//! FIFO queue acceptance is independent of the device resource worker.
use super::*;

pub(super) enum Request {
    Run(Box<dyn FnOnce() + Send>),
    Stop,
}
pub(super) fn start() -> VkResult<(mpsc::Sender<Request>, std::thread::JoinHandle<()>)> {
    let (tx, rx) = mpsc::channel();
    let thread = std::thread::Builder::new()
        .name("sgfx-vk-queue".into())
        .spawn(move || {
            while let Ok(Request::Run(job)) = rx.recv() {
                job();
            }
        })
        .map_err(|_| vk::Result::ERROR_OUT_OF_HOST_MEMORY)?;
    Ok((tx, thread))
}
#[derive(Clone)]
enum Semaphore {
    Binary(Arc<AtomicU8>),
    Timeline(Arc<timeline::Timeline>, u64),
}
impl Semaphore {
    fn get(d: &Driver, handle: vk::Semaphore, value: Option<u64>) -> VkResult<Self> {
        if let Some(state) = d
            .semaphores
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&handle.as_raw())
            .cloned()
        {
            return Ok(Self::Binary(state));
        }
        Ok(Self::Timeline(
            timeline::get(d, handle)?,
            value.ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?,
        ))
    }
    fn wait(&self, d: &Driver) -> VkResult<()> {
        loop {
            if d.lost.load(Ordering::Acquire) {
                return Err(vk::Result::ERROR_DEVICE_LOST);
            }
            let ready = match self {
                Self::Binary(s) => s
                    .compare_exchange(1, 0, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok(),
                Self::Timeline(s, v) => s.value.load(Ordering::Acquire) >= *v,
            };
            if ready {
                return Ok(());
            }
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
    }
}
#[derive(Default)]
struct Packet {
    waits: Vec<Semaphore>,
    binary_signals: Vec<Arc<AtomicU8>>,
    timeline_signals: Vec<(Arc<timeline::Timeline>, u64)>,
    recordings: Vec<Recording>,
    originals: Vec<RecordingCell>,
}
unsafe fn decode(d: &Arc<Driver>, info: &vk::SubmitInfo<'_>) -> VkResult<Packet> {
    if info.s_type != vk::StructureType::SUBMIT_INFO {
        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
    }
    let timeline = if info.p_next.is_null() {
        None
    } else {
        let node = &*info.p_next.cast::<vk::BaseInStructure<'_>>();
        if node.s_type != vk::StructureType::TIMELINE_SEMAPHORE_SUBMIT_INFO
            || !node.p_next.is_null()
            || !d.timeline_enabled
        {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        Some(&*info.p_next.cast::<vk::TimelineSemaphoreSubmitInfo<'_>>())
    };
    let wait_values = timeline
        .map(|i| slice(i.p_wait_semaphore_values, i.wait_semaphore_value_count))
        .transpose()?;
    let signal_values = timeline
        .map(|i| slice(i.p_signal_semaphore_values, i.signal_semaphore_value_count))
        .transpose()?;
    let waits = slice(info.p_wait_semaphores, info.wait_semaphore_count)?;
    let signals = slice(info.p_signal_semaphores, info.signal_semaphore_count)?;
    if !waits.is_empty() {
        let stages = slice(info.p_wait_dst_stage_mask, info.wait_semaphore_count)?;
        if stages.iter().any(|s| s.is_empty()) {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
    }
    let mut packet = Packet::default();
    for (i, &handle) in waits.iter().enumerate() {
        let sem = Semaphore::get(d, handle, wait_values.and_then(|v| v.get(i).copied()))?;
        if matches!(sem, Semaphore::Timeline(..))
            && wait_values.map(|v| v.len()) != Some(waits.len())
        {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        packet.waits.push(sem);
    }
    for (i, &handle) in signals.iter().enumerate() {
        match Semaphore::get(d, handle, signal_values.and_then(|v| v.get(i).copied()))? {
            Semaphore::Binary(s) => {
                if packet.binary_signals.iter().any(|p| Arc::ptr_eq(p, &s)) {
                    return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                }
                packet.binary_signals.push(s);
            }
            Semaphore::Timeline(s, v) => {
                if signal_values.map(|v| v.len()) != Some(signals.len()) {
                    return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                }
                packet.timeline_signals.push((s, v));
            }
        }
    }
    for cmd in slice(info.p_command_buffers, info.command_buffer_count)? {
        if !driver(cmd.as_raw(), Kind::Command).is_ok_and(|owner| Arc::ptr_eq(&owner, d)) {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        let original = d
            .recordings
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .commands
            .get(&cmd.as_raw())
            .cloned()
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let rec = original.lock().unwrap_or_else(|e| e.into_inner()).clone();
        if rec.state != RecordingState::Executable {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        if let Some(error) = rec.error {
            return Err(error);
        }
        packet.recordings.push(rec);
        packet.originals.push(original);
    }
    Ok(packet)
}
fn cancel(packets: &[Packet]) {
    for packet in packets {
        for (s, v) in &packet.timeline_signals {
            s.cancel(*v);
        }
    }
}
pub(super) unsafe fn submit(
    queue: vk::Queue,
    count: u32,
    infos: *const vk::SubmitInfo<'_>,
    fence: vk::Fence,
) -> VkResult<()> {
    let d = driver(queue.as_raw(), Kind::Queue)?;
    if d.lost.load(Ordering::Acquire) {
        return Err(vk::Result::ERROR_DEVICE_LOST);
    }
    let infos = slice(infos, count)?;
    // Admission is finite even when the application queues future timeline waits.
    let references = infos
        .iter()
        .try_fold(0usize, |total, info| {
            total
                .checked_add(info.command_buffer_count as usize)
                .and_then(|total| total.checked_add(info.wait_semaphore_count as usize))
                .and_then(|total| total.checked_add(info.signal_semaphore_count as usize))
        })
        .ok_or(vk::Result::ERROR_OUT_OF_HOST_MEMORY)?;
    if references > 16384 {
        return Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY);
    }
    let mut packets = infos
        .iter()
        .map(|info| decode(&d, info))
        .collect::<VkResult<Vec<_>>>()?;
    if packets.is_empty() {
        packets.push(Packet::default());
    }
    let fence = if fence == vk::Fence::null() {
        None
    } else {
        Some(fence_refs(&d, &[fence])?.remove(0))
    };
    let mut reserved: Vec<(Arc<timeline::Timeline>, u64)> = Vec::new();
    for p in &packets {
        for (s, v) in &p.timeline_signals {
            if let Err(e) = s.reserve(*v) {
                for (s, v) in reserved {
                    timeline::Timeline::cancel(&s, v);
                }
                return Err(e);
            }
            reserved.push((s.clone(), *v));
        }
    }
    if fence.as_ref().is_some_and(|s| {
        s.compare_exchange(0, 2, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
    }) {
        cancel(&packets);
        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
    }
    let packet_count = packets.len();
    if let Err(error) = d.pending.reserve_pending(packet_count) {
        cancel(&packets);
        if let Some(state) = &fence {
            state.store(0, Ordering::Release);
        }
        return Err(error);
    }
    let task_driver = d.clone();
    let task_fence = fence.clone();
    let sent = d.queue_sender.send(Request::Run(Box::new(move || {
        run(task_driver, packets, task_fence)
    })));
    if sent.is_err() {
        d.lost.store(true, Ordering::Release);
        for (s, v) in reserved {
            s.cancel(v);
        }
        if let Some(s) = fence {
            s.store(0, Ordering::Release);
        }
        for _ in 0..packet_count {
            d.pending.finish();
        }
        return Err(vk::Result::ERROR_DEVICE_LOST);
    }
    Ok(())
}
fn run(d: Arc<Driver>, packets: Vec<Packet>, fence: Option<Arc<AtomicU8>>) {
    for (index, packet) in packets.iter().enumerate() {
        let result = (|| {
            for wait in &packet.waits {
                wait.wait(&d)?;
            }
            let mut reserved: Vec<Arc<AtomicU8>> = Vec::new();
            for s in &packet.binary_signals {
                if s.compare_exchange(0, 2, Ordering::AcqRel, Ordering::Acquire)
                    .is_err()
                {
                    for s in reserved {
                        AtomicU8::store(&s, 0, Ordering::Release);
                    }
                    return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                }
                reserved.push(s.clone());
            }
            let recordings = packet.recordings.clone();
            let submissions = call(&d, move |rt| {
                let result = (|| {
                    let recordings = recordings
                        .iter()
                        .map(|r| {
                            resolve_recording(rt, r).inspect_err(|error| {
                                if std::env::var_os("SGFX_VULKAN_TRACE").is_some() {
                                    eprintln!(
                                        "[SGFX Vulkan] queue command resolution failed: {error:?}"
                                    );
                                }
                            })
                        })
                        .collect::<VkResult<Vec<_>>>()?;
                    for r in &recordings {
                        validate_objects(rt, r).inspect_err(|error| {
                            if std::env::var_os("SGFX_VULKAN_TRACE").is_some() {
                                eprintln!(
                                    "[SGFX Vulkan] queue object validation failed: {error:?}"
                                );
                            }
                        })?;
                    }
                    let mut submissions = Vec::new();
                    for r in &recordings {
                        match execute(rt, r) {
                            Ok(mut receipts) => submissions.append(&mut receipts),
                            Err(error) => {
                                if std::env::var_os("SGFX_VULKAN_TRACE").is_some() {
                                    eprintln!("[SGFX Vulkan] queue execution failed: {error:?}");
                                }
                                for s in &submissions {
                                    let _ = wait_submission(rt, s);
                                }
                                rt.resources.uploaded_unmapped_buffers.clear();
                                rt.lost = true;
                                return Err(error);
                            }
                        }
                    }
                    Ok(submissions)
                })();
                // All resolved recordings in this packet have now been
                // consumed. Backends retain native bindings for submitted
                // commands, so these immutable descriptor snapshots no longer
                // need logical table slots or materialization-cache entries.
                let cleanup = rt.resources.release_descriptor_groups(&rt.table, |id| {
                    rt.cache
                        .release_bind_group(id)
                        .map_err(crate::runtime::backend_failure)
                });
                let submissions = result?;
                if let Err(error) = cleanup {
                    for submission in &submissions {
                        let _ = wait_submission(rt, submission);
                    }
                    rt.lost = true;
                    return Err(error);
                }
                rt.in_flight.begin();
                Ok(submissions)
            })?;
            for original in &packet.originals {
                let mut r = original.lock().unwrap_or_else(|e| e.into_inner());
                if r.one_time {
                    r.state = RecordingState::Invalid;
                }
            }
            let request = CompletionRequest::Observe {
                submissions,
                fence: if index + 1 == packets.len() {
                    fence.clone()
                } else {
                    None
                },
                signals: packet.binary_signals.clone(),
                timeline_signals: packet.timeline_signals.clone(),
            };
            if let Err(error) = d.completion_sender.send(request)
                && let CompletionRequest::Observe {
                    submissions,
                    fence,
                    signals,
                    timeline_signals,
                } = error.0
            {
                finish_submissions(
                    &submissions,
                    fence.as_ref(),
                    &signals,
                    &timeline_signals,
                    &d.in_flight,
                    &d.pending,
                    &d.lost,
                );
            }
            Ok(())
        })();
        if let Err(error) = result {
            if std::env::var_os("SGFX_VULKAN_TRACE").is_some() {
                eprintln!("[SGFX Vulkan] queue packet {index} failed: {error:?}");
            }
            d.lost.store(true, Ordering::Release);
            cancel(&packets[index..]);
            if let Some(s) = &fence {
                s.store(0, Ordering::Release);
            }
            for remaining in &packets[index..] {
                for s in &remaining.binary_signals {
                    let _ = s.compare_exchange(2, 0, Ordering::AcqRel, Ordering::Acquire);
                }
                d.pending.finish();
            }
            break;
        }
    }
}
