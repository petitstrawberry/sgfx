//! Timeline values become visible only after GPU retirement or a host signal.
use super::*;
use std::collections::BTreeSet;

pub(super) struct Timeline {
    pub value: AtomicU64,
    pending: Mutex<BTreeSet<u64>>,
}
impl Timeline {
    fn new(value: u64) -> Self {
        Self {
            value: AtomicU64::new(value),
            pending: Default::default(),
        }
    }
    pub fn reserve(&self, value: u64) -> VkResult<()> {
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        if value <= self.value.load(Ordering::Acquire)
            || pending.last().is_some_and(|&last| value <= last)
        {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        pending.insert(value);
        Ok(())
    }
    pub fn cancel(&self, value: u64) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&value);
    }
    pub fn complete(&self, value: u64) {
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        self.value.fetch_max(value, Ordering::Release);
        pending.remove(&value);
    }
    fn signal(&self, value: u64) -> VkResult<()> {
        let pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        if value <= self.value.load(Ordering::Acquire)
            || pending.first().is_some_and(|&first| value >= first)
        {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        self.value.store(value, Ordering::Release);
        Ok(())
    }
}
pub(super) fn get(d: &Driver, handle: vk::Semaphore) -> VkResult<Arc<Timeline>> {
    if d.lost.load(Ordering::Acquire) {
        return Err(vk::Result::ERROR_DEVICE_LOST);
    }
    d.timelines
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&handle.as_raw())
        .cloned()
        .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)
}
pub(super) unsafe fn create(
    d: &Driver,
    info: &vk::SemaphoreCreateInfo<'_>,
    out: *mut vk::Semaphore,
) -> VkResult<()> {
    let ty = &*info.p_next.cast::<vk::SemaphoreTypeCreateInfo<'_>>();
    if ty.s_type != vk::StructureType::SEMAPHORE_TYPE_CREATE_INFO || !ty.p_next.is_null() {
        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
    }
    if ty.semaphore_type == vk::SemaphoreType::BINARY && ty.initial_value == 0 {
        let mut sems = d.semaphores.lock().unwrap_or_else(|e| e.into_inner());
        if sems.len() >= 4096 {
            return Err(vk::Result::ERROR_TOO_MANY_OBJECTS);
        }
        let id = next_id();
        sems.insert(id, Arc::new(AtomicU8::new(0)));
        *out = vk::Semaphore::from_raw(id);
        return Ok(());
    }
    if ty.semaphore_type != vk::SemaphoreType::TIMELINE || !d.timeline_enabled {
        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
    }
    let mut sems = d.timelines.lock().unwrap_or_else(|e| e.into_inner());
    if sems.len() >= 4096 {
        return Err(vk::Result::ERROR_TOO_MANY_OBJECTS);
    }
    let id = next_id();
    sems.insert(id, Arc::new(Timeline::new(ty.initial_value)));
    *out = vk::Semaphore::from_raw(id);
    Ok(())
}
unsafe extern "system" fn counter(
    device: vk::Device,
    semaphore: vk::Semaphore,
    out: *mut u64,
) -> vk::Result {
    if out.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    status((|| {
        let d = driver(device.as_raw(), Kind::Device)?;
        let sem = get(&d, semaphore)?;
        *out = sem.value.load(Ordering::Acquire);
        Ok(())
    })())
}
unsafe extern "system" fn signal(
    device: vk::Device,
    info: *const vk::SemaphoreSignalInfo<'_>,
) -> vk::Result {
    status((|| {
        let i = info
            .as_ref()
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        if i.s_type != vk::StructureType::SEMAPHORE_SIGNAL_INFO || !i.p_next.is_null() {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let d = driver(device.as_raw(), Kind::Device)?;
        get(&d, i.semaphore)?.signal(i.value)
    })())
}
unsafe extern "system" fn wait(
    device: vk::Device,
    info: *const vk::SemaphoreWaitInfo<'_>,
    timeout: u64,
) -> vk::Result {
    let result = (|| {
        let i = info
            .as_ref()
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        if i.s_type != vk::StructureType::SEMAPHORE_WAIT_INFO
            || !i.p_next.is_null()
            || !vk::SemaphoreWaitFlags::ANY.contains(i.flags)
        {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let d = driver(device.as_raw(), Kind::Device)?;
        let handles = slice(i.p_semaphores, i.semaphore_count)?;
        let values = slice(i.p_values, i.semaphore_count)?;
        if handles.is_empty() {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        let sems = handles
            .iter()
            .map(|&h| get(&d, h))
            .collect::<VkResult<Vec<_>>>()?;
        let start = std::time::Instant::now();
        loop {
            if d.lost.load(Ordering::Acquire) {
                return Err(vk::Result::ERROR_DEVICE_LOST);
            }
            let mut ready = sems
                .iter()
                .zip(values)
                .map(|(s, v)| s.value.load(Ordering::Acquire) >= *v);
            if if i.flags.contains(vk::SemaphoreWaitFlags::ANY) {
                ready.any(|v| v)
            } else {
                ready.all(|v| v)
            } {
                return Ok(());
            }
            if timeout != u64::MAX && start.elapsed().as_nanos() >= timeout as u128 {
                return Err(vk::Result::TIMEOUT);
            }
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
    })();
    status(result)
}
pub(super) fn lookup(name: &CStr) -> vk::PFN_vkVoidFunction {
    macro_rules! entry {
        ($f:ident,$ty:ty) => {{
            let f: $ty = $f;
            Some(unsafe { std::mem::transmute::<$ty, unsafe extern "system" fn()>(f) })
        }};
    }
    match name.to_bytes() {
        b"vkGetSemaphoreCounterValueKHR" => entry!(counter, vk::PFN_vkGetSemaphoreCounterValue),
        b"vkSignalSemaphoreKHR" => entry!(signal, vk::PFN_vkSignalSemaphore),
        b"vkWaitSemaphoresKHR" => entry!(wait, vk::PFN_vkWaitSemaphores),
        _ => None,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn host_queries_are_nonconsuming_and_validate_device_ownership() {
        let (sender, _rx) = mpsc::channel();
        let (completion_sender, _completion_rx) = mpsc::channel();
        let d = Arc::new(Driver {
            sender,
            thread: Mutex::new(None),
            completion_sender,
            completion_thread: Mutex::new(None),
            queue: AtomicU64::new(0),
            queue_sender: mpsc::channel().0,
            queue_thread: Mutex::new(None),
            timeline_enabled: true,
            timelines: Default::default(),
            events: Default::default(),
            fences: Default::default(),
            semaphores: Default::default(),
            recordings: Default::default(),
            in_flight: Default::default(),
            pending: Default::default(),
            lost: Arc::new(AtomicBool::new(false)),
        });
        let device = vk::Device::from_raw(add_handle(Kind::Device, &d));
        let handles = [
            vk::Semaphore::from_raw(next_id()),
            vk::Semaphore::from_raw(next_id()),
        ];
        d.timelines
            .lock()
            .unwrap()
            .insert(handles[0].as_raw(), Arc::new(Timeline::new(0)));
        d.timelines
            .lock()
            .unwrap()
            .insert(handles[1].as_raw(), Arc::new(Timeline::new(3)));
        let values = [1, 3];
        let mut info = vk::SemaphoreWaitInfo::default()
            .semaphores(&handles)
            .values(&values);
        unsafe {
            assert_eq!(wait(device, &info, 0), vk::Result::TIMEOUT);
            info.flags = vk::SemaphoreWaitFlags::ANY;
            assert_eq!(wait(device, &info, 0), vk::Result::SUCCESS);
            let mut counter_value = 0;
            assert_eq!(
                counter(device, handles[1], &mut counter_value),
                vk::Result::SUCCESS
            );
            assert_eq!(counter_value, 3);
            assert_eq!(
                signal(
                    device,
                    &vk::SemaphoreSignalInfo::default()
                        .semaphore(handles[0])
                        .value(1)
                ),
                vk::Result::SUCCESS
            );
            info.flags = vk::SemaphoreWaitFlags::empty();
            assert_eq!(wait(device, &info, 0), vk::Result::SUCCESS);
            assert_eq!(wait(device, &info, 0), vk::Result::SUCCESS);
            assert_eq!(
                counter(
                    device,
                    vk::Semaphore::from_raw(next_id()),
                    &mut counter_value
                ),
                vk::Result::ERROR_INITIALIZATION_FAILED
            );
            d.lost.store(true, Ordering::Release);
            assert_eq!(wait(device, &info, 0), vk::Result::ERROR_DEVICE_LOST);
        }
        remove_handle(device.as_raw());
    }
    #[test]
    fn queued_values_are_invisible_until_retirement_and_host_cannot_overtake_them() {
        let sem = Timeline::new(3);
        sem.reserve(7).unwrap();
        sem.reserve(9).unwrap();
        assert_eq!(sem.value.load(Ordering::Acquire), 3);
        assert!(sem.signal(7).is_err());
        sem.signal(5).unwrap();
        sem.complete(7);
        assert_eq!(sem.value.load(Ordering::Acquire), 7);
        assert!(sem.reserve(8).is_err());
        sem.complete(9);
        sem.signal(10).unwrap();
        assert!(sem.signal(10).is_err());
    }
}
