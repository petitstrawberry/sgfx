//! Synchronization2 lowers to the existing resource barriers and FIFO queue.
//! GPU dependencies stay in the ordered IR stream. CPU observations flush
//! and retire the preceding queue prefix before accessing its results.
use super::*;

#[derive(Clone)]
pub(super) enum Action {
    IndirectRead(indirect::Read),
    Query(queries::Action),
    IndirectDraw(Arc<Mutex<Vec<ir::OwnedCommand>>>),
    Barrier,
    Set(Arc<bootstrap::Event>),
    Reset(Arc<bootstrap::Event>),
    Wait(Arc<bootstrap::Event>),
}

#[derive(Clone)]
pub(super) struct Insertion {
    pub position: usize,
    pub order: usize,
    pub action: Action,
}

pub(super) fn execute(
    rt: &mut Runtime,
    ops: &mut Vec<ir::OwnedCommand>,
    action: &Action,
    retained: &mut Vec<sgfx::driver::Submission>,
) -> VkResult<()> {
    if let Action::IndirectDraw(commands) = action {
        ops.extend(
            commands
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .cloned(),
        );
        return Ok(());
    }
    // CPU-authoritative arguments need no GPU boundary. Eligibility includes
    // the recorded prefix's write/alias ranges; earlier recordings finish their
    // readback before execution reaches this point. Preserve the prefix in ops
    // so the expanded draw stays in the same ordered submission.
    if let Action::IndirectRead(read) = action {
        if rt.device_lost.load(Ordering::Acquire) {
            return Err(vk::Result::ERROR_DEVICE_LOST);
        }
        if indirect::read_host(
            &rt.resources,
            read,
            rt.capabilities.supports_shader_draw_parameters(),
        )? {
            return Ok(());
        }
    }
    // Resource dependency commands remain in ops and are validated/replayed
    // in FIFO order by the backend. Only CPU observations below need a submit
    // boundary; a GPU-only dependency must not create a separate receipt.
    if matches!(action, Action::Barrier) && rt.capabilities.supports_global_resource_barriers() {
        ops.push(ir::OwnedCommand::ResourceBarrier(
            ir::OwnedResourceBarrier::Global,
        ));
        return Ok(());
    }
    // A boundary with no preceding commands has nothing new to dispatch. The
    // retained receipts below already cover the earlier queue prefix; CPU
    // observations still wait for them and for earlier in-flight packets.
    // Sending empty streams here created native fences and attachment snapshots
    // for consecutive barriers without adding any ordering guarantee.
    let submissions = if ops.is_empty() {
        Vec::new()
    } else {
        submit_owned(rt, std::mem::take(ops)).inspect_err(|error| {
            eprintln!(
                "[SGFX Vulkan] control-point submission {:?}: {error:?}",
                std::mem::discriminant(action)
            )
        })?
    };
    // CPU observations below need retirement. Keep every receipt until the
    // observation or enclosing fence/readback has consumed the queue prefix.
    retained.extend(submissions);
    if matches!(action, Action::Barrier) {
        reap_completed(rt, retained)?;
        return Ok(());
    }
    profile_work("sync::gpu_wait", || {
        // Count control points by CPU observation and receipt presence before
        // draining. `receipts` does not imply a blocking wait; `empty` can still
        // have earlier in-flight packets covered by the separate wait below.
        // Static labels use the existing cached opt-in, with no formatting or
        // environment lookup per action.
        let labels = match action {
            Action::IndirectRead(_) => [
                "sync::gpu_wait::indirect_read::empty",
                "sync::gpu_wait::indirect_read::receipts",
            ],
            Action::Query(queries::Action::Begin(..)) => [
                "sync::gpu_wait::query_begin::empty",
                "sync::gpu_wait::query_begin::receipts",
            ],
            Action::Query(queries::Action::End(..)) => [
                "sync::gpu_wait::query_end::empty",
                "sync::gpu_wait::query_end::receipts",
            ],
            Action::Query(queries::Action::Reset(..)) => [
                "sync::gpu_wait::query_reset::empty",
                "sync::gpu_wait::query_reset::receipts",
            ],
            Action::Query(queries::Action::Timestamp(..)) => [
                "sync::gpu_wait::query_timestamp::empty",
                "sync::gpu_wait::query_timestamp::receipts",
            ],
            Action::Query(queries::Action::Copy { .. }) => [
                "sync::gpu_wait::query_copy::empty",
                "sync::gpu_wait::query_copy::receipts",
            ],
            Action::Set(_) => [
                "sync::gpu_wait::event_set::empty",
                "sync::gpu_wait::event_set::receipts",
            ],
            Action::Reset(_) => [
                "sync::gpu_wait::event_reset::empty",
                "sync::gpu_wait::event_reset::receipts",
            ],
            Action::Wait(_) => [
                "sync::gpu_wait::event_wait::empty",
                "sync::gpu_wait::event_wait::receipts",
            ],
            Action::IndirectDraw(_) | Action::Barrier => unreachable!(),
        };
        profile_work(labels[usize::from(!retained.is_empty())], || {
            for submission in retained.drain(..) {
                wait_submission(rt, &submission)?;
            }
            Ok(())
        })
    })?;
    // Earlier packets may have no operations in this command buffer. Their
    // completion observer runs independently of the resource worker.
    profile_work("sync::in_flight_wait", || {
        rt.in_flight.wait();
        Ok(())
    })?;
    if rt.device_lost.load(Ordering::Acquire) {
        return Err(vk::Result::ERROR_DEVICE_LOST);
    }
    match action {
        Action::IndirectRead(read) => indirect::read(rt, read)
            .inspect_err(|error| eprintln!("[SGFX Vulkan] indirect argument read: {error:?}"))?,
        Action::Query(action) => queries::execute(rt, ops, action)?,
        Action::IndirectDraw(_) => unreachable!(),
        Action::Barrier => {}
        Action::Set(event) => event.state.store(true, Ordering::Release),
        Action::Reset(event) => event.state.store(false, Ordering::Release),
        Action::Wait(event) => {
            while !event.state.load(Ordering::Acquire) {
                if rt.device_lost.load(Ordering::Acquire) {
                    return Err(vk::Result::ERROR_DEVICE_LOST);
                }
                std::thread::sleep(std::time::Duration::from_micros(50));
            }
        }
    }
    Ok(())
}

pub(super) fn reap_completed(
    rt: &mut Runtime,
    retained: &mut Vec<sgfx::driver::Submission>,
) -> VkResult<()> {
    let mut failure = None;
    retained.retain(|receipt| match receipt.poll() {
        Ok(CompletionStatus::Complete) => false,
        Ok(_) => true,
        Err(error) => {
            failure = Some(crate::runtime::backend_failure(error));
            false
        }
    });
    if let Some(error) = failure {
        rt.lost = true;
        return Err(error);
    }
    // Logical receipts own native notification handles. Keep a bounded
    // window even when one recording contains thousands of boundaries.
    if retained.len() >= 64 {
        for receipt in retained.drain(..) {
            wait_submission(rt, &receipt)?;
        }
    }
    Ok(())
}

/// Do not truncate the 64-bit flags: split transfer, vertex-input and shader
/// read/write scopes have bits beyond the original Vulkan 1.0 masks.
pub(super) fn stages(mask: vk::PipelineStageFlags2) -> VkResult<vk::PipelineStageFlags> {
    let extra = vk::PipelineStageFlags2::COPY
        | vk::PipelineStageFlags2::RESOLVE
        | vk::PipelineStageFlags2::BLIT
        | vk::PipelineStageFlags2::CLEAR
        | vk::PipelineStageFlags2::INDEX_INPUT
        | vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT
        | vk::PipelineStageFlags2::PRE_RASTERIZATION_SHADERS;
    let low = vk::PipelineStageFlags2::from_raw(u64::from(u32::MAX));
    if !(low | extra).contains(mask) {
        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
    }
    let mut legacy = vk::PipelineStageFlags::from_raw((mask & low).as_raw() as u32);
    if mask.intersects(
        vk::PipelineStageFlags2::COPY
            | vk::PipelineStageFlags2::RESOLVE
            | vk::PipelineStageFlags2::BLIT
            | vk::PipelineStageFlags2::CLEAR,
    ) {
        legacy |= vk::PipelineStageFlags::TRANSFER;
    }
    if mask.intersects(
        vk::PipelineStageFlags2::INDEX_INPUT | vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT,
    ) {
        legacy |= vk::PipelineStageFlags::VERTEX_INPUT;
    }
    if mask.contains(vk::PipelineStageFlags2::PRE_RASTERIZATION_SHADERS) {
        legacy |= vk::PipelineStageFlags::VERTEX_SHADER;
    }
    // NONE has no scope; the execution engine may synchronize more strongly.
    if legacy.is_empty() {
        legacy = vk::PipelineStageFlags::ALL_COMMANDS;
    }
    validate_pipeline_barrier_header(legacy, legacy, vk::DependencyFlags::empty())?;
    Ok(legacy)
}

fn access(mask: vk::AccessFlags2) -> VkResult<vk::AccessFlags> {
    let extra = vk::AccessFlags2::SHADER_SAMPLED_READ
        | vk::AccessFlags2::SHADER_STORAGE_READ
        | vk::AccessFlags2::SHADER_STORAGE_WRITE;
    let low = vk::AccessFlags2::from_raw(u64::from(u32::MAX));
    if !(low | extra).contains(mask) {
        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
    }
    let mut legacy = vk::AccessFlags::from_raw((mask & low).as_raw() as u32);
    if mask
        .intersects(vk::AccessFlags2::SHADER_SAMPLED_READ | vk::AccessFlags2::SHADER_STORAGE_READ)
    {
        legacy |= vk::AccessFlags::SHADER_READ;
    }
    if mask.contains(vk::AccessFlags2::SHADER_STORAGE_WRITE) {
        legacy |= vk::AccessFlags::SHADER_WRITE;
    }
    if !supported_access(legacy) {
        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
    }
    Ok(legacy)
}

pub(super) fn buffer_access(
    scope: vk::AccessFlags,
    other: vk::AccessFlags,
    usage: vk::BufferUsageFlags,
) -> VkResult<ir::BufferAccess> {
    let scope = if scope.is_empty() { other } else { scope };
    // Global MEMORY scopes cover only the uses enabled on this resource.
    if scope.intersects(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE) {
        if usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER) {
            return Ok(if scope.contains(vk::AccessFlags::MEMORY_WRITE) {
                ir::BufferAccess::StorageReadWrite
            } else {
                ir::BufferAccess::StorageRead
            });
        }
        if scope.contains(vk::AccessFlags::MEMORY_WRITE) {
            return Ok(ir::BufferAccess::CopyDestination);
        }
        return Ok(ir::BufferAccess::CopySource);
    }
    super::buffer_access(scope)
}

fn layout(layout: vk::ImageLayout, aspects: vk::ImageAspectFlags) -> vk::ImageLayout {
    let depth = aspects.intersects(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL);
    match layout {
        vk::ImageLayout::ATTACHMENT_OPTIMAL if depth => {
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        }
        vk::ImageLayout::ATTACHMENT_OPTIMAL => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        vk::ImageLayout::READ_ONLY_OPTIMAL if depth => {
            vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
        }
        vk::ImageLayout::READ_ONLY_OPTIMAL => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        other => other,
    }
}

unsafe fn dependency(info: *const vk::DependencyInfo<'_>) -> VkResult<RecordedCommand> {
    let info = info
        .as_ref()
        .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
    if info.s_type != vk::StructureType::DEPENDENCY_INFO
        || !info.p_next.is_null()
        || !vk::DependencyFlags::BY_REGION.contains(info.dependency_flags)
    {
        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
    }
    let mut source = vk::PipelineStageFlags::empty();
    let mut destination = vk::PipelineStageFlags::empty();
    let mut masks = |src_stage, dst_stage, src, dst| -> VkResult<_> {
        source |= stages(src_stage)?;
        destination |= stages(dst_stage)?;
        Ok((access(src)?, access(dst)?))
    };
    let memory = slice(info.p_memory_barriers, info.memory_barrier_count)?
        .iter()
        .map(|barrier| {
            if barrier.s_type != vk::StructureType::MEMORY_BARRIER_2 || !barrier.p_next.is_null() {
                return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
            }
            let (src, dst) = masks(
                barrier.src_stage_mask,
                barrier.dst_stage_mask,
                barrier.src_access_mask,
                barrier.dst_access_mask,
            )?;
            Ok((false, src, dst))
        })
        .collect::<VkResult<Vec<_>>>()?;
    let buffers = slice(
        info.p_buffer_memory_barriers,
        info.buffer_memory_barrier_count,
    )?
    .iter()
    .map(|barrier| {
        if barrier.s_type != vk::StructureType::BUFFER_MEMORY_BARRIER_2 || !barrier.p_next.is_null()
        {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let (src, dst) = masks(
            barrier.src_stage_mask,
            barrier.dst_stage_mask,
            barrier.src_access_mask,
            barrier.dst_access_mask,
        )?;
        Ok((
            false,
            src,
            dst,
            barrier.src_queue_family_index,
            barrier.dst_queue_family_index,
            barrier.buffer,
            barrier.offset,
            barrier.size,
        ))
    })
    .collect::<VkResult<Vec<_>>>()?;
    let images = slice(
        info.p_image_memory_barriers,
        info.image_memory_barrier_count,
    )?
    .iter()
    .map(|barrier| {
        if barrier.s_type != vk::StructureType::IMAGE_MEMORY_BARRIER_2 || !barrier.p_next.is_null()
        {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let (src, dst) = masks(
            barrier.src_stage_mask,
            barrier.dst_stage_mask,
            barrier.src_access_mask,
            barrier.dst_access_mask,
        )?;
        let aspects = barrier.subresource_range.aspect_mask;
        Ok((
            false,
            src,
            dst,
            barrier.src_queue_family_index,
            barrier.dst_queue_family_index,
            barrier.image,
            layout(barrier.old_layout, aspects),
            layout(barrier.new_layout, aspects),
            barrier.subresource_range,
        ))
    })
    .collect::<VkResult<Vec<_>>>()?;
    if source.is_empty() {
        source = vk::PipelineStageFlags::ALL_COMMANDS;
    }
    if destination.is_empty() {
        destination = vk::PipelineStageFlags::ALL_COMMANDS;
    }
    Ok(RecordedCommand::PipelineBarrier {
        full_scope: true,
        source_stage: source,
        destination_stage: destination,
        flags: info.dependency_flags,
        memory,
        buffers,
        images,
    })
}

fn enabled(command: vk::CommandBuffer) -> VkResult<Arc<Driver>> {
    let d = driver(command.as_raw(), Kind::Command)?;
    if !d.synchronization2_enabled {
        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
    }
    Ok(d)
}

unsafe extern "system" fn barrier(command: vk::CommandBuffer, info: *const vk::DependencyInfo<'_>) {
    match enabled(command).and_then(|_| dependency(info)) {
        Ok(barrier) => record(command, barrier),
        Err(error) => record_error(command, error),
    }
}

unsafe extern "system" fn set_event(
    command: vk::CommandBuffer,
    event: vk::Event,
    info: *const vk::DependencyInfo<'_>,
) {
    let result = (|| {
        let d = enabled(command)?;
        let event = bootstrap::event_state(&d, event)?;
        let barrier = dependency(info)?;
        // Satisfy both scopes at signal time; repeating resource barriers at
        // wait time is harmless and preserves the matching dependency scopes.
        record(command, barrier);
        record(command, RecordedCommand::Synchronize(Action::Set(event)));
        Ok(())
    })();
    if let Err(error) = result {
        record_error(command, error);
    }
}

unsafe extern "system" fn reset_event(
    command: vk::CommandBuffer,
    event: vk::Event,
    stage: vk::PipelineStageFlags2,
) {
    let result = (|| {
        let d = enabled(command)?;
        stages(stage)?;
        record(
            command,
            RecordedCommand::Synchronize(Action::Reset(bootstrap::event_state(&d, event)?)),
        );
        Ok(())
    })();
    if let Err(error) = result {
        record_error(command, error);
    }
}

unsafe extern "system" fn wait_events(
    command: vk::CommandBuffer,
    count: u32,
    events: *const vk::Event,
    infos: *const vk::DependencyInfo<'_>,
) {
    let result = (|| {
        let d = enabled(command)?;
        let events = slice(events, count)?;
        let infos = slice(infos, count)?;
        // Validate the entire call before publishing any command prefix.
        let dependencies = events
            .iter()
            .zip(infos)
            .map(|(&event, info)| Ok((bootstrap::event_state(&d, event)?, dependency(info)?)))
            .collect::<VkResult<Vec<_>>>()?;
        for (event, barrier) in dependencies {
            record(command, RecordedCommand::Synchronize(Action::Wait(event)));
            record(command, barrier);
        }
        Ok(())
    })();
    if let Err(error) = result {
        record_error(command, error);
    }
}

unsafe extern "system" fn timestamp(
    command: vk::CommandBuffer,
    stage: vk::PipelineStageFlags2,
    pool: vk::QueryPool,
    index: u32,
) {
    if let Err(e) = enabled(command) {
        record_error(command, e);
        return;
    }
    queries::timestamp2(command, stage, pool, index);
}

struct Submit {
    waits: Vec<vk::Semaphore>,
    values: Vec<u64>,
    stages: Vec<vk::PipelineStageFlags>,
    signals: Vec<vk::Semaphore>,
    signal_values: Vec<u64>,
    commands: Vec<vk::CommandBuffer>,
}

unsafe fn decode_submit(info: &vk::SubmitInfo2<'_>) -> VkResult<Submit> {
    if info.s_type != vk::StructureType::SUBMIT_INFO_2
        || !info.p_next.is_null()
        || !info.flags.is_empty()
    {
        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
    }
    let mut result = Submit {
        waits: Vec::new(),
        values: Vec::new(),
        stages: Vec::new(),
        signals: Vec::new(),
        signal_values: Vec::new(),
        commands: Vec::new(),
    };
    for (wait, infos) in [
        (
            true,
            slice(info.p_wait_semaphore_infos, info.wait_semaphore_info_count)?,
        ),
        (
            false,
            slice(
                info.p_signal_semaphore_infos,
                info.signal_semaphore_info_count,
            )?,
        ),
    ] {
        for info in infos {
            if info.s_type != vk::StructureType::SEMAPHORE_SUBMIT_INFO
                || !info.p_next.is_null()
                || info.device_index != 0
            {
                return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
            }
            let stage = stages(info.stage_mask)?;
            if wait {
                result.waits.push(info.semaphore);
                result.values.push(info.value);
                result.stages.push(stage);
            } else {
                result.signals.push(info.semaphore);
                result.signal_values.push(info.value);
            }
        }
    }
    for info in slice(info.p_command_buffer_infos, info.command_buffer_info_count)? {
        if info.s_type != vk::StructureType::COMMAND_BUFFER_SUBMIT_INFO
            || !info.p_next.is_null()
            || info.device_mask > 1
        {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        result.commands.push(info.command_buffer);
    }
    Ok(result)
}

unsafe extern "system" fn submit(
    queue: vk::Queue,
    count: u32,
    infos: *const vk::SubmitInfo2<'_>,
    fence: vk::Fence,
) -> vk::Result {
    let _frame = crate::frame_profile::Span::new("submit", "queue_submit2");
    status((|| {
        let d = driver(queue.as_raw(), Kind::Queue)?;
        if !d.synchronization2_enabled {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        if d.lost.load(Ordering::Acquire) {
            return Err(vk::Result::ERROR_DEVICE_LOST);
        }
        let infos = slice(infos, count)?;
        let references = infos
            .iter()
            .try_fold(0usize, |total, info| {
                total
                    .checked_add(info.command_buffer_info_count as usize)?
                    .checked_add(info.wait_semaphore_info_count as usize)?
                    .checked_add(info.signal_semaphore_info_count as usize)
            })
            .filter(|&total| total <= 16384)
            .ok_or(vk::Result::ERROR_OUT_OF_HOST_MEMORY)?;
        let _ = references;
        let mut packets = Vec::new();
        for info in infos {
            let info = decode_submit(info)?;
            let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
                .wait_semaphore_values(&info.values)
                .signal_semaphore_values(&info.signal_values);
            let mut legacy = vk::SubmitInfo::default()
                .wait_semaphores(&info.waits)
                .wait_dst_stage_mask(&info.stages)
                .signal_semaphores(&info.signals)
                .command_buffers(&info.commands);
            // Binary values are ignored. A timeline record requires the
            // timeline feature; it is unnecessary on binary-only devices.
            if d.timeline_enabled {
                legacy = legacy.push_next(&mut timeline);
            }
            packets.push(queue::decode(&d, &legacy)?);
        }
        queue::enqueue(d, packets, fence)
    })())
}

pub(super) fn lookup(name: &CStr) -> vk::PFN_vkVoidFunction {
    macro_rules! entry {
        ($f:path,$t:ty) => {
            Some(unsafe { std::mem::transmute::<$t, unsafe extern "system" fn()>($f) })
        };
    }
    match name.to_bytes() {
        b"vkCmdPipelineBarrier2" | b"vkCmdPipelineBarrier2KHR" => {
            entry!(barrier, vk::PFN_vkCmdPipelineBarrier2)
        }
        b"vkQueueSubmit2" | b"vkQueueSubmit2KHR" => entry!(submit, vk::PFN_vkQueueSubmit2),
        b"vkCmdSetEvent2" | b"vkCmdSetEvent2KHR" => entry!(set_event, vk::PFN_vkCmdSetEvent2),
        b"vkCmdResetEvent2" | b"vkCmdResetEvent2KHR" => {
            entry!(reset_event, vk::PFN_vkCmdResetEvent2)
        }
        b"vkCmdWaitEvents2" | b"vkCmdWaitEvents2KHR" => {
            entry!(wait_events, vk::PFN_vkCmdWaitEvents2)
        }
        b"vkCmdWriteTimestamp2" | b"vkCmdWriteTimestamp2KHR" => {
            entry!(timestamp, vk::PFN_vkCmdWriteTimestamp2)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_scopes_keep_their_high_bits_and_reject_unknown_bits() {
        assert_eq!(
            stages(vk::PipelineStageFlags2::COPY | vk::PipelineStageFlags2::CLEAR).unwrap(),
            vk::PipelineStageFlags::TRANSFER
        );
        assert_eq!(
            stages(vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT).unwrap(),
            vk::PipelineStageFlags::VERTEX_INPUT
        );
        assert_eq!(
            stages(vk::PipelineStageFlags2::NONE).unwrap(),
            vk::PipelineStageFlags::ALL_COMMANDS
        );
        assert_eq!(
            access(vk::AccessFlags2::SHADER_SAMPLED_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .unwrap(),
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE
        );
        assert!(stages(vk::PipelineStageFlags2::from_raw(1 << 63)).is_err());
        assert!(access(vk::AccessFlags2::from_raw(1 << 63)).is_err());
        assert!(stages(vk::PipelineStageFlags2::GEOMETRY_SHADER).is_err());
    }

    #[test]
    fn dependency_owns_memory_and_normalizes_layout_aliases() {
        let image = vk::Image::from_raw(42);
        let recorded = unsafe {
            let images = [vk::ImageMemoryBarrier2::default()
                .image(image)
                .old_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                )];
            dependency(&vk::DependencyInfo::default().image_memory_barriers(&images)).unwrap()
        };
        let RecordedCommand::PipelineBarrier {
            images, full_scope, ..
        } = recorded
        else {
            panic!()
        };
        assert!(full_scope);
        assert_eq!(images[0].5, image);
        assert_eq!(images[0].6, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        assert_eq!(images[0].7, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        assert_eq!(
            layout(
                vk::ImageLayout::ATTACHMENT_OPTIMAL,
                vk::ImageAspectFlags::DEPTH
            ),
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        );
    }

    #[test]
    fn submit_validates_single_device_masks_and_nested_array_bounds() {
        unsafe {
            let commands = [vk::CommandBufferSubmitInfo::default().device_mask(0)];
            assert!(
                decode_submit(&vk::SubmitInfo2::default().command_buffer_infos(&commands)).is_ok()
            );
            let commands = [vk::CommandBufferSubmitInfo::default().device_mask(2)];
            assert!(
                decode_submit(&vk::SubmitInfo2::default().command_buffer_infos(&commands)).is_err()
            );
            let mut invalid = vk::SubmitInfo2 {
                wait_semaphore_info_count: 1,
                ..Default::default()
            };
            assert!(decode_submit(&invalid).is_err());
            invalid.wait_semaphore_info_count = 4097;
            assert!(decode_submit(&invalid).is_err());
        }
    }

    #[test]
    fn synchronization2_requires_an_explicit_boolean_feature() {
        unsafe {
            let supported = vk::PhysicalDeviceFeatures::default();
            let mut feature =
                vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
            assert!(device_chain_supported(
                (&feature as *const vk::PhysicalDeviceSynchronization2Features<'_>).cast(),
                &supported
            ));
            feature.synchronization2 = 2;
            assert!(!device_chain_supported(
                (&feature as *const vk::PhysicalDeviceSynchronization2Features<'_>).cast(),
                &supported
            ));
        }
    }
}
