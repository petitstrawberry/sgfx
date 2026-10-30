//! GPU query results become available only after measured work retires.
use super::*;
const INVALID: vk::Result = vk::Result::ERROR_INITIALIZATION_FAILED;
const UNSUPPORTED: vk::Result = vk::Result::ERROR_FEATURE_NOT_PRESENT;
pub(super) struct Pool {
    pub(super) kind: u32,
    statistics: u32,
    values: Mutex<Vec<Option<Vec<u64>>>>,
    changed: Condvar,
}
impl Pool {
    fn words(&self) -> usize {
        if self.kind == 1 {
            self.statistics.count_ones() as usize
        } else {
            1
        }
    }
    fn range(&self, first: u32, count: u32) -> VkResult<()> {
        if first.checked_add(count).is_none_or(|n| {
            n as usize > self.values.lock().unwrap_or_else(|e| e.into_inner()).len()
        }) {
            Err(INVALID)
        } else {
            Ok(())
        }
    }
    fn reset(&self, first: u32, count: u32) {
        self.values.lock().unwrap_or_else(|e| e.into_inner())
            [first as usize..(first + count) as usize]
            .fill(None);
        self.changed.notify_all();
    }
}
#[derive(Clone)]
pub(super) enum Action {
    Begin(Arc<Pool>, u32),
    End(Arc<Pool>, u32),
    Reset(Arc<Pool>, u32, u32),
    Timestamp(Arc<Pool>, u32),
    Copy {
        pool: Arc<Pool>,
        first: u32,
        count: u32,
        buffer: vk::Buffer,
        offset: u64,
        stride: u64,
        flags: vk::QueryResultFlags,
    },
}
fn pool(d: &Driver, handle: vk::QueryPool, first: u32, count: u32) -> VkResult<Arc<Pool>> {
    let pool = d
        .queries
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&handle.as_raw())
        .cloned()
        .ok_or(INVALID)?;
    pool.range(first, count)?;
    Ok(pool)
}
unsafe extern "system" fn create(
    device: vk::Device,
    info: *const vk::QueryPoolCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::QueryPool,
) -> vk::Result {
    status((|| {
        let out = out.as_mut().ok_or(INVALID)?;
        *out = vk::QueryPool::null();
        let info = info.as_ref().ok_or(INVALID)?;
        let d = driver(device.as_raw(), Kind::Device)?;
        if !allocator.is_null()
            || info.s_type != vk::StructureType::QUERY_POOL_CREATE_INFO
            || !info.p_next.is_null()
            || !info.flags.is_empty()
        {
            return Err(UNSUPPORTED);
        }
        let kind = info.query_type.as_raw() as u32;
        let statistics = info.pipeline_statistics.as_raw();
        match info.query_type {
            vk::QueryType::TIMESTAMP if d.timestamps && statistics == 0 => {}
            vk::QueryType::OCCLUSION if d.query_support & 1 != 0 && statistics == 0 => {}
            vk::QueryType::PIPELINE_STATISTICS
                if d.pipeline_statistics_enabled && statistics != 0 && statistics & !0x7ff == 0 => {
            }
            _ => return Err(UNSUPPORTED),
        }
        if info.query_count == 0 || info.query_count > 4096 {
            return Err(INVALID);
        }
        let mut pools = d.queries.lock().unwrap_or_else(|e| e.into_inner());
        if pools.len() >= 1024 {
            return Err(vk::Result::ERROR_TOO_MANY_OBJECTS);
        }
        let id = next_id();
        pools.insert(
            id,
            Arc::new(Pool {
                kind,
                statistics,
                values: Mutex::new(vec![None; info.query_count as usize]),
                changed: Condvar::new(),
            }),
        );
        *out = vk::QueryPool::from_raw(id);
        Ok(())
    })())
}
unsafe extern "system" fn destroy(
    device: vk::Device,
    handle: vk::QueryPool,
    _: *const vk::AllocationCallbacks<'_>,
) {
    if let Ok(d) = driver(device.as_raw(), Kind::Device) {
        d.queries
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&handle.as_raw());
    }
}
unsafe extern "system" fn host_reset(
    device: vk::Device,
    handle: vk::QueryPool,
    first: u32,
    count: u32,
) {
    if let Ok(d) = driver(device.as_raw(), Kind::Device)
        && d.host_query_reset_enabled
        && let Ok(pool) = pool(&d, handle, first, count)
    {
        pool.reset(first, count);
    }
}
unsafe extern "system" fn reset(
    command: vk::CommandBuffer,
    handle: vk::QueryPool,
    first: u32,
    count: u32,
) {
    match driver(command.as_raw(), Kind::Command).and_then(|d| pool(&d, handle, first, count)) {
        Ok(pool) => record(
            command,
            RecordedCommand::Query(Action::Reset(pool, first, count)),
        ),
        Err(e) => record_error(command, e),
    }
}
pub(super) unsafe extern "system" fn timestamp2(
    command: vk::CommandBuffer,
    stage: vk::PipelineStageFlags2,
    handle: vk::QueryPool,
    index: u32,
) {
    let result = (|| {
        let d = driver(command.as_raw(), Kind::Command)?;
        if !d.timestamps || stage.is_empty() || stage.as_raw().count_ones() != 1 {
            return Err(UNSUPPORTED);
        }
        synchronization2::stages(stage)?;
        let pool = pool(&d, handle, index, 1)?;
        if pool.kind != 2 {
            return Err(INVALID);
        }
        Ok(pool)
    })();
    match result {
        Ok(pool) => record(
            command,
            RecordedCommand::Query(Action::Timestamp(pool, index)),
        ),
        Err(e) => record_error(command, e),
    }
}
unsafe extern "system" fn timestamp(
    command: vk::CommandBuffer,
    stage: vk::PipelineStageFlags,
    handle: vk::QueryPool,
    index: u32,
) {
    timestamp2(
        command,
        vk::PipelineStageFlags2::from_raw(u64::from(stage.as_raw())),
        handle,
        index,
    )
}
fn shape(
    words: usize,
    count: u32,
    stride: u64,
    flags: vk::QueryResultFlags,
) -> VkResult<(usize, usize)> {
    let allowed = vk::QueryResultFlags::TYPE_64
        | vk::QueryResultFlags::WAIT
        | vk::QueryResultFlags::WITH_AVAILABILITY;
    if !allowed.contains(flags) {
        return Err(UNSUPPORTED);
    }
    let word = if flags.contains(vk::QueryResultFlags::TYPE_64) {
        8usize
    } else {
        4
    };
    let row = word * (words + usize::from(flags.contains(vk::QueryResultFlags::WITH_AVAILABILITY)));
    if count > 1 && (stride < row as u64 || !stride.is_multiple_of(word as u64)) {
        return Err(INVALID);
    }
    let size = if count == 0 {
        0
    } else {
        u64::from(count - 1)
            .checked_mul(stride)
            .and_then(|n| n.checked_add(row as u64))
            .ok_or(INVALID)?
    };
    Ok((word, usize::try_from(size).map_err(|_| INVALID)?))
}
fn values(
    pool: &Pool,
    first: u32,
    count: u32,
    wait: bool,
    lost: &AtomicBool,
) -> VkResult<Vec<Option<Vec<u64>>>> {
    let mut values = pool.values.lock().unwrap_or_else(|e| e.into_inner());
    while wait
        && values[first as usize..(first + count) as usize]
            .iter()
            .any(Option::is_none)
    {
        if lost.load(Ordering::Acquire) {
            return Err(vk::Result::ERROR_DEVICE_LOST);
        }
        values = pool
            .changed
            .wait_timeout(values, std::time::Duration::from_millis(1))
            .unwrap_or_else(|e| e.into_inner())
            .0;
    }
    Ok(values[first as usize..(first + count) as usize].to_vec())
}
fn word(output: &mut [u8], offset: usize, value: u64, width: usize) {
    output[offset..offset + width].copy_from_slice(&value.to_le_bytes()[..width]);
}
unsafe extern "system" fn results(
    device: vk::Device,
    handle: vk::QueryPool,
    first: u32,
    count: u32,
    size: usize,
    data: *mut std::ffi::c_void,
    stride: u64,
    flags: vk::QueryResultFlags,
) -> vk::Result {
    let result = (|| {
        let d = driver(device.as_raw(), Kind::Device)?;
        let pool = pool(&d, handle, first, count)?;
        let (width, needed) = shape(pool.words(), count, stride, flags)?;
        if needed > size || (needed != 0 && data.is_null()) {
            return Err(INVALID);
        }
        let values = values(
            &pool,
            first,
            count,
            flags.contains(vk::QueryResultFlags::WAIT),
            &d.lost,
        )?;
        let output = if needed == 0 {
            &mut []
        } else {
            std::slice::from_raw_parts_mut(data.cast::<u8>(), needed)
        };
        for (index, value) in values.iter().enumerate() {
            let offset = index * stride as usize;
            if let Some(value) = value {
                for (counter, value) in value.iter().enumerate() {
                    word(output, offset + counter * width, *value, width);
                }
            }
            if flags.contains(vk::QueryResultFlags::WITH_AVAILABILITY) {
                word(
                    output,
                    offset + pool.words() * width,
                    u64::from(value.is_some()),
                    width,
                );
            }
        }
        if values.iter().any(Option::is_none) {
            Err(vk::Result::NOT_READY)
        } else {
            Ok(())
        }
    })();
    status(result)
}
unsafe extern "system" fn copy(
    command: vk::CommandBuffer,
    handle: vk::QueryPool,
    first: u32,
    count: u32,
    buffer: vk::Buffer,
    offset: u64,
    stride: u64,
    flags: vk::QueryResultFlags,
) {
    let result = (|| {
        let d = driver(command.as_raw(), Kind::Command)?;
        let pool = pool(&d, handle, first, count)?;
        shape(pool.words(), count, stride, flags)?;
        Ok(Action::Copy {
            pool,
            first,
            count,
            buffer,
            offset,
            stride,
            flags,
        })
    })();
    match result {
        Ok(a) => record(command, RecordedCommand::Query(a)),
        Err(e) => record_error(command, e),
    }
}
pub(super) fn resolve(rt: &Runtime, rec: &mut ResolvedRecording, action: &Action) -> VkResult<()> {
    if matches!(action, Action::Reset(..) | Action::Copy { .. }) && rec.render.is_some() {
        return Err(INVALID);
    }
    if let Action::Copy {
        count,
        buffer,
        offset,
        stride,
        flags,
        pool,
        ..
    } = action
    {
        if rec.render.is_some() {
            return Err(INVALID);
        }
        let (width, needed) = shape(pool.words(), *count, *stride, *flags)?;
        let b = rt.resources.buffers.get(buffer).ok_or(INVALID)?;
        if b.bound.is_none()
            || !b.usage.contains(vk::BufferUsageFlags::TRANSFER_DST)
            || offset % width as u64 != 0
            || offset
                .checked_add(needed as u64)
                .is_none_or(|end| end > b.size)
        {
            return Err(INVALID);
        }
        rec.used_buffers.push(*buffer);
        rec.readback_buffers
            .insert(*buffer, *offset, needed as u64)?;
    }
    let active = if rec.render.is_some() {
        Some(render_passes::suspend(rec)?)
    } else {
        None
    };
    rec.synchronizations.push(synchronization2::Insertion {
        position: rec.ops.len(),
        order: rec.order,
        action: synchronization2::Action::Query(action.clone()),
    });
    if let Some(active) = active {
        render_passes::resume(rec, active)?;
    }
    Ok(())
}
pub(super) fn execute(
    rt: &mut Runtime,
    ops: &mut Vec<ir::OwnedCommand>,
    action: &Action,
) -> VkResult<()> {
    match action {
        Action::Begin(pool, index) => {
            if pool.values.lock().unwrap_or_else(|e| e.into_inner())[*index as usize].is_some() {
                return Err(INVALID);
            }
            rt.device
                .begin_query(pool.kind, pool.statistics)
                .map_err(crate::runtime::backend_failure)?;
        }
        Action::End(pool, index) => {
            let values = rt
                .device
                .end_query(pool.kind, pool.words())
                .map_err(crate::runtime::backend_failure)?;
            if values.len() != pool.words() {
                return Err(INVALID);
            }
            pool.values.lock().unwrap_or_else(|e| e.into_inner())[*index as usize] = Some(values);
            pool.changed.notify_all();
        }
        Action::Reset(pool, first, count) => pool.reset(*first, *count),
        Action::Timestamp(pool, index) => {
            let value = rt
                .device
                .timestamp_ns()
                .map_err(crate::runtime::backend_failure)?;
            pool.values.lock().unwrap_or_else(|e| e.into_inner())[*index as usize] =
                Some(vec![value]);
            pool.changed.notify_all();
        }
        Action::Copy {
            pool,
            first,
            count,
            buffer,
            offset,
            stride,
            flags,
        } => {
            let (width, _) = shape(pool.words(), *count, *stride, *flags)?;
            let values = values(
                pool,
                *first,
                *count,
                flags.contains(vk::QueryResultFlags::WAIT),
                &rt.device_lost,
            )?;
            let buffer = rt.resources.buffers.get(buffer).ok_or(INVALID)?;
            for (index, value) in values.iter().enumerate() {
                let mut bytes = vec![
                    0;
                    width
                        * (pool.words()
                            + usize::from(flags.contains(
                                vk::QueryResultFlags::WITH_AVAILABILITY
                            )))
                ];
                if let Some(values) = value {
                    for (index, value) in values.iter().enumerate() {
                        word(&mut bytes, index * width, *value, width);
                    }
                }
                if flags.contains(vk::QueryResultFlags::WITH_AVAILABILITY) {
                    word(
                        &mut bytes,
                        pool.words() * width,
                        u64::from(value.is_some()),
                        width,
                    );
                }
                if !bytes.is_empty() {
                    ops.push(ir::OwnedCommand::WriteBuffer {
                        buffer: buffer.id,
                        offset: *offset + index as u64 * stride,
                        data: bytes,
                    });
                }
            }
        }
    }
    Ok(())
}
unsafe extern "system" fn begin(
    command: vk::CommandBuffer,
    handle: vk::QueryPool,
    index: u32,
    flags: vk::QueryControlFlags,
) {
    let result = (|| {
        let d = driver(command.as_raw(), Kind::Command)?;
        let p = pool(&d, handle, index, 1)?;
        if p.kind > 1
            || !vk::QueryControlFlags::PRECISE.contains(flags)
            || (!flags.is_empty() && (p.kind != 0 || !d.occlusion_precise_enabled))
        {
            return Err(UNSUPPORTED);
        }
        Ok(p)
    })();
    match result {
        Ok(p) => record(command, RecordedCommand::Query(Action::Begin(p, index))),
        Err(e) => record_error(command, e),
    }
}
unsafe extern "system" fn end(command: vk::CommandBuffer, handle: vk::QueryPool, index: u32) {
    match driver(command.as_raw(), Kind::Command).and_then(|d| pool(&d, handle, index, 1)) {
        Ok(p) => record(command, RecordedCommand::Query(Action::End(p, index))),
        Err(e) => record_error(command, e),
    }
}

pub(super) fn lookup(name: &CStr) -> vk::PFN_vkVoidFunction {
    macro_rules! f {
        ($f:path,$t:ty) => {
            Some(unsafe { std::mem::transmute::<$t, unsafe extern "system" fn()>($f) })
        };
    }
    match name.to_bytes() {
        b"vkCmdBeginQuery" => f!(begin, vk::PFN_vkCmdBeginQuery),
        b"vkCmdEndQuery" => f!(end, vk::PFN_vkCmdEndQuery),
        b"vkCreateQueryPool" => f!(create, vk::PFN_vkCreateQueryPool),
        b"vkDestroyQueryPool" => f!(destroy, vk::PFN_vkDestroyQueryPool),
        b"vkGetQueryPoolResults" => f!(results, vk::PFN_vkGetQueryPoolResults),
        b"vkResetQueryPool" | b"vkResetQueryPoolEXT" => {
            f!(host_reset, vk::PFN_vkResetQueryPool)
        }
        b"vkCmdResetQueryPool" => f!(reset, vk::PFN_vkCmdResetQueryPool),
        b"vkCmdWriteTimestamp" => f!(timestamp, vk::PFN_vkCmdWriteTimestamp),
        b"vkCmdCopyQueryPoolResults" => f!(copy, vk::PFN_vkCmdCopyQueryPoolResults),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn query_results_validate_ranges_widths_stride_and_overflow() {
        let flags = vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY;
        assert_eq!(shape(1, 2, 24, flags), Ok((8, 40)));
        assert_eq!(
            shape(1, 2, 12, vk::QueryResultFlags::WITH_AVAILABILITY),
            Ok((4, 20))
        );
        assert!(shape(1, 2, 8, flags).is_err());
        assert!(shape(1, 2, 17, flags).is_err());
        assert!(shape(1, 3, u64::MAX - 7, flags).is_err());
        assert!(shape(1, 1, 16, flags | vk::QueryResultFlags::PARTIAL).is_err());
        let pool = Pool {
            kind: 2,
            statistics: 0,
            values: Mutex::new(vec![Some(vec![9]), None]),
            changed: Condvar::new(),
        };
        assert!(pool.range(1, 1).is_ok());
        assert!(pool.range(1, 2).is_err());
        assert!(pool.range(u32::MAX, 2).is_err());
        pool.reset(0, 1);
        assert_eq!(*pool.values.lock().unwrap(), vec![None, None]);
    }
    #[test]
    fn query_reset_and_copy_are_rejected_during_rendering() {
        let pool = Arc::new(Pool {
            kind: 2,
            statistics: 0,
            values: Mutex::new(vec![None]),
            changed: Condvar::new(),
        });
        let mut r = Recording::new(0);
        r.state = RecordingState::Recording;
        r.render_active = true;
        assert_eq!(
            r.push(RecordedCommand::Query(Action::Reset(pool, 0, 1))),
            Err(INVALID)
        );
    }
    fn scoped(kind: u32) -> Arc<Pool> {
        Arc::new(Pool {
            kind,
            statistics: if kind == 1 { 0x85 } else { 0 },
            values: Mutex::new(vec![None; 2]),
            changed: Condvar::new(),
        })
    }
    #[test]
    fn scoped_queries_reject_nested_mismatched_reset_and_render_scope_end() {
        let p = scoped(0);
        let other = scoped(0);
        let recording = || {
            let mut r = Recording::new(0);
            r.state = RecordingState::Recording;
            r
        };
        let mut r = recording();
        r.push(RecordedCommand::Query(Action::Begin(p.clone(), 0)))
            .unwrap();
        assert_eq!(
            r.push(RecordedCommand::Query(Action::Begin(p.clone(), 1))),
            Err(INVALID)
        );
        let mut r = recording();
        r.push(RecordedCommand::Query(Action::Begin(p.clone(), 0)))
            .unwrap();
        assert_eq!(
            r.push(RecordedCommand::Query(Action::End(other, 0))),
            Err(INVALID)
        );
        let mut r = recording();
        r.push(RecordedCommand::Query(Action::Begin(p.clone(), 0)))
            .unwrap();
        assert_eq!(
            r.push(RecordedCommand::Query(Action::Reset(p.clone(), 0, 1))),
            Err(INVALID)
        );
        let mut r = recording();
        r.render_active = true;
        r.push(RecordedCommand::Query(Action::Begin(p.clone(), 0)))
            .unwrap();
        assert_eq!(r.push(RecordedCommand::EndRenderPass), Err(INVALID));
        let mut r = recording();
        r.render_active = true;
        r.push(RecordedCommand::Query(Action::Begin(p.clone(), 0)))
            .unwrap();
        r.push(RecordedCommand::Query(Action::End(p, 0))).unwrap();
        assert!(r.active_queries.is_empty());
    }
    #[test]
    fn statistics_rows_include_each_requested_counter_before_availability() {
        let p = scoped(1);
        assert_eq!(p.words(), 3);
        let flags = vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY;
        assert_eq!(shape(p.words(), 2, 40, flags), Ok((8, 72)));
        assert!(shape(p.words(), 2, 24, flags).is_err());
        assert_eq!(shape(11, 2, 104, flags), Ok((8, 200)));
    }
}
