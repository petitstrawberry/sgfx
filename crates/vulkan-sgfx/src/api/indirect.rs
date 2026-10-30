//! Resolve indirect parameters from completed GPU work, then resume the pass.
//! This portable path preserves GPU-produced counts and first-instance values.
use super::*;
const INVALID: vk::Result = vk::Result::ERROR_INITIALIZATION_FAILED;

#[derive(Clone)]
pub(super) struct Draw {
    buffer: vk::Buffer,
    offset: u64,
    count: u32,
    stride: u32,
    indexed: bool,
    counter: Option<(vk::Buffer, u64)>,
    first_instance_enabled: bool,
}
#[derive(Clone)]
pub(super) struct Read {
    draw: Draw,
    pub commands: Arc<Mutex<Vec<ir::OwnedCommand>>>,
}

pub(super) fn resolve(rt: &mut Runtime, rec: &mut ResolvedRecording, draw: &Draw) -> VkResult<()> {
    let size = if draw.indexed { 20u64 } else { 16 };
    let buffer = rt.resources.buffers.get(&draw.buffer).ok_or(INVALID)?;
    let bytes = if draw.count == 0 {
        0
    } else {
        u64::from(draw.count - 1)
            .checked_mul(u64::from(draw.stride))
            .and_then(|n| n.checked_add(size))
            .ok_or(INVALID)?
    };
    if buffer.bound.is_none()
        || !buffer.usage.contains(vk::BufferUsageFlags::INDIRECT_BUFFER)
        || draw
            .offset
            .checked_add(bytes)
            .is_none_or(|end| end > buffer.size)
    {
        return Err(INVALID);
    }
    rec.used_buffers.push(draw.buffer);
    if let Some((buffer, offset)) = draw.counter {
        let data = rt.resources.buffers.get(&buffer).ok_or(INVALID)?;
        if data.bound.is_none()
            || !data.usage.contains(vk::BufferUsageFlags::INDIRECT_BUFFER)
            || offset.checked_add(4).is_none_or(|end| end > data.size)
        {
            return Err(INVALID);
        }
        rec.used_buffers.push(buffer);
    }
    if draw.count == 0 {
        return Ok(());
    }
    // Materialize and retain the same pipeline/descriptor state as direct draws.
    let active = render_passes::suspend(rec)?;
    let commands = Arc::new(Mutex::new(Vec::new()));
    rec.synchronizations.push(synchronization2::Insertion {
        position: rec.ops.len(),
        order: rec.order,
        action: synchronization2::Action::IndirectRead(Read {
            draw: draw.clone(),
            commands: commands.clone(),
        }),
    });
    render_passes::resume(rec, active)?;
    let visible = graphics_bindings(&rt.table, &mut rt.resources, rec)?;
    if draw.indexed {
        let (handle, offset, format) = rec.index_buffer.ok_or(INVALID)?;
        let data = rt.resources.buffers.get(&handle).ok_or(INVALID)?;
        emit_changed(
            &mut rec.emitted_graphics.index_buffer,
            (data.id, offset, format),
            &mut rec.ops,
            |(buffer, offset, format)| ir::OwnedCommand::SetIndexBuffer {
                buffer,
                offset,
                format,
            },
        );
    }
    if visible {
        rec.synchronizations.push(synchronization2::Insertion {
            position: rec.ops.len(),
            order: rec.order,
            action: synchronization2::Action::IndirectDraw(commands),
        });
    }
    Ok(())
}

pub(super) fn read(rt: &mut Runtime, read: &Read) -> VkResult<()> {
    let d = &read.draw;
    let mut count = d.count;
    if let Some((handle, offset)) = d.counter {
        let data = rt.resources.buffers.get(&handle).ok_or(INVALID)?;
        let bytes = rt
            .cache
            .read_buffer(data.id, offset, 4)
            .map_err(crate::runtime::backend_failure)?;
        count = count.min(u32::from_le_bytes(bytes[..4].try_into().unwrap()));
    }
    let mut commands = Vec::new();
    if count != 0 {
        let size = if d.indexed { 20 } else { 16 };
        let length = u64::from(count - 1) * u64::from(d.stride) + size;
        let data = rt.resources.buffers.get(&d.buffer).ok_or(INVALID)?;
        let bytes = rt
            .cache
            .read_buffer(data.id, d.offset, length)
            .map_err(crate::runtime::backend_failure)?;
        for index in 0..count {
            let base = index as usize * d.stride as usize;
            let word = |offset: usize| {
                u32::from_le_bytes(bytes[base + offset..base + offset + 4].try_into().unwrap())
            };
            let (vertices, instances, first) = (word(0), word(4), word(8));
            let first_instance = word(if d.indexed { 16 } else { 12 });
            if instances.checked_add(first_instance).is_none()
                || (!d.first_instance_enabled && first_instance != 0)
                || (!d.indexed && vertices.checked_add(first).is_none())
            {
                return Err(INVALID);
            }
            if vertices == 0 || instances == 0 {
                continue;
            }
            if rt.capabilities.supports_shader_draw_parameters() {
                commands.push(ir::OwnedCommand::SetDrawIndex(index));
            }
            commands.push(if d.indexed {
                ir::OwnedCommand::DrawIndexedInstanced {
                    index_count: vertices,
                    first_index: first,
                    base_vertex: word(12) as i32,
                    instance_count: instances,
                    first_instance,
                }
            } else {
                ir::OwnedCommand::DrawInstanced {
                    vertex_count: vertices,
                    first_vertex: first,
                    instance_count: instances,
                    first_instance,
                }
            });
        }
    }
    *read.commands.lock().unwrap_or_else(|e| e.into_inner()) = commands;
    Ok(())
}

unsafe fn record_draw(
    command: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: u64,
    count: u32,
    stride: u32,
    indexed: bool,
    counter: Option<(vk::Buffer, u64)>,
) {
    let Ok(driver) = driver(command.as_raw(), Kind::Command) else {
        return;
    };
    if (count > 1 && !driver.multi_draw_indirect_enabled)
        || (counter.is_some() && !driver.indirect_count_enabled)
    {
        record_error(command, vk::Result::ERROR_FEATURE_NOT_PRESENT);
        return;
    }
    let size = if indexed { 20 } else { 16 };
    if count > 4096
        || !offset.is_multiple_of(4)
        || (count > 1 && (stride < size || !stride.is_multiple_of(4)))
        || counter.is_some_and(|(_, offset)| !offset.is_multiple_of(4))
    {
        record_error(command, INVALID);
        return;
    }
    record(
        command,
        RecordedCommand::DrawIndirect(Draw {
            buffer,
            offset,
            count,
            stride: if count <= 1 { size } else { stride },
            indexed,
            counter,
            first_instance_enabled: driver.indirect_first_instance_enabled,
        }),
    );
}
unsafe extern "system" fn draw(
    command: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: u64,
    count: u32,
    stride: u32,
) {
    record_draw(command, buffer, offset, count, stride, false, None)
}
unsafe extern "system" fn indexed(
    command: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: u64,
    count: u32,
    stride: u32,
) {
    record_draw(command, buffer, offset, count, stride, true, None)
}
unsafe extern "system" fn count(
    command: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: u64,
    counter: vk::Buffer,
    counter_offset: u64,
    max: u32,
    stride: u32,
) {
    record_draw(
        command,
        buffer,
        offset,
        max,
        stride,
        false,
        Some((counter, counter_offset)),
    )
}
unsafe extern "system" fn indexed_count(
    command: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: u64,
    counter: vk::Buffer,
    counter_offset: u64,
    max: u32,
    stride: u32,
) {
    record_draw(
        command,
        buffer,
        offset,
        max,
        stride,
        true,
        Some((counter, counter_offset)),
    )
}
pub(super) fn lookup(name: &CStr) -> vk::PFN_vkVoidFunction {
    macro_rules! f {
        ($f:path,$t:ty) => {
            Some(unsafe { std::mem::transmute::<$t, unsafe extern "system" fn()>($f) })
        };
    }
    match name.to_bytes() {
        b"vkCmdDrawIndirect" => f!(draw, vk::PFN_vkCmdDrawIndirect),
        b"vkCmdDrawIndexedIndirect" => f!(indexed, vk::PFN_vkCmdDrawIndexedIndirect),
        b"vkCmdDrawIndirectCount" | b"vkCmdDrawIndirectCountKHR" => {
            f!(count, vk::PFN_vkCmdDrawIndirectCount)
        }
        b"vkCmdDrawIndexedIndirectCount" | b"vkCmdDrawIndexedIndirectCountKHR" => {
            f!(indexed_count, vk::PFN_vkCmdDrawIndexedIndirectCount)
        }
        _ => None,
    }
}
