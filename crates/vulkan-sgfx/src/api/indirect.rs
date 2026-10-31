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
    // This proof is captured before resolving this draw's writable descriptors.
    // Readbacks retains preceding writes even when barriers consume written_buffers.
    host_prefix: bool,
    pub commands: Arc<Mutex<Vec<ir::OwnedCommand>>>,
}

impl Draw {
    fn byte_len(&self, count: u32) -> Option<u64> {
        if count == 0 {
            Some(0)
        } else {
            u64::from(count - 1)
                .checked_mul(u64::from(self.stride))?
                .checked_add(if self.indexed { 20 } else { 16 })
        }
    }

    fn range(&self) -> Option<std::ops::Range<u64>> {
        Some(self.offset..self.offset.checked_add(self.byte_len(self.count)?)?)
    }
}

impl Read {
    fn new(
        resources: &crate::resources::Resources,
        writes: &buffer_readback::Readbacks,
        draw: Draw,
    ) -> Self {
        let host_prefix = draw.counter.is_none()
            && draw
                .range()
                .is_some_and(|range| !writes.may_write(resources, draw.buffer, range));
        Self {
            draw,
            host_prefix,
            commands: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

enum Source<'a> {
    Host(&'a [u8]),
    Wait(&'static str),
}

fn source<'a>(resources: &'a crate::resources::Resources, read: &Read) -> Source<'a> {
    if read.draw.counter.is_some() {
        return Source::Wait("indirect::fallback_counter");
    }
    if !read.host_prefix {
        return Source::Wait("indirect::fallback_write_or_unknown");
    }
    let bytes = (|| {
        let (memory, range) =
            buffer_readback::allocation_range(resources, read.draw.buffer, read.draw.range()?)?;
        resources
            .memories
            .get(&memory)?
            .bytes
            .materialized_slice()?
            .get(usize::try_from(range.start).ok()?..usize::try_from(range.end).ok()?)
    })();
    match bytes {
        Some(bytes) => Source::Host(bytes),
        None => Source::Wait("indirect::fallback_mirror"),
    }
}

/// Decide at execution, not resolution: queue::run resolves every recording
/// first. execute() retires and reads back all tracked GPU writes before it
/// returns, so an earlier recording's mirror is current only at this point.
pub(super) fn read_host(
    resources: &crate::resources::Resources,
    read: &Read,
    draw_parameters: bool,
) -> VkResult<bool> {
    match source(resources, read) {
        Source::Host(bytes) => profile_work("indirect::cpu_authored", || {
            let commands = expand(&read.draw, read.draw.count, bytes, draw_parameters)?;
            *read.commands.lock().unwrap_or_else(|e| e.into_inner()) = commands;
            Ok(true)
        }),
        Source::Wait(label) => {
            // Eligibility counts, not fallback wait durations. The original
            // sync::gpu_wait / in_flight_wait scopes still time the fallback.
            profile_work(label, || Ok(false))
        }
    }
}

pub(super) fn resolve(rt: &mut Runtime, rec: &mut ResolvedRecording, draw: &Draw) -> VkResult<()> {
    let buffer = rt.resources.buffers.get(&draw.buffer).ok_or(INVALID)?;
    let bytes = draw.byte_len(draw.count).ok_or(INVALID)?;
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
    let read = Read::new(&rt.resources, &rec.readback_buffers, draw.clone());
    let commands = read.commands.clone();
    rec.synchronizations.push(synchronization2::Insertion {
        position: rec.ops.len(),
        order: rec.order,
        action: synchronization2::Action::IndirectRead(read),
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
    let commands = if count == 0 {
        Vec::new()
    } else {
        let length = d.byte_len(count).ok_or(INVALID)?;
        let data = rt.resources.buffers.get(&d.buffer).ok_or(INVALID)?;
        let bytes = rt
            .cache
            .read_buffer(data.id, d.offset, length)
            .map_err(crate::runtime::backend_failure)?;
        expand(
            d,
            count,
            &bytes,
            rt.capabilities.supports_shader_draw_parameters(),
        )?
    };
    *read.commands.lock().unwrap_or_else(|e| e.into_inner()) = commands;
    Ok(())
}

fn expand(
    d: &Draw,
    count: u32,
    bytes: &[u8],
    draw_parameters: bool,
) -> VkResult<Vec<ir::OwnedCommand>> {
    let needed = usize::try_from(d.byte_len(count).ok_or(INVALID)?).map_err(|_| INVALID)?;
    if count > d.count || bytes.len() < needed {
        return Err(INVALID);
    }
    let mut commands = Vec::new();
    if count != 0 {
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
            if draw_parameters {
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
    Ok(commands)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::{AlignedBytes, Buffer, Memory, Resources};

    fn fixture() -> (Resources, Draw) {
        let table = ir::ResourceTable::new();
        let id = table
            .define_buffer(ir::BufferDesc::new(128, ir::BufferUsage::COPY_DST).unwrap())
            .unwrap()
            .id();
        let memory = vk::DeviceMemory::from_raw(1);
        let mut resources = Resources::new();
        resources.memories.insert(
            memory,
            Memory {
                host_written: true,
                bytes: AlignedBytes::zeroed(256).unwrap(),
                mapped: true,
            },
        );
        for raw in [1, 2] {
            resources.buffers.insert(
                vk::Buffer::from_raw(raw),
                Buffer {
                    id,
                    size: 128,
                    usage: vk::BufferUsageFlags::INDIRECT_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST,
                    bound: Some((memory, 32)),
                },
            );
        }
        let draw = Draw {
            buffer: vk::Buffer::from_raw(1),
            offset: 16,
            count: 1,
            stride: 16,
            indexed: false,
            counter: None,
            first_instance_enabled: true,
        };
        write_words(&mut resources, 48, &[3, 2, 7, 4]);
        (resources, draw)
    }

    fn write_words(resources: &mut Resources, offset: usize, words: &[u32]) {
        let memory = resources
            .memories
            .get_mut(&vk::DeviceMemory::from_raw(1))
            .unwrap();
        for (index, word) in words.iter().enumerate() {
            memory.bytes[offset + index * 4..offset + index * 4 + 4]
                .copy_from_slice(&word.to_le_bytes());
        }
    }

    #[test]
    fn cpu_authored_arguments_use_mirror_and_keep_draw_values() {
        let (resources, draw) = fixture();
        let read = Read::new(&resources, &Default::default(), draw);
        assert!(read_host(&resources, &read, true).unwrap());
        let commands = read.commands.lock().unwrap();
        assert!(matches!(
            commands.as_slice(),
            [
                ir::OwnedCommand::SetDrawIndex(0),
                ir::OwnedCommand::DrawInstanced {
                    vertex_count: 3,
                    instance_count: 2,
                    first_vertex: 7,
                    first_instance: 4,
                }
            ]
        ));
    }

    #[test]
    fn alias_overlap_and_stride_holes_fall_back_but_disjoint_writes_do_not() {
        let (mut resources, mut draw) = fixture();
        // Different binding offsets: target 16..32 resolves to allocation 48..64.
        let alias = vk::Buffer::from_raw(2);
        resources.buffers.get_mut(&alias).unwrap().bound = Some((vk::DeviceMemory::from_raw(1), 0));
        for (offset, blocked) in [(44, false), (48, true), (60, true), (64, false)] {
            let mut writes = buffer_readback::Readbacks::default();
            writes.insert(alias, offset, 4).unwrap();
            let read = Read::new(&resources, &writes, draw.clone());
            assert_eq!(read_host(&resources, &read, false).unwrap(), !blocked);
        }
        let other = vk::DeviceMemory::from_raw(2);
        resources.memories.insert(
            other,
            Memory {
                host_written: true,
                bytes: AlignedBytes::zeroed(256).unwrap(),
                mapped: false,
            },
        );
        resources.buffers.get_mut(&alias).unwrap().bound = Some((other, 0));
        let mut writes = buffer_readback::Readbacks::default();
        writes.insert(alias, 48, 4).unwrap();
        assert!(Read::new(&resources, &writes, draw.clone()).host_prefix);
        resources.buffers.get_mut(&alias).unwrap().bound = Some((vk::DeviceMemory::from_raw(1), 0));
        // A write in stride padding conservatively blocks the covered span.
        draw.count = 2;
        draw.stride = 32;
        let mut writes = buffer_readback::Readbacks::default();
        writes.insert(alias, 68, 4).unwrap();
        assert!(!Read::new(&resources, &writes, draw).host_prefix);
    }

    #[test]
    fn gpu_transfer_write_and_barrier_do_not_authorize_stale_mirror() {
        let (resources, draw) = fixture();
        let mut rec = ResolvedRecording::new();
        buffer_operations::apply_write(
            &resources,
            &mut rec,
            draw.buffer,
            draw.offset,
            [9u32, 1, 0, 0]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect(),
        )
        .unwrap();
        // This is the memory-barrier path's consumption of transient write state.
        let barrier = DeferredBarrier {
            position: rec.ops.len(),
            buffers: std::mem::take(&mut rec.written_buffers),
            after: ir::BufferAccess::Uniform,
        };
        assert_eq!(barrier.buffers, [draw.buffer]);
        assert!(rec.written_buffers.is_empty());
        let read = Read::new(&resources, &rec.readback_buffers, draw);
        assert!(matches!(
            source(&resources, &read),
            Source::Wait("indirect::fallback_write_or_unknown")
        ));
        assert!(!read_host(&resources, &read, false).unwrap());
        assert!(read.commands.lock().unwrap().is_empty());
    }

    #[test]
    fn counter_unknown_writer_and_invalid_ranges_always_fall_back() {
        let (mut resources, mut draw) = fixture();
        draw.counter = Some((draw.buffer, 0));
        let read = Read::new(&resources, &Default::default(), draw.clone());
        assert!(matches!(
            source(&resources, &read),
            Source::Wait("indirect::fallback_counter")
        ));
        draw.counter = None;
        let mut writes = buffer_readback::Readbacks::default();
        writes.insert(vk::Buffer::from_raw(999), 0, 4).unwrap();
        assert!(!Read::new(&resources, &writes, draw.clone()).host_prefix);
        resources.buffers.get_mut(&draw.buffer).unwrap().bound =
            Some((vk::DeviceMemory::from_raw(1), u64::MAX));
        assert!(!Read::new(&resources, &Default::default(), draw.clone()).host_prefix);
        draw.offset = u64::MAX - 4;
        assert!(!Read::new(&resources, &Default::default(), draw).host_prefix);
    }

    #[test]
    fn earlier_recording_readback_is_observed_only_at_execution() {
        let (mut resources, draw) = fixture();
        let mut earlier = ResolvedRecording::new();
        buffer_operations::apply_write(
            &resources,
            &mut earlier,
            draw.buffer,
            draw.offset,
            [11u32, 1, 5, 0]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect(),
        )
        .unwrap();
        // queue::run resolves both recordings before executing the writer.
        let read = Read::new(&resources, &Default::default(), draw);
        assert!(read.commands.lock().unwrap().is_empty());
        // Simulate the earlier execute()'s synchronous backend-to-mirror copy.
        write_words(&mut resources, 48, &[11, 1, 5, 0]);
        assert!(read_host(&resources, &read, false).unwrap());
        assert!(matches!(
            read.commands.lock().unwrap().as_slice(),
            [ir::OwnedCommand::DrawInstanced {
                vertex_count: 11,
                first_vertex: 5,
                ..
            }]
        ));
        // Losing the mirror after resolution cannot turn a proof into a panic.
        resources.memories.clear();
        assert!(!read_host(&resources, &read, false).unwrap());
    }

    #[test]
    fn shared_decoder_preserves_indexed_multidraw_stride_and_validation() {
        let (_, mut draw) = fixture();
        draw.indexed = true;
        draw.count = 3;
        draw.stride = 24;
        let words = [
            0,
            1,
            0,
            0,
            0,
            0xdeadbeef,
            4,
            2,
            3,
            (-7i32) as u32,
            5,
            0xdeadbeef,
            6,
            0,
            0,
            0,
            0,
        ];
        let bytes: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
        let commands = expand(&draw, 3, &bytes, true).unwrap();
        assert!(matches!(
            commands.as_slice(),
            [
                ir::OwnedCommand::SetDrawIndex(1),
                ir::OwnedCommand::DrawIndexedInstanced {
                    index_count: 4,
                    instance_count: 2,
                    first_index: 3,
                    base_vertex: -7,
                    first_instance: 5,
                }
            ]
        ));
        draw.first_instance_enabled = false;
        assert!(expand(&draw, 3, &bytes, true).is_err());
        assert!(expand(&draw, 3, &bytes[..bytes.len() - 1], true).is_err());
        assert!(expand(&draw, 0, &[], true).unwrap().is_empty());
        draw.indexed = false;
        draw.count = 1;
        let overflow: Vec<_> = [u32::MAX, 1, 1, 0]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        assert!(expand(&draw, 1, &overflow, false).is_err());
        let overflow: Vec<_> = [1, u32::MAX, 0, 1]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        assert!(expand(&draw, 1, &overflow, false).is_err());
    }
}
