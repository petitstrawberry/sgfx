//! Snapshot host-visible bytes actually consumed by a submitted recording.

use super::*;
use crate::resources::{DescriptorBinding, Resources};
use std::ops::Range;

type Ranges = BTreeMap<vk::Buffer, Vec<Range<u64>>>;

pub(super) fn descriptor_range(
    size: u64,
    offset: u64,
    range: u64,
    dynamic: u64,
) -> VkResult<Range<u64>> {
    let invalid = vk::Result::ERROR_INITIALIZATION_FAILED;
    let length = if range == vk::WHOLE_SIZE {
        if dynamic != 0 {
            return Err(invalid);
        }
        size.checked_sub(offset).ok_or(invalid)?
    } else {
        range
    };
    let start = offset.checked_add(dynamic).ok_or(invalid)?;
    let end = start
        .checked_add(length)
        .filter(|end| *end <= size)
        .ok_or(invalid)?;
    if length == 0 {
        return Err(invalid);
    }
    Ok(start..end)
}

pub(super) fn dynamic_offset(
    set: &crate::resources::DescriptorSet,
    binding: u32,
    offsets: &[u32],
) -> VkResult<u64> {
    let is_dynamic = |ty: &&vk::DescriptorType| {
        matches!(
            **ty,
            vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
        )
    };
    if set.types.values().filter(is_dynamic).count() != offsets.len() {
        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
    }
    if !set.types.get(&binding).is_some_and(|ty| is_dynamic(&ty)) {
        return Ok(0);
    }
    let index = set
        .types
        .range(..binding)
        .map(|(_, ty)| ty)
        .filter(is_dynamic)
        .count();
    Ok(u64::from(offsets[index]))
}

fn collect(
    resources: &Resources,
    used: &[vk::Buffer],
    descriptors: &[DescriptorInsertion],
) -> VkResult<Ranges> {
    let invalid = vk::Result::ERROR_INITIALIZATION_FAILED;
    let mut ranges = Ranges::new();
    // Vertex/index/indirect and explicit barriers remain conservative until
    // their exact shader fetch bounds are known. A whole use dominates ranges.
    for handle in used {
        let buffer = resources.buffers.get(handle).ok_or(invalid)?;
        ranges
            .entry(*handle)
            .or_insert_with(|| std::iter::once(0..buffer.size).collect());
    }
    for insertion in descriptors {
        let set = resources
            .descriptor_sets
            .get(&insertion.set)
            .ok_or(invalid)?;
        let is_dynamic = |ty: &&vk::DescriptorType| {
            matches!(
                **ty,
                vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                    | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
            )
        };
        if set.types.values().filter(is_dynamic).count() != insertion.dynamic_offsets.len() {
            return Err(invalid);
        }
        for (&binding, value) in &set.bindings {
            let DescriptorBinding::Buffer {
                buffer,
                offset,
                range,
            } = *value
            else {
                continue;
            };
            let slot = set.source_to_ir(binding).ok_or(invalid)?;
            if !insertion
                .layout
                .entries()
                .iter()
                .any(|entry| entry.binding() == slot)
            {
                continue;
            }
            let dynamic = dynamic_offset(set, binding, &insertion.dynamic_offsets)?;
            let size = resources.buffers.get(&buffer).ok_or(invalid)?.size;
            let range = descriptor_range(size, offset, range, dynamic)?;
            // Upload complete dwords, preserving the neighbouring texel bytes.
            let end = range
                .end
                .checked_next_multiple_of(4)
                .ok_or(invalid)?
                .min(size);
            ranges
                .entry(buffer)
                .or_default()
                .push((range.start & !3)..end);
        }
    }
    Ok(ranges)
}

fn merge(ranges: &mut Vec<Range<u64>>) {
    ranges.sort_unstable_by_key(|range| range.start);
    let mut count = 0;
    for index in 0..ranges.len() {
        if count != 0 && ranges[index].start <= ranges[count - 1].end {
            ranges[count - 1].end = ranges[count - 1].end.max(ranges[index].end);
        } else {
            ranges.swap(count, index);
            count += 1;
        }
    }
    ranges.truncate(count);
}

fn snapshot(resources: &Resources, ranges: Ranges) -> VkResult<BufferUploads> {
    let invalid = vk::Result::ERROR_INITIALIZATION_FAILED;
    let mut ops = Vec::new();
    let mut fresh = Vec::new();
    for (handle, mut ranges) in ranges {
        let buffer = resources.buffers.get(&handle).ok_or(invalid)?;
        let Some((memory, offset)) = buffer.bound else {
            continue;
        };
        let allocation = resources.memories.get(&memory).ok_or(invalid)?;
        if !allocation.mapped {
            if resources.uploaded_unmapped_buffers.contains(&handle) {
                continue;
            }
            // This cache tracks fully synchronized buffers. Do not mark a
            // partially uploaded unmapped buffer as fully synchronized.
            ranges.clear();
            ranges.push(0..buffer.size);
            fresh.push((handle, memory));
        }
        merge(&mut ranges);
        for range in ranges {
            let start = offset
                .checked_add(range.start)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or(invalid)?;
            let end = offset
                .checked_add(range.end)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or(invalid)?;
            let mut bytes = allocation.bytes.get(start..end).ok_or(invalid)?.to_vec();
            bytes.resize(bytes.len().next_multiple_of(4), 0);
            ops.push(ir::OwnedCommand::WriteBuffer {
                buffer: buffer.id,
                offset: range.start,
                data: bytes,
            });
        }
    }
    Ok((ops, fresh))
}

pub(super) fn for_recording(
    resources: &Resources,
    used: &[vk::Buffer],
    descriptors: &[DescriptorInsertion],
) -> VkResult<BufferUploads> {
    snapshot(resources, collect(resources, used, descriptors)?)
}

#[cfg(test)]
pub(super) fn whole(resources: &Resources, used: &[vk::Buffer]) -> VkResult<BufferUploads> {
    for_recording(resources, used, &[])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::{AlignedBytes, Buffer, Memory};

    #[test]
    fn recording_collects_dynamic_descriptor_uses_and_preserves_whole_buffer_uses() {
        use crate::resources::DescriptorSet;
        let table = ir::ResourceTable::new();
        let buffer = vk::Buffer::from_raw(1);
        let set = vk::DescriptorSet::from_raw(2);
        let id = table
            .define_buffer(ir::BufferDesc::new(4096, ir::BufferUsage::UNIFORM).unwrap())
            .unwrap()
            .id();
        let layout = ir::BindGroupLayoutDesc::new(vec![ir::BindGroupLayoutEntry::new(
            0,
            ir::ShaderStages::VERTEX,
            ir::BindingType::UniformBuffer,
        )])
        .unwrap();
        let mut resources = Resources::new();
        resources.buffers.insert(
            buffer,
            Buffer {
                id,
                size: 4096,
                usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                bound: None,
            },
        );
        resources.descriptor_sets.insert(
            set,
            DescriptorSet::new(
                HashMap::new(),
                vk::DescriptorPool::null(),
                layout.clone(),
                BTreeMap::from([
                    (0, vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC),
                    (8, vk::DescriptorType::UNIFORM_BUFFER),
                ]),
                BTreeMap::new(),
                HashMap::from([
                    (
                        0,
                        DescriptorBinding::Buffer {
                            buffer,
                            offset: 256,
                            range: 64,
                        },
                    ),
                    // A statically unused descriptor need not reference a live buffer.
                    (
                        8,
                        DescriptorBinding::Buffer {
                            buffer: vk::Buffer::from_raw(999),
                            offset: 0,
                            range: 64,
                        },
                    ),
                ]),
            )
            .unwrap(),
        );
        let insertion = DescriptorInsertion {
            layout,
            position: 0,
            index: 0,
            set,
            dynamic_offsets: vec![512],
        };
        assert_eq!(
            collect(&resources, &[], std::slice::from_ref(&insertion)).unwrap()[&buffer],
            std::slice::from_ref(&(768..832))
        );
        let mut whole = collect(
            &resources,
            &[buffer, buffer],
            std::slice::from_ref(&insertion),
        )
        .unwrap();
        merge(whole.get_mut(&buffer).unwrap());
        assert_eq!(whole[&buffer], std::slice::from_ref(&(0..4096)));
        let invalid = DescriptorInsertion {
            dynamic_offsets: vec![],
            ..insertion
        };
        assert!(collect(&resources, &[], &[invalid]).is_err());
    }

    #[test]
    fn mapped_large_buffer_snapshots_only_merged_descriptor_ranges() {
        let table = ir::ResourceTable::new();
        let size = 64 * 1024 * 1024;
        let id = table
            .define_buffer(ir::BufferDesc::new(size, ir::BufferUsage::COPY_DST).unwrap())
            .unwrap()
            .id();
        let buffer = vk::Buffer::from_raw(1);
        let memory = vk::DeviceMemory::from_raw(2);
        let mut resources = Resources::new();
        let mut bytes = AlignedBytes::zeroed(size as usize + 16).unwrap();
        bytes[272..336].fill(0x5a);
        bytes[size as usize..size as usize + 16].fill(0x6b);
        resources.memories.insert(
            memory,
            Memory {
                bytes,
                mapped: true,
            },
        );
        resources.buffers.insert(
            buffer,
            Buffer {
                id,
                size,
                usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                bound: Some((memory, 16)),
            },
        );
        let ranges = Ranges::from([(buffer, vec![256..288, 272..320, 256..320, size - 16..size])]);
        let (ops, fresh) = snapshot(&resources, ranges).unwrap();
        assert!(fresh.is_empty());
        assert!(matches!(ops.as_slice(), [
            ir::OwnedCommand::WriteBuffer { offset: 256, data: first, .. },
            ir::OwnedCommand::WriteBuffer { offset, data: last, .. },
        ] if *offset == size - 16 && first == &[0x5a; 64] && last == &[0x6b; 16]));
    }

    #[test]
    fn writable_descriptor_readbacks_keep_graphics_and_compute_dynamic_ranges() {
        use crate::resources::DescriptorSet;
        let table = ir::ResourceTable::new();
        let buffer = vk::Buffer::from_raw(1);
        let set = vk::DescriptorSet::from_raw(2);
        let id = table
            .define_buffer(ir::BufferDesc::new(64 * 1024 * 1024, ir::BufferUsage::STORAGE).unwrap())
            .unwrap()
            .id();
        let group_layout = ir::BindGroupLayoutDesc::new(vec![ir::BindGroupLayoutEntry::new(
            0,
            ir::ShaderStages::VERTEX | ir::ShaderStages::COMPUTE,
            ir::BindingType::StorageBuffer { read_only: false },
        )])
        .unwrap();
        let layout = ir::PipelineLayoutDesc::new(vec![group_layout.clone()]).unwrap();
        let mut resources = Resources::new();
        resources.buffers.insert(
            buffer,
            Buffer {
                id,
                size: 64 * 1024 * 1024,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                bound: None,
            },
        );
        resources.descriptor_sets.insert(
            set,
            DescriptorSet::new(
                HashMap::new(),
                vk::DescriptorPool::null(),
                group_layout,
                BTreeMap::from([(0, vk::DescriptorType::STORAGE_BUFFER_DYNAMIC)]),
                BTreeMap::new(),
                HashMap::from([(
                    0,
                    DescriptorBinding::Buffer {
                        buffer,
                        offset: 256,
                        range: 64,
                    },
                )]),
            )
            .unwrap(),
        );
        let mut rec = ResolvedRecording::new();
        rec.graphics_offsets.insert(0, vec![512]);
        rec.compute_offsets.insert(0, vec![1024]);
        mark_writable_descriptor_buffers(
            &resources,
            &mut rec,
            &[(0, set)],
            &layout,
            vk::PipelineBindPoint::GRAPHICS,
        )
        .unwrap();
        mark_writable_descriptor_buffers(
            &resources,
            &mut rec,
            &[(0, set)],
            &layout,
            vk::PipelineBindPoint::COMPUTE,
        )
        .unwrap();
        assert_eq!(rec.written_buffers, [buffer]);
        assert_eq!(
            rec.readback_buffers.into_ranges().collect::<Vec<_>>(),
            [(buffer, vec![768..832, 1280..1344])]
        );
    }

    #[test]
    fn descriptor_ranges_account_for_dynamic_offsets_and_byte_texel_padding() {
        assert_eq!(descriptor_range(4096, 256, 64, 512).unwrap(), 768..832);
        assert_eq!(
            descriptor_range(4096, 256, vk::WHOLE_SIZE, 0).unwrap(),
            256..4096
        );
        assert!(descriptor_range(4096, 256, vk::WHOLE_SIZE, 256).is_err());
        assert!(descriptor_range(4096, 4000, 64, 64).is_err());
        assert!(descriptor_range(4096, u64::MAX, 1, 1).is_err());
        assert_eq!(descriptor_range(5, 0, 5, 0).unwrap(), 0..5);
        assert_eq!(descriptor_range(4096, 256, 5, 0).unwrap(), 256..261);
    }
}
