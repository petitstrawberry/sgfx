//! Snapshot host-visible bytes actually consumed by a submitted recording.

use super::*;
use crate::resources::{DescriptorBinding, Resources};
use std::ops::Range;

pub(super) type Ranges = BTreeMap<vk::Buffer, Vec<Range<u64>>>;

pub(super) fn resolve_copy(
    resources: &Resources,
    rec: &mut ResolvedRecording,
    source: vk::Buffer,
    destination: vk::Buffer,
    regions: &[vk::BufferCopy],
) -> VkResult<()> {
    let invalid = vk::Result::ERROR_INITIALIZATION_FAILED;
    if rec.render.is_some() {
        return Err(invalid);
    }
    let source_data = resources.buffers.get(&source).ok_or(invalid)?;
    let destination_data = resources.buffers.get(&destination).ok_or(invalid)?;
    if source_data.bound.is_none()
        || destination_data.bound.is_none()
        || !source_data
            .usage
            .contains(vk::BufferUsageFlags::TRANSFER_SRC)
        || !destination_data
            .usage
            .contains(vk::BufferUsageFlags::TRANSFER_DST)
    {
        return Err(invalid);
    }
    for region in regions {
        let source_end = region
            .src_offset
            .checked_add(region.size)
            .filter(|end| *end <= source_data.size)
            .ok_or(invalid)?;
        region
            .dst_offset
            .checked_add(region.size)
            .filter(|end| *end <= destination_data.size)
            .ok_or(invalid)?;
        rec.ops.push(ir::OwnedCommand::CopyBufferToBuffer {
            source: source_data.id,
            source_offset: region.src_offset,
            destination: destination_data.id,
            destination_offset: region.dst_offset,
            size: region.size,
        });
        rec.source_upload_ranges
            .entry(source)
            .or_default()
            .push(region.src_offset..source_end);
        rec.readback_buffers
            .insert(destination, region.dst_offset, region.size)?;
    }
    // Object liveness is separate from host synchronization. A destination
    // needs its copied ranges read back, not an implicit whole host upload.
    rec.referenced_buffers.extend([source, destination]);
    Ok(())
}

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
    sources: &Ranges,
) -> VkResult<Ranges> {
    let invalid = vk::Result::ERROR_INITIALIZATION_FAILED;
    let mut ranges = Ranges::new();
    // Vertex/index/indirect reads remain conservative until
    // their exact shader fetch bounds are known. A whole use dominates ranges.
    for handle in used {
        let buffer = resources.buffers.get(handle).ok_or(invalid)?;
        ranges
            .entry(*handle)
            .or_insert_with(|| std::iter::once(0..buffer.size).collect());
    }
    for (handle, source_ranges) in sources {
        let buffer = resources.buffers.get(handle).ok_or(invalid)?;
        for range in source_ranges {
            if range.start > range.end || range.end > buffer.size {
                return Err(invalid);
            }
            if range.is_empty() {
                continue;
            }
            // As for descriptor snapshots, upload complete transport dwords
            // using actual neighbouring host bytes, never zeroing a live byte.
            let end = range
                .end
                .checked_next_multiple_of(4)
                .ok_or(invalid)?
                .min(buffer.size);
            ranges
                .entry(*handle)
                .or_default()
                .push((range.start & !3)..end);
        }
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
    let mut measured = crate::progress::host_uploads();
    snapshot_measured(
        resources,
        ranges,
        measured.as_mut().map(|batch| &mut batch.stats),
    )
}

fn snapshot_measured(
    resources: &Resources,
    ranges: Ranges,
    mut measured: Option<&mut crate::progress::HostUploads>,
) -> VkResult<BufferUploads> {
    let _frame = crate::frame_profile::Span::new("work", "upload::host_snapshot");
    let invalid = vk::Result::ERROR_INITIALIZATION_FAILED;
    let mut ops = Vec::new();
    let mut fresh = Vec::new();
    for (handle, mut ranges) in ranges {
        let buffer = resources.buffers.get(&handle).ok_or(invalid)?;
        let Some((memory, offset)) = buffer.bound else {
            continue;
        };
        let allocation = resources.memories.get(&memory).ok_or(invalid)?;
        if !allocation.host_written { continue; }
        merge(&mut ranges);
        let full = ranges.len() == 1 && ranges[0].start == 0 && ranges[0].end == buffer.size;
        if !allocation.mapped {
            if resources.uploaded_unmapped_buffers.contains(&handle) {
                continue;
            }
            // Partial source/descriptor uploads never prove the entire buffer
            // synchronized, even after this recording has completed.
            if full {
                fresh.push((handle, memory));
            }
        }
        let mut buffer_bytes = 0;
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
            if let Some(stats) = measured.as_mut() {
                stats.copied_span(bytes.len(), &mut buffer_bytes, full, allocation.mapped);
            }
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
    sources: &Ranges,
) -> VkResult<BufferUploads> {
    snapshot(resources, collect(resources, used, descriptors, sources)?)
}

#[cfg(test)]
pub(super) fn whole(resources: &Resources, used: &[vk::Buffer]) -> VkResult<BufferUploads> {
    for_recording(resources, used, &[], &Ranges::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::{AlignedBytes, Buffer, Memory};

    fn copy_fixture(
        size: u64,
        mapped: bool,
    ) -> (ir::ResourceTable, Resources, vk::Buffer, vk::Buffer) {
        let table = ir::ResourceTable::new();
        let mut resources = Resources::new();
        for raw in [1, 2] {
            let handle = vk::Buffer::from_raw(raw);
            let memory = vk::DeviceMemory::from_raw(raw);
            let id = table
                .define_buffer(
                    ir::BufferDesc::new(
                        size,
                        ir::BufferUsage::COPY_SRC | ir::BufferUsage::COPY_DST,
                    )
                    .unwrap(),
                )
                .unwrap()
                .id();
            let mut bytes = AlignedBytes::zeroed(size as usize).unwrap();
            bytes.fill(raw as u8);
            resources.memories.insert(memory, Memory { bytes, mapped, host_written: true });
            resources.buffers.insert(
                handle,
                Buffer {
                    id,
                    size,
                    usage: vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                    bound: Some((memory, 0)),
                },
            );
        }
        (
            table,
            resources,
            vk::Buffer::from_raw(1),
            vk::Buffer::from_raw(2),
        )
    }

    fn snapshot_recording(
        resources: &Resources,
        rec: &ResolvedRecording,
    ) -> VkResult<BufferUploads> {
        for_recording(
            resources,
            &rec.used_buffers,
            &rec.descriptors,
            &rec.source_upload_ranges,
        )
    }

    fn measured_recording(
        resources: &Resources,
        rec: &ResolvedRecording,
    ) -> (BufferUploads, crate::progress::HostUploads) {
        let mut measured = crate::progress::HostUploads::default();
        let ranges = collect(
            resources,
            &rec.used_buffers,
            &rec.descriptors,
            &rec.source_upload_ranges,
        )
        .unwrap();
        let uploads = snapshot_measured(resources, ranges, Some(&mut measured)).unwrap();
        (uploads, measured)
    }

    #[test]
    fn small_copies_from_64_and_128_mib_buffers_upload_only_source_union() {
        for size in [64 * 1024 * 1024, 128 * 1024 * 1024] {
            let (_, resources, source, destination) = copy_fixture(size, true);
            let mut rec = ResolvedRecording::new();
            let regions = [
                vk::BufferCopy {
                    src_offset: 4096,
                    dst_offset: 0,
                    size: 1024,
                },
                vk::BufferCopy {
                    src_offset: 4608,
                    dst_offset: 2048,
                    size: 1024,
                },
                vk::BufferCopy {
                    src_offset: size - 16,
                    dst_offset: 4096,
                    size: 16,
                },
            ];
            resolve_copy(&resources, &mut rec, source, destination, &regions).unwrap();
            assert!(
                rec.used_buffers.is_empty(),
                "copy references must not become whole host uses"
            );
            assert_eq!(rec.referenced_buffers, [source, destination]);
            assert_eq!(rec.ops.len(), 3);
            let ((ops, fresh), measured) = measured_recording(&resources, &rec);
            assert!(fresh.is_empty());
            assert_eq!(measured.bytes, 1552);
            assert_eq!((measured.full_buffers, measured.partial_buffers), (0, 1));
            assert_eq!(measured.max_buffer_bytes, 1552);
            assert_eq!((measured.mapped_bytes, measured.unmapped_bytes), (1552, 0));
            let source_id = resources.buffers[&source].id;
            assert!(matches!(ops.as_slice(), [
                ir::OwnedCommand::WriteBuffer { buffer: first, offset: 4096, data: a },
                ir::OwnedCommand::WriteBuffer { buffer: last, offset, data: b },
            ] if *first == source_id && *last == source_id && *offset == size - 16 && a == &[1; 1536] && b == &[1; 16]));
            assert_eq!(
                rec.readback_buffers.into_ranges().collect::<Vec<_>>(),
                [(destination, vec![0..1024, 2048..3072, 4096..4112])]
            );
        }
    }

    #[test]
    fn destination_with_a_real_whole_use_still_uploads_all_untouched_cpu_bytes() {
        let (_, resources, source, destination) = copy_fixture(32, true);
        let mut rec = ResolvedRecording::new();
        resolve_copy(
            &resources,
            &mut rec,
            source,
            destination,
            &[vk::BufferCopy {
                src_offset: 4,
                dst_offset: 8,
                size: 4,
            }],
        )
        .unwrap();
        rec.used_buffers.push(destination); // e.g. an index/vertex use or unknown barrier scope
        let ((ops, _), measured) = measured_recording(&resources, &rec);
        assert_eq!(measured.bytes, 36);
        assert_eq!((measured.full_buffers, measured.partial_buffers), (1, 1));
        assert_eq!(measured.max_buffer_bytes, 32);
        assert!(matches!(ops.as_slice(), [
            ir::OwnedCommand::WriteBuffer { offset: 4, data: a, .. },
            ir::OwnedCommand::WriteBuffer { offset: 0, data: b, .. },
        ] if a == &[1; 4] && b == &[2; 32]));
    }

    #[test]
    fn measured_spans_exclude_cached_buffers_padding_and_failed_reads() {
        let (_, mut resources, source, destination) = copy_fixture(8, false);
        resources.uploaded_unmapped_buffers.insert(source);
        resources.buffers.get_mut(&destination).unwrap().size = 5;
        let ranges = Ranges::from([
            (source, std::iter::once(0..8).collect()),
            (destination, std::iter::once(0..5).collect()),
        ]);
        let mut measured = crate::progress::HostUploads::default();
        let (ops, _) = snapshot_measured(&resources, ranges, Some(&mut measured)).unwrap();
        assert!(
            matches!(ops.as_slice(), [ir::OwnedCommand::WriteBuffer { data, .. }] if data.len() == 8)
        );
        assert_eq!(
            measured.bytes, 5,
            "count actual cloned bytes, not the padded command payload"
        );
        assert_eq!((measured.full_buffers, measured.partial_buffers), (1, 0));
        assert_eq!((measured.max_buffer_bytes, measured.unmapped_bytes), (5, 5));
        let mut measured = crate::progress::HostUploads::default();
        let ranges = Ranges::from([(destination, vec![0..2, 16..20])]);
        assert!(snapshot_measured(&resources, ranges, Some(&mut measured)).is_err());
        assert_eq!(
            measured.bytes, 2,
            "an error does not erase bytes already cloned"
        );
        assert_eq!((measured.full_buffers, measured.partial_buffers), (0, 1));
        assert_eq!(measured.max_buffer_bytes, 2);
    }

    #[test]
    fn barriers_keep_validation_references_without_promoting_copy_host_ranges() {
        let (table, resources, source, destination) = copy_fixture(32, true);
        let mut rec = ResolvedRecording::new();
        resolve_copy(
            &resources,
            &mut rec,
            source,
            destination,
            &[vk::BufferCopy {
                src_offset: 4,
                dst_offset: 8,
                size: 4,
            }],
        )
        .unwrap();
        for handle in [source, destination] {
            // PipelineBarrier retains this reference and the original IR,
            // rather than declaring a whole-buffer host read.
            rec.referenced_buffers.push(handle);
            rec.ops.push(ir::OwnedCommand::ResourceBarrier(
                ir::OwnedResourceBarrier::Buffer {
                    buffer: resources.buffers[&handle].id,
                    before: ir::BufferAccess::CopyDestination,
                    after: ir::BufferAccess::CopySource,
                },
            ));
        }
        let (mut ops, _) = snapshot_recording(&resources, &rec).unwrap();
        assert!(
            matches!(ops.as_slice(), [ir::OwnedCommand::WriteBuffer { offset: 4, data, .. }] if data == &[1; 4])
        );
        ops.extend(rec.ops);
        assert_eq!(
            ops.iter()
                .filter(|op| matches!(op, ir::OwnedCommand::ResourceBarrier(_)))
                .count(),
            2
        );
        ir::OwnedCommandBuffer::new(ops).record(&table).unwrap();
    }

    #[test]
    fn mapped_copy_sources_reread_the_requested_bytes_on_every_submission() {
        let (_, mut resources, source, destination) = copy_fixture(32, true);
        let mut rec = ResolvedRecording::new();
        resolve_copy(
            &resources,
            &mut rec,
            source,
            destination,
            &[vk::BufferCopy {
                src_offset: 4,
                dst_offset: 8,
                size: 4,
            }],
        )
        .unwrap();
        resources.uploaded_unmapped_buffers.insert(source);
        for value in [3, 9] {
            resources
                .memories
                .get_mut(&vk::DeviceMemory::from_raw(1))
                .unwrap()
                .bytes[4..8]
                .fill(value);
            let (ops, fresh) = snapshot_recording(&resources, &rec).unwrap();
            assert!(fresh.is_empty());
            assert!(
                matches!(ops.as_slice(), [ir::OwnedCommand::WriteBuffer { offset: 4, data, .. }] if data == &[value; 4])
            );
        }
    }

    #[test]
    fn partial_unmapped_uploads_never_skip_a_second_range_or_mark_full_sync() {
        let (_, mut resources, source, destination) = copy_fixture(32, false);
        for offset in [4, 20] {
            let mut rec = ResolvedRecording::new();
            resolve_copy(
                &resources,
                &mut rec,
                source,
                destination,
                &[vk::BufferCopy {
                    src_offset: offset,
                    dst_offset: 0,
                    size: 4,
                }],
            )
            .unwrap();
            let (ops, fresh) = snapshot_recording(&resources, &rec).unwrap();
            assert!(fresh.is_empty());
            assert!(
                matches!(ops.as_slice(), [ir::OwnedCommand::WriteBuffer { offset: actual, data, .. }] if *actual == offset && data.len() == 4)
            );
            assert!(!resources.uploaded_unmapped_buffers.contains(&source));
        }
        let mut rec = ResolvedRecording::new();
        resolve_copy(
            &resources,
            &mut rec,
            source,
            destination,
            &[
                vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: 16,
                },
                vk::BufferCopy {
                    src_offset: 16,
                    dst_offset: 16,
                    size: 16,
                },
            ],
        )
        .unwrap();
        let (ops, fresh) = snapshot_recording(&resources, &rec).unwrap();
        assert!(
            matches!(ops.as_slice(), [ir::OwnedCommand::WriteBuffer { offset: 0, data, .. }] if data.len() == 32)
        );
        assert_eq!(fresh, [(source, vk::DeviceMemory::from_raw(1))]);
        resources.uploaded_unmapped_buffers.insert(source); // successful whole upload publication
        assert!(snapshot_recording(&resources, &rec).unwrap().0.is_empty());
    }

    #[test]
    fn partial_destination_readback_preserves_cpu_tail_and_invalidates_alias_cache() {
        let (_, mut resources, source, destination) = copy_fixture(32, false);
        let alias = vk::Buffer::from_raw(3);
        let data = &resources.buffers[&destination];
        resources.buffers.insert(
            alias,
            Buffer {
                id: data.id,
                size: data.size,
                usage: data.usage,
                bound: data.bound,
            },
        );
        resources.uploaded_unmapped_buffers.insert(alias);
        let mut rec = ResolvedRecording::new();
        resolve_copy(
            &resources,
            &mut rec,
            source,
            destination,
            &[vk::BufferCopy {
                src_offset: 4,
                dst_offset: 8,
                size: 4,
            }],
        )
        .unwrap();
        assert_eq!(snapshot_recording(&resources, &rec).unwrap().0.len(), 1);
        // execute() reads only the copy destination bytes into the shared mirror.
        let memory = vk::DeviceMemory::from_raw(2);
        resources.memories.get_mut(&memory).unwrap().bytes[8..12].fill(1);
        resources.forget_uploads_for_memory_range(memory, 8..12, destination);
        assert!(!resources.uploaded_unmapped_buffers.contains(&alias));
        let mut next = ResolvedRecording::new();
        next.used_buffers.push(alias);
        let (ops, fresh) = snapshot_recording(&resources, &next).unwrap();
        let ir::OwnedCommand::WriteBuffer { data, .. } = &ops[0] else {
            panic!()
        };
        assert_eq!(&data[..8], &[2; 8]);
        assert_eq!(&data[8..12], &[1; 4]);
        assert_eq!(&data[12..], &[2; 20]);
        assert_eq!(fresh, [(alias, memory)]);
    }

    #[test]
    fn gpu_produced_copy_keeps_the_original_write_then_copy_order() {
        let (table, resources, source, destination) = copy_fixture(32, true);
        let mut rec = ResolvedRecording::new();
        buffer_operations::apply_write(&resources, &mut rec, source, 4, vec![9; 4]).unwrap();
        resolve_copy(
            &resources,
            &mut rec,
            source,
            destination,
            &[vk::BufferCopy {
                src_offset: 4,
                dst_offset: 8,
                size: 4,
            }],
        )
        .unwrap();
        let (mut ops, _) = snapshot_recording(&resources, &rec).unwrap();
        assert!(matches!(
            rec.ops.as_slice(),
            [
                ir::OwnedCommand::WriteBuffer { .. },
                ir::OwnedCommand::CopyBufferToBuffer { .. }
            ]
        ));
        ops.extend(rec.ops);
        ir::OwnedCommandBuffer::new(ops.clone())
            .record(&table)
            .unwrap();
        let mut bytes = HashMap::from([
            (resources.buffers[&source].id, vec![0; 32]),
            (resources.buffers[&destination].id, vec![0; 32]),
        ]);
        for op in ops {
            match op {
                ir::OwnedCommand::WriteBuffer {
                    buffer,
                    offset,
                    data,
                } => bytes.get_mut(&buffer).unwrap()[offset as usize..offset as usize + data.len()]
                    .copy_from_slice(&data),
                ir::OwnedCommand::CopyBufferToBuffer {
                    source,
                    source_offset,
                    destination,
                    destination_offset,
                    size,
                } => {
                    let produced = bytes[&source]
                        [source_offset as usize..(source_offset + size) as usize]
                        .to_vec();
                    bytes.get_mut(&destination).unwrap()
                        [destination_offset as usize..(destination_offset + size) as usize]
                        .copy_from_slice(&produced);
                }
                _ => panic!(),
            }
        }
        assert_eq!(&bytes[&resources.buffers[&destination].id][8..12], &[9; 4]);
    }

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
            collect(
                &resources,
                &[],
                std::slice::from_ref(&insertion),
                &Ranges::new()
            )
            .unwrap()[&buffer],
            std::slice::from_ref(&(768..832))
        );
        let sources = Ranges::from([(buffer, vec![800..848, 4092..4096])]);
        let mut union =
            collect(&resources, &[], std::slice::from_ref(&insertion), &sources).unwrap();
        merge(union.get_mut(&buffer).unwrap());
        assert_eq!(union[&buffer], [768..848, 4092..4096]);
        let mut whole = collect(
            &resources,
            &[buffer, buffer],
            std::slice::from_ref(&insertion),
            &Ranges::new(),
        )
        .unwrap();
        merge(whole.get_mut(&buffer).unwrap());
        assert_eq!(whole[&buffer], std::slice::from_ref(&(0..4096)));
        let invalid = DescriptorInsertion {
            dynamic_offsets: vec![],
            ..insertion
        };
        assert!(collect(&resources, &[], &[invalid], &Ranges::new()).is_err());
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
                host_written: true,
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
