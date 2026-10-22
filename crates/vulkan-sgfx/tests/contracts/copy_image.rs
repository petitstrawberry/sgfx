//! Exact raw GPU image transfers, including hidden RGB at zero alpha and sRGB.
use super::*;

#[test]
#[ignore = "requires a native GPU and freshly built SGFX ICD"]
fn image_copy_keeps_all_bits_alpha_channels_and_untouched_regions() {
    let context = Context::new();
    let source_upload = Storage::with_usage(&context, vk::BufferUsageFlags::TRANSFER_SRC);
    let target_upload = Storage::with_usage(&context, vk::BufferUsageFlags::TRANSFER_SRC);
    let output = Storage::with_usage(&context, vk::BufferUsageFlags::TRANSFER_DST);
    let source: Vec<u32> = (0..64)
        .map(|i| {
            let alpha = (i * 31 % 192 + 63) as u8;
            u32::from_le_bytes([
                (i * 7 % 63) as u8,
                (i * 11 % 63) as u8,
                (i * 17 % 63) as u8,
                alpha,
            ])
        })
        .enumerate()
        .map(|(i, p)| if i == 0 { 0x00123456 } else { p })
        .collect();
    let sentinel = 0x59472f1fu32;
    unsafe {
        let d = &context.device;
        for (storage, words) in [
            (&source_upload, source.clone()),
            (&target_upload, vec![sentinel; 64]),
        ] {
            let mapped = d
                .map_memory(storage.memory, 0, 256, vk::MemoryMapFlags::empty())
                .unwrap()
                .cast::<u32>();
            std::ptr::copy_nonoverlapping(words.as_ptr(), mapped, words.len());
            d.unmap_memory(storage.memory);
        }
        for format in [
            vk::Format::B8G8R8A8_UNORM,
            vk::Format::R8G8B8A8_UNORM,
            vk::Format::R8G8B8A8_SRGB,
        ] {
            let mut allocations = Vec::new();
            for _ in 0..2 {
                let image = d
                    .create_image(
                        &vk::ImageCreateInfo::default()
                            .image_type(vk::ImageType::TYPE_2D)
                            .format(format)
                            .extent(vk::Extent3D {
                                width: 8,
                                height: 8,
                                depth: 1,
                            })
                            .mip_levels(1)
                            .array_layers(1)
                            .samples(vk::SampleCountFlags::TYPE_1)
                            .tiling(vk::ImageTiling::OPTIMAL)
                            .usage(
                                vk::ImageUsageFlags::TRANSFER_SRC
                                    | vk::ImageUsageFlags::TRANSFER_DST,
                            ),
                        None,
                    )
                    .unwrap();
                let req = d.get_image_memory_requirements(image);
                let memory = d
                    .allocate_memory(
                        &vk::MemoryAllocateInfo::default()
                            .allocation_size(req.size)
                            .memory_type_index(req.memory_type_bits.trailing_zeros()),
                        None,
                    )
                    .unwrap();
                d.bind_image_memory(image, memory, 0).unwrap();
                allocations.push((image, memory));
            }
            let src = allocations[0].0;
            let dst = allocations[1].0;
            let sub = vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1);
            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1);
            let barrier = |image, old, new, read, write| {
                vk::ImageMemoryBarrier::default()
                    .image(image)
                    .old_layout(old)
                    .new_layout(new)
                    .src_access_mask(read)
                    .dst_access_mask(write)
                    .subresource_range(range)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            };
            context.reset();
            d.begin_command_buffer(context.command, &vk::CommandBufferBeginInfo::default())
                .unwrap();
            d.cmd_pipeline_barrier(
                context.command,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[src, dst].map(|image| {
                    barrier(
                        image,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::AccessFlags::empty(),
                        vk::AccessFlags::TRANSFER_WRITE,
                    )
                }),
            );
            let full = vk::BufferImageCopy::default()
                .image_subresource(sub)
                .image_extent(vk::Extent3D {
                    width: 8,
                    height: 8,
                    depth: 1,
                });
            for (image, buffer) in [(src, source_upload.buffer), (dst, target_upload.buffer)] {
                d.cmd_copy_buffer_to_image(
                    context.command,
                    buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[full],
                );
            }
            d.cmd_pipeline_barrier(
                context.command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[
                    barrier(
                        src,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::TRANSFER_READ,
                    ),
                    barrier(
                        dst,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::TRANSFER_WRITE,
                    ),
                ],
            );
            let region = |sx, sy, dx, dy, width, height| {
                vk::ImageCopy::default()
                    .src_subresource(sub)
                    .dst_subresource(sub)
                    .src_offset(vk::Offset3D { x: sx, y: sy, z: 0 })
                    .dst_offset(vk::Offset3D { x: dx, y: dy, z: 0 })
                    .extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
            };
            d.cmd_copy_image(
                context.command,
                src,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region(1, 1, 2, 3, 3, 2), region(0, 0, 6, 0, 2, 2)],
            );
            d.cmd_pipeline_barrier(
                context.command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier(
                    dst,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::TRANSFER_READ,
                )],
            );
            d.cmd_copy_image_to_buffer(
                context.command,
                dst,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                output.buffer,
                &[full],
            );
            d.cmd_copy_image_to_buffer(
                context.command,
                src,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                output.buffer,
                &[full.buffer_offset(256)],
            );
            d.end_command_buffer(context.command).unwrap();
            let fence = d
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .unwrap();
            context.submit(fence).unwrap();
            d.wait_for_fences(&[fence], true, 5_000_000_000).unwrap();
            let actual = output.read();
            for y in 0..8 {
                for x in 0..8 {
                    let expected = if (2..5).contains(&x) && (3..5).contains(&y) {
                        source[(y - 3 + 1) * 8 + x - 2 + 1]
                    } else if (6..8).contains(&x) && (0..2).contains(&y) {
                        source[y * 8 + x - 6]
                    } else {
                        sentinel
                    };
                    assert_eq!(
                        actual[y * 8 + x],
                        expected,
                        "{format:?} GPU destination ({x},{y})"
                    );
                    assert_eq!(
                        actual[64 + y * 8 + x],
                        source[y * 8 + x],
                        "{format:?} source was modified"
                    );
                }
            }
            println!(
                "GPU_RAW_COPY_PASS format={format:?} source_pixels=64 destination_pixels=64 hidden_zero_alpha_rgb=preserved"
            );
            d.destroy_fence(fence, None);
            for (image, memory) in allocations {
                d.destroy_image(image, None);
                d.free_memory(memory, None);
            }
        }
    }
}
