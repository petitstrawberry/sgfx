//! Actual GPU readback contract for offset render areas and subpass loads.
use super::*;

#[test]
#[ignore = "requires a native GPU and freshly built SGFX ICD"]
fn partial_render_area_clear_preserves_pixels_outside_and_subpass_loads() {
    let context = Context::new();
    let output = Storage::with_usage(&context, vk::BufferUsageFlags::TRANSFER_DST);
    unsafe {
        let d = &context.device;
        let image = d
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
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
                        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
                    ),
                None,
            )
            .unwrap();
        let requirements = d.get_image_memory_requirements(image);
        let memory = d
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(requirements.memory_type_bits.trailing_zeros()),
                None,
            )
            .unwrap();
        d.bind_image_memory(image, memory, 0).unwrap();
        let view = d
            .create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                    ),
                None,
            )
            .unwrap();
        let colors = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];
        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&colors);
        let single = [subpass];
        let double = [subpass, subpass];
        let dependency = [vk::SubpassDependency::default()
            .src_subpass(0)
            .dst_subpass(1)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .dependency_flags(vk::DependencyFlags::BY_REGION)];
        let attachment = vk::AttachmentDescription::default()
            .format(vk::Format::R8G8B8A8_UNORM)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE);
        let full_pass = d
            .create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(&[attachment
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)])
                    .subpasses(&single),
                None,
            )
            .unwrap();
        let partial_pass = d
            .create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(&[attachment
                        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)])
                    .subpasses(&double)
                    .dependencies(&dependency),
                None,
            )
            .unwrap();
        let views = [view];
        let framebuffers = [full_pass, partial_pass].map(|pass| {
            d.create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(pass)
                    .attachments(&views)
                    .width(8)
                    .height(8)
                    .layers(1),
                None,
            )
            .unwrap()
        });
        context.reset();
        d.begin_command_buffer(context.command, &vk::CommandBufferBeginInfo::default())
            .unwrap();
        let full = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D {
                width: 8,
                height: 8,
            },
        };
        let partial = vk::Rect2D {
            offset: vk::Offset2D { x: 2, y: 3 },
            extent: vk::Extent2D {
                width: 4,
                height: 2,
            },
        };
        for (index, (pass, area, color)) in [
            (full_pass, full, [1.0; 4]),
            (partial_pass, partial, [1.0, 0.0, 0.0, 1.0]),
        ]
        .into_iter()
        .enumerate()
        {
            d.cmd_begin_render_pass(
                context.command,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(pass)
                    .framebuffer(framebuffers[index])
                    .render_area(area)
                    .clear_values(&[vk::ClearValue {
                        color: vk::ClearColorValue { float32: color },
                    }]),
                vk::SubpassContents::INLINE,
            );
            if index == 1 {
                d.cmd_next_subpass(context.command, vk::SubpassContents::INLINE);
            }
            d.cmd_end_render_pass(context.command);
        }
        d.cmd_copy_image_to_buffer(
            context.command,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            output.buffer,
            &[vk::BufferImageCopy::default()
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D {
                    width: 8,
                    height: 8,
                    depth: 1,
                })],
        );
        d.end_command_buffer(context.command).unwrap();
        let fence = d
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .unwrap();
        context
            .submit(fence)
            .expect("offset renderArea must be submitted to the GPU");
        d.wait_for_fences(&[fence], true, 5_000_000_000).unwrap();
        let pixels = output.read();
        for y in 0..8 {
            for x in 0..8 {
                let expected = if (2..6).contains(&x) && (3..5).contains(&y) {
                    0xff0000ff
                } else {
                    0xffffffff
                };
                assert_eq!(pixels[y * 8 + x], expected, "GPU pixel ({x}, {y})");
            }
        }
        d.destroy_fence(fence, None);
        for fb in framebuffers {
            d.destroy_framebuffer(fb, None);
        }
        d.destroy_render_pass(partial_pass, None);
        d.destroy_render_pass(full_pass, None);
        d.destroy_image_view(view, None);
        d.destroy_image(image, None);
        d.free_memory(memory, None);
    }
}
