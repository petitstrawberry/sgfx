use super::*;
#[path = "../fixtures/input_attachment.rs"]
mod fixture;

#[test]
#[ignore = "requires a native GPU and freshly built SGFX ICD"]
fn deferred_subpasses_preserve_mrt_depth_and_same_pixel_input_reads() {
    let context = Context::new();
    let output = Storage::with_usage(&context, vk::BufferUsageFlags::TRANSFER_DST);
    unsafe {
        let d = &context.device;
        let area = vk::Rect2D {
            offset: Default::default(),
            extent: vk::Extent2D {
                width: 8,
                height: 8,
            },
        };
        let mut images = Vec::new();
        let mut memories = Vec::new();
        let mut views = Vec::new();
        let mut attachments = Vec::new();
        for index in 0..5 {
            let depth = index == 2;
            let format = if depth {
                vk::Format::D32_SFLOAT
            } else {
                vk::Format::R8G8B8A8_UNORM
            };
            let usage = if depth {
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    | vk::ImageUsageFlags::INPUT_ATTACHMENT
            } else if index == 4 {
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC
            } else {
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::INPUT_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT
            };
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
                        .usage(usage),
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
                        .format(format)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(if depth {
                                    vk::ImageAspectFlags::DEPTH
                                } else {
                                    vk::ImageAspectFlags::COLOR
                                })
                                .level_count(1)
                                .layer_count(1),
                        ),
                    None,
                )
                .unwrap();
            images.push(image);
            memories.push(memory);
            views.push(view);
            attachments.push(
                vk::AttachmentDescription::default()
                    .format(format)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(if index == 4 {
                        vk::AttachmentStoreOp::STORE
                    } else {
                        vk::AttachmentStoreOp::DONT_CARE
                    })
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(if depth {
                        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                    } else if index == 4 {
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL
                    } else {
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                    }),
            );
        }
        let reference = |attachment, layout| vk::AttachmentReference { attachment, layout };
        let colors0 = [
            reference(0, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            reference(1, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
        ];
        let colors1 = [reference(3, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
        let colors2 = [reference(4, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
        let depth = reference(2, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        let depth_read = reference(2, vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        let inputs1 = [
            reference(0, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            reference(1, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            depth_read,
        ];
        let inputs2 = [reference(3, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let subpasses = [
            vk::SubpassDescription::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&colors0)
                .depth_stencil_attachment(&depth),
            vk::SubpassDescription::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&colors1)
                .depth_stencil_attachment(&depth_read)
                .input_attachments(&inputs1),
            vk::SubpassDescription::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&colors2)
                .depth_stencil_attachment(&depth)
                .input_attachments(&inputs2),
        ];
        let dependencies = (0..2)
            .map(|src| {
                vk::SubpassDependency::default()
                    .src_subpass(src)
                    .dst_subpass(src + 1)
                    .src_stage_mask(vk::PipelineStageFlags::ALL_GRAPHICS)
                    .dst_stage_mask(vk::PipelineStageFlags::ALL_GRAPHICS)
                    .src_access_mask(
                        vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    )
                    .dst_access_mask(
                        vk::AccessFlags::INPUT_ATTACHMENT_READ
                            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
                    )
                    .dependency_flags(vk::DependencyFlags::BY_REGION)
            })
            .collect::<Vec<_>>();
        let pass = d
            .create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(&attachments)
                    .subpasses(&subpasses)
                    .dependencies(&dependencies),
                None,
            )
            .unwrap();
        let fb = d
            .create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(pass)
                    .attachments(&views)
                    .width(8)
                    .height(8)
                    .layers(1),
                None,
            )
            .unwrap();
        let mut set_layouts = Vec::new();
        for count in [3, 1] {
            let bindings = (0..count)
                .map(|binding| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(binding)
                        .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                })
                .collect::<Vec<_>>();
            set_layouts.push(
                d.create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .unwrap(),
            );
        }
        let sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::INPUT_ATTACHMENT,
            descriptor_count: 4,
        }];
        let pool = d
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(2)
                    .pool_sizes(&sizes),
                None,
            )
            .unwrap();
        let sets = d
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&set_layouts),
            )
            .unwrap();
        for (set, indices) in [(sets[0], vec![0, 1, 2]), (sets[1], vec![3])] {
            for (binding, index) in indices.into_iter().enumerate() {
                let infos = [vk::DescriptorImageInfo::default()
                    .image_view(views[index])
                    .image_layout(if index == 2 {
                        vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
                    } else {
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                    })];
                d.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(binding as u32)
                        .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                        .image_info(&infos)],
                    &[],
                );
            }
        }
        let module=naga::front::wgsl::parse_str(r#"
            @vertex fn vertex(@builtin(vertex_index) i:u32)->@builtin(position) vec4<f32> {
                var p=array<vec2<f32>,3>(vec2(-1.0,-1.0),vec2(3.0,-1.0),vec2(-1.0,3.0));return vec4(p[i],0.25,1.0);
            }
            struct GBuffer { @location(0) color:vec4<f32>,@location(1) normal:vec2<f32> }
            @fragment fn geometry(@builtin(position) p:vec4<f32>)->GBuffer {return GBuffer(vec4(p.x/8.0,0.0,0.0,1.0),vec2(0.0,p.y/8.0));}
        "#).unwrap();
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
        let words = naga::back::spv::write_vec(&module, &info, &Default::default(), None).unwrap();
        let geometry = d
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            .unwrap();
        let fragments = [
            geometry,
            d.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&fixture::input_fragment(3)),
                None,
            )
            .unwrap(),
            d.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&fixture::input_fragment(1)),
                None,
            )
            .unwrap(),
        ];
        let mut layouts = Vec::new();
        let mut pipelines = Vec::new();
        for subpass in 0..3 {
            let sets = if subpass == 0 {
                vec![]
            } else {
                vec![set_layouts[subpass - 1]]
            };
            let layout = d
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&sets),
                    None,
                )
                .unwrap();
            layouts.push(layout);
            let stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(geometry)
                    .name(c"vertex"),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(fragments[subpass])
                    .name(if subpass == 0 { c"geometry" } else { c"main" }),
            ];
            let vertex = vk::PipelineVertexInputStateCreateInfo::default();
            let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let viewport = [vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: 8.0,
                height: 8.0,
                min_depth: 0.0,
                max_depth: 1.0,
            }];
            let scissors = [area];
            let vp = vk::PipelineViewportStateCreateInfo::default()
                .viewports(&viewport)
                .scissors(&scissors);
            let raster = vk::PipelineRasterizationStateCreateInfo::default()
                .line_width(1.0)
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE);
            let samples = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let depth = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(subpass == 0)
                .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);
            let colors = vec![
                vk::PipelineColorBlendAttachmentState::default()
                    .color_write_mask(vk::ColorComponentFlags::RGBA);
                if subpass == 0 { 2 } else { 1 }
            ];
            let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&colors);
            pipelines.push(
                d.create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::GraphicsPipelineCreateInfo::default()
                        .stages(&stages)
                        .vertex_input_state(&vertex)
                        .input_assembly_state(&assembly)
                        .viewport_state(&vp)
                        .rasterization_state(&raster)
                        .multisample_state(&samples)
                        .depth_stencil_state(&depth)
                        .color_blend_state(&blend)
                        .layout(layout)
                        .render_pass(pass)
                        .subpass(subpass as u32)],
                    None,
                )
                .unwrap()[0],
            );
        }
        d.begin_command_buffer(context.command, &Default::default())
            .unwrap();
        let mut clears = [vk::ClearValue {
            color: vk::ClearColorValue { float32: [0.0; 4] },
        }; 5];
        clears[2] = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };
        d.cmd_begin_render_pass(
            context.command,
            &vk::RenderPassBeginInfo::default()
                .render_pass(pass)
                .framebuffer(fb)
                .render_area(area)
                .clear_values(&clears),
            vk::SubpassContents::INLINE,
        );
        for subpass in 0..3 {
            if subpass != 0 {
                d.cmd_next_subpass(context.command, vk::SubpassContents::INLINE);
            }
            d.cmd_bind_pipeline(
                context.command,
                vk::PipelineBindPoint::GRAPHICS,
                pipelines[subpass],
            );
            if subpass != 0 {
                d.cmd_bind_descriptor_sets(
                    context.command,
                    vk::PipelineBindPoint::GRAPHICS,
                    layouts[subpass],
                    0,
                    &[sets[subpass - 1]],
                    &[],
                );
            }
            d.cmd_draw(context.command, 3, 1, 0, 0);
        }
        d.cmd_end_render_pass(context.command);
        let region = vk::BufferImageCopy::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D {
                width: 8,
                height: 8,
                depth: 1,
            });
        d.cmd_copy_image_to_buffer(
            context.command,
            images[4],
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            output.buffer,
            &[region],
        );
        d.end_command_buffer(context.command).unwrap();
        context.submit(vk::Fence::null()).unwrap();
        d.queue_wait_idle(context.queue).unwrap();
        let pixels = output.read();
        for y in 0..8 {
            for x in 0..8 {
                let pixel = pixels[y * 8 + x].to_ne_bytes();
                let expected = [
                    ((x as f32 + 0.5) * 255.0 / 8.0).round() as u8,
                    ((y as f32 + 0.5) * 255.0 / 8.0).round() as u8,
                    64,
                    255,
                ];
                assert_eq!(pixel, expected, "pixel {x},{y}");
            }
        }
        d.destroy_framebuffer(fb, None);
        d.destroy_render_pass(pass, None);
        d.destroy_descriptor_pool(pool, None);
        for p in pipelines {
            d.destroy_pipeline(p, None);
        }
        for l in layouts {
            d.destroy_pipeline_layout(l, None);
        }
        for l in set_layouts {
            d.destroy_descriptor_set_layout(l, None);
        }
        for s in fragments {
            d.destroy_shader_module(s, None);
        }
        for ((image, memory), view) in images.into_iter().zip(memories).zip(views) {
            d.destroy_image_view(view, None);
            d.destroy_image(image, None);
            d.free_memory(memory, None);
        }
    }
}
