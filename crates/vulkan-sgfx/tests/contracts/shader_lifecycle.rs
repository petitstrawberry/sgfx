//! Real Vulkan ownership regression: recycle shader/program slots on one device.
use super::*;

fn stage_words(stage: naga::ShaderStage, entry: &str, colour: &str) -> Vec<u32> {
    let source = format!(
        r#"
        @vertex fn vs(@location(0) position: vec2<f32>) -> @builtin(position) vec4<f32> {{
            return vec4<f32>(position, 0.0, 1.0);
        }}
        @fragment fn fs() -> @location(0) vec4<f32> {{ return {colour}; }}
    "#
    );
    let module = naga::front::wgsl::parse_str(&source).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    naga::back::spv::write_vec(
        &module,
        &info,
        &Default::default(),
        Some(&naga::back::spv::PipelineOptions {
            shader_stage: stage,
            entry_point: entry.into(),
        }),
    )
    .unwrap()
}

#[test]
#[ignore = "requires a native GPU and freshly built SGFX ICD"]
fn shader_modules_and_pipelines_recycle_beyond_256_with_failed_creation_rollback() {
    graphics_lifecycle(false, 0, 0);
}

#[test]
#[ignore = "requires a native GPU and freshly built SGFX ICD"]
fn live_shader_modules_recycle_1024_unique_specialized_pipelines() {
    graphics_lifecycle(true, 0, 0);
}

#[test]
#[ignore = "requires a native GPU and freshly built SGFX ICD"]
fn simultaneous_graphics_pipelines_retain_more_than_256_shader_stages() {
    graphics_lifecycle(false, 160, 0);
}

#[test]
#[ignore = "requires a native GPU and freshly built SGFX ICD"]
fn graphics_resource_reload_overlaps_two_240_pipeline_sets() {
    graphics_lifecycle(false, 0, 240);
}

fn specialized_fragment_words() -> Vec<u32> {
    let mut words = stage_words(
        naga::ShaderStage::Fragment,
        "fs",
        "vec4<f32>(0.25, 0.0, 0.0, 1.0)",
    );
    let mut offset = 5;
    let mut types_start = None;
    let mut spec_id = None;
    while offset < words.len() {
        let opcode = words[offset] & 0xffff;
        let count = (words[offset] >> 16) as usize;
        if types_start.is_none() && (19..=39).contains(&opcode) {
            types_start = Some(offset);
        }
        if opcode == 43 && count == 4 && words[offset + 3] == 0.25f32.to_bits() {
            words[offset] = (4 << 16) | 50; // OpConstant -> OpSpecConstant.
            assert!(spec_id.replace(words[offset + 2]).is_none());
        }
        offset += count;
    }
    let at = types_start.expect("SPIR-V types");
    words.splice(
        at..at,
        [(4 << 16) | 71, spec_id.expect("red scalar constant"), 1, 7],
    );
    words
}

fn graphics_lifecycle(long_lived_modules: bool, keep_live: usize, reload_batch: usize) {
    let context = Context::new();
    let vertices = Storage::with_usage(&context, vk::BufferUsageFlags::VERTEX_BUFFER);
    let output = Storage::with_usage(&context, vk::BufferUsageFlags::TRANSFER_DST);
    let vertex_words = stage_words(
        naga::ShaderStage::Vertex,
        "vs",
        "vec4<f32>(1.0, 0.0, 0.0, 1.0)",
    );
    let mut fragment_words = [
        stage_words(
            naga::ShaderStage::Fragment,
            "fs",
            "vec4<f32>(1.0, 0.0, 0.0, 1.0)",
        ),
        stage_words(
            naga::ShaderStage::Fragment,
            "fs",
            "vec4<f32>(0.0, 0.0, 1.0, 1.0)",
        ),
    ];
    if long_lived_modules {
        fragment_words[0] = specialized_fragment_words();
    }
    unsafe {
        let d = &context.device;
        let packed: Vec<u8> = [(-1.0f32, -1.0f32), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]
            .into_iter()
            .flat_map(|(x, y)| [x.to_le_bytes(), y.to_le_bytes()].concat())
            .collect();
        let mapping = d
            .map_memory(
                vertices.memory,
                0,
                packed.len() as u64,
                vk::MemoryMapFlags::empty(),
            )
            .unwrap();
        std::ptr::copy_nonoverlapping(packed.as_ptr(), mapping.cast::<u8>(), packed.len());
        d.unmap_memory(vertices.memory);
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
        let attachments = [vk::AttachmentDescription::default()
            .format(vk::Format::R8G8B8A8_UNORM)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)];
        let reference = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];
        let subpasses = [vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&reference)];
        let pass = d
            .create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(&attachments)
                    .subpasses(&subpasses),
                None,
            )
            .unwrap();
        let views = [view];
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
        let layout = d
            .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)
            .unwrap();
        let bindings = [vk::VertexInputBindingDescription {
            binding: 0,
            stride: 8,
            input_rate: vk::VertexInputRate::VERTEX,
        }];
        let attributes = [vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32_SFLOAT,
            offset: 0,
        }];
        let vertex = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_STRIP);
        let area = vk::Rect2D {
            offset: Default::default(),
            extent: vk::Extent2D {
                width: 8,
                height: 8,
            },
        };
        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: 8.0,
            height: 8.0,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissors = [area];
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissors);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .line_width(1.0)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let colors = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&colors);
        let create = |shaders: [vk::ShaderModule; 2],
                      fragment_entry: &std::ffi::CStr,
                      specialization: &vk::SpecializationInfo<'_>| {
            let stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(shaders[0])
                    .name(c"vs"),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(shaders[1])
                    .name(fragment_entry)
                    .specialization_info(specialization),
            ];
            d.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[vk::GraphicsPipelineCreateInfo::default()
                    .stages(&stages)
                    .vertex_input_state(&vertex)
                    .input_assembly_state(&assembly)
                    .viewport_state(&viewport)
                    .rasterization_state(&raster)
                    .multisample_state(&multisample)
                    .color_blend_state(&blend)
                    .layout(layout)
                    .render_pass(pass)],
                None,
            )
        };
        let fence = d
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .unwrap();
        let modules = |index| {
            [&vertex_words, &fragment_words[index]].map(|words| {
                d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None)
                    .unwrap()
            })
        };
        let persistent = long_lived_modules.then(|| modules(0));
        // A new VkDevice per iteration would hide the append-only 256-slot leak.
        let mut retained = std::collections::VecDeque::new();
        let iterations = if reload_batch != 0 {
            reload_batch * 3
        } else if keep_live == 0 {
            1024
        } else {
            350
        };
        for iteration in 0..iterations {
            let index = if long_lived_modules { 0 } else { iteration % 2 };
            let red = iteration as f32 / 1023.0;
            let data = red.to_ne_bytes();
            let entries = [vk::SpecializationMapEntry {
                constant_id: 7,
                offset: 0,
                size: 4,
            }];
            let specialization = vk::SpecializationInfo::default()
                .map_entries(&entries)
                .data(&data);
            let expect_failed_creation = |shaders| {
                // Missing entry validation follows both shader specializations.
                let (partial, error) = create(shaders, c"missing_fragment_entry", &specialization)
                    .expect_err("missing fragment entry must fail, not create a pipeline");
                for pipeline in partial {
                    if pipeline != vk::Pipeline::null() {
                        d.destroy_pipeline(pipeline, None);
                    }
                }
                assert_eq!(
                    error,
                    vk::Result::ERROR_INITIALIZATION_FAILED,
                    "failed-create leaked shader slots at iteration {iteration}"
                );
            };
            let shaders = persistent.unwrap_or_else(|| modules(index));
            expect_failed_creation(shaders);
            // A failed attempt must leave the same modules usable for retry.
            let pipeline = create(shaders, c"fs", &specialization).unwrap_or_else(|(_, error)| {
                panic!("pipeline allocation iteration {iteration}: {error:?}")
            })[0];
            // Vulkan pipelines own their compiled code independently of modules.
            if !long_lived_modules {
                for shader in shaders {
                    d.destroy_shader_module(shader, None);
                }
            }
            d.reset_fences(&[fence]).unwrap();
            context.reset();
            d.begin_command_buffer(context.command, &Default::default())
                .unwrap();
            let clears = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 1.0, 0.0, 1.0],
                },
            }];
            d.cmd_begin_render_pass(
                context.command,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(pass)
                    .framebuffer(fb)
                    .render_area(area)
                    .clear_values(&clears),
                vk::SubpassContents::INLINE,
            );
            // While the third set is built, render the retained second set to
            // verify that recycling the first set did not change its bindings.
            let draw_pipeline = if reload_batch != 0 && iteration >= reload_batch * 2 {
                retained[iteration - reload_batch * 2]
            } else {
                pipeline
            };
            d.cmd_bind_pipeline(
                context.command,
                vk::PipelineBindPoint::GRAPHICS,
                draw_pipeline,
            );
            d.cmd_bind_vertex_buffers(context.command, 0, &[vertices.buffer], &[0]);
            d.cmd_draw(context.command, 4, 1, 0, 0);
            d.cmd_end_render_pass(context.command);
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
            context.submit(fence).unwrap();
            // Force more allocations and rollback while the old pipeline is
            // live and its draw may still be in flight. Rollback may not mark the
            // device lost merely because another submission is pending.
            let churn = modules(index);
            expect_failed_creation(churn);
            for shader in churn {
                d.destroy_shader_module(shader, None);
            }
            d.wait_for_fences(&[fence], true, 5_000_000_000).unwrap();
            let pixels = output.read();
            let expected = if long_lived_modules {
                [(red * 255.0).round() as u8, 0, 0, 255]
            } else if iteration % 2 == 0 {
                [255, 0, 0, 255]
            } else {
                [0, 0, 255, 255]
            };
            for (index, pixel) in pixels[..64].iter().enumerate() {
                assert!(
                    pixel
                        .to_le_bytes()
                        .iter()
                        .zip(expected)
                        .all(|(&a, b)| a.abs_diff(b) <= 1),
                    "recycled program changed live pipeline at iteration {iteration}, pixel {index}: {:?} != {expected:?}",
                    pixel.to_le_bytes()
                );
            }
            retained.push_back(pipeline);
            if reload_batch != 0 {
                // Hold all of A and B before retiring A; repeat for B and C.
                if retained.len() == reload_batch * 2 {
                    for retired in retained.drain(..reload_batch) {
                        d.destroy_pipeline(retired, None);
                    }
                }
            } else if retained.len() > keep_live {
                d.destroy_pipeline(retained.pop_front().unwrap(), None);
            }
        }
        for pipeline in retained {
            d.destroy_pipeline(pipeline, None);
        }
        if let Some(shaders) = persistent {
            for shader in shaders {
                d.destroy_shader_module(shader, None);
            }
        }
        d.destroy_fence(fence, None);
        d.destroy_pipeline_layout(layout, None);
        d.destroy_framebuffer(fb, None);
        d.destroy_render_pass(pass, None);
        d.destroy_image_view(view, None);
        d.destroy_image(image, None);
        d.free_memory(memory, None);
    }
}
