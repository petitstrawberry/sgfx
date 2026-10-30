//! GPU execution of Skia's interleaved xy + scalar coverage vertex layout.
use super::*;

#[test]
#[ignore = "requires a native GPU and freshly built SGFX ICD"]
fn packed_xy_scalar_vertex_fetch_draws_distinct_coverage_regions() {
    vertex_contract(false, false, false, false);
}

#[test]
#[ignore = "requires a native GPU and freshly built SGFX ICD"]
fn instance_vertex_fetch_and_builtin_honor_nonzero_first_instance() {
    vertex_contract(true, false, false, false);
}

fn vertex_storage(context: &Context, bytes: u64) -> Storage<'_> {
    unsafe {
        let buffer = context
            .device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(bytes)
                    .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
            .unwrap();
        let requirements = context.device.get_buffer_memory_requirements(buffer);
        let memory_type = context.memory.memory_types[..context.memory.memory_type_count as usize]
            .iter()
            .enumerate()
            .find(|(index, memory)| {
                requirements.memory_type_bits & (1u32 << index) != 0
                    && memory.property_flags.contains(
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )
            })
            .unwrap()
            .0 as u32;
        let memory = context
            .device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type),
                None,
            )
            .unwrap();
        context
            .device
            .bind_buffer_memory(buffer, memory, 0)
            .unwrap();
        Storage {
            context,
            buffer,
            memory,
        }
    }
}

fn vertex_contract(instanced: bool, dynamic: bool, indirect: bool, indexed: bool) {
    query_vertex_contract(instanced, dynamic, indirect, indexed, false)
}
fn query_vertex_contract(
    instanced: bool,
    dynamic: bool,
    indirect: bool,
    indexed: bool,
    measured: bool,
) {
    let context = if dynamic || indirect {
        Context::configured(true)
    } else {
        Context::new()
    };
    let vertices = vertex_storage(&context, if instanced { 72 } else { 144 });
    let output = Storage::with_usage(&context, vk::BufferUsageFlags::TRANSFER_DST);
    let source = if instanced {
        r#"
        struct Output { @builtin(position) position: vec4<f32>, @location(0) coverage: f32, @location(1) instance_value: f32 };
        @vertex fn vs(@location(0) position: vec2<f32>, @location(1) coverage: f32, @location(2) shift: vec2<f32>, @builtin(instance_index) index: u32) -> Output {
            return Output(vec4<f32>(position + shift, 0.0, 1.0), coverage, f32(index - 2u) * 0.25);
        }
        @fragment fn fs(@location(0) coverage: f32, @location(1) instance_value: f32) -> @location(0) vec4<f32> {
            return vec4<f32>(coverage, instance_value, 1.0 - coverage, 1.0);
        }
    "#
    } else {
        r#"
        struct Output { @builtin(position) position: vec4<f32>, @location(0) coverage: f32 };
        @vertex fn vs(@location(0) position: vec2<f32>, @location(1) coverage: f32) -> Output {
            return Output(vec4<f32>(position, 0.0, 1.0), coverage);
        }
        @fragment fn fs(@location(0) coverage: f32) -> @location(0) vec4<f32> {
            return vec4<f32>(coverage, 0.0, 1.0 - coverage, 1.0);
        }
    "#
    };
    let module = naga::front::wgsl::parse_str(source).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    // As with separate GLSL stage compilation, keep each SPIR-V module's
    // interface scoped to its entry point rather than combining stage inputs.
    let words = [
        (naga::ShaderStage::Vertex, "vs"),
        (naga::ShaderStage::Fragment, "fs"),
    ]
    .map(|(shader_stage, entry)| {
        naga::back::spv::write_vec(
            &module,
            &info,
            &Default::default(),
            Some(&naga::back::spv::PipelineOptions {
                shader_stage,
                entry_point: entry.into(),
            }),
        )
        .unwrap()
    });
    unsafe {
        let d = &context.device;
        let mut packed = Vec::<f32>::new();
        let regions = if instanced {
            vec![(-1.0, 0.0, 0.0)]
        } else {
            vec![(-1.0, 0.0, 0.25), (0.0, 1.0, 0.75)]
        };
        for (left, right, coverage) in regions {
            for (x, y) in [
                (left, -1.0),
                (right, -1.0),
                (left, 1.0),
                (left, 1.0),
                (right, -1.0),
                (right, 1.0),
            ] {
                packed.extend([x, y, coverage]);
            }
        }
        let mapping = d
            .map_memory(
                vertices.memory,
                0,
                (packed.len() * 4) as u64,
                vk::MemoryMapFlags::empty(),
            )
            .unwrap()
            .cast::<f32>();
        std::ptr::copy_nonoverlapping(packed.as_ptr(), mapping, packed.len());
        d.unmap_memory(vertices.memory);
        let instances = instanced.then(|| {
            let instances = vertex_storage(&context, 48);
            let values = [
                1000.0f32, 1000.0, 0.0, 1000.0, 1000.0, 0.0, 0.0, 0.0, 0.25, 1.0, 0.0, 0.75,
            ];
            let mapping = d
                .map_memory(instances.memory, 0, 48, vk::MemoryMapFlags::empty())
                .unwrap()
                .cast::<f32>();
            std::ptr::copy_nonoverlapping(values.as_ptr(), mapping, values.len());
            d.unmap_memory(instances.memory);
            instances
        });
        let properties = context.instance.get_physical_device_format_properties(
            context.instance.enumerate_physical_devices().unwrap()[0],
            vk::Format::R32_SFLOAT,
        );
        assert!(
            properties
                .buffer_features
                .contains(vk::FormatFeatureFlags::VERTEX_BUFFER)
        );
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
        let pass = if dynamic {
            vk::RenderPass::null()
        } else {
            d.create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(&attachments)
                    .subpasses(&subpasses),
                None,
            )
            .unwrap()
        };
        let views = [view];
        let fb = if dynamic {
            vk::Framebuffer::null()
        } else {
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
        };
        let shaders = words.each_ref().map(|words| {
            d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None)
                .unwrap()
        });
        let layout = d
            .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)
            .unwrap();
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(shaders[0])
                .name(c"vs"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(shaders[1])
                .name(c"fs"),
        ];
        let mut bindings = vec![vk::VertexInputBindingDescription {
            binding: 0,
            stride: 12,
            input_rate: vk::VertexInputRate::VERTEX,
        }];
        let mut attributes = vec![
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: u32::from(instanced),
                format: vk::Format::R32_SFLOAT,
                offset: 8,
            },
        ];
        if instanced {
            bindings.push(vk::VertexInputBindingDescription {
                binding: 1,
                stride: 12,
                input_rate: vk::VertexInputRate::INSTANCE,
            });
            attributes.push(vk::VertexInputAttributeDescription {
                location: 2,
                binding: 1,
                format: vk::Format::R32G32_SFLOAT,
                offset: 0,
            });
        }
        let vertex = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
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
        let formats = [vk::Format::R8G8B8A8_UNORM];
        let mut rendering =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);
        let pipeline = d
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[{
                    let info = vk::GraphicsPipelineCreateInfo::default()
                        .stages(&stages)
                        .vertex_input_state(&vertex)
                        .input_assembly_state(&assembly)
                        .viewport_state(&viewport)
                        .rasterization_state(&raster)
                        .multisample_state(&multisample)
                        .color_blend_state(&blend)
                        .layout(layout)
                        .render_pass(pass);
                    if dynamic {
                        info.render_pass(vk::RenderPass::null())
                            .push_next(&mut rendering)
                    } else {
                        info
                    }
                }],
                None,
            )
            .unwrap()[0];
        context.reset();
        let supported = context.instance.get_physical_device_features(
            context.instance.enumerate_physical_devices().unwrap()[0],
        );
        let occlusion = if measured {
            Some(
                d.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::OCCLUSION)
                        .query_count(2),
                    None,
                )
                .unwrap(),
            )
        } else {
            None
        };
        let stats = if measured && supported.pipeline_statistics_query == vk::TRUE {
            Some(
                d.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::PIPELINE_STATISTICS)
                        .query_count(1)
                        .pipeline_statistics(
                            vk::QueryPipelineStatisticFlags::INPUT_ASSEMBLY_VERTICES
                                | vk::QueryPipelineStatisticFlags::INPUT_ASSEMBLY_PRIMITIVES
                                | vk::QueryPipelineStatisticFlags::VERTEX_SHADER_INVOCATIONS
                                | vk::QueryPipelineStatisticFlags::FRAGMENT_SHADER_INVOCATIONS,
                        ),
                    None,
                )
                .unwrap(),
            )
        } else {
            None
        };
        d.begin_command_buffer(context.command, &Default::default())
            .unwrap();
        let arguments = Storage::with_usage(
            &context,
            vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        );
        let counter = Storage::with_usage(
            &context,
            vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        );
        let indices = Storage::with_usage(
            &context,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        );
        if let Some(pool) = occlusion {
            d.cmd_reset_query_pool(context.command, pool, 0, 2);
        }
        if let Some(pool) = stats {
            d.cmd_reset_query_pool(context.command, pool, 0, 1);
            d.cmd_begin_query(context.command, pool, 0, vk::QueryControlFlags::empty());
        }
        if indirect {
            let count = if instanced { 6u32 } else { 12 };
            let instances = if instanced { 2 } else { 1 };
            let first_instance = if instanced { 2 } else { 0 };
            let words = if indexed {
                vec![count, instances, 0, u32::MAX, first_instance, 0, 1, 0, 0, 0]
            } else {
                vec![count, instances, 0, first_instance, 0, 1, 0, 0]
            };
            let bytes: Vec<u8> = words.iter().flat_map(|v| v.to_le_bytes()).collect();
            d.cmd_update_buffer(context.command, arguments.buffer, 0, &bytes);
            d.cmd_update_buffer(context.command, counter.buffer, 0, &1u32.to_le_bytes());
            let indices_bytes: Vec<u8> = (1..=12u32).flat_map(|v| v.to_le_bytes()).collect();
            d.cmd_update_buffer(context.command, indices.buffer, 0, &indices_bytes);
        }
        let clears = [vk::ClearValue {
            color: vk::ClearColorValue { float32: [0.0; 4] },
        }];
        let dynamic_api = ash::khr::dynamic_rendering::Device::new(&context.instance, d);
        if dynamic {
            dynamic_api.cmd_begin_rendering(
                context.command,
                &vk::RenderingInfo::default()
                    .render_area(area)
                    .layer_count(1)
                    .color_attachments(&[vk::RenderingAttachmentInfo::default()
                        .image_view(view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(clears[0])]),
            );
        } else {
            d.cmd_begin_render_pass(
                context.command,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(pass)
                    .framebuffer(fb)
                    .render_area(area)
                    .clear_values(&clears),
                vk::SubpassContents::INLINE,
            );
        }
        if let Some(pool) = occlusion {
            d.cmd_begin_query(
                context.command,
                pool,
                0,
                if supported.occlusion_query_precise == vk::TRUE {
                    vk::QueryControlFlags::PRECISE
                } else {
                    vk::QueryControlFlags::empty()
                },
            );
        }
        d.cmd_bind_pipeline(context.command, vk::PipelineBindPoint::GRAPHICS, pipeline);
        if let Some(instances) = instances.as_ref() {
            d.cmd_bind_vertex_buffers(
                context.command,
                0,
                &[vertices.buffer, instances.buffer],
                &[0, 0],
            );
            if !indirect {
                d.cmd_draw(context.command, 6, 2, 0, 2);
            }
        } else {
            d.cmd_bind_vertex_buffers(context.command, 0, &[vertices.buffer], &[0]);
            if !indirect {
                d.cmd_draw(context.command, 12, 1, 0, 0);
            }
        }
        if indirect {
            let api = ash::khr::draw_indirect_count::Device::new(&context.instance, d);
            if indexed {
                d.cmd_bind_index_buffer(context.command, indices.buffer, 0, vk::IndexType::UINT32);
                api.cmd_draw_indexed_indirect_count(
                    context.command,
                    arguments.buffer,
                    0,
                    counter.buffer,
                    0,
                    2,
                    20,
                );
            } else {
                api.cmd_draw_indirect_count(
                    context.command,
                    arguments.buffer,
                    0,
                    counter.buffer,
                    0,
                    2,
                    16,
                );
            }
        }
        if let Some(pool) = occlusion {
            d.cmd_end_query(context.command, pool, 0);
            d.cmd_begin_query(context.command, pool, 1, vk::QueryControlFlags::empty());
            d.cmd_end_query(context.command, pool, 1);
        }
        if dynamic {
            dynamic_api.cmd_end_rendering(context.command);
        } else {
            d.cmd_end_render_pass(context.command);
        }

        if let Some(pool) = stats {
            d.cmd_end_query(context.command, pool, 0);
        }
        if let Some(pool) = occlusion {
            d.cmd_copy_query_pool_results(
                context.command,
                pool,
                0,
                2,
                output.buffer,
                256,
                24,
                vk::QueryResultFlags::TYPE_64
                    | vk::QueryResultFlags::WITH_AVAILABILITY
                    | vk::QueryResultFlags::WAIT,
            );
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
        context.submit(fence).unwrap();
        d.wait_for_fences(&[fence], true, 5_000_000_000).unwrap();
        if let Some(pool) = occlusion {
            let flags = vk::QueryResultFlags::TYPE_64
                | vk::QueryResultFlags::WITH_AVAILABILITY
                | vk::QueryResultFlags::WAIT;
            let mut values = [[0u64; 3]; 2];
            d.get_query_pool_results(pool, 0, &mut values, flags)
                .unwrap();
            if supported.occlusion_query_precise == vk::TRUE {
                assert_eq!(values[0][0], 64);
            } else {
                assert!(values[0][0] > 0);
            }
            assert_eq!(values[0][1], 1);
            assert_eq!(values[1][..2], [0, 1]);
            let bytes = output.read();
            assert_eq!(
                u64::from(bytes[64]) | (u64::from(bytes[65]) << 32),
                values[0][0]
            );
            assert_eq!(bytes[66], 1);
            assert_eq!(bytes[70], 0);
            assert_eq!(bytes[72], 1);
            let mut narrow = [[0u32; 2]; 2];
            d.get_query_pool_results(
                pool,
                0,
                &mut narrow,
                vk::QueryResultFlags::WITH_AVAILABILITY,
            )
            .unwrap();
            assert_eq!(narrow[0], [values[0][0] as u32, 1]);
            d.destroy_query_pool(pool, None);
        }
        if let Some(pool) = stats {
            let mut values = [[0u64; 5]; 1];
            d.get_query_pool_results(
                pool,
                0,
                &mut values,
                vk::QueryResultFlags::TYPE_64
                    | vk::QueryResultFlags::WITH_AVAILABILITY
                    | vk::QueryResultFlags::WAIT,
            )
            .unwrap();
            assert_eq!(&values[0][..2], &[12, 4]);
            assert!(values[0][2] > 0 && values[0][3] > 0);
            assert_eq!(values[0][4], 1);
            d.destroy_query_pool(pool, None);
        }
        let pixels = output.read();
        for y in 0..8 {
            for x in 0..8 {
                let expected = if x < 4 {
                    [64, 0, 191, 255]
                } else {
                    [191, if instanced { 64 } else { 0 }, 64, 255]
                };
                assert_eq!(
                    pixels[y * 8 + x].to_le_bytes(),
                    expected,
                    "GPU pixel ({x}, {y})"
                );
            }
        }
        d.destroy_fence(fence, None);
        d.destroy_pipeline(pipeline, None);
        d.destroy_pipeline_layout(layout, None);
        for shader in shaders {
            d.destroy_shader_module(shader, None);
        }
        d.destroy_framebuffer(fb, None);
        d.destroy_render_pass(pass, None);
        d.destroy_image_view(view, None);
        d.destroy_image(image, None);
        d.free_memory(memory, None);
    }
}

#[test]
#[ignore = "requires the built ICD and a native GPU"]
fn dynamic_rendering_draws_without_renderpass_objects() {
    vertex_contract(false, true, false, false);
}
#[test]
#[ignore = "requires the built ICD and a native GPU"]
fn gpu_written_indirect_count_preserves_first_instance_and_attachments() {
    vertex_contract(true, true, true, false);
}
#[test]
#[ignore = "requires the built ICD and a native GPU"]
fn gpu_written_indexed_indirect_preserves_signed_vertex_offset() {
    vertex_contract(true, true, true, true);
}

#[test]
#[ignore = "requires the built ICD and a native GPU"]
fn scoped_occlusion_and_statistics_measure_indirect_draws_and_empty_scopes() {
    query_vertex_contract(true, true, true, true, true);
}
