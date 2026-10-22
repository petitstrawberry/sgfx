//! GPU execution of Skia's interleaved xy + scalar coverage vertex layout.
use super::*;

#[test]
#[ignore = "requires a native GPU and freshly built SGFX ICD"]
fn packed_xy_scalar_vertex_fetch_draws_distinct_coverage_regions() {
    vertex_contract(false);
}

#[test]
#[ignore = "requires a native GPU and freshly built SGFX ICD"]
fn instance_vertex_fetch_and_builtin_honor_nonzero_first_instance() {
    vertex_contract(true);
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

fn vertex_contract(instanced: bool) {
    let context = Context::new();
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
        let pipeline = d
            .create_graphics_pipelines(
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
            .unwrap()[0];
        context.reset();
        d.begin_command_buffer(context.command, &Default::default())
            .unwrap();
        let clears = [vk::ClearValue {
            color: vk::ClearColorValue { float32: [0.0; 4] },
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
        d.cmd_bind_pipeline(context.command, vk::PipelineBindPoint::GRAPHICS, pipeline);
        if let Some(instances) = instances.as_ref() {
            d.cmd_bind_vertex_buffers(
                context.command,
                0,
                &[vertices.buffer, instances.buffer],
                &[0, 0],
            );
            d.cmd_draw(context.command, 6, 2, 0, 2);
        } else {
            d.cmd_bind_vertex_buffers(context.command, 0, &[vertices.buffer], &[0]);
            d.cmd_draw(context.command, 12, 1, 0, 0);
        }
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
        let fence = d
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .unwrap();
        context.submit(fence).unwrap();
        d.wait_for_fences(&[fence], true, 5_000_000_000).unwrap();
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
