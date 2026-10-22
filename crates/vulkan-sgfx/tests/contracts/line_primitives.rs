//! One-pixel lines are actual GPU line primitives, with Skia's packed float2 layout.
use super::*;

#[test]
#[ignore = "requires a native GPU and freshly built SGFX ICD"]
fn line_primitives_cover_one_pixel_without_triangle_fill() {
    line_contract(vk::PrimitiveTopology::LINE_LIST, false);
    line_contract(vk::PrimitiveTopology::LINE_LIST, true);
    line_contract(vk::PrimitiveTopology::LINE_STRIP, false);
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

fn line_contract(topology: vk::PrimitiveTopology, indexed: bool) {
    let context = Context::new();
    let vertices = vertex_storage(&context, 32);
    let indices = Storage::with_usage(&context, vk::BufferUsageFlags::INDEX_BUFFER);
    let output = Storage::with_usage(&context, vk::BufferUsageFlags::TRANSFER_DST);
    let source = r#"
        @vertex fn vs(@location(0) position: vec2<f32>) -> @builtin(position) vec4<f32> {
            return vec4<f32>(position, 0.0, 1.0);
        }
        @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(0.5, 0.25, 0.125, 0.5); }
    "#;
    let module = naga::front::wgsl::parse_str(source).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    // As with separate GLSL stage compilation, keep each SPIR-V module's
    // interface scoped to its entry point rather than combining stage inputs.
    // Vertex positions below are Vulkan NDC, not WGSL/WebGPU coordinates.
    let mut spirv = naga::back::spv::Options::default();
    spirv
        .flags
        .remove(naga::back::spv::WriterFlags::ADJUST_COORDINATE_SPACE);
    let words = [
        (naga::ShaderStage::Vertex, "vs"),
        (naga::ShaderStage::Fragment, "fs"),
    ]
    .map(|(shader_stage, entry)| {
        naga::back::spv::write_vec(
            &module,
            &info,
            &spirv,
            Some(&naga::back::spv::PipelineOptions {
                shader_stage,
                entry_point: entry.into(),
            }),
        )
        .unwrap()
    });
    unsafe {
        let d = &context.device;
        // Skia's observed float2-only, 8-byte vertex stride. Endpoints are
        // clipped outside the viewport to avoid endpoint coverage ambiguity.
        let points: &[(f32, f32)] = if topology == vk::PrimitiveTopology::LINE_LIST {
            &[(-2.0, -0.625), (2.0, -0.625), (-2.0, 0.375), (2.0, 0.375)]
        } else {
            &[(-2.0, -0.625), (0.0, -0.625), (2.0, -0.625)]
        };
        let packed: Vec<u8> = points
            .iter()
            .flat_map(|(x, y)| x.to_le_bytes().into_iter().chain(y.to_le_bytes()))
            .collect();
        let mapping = d
            .map_memory(vertices.memory, 0, 32, vk::MemoryMapFlags::empty())
            .unwrap()
            .cast::<u8>();
        std::ptr::copy_nonoverlapping(packed.as_ptr(), mapping, packed.len());
        d.unmap_memory(vertices.memory);
        let mapping = d
            .map_memory(indices.memory, 0, 8, vk::MemoryMapFlags::empty())
            .unwrap()
            .cast::<u16>();
        std::ptr::copy_nonoverlapping([2u16, 3, 0, 1].as_ptr(), mapping, 4);
        d.unmap_memory(indices.memory);
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
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(topology);
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
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
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
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
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
        d.cmd_bind_pipeline(context.command, vk::PipelineBindPoint::GRAPHICS, pipeline);
        d.cmd_bind_vertex_buffers(context.command, 0, &[vertices.buffer], &[0]);
        if indexed {
            d.cmd_bind_index_buffer(context.command, indices.buffer, 0, vk::IndexType::UINT16);
            d.cmd_draw_indexed(context.command, 4, 1, 0, 0, 0);
        } else {
            d.cmd_draw(context.command, points.len() as u32, 1, 0, 0);
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
        let row_coverage: Vec<_> = pixels[..64]
            .chunks_exact(8)
            .map(|row| {
                row.iter()
                    .filter(|pixel| pixel.to_le_bytes()[0] != 0)
                    .count()
            })
            .collect();
        eprintln!("{topology:?} indexed={indexed} GPU row coverage: {row_coverage:?}");
        for y in 0..8 {
            for x in 0..8 {
                let actual = pixels[y * 8 + x].to_le_bytes();
                let on_line = y == 1 || (topology == vk::PrimitiveTopology::LINE_LIST && y == 5);
                let expected = if on_line {
                    [128u8, 64, 32, 255]
                } else {
                    [0, 0, 0, 255]
                };
                for channel in 0..4 {
                    assert!(
                        actual[channel].abs_diff(expected[channel]) <= 1,
                        "{topology:?} indexed={indexed} pixel ({x},{y}): {actual:?}, expected {expected:?}"
                    );
                }
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
