//! Shared real Vulkan offscreen render/readback path for smoke and image demo.
use ash::{Entry, vk};
use std::error::Error;

pub fn shader_words(
    source: &str,
    stage: naga::ShaderStage,
    entry: &str,
) -> Result<Vec<u32>, Box<dyn Error>> {
    let module = naga::front::wgsl::parse_str(source)?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)?;
    let pipeline = naga::back::spv::PipelineOptions {
        shader_stage: stage,
        entry_point: entry.into(),
    };
    Ok(naga::back::spv::write_vec(
        &module,
        &info,
        &naga::back::spv::Options::default(),
        Some(&pipeline),
    )?)
}

fn memory_type(
    properties: &vk::PhysicalDeviceMemoryProperties,
    requirements: vk::MemoryRequirements,
    flags: vk::MemoryPropertyFlags,
) -> Result<u32, Box<dyn Error>> {
    properties.memory_types[..properties.memory_type_count as usize]
        .iter()
        .enumerate()
        .find(|(index, memory)| {
            requirements.memory_type_bits & (1u32 << index) != 0
                && memory.property_flags.contains(flags)
        })
        .map(|(index, _)| index as u32)
        .ok_or_else(|| format!("no compatible memory type with flags {flags:?}").into())
}

// Handles are used with their creating device. The host reads only the mapped
// allocation's bounds, after both the transfer-to-host barrier and fence wait.
pub fn render(
    entry: &Entry,
    vertex_words: &[u32],
    fragment_words: &[u32],
    size: [u32; 2],
    clear: [f32; 4],
) -> Result<Vec<u8>, Box<dyn Error>> {
    let [width, height] = size;
    if width == 0 || height == 0 || width > 2048 || height > 2048 {
        return Err("image dimensions outside the ICD subset".into());
    }
    let buffer_bytes = width as vk::DeviceSize * height as vk::DeviceSize * 4;
    unsafe {
        let application = vk::ApplicationInfo::default()
            .application_name(c"Vulkan offscreen demo")
            .api_version(vk::API_VERSION_1_0);
        let instance = entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(&application),
            None,
        )?;
        let (physical_device, queue_family) = instance
            .enumerate_physical_devices()?
            .into_iter()
            .find_map(|physical| {
                instance
                    .get_physical_device_queue_family_properties(physical)
                    .iter()
                    .enumerate()
                    .find(|(_, properties)| {
                        properties.queue_count > 0
                            && properties.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    })
                    .map(|(index, _)| (physical, index as u32))
            })
            .ok_or("no Vulkan graphics queue is available")?;
        let properties = instance.get_physical_device_properties(physical_device);
        let name = std::ffi::CStr::from_ptr(properties.device_name.as_ptr()).to_string_lossy();
        println!("Vulkan physical device: {name}");
        let queue_priorities = [1.0];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&queue_priorities)];
        let device = instance.create_device(
            physical_device,
            &vk::DeviceCreateInfo::default().queue_create_infos(&queue_info),
            None,
        )?;
        let queue = device.get_device_queue(queue_family, 0);
        let memory_properties = instance.get_physical_device_memory_properties(physical_device);
        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };
        let image = device.create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .extent(extent)
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED),
            None,
        )?;
        let image_requirements = device.get_image_memory_requirements(image);
        let image_memory_type = memory_type(
            &memory_properties,
            image_requirements,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .or_else(|_| {
            memory_type(
                &memory_properties,
                image_requirements,
                vk::MemoryPropertyFlags::empty(),
            )
        })?;
        let image_memory = device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(image_requirements.size)
                .memory_type_index(image_memory_type),
            None,
        )?;
        device.bind_image_memory(image, image_memory, 0)?;
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);
        let image_view = device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .subresource_range(subresource_range),
            None,
        )?;
        let buffer = device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(buffer_bytes)
                .usage(vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )?;
        let buffer_requirements = device.get_buffer_memory_requirements(buffer);
        let buffer_memory = device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(buffer_requirements.size)
                .memory_type_index(memory_type(
                    &memory_properties,
                    buffer_requirements,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )?),
            None,
        )?;
        device.bind_buffer_memory(buffer, buffer_memory, 0)?;
        let mapped =
            device.map_memory(buffer_memory, 0, buffer_bytes, vk::MemoryMapFlags::empty())?;
        std::ptr::write_bytes(mapped.cast::<u8>(), 0x7f, buffer_bytes as usize);
        device.unmap_memory(buffer_memory);

        let attachments = [vk::AttachmentDescription::default()
            .format(vk::Format::R8G8B8A8_UNORM)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
        let color_references = [vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
        let subpasses = [vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_references)];
        let dependencies = [vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::TOP_OF_PIPE)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];
        let render_pass = device.create_render_pass(
            &vk::RenderPassCreateInfo::default()
                .attachments(&attachments)
                .subpasses(&subpasses)
                .dependencies(&dependencies),
            None,
        )?;
        let framebuffer_attachments = [image_view];
        let framebuffer = device.create_framebuffer(
            &vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&framebuffer_attachments)
                .width(width)
                .height(height)
                .layers(1),
            None,
        )?;

        let vertex_shader = device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(vertex_words),
            None,
        )?;
        let fragment_shader = device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(fragment_words),
            None,
        )?;
        let pipeline_layout =
            device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)?;
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex_shader)
                .name(c"vs_main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment_shader)
                .name(c"fs_main"),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewports = [vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(width as f32)
            .height(height as f32)
            .min_depth(0.0)
            .max_depth(1.0)];
        let scissors = [vk::Rect2D::default().extent(vk::Extent2D { width, height })];
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissors);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )];
        let blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
        let pipeline_info = [vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .color_blend_state(&blend)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0)];
        let pipeline = device
            .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
            .map_err(|(_, error)| error)?[0];

        let command_pool = device.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(queue_family),
            None,
        )?;
        let command_buffer = device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )?[0];
        device.begin_command_buffer(
            command_buffer,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue { float32: clear },
        }];
        // Binding before the render pass is legal and must survive its begin.
        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);
        device.cmd_begin_render_pass(
            command_buffer,
            &vk::RenderPassBeginInfo::default()
                .render_pass(render_pass)
                .framebuffer(framebuffer)
                .render_area(scissors[0])
                .clear_values(&clear_values),
            vk::SubpassContents::INLINE,
        );
        device.cmd_draw(command_buffer, 3, 1, 0, 0);
        device.cmd_end_render_pass(command_buffer);

        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(subresource_range)],
        );
        device.cmd_copy_image_to_buffer(
            command_buffer,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            buffer,
            &[vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .image_extent(extent)],
        );
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)],
            &[],
            &[],
        );
        device.end_command_buffer(command_buffer)?;
        let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
        let command_buffers = [command_buffer];
        let submits = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
        device.queue_submit(queue, &submits, fence)?;
        device.wait_for_fences(&[fence], true, u64::MAX)?;

        let mapped =
            device.map_memory(buffer_memory, 0, buffer_bytes, vk::MemoryMapFlags::empty())?;
        let pixels = std::slice::from_raw_parts(mapped.cast::<u8>(), buffer_bytes as usize);
        let pixels = pixels.to_vec();
        device.unmap_memory(buffer_memory);

        device.destroy_fence(fence, None);
        device.destroy_command_pool(command_pool, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_shader_module(fragment_shader, None);
        device.destroy_shader_module(vertex_shader, None);
        device.destroy_framebuffer(framebuffer, None);
        device.destroy_render_pass(render_pass, None);
        device.destroy_image_view(image_view, None);
        device.destroy_image(image, None);
        device.free_memory(image_memory, None);
        device.destroy_buffer(buffer, None);
        device.free_memory(buffer_memory, None);
        device.destroy_device(None);
        instance.destroy_instance(None);

        Ok(pixels)
    }
}
