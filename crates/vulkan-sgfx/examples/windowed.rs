//! Ordinary Vulkan-loader client that presents through VK_EXT_metal_surface.

#[cfg(target_os = "macos")]
#[path = "support/cube.rs"]
mod cube_support;

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("the SGFX windowed Vulkan example currently requires macOS");
}

#[cfg(target_os = "macos")]
mod macos {
    use super::cube_support;
    use ash::{Entry, vk};
    use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
    use std::{error::Error, ffi::CStr, time::Instant};
    use winit::{
        application::ApplicationHandler,
        dpi::LogicalSize,
        event::WindowEvent,
        event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
        window::{Window, WindowAttributes, WindowId},
    };

    pub fn run() -> Result<(), Box<dyn Error>> {
        let mut frame_limit = None;
        let mut arguments = std::env::args().skip(1);
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--frames" => {
                    let frames: u64 = arguments
                        .next()
                        .ok_or("--frames requires a count")?
                        .parse()?;
                    if frames == 0 {
                        return Err("--frames must be greater than zero".into());
                    }
                    frame_limit = Some(frames);
                }
                "--help" | "-h" => {
                    println!("Usage: windowed [--frames COUNT]");
                    return Ok(());
                }
                _ => return Err(format!("unknown argument {argument:?}").into()),
            }
        }
        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);
        let mut app = App {
            frame_limit,
            ..Default::default()
        };
        event_loop.run_app(&mut app)?;
        if let Some(error) = app.error {
            return Err(error.into());
        }
        Ok(())
    }

    #[derive(Default)]
    struct App {
        renderer: Option<Renderer>,
        error: Option<String>,
        frames: u64,
        frame_limit: Option<u64>,
    }

    impl ApplicationHandler for App {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.renderer.is_some() || self.error.is_some() {
                return;
            }
            let attributes = WindowAttributes::default()
                .with_title("Standard Vulkan loader → vulkan-sgfx → SGFX → Metal")
                .with_inner_size(LogicalSize::new(640.0, 640.0))
                .with_resizable(false);
            let result = event_loop
                .create_window(attributes)
                .map_err(|error| error.to_string())
                .and_then(|window| unsafe { Renderer::new(window).map_err(|e| e.to_string()) });
            match result {
                Ok(renderer) => {
                    println!(
                        "PASS: standard Vulkan loader created a Metal surface and {}-image swapchain",
                        renderer.image_count()
                    );
                    renderer.window.request_redraw();
                    self.renderer = Some(renderer);
                }
                Err(error) => {
                    self.error = Some(error);
                    event_loop.exit();
                }
            }
        }

        fn window_event(
            &mut self,
            event_loop: &ActiveEventLoop,
            window_id: WindowId,
            event: WindowEvent,
        ) {
            let Some(renderer) = self.renderer.as_mut() else {
                return;
            };
            if renderer.window.id() != window_id {
                return;
            }
            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::RedrawRequested => match unsafe { renderer.draw() } {
                    Ok(()) => {
                        self.frames += 1;
                        if self.frame_limit == Some(self.frames) {
                            println!("PASS: presented {} rotating cube frames", self.frames);
                            event_loop.exit();
                        } else {
                            renderer.window.request_redraw();
                        }
                    }
                    Err(error) => {
                        self.error = Some(error.to_string());
                        event_loop.exit();
                    }
                },
                _ => {}
            }
        }
    }

    struct Renderer {
        window: Window,
        _entry: Entry,
        instance: ash::Instance,
        surface_api: ash::khr::surface::Instance,
        surface: vk::SurfaceKHR,
        device: ash::Device,
        swapchain_api: ash::khr::swapchain::Device,
        swapchain: vk::SwapchainKHR,
        image_views: Vec<vk::ImageView>,
        depth_image: vk::Image,
        depth_view: vk::ImageView,
        render_pass: vk::RenderPass,
        framebuffers: Vec<vk::Framebuffer>,
        buffers: Vec<vk::Buffer>,
        memories: Vec<vk::DeviceMemory>,
        uniform_memory: vk::DeviceMemory,
        shader_modules: Vec<vk::ShaderModule>,
        descriptor_layout: vk::DescriptorSetLayout,
        descriptor_pool: vk::DescriptorPool,
        pipeline_layout: vk::PipelineLayout,
        descriptor_set: vk::DescriptorSet,
        pipeline: vk::Pipeline,
        vertex_buffer: vk::Buffer,
        index_buffer: vk::Buffer,
        command_pool: vk::CommandPool,
        command_buffers: Vec<vk::CommandBuffer>,
        image_available: vk::Semaphore,
        render_finished: vk::Semaphore,
        fence: vk::Fence,
        queue: vk::Queue,
        extent: vk::Extent2D,
        started: Instant,
    }

    impl Renderer {
        unsafe fn new(window: Window) -> Result<Self, Box<dyn Error>> {
            // This is deliberately the operating system's Vulkan loader. The
            // SGFX ICD is selected externally with the standard VK_DRIVER_FILES.
            let entry = unsafe { Entry::load()? };
            let display = window.display_handle()?.as_raw();
            let window_handle = window.window_handle()?.as_raw();
            let extensions = ash_window::enumerate_required_extensions(display)?;
            let app = vk::ApplicationInfo::default()
                .application_name(c"vulkan-sgfx-windowed")
                .application_version(1)
                .engine_name(c"none")
                .api_version(vk::API_VERSION_1_0);
            let instance_info = vk::InstanceCreateInfo::default()
                .application_info(&app)
                .enabled_extension_names(extensions);
            let instance = unsafe { entry.create_instance(&instance_info, None)? };
            let surface = unsafe {
                ash_window::create_surface(&entry, &instance, display, window_handle, None)?
            };
            let surface_api = ash::khr::surface::Instance::new(&entry, &instance);
            let (physical, family) = unsafe { instance.enumerate_physical_devices()? }
                .into_iter()
                .find_map(|physical| {
                    unsafe { instance.get_physical_device_queue_family_properties(physical) }
                        .into_iter()
                        .enumerate()
                        .find_map(|(index, properties)| {
                            let graphics =
                                properties.queue_flags.contains(vk::QueueFlags::GRAPHICS);
                            let present = unsafe {
                                surface_api
                                    .get_physical_device_surface_support(
                                        physical,
                                        index as u32,
                                        surface,
                                    )
                                    .ok()?
                            };
                            (graphics && present).then_some((physical, index as u32))
                        })
                })
                .ok_or("no graphics/presentation queue")?;
            let properties = unsafe { instance.get_physical_device_properties(physical) };
            let device_name =
                unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }.to_string_lossy();
            println!("Vulkan physical device: {device_name}");

            let priorities = [1.0];
            let queues = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&priorities)];
            let device_extensions = [ash::khr::swapchain::NAME.as_ptr()];
            let device_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queues)
                .enabled_extension_names(&device_extensions);
            let device = unsafe { instance.create_device(physical, &device_info, None)? };
            let queue = unsafe { device.get_device_queue(family, 0) };
            let swapchain_api = ash::khr::swapchain::Device::new(&instance, &device);
            let capabilities =
                unsafe { surface_api.get_physical_device_surface_capabilities(physical, surface)? };
            let formats =
                unsafe { surface_api.get_physical_device_surface_formats(physical, surface)? };
            let format = formats
                .iter()
                .copied()
                .find(|format| format.format == vk::Format::B8G8R8A8_UNORM)
                .or_else(|| formats.first().copied())
                .ok_or("surface has no formats")?;
            let present_modes = unsafe {
                surface_api.get_physical_device_surface_present_modes(physical, surface)?
            };
            if !present_modes.contains(&vk::PresentModeKHR::FIFO) {
                return Err("surface does not support FIFO present".into());
            }
            let size = window.inner_size();
            let extent = vk::Extent2D {
                width: size.width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: size.height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            };
            let mut image_count = capabilities.min_image_count.saturating_add(1);
            if capabilities.max_image_count != 0 {
                image_count = image_count.min(capabilities.max_image_count);
            }
            let swapchain_info = vk::SwapchainCreateInfoKHR::default()
                .surface(surface)
                .min_image_count(image_count)
                .image_format(format.format)
                .image_color_space(format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(vk::PresentModeKHR::FIFO)
                .clipped(true);
            let swapchain = unsafe { swapchain_api.create_swapchain(&swapchain_info, None)? };
            let images = unsafe { swapchain_api.get_swapchain_images(swapchain)? };
            let image_views = images
                .iter()
                .map(|&image| {
                    let range = vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1);
                    let info = vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(format.format)
                        .subresource_range(range);
                    unsafe { device.create_image_view(&info, None) }
                })
                .collect::<Result<Vec<_>, _>>()?;
            let memory_properties =
                unsafe { instance.get_physical_device_memory_properties(physical) };
            let depth_image = unsafe {
                device.create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(vk::Format::D32_SFLOAT)
                        .extent(vk::Extent3D {
                            width: extent.width,
                            height: extent.height,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    None,
                )?
            };
            let depth_requirements = unsafe { device.get_image_memory_requirements(depth_image) };
            let depth_memory = unsafe {
                device.allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(depth_requirements.size)
                        .memory_type_index(cube_support::memory_type(
                            &memory_properties,
                            depth_requirements,
                            vk::MemoryPropertyFlags::DEVICE_LOCAL,
                        )?),
                    None,
                )?
            };
            unsafe { device.bind_image_memory(depth_image, depth_memory, 0)? };
            let depth_view = unsafe {
                device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(depth_image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(vk::Format::D32_SFLOAT)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                                .level_count(1)
                                .layer_count(1),
                        ),
                    None,
                )?
            };
            let options = cube_support::Options {
                size: [extent.width, extent.height],
                cull_back: true,
                ..Default::default()
            };
            let (vertex_buffer, vertex_memory) = unsafe {
                upload_buffer(
                    &device,
                    &memory_properties,
                    vk::BufferUsageFlags::VERTEX_BUFFER,
                    &cube_support::cube_vertices(),
                )?
            };
            let (index_buffer, index_memory) = unsafe {
                upload_buffer(
                    &device,
                    &memory_properties,
                    vk::BufferUsageFlags::INDEX_BUFFER,
                    &cube_support::cube_indices(options),
                )?
            };
            let (uniform_buffer, uniform_memory) = unsafe {
                upload_buffer(
                    &device,
                    &memory_properties,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    &cube_support::transform(options),
                )?
            };
            let buffers = vec![vertex_buffer, index_buffer, uniform_buffer];
            let memories = vec![vertex_memory, index_memory, uniform_memory, depth_memory];

            let vertex_words = cube_support::shader_words(
                include_str!("assets/cube.wgsl"),
                naga::ShaderStage::Vertex,
                "vs_main",
            )?;
            let fragment_words = cube_support::shader_words(
                include_str!("assets/cube.wgsl"),
                naga::ShaderStage::Fragment,
                "fs_main",
            )?;
            let vertex_shader = unsafe {
                device.create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(&vertex_words),
                    None,
                )?
            };
            let fragment_shader = unsafe {
                device.create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(&fragment_words),
                    None,
                )?
            };
            let shader_modules = vec![vertex_shader, fragment_shader];
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX)];
            let descriptor_layout = unsafe {
                device.create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )?
            };
            let set_layouts = [descriptor_layout];
            let pipeline_layout = unsafe {
                device.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                    None,
                )?
            };
            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)];
            let descriptor_pool = unsafe {
                device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&pool_sizes),
                    None,
                )?
            };
            let descriptor_set = unsafe {
                device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&set_layouts),
                )?[0]
            };
            let uniform_info = [vk::DescriptorBufferInfo::default()
                .buffer(uniform_buffer)
                .range(64)];
            unsafe {
                device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&uniform_info)],
                    &[],
                );
            }

            let attachments = [
                vk::AttachmentDescription::default()
                    .format(format.format)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(vk::ImageLayout::PRESENT_SRC_KHR),
                vk::AttachmentDescription::default()
                    .format(vk::Format::D32_SFLOAT)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
            ];
            let color = [vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            }];
            let depth = vk::AttachmentReference {
                attachment: 1,
                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            };
            let subpasses = [vk::SubpassDescription::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&color)
                .depth_stencil_attachment(&depth)];
            let render_pass_info = vk::RenderPassCreateInfo::default()
                .attachments(&attachments)
                .subpasses(&subpasses);
            let render_pass = unsafe { device.create_render_pass(&render_pass_info, None)? };
            let framebuffers = image_views
                .iter()
                .map(|view| {
                    let attachments = [*view, depth_view];
                    let info = vk::FramebufferCreateInfo::default()
                        .render_pass(render_pass)
                        .attachments(&attachments)
                        .width(extent.width)
                        .height(extent.height)
                        .layers(1);
                    unsafe { device.create_framebuffer(&info, None) }
                })
                .collect::<Result<Vec<_>, _>>()?;

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
            let vertex_bindings = [vk::VertexInputBindingDescription::default()
                .binding(0)
                .stride(24)
                .input_rate(vk::VertexInputRate::VERTEX)];
            let vertex_attributes = [
                vk::VertexInputAttributeDescription::default()
                    .location(0)
                    .binding(0)
                    .format(vk::Format::R32G32B32_SFLOAT),
                vk::VertexInputAttributeDescription::default()
                    .location(1)
                    .binding(0)
                    .format(vk::Format::R32G32B32_SFLOAT)
                    .offset(12),
            ];
            let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&vertex_bindings)
                .vertex_attribute_descriptions(&vertex_attributes);
            let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let viewports = [vk::Viewport::default()
                .width(extent.width as f32)
                .height(extent.height as f32)
                .max_depth(1.0)];
            let scissors = [vk::Rect2D::default().extent(extent)];
            let viewport = vk::PipelineViewportStateCreateInfo::default()
                .viewports(&viewports)
                .scissors(&scissors);
            let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::BACK)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .line_width(1.0);
            let multisample = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)];
            let blend =
                vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
            let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS);
            let pipeline_info = [vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages)
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&input_assembly)
                .viewport_state(&viewport)
                .rasterization_state(&rasterization)
                .multisample_state(&multisample)
                .color_blend_state(&blend)
                .depth_stencil_state(&depth_stencil)
                .layout(pipeline_layout)
                .render_pass(render_pass)];
            let pipeline = unsafe {
                device
                    .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
                    .map_err(|(_, error)| error)?[0]
            };
            let pool_info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(family);
            let command_pool = unsafe { device.create_command_pool(&pool_info, None)? };
            let allocate = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(images.len() as u32);
            let command_buffers = unsafe { device.allocate_command_buffers(&allocate)? };
            let semaphore = vk::SemaphoreCreateInfo::default();
            let image_available = unsafe { device.create_semaphore(&semaphore, None)? };
            let render_finished = unsafe { device.create_semaphore(&semaphore, None)? };
            let fence = unsafe {
                device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )?
            };
            Ok(Self {
                window,
                _entry: entry,
                instance,
                surface_api,
                surface,
                device,
                swapchain_api,
                swapchain,
                image_views,
                depth_image,
                depth_view,
                render_pass,
                framebuffers,
                buffers,
                memories,
                uniform_memory,
                shader_modules,
                descriptor_layout,
                descriptor_pool,
                pipeline_layout,
                descriptor_set,
                pipeline,
                vertex_buffer,
                index_buffer,
                command_pool,
                command_buffers,
                image_available,
                render_finished,
                fence,
                queue,
                extent,
                started: Instant::now(),
            })
        }

        fn image_count(&self) -> usize {
            self.image_views.len()
        }

        unsafe fn draw(&mut self) -> Result<(), vk::Result> {
            unsafe {
                self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
                self.device.reset_fences(&[self.fence])?;
            }
            let (index, _) = unsafe {
                self.swapchain_api.acquire_next_image(
                    self.swapchain,
                    u64::MAX,
                    self.image_available,
                    vk::Fence::null(),
                )?
            };
            let command = self.command_buffers[index as usize];
            let phase = self.started.elapsed().as_secs_f32();
            let transform = cube_support::transform(cube_support::Options {
                size: [self.extent.width, self.extent.height],
                angle: 0.45 + phase * 0.7,
                ..Default::default()
            });
            unsafe {
                let mapped = self.device.map_memory(
                    self.uniform_memory,
                    0,
                    transform.len() as u64,
                    vk::MemoryMapFlags::empty(),
                )?;
                std::ptr::copy_nonoverlapping(
                    transform.as_ptr(),
                    mapped.cast::<u8>(),
                    transform.len(),
                );
                self.device.unmap_memory(self.uniform_memory);
            }
            unsafe {
                self.device
                    .reset_command_buffer(command, vk::CommandBufferResetFlags::empty())?;
                self.device.begin_command_buffer(
                    command,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;
            }
            let clears = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.025, 0.035, 0.06, 1.0],
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];
            let render = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(self.framebuffers[index as usize])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent: self.extent,
                })
                .clear_values(&clears);
            unsafe {
                self.device.cmd_bind_pipeline(
                    command,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline,
                );
                self.device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    0,
                    &[self.descriptor_set],
                    &[],
                );
                self.device
                    .cmd_bind_vertex_buffers(command, 0, &[self.vertex_buffer], &[0]);
                self.device.cmd_bind_index_buffer(
                    command,
                    self.index_buffer,
                    0,
                    vk::IndexType::UINT16,
                );
                self.device
                    .cmd_begin_render_pass(command, &render, vk::SubpassContents::INLINE);
                self.device.cmd_draw_indexed(command, 36, 1, 0, 0, 0);
                self.device.cmd_end_render_pass(command);
                self.device.end_command_buffer(command)?;
            }
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let wait_semaphores = [self.image_available];
            let signal_semaphores = [self.render_finished];
            let commands = [command];
            let submits = [vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&commands)
                .signal_semaphores(&signal_semaphores)];
            unsafe { self.device.queue_submit(self.queue, &submits, self.fence)? };
            let swapchains = [self.swapchain];
            let indices = [index];
            let present = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&indices);
            unsafe { self.swapchain_api.queue_present(self.queue, &present)? };
            Ok(())
        }
    }

    unsafe fn upload_buffer(
        device: &ash::Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        usage: vk::BufferUsageFlags,
        bytes: &[u8],
    ) -> Result<(vk::Buffer, vk::DeviceMemory), Box<dyn Error>> {
        let buffer = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(bytes.len() as u64)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?
        };
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(cube_support::memory_type(
                        memory_properties,
                        requirements,
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )?),
                None,
            )?
        };
        unsafe {
            device.bind_buffer_memory(buffer, memory, 0)?;
            let mapped =
                device.map_memory(memory, 0, bytes.len() as u64, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.cast::<u8>(), bytes.len());
            device.unmap_memory(memory);
        }
        Ok((buffer, memory))
    }

    impl Drop for Renderer {
        fn drop(&mut self) {
            unsafe {
                let _ = self.device.device_wait_idle();
                self.device.destroy_fence(self.fence, None);
                self.device.destroy_semaphore(self.render_finished, None);
                self.device.destroy_semaphore(self.image_available, None);
                self.device.destroy_command_pool(self.command_pool, None);
                self.device.destroy_pipeline(self.pipeline, None);
                for framebuffer in self.framebuffers.drain(..) {
                    self.device.destroy_framebuffer(framebuffer, None);
                }
                self.device.destroy_render_pass(self.render_pass, None);
                self.device
                    .destroy_descriptor_pool(self.descriptor_pool, None);
                self.device
                    .destroy_pipeline_layout(self.pipeline_layout, None);
                self.device
                    .destroy_descriptor_set_layout(self.descriptor_layout, None);
                for shader in self.shader_modules.drain(..) {
                    self.device.destroy_shader_module(shader, None);
                }
                for view in self.image_views.drain(..) {
                    self.device.destroy_image_view(view, None);
                }
                self.device.destroy_image_view(self.depth_view, None);
                self.device.destroy_image(self.depth_image, None);
                for buffer in self.buffers.drain(..) {
                    self.device.destroy_buffer(buffer, None);
                }
                for memory in self.memories.drain(..) {
                    self.device.free_memory(memory, None);
                }
                self.swapchain_api.destroy_swapchain(self.swapchain, None);
                self.device.destroy_device(None);
                self.surface_api.destroy_surface(self.surface, None);
                self.instance.destroy_instance(None);
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}
