//! An indexed, depth-tested Vulkan cube. The caller supplies the entry, so this
//! path can run through a host loader or a statically linked Scarlet ICD.
#![allow(dead_code)]
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

pub(crate) fn memory_type(
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

pub const CLEAR: [f32; 4] = [0.025, 0.035, 0.06, 1.0];

#[derive(Clone, Copy)]
pub struct Options {
    pub size: [u32; 2],
    pub angle: f32,
    pub reverse_triangles: bool,
    pub index_u32: bool,
    pub depth_test: bool,
    pub cull_back: bool,
    pub front_clockwise: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            size: [512, 512],
            angle: 0.58,
            reverse_triangles: false,
            index_u32: false,
            depth_test: true,
            cull_back: false,
            front_clockwise: false,
        }
    }
}

pub(crate) fn cube_vertices() -> Vec<u8> {
    // Independent face vertices keep the colors flat across each cube face.
    let faces = [
        (
            [0.95f32, 0.24, 0.32],
            [
                [-1., -1., -1.],
                [1., -1., -1.],
                [1., 1., -1.],
                [-1., 1., -1.],
            ],
        ),
        (
            [0.18, 0.75, 0.98],
            [[1., -1., -1.], [1., -1., 1.], [1., 1., 1.], [1., 1., -1.]],
        ),
        (
            [0.64, 0.28, 0.92],
            [[1., -1., 1.], [-1., -1., 1.], [-1., 1., 1.], [1., 1., 1.]],
        ),
        (
            [0.13, 0.78, 0.49],
            [
                [-1., -1., 1.],
                [-1., -1., -1.],
                [-1., 1., -1.],
                [-1., 1., 1.],
            ],
        ),
        (
            [1.0, 0.72, 0.17],
            [[-1., 1., -1.], [1., 1., -1.], [1., 1., 1.], [-1., 1., 1.]],
        ),
        (
            [0.23, 0.38, 0.90],
            [
                [-1., -1., 1.],
                [1., -1., 1.],
                [1., -1., -1.],
                [-1., -1., -1.],
            ],
        ),
    ];
    let mut bytes = Vec::with_capacity(24 * 24);
    for (color, positions) in faces {
        for position in positions {
            for component in position.into_iter().chain(color) {
                bytes.extend_from_slice(&component.to_le_bytes());
            }
        }
    }
    bytes
}

pub(crate) fn cube_indices(options: Options) -> Vec<u8> {
    let mut triangles = Vec::with_capacity(12);
    for face in 0..6 {
        let base = face * 4;
        triangles.push([base, base + 1, base + 2]);
        triangles.push([base, base + 2, base + 3]);
    }
    if options.reverse_triangles {
        triangles.reverse();
    }
    let mut bytes = Vec::new();
    for index in triangles.into_iter().flatten() {
        if options.index_u32 {
            bytes.extend_from_slice(&(index as u32).to_le_bytes());
        } else {
            bytes.extend_from_slice(&(index as u16).to_le_bytes());
        }
    }
    bytes
}

pub(crate) fn transform(options: Options) -> Vec<u8> {
    let (sy, cy) = options.angle.sin_cos();
    let (sx, cx) = (-0.42f32).sin_cos();
    let aspect = options.size[0] as f32 / options.size[1] as f32;
    let focal = 1.8;
    let near = 0.1;
    let far = 20.0;
    let depth_scale = far / (far - near);
    // Column-major projection * translation * rotation-X * rotation-Y.
    // Positive view Z and the perspective divide map near/far onto Vulkan 0..1.
    let model_columns = [
        [cy, sx * sy, -cx * sy, 0.0],
        [0.0, cx, sx, 0.0],
        [sy, -sx * cy, cx * cy, 0.0],
        [0.0, 0.0, 4.5, 1.0],
    ];
    model_columns
        .into_iter()
        .flat_map(|[x, y, z, w]| {
            [
                focal * x / aspect,
                focal * y,
                depth_scale * z - near * depth_scale * w,
                z,
            ]
        })
        .flat_map(f32::to_le_bytes)
        .collect()
}

// Track objects as soon as they are created, so an unsupported capability or
// failed allocation also tears down the partially constructed scene.
struct SceneResources {
    instance: ash::Instance,
    device: Option<ash::Device>,
    buffers: Vec<vk::Buffer>,
    memories: Vec<vk::DeviceMemory>,
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
    shaders: Vec<vk::ShaderModule>,
    descriptor_layouts: Vec<vk::DescriptorSetLayout>,
    descriptor_pools: Vec<vk::DescriptorPool>,
    pipeline_layouts: Vec<vk::PipelineLayout>,
    pipelines: Vec<vk::Pipeline>,
    render_passes: Vec<vk::RenderPass>,
    framebuffers: Vec<vk::Framebuffer>,
    command_pools: Vec<vk::CommandPool>,
    fences: Vec<vk::Fence>,
}

impl SceneResources {
    fn new(instance: ash::Instance) -> Self {
        Self {
            instance,
            device: None,
            buffers: vec![],
            memories: vec![],
            images: vec![],
            views: vec![],
            shaders: vec![],
            descriptor_layouts: vec![],
            descriptor_pools: vec![],
            pipeline_layouts: vec![],
            pipelines: vec![],
            render_passes: vec![],
            framebuffers: vec![],
            command_pools: vec![],
            fences: vec![],
        }
    }
}

impl Drop for SceneResources {
    fn drop(&mut self) {
        unsafe {
            if let Some(device) = &self.device {
                // Required on the error path if submission succeeded but waiting failed.
                let _ = device.device_wait_idle();
                for &h in &self.fences {
                    device.destroy_fence(h, None);
                }
                for &h in &self.command_pools {
                    device.destroy_command_pool(h, None);
                }
                for &h in &self.pipelines {
                    device.destroy_pipeline(h, None);
                }
                for &h in &self.framebuffers {
                    device.destroy_framebuffer(h, None);
                }
                for &h in &self.render_passes {
                    device.destroy_render_pass(h, None);
                }
                for &h in &self.descriptor_pools {
                    device.destroy_descriptor_pool(h, None);
                }
                for &h in &self.pipeline_layouts {
                    device.destroy_pipeline_layout(h, None);
                }
                for &h in &self.descriptor_layouts {
                    device.destroy_descriptor_set_layout(h, None);
                }
                for &h in &self.shaders {
                    device.destroy_shader_module(h, None);
                }
                for &h in &self.views {
                    device.destroy_image_view(h, None);
                }
                for &h in &self.images {
                    device.destroy_image(h, None);
                }
                for &h in &self.buffers {
                    device.destroy_buffer(h, None);
                }
                for &h in &self.memories {
                    device.free_memory(h, None);
                }
                device.destroy_device(None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

// The full allocation range is coherent and accessed only while mapped.
unsafe fn upload_buffer(
    device: &ash::Device,
    properties: &vk::PhysicalDeviceMemoryProperties,
    usage: vk::BufferUsageFlags,
    bytes: &[u8],
    resources: &mut SceneResources,
) -> Result<vk::Buffer, Box<dyn Error>> {
    unsafe {
        let buffer = device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(bytes.len() as u64)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )?;
        resources.buffers.push(buffer);
        let requirements = device.get_buffer_memory_requirements(buffer);
        let memory = device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type(
                    properties,
                    requirements,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )?),
            None,
        )?;
        resources.memories.push(memory);
        device.bind_buffer_memory(buffer, memory, 0)?;
        let mapped =
            device.map_memory(memory, 0, bytes.len() as u64, vk::MemoryMapFlags::empty())?;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.cast::<u8>(), bytes.len());
        device.unmap_memory(memory);
        Ok(buffer)
    }
}

// Handles are used only with their creating device. Readback follows the
// transfer-to-host barrier and fence wait, and preserves exact RGBA bytes.
pub fn render(entry: &Entry, options: Options) -> Result<Vec<u8>, Box<dyn Error>> {
    let vertex_words = shader_words(
        include_str!("../assets/cube.wgsl"),
        naga::ShaderStage::Vertex,
        "vs_main",
    )?;
    let fragment_words = shader_words(
        include_str!("../assets/cube.wgsl"),
        naga::ShaderStage::Fragment,
        "fs_main",
    )?;
    let [width, height] = options.size;
    if !options.angle.is_finite() {
        return Err("cube angle must be finite".into());
    }
    if width == 0 || height == 0 || width > 2048 || height > 2048 {
        return Err("image dimensions outside the ICD subset".into());
    }
    let buffer_bytes = width as vk::DeviceSize * height as vk::DeviceSize * 4;
    unsafe {
        let application = vk::ApplicationInfo::default()
            .application_name(c"sgfx-indexed-cube")
            .api_version(vk::API_VERSION_1_0);
        let instance = entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(&application),
            None,
        )?;
        let mut resources = SceneResources::new(instance.clone());
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
        if !name.starts_with("SGFX Vulkan (") {
            return Err(
                format!("expected the SGFX ICD, selected {name}; check VK_DRIVER_FILES").into(),
            );
        }
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
        resources.device = Some(device.clone());
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
        resources.images.push(image);
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
        resources.memories.push(image_memory);
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
        resources.views.push(image_view);
        let depth_image = device.create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::D32_SFLOAT)
                .extent(extent)
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED),
            None,
        )?;
        resources.images.push(depth_image);
        let depth_requirements = device.get_image_memory_requirements(depth_image);
        let depth_memory = device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(depth_requirements.size)
                .memory_type_index(memory_type(
                    &memory_properties,
                    depth_requirements,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )?),
            None,
        )?;
        resources.memories.push(depth_memory);
        device.bind_image_memory(depth_image, depth_memory, 0)?;
        let depth_view = device.create_image_view(
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
        )?;
        resources.views.push(depth_view);
        let vertex_buffer = upload_buffer(
            &device,
            &memory_properties,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &cube_vertices(),
            &mut resources,
        )?;
        let index_buffer = upload_buffer(
            &device,
            &memory_properties,
            vk::BufferUsageFlags::INDEX_BUFFER,
            &cube_indices(options),
            &mut resources,
        )?;
        let uniform_buffer = upload_buffer(
            &device,
            &memory_properties,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            &transform(options),
            &mut resources,
        )?;

        let buffer = device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(buffer_bytes)
                .usage(vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )?;
        resources.buffers.push(buffer);
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
        resources.memories.push(buffer_memory);
        device.bind_buffer_memory(buffer, buffer_memory, 0)?;
        let mapped =
            device.map_memory(buffer_memory, 0, buffer_bytes, vk::MemoryMapFlags::empty())?;
        std::ptr::write_bytes(mapped.cast::<u8>(), 0x7f, buffer_bytes as usize);
        device.unmap_memory(buffer_memory);

        let attachments = [
            vk::AttachmentDescription::default()
                .format(vk::Format::R8G8B8A8_UNORM)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
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
        let color_references = [vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
        let depth_reference = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        let subpasses = [vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_references)
            .depth_stencil_attachment(&depth_reference)];
        let dependencies = [vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::TOP_OF_PIPE)
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )];
        let render_pass = device.create_render_pass(
            &vk::RenderPassCreateInfo::default()
                .attachments(&attachments)
                .subpasses(&subpasses)
                .dependencies(&dependencies),
            None,
        )?;
        resources.render_passes.push(render_pass);
        let framebuffer_attachments = [image_view, depth_view];
        let framebuffer = device.create_framebuffer(
            &vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&framebuffer_attachments)
                .width(width)
                .height(height)
                .layers(1),
            None,
        )?;
        resources.framebuffers.push(framebuffer);

        let vertex_shader = device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&vertex_words),
            None,
        )?;
        resources.shaders.push(vertex_shader);
        let fragment_shader = device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&fragment_words),
            None,
        )?;
        resources.shaders.push(fragment_shader);
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)];
        let descriptor_layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )?;
        resources.descriptor_layouts.push(descriptor_layout);
        let set_layouts = [descriptor_layout];
        let pipeline_layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
            None,
        )?;
        resources.pipeline_layouts.push(pipeline_layout);
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)];
        let descriptor_pool = device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes),
            None,
        )?;
        resources.descriptor_pools.push(descriptor_pool);
        let descriptor_set = device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&set_layouts),
        )?[0];
        let uniform_info = [vk::DescriptorBufferInfo::default()
            .buffer(uniform_buffer)
            .offset(0)
            .range(64)];
        device.update_descriptor_sets(
            &[vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&uniform_info)],
            &[],
        );
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
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0),
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
            .cull_mode(if options.cull_back {
                vk::CullModeFlags::BACK
            } else {
                vk::CullModeFlags::NONE
            })
            .front_face(if options.front_clockwise {
                vk::FrontFace::CLOCKWISE
            } else {
                vk::FrontFace::COUNTER_CLOCKWISE
            })
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
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(options.depth_test)
            .depth_write_enable(options.depth_test)
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
            .render_pass(render_pass)
            .subpass(0)];
        let pipeline = device
            .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
            .map_err(|(_, error)| error)?[0];

        resources.pipelines.push(pipeline);
        let command_pool = device.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(queue_family),
            None,
        )?;
        resources.command_pools.push(command_pool);
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
        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue { float32: CLEAR },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];
        // Binding before the render pass is legal and must survive its begin.
        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline_layout,
            0,
            &[descriptor_set],
            &[],
        );
        device.cmd_bind_vertex_buffers(command_buffer, 0, &[vertex_buffer], &[0]);
        device.cmd_bind_index_buffer(
            command_buffer,
            index_buffer,
            0,
            if options.index_u32 {
                vk::IndexType::UINT32
            } else {
                vk::IndexType::UINT16
            },
        );
        device.cmd_begin_render_pass(
            command_buffer,
            &vk::RenderPassBeginInfo::default()
                .render_pass(render_pass)
                .framebuffer(framebuffer)
                .render_area(scissors[0])
                .clear_values(&clear_values),
            vk::SubpassContents::INLINE,
        );
        device.cmd_draw_indexed(command_buffer, 36, 1, 0, 0, 0);
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
        resources.fences.push(fence);
        let command_buffers = [command_buffer];
        let submits = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
        device.queue_submit(queue, &submits, fence)?;
        device.wait_for_fences(&[fence], true, u64::MAX)?;

        let mapped =
            device.map_memory(buffer_memory, 0, buffer_bytes, vk::MemoryMapFlags::empty())?;
        let pixels = std::slice::from_raw_parts(mapped.cast::<u8>(), buffer_bytes as usize);
        let pixels = pixels.to_vec();
        device.unmap_memory(buffer_memory);

        drop(resources);

        Ok(pixels)
    }
}

pub fn validate_image(pixels: &[u8], size: [u32; 2]) -> Result<(), Box<dyn Error>> {
    let [width, height] = size;
    let total = width as usize * height as usize;
    if pixels.len() != total * 4 || pixels.chunks_exact(4).any(|pixel| pixel[3] != 255) {
        return Err("cube readback has incorrect dimensions or alpha".into());
    }
    let is_clear = |pixel: &[u8]| {
        pixel
            .iter()
            .zip([6u8, 9, 15, 255])
            .all(|(&a, b)| a.abs_diff(b) <= 1)
    };
    let foreground = pixels
        .chunks_exact(4)
        .filter(|pixel| !is_clear(pixel))
        .count();
    if foreground < total / 10 || foreground > total * 3 / 4 {
        return Err(
            format!("cube silhouette has unexpected area: {foreground}/{total} pixels").into(),
        );
    }
    for [x, y] in [
        [0, 0],
        [width - 1, 0],
        [0, height - 1],
        [width - 1, height - 1],
    ] {
        let offset = ((y * width + x) * 4) as usize;
        if !is_clear(&pixels[offset..offset + 4]) {
            return Err(format!("cube overwrote clear corner ({x}, {y})").into());
        }
    }
    let center = ((height / 2 * width + width / 2) * 4) as usize;
    if is_clear(&pixels[center..center + 4]) {
        return Err("cube center is empty".into());
    }
    Ok(())
}

fn differing_pixels(a: &[u8], b: &[u8]) -> usize {
    a.chunks_exact(4)
        .zip(b.chunks_exact(4))
        .filter(|(a, b)| a.iter().zip(b.iter()).any(|(&a, &b)| a.abs_diff(b) > 2))
        .count()
}

/// Test actual depth occlusion, not only successful API return values. Reversing
/// every triangle must preserve a depth-tested frame, but visibly change the
/// same overlapping faces with depth disabled. Both index widths must agree.
pub fn verify(entry: &Entry) -> Result<(), Box<dyn Error>> {
    let options = Options {
        size: [256, 256],
        ..Options::default()
    };
    let total = options.size[0] as usize * options.size[1] as usize;
    let normal = render(entry, options)?;
    validate_image(&normal, options.size)?;
    let reversed = render(
        entry,
        Options {
            reverse_triangles: true,
            ..options
        },
    )?;
    let order_difference = differing_pixels(&normal, &reversed);
    // A small boundary allowance accommodates rasterization precision at the
    // shared edges; a missing depth test changes thousands of interior pixels.
    if order_difference > total / 500 {
        return Err(format!(
            "depth failed draw-order invariance: {order_difference} changed pixels"
        )
        .into());
    }
    let wide = render(
        entry,
        Options {
            index_u32: true,
            ..options
        },
    )?;
    if normal != wide {
        return Err("UINT16 and UINT32 indexed cube images differ".into());
    }
    let rotated = render(
        entry,
        Options {
            angle: options.angle + 0.8,
            ..options
        },
    )?;
    validate_image(&rotated, options.size)?;
    let rotation_difference = differing_pixels(&normal, &rotated);
    if rotation_difference < total / 20 {
        return Err(format!(
            "uniform rotation did not change enough pixels: {rotation_difference}"
        )
        .into());
    }
    let no_depth = render(
        entry,
        Options {
            depth_test: false,
            ..options
        },
    )?;
    let no_depth_reversed = render(
        entry,
        Options {
            depth_test: false,
            reverse_triangles: true,
            ..options
        },
    )?;
    let disabled_difference = differing_pixels(&no_depth, &no_depth_reversed);
    if disabled_difference < total / 20 {
        return Err(format!(
            "depth-disabled control lacks expected face overlap: {disabled_difference}"
        )
        .into());
    }
    let ccw = render(
        entry,
        Options {
            cull_back: true,
            ..options
        },
    )?;
    let cw = render(
        entry,
        Options {
            cull_back: true,
            front_clockwise: true,
            ..options
        },
    )?;
    let ccw_difference = differing_pixels(&normal, &ccw);
    let cw_difference = differing_pixels(&normal, &cw);
    // The front z=-1 quad uses counter-clockwise XY vertices. Coordinate
    // normalization must preserve that Vulkan-facing front-face convention.
    if ccw_difference > total / 500 || cw_difference < total / 20 {
        return Err(format!("front-face culling did not select opposite cube faces: CCW={ccw_difference}, CW={cw_difference}").into());
    }
    println!(
        "Cull check: front CCW difference={ccw_difference}, front CW difference={cw_difference}"
    );
    println!(
        "PASS: 256x256 cube: depth draw-order difference={order_difference}, depth-disabled control={disabled_difference}, rotation difference={rotation_difference}; UINT16/UINT32 readbacks identical"
    );
    Ok(())
}
