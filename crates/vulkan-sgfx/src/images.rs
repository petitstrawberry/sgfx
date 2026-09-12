//! The bounded offscreen color-attachment portion of the experimental ICD.

use ash::vk::{self, Handle};
use sgfx_core::ir;
use std::ffi::CStr;

use crate::api::{next_id, with_device};

pub(crate) struct Image {
    pub id: ir::TextureId,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub bound: Option<(vk::DeviceMemory, u64)>,
}

pub(crate) struct RenderPass {
    pub format: vk::Format,
    pub load_op: vk::AttachmentLoadOp,
}

pub(crate) struct Framebuffer {
    pub view: vk::ImageView,
    pub render_pass: vk::RenderPass,
    pub image: vk::Image,
    pub width: u32,
    pub height: u32,
}

const UNSUPPORTED: vk::Result = vk::Result::ERROR_FEATURE_NOT_PRESENT;
const INVALID: vk::Result = vk::Result::ERROR_INITIALIZATION_FAILED;

fn status(result: Result<(), vk::Result>) -> vk::Result {
    result.err().unwrap_or(vk::Result::SUCCESS)
}

fn image_size(extent: vk::Extent3D) -> u64 {
    u64::from(extent.width) * u64::from(extent.height) * 4
}

unsafe extern "system" fn create_image(
    device: vk::Device,
    info: *const vk::ImageCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    output: *mut vk::Image,
) -> vk::Result {
    if info.is_null() || output.is_null() {
        return INVALID;
    }
    *output = vk::Image::null();
    let info = &*info;
    let supported = vk::ImageUsageFlags::COLOR_ATTACHMENT
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST;
    if !allocator.is_null()
        || !info.p_next.is_null()
        || !info.flags.is_empty()
        || info.s_type != vk::StructureType::IMAGE_CREATE_INFO
        || info.image_type != vk::ImageType::TYPE_2D
        || info.format != vk::Format::R8G8B8A8_UNORM
        || info.tiling != vk::ImageTiling::OPTIMAL
        || info.samples != vk::SampleCountFlags::TYPE_1
        || info.mip_levels != 1
        || info.array_layers != 1
        || info.extent.depth != 1
        || info.extent.width == 0
        || info.extent.height == 0
        || info.extent.width > 2048
        || info.extent.height > 2048
        || info.usage.is_empty()
        || !supported.contains(info.usage)
        || info.sharing_mode != vk::SharingMode::EXCLUSIVE
        || info.initial_layout != vk::ImageLayout::UNDEFINED
    {
        return UNSUPPORTED;
    }
    let extent = info.extent;
    let format = info.format;
    let mut usage = ir::TextureUsage::empty();
    if info.usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) {
        usage |= ir::TextureUsage::RENDER_ATTACHMENT;
    }
    if info.usage.contains(vk::ImageUsageFlags::TRANSFER_SRC) {
        usage |= ir::TextureUsage::COPY_SRC;
    }
    if info.usage.contains(vk::ImageUsageFlags::TRANSFER_DST) {
        usage |= ir::TextureUsage::COPY_DST;
    }
    match with_device(device, move |runtime| {
        let size = ir::Extent2D::new(extent.width, extent.height).map_err(|_| INVALID)?;
        let desc = ir::TextureDesc::new(ir::TextureFormat::Rgba8Unorm, size, usage)
            .map_err(|_| INVALID)?;
        let id = runtime
            .table
            .define_texture(desc)
            .map_err(crate::resources::failure)?
            .id();
        let handle = vk::Image::from_raw(next_id());
        runtime.resources.images.insert(
            handle,
            Image {
                id,
                format,
                extent,
                bound: None,
            },
        );
        Ok(handle)
    }) {
        Ok(handle) => {
            *output = handle;
            vk::Result::SUCCESS
        }
        Err(error) => error,
    }
}

unsafe extern "system" fn destroy_image(
    device: vk::Device,
    image: vk::Image,
    _: *const vk::AllocationCallbacks<'_>,
) {
    if image != vk::Image::null() {
        let _ = with_device(device, move |runtime| {
            runtime.resources.images.remove(&image);
            Ok(())
        });
    }
}

unsafe extern "system" fn get_image_memory_requirements(
    device: vk::Device,
    image: vk::Image,
    output: *mut vk::MemoryRequirements,
) {
    if output.is_null() {
        return;
    }
    *output = vk::MemoryRequirements::default();
    if let Ok(requirements) = with_device(device, move |runtime| {
        let image = runtime.resources.images.get(&image).ok_or(INVALID)?;
        Ok(vk::MemoryRequirements {
            size: image_size(image.extent),
            alignment: 4,
            memory_type_bits: 1,
        })
    }) {
        *output = requirements;
    }
}

unsafe extern "system" fn bind_image_memory(
    device: vk::Device,
    image: vk::Image,
    memory: vk::DeviceMemory,
    offset: vk::DeviceSize,
) -> vk::Result {
    status(with_device(device, move |runtime| {
        let size = image_size(runtime.resources.images.get(&image).ok_or(INVALID)?.extent);
        if !offset.is_multiple_of(4)
            || !crate::resources::memory_available(&runtime.resources, memory, offset, size)
        {
            return Err(INVALID);
        }
        let image = runtime.resources.images.get_mut(&image).ok_or(INVALID)?;
        if image.bound.is_some() {
            return Err(INVALID);
        }
        image.bound = Some((memory, offset));
        Ok(())
    }))
}

unsafe extern "system" fn create_image_view(
    device: vk::Device,
    info: *const vk::ImageViewCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    output: *mut vk::ImageView,
) -> vk::Result {
    if info.is_null() || output.is_null() {
        return INVALID;
    }
    *output = vk::ImageView::null();
    let info = &*info;
    let range = info.subresource_range;
    if !allocator.is_null()
        || !info.p_next.is_null()
        || !info.flags.is_empty()
        || info.s_type != vk::StructureType::IMAGE_VIEW_CREATE_INFO
        || info.view_type != vk::ImageViewType::TYPE_2D
        || info.format != vk::Format::R8G8B8A8_UNORM
        || range.aspect_mask != vk::ImageAspectFlags::COLOR
        || range.base_mip_level != 0
        || !matches!(range.level_count, 1 | vk::REMAINING_MIP_LEVELS)
        || range.base_array_layer != 0
        || !matches!(range.layer_count, 1 | vk::REMAINING_ARRAY_LAYERS)
        || [
            info.components.r,
            info.components.g,
            info.components.b,
            info.components.a,
        ]
        .iter()
        .any(|&component| component != vk::ComponentSwizzle::IDENTITY)
    {
        return UNSUPPORTED;
    }
    let image = info.image;
    let format = info.format;
    match with_device(device, move |runtime| {
        let data = runtime.resources.images.get(&image).ok_or(INVALID)?;
        if data.bound.is_none() || data.format != format {
            return Err(INVALID);
        }
        let handle = vk::ImageView::from_raw(next_id());
        runtime.resources.views.insert(handle, image);
        Ok(handle)
    }) {
        Ok(handle) => {
            *output = handle;
            vk::Result::SUCCESS
        }
        Err(error) => error,
    }
}

unsafe extern "system" fn destroy_image_view(
    device: vk::Device,
    view: vk::ImageView,
    _: *const vk::AllocationCallbacks<'_>,
) {
    if view != vk::ImageView::null() {
        let _ = with_device(device, move |runtime| {
            runtime.resources.views.remove(&view);
            Ok(())
        });
    }
}

unsafe extern "system" fn create_render_pass(
    device: vk::Device,
    info: *const vk::RenderPassCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    output: *mut vk::RenderPass,
) -> vk::Result {
    if info.is_null() || output.is_null() {
        return INVALID;
    }
    *output = vk::RenderPass::null();
    let info = &*info;
    if !allocator.is_null()
        || !info.p_next.is_null()
        || !info.flags.is_empty()
        || info.s_type != vk::StructureType::RENDER_PASS_CREATE_INFO
        || info.attachment_count != 1
        || info.p_attachments.is_null()
        || info.subpass_count != 1
        || info.p_subpasses.is_null()
    {
        return UNSUPPORTED;
    }
    let attachment = &*info.p_attachments;
    let subpass = &*info.p_subpasses;
    if !attachment.flags.is_empty()
        || attachment.format != vk::Format::R8G8B8A8_UNORM
        || attachment.samples != vk::SampleCountFlags::TYPE_1
        || !matches!(
            attachment.load_op,
            vk::AttachmentLoadOp::CLEAR | vk::AttachmentLoadOp::LOAD
        )
        || attachment.store_op != vk::AttachmentStoreOp::STORE
        || attachment.stencil_load_op != vk::AttachmentLoadOp::DONT_CARE
        || attachment.stencil_store_op != vk::AttachmentStoreOp::DONT_CARE
        || !matches!(
            attachment.initial_layout,
            vk::ImageLayout::UNDEFINED
                | vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                | vk::ImageLayout::GENERAL
        )
        || (attachment.load_op == vk::AttachmentLoadOp::LOAD
            && attachment.initial_layout == vk::ImageLayout::UNDEFINED)
        || !matches!(
            attachment.final_layout,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                | vk::ImageLayout::TRANSFER_SRC_OPTIMAL
                | vk::ImageLayout::GENERAL
        )
        || !subpass.flags.is_empty()
        || subpass.pipeline_bind_point != vk::PipelineBindPoint::GRAPHICS
        || subpass.input_attachment_count != 0
        || subpass.color_attachment_count != 1
        || subpass.p_color_attachments.is_null()
        || !subpass.p_resolve_attachments.is_null()
        || !subpass.p_depth_stencil_attachment.is_null()
        || subpass.preserve_attachment_count != 0
    {
        return UNSUPPORTED;
    }
    let color = &*subpass.p_color_attachments;
    if color.attachment != 0 || color.layout != vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
        return UNSUPPORTED;
    }
    if info.dependency_count > 2 || (info.dependency_count != 0 && info.p_dependencies.is_null()) {
        return UNSUPPORTED;
    }
    for index in 0..info.dependency_count as usize {
        let dependency = &*info.p_dependencies.add(index);
        let stages = vk::PipelineStageFlags::TOP_OF_PIPE
            | vk::PipelineStageFlags::BOTTOM_OF_PIPE
            | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
            | vk::PipelineStageFlags::TRANSFER;
        let accesses = vk::AccessFlags::COLOR_ATTACHMENT_READ
            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
            | vk::AccessFlags::TRANSFER_READ
            | vk::AccessFlags::TRANSFER_WRITE;
        let external_edge = (dependency.src_subpass == vk::SUBPASS_EXTERNAL
            && dependency.dst_subpass == 0)
            || (dependency.src_subpass == 0 && dependency.dst_subpass == vk::SUBPASS_EXTERNAL);
        if !external_edge
            || !stages.contains(dependency.src_stage_mask)
            || !stages.contains(dependency.dst_stage_mask)
            || !accesses.contains(dependency.src_access_mask)
            || !accesses.contains(dependency.dst_access_mask)
            || !vk::DependencyFlags::BY_REGION.contains(dependency.dependency_flags)
        {
            return UNSUPPORTED;
        }
    }
    let format = attachment.format;
    let load_op = attachment.load_op;
    match with_device(device, move |runtime| {
        let handle = vk::RenderPass::from_raw(next_id());
        runtime
            .resources
            .render_passes
            .insert(handle, RenderPass { format, load_op });
        Ok(handle)
    }) {
        Ok(handle) => {
            *output = handle;
            vk::Result::SUCCESS
        }
        Err(error) => error,
    }
}

unsafe extern "system" fn destroy_render_pass(
    device: vk::Device,
    pass: vk::RenderPass,
    _: *const vk::AllocationCallbacks<'_>,
) {
    if pass != vk::RenderPass::null() {
        let _ = with_device(device, move |runtime| {
            runtime.resources.render_passes.remove(&pass);
            Ok(())
        });
    }
}

unsafe extern "system" fn create_framebuffer(
    device: vk::Device,
    info: *const vk::FramebufferCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    output: *mut vk::Framebuffer,
) -> vk::Result {
    if info.is_null() || output.is_null() {
        return INVALID;
    }
    *output = vk::Framebuffer::null();
    let info = &*info;
    if !allocator.is_null()
        || !info.p_next.is_null()
        || !info.flags.is_empty()
        || info.s_type != vk::StructureType::FRAMEBUFFER_CREATE_INFO
        || info.attachment_count != 1
        || info.p_attachments.is_null()
        || info.layers != 1
        || info.width == 0
        || info.height == 0
    {
        return UNSUPPORTED;
    }
    let view = *info.p_attachments;
    let render_pass = info.render_pass;
    let (width, height) = (info.width, info.height);
    match with_device(device, move |runtime| {
        let pass = runtime
            .resources
            .render_passes
            .get(&render_pass)
            .ok_or(INVALID)?;
        let image = *runtime.resources.views.get(&view).ok_or(INVALID)?;
        let data = runtime.resources.images.get(&image).ok_or(INVALID)?;
        let desc = runtime
            .table
            .texture(runtime.table.texture_ref(data.id).map_err(|_| INVALID)?)
            .map_err(|_| INVALID)?;
        if data.bound.is_none()
            || data.format != pass.format
            || data.extent.width != width
            || data.extent.height != height
            || !desc.usage().contains(ir::TextureUsage::RENDER_ATTACHMENT)
        {
            return Err(UNSUPPORTED);
        }
        let handle = vk::Framebuffer::from_raw(next_id());
        runtime.resources.framebuffers.insert(
            handle,
            Framebuffer {
                view,
                render_pass,
                image,
                width,
                height,
            },
        );
        Ok(handle)
    }) {
        Ok(handle) => {
            *output = handle;
            vk::Result::SUCCESS
        }
        Err(error) => error,
    }
}

unsafe extern "system" fn destroy_framebuffer(
    device: vk::Device,
    framebuffer: vk::Framebuffer,
    _: *const vk::AllocationCallbacks<'_>,
) {
    if framebuffer != vk::Framebuffer::null() {
        let _ = with_device(device, move |runtime| {
            runtime.resources.framebuffers.remove(&framebuffer);
            Ok(())
        });
    }
}

/// Copies shader names and handles before crossing the device worker boundary.
unsafe fn shader_stages(
    info: &vk::GraphicsPipelineCreateInfo<'_>,
) -> Result<[(vk::ShaderModule, String); 2], vk::Result> {
    if info.stage_count != 2 || info.p_stages.is_null() {
        return Err(UNSUPPORTED);
    }
    let mut vertex = None;
    let mut fragment = None;
    for index in 0..2 {
        let stage = &*info.p_stages.add(index);
        if stage.s_type != vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO
            || !stage.p_next.is_null()
            || !stage.flags.is_empty()
            || stage.p_name.is_null()
            || !stage.p_specialization_info.is_null()
        {
            return Err(UNSUPPORTED);
        }
        let name = CStr::from_ptr(stage.p_name)
            .to_str()
            .map_err(|_| INVALID)?
            .to_owned();
        let slot = if stage.stage == vk::ShaderStageFlags::VERTEX {
            &mut vertex
        } else if stage.stage == vk::ShaderStageFlags::FRAGMENT {
            &mut fragment
        } else {
            return Err(UNSUPPORTED);
        };
        if slot.replace((stage.module, name)).is_some() {
            return Err(INVALID);
        }
    }
    Ok([vertex.ok_or(INVALID)?, fragment.ok_or(INVALID)?])
}

unsafe fn graphics_state(
    info: &vk::GraphicsPipelineCreateInfo<'_>,
) -> Result<vk::Extent2D, vk::Result> {
    if !info.p_next.is_null()
        || !info.flags.is_empty()
        || info.s_type != vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO
        || info.subpass != 0
        || !info.p_tessellation_state.is_null()
        || info.p_vertex_input_state.is_null()
        || info.p_input_assembly_state.is_null()
        || info.p_viewport_state.is_null()
        || info.p_rasterization_state.is_null()
        || info.p_multisample_state.is_null()
        || info.p_color_blend_state.is_null()
    {
        return Err(UNSUPPORTED);
    }
    let vertex = &*info.p_vertex_input_state;
    let assembly = &*info.p_input_assembly_state;
    let viewport = &*info.p_viewport_state;
    let raster = &*info.p_rasterization_state;
    let samples = &*info.p_multisample_state;
    let blend = &*info.p_color_blend_state;
    if vertex.s_type != vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO
        || assembly.s_type != vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO
        || viewport.s_type != vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO
        || raster.s_type != vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO
        || samples.s_type != vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO
        || blend.s_type != vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO
        || !vertex.p_next.is_null()
        || !vertex.flags.is_empty()
        || vertex.vertex_binding_description_count != 0
        || vertex.vertex_attribute_description_count != 0
        || !assembly.p_next.is_null()
        || !assembly.flags.is_empty()
        || assembly.topology != vk::PrimitiveTopology::TRIANGLE_LIST
        || assembly.primitive_restart_enable != vk::FALSE
        || !viewport.p_next.is_null()
        || !viewport.flags.is_empty()
        || viewport.viewport_count != 1
        || viewport.scissor_count != 1
        || viewport.p_viewports.is_null()
        || viewport.p_scissors.is_null()
        || !raster.p_next.is_null()
        || !raster.flags.is_empty()
        || raster.depth_clamp_enable != vk::FALSE
        || raster.rasterizer_discard_enable != vk::FALSE
        || raster.polygon_mode != vk::PolygonMode::FILL
        || raster.cull_mode != vk::CullModeFlags::NONE
        || raster.depth_bias_enable != vk::FALSE
        || raster.line_width != 1.0
        || !matches!(
            raster.front_face,
            vk::FrontFace::CLOCKWISE | vk::FrontFace::COUNTER_CLOCKWISE
        )
        || !samples.p_next.is_null()
        || !samples.flags.is_empty()
        || samples.rasterization_samples != vk::SampleCountFlags::TYPE_1
        || samples.sample_shading_enable != vk::FALSE
        || samples.alpha_to_coverage_enable != vk::FALSE
        || samples.alpha_to_one_enable != vk::FALSE
        || (!samples.p_sample_mask.is_null() && *samples.p_sample_mask & 1 == 0)
        || !blend.p_next.is_null()
        || !blend.flags.is_empty()
        || blend.logic_op_enable != vk::FALSE
        || blend.attachment_count != 1
        || blend.p_attachments.is_null()
    {
        return Err(UNSUPPORTED);
    }
    let color = &*blend.p_attachments;
    let rgba = vk::ColorComponentFlags::R
        | vk::ColorComponentFlags::G
        | vk::ColorComponentFlags::B
        | vk::ColorComponentFlags::A;
    if color.blend_enable != vk::FALSE || color.color_write_mask != rgba {
        return Err(UNSUPPORTED);
    }
    if let Some(depth) = info.p_depth_stencil_state.as_ref()
        && (depth.s_type != vk::StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO
            || !depth.p_next.is_null()
            || !depth.flags.is_empty()
            || depth.depth_test_enable != vk::FALSE
            || depth.depth_write_enable != vk::FALSE
            || depth.depth_bounds_test_enable != vk::FALSE
            || depth.stencil_test_enable != vk::FALSE)
    {
        return Err(UNSUPPORTED);
    }
    if let Some(dynamic) = info.p_dynamic_state.as_ref()
        && (dynamic.s_type != vk::StructureType::PIPELINE_DYNAMIC_STATE_CREATE_INFO
            || !dynamic.p_next.is_null()
            || !dynamic.flags.is_empty()
            || dynamic.dynamic_state_count != 0)
    {
        return Err(UNSUPPORTED);
    }
    let view = &*viewport.p_viewports;
    let scissor = &*viewport.p_scissors;
    if view.x != 0.0
        || view.y != 0.0
        || view.min_depth != 0.0
        || view.max_depth != 1.0
        || !view.width.is_finite()
        || !view.height.is_finite()
        || view.width < 1.0
        || view.height < 1.0
        || view.width > 2048.0
        || view.height > 2048.0
        || view.width.fract() != 0.0
        || view.height.fract() != 0.0
        || scissor.offset != (vk::Offset2D { x: 0, y: 0 })
        || scissor.extent.width != view.width as u32
        || scissor.extent.height != view.height as u32
    {
        return Err(UNSUPPORTED);
    }
    Ok(scissor.extent)
}

unsafe extern "system" fn create_graphics_pipelines(
    device: vk::Device,
    cache: vk::PipelineCache,
    count: u32,
    infos: *const vk::GraphicsPipelineCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    output: *mut vk::Pipeline,
) -> vk::Result {
    if infos.is_null() || output.is_null() || count == 0 {
        return INVALID;
    }
    for index in 0..count as usize {
        *output.add(index) = vk::Pipeline::null();
    }
    if !allocator.is_null() || cache != vk::PipelineCache::null() {
        return UNSUPPORTED;
    }
    // Each successful element remains usable if a later descriptor is rejected,
    // as required for Vulkan's partial-success pipeline creation contract.
    let mut first_error = None;
    for index in 0..count as usize {
        let info = &*infos.add(index);
        let parsed = graphics_state(info)
            .and_then(|extent| shader_stages(info).map(|stages| (extent, stages)));
        let (
            extent,
            [
                (vertex_module, vertex_name),
                (fragment_module, fragment_name),
            ],
        ) = match parsed {
            Ok(parsed) => parsed,
            Err(error) => {
                first_error.get_or_insert(error);
                continue;
            }
        };
        let layout = info.layout;
        let render_pass = info.render_pass;
        // Shader-module normalization already compensates for Vulkan's positive
        // viewport height. Framebuffer winding therefore maps without reversal.
        let front_face = if (*info.p_rasterization_state).front_face == vk::FrontFace::CLOCKWISE {
            ir::FrontFace::Clockwise
        } else {
            ir::FrontFace::CounterClockwise
        };
        match with_device(device, move |runtime| {
            let pass = runtime
                .resources
                .render_passes
                .get(&render_pass)
                .ok_or(INVALID)?;
            if pass.format != vk::Format::R8G8B8A8_UNORM {
                return Err(UNSUPPORTED);
            }
            let layout = runtime
                .resources
                .pipeline_layouts
                .get(&layout)
                .ok_or(INVALID)?
                .clone();
            let vertex_id = *runtime
                .resources
                .shaders
                .get(&vertex_module)
                .ok_or(INVALID)?;
            let fragment_id = *runtime
                .resources
                .shaders
                .get(&fragment_module)
                .ok_or(INVALID)?;
            let vertex = ir::ShaderEntryPoint::new(
                runtime
                    .table
                    .shader_module_ref(vertex_id)
                    .map_err(|_| INVALID)?,
                ir::ShaderStage::Vertex,
                vertex_name,
            )
            .map_err(|_| INVALID)?;
            let fragment = ir::ShaderEntryPoint::new(
                runtime
                    .table
                    .shader_module_ref(fragment_id)
                    .map_err(|_| INVALID)?,
                ir::ShaderStage::Fragment,
                fragment_name,
            )
            .map_err(|_| INVALID)?;
            let desc = ir::ProgrammableRenderPipelineDesc::new(
                vertex,
                fragment,
                layout,
                ir::TextureFormat::Rgba8Unorm,
                None,
                ir::PrimitiveTopology::TriangleList,
                ir::BlendState::REPLACE,
                ir::RasterState::new(ir::CullMode::None, front_face),
            )
            .map_err(|_| INVALID)?;
            let id = runtime
                .table
                .define_programmable_render_pipeline(desc)
                .map_err(crate::resources::failure)?
                .id();
            runtime
                .cache
                .validate_programmable_render_pipeline(id)
                .map_err(|error| match error {
                    sgfx_backend_wgpu::Error::DeviceLost => vk::Result::ERROR_DEVICE_LOST,
                    sgfx_backend_wgpu::Error::InvalidIr(error) => crate::resources::failure(error),
                    _ => UNSUPPORTED,
                })?;
            let handle = vk::Pipeline::from_raw(next_id());
            runtime
                .resources
                .pipelines
                .insert(handle, crate::resources::Pipeline::Graphics(id));
            runtime.resources.graphics_extents.insert(handle, extent);
            Ok(handle)
        }) {
            Ok(handle) => *output.add(index) = handle,
            Err(error) => {
                first_error.get_or_insert(error);
            }
        }
    }
    first_error.unwrap_or(vk::Result::SUCCESS)
}

pub(crate) fn lookup(name: &CStr) -> vk::PFN_vkVoidFunction {
    macro_rules! function {
        ($name:ident, $ty:ident) => {
            Some(unsafe { std::mem::transmute::<vk::$ty, unsafe extern "system" fn()>($name) })
        };
    }
    match name.to_bytes() {
        b"vkCreateImage" => function!(create_image, PFN_vkCreateImage),
        b"vkDestroyImage" => function!(destroy_image, PFN_vkDestroyImage),
        b"vkGetImageMemoryRequirements" => function!(
            get_image_memory_requirements,
            PFN_vkGetImageMemoryRequirements
        ),
        b"vkBindImageMemory" => function!(bind_image_memory, PFN_vkBindImageMemory),
        b"vkCreateImageView" => function!(create_image_view, PFN_vkCreateImageView),
        b"vkDestroyImageView" => function!(destroy_image_view, PFN_vkDestroyImageView),
        b"vkCreateRenderPass" => function!(create_render_pass, PFN_vkCreateRenderPass),
        b"vkDestroyRenderPass" => function!(destroy_render_pass, PFN_vkDestroyRenderPass),
        b"vkCreateFramebuffer" => function!(create_framebuffer, PFN_vkCreateFramebuffer),
        b"vkDestroyFramebuffer" => function!(destroy_framebuffer, PFN_vkDestroyFramebuffer),
        b"vkCreateGraphicsPipelines" => {
            function!(create_graphics_pipelines, PFN_vkCreateGraphicsPipelines)
        }
        _ => None,
    }
}
