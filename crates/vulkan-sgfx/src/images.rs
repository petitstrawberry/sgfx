//! Bounded offscreen color/depth attachments and graphics pipeline state.

use ash::vk::{self, Handle};
use sgfx::ir;
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
    pub depth_load_op: Option<vk::AttachmentLoadOp>,
    pub depth_store_op: vk::AttachmentStoreOp,
}

pub(crate) struct Framebuffer {
    pub view: vk::ImageView,
    pub render_pass: vk::RenderPass,
    pub image: vk::Image,
    pub depth: Option<(vk::ImageView, vk::Image)>,
    pub width: u32,
    pub height: u32,
}

const UNSUPPORTED: vk::Result = vk::Result::ERROR_FEATURE_NOT_PRESENT;
const INVALID: vk::Result = vk::Result::ERROR_INITIALIZATION_FAILED;

pub(crate) fn image_usage(format: vk::Format) -> vk::ImageUsageFlags {
    match format {
        vk::Format::R8G8B8A8_UNORM => {
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
        }
        vk::Format::D32_SFLOAT => vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        _ => vk::ImageUsageFlags::empty(),
    }
}

pub(crate) fn image_aspect(format: vk::Format) -> vk::ImageAspectFlags {
    if format == vk::Format::D32_SFLOAT {
        vk::ImageAspectFlags::DEPTH
    } else {
        vk::ImageAspectFlags::COLOR
    }
}

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
    let supported = image_usage(info.format);
    if !allocator.is_null()
        || !info.p_next.is_null()
        || !info.flags.is_empty()
        || info.s_type != vk::StructureType::IMAGE_CREATE_INFO
        || info.image_type != vk::ImageType::TYPE_2D
        || supported.is_empty()
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
    if info.usage.intersects(
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
    ) {
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
        let ir_format = if format == vk::Format::D32_SFLOAT {
            ir::TextureFormat::Depth32Float
        } else {
            ir::TextureFormat::Rgba8Unorm
        };
        let desc = ir::TextureDesc::new(ir_format, size, usage).map_err(|_| INVALID)?;
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
        || image_usage(info.format).is_empty()
        || range.aspect_mask != image_aspect(info.format)
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
        || !matches!(info.attachment_count, 1 | 2)
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
        || subpass.preserve_attachment_count != 0
    {
        return UNSUPPORTED;
    }
    let color = &*subpass.p_color_attachments;
    if color.attachment != 0 || color.layout != vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
        return UNSUPPORTED;
    }
    let mut depth_load_op = None;
    let mut depth_store_op = vk::AttachmentStoreOp::DONT_CARE;
    if info.attachment_count == 2 {
        let Some(reference) = subpass.p_depth_stencil_attachment.as_ref() else {
            return UNSUPPORTED;
        };
        let depth = &*info.p_attachments.add(1);
        if reference.attachment != 1
            || reference.layout != vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            || !depth.flags.is_empty()
            || depth.format != vk::Format::D32_SFLOAT
            || depth.samples != vk::SampleCountFlags::TYPE_1
            || !matches!(
                depth.load_op,
                vk::AttachmentLoadOp::CLEAR
                    | vk::AttachmentLoadOp::LOAD
                    | vk::AttachmentLoadOp::DONT_CARE
            )
            || !matches!(
                depth.store_op,
                vk::AttachmentStoreOp::STORE | vk::AttachmentStoreOp::DONT_CARE
            )
            || depth.stencil_load_op != vk::AttachmentLoadOp::DONT_CARE
            || depth.stencil_store_op != vk::AttachmentStoreOp::DONT_CARE
            || !matches!(
                depth.initial_layout,
                vk::ImageLayout::UNDEFINED
                    | vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                    | vk::ImageLayout::GENERAL
            )
            || (depth.load_op == vk::AttachmentLoadOp::LOAD
                && depth.initial_layout == vk::ImageLayout::UNDEFINED)
            || !matches!(
                depth.final_layout,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL | vk::ImageLayout::GENERAL
            )
        {
            return UNSUPPORTED;
        }
        depth_load_op = Some(depth.load_op);
        depth_store_op = depth.store_op;
    } else if !subpass.p_depth_stencil_attachment.is_null() {
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
            | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
            | vk::PipelineStageFlags::TRANSFER;
        let accesses = vk::AccessFlags::COLOR_ATTACHMENT_READ
            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
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
        runtime.resources.render_passes.insert(
            handle,
            RenderPass {
                format,
                load_op,
                depth_load_op,
                depth_store_op,
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
        || !matches!(info.attachment_count, 1 | 2)
        || info.p_attachments.is_null()
        || info.layers != 1
        || info.width == 0
        || info.height == 0
    {
        return UNSUPPORTED;
    }
    let view = *info.p_attachments;
    let depth_view = (info.attachment_count == 2).then(|| *info.p_attachments.add(1));
    let render_pass = info.render_pass;
    let (width, height) = (info.width, info.height);
    match with_device(device, move |runtime| {
        let pass = runtime
            .resources
            .render_passes
            .get(&render_pass)
            .ok_or(INVALID)?;
        let image = *runtime.resources.views.get(&view).ok_or(INVALID)?;
        if depth_view.is_some() != pass.depth_load_op.is_some() {
            return Err(UNSUPPORTED);
        }
        let depth = depth_view
            .map(|view| {
                let image = *runtime.resources.views.get(&view).ok_or(INVALID)?;
                let data = runtime.resources.images.get(&image).ok_or(INVALID)?;
                if data.bound.is_none()
                    || data.format != vk::Format::D32_SFLOAT
                    || data.extent.width != width
                    || data.extent.height != height
                {
                    return Err(UNSUPPORTED);
                }
                Ok((view, image))
            })
            .transpose()?;
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
                depth,
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

struct GraphicsState {
    extent: vk::Extent2D,
    vertex: Option<ir::VertexBufferLayout>,
    depth: Option<ir::DepthState>,
    raster: ir::RasterState,
}

fn compare(op: vk::CompareOp) -> Result<ir::CompareFunction, vk::Result> {
    Ok(match op {
        vk::CompareOp::NEVER => ir::CompareFunction::Never,
        vk::CompareOp::LESS => ir::CompareFunction::Less,
        vk::CompareOp::EQUAL => ir::CompareFunction::Equal,
        vk::CompareOp::LESS_OR_EQUAL => ir::CompareFunction::LessEqual,
        vk::CompareOp::GREATER => ir::CompareFunction::Greater,
        vk::CompareOp::NOT_EQUAL => ir::CompareFunction::NotEqual,
        vk::CompareOp::GREATER_OR_EQUAL => ir::CompareFunction::GreaterEqual,
        vk::CompareOp::ALWAYS => ir::CompareFunction::Always,
        _ => return Err(UNSUPPORTED),
    })
}

unsafe fn vertex_layout(
    vertex: &vk::PipelineVertexInputStateCreateInfo<'_>,
) -> Result<Option<ir::VertexBufferLayout>, vk::Result> {
    if vertex.vertex_binding_description_count == 0
        && vertex.vertex_attribute_description_count == 0
    {
        return Ok(None);
    }
    if vertex.vertex_binding_description_count != 1
        || vertex.p_vertex_binding_descriptions.is_null()
        || vertex.vertex_attribute_description_count == 0
        || vertex.vertex_attribute_description_count > 16
        || vertex.p_vertex_attribute_descriptions.is_null()
    {
        return Err(UNSUPPORTED);
    }
    let binding = &*vertex.p_vertex_binding_descriptions;
    if binding.binding != 0
        || binding.input_rate != vk::VertexInputRate::VERTEX
        || binding.stride > 2048
        || !binding.stride.is_multiple_of(4)
    {
        return Err(UNSUPPORTED);
    }
    let mut attributes = Vec::new();
    for index in 0..vertex.vertex_attribute_description_count as usize {
        let attribute = &*vertex.p_vertex_attribute_descriptions.add(index);
        if attribute.binding != 0
            || attribute.location >= 16
            || attribute.offset > 2047
            || !attribute.offset.is_multiple_of(4)
        {
            return Err(UNSUPPORTED);
        }
        let format = match attribute.format {
            vk::Format::R32G32_SFLOAT => ir::VertexFormat::Float32x2,
            vk::Format::R32G32B32_SFLOAT => ir::VertexFormat::Float32x3,
            vk::Format::R32G32B32A32_SFLOAT => ir::VertexFormat::Float32x4,
            vk::Format::R8G8B8A8_UNORM => ir::VertexFormat::Unorm8x4,
            _ => return Err(UNSUPPORTED),
        };
        attributes.push(ir::VertexAttribute::new(
            attribute.location,
            format,
            attribute.offset,
        ));
    }
    ir::VertexBufferLayout::new(binding.stride, attributes)
        .map(Some)
        .map_err(|_| UNSUPPORTED)
}

unsafe fn graphics_state(
    info: &vk::GraphicsPipelineCreateInfo<'_>,
) -> Result<GraphicsState, vk::Result> {
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
        || !matches!(
            raster.cull_mode,
            vk::CullModeFlags::NONE | vk::CullModeFlags::FRONT | vk::CullModeFlags::BACK
        )
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
            || !matches!(depth.depth_test_enable, vk::FALSE | vk::TRUE)
            || !matches!(depth.depth_write_enable, vk::FALSE | vk::TRUE)
            || depth.depth_bounds_test_enable != vk::FALSE
            || depth.stencil_test_enable != vk::FALSE)
    {
        return Err(UNSUPPORTED);
    }
    let depth = info
        .p_depth_stencil_state
        .as_ref()
        .map(|depth| {
            let enabled = depth.depth_test_enable == vk::TRUE;
            // Vulkan disables depth writes together with depth testing, irrespective
            // of depthWriteEnable; SGFX's explicit Always state preserves that rule.
            Ok(ir::DepthState::new(
                ir::TextureFormat::Depth32Float,
                if enabled {
                    compare(depth.depth_compare_op)?
                } else {
                    ir::CompareFunction::Always
                },
                enabled && depth.depth_write_enable == vk::TRUE,
            ))
        })
        .transpose()?;
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
    let front = if raster.front_face == vk::FrontFace::CLOCKWISE {
        ir::FrontFace::Clockwise
    } else {
        ir::FrontFace::CounterClockwise
    };
    let cull = match raster.cull_mode {
        vk::CullModeFlags::FRONT => ir::CullMode::Front,
        vk::CullModeFlags::BACK => ir::CullMode::Back,
        _ => ir::CullMode::None,
    };
    Ok(GraphicsState {
        extent: scissor.extent,
        vertex: vertex_layout(vertex)?,
        depth,
        raster: ir::RasterState::new(cull, front),
    })
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
            state,
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
            let mut desc = ir::ProgrammableRenderPipelineDesc::new(
                vertex,
                fragment,
                layout,
                ir::TextureFormat::Rgba8Unorm,
                state.vertex,
                ir::PrimitiveTopology::TriangleList,
                ir::BlendState::REPLACE,
                state.raster,
            )
            .map_err(|_| INVALID)?;
            if pass.depth_load_op.is_some() {
                desc = desc
                    .with_depth_state(state.depth.ok_or(UNSUPPORTED)?)
                    .map_err(|_| INVALID)?;
            } else if state.depth.is_some_and(|depth| {
                depth.write_enabled() || depth.compare() != ir::CompareFunction::Always
            }) {
                return Err(UNSUPPORTED);
            }
            let id = runtime
                .table
                .define_programmable_render_pipeline(desc)
                .map_err(crate::resources::failure)?
                .id();
            runtime
                .cache
                .validate_programmable_render_pipeline(id)
                .map_err(crate::resources::backend_failure)?;
            let handle = vk::Pipeline::from_raw(next_id());
            runtime
                .resources
                .pipelines
                .insert(handle, crate::resources::Pipeline::Graphics(id));
            runtime
                .resources
                .graphics_extents
                .insert(handle, state.extent);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleaved_vertex_layout_is_bounded_and_formats_are_preserved() {
        let bindings = [vk::VertexInputBindingDescription {
            binding: 0,
            stride: 28,
            input_rate: vk::VertexInputRate::VERTEX,
        }];
        let mut attributes = [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 12,
            },
        ];
        let info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);
        let parsed = unsafe { vertex_layout(&info) }.unwrap().unwrap();
        assert_eq!(parsed.stride(), 28);
        assert_eq!(parsed.attributes()[0].format(), ir::VertexFormat::Float32x3);
        assert_eq!(parsed.attributes()[1].offset(), 12);
        attributes[1].location = 0;
        let duplicate = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);
        assert_eq!(unsafe { vertex_layout(&duplicate) }, Err(UNSUPPORTED));
        attributes[1].location = 1;
        attributes[1].offset = 16;
        let overflow = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);
        assert_eq!(unsafe { vertex_layout(&overflow) }, Err(UNSUPPORTED));
        let bindings = [vk::VertexInputBindingDescription {
            input_rate: vk::VertexInputRate::INSTANCE,
            ..bindings[0]
        }];
        let instanced = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);
        assert_eq!(unsafe { vertex_layout(&instanced) }, Err(UNSUPPORTED));
    }

    #[test]
    fn depth_and_culling_translate_without_enabling_stencil_or_bounds() {
        let vertex = vk::PipelineVertexInputStateCreateInfo::default();
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let views = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: 64.0,
            height: 64.0,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissors = [vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D {
                width: 64,
                height: 64,
            },
        }];
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&views)
            .scissors(&scissors);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .line_width(1.0);
        let samples = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let attachments = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
        let mut depth = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS);
        let base = vk::GraphicsPipelineCreateInfo::default()
            .vertex_input_state(&vertex)
            .input_assembly_state(&assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&samples)
            .color_blend_state(&blend);
        let parsed = unsafe { graphics_state(&base.depth_stencil_state(&depth)) }.unwrap();
        assert_eq!(
            parsed.depth.unwrap(),
            ir::DepthState::new(
                ir::TextureFormat::Depth32Float,
                ir::CompareFunction::Less,
                true
            )
        );
        assert_eq!(
            parsed.raster,
            ir::RasterState::new(ir::CullMode::Back, ir::FrontFace::Clockwise)
        );
        depth.depth_test_enable = vk::FALSE;
        let parsed = unsafe { graphics_state(&base.depth_stencil_state(&depth)) }.unwrap();
        assert_eq!(
            parsed.depth.unwrap(),
            ir::DepthState::new(
                ir::TextureFormat::Depth32Float,
                ir::CompareFunction::Always,
                false
            )
        );
        depth.depth_bounds_test_enable = vk::TRUE;
        assert!(unsafe { graphics_state(&base.depth_stencil_state(&depth)) }.is_err());
        depth.depth_bounds_test_enable = vk::FALSE;
        depth.stencil_test_enable = vk::TRUE;
        assert!(unsafe { graphics_state(&base.depth_stencil_state(&depth)) }.is_err());
    }
}
