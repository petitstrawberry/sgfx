//! Bounded offscreen color/depth attachments and graphics pipeline state.

// Bound image storage and readback to 64 MiB for four-byte pixels.
pub(crate) const MAX_IMAGE_DIMENSION: u32 = 4096;

use ash::vk::{self, Handle};
use sgfx::ir;
use std::ffi::CStr;

use crate::api::{next_id, with_device};

pub(crate) struct Image {
    pub id: ir::TextureId,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub flags: vk::ImageCreateFlags,
    pub usage: vk::ImageUsageFlags,
    pub bound: Option<(vk::DeviceMemory, u64)>,
    pub swapchain: Option<vk::SwapchainKHR>,
    #[cfg(target_os = "scarlet")]
    pub shared: Option<sgfx::driver::PresentationImage>,
}

impl Image {
    pub(crate) fn mip_extent(&self, mip: u32) -> Result<vk::Extent3D, vk::Result> {
        if mip >= self.mip_levels {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        Ok(vk::Extent3D {
            width: (self.extent.width >> mip).max(1),
            height: (self.extent.height >> mip).max(1),
            depth: 1,
        })
    }
    pub(crate) fn byte_size(&self) -> u64 {
        (0..self.mip_levels)
            .map(|mip| {
                let size = self.mip_extent(mip).expect("validated image mip count");
                u64::from(size.width)
                    * u64::from(size.height)
                    * u64::from(texture_format(self.format).unwrap().bytes_per_pixel())
                    * u64::from(self.array_layers)
            })
            .sum()
    }
    pub(crate) fn usable(&self) -> bool {
        self.bound.is_some() || self.swapchain.is_some()
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ImageView {
    pub image: vk::Image,
    pub desc: ir::TextureViewDesc,
    pub components: [u8; 4],
}

pub(crate) use crate::render_pass::RenderPass;

pub(crate) struct Framebuffer {
    pub attachments: Vec<vk::ImageView>,
    pub render_pass: RenderPass,
    pub width: u32,
    pub height: u32,
}

const UNSUPPORTED: vk::Result = vk::Result::ERROR_FEATURE_NOT_PRESENT;
const INVALID: vk::Result = vk::Result::ERROR_INITIALIZATION_FAILED;

pub(crate) fn image_usage(format: vk::Format) -> vk::ImageUsageFlags {
    match format {
        vk::Format::R8G8B8A8_UNORM
        | vk::Format::B8G8R8A8_UNORM
        | vk::Format::R8G8B8A8_SRGB
        | vk::Format::B8G8R8A8_SRGB
        | vk::Format::R8_UNORM => {
            (if format == vk::Format::R8G8B8A8_UNORM {
                vk::ImageUsageFlags::STORAGE
            } else {
                vk::ImageUsageFlags::empty()
            }) | vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::INPUT_ATTACHMENT
                | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
        }
        vk::Format::D32_SFLOAT => {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::INPUT_ATTACHMENT
                | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT
        }
        _ => vk::ImageUsageFlags::empty(),
    }
}

pub(crate) fn valid_image_usage(usage: vk::ImageUsageFlags) -> bool {
    // Transient attachments can use ordinary device memory. Lazy allocation is
    // an optional memory type, not a requirement on the image's backing store.
    !usage.contains(vk::ImageUsageFlags::TRANSIENT_ATTACHMENT)
        || (vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
            | vk::ImageUsageFlags::INPUT_ATTACHMENT
            | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT)
            .contains(usage)
}

pub(crate) fn texture_format(format: vk::Format) -> Option<ir::TextureFormat> {
    match format {
        vk::Format::R8G8B8A8_UNORM => Some(ir::TextureFormat::Rgba8Unorm),
        vk::Format::R8G8B8A8_SRGB => Some(ir::TextureFormat::Rgba8UnormSrgb),
        vk::Format::B8G8R8A8_UNORM => Some(ir::TextureFormat::Bgra8Unorm),
        vk::Format::B8G8R8A8_SRGB => Some(ir::TextureFormat::Bgra8UnormSrgb),
        vk::Format::R8_UNORM => Some(ir::TextureFormat::R8Unorm),
        vk::Format::D32_SFLOAT => Some(ir::TextureFormat::Depth32Float),
        _ => None,
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
        || !(vk::ImageCreateFlags::MUTABLE_FORMAT | vk::ImageCreateFlags::CUBE_COMPATIBLE)
            .contains(info.flags)
        || info.s_type != vk::StructureType::IMAGE_CREATE_INFO
        || info.image_type != vk::ImageType::TYPE_2D
        || supported.is_empty()
        || info.tiling != vk::ImageTiling::OPTIMAL
        || info.samples != vk::SampleCountFlags::TYPE_1
        || info.mip_levels == 0
        || (info.mip_levels > 1
            && info.usage.intersects(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ))
        || info.array_layers == 0
        || (info.flags.contains(vk::ImageCreateFlags::CUBE_COMPATIBLE)
            && (info.array_layers < 6 || info.extent.width != info.extent.height))
        || info.extent.depth != 1
        || info.extent.width == 0
        || info.extent.height == 0
        || info.extent.width > MAX_IMAGE_DIMENSION
        || info.extent.height > MAX_IMAGE_DIMENSION
        || info.usage.is_empty()
        || !supported.contains(info.usage)
        || !valid_image_usage(info.usage)
        || info.sharing_mode != vk::SharingMode::EXCLUSIVE
        || info.initial_layout != vk::ImageLayout::UNDEFINED
    {
        return UNSUPPORTED;
    }
    let extent = info.extent;
    let mip_levels = info.mip_levels;
    let array_layers = info.array_layers;
    let flags = info.flags;
    let format = info.format;
    let image_usage = info.usage;
    let mut usage = ir::TextureUsage::empty();
    if info.usage.contains(vk::ImageUsageFlags::STORAGE) {
        usage |= ir::TextureUsage::STORAGE;
    }
    if info
        .usage
        .intersects(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::INPUT_ATTACHMENT)
    {
        usage |= ir::TextureUsage::SAMPLED;
    }
    if info.usage.intersects(
        vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
            | vk::ImageUsageFlags::INPUT_ATTACHMENT,
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
        // Some applications create a sampled image with STORAGE usage even
        // when this device exposes no storage-image shader support. The
        // resource can still be uploaded and sampled; an actual storage
        // binding remains subject to the backend's capability check.
        if image_usage.contains(vk::ImageUsageFlags::STORAGE)
            && !runtime.capabilities.supports_storage_images()
            && !(format == vk::Format::R8G8B8A8_UNORM
                && image_usage
                    .contains(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST))
        {
            return Err(UNSUPPORTED);
        }
        let ir_format = texture_format(format).ok_or(UNSUPPORTED)?;
        if (!runtime.capabilities.supports_srgb_texture_views()
            && matches!(
                format,
                vk::Format::R8G8B8A8_SRGB | vk::Format::B8G8R8A8_SRGB
            ))
            || (!runtime.capabilities.supports_typed_texture_views()
                && format == vk::Format::D32_SFLOAT
                && image_usage.contains(vk::ImageUsageFlags::SAMPLED))
        {
            return Err(UNSUPPORTED);
        }
        if extent.width > runtime.capabilities.limits().max_image_dimension_2d
            || extent.height > runtime.capabilities.limits().max_image_dimension_2d
            || mip_levels > runtime.capabilities.limits().max_image_mip_levels
            || array_layers > runtime.capabilities.limits().max_image_array_layers
        {
            return Err(UNSUPPORTED);
        }
        let desc = ir::TextureDesc::new(ir_format, size, usage)
            .map_err(|_| INVALID)?
            .with_mip_level_count(mip_levels)
            .map_err(|_| UNSUPPORTED)?
            .with_array_layer_count(array_layers)
            .map_err(|_| UNSUPPORTED)?
            .with_cube_compatible(flags.contains(vk::ImageCreateFlags::CUBE_COMPATIBLE))
            .map_err(|_| UNSUPPORTED)?;
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
                mip_levels,
                array_layers,
                flags,
                usage: image_usage,
                bound: None,
                swapchain: None,
                #[cfg(target_os = "scarlet")]
                shared: None,
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
            let removable = runtime
                .resources
                .images
                .get(&image)
                .is_some_and(|image| image.swapchain.is_none());
            if removable && let Some(_data) = runtime.resources.images.remove(&image) {
                #[cfg(target_os = "scarlet")]
                if _data.shared.is_some() {
                    runtime.cache.unmap_presentation_image(_data.id);
                }
            }
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
            size: image.byte_size(),
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
        let size = runtime
            .resources
            .images
            .get(&image)
            .ok_or(INVALID)?
            .byte_size();
        if !offset.is_multiple_of(4)
            || !crate::resources::memory_available(&runtime.resources, memory, offset, size)
        {
            return Err(INVALID);
        }
        let image = runtime.resources.images.get_mut(&image).ok_or(INVALID)?;
        if image.bound.is_some() || image.swapchain.is_some() {
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
        || !matches!(
            info.view_type,
            vk::ImageViewType::TYPE_2D | vk::ImageViewType::TYPE_2D_ARRAY | vk::ImageViewType::CUBE
        )
        || image_usage(info.format).is_empty()
        || range.aspect_mask != image_aspect(info.format)
        || range.level_count == 0
    {
        return UNSUPPORTED;
    }
    let mut components = [0, 1, 2, 3];
    for (slot, value) in [
        info.components.r,
        info.components.g,
        info.components.b,
        info.components.a,
    ]
    .into_iter()
    .enumerate()
    {
        components[slot] = match value {
            vk::ComponentSwizzle::IDENTITY => slot as u8,
            vk::ComponentSwizzle::R => 0,
            vk::ComponentSwizzle::G => 1,
            vk::ComponentSwizzle::B => 2,
            vk::ComponentSwizzle::A => 3,
            vk::ComponentSwizzle::ZERO => 4,
            vk::ComponentSwizzle::ONE => 5,
            _ => return UNSUPPORTED,
        };
    }
    if components != [0, 1, 2, 3] && info.format == vk::Format::D32_SFLOAT {
        return UNSUPPORTED;
    }
    let image = info.image;
    let view_type = info.view_type;
    let format = info.format;
    match with_device(device, move |runtime| {
        let data = runtime.resources.images.get(&image).ok_or(INVALID)?;
        if !data.usable()
            || (data.format != format && !data.flags.contains(vk::ImageCreateFlags::MUTABLE_FORMAT))
        {
            return Err(INVALID);
        }
        if view_type == vk::ImageViewType::CUBE
            && !data.flags.contains(vk::ImageCreateFlags::CUBE_COMPATIBLE)
        {
            return Err(UNSUPPORTED);
        }
        let desc = runtime
            .table
            .texture(runtime.table.texture_ref(data.id).map_err(|_| INVALID)?)
            .map_err(|_| INVALID)?;
        let levels = if range.level_count == vk::REMAINING_MIP_LEVELS {
            data.mip_levels
                .checked_sub(range.base_mip_level)
                .ok_or(INVALID)?
        } else {
            range.level_count
        };
        let layers = if range.layer_count == vk::REMAINING_ARRAY_LAYERS {
            data.array_layers
                .checked_sub(range.base_array_layer)
                .ok_or(INVALID)?
        } else {
            range.layer_count
        };
        if !runtime.capabilities.supports_srgb_texture_views()
            && matches!(
                format,
                vk::Format::R8G8B8A8_SRGB | vk::Format::B8G8R8A8_SRGB
            )
        {
            return Err(UNSUPPORTED);
        }
        if !runtime.capabilities.supports_typed_texture_views()
            && (view_type != vk::ImageViewType::TYPE_2D
                || data.format != format
                || range.base_mip_level != 0
                || levels != data.mip_levels
                || range.base_array_layer != 0
                || layers != 1
                || components != [0, 1, 2, 3])
        {
            return Err(UNSUPPORTED);
        }
        let view = ir::TextureViewDesc::new(
            desc,
            texture_format(format).ok_or(UNSUPPORTED)?,
            match view_type {
                vk::ImageViewType::TYPE_2D_ARRAY => ir::TextureViewDimension::D2Array,
                vk::ImageViewType::CUBE => ir::TextureViewDimension::Cube,
                _ => ir::TextureViewDimension::D2,
            },
            range.base_mip_level,
            levels,
            range.base_array_layer,
            layers,
        )
        .map_err(|_| UNSUPPORTED)?;
        let handle = vk::ImageView::from_raw(next_id());
        runtime.resources.views.insert(
            handle,
            ImageView {
                image,
                desc: view,
                components,
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
    if !allocator.is_null() {
        return UNSUPPORTED;
    }
    let pass = match crate::render_pass::parse(&*info) {
        Ok(pass) => pass,
        Err(error) => return error,
    };
    match with_device(device, move |runtime| {
        if pass
            .subpasses
            .iter()
            .any(|s| s.colors.len() > runtime.capabilities.limits().max_color_attachments as usize)
            || (pass
                .subpasses
                .iter()
                .any(|s| !s.inputs.is_empty() || s.read_only_depth)
                && !runtime.capabilities.supports_typed_texture_views())
        {
            return Err(UNSUPPORTED);
        }
        let handle = vk::RenderPass::from_raw(next_id());
        runtime.resources.render_passes.insert(handle, pass);
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
        || !(1..=8).contains(&info.attachment_count)
        || info.p_attachments.is_null()
        || info.layers != 1
        || info.width == 0
        || info.height == 0
    {
        return UNSUPPORTED;
    }
    let views =
        std::slice::from_raw_parts(info.p_attachments, info.attachment_count as usize).to_vec();
    let render_pass = info.render_pass;
    let (width, height) = (info.width, info.height);
    match with_device(device, move |runtime| {
        let pass = runtime
            .resources
            .render_passes
            .get(&render_pass)
            .ok_or(INVALID)?;
        if views.len() != pass.attachments.len() {
            return Err(INVALID);
        }
        for (index, view) in views.iter().enumerate() {
            let view = runtime.resources.views.get(view).ok_or(INVALID)?;
            let image = runtime.resources.images.get(&view.image).ok_or(INVALID)?;
            if image.format != pass.attachments[index].format || !image.usable() {
                return Err(INVALID);
            }
            if view.components != [0, 1, 2, 3]
                || image.array_layers != 1
                || view.desc.dimension() != ir::TextureViewDimension::D2
                || view.desc.base_mip_level() != 0
                || view.desc.mip_level_count() != 1
                || view.desc.base_array_layer() != 0
                || view.desc.array_layer_count() != 1
                || Some(view.desc.format()) != texture_format(image.format)
            {
                return Err(UNSUPPORTED);
            }
        }
        for (index, view) in views.iter().enumerate() {
            let view = runtime.resources.views.get(view).ok_or(INVALID)?;
            let image = runtime.resources.images.get(&view.image).ok_or(INVALID)?;
            let color = pass.subpasses.iter().any(|s| s.colors.contains(&index));
            let depth = pass.subpasses.iter().any(|s| s.depth == Some(index));
            let input = pass
                .subpasses
                .iter()
                .any(|s| s.inputs.contains(&Some(index)));
            let mut required = vk::ImageUsageFlags::empty();
            if color {
                required |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
            }
            if depth {
                required |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
            }
            if input {
                required |= vk::ImageUsageFlags::INPUT_ATTACHMENT;
            }
            if image.extent.width != width
                || image.extent.height != height
                || !image.usage.contains(required)
            {
                return Err(UNSUPPORTED);
            }
        }
        let render_pass = pass.clone();
        let handle = vk::Framebuffer::from_raw(next_id());
        runtime.resources.framebuffers.insert(
            handle,
            Framebuffer {
                attachments: views,
                render_pass,
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
) -> Result<[(vk::ShaderModule, String, crate::resources::Specialization); 2], vk::Result> {
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
        if slot
            .replace((
                stage.module,
                name,
                crate::resources::specialization(stage.p_specialization_info)?,
            ))
            .is_some()
        {
            return Err(INVALID);
        }
    }
    Ok([vertex.ok_or(INVALID)?, fragment.ok_or(INVALID)?])
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GraphicsDynamicState {
    pub viewport: Option<vk::Viewport>,
    pub scissor: Option<vk::Rect2D>,
}

struct GraphicsState {
    dynamic: GraphicsDynamicState,
    colors: Vec<(ir::BlendState, ir::ColorWriteMask)>,
    extent: vk::Extent2D,
    vertex: Vec<ir::VertexBufferLayout>,
    depth: Option<ir::DepthState>,
    raster: ir::RasterState,
    topology: ir::PrimitiveTopology,
}

pub(crate) fn compare(op: vk::CompareOp) -> Result<ir::CompareFunction, vk::Result> {
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
) -> Result<Vec<ir::VertexBufferLayout>, vk::Result> {
    if vertex.vertex_binding_description_count == 0
        && vertex.vertex_attribute_description_count == 0
    {
        return Ok(Vec::new());
    }
    if vertex.vertex_binding_description_count == 0
        || vertex.vertex_binding_description_count > ir::MAX_VERTEX_BUFFERS as u32
        || vertex.p_vertex_binding_descriptions.is_null()
        || vertex.vertex_attribute_description_count == 0
        || vertex.vertex_attribute_description_count > 16
        || vertex.p_vertex_attribute_descriptions.is_null()
    {
        return Err(UNSUPPORTED);
    }
    let mut buffers = std::collections::BTreeMap::new();
    for binding in std::slice::from_raw_parts(
        vertex.p_vertex_binding_descriptions,
        vertex.vertex_binding_description_count as usize,
    ) {
        if binding.binding >= ir::MAX_VERTEX_BUFFERS as u32
            || binding.input_rate != vk::VertexInputRate::VERTEX
            || binding.stride > 2048
            || !binding.stride.is_multiple_of(4)
            || buffers
                .insert(binding.binding, (binding.stride, Vec::new()))
                .is_some()
        {
            return Err(UNSUPPORTED);
        }
    }
    for attribute in std::slice::from_raw_parts(
        vertex.p_vertex_attribute_descriptions,
        vertex.vertex_attribute_description_count as usize,
    ) {
        if attribute.location >= 16
            || attribute.offset > 2047
            || !attribute.offset.is_multiple_of(4)
        {
            return Err(UNSUPPORTED);
        }
        let format = match attribute.format {
            vk::Format::R32G32_SFLOAT => ir::VertexFormat::Float32x2,
            vk::Format::R32G32B32_SFLOAT => ir::VertexFormat::Float32x3,
            vk::Format::R32G32B32A32_SFLOAT => ir::VertexFormat::Float32x4,
            vk::Format::R8G8B8A8_UNORM | vk::Format::A8B8G8R8_UNORM_PACK32 => {
                ir::VertexFormat::Unorm8x4
            }
            vk::Format::R32_SINT => ir::VertexFormat::Sint32,
            vk::Format::R32_UINT => ir::VertexFormat::Uint32,
            vk::Format::R16G16_SFLOAT => ir::VertexFormat::Float16x2,
            vk::Format::R16G16B16A16_SFLOAT => ir::VertexFormat::Float16x4,
            vk::Format::R16G16B16A16_SINT => ir::VertexFormat::Sint16x4,
            vk::Format::A2B10G10R10_SNORM_PACK32 => ir::VertexFormat::Snorm10_10_10_2,
            _ => return Err(UNSUPPORTED),
        };

        let (_, attributes) = buffers.get_mut(&attribute.binding).ok_or(UNSUPPORTED)?;
        attributes.push(ir::VertexAttribute::new(
            attribute.location,
            format,
            attribute.offset,
        ));
    }
    buffers
        .into_iter()
        .enumerate()
        .map(|(index, (slot, (stride, attributes)))| {
            if slot != index as u32 {
                return Err(UNSUPPORTED);
            }
            ir::VertexBufferLayout::new(stride, attributes).map_err(|_| UNSUPPORTED)
        })
        .collect()
}

unsafe fn graphics_state(
    info: &vk::GraphicsPipelineCreateInfo<'_>,
) -> Result<GraphicsState, vk::Result> {
    if !info.p_next.is_null()
        || !info.flags.is_empty()
        || info.s_type != vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO
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
    let mut dynamic_viewport = false;
    let mut dynamic_scissor = false;
    if let Some(dynamic) = info.p_dynamic_state.as_ref() {
        if dynamic.s_type != vk::StructureType::PIPELINE_DYNAMIC_STATE_CREATE_INFO
            || !dynamic.p_next.is_null()
            || !dynamic.flags.is_empty()
            || dynamic.dynamic_state_count > 2
            || (dynamic.dynamic_state_count > 0 && dynamic.p_dynamic_states.is_null())
        {
            return Err(UNSUPPORTED);
        }
        for i in 0..dynamic.dynamic_state_count as usize {
            match *dynamic.p_dynamic_states.add(i) {
                vk::DynamicState::VIEWPORT if !dynamic_viewport => dynamic_viewport = true,
                vk::DynamicState::SCISSOR if !dynamic_scissor => dynamic_scissor = true,
                _ => return Err(UNSUPPORTED),
            }
        }
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
        || !matches!(
            assembly.topology,
            vk::PrimitiveTopology::TRIANGLE_LIST | vk::PrimitiveTopology::TRIANGLE_STRIP
        )
        || assembly.primitive_restart_enable != vk::FALSE
        || !viewport.p_next.is_null()
        || !viewport.flags.is_empty()
        || viewport.viewport_count != 1
        || viewport.scissor_count != 1
        || (!dynamic_viewport && viewport.p_viewports.is_null())
        || (!dynamic_scissor && viewport.p_scissors.is_null())
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
        || !(1..=ir::MAX_COLOR_ATTACHMENTS as u32).contains(&blend.attachment_count)
        || blend.p_attachments.is_null()
    {
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
    let view = if dynamic_viewport {
        None
    } else {
        Some(*viewport.p_viewports)
    };
    let scissor = if dynamic_scissor {
        None
    } else {
        Some(*viewport.p_scissors)
    };
    if let Some(view) = view {
        ir::Viewport::new(
            view.x,
            view.y,
            view.width,
            view.height,
            view.min_depth,
            view.max_depth,
        )
        .map_err(|_| UNSUPPORTED)?;
        if view.width > MAX_IMAGE_DIMENSION as f32 || view.height > MAX_IMAGE_DIMENSION as f32 {
            return Err(UNSUPPORTED);
        }
    }
    if scissor.is_some_and(|s| s.offset.x < 0 || s.offset.y < 0) {
        return Err(UNSUPPORTED);
    }
    let factor = |v| {
        Ok(match v {
            vk::BlendFactor::ZERO => ir::BlendFactor::Zero,
            vk::BlendFactor::ONE => ir::BlendFactor::One,
            vk::BlendFactor::SRC_ALPHA => ir::BlendFactor::SourceAlpha,
            vk::BlendFactor::ONE_MINUS_SRC_ALPHA => ir::BlendFactor::OneMinusSourceAlpha,
            vk::BlendFactor::DST_ALPHA => ir::BlendFactor::DestinationAlpha,
            vk::BlendFactor::ONE_MINUS_DST_ALPHA => ir::BlendFactor::OneMinusDestinationAlpha,
            _ => return Err(UNSUPPORTED),
        })
    };
    let op = |v| {
        Ok(match v {
            vk::BlendOp::ADD => ir::BlendOp::Add,
            vk::BlendOp::SUBTRACT => ir::BlendOp::Subtract,
            vk::BlendOp::REVERSE_SUBTRACT => ir::BlendOp::ReverseSubtract,
            _ => return Err(UNSUPPORTED),
        })
    };
    let mut colors = Vec::new();
    for color in std::slice::from_raw_parts(blend.p_attachments, blend.attachment_count as usize) {
        if !matches!(color.blend_enable, vk::FALSE | vk::TRUE)
            || !vk::ColorComponentFlags::RGBA.contains(color.color_write_mask)
        {
            return Err(UNSUPPORTED);
        }
        let blending = if color.blend_enable == vk::FALSE {
            ir::BlendState::REPLACE
        } else {
            ir::BlendState::new(
                ir::BlendComponent::new(
                    factor(color.src_color_blend_factor)?,
                    factor(color.dst_color_blend_factor)?,
                    op(color.color_blend_op)?,
                ),
                ir::BlendComponent::new(
                    factor(color.src_alpha_blend_factor)?,
                    factor(color.dst_alpha_blend_factor)?,
                    op(color.alpha_blend_op)?,
                ),
            )
        };
        colors.push((
            blending,
            ir::ColorWriteMask::from_bits(color.color_write_mask.as_raw() as u8)
                .map_err(|_| UNSUPPORTED)?,
        ));
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
        extent: scissor.map(|s| s.extent).unwrap_or_default(),
        dynamic: GraphicsDynamicState {
            viewport: view,
            scissor,
        },
        colors,
        vertex: vertex_layout(vertex)?,
        depth,
        raster: ir::RasterState::new(cull, front),
        topology: if assembly.topology == vk::PrimitiveTopology::TRIANGLE_STRIP {
            ir::PrimitiveTopology::TriangleStrip
        } else {
            ir::PrimitiveTopology::TriangleList
        },
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
                (vertex_module, vertex_name, vertex_values),
                (fragment_module, fragment_name, fragment_values),
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
        let subpass_index = info.subpass as usize;
        // Shader-module normalization already compensates for Vulkan's positive
        // viewport height. Framebuffer winding therefore maps without reversal.
        match with_device(device, move |runtime| {
            let pass = runtime
                .resources
                .render_passes
                .get(&render_pass)
                .ok_or(INVALID)?;
            let pass = pass.clone();
            let subpass = pass.subpasses.get(subpass_index).ok_or(INVALID)?;
            if subpass.colors.len() != state.colors.len() {
                return Err(INVALID);
            }
            let targets = subpass
                .colors
                .iter()
                .zip(&state.colors)
                .map(|(&i, &(blend, mask))| {
                    ir::ColorTargetState::new(
                        texture_format(pass.attachments[i].format).ok_or(UNSUPPORTED)?,
                        blend,
                        mask,
                    )
                    .map_err(|_| INVALID)
                })
                .collect::<Result<Vec<_>, vk::Result>>()?;
            let target_format = targets[0].format();
            let input_depths = pass.input_depths(subpass_index);
            let layout = runtime
                .resources
                .pipeline_layouts
                .get(&layout)
                .ok_or(INVALID)?
                .clone();
            let vertex_id = runtime
                .resources
                .shaders
                .get_mut(&vertex_module)
                .ok_or(INVALID)?
                .variant(&runtime.table, &vertex_values)?;
            let input_bindings = runtime
                .resources
                .shaders
                .get(&fragment_module)
                .ok_or(INVALID)?
                .input_bindings()?;
            let fragment_id = runtime
                .resources
                .shaders
                .get_mut(&fragment_module)
                .ok_or(INVALID)?
                .variant_for_subpass(&runtime.table, &fragment_values, &input_depths)?;
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
                vertex.clone(),
                fragment.clone(),
                crate::spirv::specialize_layout(&runtime.table, &layout, &[&vertex, &fragment])?,
                target_format,
                None,
                state.topology,
                targets[0].blend(),
                state.raster,
            )
            .map_err(|_| INVALID)?
            .with_vertex_buffers(state.vertex)
            .map_err(|_| INVALID)?
            .with_color_targets(targets)
            .map_err(|_| INVALID)?;
            if subpass.read_only_depth && state.depth.is_some_and(|d| d.write_enabled()) {
                return Err(INVALID);
            }
            if subpass.depth.is_some() {
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
                .graphics_subpasses
                .insert(handle, (pass, subpass_index));
            runtime
                .resources
                .graphics_inputs
                .insert(handle, input_bindings);
            runtime
                .resources
                .graphics_extents
                .insert(handle, state.extent);
            runtime
                .resources
                .graphics_state
                .insert(handle, state.dynamic);
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
        let parsed = unsafe { vertex_layout(&info) }.unwrap().remove(0);
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
