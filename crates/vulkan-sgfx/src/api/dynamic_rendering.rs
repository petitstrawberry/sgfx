//! Retain rendering attachments without manufacturing Vulkan render-pass handles.
use super::*;
const INVALID: vk::Result = vk::Result::ERROR_INITIALIZATION_FAILED;
const UNSUPPORTED: vk::Result = vk::Result::ERROR_FEATURE_NOT_PRESENT;

#[derive(Clone)]
pub(super) struct Attachment {
    view: vk::ImageView,
    layout: vk::ImageLayout,
    load: vk::AttachmentLoadOp,
    store: vk::AttachmentStoreOp,
    clear: vk::ClearValue,
}
#[derive(Clone)]
pub(super) struct Rendering {
    area: vk::Rect2D,
    colors: Vec<Attachment>,
    depth: Option<Attachment>,
}
unsafe fn attachment(info: &vk::RenderingAttachmentInfo<'_>) -> VkResult<Attachment> {
    if info.s_type != vk::StructureType::RENDERING_ATTACHMENT_INFO
        || !info.p_next.is_null()
        || info.image_view == vk::ImageView::null()
        || !info.resolve_mode.is_empty()
        || !matches!(
            info.load_op,
            vk::AttachmentLoadOp::LOAD
                | vk::AttachmentLoadOp::CLEAR
                | vk::AttachmentLoadOp::DONT_CARE
        )
        || !matches!(
            info.store_op,
            vk::AttachmentStoreOp::STORE | vk::AttachmentStoreOp::DONT_CARE
        )
    {
        return Err(UNSUPPORTED);
    }
    Ok(Attachment {
        view: info.image_view,
        layout: info.image_layout,
        load: info.load_op,
        store: info.store_op,
        clear: info.clear_value,
    })
}
pub(crate) fn enabled(device: vk::Device) -> bool {
    driver(device.as_raw(), Kind::Device).is_ok_and(|d| d.dynamic_rendering_enabled)
}

/// A dynamic pipeline retains only its formats, never application pointers.
pub(crate) unsafe fn pipeline(
    info: &vk::GraphicsPipelineCreateInfo<'_>,
) -> VkResult<Option<crate::render_pass::RenderPass>> {
    if info.render_pass != vk::RenderPass::null() {
        return Ok(None);
    }
    let info = info
        .p_next
        .cast::<vk::PipelineRenderingCreateInfo<'_>>()
        .as_ref()
        .ok_or(INVALID)?;
    if info.s_type != vk::StructureType::PIPELINE_RENDERING_CREATE_INFO
        || !info.p_next.is_null()
        || info.view_mask != 0
        || info.stencil_attachment_format != vk::Format::UNDEFINED
    {
        return Err(UNSUPPORTED);
    }
    let formats = slice(info.p_color_attachment_formats, info.color_attachment_count)?;
    pass(formats, info.depth_attachment_format, false).inspect_err(|error| {
        eprintln!("[SGFX Vulkan] pipeline attachment formats: colors={formats:?} depth={:?}: {error:?}", info.depth_attachment_format);
    }).map(Some)
}

fn pass(
    colors: &[vk::Format],
    depth: vk::Format,
    read_only_depth: bool,
) -> VkResult<crate::render_pass::RenderPass> {
    if (colors.is_empty() && depth == vk::Format::UNDEFINED)
        || colors.len() > 8
        || colors
            .iter()
            .any(|&f| crate::images::texture_format(f).is_none() || f == vk::Format::D32_SFLOAT)
        || !matches!(depth, vk::Format::UNDEFINED | vk::Format::D32_SFLOAT)
    {
        return Err(UNSUPPORTED);
    }
    let mut attachments: Vec<_> = colors
        .iter()
        .map(|&format| {
            vk::AttachmentDescription::default()
                .format(format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE)
                .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        })
        .collect();
    let depth = if depth == vk::Format::UNDEFINED {
        None
    } else {
        attachments.push(
            vk::AttachmentDescription::default()
                .format(depth)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE)
                .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        );
        Some(colors.len())
    };
    Ok(crate::render_pass::RenderPass {
        dynamic: true,
        attachments,
        subpasses: vec![crate::render_pass::Subpass {
            colors: (0..colors.len()).collect(),
            inputs: vec![],
            depth,
            read_only_depth,
        }],
    })
}

pub(super) fn begin(
    rt: &mut Runtime,
    rec: &mut ResolvedRecording,
    rendering: &Rendering,
) -> VkResult<()> {
    if rec.render.is_some() {
        return Err(INVALID);
    }
    let mut formats = Vec::new();
    let mut views = Vec::new();
    let mut clears = Vec::new();
    let mut descriptions = Vec::new();
    let mut dimensions = (u32::MAX, u32::MAX);
    let mut depth_format = vk::Format::UNDEFINED;
    let mut read_only = false;
    for (index, a) in rendering.colors.iter().chain(&rendering.depth).enumerate() {
        let view = rt.resources.views.get(&a.view).ok_or(INVALID)?;
        let image = rt.resources.images.get(&view.image).ok_or(INVALID)?;
        let depth = index == rendering.colors.len();
        let layout = crate::render_pass::normalize_depth_layout(match a.layout {
            vk::ImageLayout::ATTACHMENT_OPTIMAL if depth => {
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            }
            vk::ImageLayout::ATTACHMENT_OPTIMAL => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::READ_ONLY_OPTIMAL if depth => {
                vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
            }
            other => other,
        });
        if !image.usable()
            || !image.usage.contains(if depth {
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
            } else {
                vk::ImageUsageFlags::COLOR_ATTACHMENT
            })
            || Some(view.desc.format()) != crate::images::texture_format(image.format)
            || view.components != [0, 1, 2, 3]
            || view.desc.base_mip_level() != 0
            || view.desc.mip_level_count() != 1
            || view.desc.base_array_layer() != 0
            || view.desc.array_layer_count() != 1
            || (depth
                && (image.format != vk::Format::D32_SFLOAT
                    || layout == vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL))
            || !matches!(
                layout,
                vk::ImageLayout::GENERAL
                    | vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                    | vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                    | vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
            )
        {
            eprintln!(
                "[SGFX Vulkan] rendering attachment rejected: index={index} depth={depth} image={:?} format={:?} extent={:?} usage={:?} layout={layout:?} view={:?} components={:?}",
                view.image, image.format, image.extent, image.usage, view.desc, view.components
            );
            return Err(UNSUPPORTED);
        }
        if depth {
            depth_format = image.format;
            read_only = layout == vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        } else {
            formats.push(image.format);
            if layout == vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
                || layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            {
                return Err(UNSUPPORTED);
            }
        }
        if read_only && a.load == vk::AttachmentLoadOp::CLEAR {
            return Err(INVALID);
        }
        dimensions.0 = dimensions.0.min(image.extent.width);
        dimensions.1 = dimensions.1.min(image.extent.height);
        views.push(a.view);
        clears.push(a.clear);
        descriptions.push(
            vk::AttachmentDescription::default()
                .format(image.format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(a.load)
                .store_op(a.store)
                .initial_layout(layout)
                .final_layout(layout),
        );
    }
    if formats.len() > rt.capabilities.limits().max_color_attachments as usize
        || (read_only && !rt.capabilities.supports_typed_texture_views())
    {
        return Err(UNSUPPORTED);
    }
    let mut pass = pass(&formats, depth_format, read_only)?;
    pass.attachments = descriptions;
    let x = u32::try_from(rendering.area.offset.x).map_err(|_| INVALID)?;
    let y = u32::try_from(rendering.area.offset.y).map_err(|_| INVALID)?;
    let area = ir::PixelRect::new(
        x,
        y,
        rendering.area.extent.width,
        rendering.area.extent.height,
    )
    .map_err(|_| INVALID)?;
    if x.checked_add(area.width()).is_none_or(|x| x > dimensions.0)
        || y.checked_add(area.height())
            .is_none_or(|y| y > dimensions.1)
    {
        return Err(INVALID);
    }
    render_passes::begin_owned(rt, rec, &pass, &views, dimensions, area, &clears)
}

unsafe extern "system" fn begin_rendering(
    command: vk::CommandBuffer,
    info: *const vk::RenderingInfo<'_>,
) {
    let result = (|| {
        let d = driver(command.as_raw(), Kind::Command)?;
        if !d.dynamic_rendering_enabled {
            return Err(UNSUPPORTED);
        }
        let info = info.as_ref().ok_or(INVALID)?;
        if info.s_type != vk::StructureType::RENDERING_INFO
            || !info.p_next.is_null()
            || !info.flags.is_empty()
            || info.view_mask != 0
            || info.layer_count != 1
            || !info.p_stencil_attachment.is_null()
        {
            return Err(UNSUPPORTED);
        }
        let colors = slice(info.p_color_attachments, info.color_attachment_count)?
            .iter()
            .map(|a| attachment(a))
            .collect::<VkResult<Vec<_>>>()?;
        let depth = info
            .p_depth_attachment
            .as_ref()
            .map(|a| attachment(a))
            .transpose()?;
        Ok(Rendering {
            area: info.render_area,
            colors,
            depth,
        })
    })();
    match result {
        Ok(info) => record(command, RecordedCommand::BeginRendering(info)),
        Err(e) => record_error(command, e),
    }
}
unsafe extern "system" fn end_rendering(command: vk::CommandBuffer) {
    record(command, RecordedCommand::EndRendering)
}
pub(super) fn lookup(name: &CStr) -> vk::PFN_vkVoidFunction {
    match name.to_bytes() {
        b"vkCmdBeginRendering" | b"vkCmdBeginRenderingKHR" => Some(unsafe {
            std::mem::transmute::<vk::PFN_vkCmdBeginRendering, unsafe extern "system" fn()>(
                begin_rendering,
            )
        }),
        b"vkCmdEndRendering" | b"vkCmdEndRenderingKHR" => Some(unsafe {
            std::mem::transmute::<vk::PFN_vkCmdEndRendering, unsafe extern "system" fn()>(
                end_rendering,
            )
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dynamic_formats_match_independently_of_depth_attachment_access() {
        let a = pass(&[vk::Format::R8G8B8A8_UNORM], vk::Format::D32_SFLOAT, false).unwrap();
        let b = pass(&[vk::Format::R8G8B8A8_UNORM], vk::Format::D32_SFLOAT, true).unwrap();
        assert!(a.compatible(&b));
        let mut legacy = a.clone();
        legacy.dynamic = false;
        assert!(!a.compatible(&legacy));
        assert!(!a.compatible(
            &pass(&[vk::Format::B8G8R8A8_UNORM], vk::Format::D32_SFLOAT, false).unwrap()
        ));
        assert!(pass(&[], vk::Format::UNDEFINED, false).is_err());
    }
}
