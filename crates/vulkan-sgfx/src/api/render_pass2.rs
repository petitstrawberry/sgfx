//! RenderPass2 lowers the supported non-multiview subset to the same retained IR.
use super::*;
const UNSUPPORTED: vk::Result = vk::Result::ERROR_FEATURE_NOT_PRESENT;
unsafe fn trace_rejected(info: &vk::RenderPassCreateInfo2<'_>, error: vk::Result) {
    if std::env::var_os("SGFX_VULKAN_TRACE").is_none() {
        return;
    }
    eprintln!("[SGFX Vulkan] renderpass2 rejected: {error:?} info={info:?}");
    let next_type = |next: *const std::ffi::c_void| {
        next.cast::<vk::BaseInStructure<'_>>()
            .as_ref()
            .map(|base| base.s_type)
    };
    eprintln!(
        "[SGFX Vulkan] renderpass2 pnext={:?}",
        next_type(info.p_next)
    );
    if let Ok(attachments) = slice(info.p_attachments, info.attachment_count.min(16)) {
        for (index, a) in attachments.iter().enumerate() {
            eprintln!(
                "[SGFX Vulkan] renderpass2 attachment[{index}]={a:?} pnext={:?}",
                next_type(a.p_next)
            );
        }
    }
    if let Ok(subpasses) = slice(info.p_subpasses, info.subpass_count.min(16)) {
        for (index, s) in subpasses.iter().enumerate() {
            eprintln!(
                "[SGFX Vulkan] renderpass2 subpass[{index}]={s:?} pnext={:?}",
                next_type(s.p_next)
            );
            if let Ok(refs) = slice(s.p_color_attachments, s.color_attachment_count.min(16)) {
                eprintln!("[SGFX Vulkan] renderpass2 colors[{index}]={refs:?}");
            }
            if let Ok(refs) = slice(s.p_input_attachments, s.input_attachment_count.min(16)) {
                eprintln!("[SGFX Vulkan] renderpass2 inputs[{index}]={refs:?}");
            }
            if let Some(depth) = s.p_depth_stencil_attachment.as_ref() {
                eprintln!("[SGFX Vulkan] renderpass2 depth[{index}]={depth:?}");
            }
        }
    }
    if let Ok(dependencies) = slice(info.p_dependencies, info.dependency_count.min(64)) {
        for (index, d) in dependencies.iter().enumerate() {
            eprintln!(
                "[SGFX Vulkan] renderpass2 dependency[{index}]={d:?} pnext={:?}",
                next_type(d.p_next)
            );
        }
    }
}
unsafe fn reference(
    r: &vk::AttachmentReference2<'_>,
    input: bool,
) -> VkResult<vk::AttachmentReference> {
    if r.s_type != vk::StructureType::ATTACHMENT_REFERENCE_2
        || !r.p_next.is_null()
        // Vulkan ignores aspectMask for every reference except input
        // attachments. Mesa does not initialize it for depth attachments.
        || (input && !(vk::ImageAspectFlags::COLOR | vk::ImageAspectFlags::DEPTH).contains(r.aspect_mask))
    {
        return Err(UNSUPPORTED);
    }
    Ok(vk::AttachmentReference {
        attachment: r.attachment,
        layout: r.layout,
    })
}
unsafe fn parse(info: &vk::RenderPassCreateInfo2<'_>) -> VkResult<crate::render_pass::RenderPass> {
    if info.s_type != vk::StructureType::RENDER_PASS_CREATE_INFO_2
        || !info.p_next.is_null()
        || info.correlated_view_mask_count != 0
    {
        return Err(UNSUPPORTED);
    }
    let attachments = slice(info.p_attachments, info.attachment_count)?
        .iter()
        .map(|a| {
            if a.s_type != vk::StructureType::ATTACHMENT_DESCRIPTION_2 || !a.p_next.is_null() {
                return Err(UNSUPPORTED);
            }
            Ok(vk::AttachmentDescription {
                flags: a.flags,
                format: a.format,
                samples: a.samples,
                load_op: a.load_op,
                store_op: a.store_op,
                stencil_load_op: a.stencil_load_op,
                stencil_store_op: a.stencil_store_op,
                initial_layout: a.initial_layout,
                final_layout: a.final_layout,
            })
        })
        .collect::<VkResult<Vec<_>>>()?;
    let sources = slice(info.p_subpasses, info.subpass_count)?;
    let mut refs = Vec::new();
    for s in sources {
        if s.s_type != vk::StructureType::SUBPASS_DESCRIPTION_2
            || !s.p_next.is_null()
            || s.view_mask != 0
        {
            return Err(UNSUPPORTED);
        }
        let colors = slice(s.p_color_attachments, s.color_attachment_count)?
            .iter()
            .map(|r| reference(r, false))
            .collect::<VkResult<Vec<_>>>()?;
        let inputs = slice(s.p_input_attachments, s.input_attachment_count)?
            .iter()
            .map(|r| reference(r, true))
            .collect::<VkResult<Vec<_>>>()?;
        let resolves = if s.p_resolve_attachments.is_null() {
            Vec::new()
        } else {
            slice(s.p_resolve_attachments, s.color_attachment_count)?
                .iter()
                .map(|r| reference(r, false))
                .collect::<VkResult<Vec<_>>>()?
        };
        let depth = s
            .p_depth_stencil_attachment
            .as_ref()
            .map(|r| reference(r, false))
            .transpose()?;
        let preserve = slice(s.p_preserve_attachments, s.preserve_attachment_count)?.to_vec();
        refs.push((colors, inputs, resolves, depth, preserve));
    }
    let subpasses = sources
        .iter()
        .zip(&refs)
        .map(|(s, (colors, inputs, resolves, depth, preserve))| {
            let mut old = vk::SubpassDescription::default()
                .flags(s.flags)
                .pipeline_bind_point(s.pipeline_bind_point)
                .color_attachments(colors)
                .input_attachments(inputs)
                .preserve_attachments(preserve);
            if !resolves.is_empty() {
                old.p_resolve_attachments = resolves.as_ptr();
            }
            if let Some(depth) = depth {
                old.p_depth_stencil_attachment = depth;
            }
            old
        })
        .collect::<Vec<_>>();
    let dependencies = slice(info.p_dependencies, info.dependency_count)?
        .iter()
        .map(|d| {
            if d.s_type != vk::StructureType::SUBPASS_DEPENDENCY_2
                || !d.p_next.is_null()
                || d.view_offset != 0
            {
                return Err(UNSUPPORTED);
            }
            Ok(vk::SubpassDependency {
                src_subpass: d.src_subpass,
                dst_subpass: d.dst_subpass,
                src_stage_mask: d.src_stage_mask,
                dst_stage_mask: d.dst_stage_mask,
                src_access_mask: d.src_access_mask,
                dst_access_mask: d.dst_access_mask,
                dependency_flags: d.dependency_flags,
            })
        })
        .collect::<VkResult<Vec<_>>>()?;
    crate::render_pass::parse(
        &vk::RenderPassCreateInfo::default()
            .flags(info.flags)
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies),
    )
}
unsafe extern "system" fn create(
    device: vk::Device,
    info: *const vk::RenderPassCreateInfo2<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::RenderPass,
) -> vk::Result {
    if out.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    *out = vk::RenderPass::null();
    if !allocator.is_null() {
        return UNSUPPORTED;
    }
    status((|| {
        let info = info
            .as_ref()
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let pass = parse(info).inspect_err(|error| trace_rejected(info, *error))?;
        let handle = with_device(device, move |rt| {
            crate::images::insert_render_pass(rt, pass)
        })?;
        *out = handle;
        Ok(())
    })())
}
unsafe extern "system" fn begin(
    command: vk::CommandBuffer,
    info: *const vk::RenderPassBeginInfo<'_>,
    begin: *const vk::SubpassBeginInfo<'_>,
) {
    let Some(begin) = begin.as_ref() else {
        record_error(command, vk::Result::ERROR_INITIALIZATION_FAILED);
        return;
    };
    if begin.s_type != vk::StructureType::SUBPASS_BEGIN_INFO || !begin.p_next.is_null() {
        record_error(command, UNSUPPORTED);
        return;
    }
    cmd_begin_render_pass(command, info, begin.contents);
}
unsafe extern "system" fn next(
    command: vk::CommandBuffer,
    begin: *const vk::SubpassBeginInfo<'_>,
    end: *const vk::SubpassEndInfo<'_>,
) {
    let (Some(begin), Some(end)) = (begin.as_ref(), end.as_ref()) else {
        record_error(command, vk::Result::ERROR_INITIALIZATION_FAILED);
        return;
    };
    if begin.s_type != vk::StructureType::SUBPASS_BEGIN_INFO
        || !begin.p_next.is_null()
        || end.s_type != vk::StructureType::SUBPASS_END_INFO
        || !end.p_next.is_null()
    {
        record_error(command, UNSUPPORTED);
        return;
    }
    cmd_next_subpass(command, begin.contents);
}
unsafe extern "system" fn end(command: vk::CommandBuffer, info: *const vk::SubpassEndInfo<'_>) {
    let Some(info) = info.as_ref() else {
        record_error(command, vk::Result::ERROR_INITIALIZATION_FAILED);
        return;
    };
    if info.s_type != vk::StructureType::SUBPASS_END_INFO || !info.p_next.is_null() {
        record_error(command, UNSUPPORTED);
        return;
    }
    cmd_end_render_pass(command);
}
pub(super) fn lookup(name: &CStr) -> vk::PFN_vkVoidFunction {
    macro_rules! entry {
        ($f:ident,$ty:ty) => {{
            let f: $ty = $f;
            Some(unsafe { std::mem::transmute::<$ty, unsafe extern "system" fn()>(f) })
        }};
    }
    match name.to_bytes() {
        b"vkCreateRenderPass2KHR" => entry!(create, vk::PFN_vkCreateRenderPass2),
        b"vkCmdBeginRenderPass2KHR" => entry!(begin, vk::PFN_vkCmdBeginRenderPass2),
        b"vkCmdNextSubpass2KHR" => entry!(next, vk::PFN_vkCmdNextSubpass2),
        b"vkCmdEndRenderPass2KHR" => entry!(end, vk::PFN_vkCmdEndRenderPass2),
        _ => None,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn zink_depth_only_attachment_ignores_non_input_aspect_and_stencil_ops() {
        unsafe {
            let mut attachments = [
                vk::AttachmentDescription2::default()
                    .format(vk::Format::B8G8R8A8_UNORM)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::LOAD)
                    .stencil_store_op(vk::AttachmentStoreOp::STORE)
                    .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                vk::AttachmentDescription2::default()
                    .format(vk::Format::D32_SFLOAT)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::LOAD)
                    .stencil_store_op(vk::AttachmentStoreOp::STORE)
                    .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
            ];
            let colors = [vk::AttachmentReference2::default()
                .attachment(0)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .aspect_mask(vk::ImageAspectFlags::STENCIL)];
            let mut depth = vk::AttachmentReference2::default()
                .attachment(1)
                .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .aspect_mask(vk::ImageAspectFlags::from_raw(u32::MAX));
            let subpasses = [vk::SubpassDescription2::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&colors)
                .depth_stencil_attachment(&depth)];
            let stage = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;
            let access = vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
            let dependencies = [
                vk::SubpassDependency2::default()
                    .src_subpass(vk::SUBPASS_EXTERNAL)
                    .dst_subpass(0)
                    .src_stage_mask(stage)
                    .dst_stage_mask(stage)
                    .dst_access_mask(access),
                vk::SubpassDependency2::default()
                    .src_subpass(0)
                    .dst_subpass(vk::SUBPASS_EXTERNAL)
                    .src_stage_mask(stage)
                    .dst_stage_mask(vk::PipelineStageFlags::BOTTOM_OF_PIPE)
                    .src_access_mask(access),
            ];
            let pass = parse(
                &vk::RenderPassCreateInfo2::default()
                    .attachments(&attachments)
                    .subpasses(&subpasses)
                    .dependencies(&dependencies),
            )
            .unwrap();
            assert_eq!(pass.subpasses[0].colors, [0]);
            assert_eq!(pass.subpasses[0].depth, Some(1));
            assert!(!pass.subpasses[0].read_only_depth);
            assert_eq!(pass.attachments[1].load_op, vk::AttachmentLoadOp::CLEAR);
            assert_eq!(pass.attachments[1].store_op, vk::AttachmentStoreOp::STORE);
            assert_eq!(
                pass.attachments[1].stencil_store_op,
                vk::AttachmentStoreOp::DONT_CARE
            );
            // Actual OpenTTD/Zink r6: depth is unused and read-only, while its
            // absent stencil aspect still has STORE and a garbage aspectMask.
            attachments[1].load_op = vk::AttachmentLoadOp::DONT_CARE;
            attachments[1].initial_layout = vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            attachments[1].final_layout = vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            depth.layout = vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            let subpasses = [vk::SubpassDescription2::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&colors)
                .depth_stencil_attachment(&depth)];
            let pass = parse(
                &vk::RenderPassCreateInfo2::default()
                    .attachments(&attachments)
                    .subpasses(&subpasses)
                    .dependencies(&dependencies),
            )
            .unwrap();
            assert!(pass.subpasses[0].read_only_depth);
            assert_eq!(pass.attachments[1].load_op, vk::AttachmentLoadOp::DONT_CARE);
        }
    }

    #[test]
    fn input_attachment_aspect_is_still_checked() {
        unsafe {
            let reference = vk::AttachmentReference2::default()
                .attachment(0)
                .layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .aspect_mask(vk::ImageAspectFlags::STENCIL);
            assert!(super::reference(&reference, true).is_err());
            assert!(super::reference(&reference, false).is_ok());
        }
    }

    #[test]
    fn lowering_retains_layouts_and_rejects_multiview() {
        unsafe {
            let attachment = [vk::AttachmentDescription2::default()
                .format(vk::Format::R8G8B8A8_UNORM)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)];
            let color = [vk::AttachmentReference2::default()
                .attachment(0)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
            let mut subpasses = [vk::SubpassDescription2::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&color)];
            let parsed = parse(
                &vk::RenderPassCreateInfo2::default()
                    .attachments(&attachment)
                    .subpasses(&subpasses),
            )
            .unwrap();
            assert_eq!(
                parsed.attachments[0].final_layout,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL
            );
            assert_eq!(parsed.subpasses[0].colors, [0]);
            subpasses[0].view_mask = 1;
            assert!(
                parse(
                    &vk::RenderPassCreateInfo2::default()
                        .attachments(&attachment)
                        .subpasses(&subpasses)
                )
                .is_err()
            );
        }
    }
}
