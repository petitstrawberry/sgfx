//! Vulkan attachment/subpass descriptions retained independently of application pointers.
//! Backends without native subpasses execute ordered GPU passes and preserve intermediates.
use ash::vk;

const UNSUPPORTED: vk::Result = vk::Result::ERROR_FEATURE_NOT_PRESENT;

#[derive(Clone)]
pub(crate) struct RenderPass {
    pub attachments: Vec<vk::AttachmentDescription>,
    pub subpasses: Vec<Subpass>,
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct Subpass {
    pub colors: Vec<usize>,
    pub inputs: Vec<Option<usize>>,
    pub depth: Option<usize>,
    pub read_only_depth: bool,
}

impl RenderPass {
    pub fn compatible(&self, other: &Self) -> bool {
        self.attachments
            .iter()
            .map(|a| (a.format, a.samples))
            .eq(other.attachments.iter().map(|a| (a.format, a.samples)))
            && self.subpasses == other.subpasses
    }

    pub fn input_depths(&self, subpass: usize) -> Vec<Option<bool>> {
        self.subpasses[subpass]
            .inputs
            .iter()
            .map(|i| i.map(|i| self.attachments[i].format == vk::Format::D32_SFLOAT))
            .collect()
    }
}

pub(crate) unsafe fn parse(info: &vk::RenderPassCreateInfo<'_>) -> Result<RenderPass, vk::Result> {
    if !info.flags.is_empty()
        || info.s_type != vk::StructureType::RENDER_PASS_CREATE_INFO
        || !(1..=8).contains(&info.attachment_count)
        || info.p_attachments.is_null()
        || !(1..=8).contains(&info.subpass_count)
        || info.p_subpasses.is_null()
        || info.dependency_count > 64
        || (info.dependency_count != 0 && info.p_dependencies.is_null())
    {
        return Err(UNSUPPORTED);
    }
    let attachments =
        std::slice::from_raw_parts(info.p_attachments, info.attachment_count as usize).to_vec();
    let mut attachments = attachments;
    for a in &mut attachments {
        a.initial_layout = normalize_depth_layout(a.initial_layout);
        a.final_layout = normalize_depth_layout(a.final_layout);
        // Stencil operations affect only a stencil aspect. All currently
        // supported formats are color or depth-only, and Zink legitimately
        // supplies STORE even for D32. Canonicalize these ignored operations;
        // this does not add stencil-format or stencil-operation support.
        if !crate::images::image_aspect(a.format).contains(vk::ImageAspectFlags::STENCIL) {
            a.stencil_load_op = vk::AttachmentLoadOp::DONT_CARE;
            a.stencil_store_op = vk::AttachmentStoreOp::DONT_CARE;
        }
        let depth = a.format == vk::Format::D32_SFLOAT;
        let valid_layout = |layout| match layout {
            vk::ImageLayout::GENERAL | vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => true,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            | vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL => depth,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
            | vk::ImageLayout::TRANSFER_SRC_OPTIMAL
            | vk::ImageLayout::TRANSFER_DST_OPTIMAL
            | vk::ImageLayout::PRESENT_SRC_KHR => !depth,
            _ => false,
        };
        if crate::images::texture_format(a.format).is_none()
            || !a.flags.is_empty()
            || a.samples != vk::SampleCountFlags::TYPE_1
            || !matches!(
                a.load_op,
                vk::AttachmentLoadOp::CLEAR
                    | vk::AttachmentLoadOp::LOAD
                    | vk::AttachmentLoadOp::DONT_CARE
            )
            || !matches!(
                a.store_op,
                vk::AttachmentStoreOp::STORE | vk::AttachmentStoreOp::DONT_CARE
            )
            || a.stencil_load_op != vk::AttachmentLoadOp::DONT_CARE
            || a.stencil_store_op != vk::AttachmentStoreOp::DONT_CARE
            || (a.initial_layout != vk::ImageLayout::UNDEFINED && !valid_layout(a.initial_layout))
            || (a.load_op == vk::AttachmentLoadOp::LOAD
                && a.initial_layout == vk::ImageLayout::UNDEFINED)
            || !valid_layout(a.final_layout)
        {
            return Err(UNSUPPORTED);
        }
    }
    validate_extensions(info, &attachments)?;
    let mut subpasses = Vec::new();
    let mut first_use = vec![true; attachments.len()];
    for subpass in std::slice::from_raw_parts(info.p_subpasses, info.subpass_count as usize) {
        if !subpass.flags.is_empty()
            || subpass.pipeline_bind_point != vk::PipelineBindPoint::GRAPHICS
            || !(1..=8).contains(&subpass.color_attachment_count)
            || subpass.p_color_attachments.is_null()
            || subpass.input_attachment_count > 8
            || (subpass.input_attachment_count != 0 && subpass.p_input_attachments.is_null())
            || !subpass.p_resolve_attachments.is_null()
            || subpass.preserve_attachment_count > info.attachment_count
            || (subpass.preserve_attachment_count != 0 && subpass.p_preserve_attachments.is_null())
        {
            return Err(UNSUPPORTED);
        }
        let mut colors = Vec::new();
        for reference in std::slice::from_raw_parts(
            subpass.p_color_attachments,
            subpass.color_attachment_count as usize,
        ) {
            let index = reference.attachment as usize;
            let a = attachments.get(index).ok_or(UNSUPPORTED)?;
            if a.format == vk::Format::D32_SFLOAT
                || colors.contains(&index)
                || !matches!(
                    reference.layout,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL | vk::ImageLayout::GENERAL
                )
            {
                return Err(UNSUPPORTED);
            }
            colors.push(index);
        }
        let mut read_only_depth = false;
        let depth = subpass
            .p_depth_stencil_attachment
            .as_ref()
            .filter(|r| r.attachment != vk::ATTACHMENT_UNUSED)
            .map(|reference| {
                let mut reference = *reference;
                reference.layout = normalize_depth_layout(reference.layout);
                let index = reference.attachment as usize;
                if attachments
                    .get(index)
                    .is_none_or(|a| a.format != vk::Format::D32_SFLOAT)
                    || !matches!(
                        reference.layout,
                        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                            | vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
                            | vk::ImageLayout::GENERAL
                    )
                {
                    return Err(UNSUPPORTED);
                }
                read_only_depth =
                    reference.layout == vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL;
                if read_only_depth
                    && first_use[index]
                    && attachments[index].load_op == vk::AttachmentLoadOp::CLEAR
                {
                    return Err(UNSUPPORTED);
                }
                Ok(index)
            })
            .transpose()?;
        let mut inputs = Vec::new();
        for i in 0..subpass.input_attachment_count as usize {
            let mut reference = *subpass.p_input_attachments.add(i);
            reference.layout = normalize_depth_layout(reference.layout);
            if reference.attachment == vk::ATTACHMENT_UNUSED {
                inputs.push(None);
                continue;
            }
            let index = reference.attachment as usize;
            let a = attachments.get(index).ok_or(UNSUPPORTED)?;
            if colors.contains(&index)
                || (depth == Some(index) && !read_only_depth)
                || !matches!(
                    reference.layout,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                        | vk::ImageLayout::GENERAL
                        | vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
                )
                || (reference.layout == vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
                    && a.format != vk::Format::D32_SFLOAT)
                || (first_use[index] && a.load_op == vk::AttachmentLoadOp::CLEAR)
            {
                return Err(UNSUPPORTED);
            }
            inputs.push(Some(index));
        }
        for i in 0..subpass.preserve_attachment_count as usize {
            let index = *subpass.p_preserve_attachments.add(i) as usize;
            if index >= attachments.len()
                || colors.contains(&index)
                || depth == Some(index)
                || inputs.contains(&Some(index))
            {
                return Err(UNSUPPORTED);
            }
        }
        for &i in colors
            .iter()
            .chain(depth.iter())
            .chain(inputs.iter().flatten())
        {
            first_use[i] = false;
        }
        subpasses.push(Subpass {
            colors,
            inputs,
            depth,
            read_only_depth,
        });
    }
    let stages = vk::PipelineStageFlags::TOP_OF_PIPE
        | vk::PipelineStageFlags::BOTTOM_OF_PIPE
        | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
        | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
        | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
        | vk::PipelineStageFlags::TRANSFER
        | vk::PipelineStageFlags::VERTEX_SHADER
        | vk::PipelineStageFlags::FRAGMENT_SHADER
        | vk::PipelineStageFlags::ALL_GRAPHICS
        | vk::PipelineStageFlags::ALL_COMMANDS;
    let accesses = vk::AccessFlags::COLOR_ATTACHMENT_READ
        | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
        | vk::AccessFlags::INPUT_ATTACHMENT_READ
        | vk::AccessFlags::SHADER_READ
        | vk::AccessFlags::TRANSFER_READ
        | vk::AccessFlags::TRANSFER_WRITE
        | vk::AccessFlags::MEMORY_READ
        | vk::AccessFlags::MEMORY_WRITE;
    for i in 0..info.dependency_count as usize {
        let d = &*info.p_dependencies.add(i);
        let external = vk::SUBPASS_EXTERNAL;
        if (d.src_subpass != external && d.src_subpass >= info.subpass_count)
            || (d.dst_subpass != external && d.dst_subpass >= info.subpass_count)
            || (d.src_subpass == external && d.dst_subpass == external)
            || (d.src_subpass != external
                && d.dst_subpass != external
                && d.src_subpass >= d.dst_subpass)
            || !stages.contains(d.src_stage_mask | d.dst_stage_mask)
            || !accesses.contains(d.src_access_mask | d.dst_access_mask)
            || !vk::DependencyFlags::BY_REGION.contains(d.dependency_flags)
        {
            return Err(UNSUPPORTED);
        }
    }
    Ok(RenderPass {
        attachments,
        subpasses,
    })
}

// D32 is the only supported depth format: there is no stencil aspect to carry.
pub(crate) fn normalize_depth_layout(layout: vk::ImageLayout) -> vk::ImageLayout {
    match layout {
        vk::ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL => {
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        }
        vk::ImageLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL => {
            vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
        }
        _ => layout,
    }
}
unsafe fn validate_extensions(
    info: &vk::RenderPassCreateInfo<'_>,
    attachments: &[vk::AttachmentDescription],
) -> Result<(), vk::Result> {
    let mut node = info.p_next.cast::<vk::BaseInStructure<'_>>();
    for _ in 0..32 {
        let Some(base) = node.as_ref() else {
            return Ok(());
        };
        match base.s_type {
            vk::StructureType::RENDER_PASS_MULTIVIEW_CREATE_INFO => {
                let mv = &*node.cast::<vk::RenderPassMultiviewCreateInfo<'_>>();
                if !(mv.subpass_count == 0 || mv.subpass_count == info.subpass_count)
                    || !(mv.dependency_count == 0 || mv.dependency_count == info.dependency_count)
                    || mv.correlation_mask_count != 0
                {
                    return Err(UNSUPPORTED);
                }
                if crate::api::slice(mv.p_view_masks, mv.subpass_count)?
                    .iter()
                    .any(|&m| m != 0)
                    || crate::api::slice(mv.p_view_offsets, mv.dependency_count)?
                        .iter()
                        .any(|&o| o != 0)
                {
                    return Err(UNSUPPORTED);
                }
            }
            vk::StructureType::RENDER_PASS_INPUT_ATTACHMENT_ASPECT_CREATE_INFO => {
                let aspects = &*node.cast::<vk::RenderPassInputAttachmentAspectCreateInfo<'_>>();
                for r in
                    crate::api::slice(aspects.p_aspect_references, aspects.aspect_reference_count)?
                {
                    if r.subpass >= info.subpass_count {
                        return Err(UNSUPPORTED);
                    }
                    let subpass = &*info.p_subpasses.add(r.subpass as usize);
                    let inputs = crate::api::slice(
                        subpass.p_input_attachments,
                        subpass.input_attachment_count,
                    )?;
                    let reference = inputs
                        .get(r.input_attachment_index as usize)
                        .ok_or(UNSUPPORTED)?;
                    if reference.attachment != vk::ATTACHMENT_UNUSED {
                        let attachment = attachments
                            .get(reference.attachment as usize)
                            .ok_or(UNSUPPORTED)?;
                        if r.aspect_mask != crate::images::image_aspect(attachment.format) {
                            return Err(UNSUPPORTED);
                        }
                    }
                }
            }
            _ => return Err(UNSUPPORTED),
        }
        node = base.p_next;
    }
    Err(UNSUPPORTED)
}
