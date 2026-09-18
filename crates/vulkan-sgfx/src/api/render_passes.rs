//! Resolve a Vulkan render pass into ordered SGFX passes without CPU readback.
use super::*;
const INVALID: vk::Result = vk::Result::ERROR_INITIALIZATION_FAILED;
const UNSUPPORTED: vk::Result = vk::Result::ERROR_FEATURE_NOT_PRESENT;

#[derive(Clone)]
pub(super) struct ActiveRenderPass {
    pub pass: crate::render_pass::RenderPass,
    pub framebuffer: vk::Framebuffer,
    pub subpass: usize,
    clears: Vec<vk::ClearValue>,
    images: Vec<(vk::Image, ir::TextureId)>,
    used: Vec<bool>,
    accesses: Vec<Option<ir::TextureAccess>>,
    area: ir::PixelRect,
}

pub(super) fn begin(
    rt: &Runtime,
    rec: &mut ResolvedRecording,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    area: vk::Rect2D,
    clears: &[vk::ClearValue],
) -> VkResult<()> {
    let pass = rt
        .resources
        .render_passes
        .get(&render_pass)
        .ok_or(INVALID)?;
    let fb = rt.resources.framebuffers.get(&framebuffer).ok_or(INVALID)?;
    if !pass.compatible(&fb.render_pass)
        || area.offset.x != 0
        || area.offset.y != 0
        || area.extent.width != fb.width
        || area.extent.height != fb.height
    {
        return Err(UNSUPPORTED);
    }
    let images = fb
        .attachments
        .iter()
        .map(|view| {
            let view = rt.resources.views.get(view).ok_or(INVALID)?;
            let image = rt.resources.images.get(&view.image).ok_or(INVALID)?;
            if !image.usable() {
                return Err(INVALID);
            }
            Ok((view.image, image.id))
        })
        .collect::<VkResult<Vec<_>>>()?;
    rec.used_images
        .extend(images.iter().map(|(image, _)| *image));
    rec.used_framebuffers.push(framebuffer);
    rec.used_render_passes.push(render_pass);
    rec.render = Some((fb.width, fb.height));
    let mut active = ActiveRenderPass {
        pass: pass.clone(),
        framebuffer,
        subpass: 0,
        clears: clears.to_vec(),
        images,
        used: vec![false; pass.attachments.len()],
        accesses: vec![None; pass.attachments.len()],
        area: ir::PixelRect::new(0, 0, fb.width, fb.height).map_err(crate::resources::failure)?,
    };
    active.emit(rec)?;
    rec.active_render_pass = Some(active);
    Ok(())
}

pub(super) fn next(rec: &mut ResolvedRecording) -> VkResult<()> {
    let mut active = rec.active_render_pass.take().ok_or(INVALID)?;
    if active.subpass + 1 >= active.pass.subpasses.len() {
        return Err(INVALID);
    }
    rec.ops.push(ir::OwnedCommand::EndRenderPass);
    active.subpass += 1;
    active.emit(rec)?;
    rec.active_render_pass = Some(active);
    Ok(())
}

pub(super) fn end(rec: &mut ResolvedRecording) -> VkResult<()> {
    let active = rec.active_render_pass.take().ok_or(INVALID)?;
    if active.subpass + 1 != active.pass.subpasses.len() {
        return Err(INVALID);
    }
    rec.ops.push(ir::OwnedCommand::EndRenderPass);
    for (index, before) in active.accesses.iter().enumerate() {
        if let (Some(before), Ok(after)) = (
            before,
            texture_access(active.pass.attachments[index].final_layout),
        ) && *before != after
        {
            rec.ops.push(ir::OwnedCommand::ResourceBarrier(
                ir::OwnedResourceBarrier::Texture {
                    texture: active.images[index].1,
                    before: *before,
                    after,
                },
            ));
        }
    }
    rec.render = None;
    Ok(())
}

impl ActiveRenderPass {
    fn access(&mut self, index: usize, after: ir::TextureAccess, rec: &mut ResolvedRecording) {
        let before = self.accesses[index]
            .or_else(|| texture_access(self.pass.attachments[index].initial_layout).ok());
        if let Some(before) = before
            && before != after
        {
            rec.ops.push(ir::OwnedCommand::ResourceBarrier(
                ir::OwnedResourceBarrier::Texture {
                    texture: self.images[index].1,
                    before,
                    after,
                },
            ));
        }
        self.accesses[index] = Some(after);
    }
    fn load(&self, index: usize) -> vk::AttachmentLoadOp {
        if self.used[index] {
            vk::AttachmentLoadOp::LOAD
        } else {
            self.pass.attachments[index].load_op
        }
    }
    fn store(&self, index: usize) -> ir::StoreOp {
        let later = self.pass.subpasses[self.subpass + 1..].iter().any(|p| {
            p.colors.contains(&index) || p.depth == Some(index) || p.inputs.contains(&Some(index))
        });
        if later || self.pass.attachments[index].store_op == vk::AttachmentStoreOp::STORE {
            ir::StoreOp::Store
        } else {
            ir::StoreOp::DontCare
        }
    }
    fn emit(&mut self, rec: &mut ResolvedRecording) -> VkResult<()> {
        let subpass = self.pass.subpasses[self.subpass].clone();
        let mut colors = Vec::new();
        for &index in &subpass.colors {
            let load = match self.load(index) {
                vk::AttachmentLoadOp::LOAD => ir::LoadOp::Load,
                vk::AttachmentLoadOp::DONT_CARE => ir::LoadOp::DontCare,
                vk::AttachmentLoadOp::CLEAR => {
                    let color = unsafe { self.clears.get(index).ok_or(INVALID)?.color.float32 };
                    ir::LoadOp::Clear(
                        ir::Color::rgba(color[0], color[1], color[2], color[3])
                            .map_err(crate::resources::failure)?,
                    )
                }
                _ => return Err(UNSUPPORTED),
            };
            colors.push(ir::OwnedColorAttachment {
                target: self.images[index].1,
                load,
                store: self.store(index),
            });
            self.access(index, ir::TextureAccess::RenderAttachment, rec);
            self.used[index] = true;
        }
        let depth = subpass
            .depth
            .map(|index| {
                let load = match self.load(index) {
                    vk::AttachmentLoadOp::LOAD => ir::DepthLoadOp::Load,
                    vk::AttachmentLoadOp::DONT_CARE => ir::DepthLoadOp::DontCare,
                    vk::AttachmentLoadOp::CLEAR => {
                        let value =
                            unsafe { self.clears.get(index).ok_or(INVALID)?.depth_stencil.depth };
                        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                            return Err(INVALID);
                        }
                        ir::DepthLoadOp::Clear(value)
                    }
                    _ => return Err(UNSUPPORTED),
                };
                self.access(
                    index,
                    if subpass.read_only_depth {
                        ir::TextureAccess::Sampled
                    } else {
                        ir::TextureAccess::RenderAttachment
                    },
                    rec,
                );
                self.used[index] = true;
                Ok(ir::OwnedDepthAttachment {
                    target: self.images[index].1,
                    load,
                    store: self.store(index),
                })
            })
            .transpose()?;
        for index in subpass.inputs.into_iter().flatten() {
            self.access(index, ir::TextureAccess::Sampled, rec);
            self.used[index] = true;
        }
        let first = colors.remove(0);
        let desc = ir::OwnedRenderPassDesc {
            target: first.target,
            area: self.area,
            load: first.load,
            store: first.store,
            depth,
        };
        rec.ops
            .push(if colors.is_empty() && !subpass.read_only_depth {
                ir::OwnedCommand::BeginRenderPass(desc)
            } else {
                ir::OwnedCommand::BeginRenderPassWithAttachments {
                    desc,
                    colors,
                    read_only_depth: subpass.read_only_depth,
                }
            });
        rec.emitted_graphics = Default::default();
        Ok(())
    }
}

pub(super) fn validate_bindings(
    resources: &crate::resources::Resources,
    rec: &ResolvedRecording,
    pipeline: vk::Pipeline,
    layout: &ir::PipelineLayoutDesc,
) -> VkResult<()> {
    let Some(active) = rec.active_render_pass.as_ref() else {
        return Ok(());
    };
    let (pass, subpass) = resources.graphics_subpasses.get(&pipeline).ok_or(INVALID)?;
    if *subpass != active.subpass || !pass.compatible(&active.pass) {
        return Err(INVALID);
    }
    let inputs = resources.graphics_inputs.get(&pipeline).ok_or(INVALID)?;
    for input in inputs {
        if layout
            .bind_groups()
            .get(input.group as usize)
            .is_none_or(|g| !g.entries().iter().any(|e| e.binding() == input.binding * 2))
        {
            continue;
        }
        let set = rec
            .graphics_sets
            .get(&input.group)
            .and_then(|set| resources.descriptor_sets.get(set))
            .ok_or(INVALID)?;
        if set.types.get(&input.binding) != Some(&vk::DescriptorType::INPUT_ATTACHMENT) {
            return Err(INVALID);
        }
        let crate::resources::DescriptorBinding::Image { view, .. } =
            set.bindings.get(&input.binding).ok_or(INVALID)?
        else {
            return Err(INVALID);
        };
        let index = active.pass.subpasses[active.subpass]
            .inputs
            .get(input.index)
            .copied()
            .flatten()
            .ok_or(INVALID)?;
        let fb = resources
            .framebuffers
            .get(&active.framebuffer)
            .ok_or(INVALID)?;
        let bound = resources.views.get(view).ok_or(INVALID)?;
        let attached = resources.views.get(&fb.attachments[index]).ok_or(INVALID)?;
        if bound.image != attached.image
            || bound.desc != attached.desc
            || bound.components != [0, 1, 2, 3]
        {
            return Err(INVALID);
        }
    }
    Ok(())
}
