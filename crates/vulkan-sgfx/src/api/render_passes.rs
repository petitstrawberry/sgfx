//! Resolve a Vulkan render pass into ordered SGFX passes without CPU readback.
use super::*;
const INVALID: vk::Result = vk::Result::ERROR_INITIALIZATION_FAILED;
const UNSUPPORTED: vk::Result = vk::Result::ERROR_FEATURE_NOT_PRESENT;

#[derive(Clone)]
pub(super) struct ActiveRenderPass {
    pub pass: crate::render_pass::RenderPass,
    pub views: Vec<vk::ImageView>,
    pub subpass: usize,
    clears: Vec<vk::ClearValue>,
    images: Vec<(vk::Image, ir::TextureId)>,
    used: Vec<bool>,
    accesses: Vec<Option<ir::TextureAccess>>,
    pub(super) area: ir::PixelRect,
}

pub(super) fn begin(
    rt: &Runtime,
    rec: &mut ResolvedRecording,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    area: vk::Rect2D,
    clears: &[vk::ClearValue],
    attachments: Option<&[vk::ImageView]>,
) -> VkResult<()> {
    let pass = rt
        .resources
        .render_passes
        .get(&render_pass)
        .ok_or(INVALID)?;
    let fb = rt.resources.framebuffers.get(&framebuffer).ok_or(INVALID)?;
    if !pass.compatible(&fb.render_pass) {
        return Err(INVALID);
    }
    let area = render_area(area, fb.width, fb.height)?;
    let views = if let Some(descriptions) = &fb.imageless {
        let views = attachments.ok_or(INVALID)?;
        if views.len() != descriptions.len() {
            return Err(INVALID);
        }
        for ((view, d), a) in views.iter().zip(descriptions).zip(&pass.attachments) {
            let view = rt.resources.views.get(view).ok_or(INVALID)?;
            let image = rt.resources.images.get(&view.image).ok_or(INVALID)?;
            if image.flags != d.flags
                || !image.usage.contains(d.usage)
                || image.extent.width < d.width
                || image.extent.height < d.height
                || image.array_layers != d.layers
                || image.format != a.format
                || !d.formats.contains(&image.format)
                || view.components != [0, 1, 2, 3]
                || view.desc.base_mip_level() != 0
                || view.desc.mip_level_count() != 1
                || view.desc.base_array_layer() != 0
                || view.desc.array_layer_count() != 1
            {
                return Err(UNSUPPORTED);
            }
        }
        views
    } else {
        if attachments.is_some_and(|a| !a.is_empty()) {
            return Err(INVALID);
        }
        &fb.attachments
    };
    rec.used_framebuffers.push(framebuffer);
    rec.used_render_passes.push(render_pass);
    begin_owned(rt, rec, pass, views, (fb.width, fb.height), area, clears)
}

pub(super) fn begin_owned(
    rt: &Runtime,
    rec: &mut ResolvedRecording,
    pass: &crate::render_pass::RenderPass,
    views: &[vk::ImageView],
    (width, height): (u32, u32),
    area: ir::PixelRect,
    clears: &[vk::ClearValue],
) -> VkResult<()> {
    let images = views
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
    rec.render = Some((width, height));
    let mut active = ActiveRenderPass {
        pass: pass.clone(),
        views: views.to_vec(),
        subpass: 0,
        clears: clears.to_vec(),
        images,
        used: vec![false; pass.attachments.len()],
        accesses: vec![None; pass.attachments.len()],
        area,
    };
    active.emit(rec)?;
    rec.active_render_pass = Some(active);
    Ok(())
}

/// Retain the caller's render area; load/clear/store operations use these bounds.
fn render_area(area: vk::Rect2D, width: u32, height: u32) -> VkResult<ir::PixelRect> {
    let x = u32::try_from(area.offset.x).map_err(|_| INVALID)?;
    let y = u32::try_from(area.offset.y).map_err(|_| INVALID)?;
    let area =
        ir::PixelRect::new(x, y, area.extent.width, area.extent.height).map_err(|_| INVALID)?;
    if area.x() + area.width() > width || area.y() + area.height() > height {
        return Err(INVALID);
    }
    Ok(area)
}

/// SGFX requires scissors to lie within the active render area. An empty
/// intersection is a valid Vulkan scissor and suppresses the draw.
pub(super) fn clip_scissor(
    scissor: vk::Rect2D,
    area: ir::PixelRect,
) -> VkResult<Option<ir::PixelRect>> {
    let x = u32::try_from(scissor.offset.x).map_err(|_| UNSUPPORTED)?;
    let y = u32::try_from(scissor.offset.y).map_err(|_| UNSUPPORTED)?;
    let right = x.checked_add(scissor.extent.width).ok_or(INVALID)?;
    let bottom = y.checked_add(scissor.extent.height).ok_or(INVALID)?;
    if right > i32::MAX as u32 || bottom > i32::MAX as u32 {
        return Err(INVALID);
    }
    let left = x.max(area.x());
    let top = y.max(area.y());
    let right = right.min(area.x() + area.width());
    let bottom = bottom.min(area.y() + area.height());
    if left >= right || top >= bottom {
        return Ok(None);
    }
    Ok(Some(
        ir::PixelRect::new(left, top, right - left, bottom - top).map_err(|_| INVALID)?,
    ))
}

pub(super) fn retain_current_pass(rec: &mut ResolvedRecording) -> VkResult<()> {
    let original = rec
        .ops
        .iter_mut()
        .rev()
        .find(|op| {
            matches!(
                op,
                ir::OwnedCommand::BeginRenderPass(_)
                    | ir::OwnedCommand::BeginRenderPassWithAttachments { .. }
            )
        })
        .ok_or(INVALID)?;
    let (desc, colors) = match original {
        ir::OwnedCommand::BeginRenderPass(desc) => (desc, None),
        ir::OwnedCommand::BeginRenderPassWithAttachments { desc, colors, .. } => {
            (desc, Some(colors))
        }
        _ => unreachable!(),
    };
    desc.store = ir::StoreOp::Store;
    if let Some(depth) = &mut desc.depth {
        depth.store = ir::StoreOp::Store;
    }
    if let Some(colors) = colors {
        for color in colors {
            color.store = ir::StoreOp::Store;
        }
    }
    Ok(())
}

/// Clear attachment rectangles in rasterization order through genuine GPU
/// load-op clears. Vulkan bindings survive these internal pass boundaries.
pub(super) fn clear_attachments(
    rec: &mut ResolvedRecording,
    attachments: &[vk::ClearAttachment],
    rects: &[vk::ClearRect],
) -> VkResult<()> {
    let active = rec.active_render_pass.as_ref().ok_or(INVALID)?;
    if attachments.is_empty() || rects.is_empty() {
        return Ok(());
    }
    // Validate everything before changing the original pass. A bad later
    // rectangle must not leave a partially emitted clear sequence.
    let areas = rects
        .iter()
        .map(|rect| {
            if rect.base_array_layer != 0 || rect.layer_count != 1 {
                return Err(UNSUPPORTED);
            }
            let x = u32::try_from(rect.rect.offset.x).map_err(|_| INVALID)?;
            let y = u32::try_from(rect.rect.offset.y).map_err(|_| INVALID)?;
            let area = ir::PixelRect::new(x, y, rect.rect.extent.width, rect.rect.extent.height)
                .map_err(|_| INVALID)?;
            // Vulkan requires each rectangle to lie within the render area.
            // Pipeline viewport/scissor must not affect this command.
            if area.x() < active.area.x()
                || area.y() < active.area.y()
                || area.x() + area.width() > active.area.x() + active.area.width()
                || area.y() + area.height() > active.area.y() + active.area.height()
            {
                return Err(INVALID);
            }
            Ok(area)
        })
        .collect::<VkResult<Vec<_>>>()?;
    let subpass = &active.pass.subpasses[active.subpass];
    let clears = attachments
        .iter()
        .map(|attachment| {
            if attachment.aspect_mask != vk::ImageAspectFlags::COLOR {
                return Err(UNSUPPORTED);
            }
            let index = *subpass
                .colors
                .get(attachment.color_attachment as usize)
                .ok_or(INVALID)?;
            let color = unsafe { attachment.clear_value.color.float32 };
            let color = ir::Color::rgba(color[0], color[1], color[2], color[3])
                .map_err(crate::resources::failure)?;
            Ok((active.images[index].1, color))
        })
        .collect::<VkResult<Vec<_>>>()?;
    if clears
        .len()
        .checked_mul(areas.len())
        .filter(|&count| count <= ir::MAX_COMMANDS / 4)
        .is_none()
    {
        return Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY);
    }
    // Retain intermediate color/depth contents, even when the Vulkan final
    // store is DONT_CARE. Vulkan permits preserving those otherwise undefined
    // contents; SGFX needs LOAD/STORE to resume an empty pass correctly.
    retain_current_pass(rec)?;
    rec.ops.push(ir::OwnedCommand::EndRenderPass);
    for &(target, color) in &clears {
        for &area in &areas {
            rec.ops
                .push(ir::OwnedCommand::BeginRenderPass(ir::OwnedRenderPassDesc {
                    target,
                    area,
                    load: ir::LoadOp::Clear(color),
                    store: ir::StoreOp::Store,
                    depth: None,
                }));
            rec.ops.push(ir::OwnedCommand::EndRenderPass);
        }
    }
    let mut active = rec.active_render_pass.take().ok_or(INVALID)?;
    active.emit(rec)?;
    retain_current_pass(rec)?;
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
        let Some(group) = layout.bind_groups().get(input.group as usize) else {
            continue;
        };
        if group.entries().is_empty() {
            continue;
        }
        let set = rec
            .graphics_sets
            .get(&input.group)
            .and_then(|set| resources.descriptor_sets.get(set))
            .ok_or(INVALID)?;
        let binding = set.source_to_ir(input.binding).ok_or(INVALID)?;
        if !group
            .entries()
            .iter()
            .any(|entry| entry.binding() == binding)
        {
            continue;
        }
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
        let bound = resources.views.get(view).ok_or(INVALID)?;
        let attached = resources.views.get(&active.views[index]).ok_or(INVALID)?;
        if bound.image != attached.image
            || bound.desc != attached.desc
            || bound.components != [0, 1, 2, 3]
        {
            return Err(INVALID);
        }
    }
    Ok(())
}

/// Split a pass without losing attachment contents or implicit binding state.
pub(super) fn suspend(rec: &mut ResolvedRecording) -> VkResult<ActiveRenderPass> {
    retain_current_pass(rec)?;
    let active = rec.active_render_pass.take().ok_or(INVALID)?;
    rec.ops.push(ir::OwnedCommand::EndRenderPass);
    Ok(active)
}
pub(super) fn resume(rec: &mut ResolvedRecording, mut active: ActiveRenderPass) -> VkResult<()> {
    active.emit(rec)?;
    rec.active_render_pass = Some(active);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x: i32, y: i32, width: u32, height: u32) -> vk::Rect2D {
        vk::Rect2D {
            offset: vk::Offset2D { x, y },
            extent: vk::Extent2D { width, height },
        }
    }

    fn clear_fixture() -> ResolvedRecording {
        let table = ir::ResourceTable::new();
        let target = table
            .define_texture(
                ir::TextureDesc::new(
                    ir::TextureFormat::Rgba8Unorm,
                    ir::Extent2D::new(32, 32).unwrap(),
                    ir::TextureUsage::RENDER_ATTACHMENT,
                )
                .unwrap(),
            )
            .unwrap()
            .id();
        let attachment = vk::AttachmentDescription::default()
            .format(vk::Format::R8G8B8A8_UNORM)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE);
        let mut active = ActiveRenderPass {
            pass: crate::render_pass::RenderPass {
                dynamic: false,
                attachments: vec![attachment],
                subpasses: vec![crate::render_pass::Subpass {
                    colors: vec![0],
                    inputs: vec![],
                    depth: None,
                    read_only_depth: false,
                }],
            },
            views: vec![],
            subpass: 0,
            clears: vec![vk::ClearValue {
                color: vk::ClearColorValue { float32: [1.0; 4] },
            }],
            images: vec![(vk::Image::null(), target)],
            used: vec![false],
            accesses: vec![None],
            area: ir::PixelRect::new(7, 9, 25, 23).unwrap(),
        };
        let mut rec = ResolvedRecording::new();
        active.emit(&mut rec).unwrap();
        rec.active_render_pass = Some(active);
        rec
    }
    fn color_clear() -> vk::ClearAttachment {
        vk::ClearAttachment::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .color_attachment(0)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 1.0, 1.0],
                },
            })
    }
    fn clear_rect(rect: vk::Rect2D) -> vk::ClearRect {
        vk::ClearRect::default()
            .rect(rect)
            .base_array_layer(0)
            .layer_count(1)
    }
    #[test]
    fn attachment_clear_splits_gpu_pass_preserves_store_and_resumes_load() {
        let mut rec = clear_fixture();
        clear_attachments(&mut rec, &[color_clear()], &[clear_rect(rect(8, 10, 3, 4))]).unwrap();
        let passes = rec
            .ops
            .iter()
            .filter_map(|op| match op {
                ir::OwnedCommand::BeginRenderPass(desc) => Some(desc),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(passes.len(), 3);
        assert_eq!(passes[0].store, ir::StoreOp::Store);
        assert_eq!(passes[1].area, ir::PixelRect::new(8, 10, 3, 4).unwrap());
        assert_eq!(
            passes[1].load,
            ir::LoadOp::Clear(ir::Color::rgba(0.0, 0.0, 1.0, 1.0).unwrap())
        );
        assert_eq!(passes[1].store, ir::StoreOp::Store);
        assert_eq!(passes[2].area, ir::PixelRect::new(7, 9, 25, 23).unwrap());
        assert_eq!(passes[2].load, ir::LoadOp::Load);
        assert_eq!(passes[2].store, ir::StoreOp::Store);
    }
    #[test]
    fn attachment_clear_rejects_bad_regions_layers_aspects_without_partial_emission() {
        for bad in [
            rect(6, 9, 1, 1),
            rect(7, 8, 1, 1),
            rect(31, 9, 2, 1),
            rect(7, 31, 1, 2),
            rect(-1, 9, 1, 1),
            rect(8, 10, 0, 1),
            rect(8, 10, 1, 0),
        ] {
            let mut rec = clear_fixture();
            let before = rec.ops.len();
            assert_eq!(
                clear_attachments(
                    &mut rec,
                    &[color_clear()],
                    &[clear_rect(rect(8, 10, 1, 1)), clear_rect(bad)]
                ),
                Err(INVALID)
            );
            assert_eq!(rec.ops.len(), before);
        }
        for invalid in [
            clear_rect(rect(8, 10, 1, 1)).base_array_layer(1),
            clear_rect(rect(8, 10, 1, 1)).layer_count(0),
            clear_rect(rect(8, 10, 1, 1)).layer_count(2),
        ] {
            let mut rec = clear_fixture();
            let before = rec.ops.len();
            assert_eq!(
                clear_attachments(&mut rec, &[color_clear()], &[invalid]),
                Err(UNSUPPORTED)
            );
            assert_eq!(rec.ops.len(), before);
        }
        let mut rec = clear_fixture();
        let before = rec.ops.len();
        assert_eq!(
            clear_attachments(
                &mut rec,
                &[color_clear().aspect_mask(vk::ImageAspectFlags::DEPTH)],
                &[clear_rect(rect(8, 10, 1, 1))]
            ),
            Err(UNSUPPORTED)
        );
        assert_eq!(rec.ops.len(), before);
        assert_eq!(
            clear_attachments(
                &mut rec,
                &[color_clear().color_attachment(1)],
                &[clear_rect(rect(8, 10, 1, 1))]
            ),
            Err(INVALID)
        );
        assert_eq!(rec.ops.len(), before);
        clear_attachments(&mut rec, &[color_clear()], &[]).unwrap();
        clear_attachments(&mut rec, &[], &[clear_rect(rect(8, 10, 1, 1))]).unwrap();
        assert_eq!(rec.ops.len(), before);
    }

    #[test]
    fn render_area_preserves_partial_bounds_and_rejects_invalid_regions() {
        assert_eq!(
            render_area(rect(7, 9, 25, 23), 32, 32),
            Ok(ir::PixelRect::new(7, 9, 25, 23).unwrap())
        );
        assert_eq!(
            render_area(rect(0, 0, 32, 32), 32, 32),
            Ok(ir::PixelRect::new(0, 0, 32, 32).unwrap())
        );
        for area in [
            rect(-1, 0, 1, 1),
            rect(0, -1, 1, 1),
            rect(0, 0, 0, 1),
            rect(0, 0, 1, 0),
            rect(32, 0, 1, 1),
            rect(0, 32, 1, 1),
            rect(1, 0, 32, 1),
            rect(0, 1, 1, 32),
            rect(1, 0, u32::MAX, 1),
            rect(0, 1, 1, u32::MAX),
        ] {
            assert_eq!(render_area(area, 32, 32), Err(INVALID), "{area:?}");
        }
    }

    #[test]
    fn scissors_intersect_the_render_area_and_empty_draws_are_valid() {
        let area = ir::PixelRect::new(7, 9, 25, 23).unwrap();
        assert_eq!(clip_scissor(rect(0, 0, 32, 32), area), Ok(Some(area)));
        assert_eq!(
            clip_scissor(rect(20, 20, 30, 30), area),
            Ok(Some(ir::PixelRect::new(20, 20, 12, 12).unwrap()))
        );
        for scissor in [
            rect(0, 0, 7, 9),
            rect(32, 0, 1, 32),
            rect(0, 32, 32, 1),
            rect(8, 10, 0, 1),
            rect(8, 10, 1, 0),
        ] {
            assert_eq!(clip_scissor(scissor, area), Ok(None), "{scissor:?}");
        }
        assert_eq!(clip_scissor(rect(-1, 0, 32, 32), area), Err(UNSUPPORTED));
        assert_eq!(clip_scissor(rect(1, 0, u32::MAX, 32), area), Err(INVALID));
        assert_eq!(clip_scissor(rect(i32::MAX, 0, 1, 1), area), Err(INVALID));
        assert_eq!(clip_scissor(rect(0, i32::MAX, 1, 1), area), Err(INVALID));
    }

    #[test]
    fn partial_subpasses_retain_area_clear_once_and_store_intermediates() {
        let table = ir::ResourceTable::new();
        let images = (0..2)
            .map(|_| {
                (
                    vk::Image::null(),
                    table
                        .define_texture(
                            ir::TextureDesc::new(
                                ir::TextureFormat::Rgba8Unorm,
                                ir::Extent2D::new(32, 32).unwrap(),
                                ir::TextureUsage::RENDER_ATTACHMENT,
                            )
                            .unwrap(),
                        )
                        .unwrap()
                        .id(),
                )
            })
            .collect();
        let attachment = vk::AttachmentDescription::default()
            .format(vk::Format::R8G8B8A8_UNORM)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let area = ir::PixelRect::new(7, 9, 25, 23).unwrap();
        let mut active = ActiveRenderPass {
            pass: crate::render_pass::RenderPass {
                dynamic: false,
                attachments: vec![attachment; 2],
                subpasses: vec![
                    crate::render_pass::Subpass {
                        colors: vec![0, 1],
                        inputs: vec![],
                        depth: None,
                        read_only_depth: false,
                    },
                    crate::render_pass::Subpass {
                        colors: vec![0],
                        inputs: vec![],
                        depth: None,
                        read_only_depth: false,
                    },
                ],
            },
            views: vec![],
            subpass: 0,
            clears: vec![
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [1.0, 0.0, 0.0, 1.0]
                    }
                };
                2
            ],
            images,
            used: vec![false; 2],
            accesses: vec![None; 2],
            area,
        };
        let mut rec = ResolvedRecording::new();
        active.emit(&mut rec).unwrap();
        let Some(ir::OwnedCommand::BeginRenderPassWithAttachments { desc, colors, .. }) =
            rec.ops.last()
        else {
            panic!("MRT pass expected")
        };
        assert_eq!(desc.area, area);
        assert!(matches!(desc.load, ir::LoadOp::Clear(_)));
        assert_eq!(desc.store, ir::StoreOp::Store);
        assert!(matches!(colors[0].load, ir::LoadOp::Clear(_)));
        assert_eq!(colors[0].store, ir::StoreOp::DontCare);
        rec.active_render_pass = Some(active);
        next(&mut rec).unwrap();
        let Some(ir::OwnedCommand::BeginRenderPass(desc)) = rec.ops.last() else {
            panic!("second pass expected")
        };
        assert_eq!(desc.area, area);
        assert_eq!(desc.load, ir::LoadOp::Load);
        assert_eq!(desc.store, ir::StoreOp::DontCare);
        end(&mut rec).unwrap();
        assert!(matches!(
            rec.ops.last(),
            Some(ir::OwnedCommand::EndRenderPass)
        ));
    }
}
