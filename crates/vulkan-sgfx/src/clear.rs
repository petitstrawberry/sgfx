//! Transfer clears lowered to real GPU render-pass clears.
use crate::images::Image;
use ash::vk;
use sgfx::ir;

const INVALID: vk::Result = vk::Result::ERROR_INITIALIZATION_FAILED;
const UNSUPPORTED: vk::Result = vk::Result::ERROR_FEATURE_NOT_PRESENT;

pub(crate) fn color_commands(
    image: &Image,
    layout: vk::ImageLayout,
    color: [f32; 4],
    ranges: &[vk::ImageSubresourceRange],
) -> Result<Vec<ir::OwnedCommand>, vk::Result> {
    if !image.usable()
        || !image.usage.contains(vk::ImageUsageFlags::TRANSFER_DST)
        || !matches!(
            layout,
            vk::ImageLayout::GENERAL | vk::ImageLayout::TRANSFER_DST_OPTIMAL
        )
        || ranges.is_empty()
    {
        return Err(INVALID);
    }
    if image.mip_levels != 1
        || image.array_layers != 1
        || !matches!(
            image.format,
            vk::Format::R8G8B8A8_UNORM
                | vk::Format::B8G8R8A8_UNORM
                | vk::Format::R8_UNORM
                | vk::Format::R8G8_UNORM
        )
    {
        return Err(UNSUPPORTED);
    }
    for range in ranges {
        if range.aspect_mask != vk::ImageAspectFlags::COLOR
            || range.base_mip_level != 0
            || range.base_array_layer != 0
            || !matches!(range.level_count, 1 | vk::REMAINING_MIP_LEVELS)
            || !matches!(range.layer_count, 1 | vk::REMAINING_ARRAY_LAYERS)
        {
            return Err(UNSUPPORTED);
        }
    }
    let color = ir::Color::rgba(color[0], color[1], color[2], color[3])
        .map_err(crate::resources::failure)?;
    let barrier = |before, after| {
        ir::OwnedCommand::ResourceBarrier(ir::OwnedResourceBarrier::Texture {
            texture: image.id,
            before,
            after,
        })
    };
    // Repeated/overlapping ranges target the same single subresource. One
    // clear has identical semantics, with no CPU pixel allocation or upload.
    Ok(vec![
        barrier(
            ir::TextureAccess::CopyDestination,
            ir::TextureAccess::RenderAttachment,
        ),
        ir::OwnedCommand::BeginRenderPass(ir::OwnedRenderPassDesc {
            target: image.id,
            area: ir::PixelRect::new(0, 0, image.extent.width, image.extent.height)
                .map_err(crate::resources::failure)?,
            load: ir::LoadOp::Clear(color),
            store: ir::StoreOp::Store,
            depth: None,
        }),
        ir::OwnedCommand::EndRenderPass,
        barrier(
            ir::TextureAccess::RenderAttachment,
            ir::TextureAccess::CopyDestination,
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    fn image() -> Image {
        let table = ir::ResourceTable::new();
        let id = table
            .define_texture(
                ir::TextureDesc::new(
                    ir::TextureFormat::Bgra8Unorm,
                    ir::Extent2D::new(64, 64).unwrap(),
                    ir::TextureUsage::RENDER_ATTACHMENT | ir::TextureUsage::COPY_DST,
                )
                .unwrap(),
            )
            .unwrap()
            .id();
        Image {
            id,
            format: vk::Format::B8G8R8A8_UNORM,
            extent: vk::Extent3D {
                width: 64,
                height: 64,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            flags: vk::ImageCreateFlags::empty(),
            usage: vk::ImageUsageFlags::TRANSFER_DST,
            bound: Some((vk::DeviceMemory::null(), 0)),
            swapchain: None,
            #[cfg(target_os = "scarlet")]
            shared: None,
        }
    }
    #[test]
    fn clear_validates_all_ranges_before_emitting_any_gpu_work() {
        let mut range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(vk::REMAINING_MIP_LEVELS)
            .layer_count(vk::REMAINING_ARRAY_LAYERS);
        let commands = color_commands(
            &image(),
            vk::ImageLayout::GENERAL,
            [1., 0., 0., 1.],
            &[range],
        )
        .unwrap();
        assert!(matches!(
            commands.as_slice(),
            [
                ir::OwnedCommand::ResourceBarrier(_),
                ir::OwnedCommand::BeginRenderPass(_),
                ir::OwnedCommand::EndRenderPass,
                ir::OwnedCommand::ResourceBarrier(_)
            ]
        ));
        let valid = range;
        range.base_mip_level = 1;
        assert_eq!(
            color_commands(&image(), vk::ImageLayout::GENERAL, [1.; 4], &[valid, range])
                .unwrap_err(),
            UNSUPPORTED
        );
        assert_eq!(
            color_commands(
                &image(),
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                [1.; 4],
                &[valid]
            )
            .unwrap_err(),
            INVALID
        );
        assert!(
            color_commands(&image(), vk::ImageLayout::GENERAL, [f32::NAN; 4], &[valid]).is_err()
        );
    }
}
