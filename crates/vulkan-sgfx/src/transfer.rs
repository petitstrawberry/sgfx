//! Checked staging uploads for the current host-visible memory implementation.
use crate::resources::Resources;
use ash::vk;
use sgfx::ir;

/// Lower complete, positive-direction color mip blits into executable IR.
/// Partial source/destination rectangles, flips, format conversion and layers
/// remain outside this subset and are rejected before backend execution.
pub(crate) fn blit(
    resources: &Resources,
    source: vk::Image,
    source_layout: vk::ImageLayout,
    destination: vk::Image,
    destination_layout: vk::ImageLayout,
    regions: &[vk::ImageBlit],
    filter: vk::Filter,
) -> Result<Vec<ir::OwnedCommand>, vk::Result> {
    let invalid = vk::Result::ERROR_INITIALIZATION_FAILED;
    let unsupported = vk::Result::ERROR_FEATURE_NOT_PRESENT;
    if !matches!(
        source_layout,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL | vk::ImageLayout::GENERAL
    ) || !matches!(
        destination_layout,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL | vk::ImageLayout::GENERAL
    ) {
        return Err(unsupported);
    }
    let filter = match filter {
        vk::Filter::NEAREST => ir::FilterMode::Nearest,
        vk::Filter::LINEAR => ir::FilterMode::Linear,
        _ => return Err(unsupported),
    };
    let src = resources.images.get(&source).ok_or(invalid)?;
    let dst = resources.images.get(&destination).ok_or(invalid)?;
    if !src.usable()
        || !dst.usable()
        || !src.usage.contains(vk::ImageUsageFlags::TRANSFER_SRC)
        || !dst.usage.contains(vk::ImageUsageFlags::TRANSFER_DST)
    {
        return Err(invalid);
    }
    if src.format != dst.format
        || !matches!(
            src.format,
            vk::Format::R8G8B8A8_UNORM | vk::Format::B8G8R8A8_UNORM | vk::Format::R8_UNORM
        )
    {
        return Err(unsupported);
    }
    let mut commands = Vec::with_capacity(regions.len());
    for region in regions {
        for (image, subresource, offsets) in [
            (src, region.src_subresource, region.src_offsets),
            (dst, region.dst_subresource, region.dst_offsets),
        ] {
            let extent = image.mip_extent(subresource.mip_level)?;
            if subresource.aspect_mask != vk::ImageAspectFlags::COLOR
                || subresource.base_array_layer != 0
                || subresource.layer_count != 1
                || offsets
                    != [
                        vk::Offset3D::default(),
                        vk::Offset3D {
                            x: extent.width as i32,
                            y: extent.height as i32,
                            z: 1,
                        },
                    ]
            {
                return Err(unsupported);
            }
        }
        if source == destination
            && region.src_subresource.mip_level == region.dst_subresource.mip_level
        {
            return Err(invalid);
        }
        commands.push(ir::OwnedCommand::BlitTexture {
            source: src.id,
            source_mip: region.src_subresource.mip_level,
            destination: dst.id,
            destination_mip: region.dst_subresource.mip_level,
            filter,
        });
    }
    Ok(commands)
}

pub(crate) fn upload(
    resources: &Resources,
    source: vk::Buffer,
    image: vk::Image,
    layout: vk::ImageLayout,
    regions: &[vk::BufferImageCopy],
) -> Result<Vec<ir::OwnedCommand>, vk::Result> {
    let invalid = vk::Result::ERROR_INITIALIZATION_FAILED;
    let unsupported = vk::Result::ERROR_FEATURE_NOT_PRESENT;
    if !matches!(
        layout,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL | vk::ImageLayout::GENERAL
    ) {
        return Err(unsupported);
    }
    let buffer = resources.buffers.get(&source).ok_or(invalid)?;
    let target = resources.images.get(&image).ok_or(invalid)?;
    if !buffer.usage.contains(vk::BufferUsageFlags::TRANSFER_SRC)
        || !target.usage.contains(vk::ImageUsageFlags::TRANSFER_DST)
        || !target.usable()
        || target.format == vk::Format::D32_SFLOAT
    {
        return Err(invalid);
    }
    let (memory, binding) = buffer.bound.ok_or(invalid)?;
    let memory = resources.memories.get(&memory).ok_or(invalid)?;
    let bpp = crate::images::texture_format(target.format)
        .ok_or(unsupported)?
        .bytes_per_pixel();
    let mut ops = Vec::with_capacity(regions.len());
    for region in regions {
        if region.image_subresource.aspect_mask != vk::ImageAspectFlags::COLOR
            || region.image_subresource.layer_count == 0
            || region
                .image_subresource
                .base_array_layer
                .checked_add(region.image_subresource.layer_count)
                .is_none_or(|end| end > target.array_layers)
            || region.image_extent.depth != 1
            || region.image_offset.z != 0
            || region.image_offset.x < 0
            || region.image_offset.y < 0
            || !region.buffer_offset.is_multiple_of(u64::from(bpp))
        {
            return Err(unsupported);
        }
        let mip_level = region.image_subresource.mip_level;
        let extent = target.mip_extent(mip_level)?;
        let destination = ir::PixelRect::new(
            region.image_offset.x as u32,
            region.image_offset.y as u32,
            region.image_extent.width,
            region.image_extent.height,
        )
        .map_err(|_| invalid)?;
        if destination.x() + destination.width() > extent.width
            || destination.y() + destination.height() > extent.height
        {
            return Err(invalid);
        }
        let row = if region.buffer_row_length == 0 {
            destination.width()
        } else {
            region.buffer_row_length
        };
        if row < destination.width()
            || (region.buffer_image_height != 0
                && region.buffer_image_height < destination.height())
        {
            return Err(invalid);
        }
        let stride = row.checked_mul(bpp).ok_or(invalid)?;
        let length = u64::from(stride)
            .checked_mul(u64::from(destination.height() - 1))
            .and_then(|v| v.checked_add(u64::from(destination.width()) * u64::from(bpp)))
            .ok_or(invalid)?;
        let rows_per_layer = if region.buffer_image_height == 0 {
            destination.height()
        } else {
            region.buffer_image_height
        };
        let layer_stride = u64::from(stride) * u64::from(rows_per_layer);
        for layer in 0..region.image_subresource.layer_count {
            let offset = region
                .buffer_offset
                .checked_add(layer_stride.checked_mul(u64::from(layer)).ok_or(invalid)?)
                .ok_or(invalid)?;
            let end = offset
                .checked_add(length)
                .filter(|end| *end <= buffer.size)
                .ok_or(invalid)?;
            let start = usize::try_from(binding.checked_add(offset).ok_or(invalid)?)
                .map_err(|_| invalid)?;
            let end =
                usize::try_from(binding.checked_add(end).ok_or(invalid)?).map_err(|_| invalid)?;
            let data = memory.bytes.get(start..end).ok_or(invalid)?.to_vec();
            let array_layer = region.image_subresource.base_array_layer + layer;
            if target.array_layers == 1 {
                ops.push(ir::OwnedCommand::WriteTextureMip {
                    mip_level,
                    texture: target.id,
                    destination,
                    bytes_per_row: stride,
                    data,
                });
            } else {
                ops.push(ir::OwnedCommand::WriteTextureLayer {
                    mip_level,
                    array_layer,
                    texture: target.id,
                    destination,
                    bytes_per_row: stride,
                    data,
                });
            }
        }
    }
    Ok(ops)
}
