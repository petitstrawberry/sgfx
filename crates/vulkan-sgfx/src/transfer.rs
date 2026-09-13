//! Checked staging uploads for the current host-visible memory implementation.
use crate::resources::Resources;
use ash::vk;
use sgfx::ir;

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
        || !matches!(
            target.format,
            vk::Format::R8G8B8A8_UNORM | vk::Format::B8G8R8A8_UNORM
        )
    {
        return Err(invalid);
    }
    let (memory, binding) = buffer.bound.ok_or(invalid)?;
    let memory = resources.memories.get(&memory).ok_or(invalid)?;
    let mut ops = Vec::with_capacity(regions.len());
    for region in regions {
        if region.image_subresource.aspect_mask != vk::ImageAspectFlags::COLOR
            || region.image_subresource.mip_level != 0
            || region.image_subresource.base_array_layer != 0
            || region.image_subresource.layer_count != 1
            || region.image_extent.depth != 1
            || region.image_offset.z != 0
            || region.image_offset.x < 0
            || region.image_offset.y < 0
            || !region.buffer_offset.is_multiple_of(4)
        {
            return Err(unsupported);
        }
        let destination = ir::PixelRect::new(
            region.image_offset.x as u32,
            region.image_offset.y as u32,
            region.image_extent.width,
            region.image_extent.height,
        )
        .map_err(|_| invalid)?;
        if destination.x() + destination.width() > target.extent.width
            || destination.y() + destination.height() > target.extent.height
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
        let stride = row.checked_mul(4).ok_or(invalid)?;
        let length = u64::from(stride)
            .checked_mul(u64::from(destination.height() - 1))
            .and_then(|v| v.checked_add(u64::from(destination.width()) * 4))
            .ok_or(invalid)?;
        let end = region
            .buffer_offset
            .checked_add(length)
            .filter(|end| *end <= buffer.size)
            .ok_or(invalid)?;
        let start = usize::try_from(binding.checked_add(region.buffer_offset).ok_or(invalid)?)
            .map_err(|_| invalid)?;
        let end = usize::try_from(binding.checked_add(end).ok_or(invalid)?).map_err(|_| invalid)?;
        let data = memory.bytes.get(start..end).ok_or(invalid)?.to_vec();
        ops.push(ir::OwnedCommand::WriteTexture {
            texture: target.id,
            destination,
            bytes_per_row: stride,
            data,
        });
    }
    Ok(ops)
}
