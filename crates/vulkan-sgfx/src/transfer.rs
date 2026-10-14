//! Checked staging uploads for the current host-visible memory implementation.
use crate::resources::Resources;
use ash::vk;
use sgfx::ir;

/// Layout of one color image region copied into a host-visible buffer.
/// GPU readback supplies the full packed mip; only the selected rows are written.
#[derive(Clone)]
pub(crate) struct ImageReadbackRegion {
    full_size: usize,
    source_offset: usize,
    source_stride: usize,
    row_bytes: usize,
    rows: usize,
    destination_offset: u64,
    destination_stride: u64,
}

impl ImageReadbackRegion {
    pub(crate) fn new(
        extent: vk::Extent3D,
        bytes_per_pixel: u32,
        buffer_size: u64,
        region: &vk::BufferImageCopy,
    ) -> Result<Self, vk::Result> {
        let invalid = vk::Result::ERROR_INITIALIZATION_FAILED;
        if region.image_subresource.aspect_mask != vk::ImageAspectFlags::COLOR
            || region.image_subresource.base_array_layer != 0
            || region.image_subresource.layer_count != 1
            || extent.depth != 1
            || region.image_extent.depth != 1
            || region.image_offset.z != 0
        {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let x = u32::try_from(region.image_offset.x).map_err(|_| invalid)?;
        let y = u32::try_from(region.image_offset.y).map_err(|_| invalid)?;
        let width = region.image_extent.width;
        let height = region.image_extent.height;
        if !matches!(bytes_per_pixel, 1 | 2 | 4)
            || width == 0
            || height == 0
            || x.checked_add(width).is_none_or(|end| end > extent.width)
            || y.checked_add(height).is_none_or(|end| end > extent.height)
            || !region
                .buffer_offset
                .is_multiple_of(u64::from(bytes_per_pixel))
        {
            return Err(invalid);
        }
        let destination_width = if region.buffer_row_length == 0 {
            width
        } else {
            region.buffer_row_length
        };
        if destination_width < width
            || (region.buffer_image_height != 0 && region.buffer_image_height < height)
        {
            return Err(invalid);
        }
        let bpp = u64::from(bytes_per_pixel);
        let source_stride = u64::from(extent.width).checked_mul(bpp).ok_or(invalid)?;
        let full_size = source_stride
            .checked_mul(u64::from(extent.height))
            .ok_or(invalid)?;
        let source_offset = u64::from(y)
            .checked_mul(source_stride)
            .and_then(|row| row.checked_add(u64::from(x) * bpp))
            .ok_or(invalid)?;
        let row_bytes = u64::from(width).checked_mul(bpp).ok_or(invalid)?;
        let destination_stride = u64::from(destination_width)
            .checked_mul(bpp)
            .ok_or(invalid)?;
        let end = destination_stride
            .checked_mul(u64::from(height - 1))
            .and_then(|tail| tail.checked_add(row_bytes))
            .and_then(|length| region.buffer_offset.checked_add(length))
            .ok_or(invalid)?;
        if end > buffer_size {
            return Err(invalid);
        }
        Ok(Self {
            full_size: usize::try_from(full_size).map_err(|_| invalid)?,
            source_offset: usize::try_from(source_offset).map_err(|_| invalid)?,
            source_stride: usize::try_from(source_stride).map_err(|_| invalid)?,
            row_bytes: usize::try_from(row_bytes).map_err(|_| invalid)?,
            rows: usize::try_from(height).map_err(|_| invalid)?,
            destination_offset: region.buffer_offset,
            destination_stride,
        })
    }

    pub(crate) fn writes(
        &self,
        pixels: &[u8],
        buffer: ir::BufferId,
    ) -> Result<Vec<ir::OwnedCommand>, vk::Result> {
        if pixels.len() != self.full_size {
            return Err(vk::Result::ERROR_DEVICE_LOST);
        }
        let mut writes = Vec::new();
        // Preserve the common full-width packed readback as one buffer write.
        if self.source_stride == self.row_bytes && self.destination_stride == self.row_bytes as u64
        {
            let end = self.source_offset + self.rows * self.row_bytes;
            writes.push(ir::OwnedCommand::WriteBuffer {
                buffer,
                offset: self.destination_offset,
                data: pixels[self.source_offset..end].to_vec(),
            });
        } else {
            for row in 0..self.rows {
                let start = self.source_offset + row * self.source_stride;
                writes.push(ir::OwnedCommand::WriteBuffer {
                    buffer,
                    offset: self.destination_offset + row as u64 * self.destination_stride,
                    data: pixels[start..start + self.row_bytes].to_vec(),
                });
            }
        }
        Ok(writes)
    }
}

/// Exact same-format color copies. Unlike blits, this uses the canonical
/// CopyTextureToTexture operation: no sampling, gamma conversion or blending.
/// The existing IR copy contract addresses only mip zero and a single layer.
pub(crate) fn copy_image(
    resources: &Resources,
    source: vk::Image,
    source_layout: vk::ImageLayout,
    destination: vk::Image,
    destination_layout: vk::ImageLayout,
    regions: &[vk::ImageCopy],
) -> Result<Vec<ir::OwnedCommand>, vk::Result> {
    let invalid = vk::Result::ERROR_INITIALIZATION_FAILED;
    let unsupported = vk::Result::ERROR_FEATURE_NOT_PRESENT;
    if !matches!(
        source_layout,
        vk::ImageLayout::GENERAL | vk::ImageLayout::TRANSFER_SRC_OPTIMAL
    ) || !matches!(
        destination_layout,
        vk::ImageLayout::GENERAL | vk::ImageLayout::TRANSFER_DST_OPTIMAL
    ) {
        return Err(invalid);
    }
    let src = resources.images.get(&source).ok_or(invalid)?;
    let dst = resources.images.get(&destination).ok_or(invalid)?;
    if !src.usable()
        || !dst.usable()
        || !src.usage.contains(vk::ImageUsageFlags::TRANSFER_SRC)
        || !dst.usage.contains(vk::ImageUsageFlags::TRANSFER_DST)
        || regions.is_empty()
    {
        return Err(invalid);
    }
    if src.format != dst.format
        || !matches!(
            src.format,
            vk::Format::B8G8R8A8_UNORM
                | vk::Format::R8G8B8A8_UNORM
                | vk::Format::A8B8G8R8_UNORM_PACK32
                | vk::Format::B8G8R8A8_SRGB
                | vk::Format::R8G8B8A8_SRGB
                | vk::Format::A8B8G8R8_SRGB_PACK32
                | vk::Format::R8_UNORM
                | vk::Format::R8G8_UNORM
        )
        || src.array_layers != 1
        || dst.array_layers != 1
    {
        return Err(unsupported);
    }
    let rect = |image: &crate::images::Image,
                subresource: vk::ImageSubresourceLayers,
                offset: vk::Offset3D,
                extent: vk::Extent3D| {
        let image_extent = image.mip_extent(subresource.mip_level)?;
        if subresource.aspect_mask != vk::ImageAspectFlags::COLOR
            || subresource.mip_level != 0
            || subresource.base_array_layer != 0
            || subresource.layer_count != 1
            || offset.z != 0
            || extent.depth != 1
        {
            return Err(unsupported);
        }
        let x = u32::try_from(offset.x).map_err(|_| invalid)?;
        let y = u32::try_from(offset.y).map_err(|_| invalid)?;
        let area = ir::PixelRect::new(x, y, extent.width, extent.height).map_err(|_| invalid)?;
        if !area.is_within(
            ir::Extent2D::new(image_extent.width, image_extent.height).map_err(|_| invalid)?,
        ) {
            return Err(invalid);
        }
        Ok(area)
    };
    let rectangles = regions
        .iter()
        .map(|region| {
            Ok((
                rect(
                    src,
                    region.src_subresource,
                    region.src_offset,
                    region.extent,
                )?,
                rect(
                    dst,
                    region.dst_subresource,
                    region.dst_offset,
                    region.extent,
                )?,
            ))
        })
        .collect::<Result<Vec<_>, vk::Result>>()?;
    let overlap = |a: ir::PixelRect, b: ir::PixelRect| {
        a.x() < b.x() + b.width()
            && b.x() < a.x() + a.width()
            && a.y() < b.y() + b.height()
            && b.y() < a.y() + a.height()
    };
    // Vulkan prohibits destination overlap and any source/destination overlap
    // across the complete region list, not merely within each individual pair.
    for (i, &(_, destination)) in rectangles.iter().enumerate() {
        if rectangles[..i]
            .iter()
            .any(|&(_, other)| overlap(destination, other))
        {
            return Err(invalid);
        }
    }
    if source == destination {
        if source_layout != vk::ImageLayout::GENERAL
            || destination_layout != vk::ImageLayout::GENERAL
            || rectangles
                .iter()
                .any(|&(a, _)| rectangles.iter().any(|&(_, b)| overlap(a, b)))
        {
            return Err(invalid);
        }
        // GENERAL read/write barriers cannot currently express both access
        // roles on one canonical texture. Do not silently ignore that contract.
        return Err(unsupported);
    }
    if let (Some((a, offset_a)), Some((b, offset_b))) = (src.bound, dst.bound) {
        if a == b
            && offset_a < offset_b.saturating_add(dst.byte_size())
            && offset_b < offset_a.saturating_add(src.byte_size())
        {
            // Distinct aliased images do not yet share physical SGFX storage.
            return Err(unsupported);
        }
    }
    Ok(rectangles
        .into_iter()
        .map(
            |(source_rect, destination_rect)| ir::OwnedCommand::CopyTextureToTexture {
                source: src.id,
                source_rect,
                destination: dst.id,
                destination_rect,
            },
        )
        .collect())
}

/// Lower bounded color mip blits, flips and narrow-to-RGBA expansion into IR.
/// Array layers remain outside this subset.
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
    if !crate::images::texture_format(src.format)
        .zip(crate::images::texture_format(dst.format))
        .is_some_and(|(a, b)| a.blit_compatible(b))
    {
        return Err(unsupported);
    }
    let mut commands = Vec::with_capacity(regions.len());
    for region in regions {
        if matches!(src.format, vk::Format::R8_UNORM | vk::Format::R8G8_UNORM)
            && matches!(
                dst.format,
                vk::Format::R8G8B8A8_UNORM | vk::Format::B8G8R8A8_UNORM
            )
            && region.dst_subresource.mip_level != 0
        {
            return Err(unsupported);
        }
        let rect = |image: &crate::images::Image,
                    subresource: vk::ImageSubresourceLayers,
                    offsets: [vk::Offset3D; 2]| {
            let extent = image.mip_extent(subresource.mip_level)?;
            if subresource.aspect_mask != vk::ImageAspectFlags::COLOR
                || subresource.base_array_layer != 0
                || subresource.layer_count != 1
                || image.array_layers != 1
                || offsets[0].z != 0
                || offsets[1].z != 1
                || offsets.iter().any(|o| o.x < 0 || o.y < 0)
                || offsets[1].x == offsets[0].x
                || offsets[1].y == offsets[0].y
            {
                return Err(unsupported);
            }
            let rect = ir::PixelRect::new(
                offsets[0].x.min(offsets[1].x) as u32,
                offsets[0].y.min(offsets[1].y) as u32,
                offsets[1].x.abs_diff(offsets[0].x),
                offsets[1].y.abs_diff(offsets[0].y),
            )
            .map_err(crate::resources::failure)?;
            if !rect.is_within(
                ir::Extent2D::new(extent.width, extent.height)
                    .map_err(crate::resources::failure)?,
            ) {
                return Err(invalid);
            }
            Ok(rect)
        };
        let source_rect = rect(src, region.src_subresource, region.src_offsets)?;
        let destination_rect = rect(dst, region.dst_subresource, region.dst_offsets)?;
        if source == destination
            && region.src_subresource.mip_level == region.dst_subresource.mip_level
        {
            return Err(invalid);
        }
        commands.push(ir::OwnedCommand::BlitTextureRegion {
            source: src.id,
            source_mip: region.src_subresource.mip_level,
            source_rect,
            destination_rect,
            destination: dst.id,
            destination_mip: region.dst_subresource.mip_level,
            filter,
            flips: [
                (region.src_offsets[0].x > region.src_offsets[1].x)
                    != (region.dst_offsets[0].x > region.dst_offsets[1].x),
                (region.src_offsets[0].y > region.src_offsets[1].y)
                    != (region.dst_offsets[0].y > region.dst_offsets[1].y),
            ],
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
        .bytes_per_pixel()
        .ok_or(unsupported)?;
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

#[cfg(test)]
mod readback_tests {
    use super::*;

    fn region(x: i32, y: i32, width: u32, height: u32) -> vk::BufferImageCopy {
        vk::BufferImageCopy::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .image_offset(vk::Offset3D { x, y, z: 0 })
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
    }

    fn buffer_id() -> ir::BufferId {
        ir::ResourceTable::new()
            .define_buffer(ir::BufferDesc::new(256, ir::BufferUsage::COPY_DST).unwrap())
            .unwrap()
            .id()
    }

    #[test]
    fn partial_image_readback_preserves_offset_and_row_padding_for_r_rg_rgba() {
        for bpp in [1u32, 2, 4] {
            let extent = vk::Extent3D {
                width: 4,
                height: 3,
                depth: 1,
            };
            let copy = region(1, 1, 2, 2)
                .buffer_offset(u64::from(bpp))
                .buffer_row_length(4)
                .buffer_image_height(7);
            let layout = ImageReadbackRegion::new(extent, bpp, u64::from(12 * bpp), &copy).unwrap();
            let pixels: Vec<_> = (0..12 * bpp).map(|byte| byte as u8).collect();
            let writes = layout.writes(&pixels, buffer_id()).unwrap();
            assert_eq!(writes.len(), 2);
            let mut output = vec![0xcd; (12 * bpp) as usize];
            for write in writes {
                let ir::OwnedCommand::WriteBuffer { offset, data, .. } = write else {
                    panic!("buffer write expected");
                };
                output[offset as usize..offset as usize + data.len()].copy_from_slice(&data);
            }
            let mut expected = vec![0xcd; (12 * bpp) as usize];
            for (dst, src) in [(bpp, 5 * bpp), (5 * bpp, 9 * bpp)] {
                expected[dst as usize..(dst + 2 * bpp) as usize]
                    .copy_from_slice(&pixels[src as usize..(src + 2 * bpp) as usize]);
            }
            assert_eq!(output, expected);
        }
    }

    #[test]
    fn partial_image_readback_single_pixel_and_contiguous_rows_are_exact() {
        let extent = vk::Extent3D {
            width: 3,
            height: 2,
            depth: 1,
        };
        let pixels: Vec<_> = (0..24).collect();
        let layout =
            ImageReadbackRegion::new(extent, 4, 8, &region(2, 1, 1, 1).buffer_offset(4)).unwrap();
        let writes = layout.writes(&pixels, buffer_id()).unwrap();
        assert!(
            matches!(writes.as_slice(), [ir::OwnedCommand::WriteBuffer { offset: 4, data, .. }] if data == &[20,21,22,23])
        );
        let layout = ImageReadbackRegion::new(extent, 4, 12, &region(0, 1, 3, 1)).unwrap();
        let writes = layout.writes(&pixels, buffer_id()).unwrap();
        assert!(
            matches!(writes.as_slice(), [ir::OwnedCommand::WriteBuffer { offset: 0, data, .. }] if data == &pixels[12..])
        );
    }

    #[test]
    fn partial_image_readback_rejects_bounds_alignment_overflow_and_short_gpu_data() {
        let extent = vk::Extent3D {
            width: 4,
            height: 3,
            depth: 1,
        };
        for invalid in [
            region(-1, 0, 1, 1),
            region(4, 0, 1, 1),
            region(0, 0, 0, 1),
            region(0, 2, 1, 2),
            region(0, 0, 2, 2).buffer_row_length(1),
            region(0, 0, 2, 2).buffer_image_height(1),
            region(0, 0, 1, 1).buffer_offset(1),
            region(0, 0, 1, 1).buffer_offset(u64::MAX - 3),
        ] {
            assert!(ImageReadbackRegion::new(extent, 4, u64::MAX, &invalid).is_err());
        }
        assert!(ImageReadbackRegion::new(extent, 4, 3, &region(0, 0, 1, 1)).is_err());
        let layout = ImageReadbackRegion::new(extent, 4, 4, &region(0, 0, 1, 1)).unwrap();
        assert!(matches!(
            layout.writes(&[0; 4], buffer_id()),
            Err(vk::Result::ERROR_DEVICE_LOST)
        ));
    }
}

#[cfg(test)]
mod copy_image_tests {
    use super::*;
    use ash::vk::Handle;
    fn src_handle() -> vk::Image {
        vk::Image::from_raw(101)
    }
    fn dst_handle() -> vk::Image {
        vk::Image::from_raw(102)
    }
    fn resources() -> Resources {
        let table = ir::ResourceTable::new();
        let mut resources = Resources::new();
        for (handle, memory) in [(src_handle(), 201), (dst_handle(), 202)] {
            let id = table
                .define_texture(
                    ir::TextureDesc::new(
                        ir::TextureFormat::Bgra8Unorm,
                        ir::Extent2D::new(8, 8).unwrap(),
                        ir::TextureUsage::COPY_SRC | ir::TextureUsage::COPY_DST,
                    )
                    .unwrap(),
                )
                .unwrap()
                .id();
            resources.images.insert(
                handle,
                crate::images::Image {
                    id,
                    format: vk::Format::B8G8R8A8_UNORM,
                    extent: vk::Extent3D {
                        width: 8,
                        height: 8,
                        depth: 1,
                    },
                    mip_levels: 1,
                    array_layers: 1,
                    flags: vk::ImageCreateFlags::empty(),
                    usage: vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
                    bound: Some((vk::DeviceMemory::from_raw(memory), 0)),
                    swapchain: None,
                    #[cfg(target_os = "scarlet")]
                    shared: None,
                },
            );
        }
        resources
    }
    fn region(sx: i32, sy: i32, dx: i32, dy: i32, width: u32, height: u32) -> vk::ImageCopy {
        let sub = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1);
        vk::ImageCopy::default()
            .src_subresource(sub)
            .dst_subresource(sub)
            .src_offset(vk::Offset3D { x: sx, y: sy, z: 0 })
            .dst_offset(vk::Offset3D { x: dx, y: dy, z: 0 })
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
    }
    fn copy(
        resources: &Resources,
        regions: &[vk::ImageCopy],
    ) -> Result<Vec<ir::OwnedCommand>, vk::Result> {
        copy_image(
            resources,
            src_handle(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            dst_handle(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            regions,
        )
    }
    #[test]
    fn image_copy_emits_exact_rectangles_without_sampling_or_blending() {
        let resources = resources();
        let ops = copy(
            &resources,
            &[region(1, 2, 4, 3, 3, 2), region(0, 0, 0, 0, 1, 1)],
        )
        .unwrap();
        assert_eq!(ops.len(), 2);
        match &ops[0] {
            ir::OwnedCommand::CopyTextureToTexture {
                source,
                source_rect,
                destination,
                destination_rect,
            } => {
                assert_eq!(*source, resources.images[&src_handle()].id);
                assert_eq!(*destination, resources.images[&dst_handle()].id);
                assert_eq!(*source_rect, ir::PixelRect::new(1, 2, 3, 2).unwrap());
                assert_eq!(*destination_rect, ir::PixelRect::new(4, 3, 3, 2).unwrap());
            }
            _ => panic!("image copy must not use a blit/draw/upload"),
        }
    }
    #[test]
    fn image_copy_rejects_invalid_regions_usage_and_overlapping_destinations() {
        let invalid = vk::Result::ERROR_INITIALIZATION_FAILED;
        assert!(
            matches!(copy_image(&resources(), src_handle(), vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            dst_handle(), vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region(0,0,0,0,1,1)]), Err(e) if e==invalid)
        );
        assert!(
            copy_image(
                &resources(),
                src_handle(),
                vk::ImageLayout::GENERAL,
                dst_handle(),
                vk::ImageLayout::GENERAL,
                &[region(0, 0, 0, 0, 1, 1)]
            )
            .is_ok()
        );
        for region in [
            region(-1, 0, 0, 0, 1, 1),
            region(0, -1, 0, 0, 1, 1),
            region(0, 0, -1, 0, 1, 1),
            region(0, 0, 0, 0, 0, 1),
            region(0, 0, 0, 0, 1, 0),
            region(7, 0, 0, 0, 2, 1),
            region(0, 0, 7, 0, 2, 1),
            region(0, 0, 1, 0, u32::MAX, 1),
        ] {
            assert!(matches!(copy(&resources(),&[region]),Err(e) if e==invalid));
        }
        assert!(matches!(copy(&resources(),&[]),Err(e) if e==invalid));
        assert!(
            matches!(copy(&resources(),&[region(0,0,2,2,2,2),region(4,4,3,3,2,2)]),Err(e) if e==invalid)
        );
        let mut resources = resources();
        resources.images.get_mut(&src_handle()).unwrap().usage = vk::ImageUsageFlags::TRANSFER_DST;
        assert!(matches!(copy(&resources,&[region(0,0,0,0,1,1)]),Err(e) if e==invalid));
    }
    #[test]
    fn image_copy_reports_unknown_subresources_formats_aliases_and_self_access() {
        let unsupported = vk::Result::ERROR_FEATURE_NOT_PRESENT;
        let sub = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1);
        for region in [
            region(0, 0, 0, 0, 1, 1).src_subresource(sub.base_array_layer(1)),
            region(0, 0, 0, 0, 1, 1).dst_subresource(sub.layer_count(2)),
            region(0, 0, 0, 0, 1, 1).src_subresource(sub.aspect_mask(vk::ImageAspectFlags::DEPTH)),
            region(0, 0, 0, 0, 1, 1).extent(vk::Extent3D {
                width: 1,
                height: 1,
                depth: 2,
            }),
        ] {
            assert!(matches!(copy(&resources(),&[region]),Err(e) if e==unsupported));
        }
        let mut resources = resources();
        resources.images.get_mut(&src_handle()).unwrap().mip_levels = 2;
        assert!(
            matches!(copy(&resources,&[region(0,0,0,0,1,1).src_subresource(sub.mip_level(1))]),Err(e) if e==unsupported)
        );
        resources.images.get_mut(&dst_handle()).unwrap().format = vk::Format::R8G8B8A8_UNORM;
        assert!(matches!(copy(&resources,&[region(0,0,0,0,1,1)]),Err(e) if e==unsupported));
        resources.images.get_mut(&dst_handle()).unwrap().format = vk::Format::B8G8R8A8_UNORM;
        resources.images.get_mut(&dst_handle()).unwrap().bound =
            resources.images[&src_handle()].bound;
        assert!(matches!(copy(&resources,&[region(0,0,0,0,1,1)]),Err(e) if e==unsupported));
        assert!(
            matches!(copy_image(&resources,src_handle(),vk::ImageLayout::GENERAL,src_handle(),vk::ImageLayout::GENERAL,&[region(0,0,4,0,2,2)]),Err(e) if e==unsupported)
        );
        // Source of the second region overlaps the first destination: cross-
        // region hazard must be rejected, not accepted as two disjoint pairs.
        assert!(matches!(
            copy_image(
                &resources,
                src_handle(),
                vk::ImageLayout::GENERAL,
                src_handle(),
                vk::ImageLayout::GENERAL,
                &[region(0, 0, 4, 0, 2, 2), region(4, 0, 0, 4, 2, 2)]
            ),
            Err(vk::Result::ERROR_INITIALIZATION_FAILED)
        ));
    }
}
