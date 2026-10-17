//! Bound texture-only transfer recordings without splitting render-pass state.

use sgfx::ir::{self, OwnedCommand, OwnedCommandBuffer};

// A pacing budget for upload-only work, independent of native packet encoding.
// Leaves room for resource/transfer metadata in bounded backend schedulers.
pub(crate) const UPLOAD_BATCH_BYTES: usize = 8 * 1024 * 1024;

fn upload_bytes(command: &OwnedCommand) -> Option<usize> {
    match command {
        OwnedCommand::WriteTexture { data, .. }
        | OwnedCommand::WriteTextureMip { data, .. }
        | OwnedCommand::WriteTextureLayer { data, .. } => Some(data.len()),
        _ => None,
    }
}

pub(crate) fn needs_batches(commands: &[OwnedCommand], budget: usize) -> bool {
    commands
        .iter()
        .try_fold(0usize, |total, command| {
            upload_bytes(command).map(|bytes| total.saturating_add(bytes))
        })
        .is_some_and(|bytes| bytes > budget)
}

/// Validate the entire recording before admitting any prefix. Only independent
/// texture writes are eligible; ordering and all mip/layer identities survive.
pub(crate) fn split(
    owned: OwnedCommandBuffer,
    table: &ir::ResourceTable,
    budget: usize,
) -> ir::Result<Vec<OwnedCommandBuffer>> {
    owned.validate(table)?;
    if !needs_batches(owned.commands(), budget) {
        return Ok(vec![owned]);
    }
    let mut batches = Vec::new();
    let mut batch = Vec::new();
    let mut bytes = 0usize;
    for command in owned.into_commands() {
        let length = upload_bytes(&command).expect("texture-only recording");
        if length <= budget {
            push(&mut batches, &mut batch, &mut bytes, command, budget);
            continue;
        }
        let (texture, mip_level, array_layer, rect, stride, data) = match command {
            OwnedCommand::WriteTexture {
                texture,
                destination,
                bytes_per_row,
                data,
            } => (texture, 0, 0, destination, bytes_per_row, data),
            OwnedCommand::WriteTextureMip {
                texture,
                mip_level,
                destination,
                bytes_per_row,
                data,
            } => (texture, mip_level, 0, destination, bytes_per_row, data),
            OwnedCommand::WriteTextureLayer {
                texture,
                mip_level,
                array_layer,
                destination,
                bytes_per_row,
                data,
            } => (
                texture,
                mip_level,
                array_layer,
                destination,
                bytes_per_row,
                data,
            ),
            _ => unreachable!("texture-only recording"),
        };
        let pixel_bytes = table
            .texture(table.texture_ref(texture)?)?
            .format()
            .bytes_per_pixel()
            .ok_or(ir::Error::InvalidValue)? as usize;
        let row_bytes = rect.width() as usize * pixel_bytes;
        // Physical image limits keep a tight row below the upload budget. Do
        // not let arbitrary source padding turn one row into a huge transfer.
        if row_bytes > budget {
            return Err(ir::Error::InvalidValue);
        }
        let rows_per_chunk = (budget / row_bytes).max(1);
        for row in (0..rect.height() as usize).step_by(rows_per_chunk) {
            let rows = rows_per_chunk.min(rect.height() as usize - row);
            let mut pixels = Vec::with_capacity(rows * row_bytes);
            for y in row..row + rows {
                let start = y * stride as usize;
                pixels.extend_from_slice(&data[start..start + row_bytes]);
            }
            let destination =
                ir::PixelRect::new(rect.x(), rect.y() + row as u32, rect.width(), rows as u32)?;
            push(
                &mut batches,
                &mut batch,
                &mut bytes,
                OwnedCommand::WriteTextureLayer {
                    texture,
                    mip_level,
                    array_layer,
                    destination,
                    bytes_per_row: row_bytes as u32,
                    data: pixels,
                },
                budget,
            );
        }
    }
    if !batch.is_empty() {
        batches.push(OwnedCommandBuffer::new(batch));
    }
    // Also check transformed row rectangles before the first backend submission.
    for batch in &batches {
        batch.validate(table)?;
    }
    Ok(batches)
}

fn push(
    batches: &mut Vec<OwnedCommandBuffer>,
    batch: &mut Vec<OwnedCommand>,
    bytes: &mut usize,
    command: OwnedCommand,
    budget: usize,
) {
    let length = upload_bytes(&command).expect("texture upload");
    if bytes.saturating_add(length) > budget && !batch.is_empty() {
        batches.push(OwnedCommandBuffer::new(std::mem::take(batch)));
        *bytes = 0;
    }
    *bytes += length;
    batch.push(command);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texture(table: &ir::ResourceTable) -> ir::TextureId {
        table
            .define_texture(
                ir::TextureDesc::new(
                    ir::TextureFormat::Rgba8Unorm,
                    ir::Extent2D::new(32, 32).unwrap(),
                    ir::TextureUsage::COPY_DST,
                )
                .unwrap()
                .with_mip_level_count(2)
                .unwrap()
                .with_array_layer_count(2)
                .unwrap(),
            )
            .unwrap()
            .id()
    }

    fn payload(command: &OwnedCommand) -> (ir::TextureId, u32, u32, ir::PixelRect, u32, &[u8]) {
        match command {
            OwnedCommand::WriteTexture {
                texture,
                destination,
                bytes_per_row,
                data,
            } => (*texture, 0, 0, *destination, *bytes_per_row, data),
            OwnedCommand::WriteTextureMip {
                texture,
                mip_level,
                destination,
                bytes_per_row,
                data,
            } => (*texture, *mip_level, 0, *destination, *bytes_per_row, data),
            OwnedCommand::WriteTextureLayer {
                texture,
                mip_level,
                array_layer,
                destination,
                bytes_per_row,
                data,
            } => (
                *texture,
                *mip_level,
                *array_layer,
                *destination,
                *bytes_per_row,
                data,
            ),
            _ => panic!("not a texture upload"),
        }
    }

    #[test]
    fn padded_subrectangles_keep_mip_layer_pixels_and_overwrite_order() {
        let table = ir::ResourceTable::new();
        let id = texture(&table);
        let mut data = vec![0xee; 5 * 16 + 12];
        for row in 0..6 {
            data[row * 16..row * 16 + 12].fill((row + 1) as u8);
        }
        let commands = vec![
            OwnedCommand::WriteTextureLayer {
                texture: id,
                mip_level: 1,
                array_layer: 1,
                destination: ir::PixelRect::new(2, 3, 3, 6).unwrap(),
                bytes_per_row: 16,
                data,
            },
            OwnedCommand::WriteTextureLayer {
                texture: id,
                mip_level: 1,
                array_layer: 1,
                destination: ir::PixelRect::new(3, 4, 1, 1).unwrap(),
                bytes_per_row: 4,
                data: vec![99; 4],
            },
        ];
        let batches = split(OwnedCommandBuffer::new(commands), &table, 24).unwrap();
        assert!(batches.len() > 1);
        let mut pixels = vec![0; 16 * 16 * 4];
        for batch in &batches {
            batch.validate(&table).unwrap();
            assert!(
                batch
                    .commands()
                    .iter()
                    .map(|c| upload_bytes(c).unwrap())
                    .sum::<usize>()
                    <= 24
            );
            for command in batch.commands() {
                let (texture, mip, layer, rect, stride, data) = payload(command);
                assert_eq!((texture, mip, layer), (id, 1, 1));
                for row in 0..rect.height() as usize {
                    let destination = ((rect.y() as usize + row) * 16 + rect.x() as usize) * 4;
                    let source = row * stride as usize;
                    pixels[destination..destination + rect.width() as usize * 4]
                        .copy_from_slice(&data[source..source + rect.width() as usize * 4]);
                }
            }
        }
        for y in 0..16 {
            for x in 0..16 {
                let expected = if x == 3 && y == 4 {
                    99
                } else if (2..5).contains(&x) && (3..9).contains(&y) {
                    (y - 2) as u8
                } else {
                    0
                };
                assert_eq!(
                    &pixels[(y * 16 + x) * 4..(y * 16 + x + 1) * 4],
                    &[expected; 4]
                );
            }
        }
    }

    #[test]
    fn invalid_late_upload_is_rejected_before_any_batches_are_published() {
        let table = ir::ResourceTable::new();
        let texture = texture(&table);
        let good = OwnedCommand::WriteTexture {
            texture,
            destination: ir::PixelRect::new(0, 0, 2, 2).unwrap(),
            bytes_per_row: 8,
            data: vec![7; 16],
        };
        let bad = OwnedCommand::WriteTextureMip {
            texture,
            mip_level: 1,
            destination: ir::PixelRect::new(15, 15, 2, 2).unwrap(),
            bytes_per_row: 8,
            data: vec![8; 16],
        };
        assert!(split(OwnedCommandBuffer::new(vec![good, bad]), &table, 8).is_err());
    }

    #[test]
    fn mixed_work_is_kept_in_one_recording_and_small_uploads_keep_their_allocation() {
        let table = ir::ResourceTable::new();
        let id = texture(&table);
        let data = vec![3; 16];
        let address = data.as_ptr();
        let commands = vec![OwnedCommand::WriteTextureMip {
            texture: id,
            mip_level: 1,
            destination: ir::PixelRect::new(0, 0, 2, 2).unwrap(),
            bytes_per_row: 8,
            data,
        }];
        let batches = split(OwnedCommandBuffer::new(commands), &table, 32).unwrap();
        assert_eq!(payload(&batches[0].commands()[0]).5.as_ptr(), address);
        assert!(!needs_batches(
            &[
                batches[0].commands()[0].clone(),
                OwnedCommand::EndRenderPass
            ],
            8
        ));
    }
}
