//! Bound uploads in transfer and mixed recordings without splitting pass state.

use sgfx::ir::{self, OwnedCommand, OwnedCommandBuffer};

// A pacing budget for uploads, independent of native packet encoding.
// Leaves room for resource/transfer metadata in bounded backend schedulers.
pub(crate) const UPLOAD_BATCH_BYTES: usize = 8 * 1024 * 1024;

fn upload_bytes(command: &OwnedCommand) -> Option<usize> {
    match command {
        OwnedCommand::WriteBuffer { data, .. }
        | OwnedCommand::WriteTexture { data, .. }
        | OwnedCommand::WriteTextureMip { data, .. }
        | OwnedCommand::WriteTextureLayer { data, .. } => Some(data.len()),
        OwnedCommand::ResourceBarrier(
            ir::OwnedResourceBarrier::Texture { .. } | ir::OwnedResourceBarrier::TextureMip { .. },
        )
        | OwnedCommand::CopyTextureToTexture { .. }
        | OwnedCommand::BlitTexture { .. }
        | OwnedCommand::BlitTextureRegion { .. } => Some(0),
        _ => None,
    }
}

pub(crate) fn needs_batches(commands: &[OwnedCommand], budget: usize) -> bool {
    commands.iter().fold(0usize, |total, command| {
        total.saturating_add(upload_bytes(command).unwrap_or(0))
    }) > budget
}

/// Validate the entire recording before admitting any prefix. Preserve command
/// order and resource identities, and split only outside render/compute passes.
pub(crate) fn split(
    owned: OwnedCommandBuffer,
    table: &ir::ResourceTable,
    budget: usize,
) -> ir::Result<Vec<OwnedCommandBuffer>> {
    if budget == 0 {
        return Err(ir::Error::InvalidValue);
    }
    owned.validate(table)?;
    if !needs_batches(owned.commands(), budget) {
        return Ok(vec![owned]);
    }
    let mut batches = Vec::new();
    let mut batch = Vec::new();
    let mut bytes = 0usize;
    let mut in_pass = false;
    for command in owned.into_commands() {
        match &command {
            OwnedCommand::BeginRenderPass(_)
            | OwnedCommand::BeginRenderPassWithAttachments { .. }
            | OwnedCommand::BeginComputePass => {
                if bytes >= budget && !batch.is_empty() {
                    batches.push(OwnedCommandBuffer::new(std::mem::take(&mut batch)));
                    bytes = 0;
                }
                in_pass = true;
                batch.push(command);
                continue;
            }
            OwnedCommand::EndRenderPass | OwnedCommand::EndComputePass => {
                in_pass = false;
                batch.push(command);
                continue;
            }
            _ if in_pass => {
                batch.push(command);
                continue;
            }
            _ => {}
        }
        let length = upload_bytes(&command).unwrap_or(0);
        if length <= budget {
            push(&mut batches, &mut batch, &mut bytes, command, budget);
            continue;
        }
        let command = match command {
            OwnedCommand::WriteBuffer {
                buffer,
                offset,
                data,
            } => {
                for (index, chunk) in data.chunks(budget).enumerate() {
                    push(
                        &mut batches,
                        &mut batch,
                        &mut bytes,
                        OwnedCommand::WriteBuffer {
                            buffer,
                            offset: offset + (index * budget) as u64,
                            data: chunk.to_vec(),
                        },
                        budget,
                    );
                }
                continue;
            }
            command => command,
        };
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
            _ => unreachable!("oversized upload"),
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
    let length = upload_bytes(&command).unwrap_or(0);
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
                    ir::TextureUsage::COPY_DST
                        | ir::TextureUsage::COPY_SRC
                        | ir::TextureUsage::SAMPLED,
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
    fn mip_uploads_keep_barriers_and_dependent_blits_in_queue_order() {
        let table = ir::ResourceTable::new();
        let texture = table
            .define_texture(
                ir::TextureDesc::new(
                    ir::TextureFormat::Rgba8Unorm,
                    ir::Extent2D::new(32, 32).unwrap(),
                    ir::TextureUsage::COPY_DST
                        | ir::TextureUsage::COPY_SRC
                        | ir::TextureUsage::SAMPLED,
                )
                .unwrap()
                .with_mip_level_count(2)
                .unwrap(),
            )
            .unwrap()
            .id();
        let commands = vec![
            OwnedCommand::WriteTexture {
                texture,
                destination: ir::PixelRect::new(0, 0, 32, 32).unwrap(),
                bytes_per_row: 128,
                data: vec![9; 4096],
            },
            OwnedCommand::ResourceBarrier(ir::OwnedResourceBarrier::TextureMip {
                texture,
                mip_level: 0,
                before: ir::TextureAccess::CopyDestination,
                after: ir::TextureAccess::CopySource,
            }),
            OwnedCommand::BlitTexture {
                source: texture,
                source_mip: 0,
                destination: texture,
                destination_mip: 1,
                filter: ir::FilterMode::Linear,
            },
            OwnedCommand::WriteTextureMip {
                texture,
                mip_level: 1,
                destination: ir::PixelRect::new(0, 0, 16, 16).unwrap(),
                bytes_per_row: 64,
                data: vec![3; 1024],
            },
            OwnedCommand::ResourceBarrier(ir::OwnedResourceBarrier::TextureMip {
                texture,
                mip_level: 1,
                before: ir::TextureAccess::CopyDestination,
                after: ir::TextureAccess::Sampled,
            }),
        ];
        assert!(needs_batches(&commands, 1024));
        let batches = split(OwnedCommandBuffer::new(commands), &table, 1024).unwrap();
        let mut order = Vec::new();
        for batch in &batches {
            batch.validate(&table).unwrap();
            assert!(
                batch
                    .commands()
                    .iter()
                    .map(|c| upload_bytes(c).unwrap())
                    .sum::<usize>()
                    <= 1024
            );
            for command in batch.commands() {
                order.push(match command {
                    OwnedCommand::WriteTextureLayer { mip_level: 0, .. } => "base",
                    OwnedCommand::WriteTextureMip { mip_level: 1, .. } => "overwrite",
                    OwnedCommand::ResourceBarrier(ir::OwnedResourceBarrier::TextureMip {
                        mip_level: 0,
                        ..
                    }) => "source-barrier",
                    OwnedCommand::ResourceBarrier(ir::OwnedResourceBarrier::TextureMip {
                        mip_level: 1,
                        ..
                    }) => "sample-barrier",
                    OwnedCommand::BlitTexture { .. } => "blit",
                    _ => panic!("unexpected transfer"),
                });
            }
        }
        assert_eq!(
            order,
            [
                "base",
                "base",
                "base",
                "base",
                "source-barrier",
                "blit",
                "overwrite",
                "sample-barrier"
            ]
        );
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
    fn small_uploads_keep_their_allocation_and_invalid_mixed_work_is_rejected() {
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
        let invalid = vec![
            batches[0].commands()[0].clone(),
            OwnedCommand::EndRenderPass,
        ];
        assert!(needs_batches(&invalid, 8));
        assert!(split(OwnedCommandBuffer::new(invalid), &table, 8).is_err());
    }

    fn render_pass(table: &ir::ResourceTable) -> Vec<OwnedCommand> {
        let target = table
            .define_texture(
                ir::TextureDesc::new(
                    ir::TextureFormat::Rgba8Unorm,
                    ir::Extent2D::new(2, 2).unwrap(),
                    ir::TextureUsage::RENDER_ATTACHMENT,
                )
                .unwrap(),
            )
            .unwrap()
            .id();
        vec![
            OwnedCommand::BeginRenderPass(ir::OwnedRenderPassDesc {
                target,
                area: ir::PixelRect::new(0, 0, 2, 2).unwrap(),
                load: ir::LoadOp::DontCare,
                store: ir::StoreOp::Store,
                depth: None,
            }),
            OwnedCommand::SetScissor(Some(ir::PixelRect::new(0, 0, 1, 1).unwrap())),
            OwnedCommand::EndRenderPass,
        ]
    }

    #[test]
    fn mixed_uploads_preserve_complete_passes_and_buffer_ranges() {
        let table = ir::ResourceTable::new();
        let texture = texture(&table);
        let buffer = table
            .define_buffer(ir::BufferDesc::new(80, ir::BufferUsage::COPY_DST).unwrap())
            .unwrap()
            .id();
        let mut commands = vec![OwnedCommand::WriteTexture {
            texture,
            destination: ir::PixelRect::new(0, 0, 4, 4).unwrap(),
            bytes_per_row: 16,
            data: (0..64).collect(),
        }];
        commands.extend(render_pass(&table));
        commands.push(OwnedCommand::WriteBuffer {
            buffer,
            offset: 4,
            data: (0..64).collect(),
        });
        commands.extend([OwnedCommand::BeginComputePass, OwnedCommand::EndComputePass]);
        commands.extend(render_pass(&table));
        assert!(needs_batches(&commands, 24));
        let batches = split(OwnedCommandBuffer::new(commands), &table, 24).unwrap();
        let mut pixels: Vec<u8> = Vec::new();
        let mut buffer_bytes: Vec<u8> = Vec::new();
        let mut passes = 0;
        let mut compute = 0;
        for batch in batches {
            batch.validate(&table).unwrap();
            assert!(
                batch
                    .commands()
                    .iter()
                    .map(|c| upload_bytes(c).unwrap_or(0))
                    .sum::<usize>()
                    <= 24
            );
            let mut active = false;
            for command in batch.commands() {
                match command {
                    OwnedCommand::WriteTextureLayer { data, .. } => pixels.extend(data),
                    OwnedCommand::WriteBuffer {
                        buffer: id,
                        offset,
                        data,
                    } => {
                        assert_eq!(*id, buffer);
                        assert_eq!(*offset, 4 + buffer_bytes.len() as u64);
                        buffer_bytes.extend(data);
                    }
                    OwnedCommand::BeginRenderPass(_) => {
                        assert!(!active);
                        active = true;
                        passes += 1;
                    }
                    OwnedCommand::SetScissor(_) => assert!(active),
                    OwnedCommand::EndRenderPass => {
                        assert!(active);
                        active = false;
                    }
                    OwnedCommand::BeginComputePass => {
                        assert!(!active);
                        active = true;
                        compute += 1;
                    }
                    OwnedCommand::EndComputePass => {
                        assert!(active);
                        active = false;
                    }
                    _ => panic!("unexpected command"),
                }
            }
            assert!(!active);
        }
        assert_eq!(passes, 2);
        assert_eq!(compute, 1);
        assert_eq!(pixels, (0..64).collect::<Vec<u8>>());
        assert_eq!(buffer_bytes, (0..64).collect::<Vec<u8>>());
    }

    #[test]
    fn page_uploads_over_backend_admission_are_batched_even_with_rendering() {
        let table = ir::ResourceTable::new();
        let texture = table
            .define_texture(
                ir::TextureDesc::new(
                    ir::TextureFormat::Rgba8Unorm,
                    ir::Extent2D::new(2048, 2048).unwrap(),
                    ir::TextureUsage::COPY_DST,
                )
                .unwrap(),
            )
            .unwrap()
            .id();
        let mut commands = Vec::new();
        for value in 0..5 {
            commands.push(OwnedCommand::WriteTexture {
                texture,
                destination: ir::PixelRect::new(0, 0, 2048, 2048).unwrap(),
                bytes_per_row: 8192,
                data: vec![value; 16 * 1024 * 1024],
            });
            commands.extend(render_pass(&table));
        }
        assert!(needs_batches(&commands, UPLOAD_BATCH_BYTES));
        let batches = split(
            OwnedCommandBuffer::new(commands),
            &table,
            UPLOAD_BATCH_BYTES,
        )
        .unwrap();
        let mut total = 0;
        let mut passes = 0;
        for batch in batches {
            batch.validate(&table).unwrap();
            let bytes: usize = batch
                .commands()
                .iter()
                .map(|c| upload_bytes(c).unwrap_or(0))
                .sum();
            assert!(bytes <= UPLOAD_BATCH_BYTES);
            total += bytes;
            passes += batch
                .commands()
                .iter()
                .filter(|c| matches!(c, OwnedCommand::BeginRenderPass(_)))
                .count();
        }
        assert_eq!(total, 80 * 1024 * 1024);
        assert_eq!(passes, 5);
    }
}
