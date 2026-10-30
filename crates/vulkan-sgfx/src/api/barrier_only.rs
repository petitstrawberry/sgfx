//! Ordinary single-queue dependencies have no physical operation in WGPU or
//! native VirGL. Replay must first check ownership, usage, mip bounds, pass
//! scope and consecutive accesses. Then inspect the original owned commands,
//! avoiding compatibility decoding of native ABI command storage.
use sgfx::ir;

pub(super) fn is_noop(commands: &[ir::OwnedCommand]) -> bool {
    !commands.is_empty() && commands.iter().all(ordinary_barrier)
}

fn ordinary_barrier(command: &ir::OwnedCommand) -> bool {
    match command {
        ir::OwnedCommand::ResourceBarrier(ir::OwnedResourceBarrier::Buffer {
            before,
            after,
            ..
        }) => ordinary_buffer_access(*before) && ordinary_buffer_access(*after),
        ir::OwnedCommand::ResourceBarrier(
            ir::OwnedResourceBarrier::Texture { before, after, .. }
            | ir::OwnedResourceBarrier::TextureMip { before, after, .. },
        ) => ordinary_texture_access(*before) && ordinary_texture_access(*after),
        _ => false,
    }
}

fn ordinary_buffer_access(access: ir::BufferAccess) -> bool {
    matches!(
        access,
        ir::BufferAccess::CopyDestination
            | ir::BufferAccess::CopySource
            | ir::BufferAccess::Vertex
            | ir::BufferAccess::Index
            | ir::BufferAccess::Uniform
    )
}

fn ordinary_texture_access(access: ir::TextureAccess) -> bool {
    // Core replay accepts storage transitions that native validate_barrier
    // rejects. Leave storage (and any future access scope) on the backend path.
    matches!(
        access,
        ir::TextureAccess::CopyDestination
            | ir::TextureAccess::CopySource
            | ir::TextureAccess::Sampled
            | ir::TextureAccess::RenderAttachment
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ir::{BufferAccess as B, OwnedCommand as C, OwnedResourceBarrier as R, TextureAccess as T};

    fn buffer(table: &ir::ResourceTable) -> ir::BufferId {
        table
            .define_buffer(
                ir::BufferDesc::new(
                    16,
                    ir::BufferUsage::COPY_SRC
                        | ir::BufferUsage::COPY_DST
                        | ir::BufferUsage::VERTEX
                        | ir::BufferUsage::INDEX
                        | ir::BufferUsage::UNIFORM
                        | ir::BufferUsage::STORAGE,
                )
                .unwrap(),
            )
            .unwrap()
            .id()
    }

    fn texture(table: &ir::ResourceTable) -> ir::TextureId {
        table
            .define_texture(
                ir::TextureDesc::new(
                    ir::TextureFormat::Rgba8Unorm,
                    ir::Extent2D::new(4, 4).unwrap(),
                    ir::TextureUsage::COPY_SRC
                        | ir::TextureUsage::COPY_DST
                        | ir::TextureUsage::SAMPLED
                        | ir::TextureUsage::RENDER_ATTACHMENT
                        | ir::TextureUsage::STORAGE,
                )
                .unwrap()
                .with_mip_level_count(2)
                .unwrap(),
            )
            .unwrap()
            .id()
    }

    fn classify(table: &ir::ResourceTable, ops: Vec<C>) -> ir::Result<bool> {
        let owned = ir::OwnedCommandBuffer::new(ops);
        owned.record(table)?;
        Ok(is_noop(owned.commands()))
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn abi_recordings_classify_owned_barriers_and_preserve_valid_write_bind_and_draw() {
        let table = ir::ResourceTable::new();
        table.enable_abi_commands();
        let buffer = buffer(&table);
        let texture = texture(&table);
        let check = |ops: Vec<C>, expected| {
            let owned = ir::OwnedCommandBuffer::new(ops);
            let recorded = owned.record(&table).unwrap();
            assert!(recorded.abi_batch().is_some());
            assert!(!recorded.is_empty());
            // Do not call recorded.commands(): that lazily decodes the ABI
            // recording and allocates a compatibility Vec. Owned IR is intact.
            assert!(!owned.commands().is_empty());
            assert_eq!(is_noop(owned.commands()), expected);
        };
        let barrier = C::ResourceBarrier(R::Buffer {
            buffer,
            before: B::CopyDestination,
            after: B::CopySource,
        });
        check(vec![barrier.clone()], true);
        check(
            vec![
                barrier.clone(),
                C::ResourceBarrier(R::TextureMip {
                    texture,
                    mip_level: 1,
                    before: T::RenderAttachment,
                    after: T::Sampled,
                }),
            ],
            true,
        );
        check(
            vec![C::WriteBuffer {
                buffer,
                offset: 0,
                data: vec![7; 4],
            }],
            false,
        );
        check(
            vec![
                C::WriteBuffer {
                    buffer,
                    offset: 0,
                    data: vec![7; 4],
                },
                barrier.clone(),
            ],
            false,
        );

        let shader = table
            .define_shader_module(
                ir::ShaderModuleDesc::wgsl(
                    "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }
             @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }"
                        .into(),
                )
                .unwrap(),
            )
            .unwrap();
        let pipeline = table
            .define_programmable_render_pipeline(
                ir::ProgrammableRenderPipelineDesc::new(
                    ir::ShaderEntryPoint::new(shader, ir::ShaderStage::Vertex, "vs".into())
                        .unwrap(),
                    ir::ShaderEntryPoint::new(shader, ir::ShaderStage::Fragment, "fs".into())
                        .unwrap(),
                    ir::PipelineLayoutDesc::new(vec![]).unwrap(),
                    ir::TextureFormat::Rgba8Unorm,
                    None,
                    ir::PrimitiveTopology::TriangleList,
                    ir::BlendState::REPLACE,
                    ir::RasterState::new(ir::CullMode::None, ir::FrontFace::CounterClockwise),
                )
                .unwrap(),
            )
            .unwrap()
            .id();
        let pass = C::BeginRenderPass(ir::OwnedRenderPassDesc {
            target: texture,
            area: ir::PixelRect::new(0, 0, 4, 4).unwrap(),
            load: ir::LoadOp::DontCare,
            store: ir::StoreOp::Store,
            depth: None,
        });
        check(
            vec![
                pass.clone(),
                C::SetProgrammablePipeline(pipeline),
                C::EndRenderPass,
            ],
            false,
        );
        check(
            vec![
                pass,
                C::SetProgrammablePipeline(pipeline),
                C::Draw {
                    vertex_count: 3,
                    first_vertex: 0,
                },
                C::EndRenderPass,
            ],
            false,
        );
        for (before, after) in [
            (T::StorageWrite, T::Sampled),
            (T::CopyDestination, T::StorageWrite),
        ] {
            for storage in [
                R::Texture {
                    texture,
                    before,
                    after,
                },
                R::TextureMip {
                    texture,
                    mip_level: 1,
                    before,
                    after,
                },
            ] {
                check(vec![barrier.clone(), C::ResourceBarrier(storage)], false);
            }
        }
    }

    #[test]
    fn ordinary_access_pairs_and_multiple_barriers_need_no_backend_work() {
        let table = ir::ResourceTable::new();
        let buffer = buffer(&table);
        let texture = texture(&table);
        for before in [
            B::CopyDestination,
            B::CopySource,
            B::Vertex,
            B::Index,
            B::Uniform,
        ] {
            for after in [
                B::CopyDestination,
                B::CopySource,
                B::Vertex,
                B::Index,
                B::Uniform,
            ] {
                assert_eq!(
                    classify(
                        &table,
                        vec![C::ResourceBarrier(R::Buffer {
                            buffer,
                            before,
                            after
                        })]
                    ),
                    Ok(true)
                );
            }
        }
        for before in [
            T::CopyDestination,
            T::CopySource,
            T::Sampled,
            T::RenderAttachment,
        ] {
            for after in [
                T::CopyDestination,
                T::CopySource,
                T::Sampled,
                T::RenderAttachment,
            ] {
                for barrier in [
                    R::Texture {
                        texture,
                        before,
                        after,
                    },
                    R::TextureMip {
                        texture,
                        mip_level: 1,
                        before,
                        after,
                    },
                ] {
                    assert_eq!(
                        classify(&table, vec![C::ResourceBarrier(barrier)]),
                        Ok(true)
                    );
                }
            }
        }
        assert_eq!(
            classify(
                &table,
                vec![
                    C::ResourceBarrier(R::Buffer {
                        buffer,
                        before: B::CopyDestination,
                        after: B::CopySource
                    }),
                    C::ResourceBarrier(R::Buffer {
                        buffer,
                        before: B::CopySource,
                        after: B::Uniform
                    }),
                    C::ResourceBarrier(R::Texture {
                        texture,
                        before: T::RenderAttachment,
                        after: T::CopySource
                    }),
                    C::ResourceBarrier(R::TextureMip {
                        texture,
                        mip_level: 1,
                        before: T::CopySource,
                        after: T::Sampled
                    }),
                ]
            ),
            Ok(true)
        );
    }

    #[test]
    fn storage_scopes_still_require_backend_validation() {
        let table = ir::ResourceTable::new();
        let buffer = buffer(&table);
        let texture = texture(&table);
        for storage in [B::StorageRead, B::StorageReadWrite] {
            for (before, after) in [(storage, B::CopySource), (B::CopyDestination, storage)] {
                assert_eq!(
                    classify(
                        &table,
                        vec![C::ResourceBarrier(R::Buffer {
                            buffer,
                            before,
                            after
                        })]
                    ),
                    Ok(false)
                );
            }
        }
        // Replay succeeds even though native VirGL rejects StorageWrite in
        // either direction. Neither whole-texture nor mip barriers may skip it.
        for (before, after) in [
            (T::StorageWrite, T::Sampled),
            (T::CopyDestination, T::StorageWrite),
        ] {
            for barrier in [
                R::Texture {
                    texture,
                    before,
                    after,
                },
                R::TextureMip {
                    texture,
                    mip_level: 1,
                    before,
                    after,
                },
            ] {
                assert_eq!(
                    classify(&table, vec![C::ResourceBarrier(barrier)]),
                    Ok(false)
                );
            }
        }
    }

    #[test]
    fn malformed_foreign_and_retired_barriers_fail_replay() {
        let table = ir::ResourceTable::new();
        let foreign = ir::ResourceTable::new();
        let buffer = buffer(&table);
        let texture = texture(&table);
        let barrier = C::ResourceBarrier(R::Buffer {
            buffer,
            before: B::CopyDestination,
            after: B::CopySource,
        });
        self::buffer(&foreign); // Matching slots cannot bypass ownership.
        assert_eq!(
            classify(&foreign, vec![barrier.clone()]),
            Err(ir::Error::ResourceTableMismatch)
        );
        assert_eq!(
            classify(
                &table,
                vec![
                    barrier.clone(),
                    C::ResourceBarrier(R::Buffer {
                        buffer,
                        before: B::CopyDestination,
                        after: B::Uniform
                    }),
                ]
            ),
            Err(ir::Error::InvalidResourceAccess)
        );
        assert_eq!(
            classify(
                &table,
                vec![C::ResourceBarrier(R::TextureMip {
                    texture,
                    mip_level: 2,
                    before: T::CopyDestination,
                    after: T::Sampled
                })]
            ),
            Err(ir::Error::OutOfBounds)
        );
        let limited = table
            .define_buffer(ir::BufferDesc::new(4, ir::BufferUsage::COPY_DST).unwrap())
            .unwrap()
            .id();
        assert_eq!(
            classify(
                &table,
                vec![C::ResourceBarrier(R::Buffer {
                    buffer: limited,
                    before: B::CopyDestination,
                    after: B::CopySource
                })]
            ),
            Err(ir::Error::InvalidUsage)
        );
        table.release_buffer(buffer).unwrap();
        assert!(classify(&table, vec![barrier]).is_err());
    }

    #[test]
    fn mixed_upload_copy_and_pass_streams_are_never_elided() {
        let table = ir::ResourceTable::new();
        let source = buffer(&table);
        let destination = buffer(&table);
        let barrier = C::ResourceBarrier(R::Buffer {
            buffer: source,
            before: B::CopyDestination,
            after: B::CopySource,
        });
        for ops in [
            vec![
                C::WriteBuffer {
                    buffer: source,
                    offset: 0,
                    data: vec![7; 4],
                },
                barrier.clone(),
            ],
            vec![
                barrier.clone(),
                C::CopyBufferToBuffer {
                    source,
                    source_offset: 0,
                    destination,
                    destination_offset: 0,
                    size: 4,
                },
            ],
            vec![barrier, C::BeginComputePass, C::EndComputePass],
        ] {
            assert_eq!(classify(&table, ops), Ok(false));
        }
        assert_eq!(classify(&table, vec![]), Ok(false));
    }

    #[test]
    fn bind_draw_dispatch_and_state_commands_are_not_barriers() {
        let table = ir::ResourceTable::new();
        let buffer = buffer(&table);
        let texture = texture(&table);
        // Test individual variants as well as the validated mixed streams above.
        for command in [
            C::SetVertexBuffer { buffer, offset: 0 },
            C::SetTexture(texture),
            C::Draw {
                vertex_count: 3,
                first_vertex: 0,
            },
            C::DrawIndexed {
                index_count: 3,
                first_index: 0,
                base_vertex: 0,
            },
            C::BeginComputePass,
            C::Dispatch { x: 1, y: 1, z: 1 },
            C::EndComputePass,
            C::EndRenderPass,
            C::SetDrawIndex(0),
        ] {
            assert!(!ordinary_barrier(&command));
        }
    }
}
