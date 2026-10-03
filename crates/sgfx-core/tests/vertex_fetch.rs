use sgfx_core::ir::*;

fn recording(
    table: &ResourceTable,
    buffer_size: u64,
    second_offset: u64,
    attribute_offset: u32,
    draw: OwnedCommand,
) -> OwnedCommandBuffer {
    let module = table
        .define_shader_module(
            ShaderModuleDesc::wgsl(
                "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }
         @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }"
                    .into(),
            )
            .unwrap(),
        )
        .unwrap();
    let pipeline = table
        .define_programmable_render_pipeline(
            ProgrammableRenderPipelineDesc::new(
                ShaderEntryPoint::new(module, ShaderStage::Vertex, "vs".into()).unwrap(),
                ShaderEntryPoint::new(module, ShaderStage::Fragment, "fs".into()).unwrap(),
                PipelineLayoutDesc::new(vec![]).unwrap(),
                TextureFormat::Rgba8Unorm,
                None,
                PrimitiveTopology::TriangleStrip,
                BlendState::REPLACE,
                RasterState::new(CullMode::None, FrontFace::CounterClockwise),
            )
            .unwrap()
            .with_vertex_buffers(vec![
                VertexBufferLayout::new(
                    16,
                    vec![VertexAttribute::new(0, VertexFormat::Float32x2, 0)],
                )
                .unwrap(),
                VertexBufferLayout::new(
                    16,
                    vec![VertexAttribute::new(
                        1,
                        VertexFormat::Float32x2,
                        attribute_offset,
                    )],
                )
                .unwrap(),
            ])
            .unwrap(),
        )
        .unwrap();
    let buffer = table
        .define_buffer(BufferDesc::new(buffer_size, BufferUsage::VERTEX).unwrap())
        .unwrap();
    let index = table
        .define_buffer(BufferDesc::new(8, BufferUsage::INDEX).unwrap())
        .unwrap();
    let target = table
        .define_texture(
            TextureDesc::new(
                TextureFormat::Rgba8Unorm,
                Extent2D::new(8, 8).unwrap(),
                TextureUsage::RENDER_ATTACHMENT,
            )
            .unwrap(),
        )
        .unwrap();
    OwnedCommandBuffer::new(vec![
        OwnedCommand::BeginRenderPass(OwnedRenderPassDesc {
            target: target.id(),
            area: PixelRect::new(0, 0, 8, 8).unwrap(),
            load: LoadOp::Load,
            store: StoreOp::Store,
            depth: None,
        }),
        OwnedCommand::SetProgrammablePipeline(pipeline.id()),
        OwnedCommand::SetVertexBuffer {
            buffer: buffer.id(),
            offset: 0,
        },
        OwnedCommand::SetVertexBufferSlot {
            slot: 1,
            buffer: buffer.id(),
            offset: second_offset,
        },
        OwnedCommand::SetIndexBuffer {
            buffer: index.id(),
            offset: 0,
            format: IndexFormat::Uint16,
        },
        draw,
        OwnedCommand::EndRenderPass,
    ])
}

#[test]
fn split_interleaved_bindings_require_only_the_final_attribute_bytes() {
    for draw in [
        OwnedCommand::Draw {
            vertex_count: 4,
            first_vertex: 0,
        },
        OwnedCommand::Draw {
            vertex_count: 3,
            first_vertex: 1,
        },
        OwnedCommand::DrawInstanced {
            vertex_count: 4,
            first_vertex: 0,
            instance_count: 5,
            first_instance: 7,
        },
    ] {
        let table = ResourceTable::new();
        // OpenTTD's actual layout: 4 * (position.xy, uv.xy), with the UV
        // binding based eight bytes into the same 64-byte allocation.
        let commands = recording(&table, 64, 8, 0, draw);
        commands.record(&table).unwrap();
    }
}

#[test]
fn final_attribute_still_rejects_short_ranges_and_counts_past_the_buffer() {
    for (size, offset, attribute_offset, first) in
        [(60, 8, 0, 0), (64, 12, 0, 0), (64, 8, 4, 0), (64, 8, 0, 1)]
    {
        let table = ResourceTable::new();
        let commands = recording(
            &table,
            size,
            offset,
            attribute_offset,
            OwnedCommand::Draw {
                vertex_count: 4,
                first_vertex: first,
            },
        );
        assert_eq!(commands.validate(&table), Err(Error::OutOfBounds));
    }
}

#[test]
fn indexed_minimum_fetch_does_not_require_unused_stride_padding() {
    let table = ResourceTable::new();
    let commands = recording(
        &table,
        16,
        8,
        0,
        OwnedCommand::DrawIndexed {
            index_count: 3,
            first_index: 0,
            base_vertex: 0,
        },
    );
    // The actual index values remain a backend validation responsibility.
    commands.record(&table).unwrap();
    let table = ResourceTable::new();
    let commands = recording(
        &table,
        12,
        8,
        0,
        OwnedCommand::DrawIndexed {
            index_count: 3,
            first_index: 0,
            base_vertex: 0,
        },
    );
    assert_eq!(commands.validate(&table), Err(Error::OutOfBounds));
}
