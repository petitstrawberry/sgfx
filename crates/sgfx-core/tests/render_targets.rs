use sgfx_core::ir::*;

#[test]
fn owned_multiple_attachments_resolve_all_identities_and_preserve_read_only_depth() {
    let table = ResourceTable::new();
    let a = texture(&table, TextureFormat::Rgba8Unorm, 8);
    let b = texture(&table, TextureFormat::Bgra8Unorm, 8);
    let depth = texture(&table, TextureFormat::Depth32Float, 8);
    let recording = OwnedCommandBuffer::new(vec![
        OwnedCommand::BeginRenderPassWithAttachments {
            desc: OwnedRenderPassDesc {
                target: a.id(),
                area: PixelRect::new(0, 0, 8, 8).unwrap(),
                load: LoadOp::Load,
                store: StoreOp::Store,
                depth: Some(OwnedDepthAttachment {
                    target: depth.id(),
                    load: DepthLoadOp::Load,
                    store: StoreOp::Store,
                }),
            },
            colors: vec![OwnedColorAttachment {
                target: b.id(),
                load: LoadOp::Load,
                store: StoreOp::Store,
            }],
            read_only_depth: true,
        },
        OwnedCommand::EndRenderPass,
    ]);
    let commands = recording.record(&table).unwrap();
    let Command::BeginRenderPass(desc) = commands.commands()[0] else {
        panic!("missing pass")
    };
    assert_eq!(
        desc.color_attachments()
            .map(|a| a.target().id())
            .collect::<Vec<_>>(),
        vec![a.id(), b.id()]
    );
    assert!(desc.depth_attachment().unwrap().read_only());
    assert!(recording.record(&ResourceTable::new()).is_err());
}

fn texture(table: &ResourceTable, format: TextureFormat, size: u32) -> TextureRef<'_> {
    table
        .define_texture(
            TextureDesc::new(
                format,
                Extent2D::new(size, size).unwrap(),
                TextureUsage::SAMPLED | TextureUsage::RENDER_ATTACHMENT,
            )
            .unwrap(),
        )
        .unwrap()
}

fn pipeline(
    table: &ResourceTable,
    groups: Vec<BindGroupLayoutDesc>,
) -> ProgrammableRenderPipelineDesc {
    let module = table.define_shader_module(ShaderModuleDesc::wgsl("@vertex fn vertex() -> @builtin(position) vec4<f32> { return vec4(0.0); } @fragment fn fragment() -> @location(0) vec4<f32> { return vec4(1.0); }".into()).unwrap()).unwrap();
    ProgrammableRenderPipelineDesc::new(
        ShaderEntryPoint::new(module, ShaderStage::Vertex, "vertex".into()).unwrap(),
        ShaderEntryPoint::new(module, ShaderStage::Fragment, "fragment".into()).unwrap(),
        PipelineLayoutDesc::new(groups).unwrap(),
        TextureFormat::Rgba8Unorm,
        None,
        PrimitiveTopology::TriangleList,
        BlendState::REPLACE,
        RasterState::new(CullMode::None, FrontFace::CounterClockwise),
    )
    .unwrap()
}

#[test]
fn multiple_color_targets_validate_counts_formats_and_all_attachment_aliases() {
    let table = ResourceTable::new();
    let a = texture(&table, TextureFormat::Rgba8Unorm, 8);
    let b = texture(&table, TextureFormat::Bgra8Unorm, 8);
    let small = texture(&table, TextureFormat::Rgba8Unorm, 4);
    let area = PixelRect::new(0, 0, 8, 8).unwrap();
    let desc = RenderPassDesc::new(&table, a, area, LoadOp::Load, StoreOp::Store).unwrap();
    assert!(
        desc.with_color_attachment(&table, a, LoadOp::Load, StoreOp::Store)
            .is_err()
    );
    assert!(
        desc.with_color_attachment(&table, small, LoadOp::Load, StoreOp::Store)
            .is_err()
    );
    let desc = desc
        .with_color_attachment(&table, b, LoadOp::Load, StoreOp::Store)
        .unwrap();
    let group_layout = BindGroupLayoutDesc::new(vec![BindGroupLayoutEntry::new(
        0,
        ShaderStages::FRAGMENT,
        BindingType::SampledTexture,
    )])
    .unwrap();
    let original = pipeline(&table, vec![group_layout.clone()]);
    let single = table
        .define_programmable_render_pipeline(original.clone())
        .unwrap();
    let multiple = table
        .define_programmable_render_pipeline(
            original
                .with_color_targets(vec![
                    ColorTargetState::new(
                        TextureFormat::Rgba8Unorm,
                        BlendState::REPLACE,
                        ColorWriteMask::ALL,
                    )
                    .unwrap(),
                    ColorTargetState::new(
                        TextureFormat::Bgra8Unorm,
                        BlendState::REPLACE,
                        ColorWriteMask::from_bits(3).unwrap(),
                    )
                    .unwrap(),
                ])
                .unwrap(),
        )
        .unwrap();
    let bindings = table
        .define_bind_group(
            BindGroupDesc::new(
                &table,
                group_layout,
                vec![BindGroupEntry::new(0, BindingResource::Texture(b.id()))],
            )
            .unwrap(),
        )
        .unwrap();
    let mut encoder = CommandEncoder::new(&table);
    let mut pass = encoder.begin_render_pass(desc).unwrap();
    assert_eq!(
        pass.set_programmable_pipeline(single),
        Err(Error::PipelineTargetMismatch)
    );
    pass.set_programmable_pipeline(multiple).unwrap();
    pass.set_bind_group(0, bindings).unwrap();
    assert_eq!(pass.draw(3, 0), Err(Error::AttachmentFeedback));
    pass.end().unwrap();
    encoder.finish().unwrap();
}

#[test]
fn depth_sampling_requires_read_only_attachment_and_disallows_depth_writes() {
    let table = ResourceTable::new();
    let color = texture(&table, TextureFormat::Rgba8Unorm, 8);
    let depth = texture(&table, TextureFormat::Depth32Float, 8);
    let area = PixelRect::new(0, 0, 8, 8).unwrap();
    let view = TextureViewDesc::new(
        table.texture(depth).unwrap(),
        TextureFormat::Depth32Float,
        TextureViewDimension::D2,
        0,
        1,
        0,
        1,
    )
    .unwrap();
    let group_layout = BindGroupLayoutDesc::new(vec![BindGroupLayoutEntry::new(
        0,
        ShaderStages::FRAGMENT,
        BindingType::SampledTextureView {
            dimension: TextureViewDimension::D2,
            depth: true,
        },
    )])
    .unwrap();
    let bindings = table
        .define_bind_group(
            BindGroupDesc::new(
                &table,
                group_layout.clone(),
                vec![BindGroupEntry::new(
                    0,
                    BindingResource::TextureView {
                        texture: depth.id(),
                        view,
                    },
                )],
            )
            .unwrap(),
        )
        .unwrap();
    let original = pipeline(&table, vec![group_layout]);
    let read_pipeline = table
        .define_programmable_render_pipeline(
            original
                .clone()
                .with_depth_state(DepthState::new(
                    TextureFormat::Depth32Float,
                    CompareFunction::LessEqual,
                    false,
                ))
                .unwrap(),
        )
        .unwrap();
    let write_pipeline = table
        .define_programmable_render_pipeline(
            original
                .with_depth_state(DepthState::new(
                    TextureFormat::Depth32Float,
                    CompareFunction::LessEqual,
                    true,
                ))
                .unwrap(),
        )
        .unwrap();
    let desc = RenderPassDesc::new(&table, color, area, LoadOp::Load, StoreOp::Store).unwrap();
    assert!(
        desc.with_depth_attachment(&table, depth, DepthLoadOp::Clear(1.0), StoreOp::Store)
            .unwrap()
            .with_read_only_depth()
            .is_err()
    );
    let desc = desc
        .with_depth_attachment(&table, depth, DepthLoadOp::Load, StoreOp::Store)
        .unwrap();
    let mut encoder = CommandEncoder::new(&table);
    for read_only in [false, true] {
        let mut pass = encoder
            .begin_render_pass(if read_only {
                desc.with_read_only_depth().unwrap()
            } else {
                desc
            })
            .unwrap();
        if read_only {
            assert_eq!(
                pass.set_programmable_pipeline(write_pipeline),
                Err(Error::PipelineTargetMismatch)
            );
        }
        pass.set_programmable_pipeline(read_pipeline).unwrap();
        pass.set_bind_group(0, bindings).unwrap();
        assert_eq!(
            pass.draw(3, 0),
            if read_only {
                Ok(())
            } else {
                Err(Error::AttachmentFeedback)
            }
        );
        pass.end().unwrap();
    }
    encoder.finish().unwrap();
}
