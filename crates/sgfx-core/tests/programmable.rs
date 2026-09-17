use sgfx_core::ir::*;

#[test]
fn instanced_draw_ranges_survive_owned_replay() {
    let table = ResourceTable::new();
    let format = TextureFormat::Rgba8Unorm;
    let target = table.define_texture(TextureDesc::new(format, Extent2D::new(8,8).unwrap(),
        TextureUsage::RENDER_ATTACHMENT).unwrap()).unwrap();
    let pipeline = table.define_programmable_render_pipeline(ProgrammableRenderPipelineDesc::new(
        shader(&table, ShaderStage::Vertex), shader(&table, ShaderStage::Fragment), PipelineLayoutDesc::new(vec![]).unwrap(),
        format, None, PrimitiveTopology::TriangleList,
        BlendState::REPLACE, RasterState::new(CullMode::None, FrontFace::CounterClockwise)).unwrap()).unwrap();
    let indices = buffer(&table, BufferUsage::INDEX);
    let recording = OwnedCommandBuffer::new(vec![
        OwnedCommand::BeginRenderPass(OwnedRenderPassDesc { target:target.id(), depth:None, area:PixelRect::new(0,0,8,8).unwrap(), load:LoadOp::Load, store:StoreOp::Store }),
        OwnedCommand::SetProgrammablePipeline(pipeline.id()),
        OwnedCommand::SetIndexBuffer { buffer:indices.id(), offset:0, format:IndexFormat::Uint16 },
        OwnedCommand::DrawInstanced { vertex_count:3, first_vertex:0, instance_count:4, first_instance:7 },
        OwnedCommand::DrawIndexedInstanced { index_count:3, first_index:0, base_vertex:2, instance_count:5, first_instance:9 },
        OwnedCommand::EndRenderPass,
    ]);
    let commands = recording.record(&table).unwrap();
    assert!(matches!(commands.commands()[3], Command::DrawInstanced { instance_count:4, first_instance:7, .. }));
    assert!(matches!(commands.commands()[4], Command::DrawIndexedInstanced { instance_count:5, first_instance:9, base_vertex:2, .. }));
    for (count, first, error) in [(0, 0, Error::InvalidValue), (2, u32::MAX, Error::Overflow)] {
        let mut invalid = recording.commands().to_vec();
        invalid[3] = OwnedCommand::DrawInstanced { vertex_count:3, first_vertex:0, instance_count:count, first_instance:first };
        assert_eq!(OwnedCommandBuffer::new(invalid).validate(&table), Err(error));
    }
}

#[test]
fn comparison_samplers_require_a_matching_binding_type() {
    let table = ResourceTable::new();
    let ordinary = SamplerDesc::new(
        FilterMode::Linear,
        FilterMode::Linear,
        AddressMode::ClampToEdge,
        AddressMode::ClampToEdge,
    );
    assert_eq!(ordinary.compare(), None);
    for compare in [None, Some(CompareFunction::LessEqual), Some(CompareFunction::Greater)] {
        let desc = ordinary.with_compare(compare);
        assert_eq!(desc.compare(), compare);
        let sampler = table.define_sampler(desc).unwrap();
        for ty in [BindingType::Sampler, BindingType::ComparisonSampler] {
            let result = BindGroupDesc::new(
                &table,
                layout(ty),
                vec![BindGroupEntry::new(0, BindingResource::Sampler(sampler.id()))],
            );
            assert_eq!(result.is_ok(), compare.is_some() == (ty == BindingType::ComparisonSampler));
        }
    }
}

fn shader(table: &ResourceTable, stage: ShaderStage) -> ShaderEntryPoint {
    let module = table
        .define_shader_module(
            ShaderModuleDesc::wgsl("@compute @workgroup_size(1) fn main() {}".into()).unwrap(),
        )
        .unwrap();
    ShaderEntryPoint::new(module, stage, "main".into()).unwrap()
}
fn layout(ty: BindingType) -> BindGroupLayoutDesc {
    BindGroupLayoutDesc::new(vec![BindGroupLayoutEntry::new(
        0,
        ShaderStages::COMPUTE,
        ty,
    )])
    .unwrap()
}
fn group<'r>(table: &'r ResourceTable, buffer: BufferRef<'r>, ty: BindingType) -> BindGroupRef<'r> {
    table
        .define_bind_group(
            BindGroupDesc::new(
                table,
                layout(ty),
                vec![BindGroupEntry::new(
                    0,
                    BindingResource::Buffer {
                        buffer: buffer.id(),
                        offset: 0,
                        size: 16,
                    },
                )],
            )
            .unwrap(),
        )
        .unwrap()
}
fn compute<'r>(table: &'r ResourceTable, ty: BindingType) -> ComputePipelineRef<'r> {
    table
        .define_compute_pipeline(
            ComputePipelineDesc::new(
                shader(table, ShaderStage::Compute),
                PipelineLayoutDesc::new(vec![layout(ty)]).unwrap(),
            )
            .unwrap(),
        )
        .unwrap()
}
fn buffer(table: &ResourceTable, usage: BufferUsage) -> BufferRef<'_> {
    table
        .define_buffer(BufferDesc::new(512, usage).unwrap())
        .unwrap()
}
fn dispatch<'r>(
    encoder: &mut CommandEncoder<'r, '_>,
    pipeline: ComputePipelineRef<'r>,
    group: BindGroupRef<'r>,
) -> Result<()> {
    let mut pass = encoder.begin_compute_pass()?;
    pass.set_pipeline(pipeline)?;
    pass.set_bind_group(0, group)?;
    pass.dispatch(1, 1, 1)?;
    pass.end()
}

#[test]
fn push_constant_ranges_validate_alignment_limits_and_stage_overlap() {
    for (stages, offset, size, expected) in [
        (ShaderStages::empty(), 0, 4, Error::InvalidDescriptor),
        (ShaderStages::VERTEX, 2, 4, Error::InvalidDescriptor),
        (ShaderStages::VERTEX, 0, 6, Error::InvalidDescriptor),
        (ShaderStages::VERTEX, 0, 0, Error::InvalidDescriptor),
        (ShaderStages::VERTEX, 128, 4, Error::OutOfBounds),
        (ShaderStages::VERTEX, u32::MAX - 3, 4, Error::Overflow),
    ] {
        assert_eq!(PushConstantRange::new(stages, offset, size), Err(expected));
    }
    let vertex = PushConstantRange::new(ShaderStages::VERTEX, 0, 64).unwrap();
    let fragment = PushConstantRange::new(ShaderStages::FRAGMENT, 0, 16).unwrap();
    let layout = PipelineLayoutDesc::new(vec![])
        .unwrap()
        .with_push_constant_ranges(vec![vertex, fragment])
        .unwrap();
    assert!(
        layout
            .validate_push_constants(ShaderStages::VERTEX | ShaderStages::FRAGMENT, 4, &[1; 12])
            .is_ok()
    );
    assert_eq!(
        layout.validate_push_constants(ShaderStages::FRAGMENT, 16, &[1; 4]),
        Err(Error::BindingLayoutMismatch)
    );
    assert_eq!(
        layout.validate_push_constants(ShaderStages::COMPUTE, 0, &[1; 4]),
        Err(Error::BindingLayoutMismatch)
    );
    assert_eq!(
        layout.validate_push_constants(ShaderStages::VERTEX, 2, &[1; 4]),
        Err(Error::InvalidValue)
    );
    assert_eq!(
        layout.validate_push_constants(ShaderStages::VERTEX, 0, &[1; 4]),
        Err(Error::BindingLayoutMismatch)
    );
    assert!(
        layout
            .validate_push_constants(ShaderStages::VERTEX, 20, &[1; 4])
            .is_ok()
    );
    assert_eq!(
        layout.validate_push_constants(ShaderStages::VERTEX, 0, &[]),
        Err(Error::InvalidValue)
    );
    assert_eq!(
        PipelineLayoutDesc::new(vec![])
            .unwrap()
            .with_push_constant_ranges(vec![vertex, vertex]),
        Err(Error::InvalidDescriptor)
    );
}

#[test]
fn owned_push_constants_replay_without_copying_and_require_pipeline_and_scope() {
    let table = ResourceTable::new();
    let layout = PipelineLayoutDesc::new(vec![])
        .unwrap()
        .with_push_constant_ranges(vec![
            PushConstantRange::new(ShaderStages::COMPUTE, 16, 16).unwrap(),
        ])
        .unwrap();
    let pipeline = table
        .define_compute_pipeline(
            ComputePipelineDesc::new(shader(&table, ShaderStage::Compute), layout).unwrap(),
        )
        .unwrap()
        .id();
    let data = vec![7; 8];
    let address = data.as_ptr();
    let update = OwnedCommand::SetPushConstants {
        stages: ShaderStages::COMPUTE,
        offset: 20,
        data,
    };
    let recording = OwnedCommandBuffer::new(vec![
        OwnedCommand::BeginComputePass,
        OwnedCommand::SetComputePipeline(pipeline),
        update.clone(),
        OwnedCommand::EndComputePass,
    ]);
    // The cloned recording owns its own bytes, and replay borrows those exact bytes.
    let OwnedCommand::SetPushConstants { data, .. } = &recording.commands()[2] else {
        panic!()
    };
    let retained = data.as_ptr();
    assert_ne!(retained, address);
    for _ in 0..2 {
        let borrowed = recording.record(&table).unwrap();
        assert!(
            matches!(&borrowed.commands()[2], Command::SetPushConstants { offset: 20, data, .. } if data.as_ptr() == retained && *data == [7; 8])
        );
    }
    assert_eq!(
        OwnedCommandBuffer::new(vec![
            OwnedCommand::BeginComputePass,
            update.clone(),
            OwnedCommand::EndComputePass,
        ])
        .validate(&table),
        Err(Error::PipelineNotSet)
    );
    assert_eq!(
        OwnedCommandBuffer::new(vec![update]).validate(&table),
        Err(Error::InvalidDescriptor)
    );
}

#[test]
fn descriptors_reject_duplicate_bindings_empty_visibility_and_invalid_storage_formats() {
    let entry = BindGroupLayoutEntry::new(0, ShaderStages::COMPUTE, BindingType::UniformBuffer);
    assert_eq!(
        BindGroupLayoutDesc::new(vec![entry, entry]),
        Err(Error::InvalidDescriptor)
    );
    assert_eq!(
        BindGroupLayoutDesc::new(vec![BindGroupLayoutEntry::new(
            0,
            ShaderStages::empty(),
            BindingType::Sampler
        )]),
        Err(Error::InvalidDescriptor)
    );
    assert_eq!(
        BindGroupLayoutDesc::new(vec![BindGroupLayoutEntry::new(
            MAX_BINDINGS_PER_GROUP as u32,
            ShaderStages::COMPUTE,
            BindingType::Sampler
        )]),
        Err(Error::InvalidDescriptor)
    );
    assert_eq!(
        TextureDesc::new(
            TextureFormat::Bgra8Unorm,
            Extent2D::new(1, 1).unwrap(),
            TextureUsage::STORAGE
        ),
        Err(Error::InvalidDescriptor)
    );
}

#[test]
fn owned_viewports_validate_values_attachment_bounds_and_pass_scope() {
    assert_eq!(
        Viewport::new(0.0, 0.0, 0.0, 8.0, 0.0, 1.0),
        Err(Error::InvalidValue)
    );
    assert_eq!(
        Viewport::new(f32::NAN, 0.0, 8.0, 8.0, 0.0, 1.0),
        Err(Error::InvalidValue)
    );
    let table = ResourceTable::new();
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
    let valid = Viewport::new(2.0, 1.0, 6.0, 7.0, 0.2, 0.8).unwrap();
    let begin = OwnedCommand::BeginRenderPass(OwnedRenderPassDesc {
        target: target.id(),
        area: PixelRect::new(0, 0, 8, 8).unwrap(),
        load: LoadOp::DontCare,
        store: StoreOp::Store,
        depth: None,
    });
    let recording = OwnedCommandBuffer::new(vec![
        begin.clone(),
        OwnedCommand::SetViewport(valid),
        OwnedCommand::EndRenderPass,
    ]);
    let commands = recording.record(&table).unwrap();
    assert!(matches!(commands.commands()[1], Command::SetViewport(v) if v == valid));
    let outside = Viewport::new(2.0, 0.0, 7.0, 8.0, 0.0, 1.0).unwrap();
    assert_eq!(
        OwnedCommandBuffer::new(vec![
            begin,
            OwnedCommand::SetViewport(outside),
            OwnedCommand::EndRenderPass
        ])
        .validate(&table),
        Err(Error::OutOfBounds)
    );
    assert_eq!(
        OwnedCommandBuffer::new(vec![OwnedCommand::SetViewport(valid)]).validate(&table),
        Err(Error::InvalidDescriptor)
    );
}

#[test]
fn binding_validation_checks_ownership_ranges_alignment_and_usage() {
    let table = ResourceTable::new();
    let other = ResourceTable::new();
    let local = buffer(&table, BufferUsage::UNIFORM);
    let foreign = buffer(&other, BufferUsage::UNIFORM);
    for (id, offset, size, expected) in [
        (foreign.id(), 0, 16, Error::ResourceTableMismatch),
        (local.id(), 4, 16, Error::InvalidValue),
        (local.id(), 512, 16, Error::OutOfBounds),
        (local.id(), 0, 0, Error::InvalidValue),
        (local.id(), 0, 3, Error::InvalidValue),
    ] {
        assert_eq!(
            BindGroupDesc::new(
                &table,
                layout(BindingType::UniformBuffer),
                vec![BindGroupEntry::new(
                    0,
                    BindingResource::Buffer {
                        buffer: id,
                        offset,
                        size
                    }
                )]
            ),
            Err(expected)
        );
    }
    assert_eq!(
        BindGroupDesc::new(
            &table,
            layout(BindingType::StorageBuffer { read_only: true }),
            vec![BindGroupEntry::new(
                0,
                BindingResource::Buffer {
                    buffer: local.id(),
                    offset: 0,
                    size: 16
                }
            )]
        ),
        Err(Error::InvalidUsage)
    );
}

#[test]
fn descriptor_ownership_is_checked_again_when_defining_resources() {
    let table = ResourceTable::new();
    let other = ResourceTable::new();
    let local = buffer(&table, BufferUsage::UNIFORM);
    let desc = BindGroupDesc::new(
        &table,
        layout(BindingType::UniformBuffer),
        vec![BindGroupEntry::new(
            0,
            BindingResource::Buffer {
                buffer: local.id(),
                offset: 0,
                size: 16,
            },
        )],
    )
    .unwrap();
    assert_eq!(
        other.define_bind_group(desc),
        Err(Error::ResourceTableMismatch)
    );
    let pipeline = ComputePipelineDesc::new(
        shader(&table, ShaderStage::Compute),
        PipelineLayoutDesc::new(vec![]).unwrap(),
    )
    .unwrap();
    assert_eq!(
        other.define_compute_pipeline(pipeline),
        Err(Error::ResourceTableMismatch)
    );
}

#[test]
fn compute_requires_pipeline_exact_bindings_and_positive_bounded_grid() {
    let table = ResourceTable::new();
    let target = buffer(&table, BufferUsage::STORAGE | BufferUsage::UNIFORM);
    let pipeline = compute(&table, BindingType::StorageBuffer { read_only: true });
    let wrong = group(&table, target, BindingType::UniformBuffer);
    let good = group(
        &table,
        target,
        BindingType::StorageBuffer { read_only: true },
    );
    let mut encoder = CommandEncoder::new(&table);
    let mut pass = encoder.begin_compute_pass().unwrap();
    assert_eq!(pass.dispatch(1, 1, 1), Err(Error::PipelineNotSet));
    pass.set_pipeline(pipeline).unwrap();
    assert_eq!(pass.dispatch(1, 1, 1), Err(Error::BindGroupNotSet));
    pass.set_bind_group(0, wrong).unwrap();
    assert_eq!(pass.dispatch(1, 1, 1), Err(Error::BindingLayoutMismatch));
    pass.set_bind_group(0, good).unwrap();
    assert_eq!(pass.dispatch(0, 1, 1), Err(Error::InvalidValue));
    assert_eq!(pass.dispatch(1, 65_536, 1), Err(Error::InvalidValue));
    pass.dispatch(1, 1, 1).unwrap();
    pass.end().unwrap();
    assert_eq!(
        encoder
            .finish()
            .unwrap()
            .commands()
            .iter()
            .filter(|command| matches!(command, Command::Dispatch { .. }))
            .count(),
        1
    );
}

#[test]
fn storage_write_requires_matching_barrier_before_copy_or_next_dispatch() {
    let table = ResourceTable::new();
    let target = buffer(
        &table,
        BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
    );
    let destination = buffer(&table, BufferUsage::COPY_DST);
    let ty = BindingType::StorageBuffer { read_only: false };
    let pipeline = compute(&table, ty);
    let bindings = group(&table, target, ty);
    let mut encoder = CommandEncoder::new(&table);
    dispatch(&mut encoder, pipeline, bindings).unwrap();
    assert_eq!(
        encoder.copy_buffer_to_buffer(target, 0, destination, 0, 16),
        Err(Error::MissingBarrier)
    );
    assert_eq!(
        encoder.write_buffer(target, 0, &[0; 16]),
        Err(Error::MissingBarrier)
    );
    assert_eq!(
        encoder.resource_barrier(ResourceBarrier::Buffer {
            buffer: target,
            before: BufferAccess::CopyDestination,
            after: BufferAccess::CopySource
        }),
        Err(Error::InvalidResourceAccess)
    );
    encoder
        .resource_barrier(ResourceBarrier::Buffer {
            buffer: target,
            before: BufferAccess::StorageReadWrite,
            after: BufferAccess::CopySource,
        })
        .unwrap();
    encoder
        .copy_buffer_to_buffer(target, 0, destination, 0, 16)
        .unwrap();
    dispatch(&mut encoder, pipeline, bindings).unwrap();
    let commands = encoder.finish().unwrap();
    assert_eq!(
        commands
            .commands()
            .iter()
            .filter(|command| matches!(command, Command::ResourceBarrier(_)))
            .count(),
        1
    );
}

#[test]
fn storage_writes_cannot_be_reused_inside_the_same_pass_without_a_dependency() {
    let table = ResourceTable::new();
    let target = buffer(&table, BufferUsage::STORAGE);
    let ty = BindingType::StorageBuffer { read_only: false };
    let pipeline = compute(&table, ty);
    let bindings = group(&table, target, ty);
    let mut encoder = CommandEncoder::new(&table);
    let mut pass = encoder.begin_compute_pass().unwrap();
    pass.set_pipeline(pipeline).unwrap();
    pass.set_bind_group(0, bindings).unwrap();
    pass.dispatch(1, 1, 1).unwrap();
    assert_eq!(pass.dispatch(1, 1, 1), Err(Error::MissingBarrier));
    pass.end().unwrap();
    encoder
        .resource_barrier(ResourceBarrier::Buffer {
            buffer: target,
            before: BufferAccess::StorageReadWrite,
            after: BufferAccess::StorageReadWrite,
        })
        .unwrap();
    dispatch(&mut encoder, pipeline, bindings).unwrap();
    encoder.finish().unwrap();
}

#[test]
fn rejects_write_aliases_across_descriptor_sets() {
    let table = ResourceTable::new();
    let target = buffer(&table, BufferUsage::STORAGE);
    let read = BindingType::StorageBuffer { read_only: true };
    let write = BindingType::StorageBuffer { read_only: false };
    let pipeline = table
        .define_compute_pipeline(
            ComputePipelineDesc::new(
                shader(&table, ShaderStage::Compute),
                PipelineLayoutDesc::new(vec![layout(read), layout(write)]).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    let mut encoder = CommandEncoder::new(&table);
    let mut pass = encoder.begin_compute_pass().unwrap();
    pass.set_pipeline(pipeline).unwrap();
    pass.set_bind_group(0, group(&table, target, read)).unwrap();
    pass.set_bind_group(1, group(&table, target, write))
        .unwrap();
    assert_eq!(pass.dispatch(1, 1, 1), Err(Error::ResourceAccessConflict));
}

#[test]
fn programmable_draw_can_generate_vertices_without_fixed_uniforms() {
    let table = ResourceTable::new();
    let pipeline = table
        .define_programmable_render_pipeline(
            ProgrammableRenderPipelineDesc::new(
                shader(&table, ShaderStage::Vertex),
                shader(&table, ShaderStage::Fragment),
                PipelineLayoutDesc::new(vec![]).unwrap(),
                TextureFormat::Rgba8Unorm,
                None,
                PrimitiveTopology::TriangleList,
                BlendState::REPLACE,
                RasterState::new(CullMode::None, FrontFace::CounterClockwise),
            )
            .unwrap(),
        )
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
    let mut encoder = CommandEncoder::new(&table);
    let mut pass = encoder
        .begin_render_pass(
            RenderPassDesc::new(
                &table,
                target,
                PixelRect::new(0, 0, 8, 8).unwrap(),
                LoadOp::DontCare,
                StoreOp::Store,
            )
            .unwrap(),
        )
        .unwrap();
    pass.set_programmable_pipeline(pipeline).unwrap();
    assert_eq!(pass.draw(3, u32::MAX), Err(Error::Overflow));
    pass.draw(3, 0).unwrap();
    pass.end().unwrap();
    assert_eq!(encoder.finish().unwrap().command_count(), 4);
}

#[test]
fn storage_texture_write_requires_barrier_before_using_it_as_attachment() {
    let table = ResourceTable::new();
    let target = table
        .define_texture(
            TextureDesc::new(
                TextureFormat::Rgba8Unorm,
                Extent2D::new(8, 8).unwrap(),
                TextureUsage::STORAGE | TextureUsage::RENDER_ATTACHMENT,
            )
            .unwrap(),
        )
        .unwrap();
    let ty = BindingType::StorageTexture {
        format: TextureFormat::Rgba8Unorm,
        access: StorageTextureAccess::WriteOnly,
    };
    let pipeline = compute(&table, ty);
    let bindings = table
        .define_bind_group(
            BindGroupDesc::new(
                &table,
                layout(ty),
                vec![BindGroupEntry::new(
                    0,
                    BindingResource::Texture(target.id()),
                )],
            )
            .unwrap(),
        )
        .unwrap();
    let mut encoder = CommandEncoder::new(&table);
    dispatch(&mut encoder, pipeline, bindings).unwrap();
    let render = RenderPassDesc::new(
        &table,
        target,
        PixelRect::new(0, 0, 8, 8).unwrap(),
        LoadOp::Load,
        StoreOp::Store,
    )
    .unwrap();
    assert!(matches!(
        encoder.begin_render_pass(render),
        Err(Error::MissingBarrier)
    ));
    encoder
        .resource_barrier(ResourceBarrier::Texture {
            texture: target,
            before: TextureAccess::StorageWrite,
            after: TextureAccess::RenderAttachment,
        })
        .unwrap();
    encoder.begin_render_pass(render).unwrap().end().unwrap();
    encoder.finish().unwrap();
}

#[test]
fn buffer_copy_rejects_foreign_alias_misaligned_and_overflow_ranges() {
    let table = ResourceTable::new();
    let other = ResourceTable::new();
    let source = buffer(&table, BufferUsage::COPY_SRC | BufferUsage::COPY_DST);
    let destination = buffer(&table, BufferUsage::COPY_DST);
    let foreign = buffer(&other, BufferUsage::COPY_DST);
    let mut encoder = CommandEncoder::new(&table);
    assert_eq!(
        encoder.copy_buffer_to_buffer(source, 0, foreign, 0, 4),
        Err(Error::ResourceTableMismatch)
    );
    assert_eq!(
        encoder.copy_buffer_to_buffer(source, 0, source, 256, 4),
        Err(Error::ResourceAccessConflict)
    );
    assert_eq!(
        encoder.copy_buffer_to_buffer(source, 1, destination, 0, 4),
        Err(Error::InvalidValue)
    );
    assert_eq!(
        encoder.copy_buffer_to_buffer(source, 512, destination, 0, 4),
        Err(Error::OutOfBounds)
    );
    assert_eq!(
        encoder.copy_buffer_to_buffer(source, u64::MAX - 3, destination, 0, 4),
        Err(Error::Overflow)
    );
    encoder
        .copy_buffer_to_buffer(source, 0, destination, 0, 16)
        .unwrap();
    assert_eq!(encoder.finish().unwrap().command_count(), 1);
}

#[test]
fn barrier_destination_is_checked_and_failed_access_does_not_consume_it() {
    let table = ResourceTable::new();
    let target = buffer(
        &table,
        BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
    );
    let destination = buffer(&table, BufferUsage::COPY_DST);
    let ty = BindingType::StorageBuffer { read_only: false };
    let pipeline = compute(&table, ty);
    let bindings = group(&table, target, ty);
    let mut encoder = CommandEncoder::new(&table);
    dispatch(&mut encoder, pipeline, bindings).unwrap();
    encoder
        .resource_barrier(ResourceBarrier::Buffer {
            buffer: target,
            before: BufferAccess::StorageReadWrite,
            after: BufferAccess::CopySource,
        })
        .unwrap();
    assert_eq!(
        encoder.write_buffer(target, 0, &[0; 16]),
        Err(Error::InvalidResourceAccess)
    );
    let mut pass = encoder.begin_compute_pass().unwrap();
    pass.set_pipeline(pipeline).unwrap();
    pass.set_bind_group(0, bindings).unwrap();
    assert_eq!(pass.dispatch(1, 1, 1), Err(Error::InvalidResourceAccess));
    pass.end().unwrap();
    encoder
        .copy_buffer_to_buffer(target, 0, destination, 0, 16)
        .unwrap();
    encoder.write_buffer(target, 0, &[0; 16]).unwrap();
    encoder.finish().unwrap();
}

fn graphics<'r>(table: &'r ResourceTable, ty: BindingType) -> ProgrammableRenderPipelineRef<'r> {
    table
        .define_programmable_render_pipeline(
            ProgrammableRenderPipelineDesc::new(
                shader(table, ShaderStage::Vertex),
                shader(table, ShaderStage::Fragment),
                PipelineLayoutDesc::new(vec![layout(ty)]).unwrap(),
                TextureFormat::Rgba8Unorm,
                None,
                PrimitiveTopology::TriangleList,
                BlendState::REPLACE,
                RasterState::new(CullMode::None, FrontFace::CounterClockwise),
            )
            .unwrap(),
        )
        .unwrap()
}

#[test]
fn render_pass_rejects_storage_write_to_resource_read_by_an_earlier_draw() {
    let table = ResourceTable::new();
    let data = buffer(&table, BufferUsage::STORAGE);
    let read = BindingType::StorageBuffer { read_only: true };
    let write = BindingType::StorageBuffer { read_only: false };
    let reader = graphics(&table, read);
    let writer = graphics(&table, write);
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
    let mut encoder = CommandEncoder::new(&table);
    let mut pass = encoder
        .begin_render_pass(
            RenderPassDesc::new(
                &table,
                target,
                PixelRect::new(0, 0, 8, 8).unwrap(),
                LoadOp::DontCare,
                StoreOp::Store,
            )
            .unwrap(),
        )
        .unwrap();
    pass.set_programmable_pipeline(reader).unwrap();
    pass.set_bind_group(0, group(&table, data, read)).unwrap();
    pass.draw(3, 0).unwrap();
    pass.set_programmable_pipeline(writer).unwrap();
    pass.set_bind_group(0, group(&table, data, write)).unwrap();
    assert_eq!(pass.draw(3, 0), Err(Error::ResourceAccessConflict));
    pass.end().unwrap();
    encoder.finish().unwrap();
}

#[test]
fn nonindexed_draw_ignores_an_unused_index_binding_when_checking_storage_aliases() {
    let table = ResourceTable::new();
    let data = buffer(&table, BufferUsage::STORAGE | BufferUsage::INDEX);
    let write = BindingType::StorageBuffer { read_only: false };
    let pipeline = graphics(&table, write);
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
    let mut encoder = CommandEncoder::new(&table);
    let mut pass = encoder
        .begin_render_pass(
            RenderPassDesc::new(
                &table,
                target,
                PixelRect::new(0, 0, 8, 8).unwrap(),
                LoadOp::DontCare,
                StoreOp::Store,
            )
            .unwrap(),
        )
        .unwrap();
    pass.set_index_buffer(data, 0, IndexFormat::Uint32).unwrap();
    pass.set_programmable_pipeline(pipeline).unwrap();
    pass.set_bind_group(0, group(&table, data, write)).unwrap();
    pass.draw(3, 0).unwrap();
    pass.end().unwrap();
    encoder.finish().unwrap();
}

#[test]
fn solid_draw_does_not_access_or_consume_a_stale_texture_binding() {
    let table = ResourceTable::new();
    let extent = Extent2D::new(8, 8).unwrap();
    let area = PixelRect::new(0, 0, 8, 8).unwrap();
    let source = table
        .define_texture(
            TextureDesc::new(
                TextureFormat::Rgba8Unorm,
                extent,
                TextureUsage::STORAGE | TextureUsage::SAMPLED | TextureUsage::COPY_SRC,
            )
            .unwrap(),
        )
        .unwrap();
    let target = table
        .define_texture(
            TextureDesc::new(
                TextureFormat::Rgba8Unorm,
                extent,
                TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_DST,
            )
            .unwrap(),
        )
        .unwrap();
    let ty = BindingType::StorageTexture {
        format: TextureFormat::Rgba8Unorm,
        access: StorageTextureAccess::WriteOnly,
    };
    let producer = compute(&table, ty);
    let bindings = table
        .define_bind_group(
            BindGroupDesc::new(
                &table,
                layout(ty),
                vec![BindGroupEntry::new(
                    0,
                    BindingResource::Texture(source.id()),
                )],
            )
            .unwrap(),
        )
        .unwrap();
    let vertices = buffer(&table, BufferUsage::VERTEX);
    let solid = table
        .define_render_pipeline(
            RenderPipelineDesc::new(
                TextureFormat::Rgba8Unorm,
                PrimitiveTopology::TriangleList,
                VertexBufferLayout::new(
                    8,
                    vec![VertexAttribute::new(0, VertexFormat::Float32x2, 0)],
                )
                .unwrap(),
                FragmentProgram::Solid,
                BlendState::REPLACE,
                RasterState::new(CullMode::None, FrontFace::CounterClockwise),
            )
            .unwrap(),
        )
        .unwrap();
    let mut encoder = CommandEncoder::new(&table);
    dispatch(&mut encoder, producer, bindings).unwrap();
    encoder
        .resource_barrier(ResourceBarrier::Texture {
            texture: source,
            before: TextureAccess::StorageWrite,
            after: TextureAccess::CopySource,
        })
        .unwrap();
    let mut pass = encoder
        .begin_render_pass(
            RenderPassDesc::new(&table, target, area, LoadOp::DontCare, StoreOp::Store).unwrap(),
        )
        .unwrap();
    pass.set_texture(source).unwrap();
    pass.set_pipeline(solid).unwrap();
    pass.set_vertex_buffer(vertices, 0).unwrap();
    pass.set_uniforms(DrawUniforms::new(
        Transform::identity(),
        Color::rgba(1.0, 1.0, 1.0, 1.0).unwrap(),
    ))
    .unwrap();
    pass.draw(3, 0).unwrap();
    pass.end().unwrap();
    encoder
        .copy_texture_to_texture(source, area, target, area)
        .unwrap();
    encoder.finish().unwrap();
}

#[test]
fn multiple_vertex_slots_validate_layouts_binding_bounds_and_write_aliases() {
    let table = ResourceTable::new();
    let input = |location, format: VertexFormat| VertexBufferLayout::new(format.byte_size(),
        vec![VertexAttribute::new(location, format, 0)]).unwrap();
    let desc = ProgrammableRenderPipelineDesc::new(shader(&table, ShaderStage::Vertex),
        shader(&table, ShaderStage::Fragment), PipelineLayoutDesc::new(vec![]).unwrap(),
        TextureFormat::Rgba8Unorm, None, PrimitiveTopology::TriangleList, BlendState::REPLACE,
        RasterState::new(CullMode::None, FrontFace::CounterClockwise)).unwrap();
    assert!(desc.clone().with_vertex_buffers(vec![input(0, VertexFormat::Float32x2), input(0, VertexFormat::Sint32)]).is_err());
    assert!(desc.clone().with_vertex_buffers((0..9).map(|i| input(i, VertexFormat::Sint32)).collect()).is_err());
    let desc = desc.with_vertex_buffers(vec![input(0, VertexFormat::Float32x2), input(1, VertexFormat::Sint32)]).unwrap();
    let pipeline = table.define_programmable_render_pipeline(desc.clone()).unwrap();
    let target = table.define_texture(TextureDesc::new(TextureFormat::Rgba8Unorm,
        Extent2D::new(4,4).unwrap(), TextureUsage::RENDER_ATTACHMENT).unwrap()).unwrap();
    let vertices = buffer(&table, BufferUsage::VERTEX);
    let values = table.define_buffer(BufferDesc::new(16, BufferUsage::VERTEX | BufferUsage::STORAGE).unwrap()).unwrap();
    let pass_desc = RenderPassDesc::new(&table, target, PixelRect::new(0,0,4,4).unwrap(), LoadOp::DontCare, StoreOp::Store).unwrap();
    let mut encoder = CommandEncoder::new(&table);
    let mut pass = encoder.begin_render_pass(pass_desc).unwrap();
    pass.set_programmable_pipeline(pipeline).unwrap();
    pass.set_vertex_buffer(vertices, 0).unwrap();
    assert_eq!(pass.draw(3,0), Err(Error::VertexBufferNotSet));
    assert_eq!(pass.set_vertex_buffer_slot(8, values, 0), Err(Error::OutOfBounds));
    pass.set_vertex_buffer_slot(1, values, 8).unwrap();
    assert_eq!(pass.draw(3,0), Err(Error::OutOfBounds));
    pass.set_vertex_buffer_slot(1, values, 0).unwrap();
    pass.draw(3,0).unwrap();
    pass.end().unwrap();
    encoder.finish().unwrap();
    let storage_layout = BindGroupLayoutDesc::new(vec![BindGroupLayoutEntry::new(0, ShaderStages::FRAGMENT,
        BindingType::StorageBuffer { read_only: false })]).unwrap();
    let pipeline = table.define_programmable_render_pipeline(ProgrammableRenderPipelineDesc::new(
        shader(&table, ShaderStage::Vertex), shader(&table, ShaderStage::Fragment),
        PipelineLayoutDesc::new(vec![storage_layout.clone()]).unwrap(), TextureFormat::Rgba8Unorm,
        None, PrimitiveTopology::TriangleList, BlendState::REPLACE,
        RasterState::new(CullMode::None, FrontFace::CounterClockwise)).unwrap()
        .with_vertex_buffers(desc.vertex_buffers().to_vec()).unwrap()).unwrap();
    let group = table.define_bind_group(BindGroupDesc::new(&table, storage_layout, vec![BindGroupEntry::new(0,
        BindingResource::Buffer { buffer: values.id(), offset: 0, size: 16 })]).unwrap()).unwrap();
    let mut encoder = CommandEncoder::new(&table);
    let mut pass = encoder.begin_render_pass(pass_desc).unwrap();
    pass.set_programmable_pipeline(pipeline).unwrap();
    pass.set_vertex_buffer(vertices,0).unwrap();
    pass.set_vertex_buffer_slot(1, values,0).unwrap();
    pass.set_bind_group(0,group).unwrap();
    assert_eq!(pass.draw(3,0), Err(Error::ResourceAccessConflict));
}
#[test]
fn storage_texture_views_select_one_mip_and_layer() {
    let table = ResourceTable::new();
    let desc = TextureDesc::new(TextureFormat::Rgba8Unorm, Extent2D::new(8, 8).unwrap(),
        TextureUsage::STORAGE | TextureUsage::SAMPLED).unwrap()
        .with_mip_level_count(4).unwrap().with_array_layer_count(6).unwrap();
    let texture = table.define_texture(desc).unwrap();
    let bindings = layout(BindingType::StorageTexture {
        format: TextureFormat::Rgba8Unorm, access: StorageTextureAccess::WriteOnly,
    });
    let bind = |resource| BindGroupDesc::new(&table, bindings.clone(),
        vec![BindGroupEntry::new(0, resource)]);
    assert!(bind(BindingResource::Texture(texture.id())).is_err());
    let view = |dimension, levels, layers| TextureViewDesc::new(desc, desc.format(),
        dimension, 1, levels, 0, layers).unwrap();
    assert!(bind(BindingResource::TextureView { texture: texture.id(),
        view: view(TextureViewDimension::D2, 1, 1) }).is_ok());
    for invalid in [view(TextureViewDimension::D2, 2, 1),
        view(TextureViewDimension::Cube, 1, 6),
        view(TextureViewDimension::D2Array, 1, 6)] {
        assert!(bind(BindingResource::TextureView { texture: texture.id(), view: invalid }).is_err());
    }
}

#[test]
fn storage_view_writes_require_a_barrier_on_the_written_mip() {
    let table = ResourceTable::new();
    let desc = TextureDesc::new(TextureFormat::Rgba8Unorm, Extent2D::new(8, 8).unwrap(),
        TextureUsage::STORAGE | TextureUsage::SAMPLED).unwrap().with_mip_level_count(4).unwrap();
    let texture = table.define_texture(desc).unwrap();
    let ty = BindingType::StorageTexture { format: desc.format(), access: StorageTextureAccess::WriteOnly };
    let pipeline = compute(&table, ty);
    let view = TextureViewDesc::new(desc, desc.format(), TextureViewDimension::D2, 1, 1, 0, 1).unwrap();
    let bindings = table.define_bind_group(BindGroupDesc::new(&table, layout(ty),
        vec![BindGroupEntry::new(0, BindingResource::TextureView { texture: texture.id(), view })]).unwrap()).unwrap();
    let mut encoder = CommandEncoder::new(&table);
    fn dispatch_checked<'r>(encoder: &mut CommandEncoder<'r, '_>, pipeline: ComputePipelineRef<'r>, bindings: BindGroupRef<'r>) -> Result<()> {
        let mut pass = encoder.begin_compute_pass()?;
        pass.set_pipeline(pipeline)?; pass.set_bind_group(0, bindings)?;
        let result = pass.dispatch(1,1,1); pass.end()?; result
    }
    dispatch_checked(&mut encoder, pipeline, bindings).unwrap();
    assert_eq!(dispatch_checked(&mut encoder, pipeline, bindings), Err(Error::MissingBarrier));
    encoder.resource_barrier(ResourceBarrier::TextureMip { texture, mip_level:0,
        before:TextureAccess::StorageWrite, after:TextureAccess::Sampled }).unwrap();
    assert_eq!(dispatch_checked(&mut encoder, pipeline, bindings), Err(Error::MissingBarrier));
    assert_eq!(encoder.resource_barrier(ResourceBarrier::TextureMip { texture, mip_level:1,
        before:TextureAccess::Sampled, after:TextureAccess::StorageWrite }), Err(Error::InvalidResourceAccess));
    encoder.resource_barrier(ResourceBarrier::TextureMip { texture, mip_level:1,
        before:TextureAccess::StorageWrite, after:TextureAccess::StorageWrite }).unwrap();
    dispatch_checked(&mut encoder, pipeline, bindings).unwrap();
    encoder.finish().unwrap();
}
