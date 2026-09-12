use sgfx_core::ir::*;

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
            16,
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
