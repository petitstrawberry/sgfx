//! Public recording invariants, independent of any GPU or backend.

use sgfx_core::ir::{
    AddressMode, BlendState, BufferDesc, BufferUsage, Color, Command, CommandEncoder,
    CompareFunction, CullMode, DepthLoadOp, DepthState, DrawUniforms, Error, Extent2D, FilterMode,
    FragmentProgram, FrontFace, IndexFormat, LoadOp, MAX_BUFFERS, MAX_COMMANDS, PixelRect,
    PrimitiveTopology, RasterState, RenderPassDesc, RenderPipelineDesc, ResourceTable, SamplerDesc,
    StoreOp, TextureDesc, TextureFormat, TextureUsage, TextureWrite, Transform, VertexAttribute,
    VertexBufferLayout, VertexFormat,
};

fn extent() -> Extent2D {
    Extent2D::new(8, 8).expect("nonzero extent")
}

fn area() -> PixelRect {
    PixelRect::new(0, 0, 8, 8).expect("full render area")
}

fn color_target(table: &ResourceTable) -> sgfx_core::ir::TextureRef<'_> {
    table
        .define_texture(
            TextureDesc::new(
                TextureFormat::Bgra8Unorm,
                extent(),
                TextureUsage::RENDER_ATTACHMENT,
            )
            .expect("color descriptor"),
        )
        .expect("color target")
}

fn pipeline(format: TextureFormat) -> sgfx_core::ir::Result<RenderPipelineDesc> {
    RenderPipelineDesc::new(
        format,
        PrimitiveTopology::TriangleList,
        VertexBufferLayout::new(8, vec![VertexAttribute::new(0, VertexFormat::Float32x2, 0)])
            .expect("position layout"),
        FragmentProgram::Solid,
        BlendState::REPLACE,
        RasterState::new(CullMode::None, FrontFace::CounterClockwise),
    )
}

fn uniforms() -> DrawUniforms {
    DrawUniforms::new(
        Transform::identity(),
        Color::rgba(1.0, 1.0, 1.0, 1.0).expect("finite color"),
    )
}

#[test]
fn rejects_depth_color_target() {
    assert_eq!(
        pipeline(TextureFormat::Depth32Float),
        Err(Error::InvalidDescriptor)
    );
}

#[test]
fn rejects_depth_pipeline_without_attachment() {
    let table = ResourceTable::new();
    let target = color_target(&table);
    let depth_pipeline = table
        .define_render_pipeline(
            pipeline(TextureFormat::Bgra8Unorm)
                .expect("color pipeline")
                .with_depth_state(DepthState::new(
                    TextureFormat::Depth32Float,
                    CompareFunction::Less,
                    true,
                ))
                .expect("depth state"),
        )
        .expect("depth pipeline");
    let desc = RenderPassDesc::new(&table, target, area(), LoadOp::Load, StoreOp::Store)
        .expect("color-only pass");
    let mut encoder = CommandEncoder::new(&table);
    let mut pass = encoder.begin_render_pass(desc).expect("begin pass");
    assert_eq!(
        pass.set_pipeline(depth_pipeline),
        Err(Error::PipelineTargetMismatch)
    );
    pass.end().expect("end rejected-bind pass");
    assert_eq!(encoder.finish().expect("finish").command_count(), 2);
}

#[test]
fn accepts_matching_depth_and_validates_its_clear() {
    let table = ResourceTable::new();
    let target = color_target(&table);
    let depth = table
        .define_texture(
            TextureDesc::new(
                TextureFormat::Depth32Float,
                extent(),
                TextureUsage::RENDER_ATTACHMENT,
            )
            .expect("depth descriptor"),
        )
        .expect("depth target");
    let desc = RenderPassDesc::new(&table, target, area(), LoadOp::Load, StoreOp::Store)
        .expect("color pass");
    for clear in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
        assert!(matches!(
            desc.with_depth_attachment(&table, depth, DepthLoadOp::Clear(clear), StoreOp::Store),
            Err(Error::InvalidDescriptor)
        ));
    }
    let desc = desc
        .with_depth_attachment(&table, depth, DepthLoadOp::Clear(1.0), StoreOp::Store)
        .expect("matching depth attachment");
    let pipeline = table
        .define_render_pipeline(
            pipeline(TextureFormat::Bgra8Unorm)
                .expect("color pipeline")
                .with_depth_state(DepthState::new(
                    TextureFormat::Depth32Float,
                    CompareFunction::LessEqual,
                    true,
                ))
                .expect("depth state"),
        )
        .expect("depth pipeline");
    let mut encoder = CommandEncoder::new(&table);
    let mut pass = encoder.begin_render_pass(desc).expect("begin pass");
    pass.set_pipeline(pipeline)
        .expect("matching depth pipeline");
    pass.end().expect("end pass");
    assert_eq!(encoder.finish().expect("finish").command_count(), 3);
}

#[test]
fn depth_attachment_does_not_require_depth_testing_in_the_ir() {
    let table = ResourceTable::new();
    let target = color_target(&table);
    let depth = table
        .define_texture(
            TextureDesc::new(
                TextureFormat::Depth32Float,
                extent(),
                TextureUsage::RENDER_ATTACHMENT,
            )
            .expect("depth descriptor"),
        )
        .expect("depth target");
    let desc = RenderPassDesc::new(&table, target, area(), LoadOp::Load, StoreOp::Store)
        .expect("color pass")
        .with_depth_attachment(&table, depth, DepthLoadOp::Load, StoreOp::Store)
        .expect("depth attachment");
    let pipeline = table
        .define_render_pipeline(pipeline(TextureFormat::Bgra8Unorm).expect("color descriptor"))
        .expect("color-only pipeline");
    let mut encoder = CommandEncoder::new(&table);
    let mut pass = encoder.begin_render_pass(desc).expect("begin pass");
    pass.set_pipeline(pipeline).expect("depth testing disabled");
    pass.end().expect("end pass");
    assert_eq!(encoder.finish().expect("finish").command_count(), 3);
}

#[test]
fn persistent_ids_survive_table_moves_but_do_not_alias_other_tables() {
    let table = ResourceTable::new();
    let texture = color_target(&table).id();
    let buffer = table
        .define_buffer(BufferDesc::new(24, BufferUsage::VERTEX).expect("buffer descriptor"))
        .expect("buffer")
        .id();
    let sampler = table
        .define_sampler(SamplerDesc::new(
            FilterMode::Linear,
            FilterMode::Nearest,
            AddressMode::ClampToEdge,
            AddressMode::Repeat,
        ))
        .expect("sampler")
        .id();
    let pipeline = table
        .define_render_pipeline(pipeline(TextureFormat::Bgra8Unorm).expect("pipeline descriptor"))
        .expect("pipeline")
        .id();
    let moved = table;
    assert_eq!(
        moved.texture_ref(texture).expect("moved texture").id(),
        texture
    );
    assert_eq!(moved.buffer_ref(buffer).expect("moved buffer").id(), buffer);
    assert_eq!(
        moved.sampler_ref(sampler).expect("moved sampler").id(),
        sampler
    );
    assert_eq!(
        moved
            .render_pipeline_ref(pipeline)
            .expect("moved pipeline")
            .id(),
        pipeline
    );
    let other = ResourceTable::new();
    assert_eq!(
        color_target(&other).slot(),
        moved.texture_ref(texture).expect("texture").slot()
    );
    assert_eq!(
        other.texture_ref(texture),
        Err(Error::ResourceTableMismatch)
    );
    assert_eq!(other.buffer_ref(buffer), Err(Error::ResourceTableMismatch));
    assert_eq!(
        other.sampler_ref(sampler),
        Err(Error::ResourceTableMismatch)
    );
    assert_eq!(
        other.render_pipeline_ref(pipeline),
        Err(Error::ResourceTableMismatch)
    );
}

#[test]
fn rejects_invalid_values_and_depth_usages() {
    assert_eq!(Extent2D::new(0, 1), Err(Error::InvalidValue));
    assert_eq!(PixelRect::new(u32::MAX, 0, 1, 1), Err(Error::InvalidValue));
    assert_eq!(
        Color::rgba(f32::NAN, 0.0, 0.0, 1.0),
        Err(Error::InvalidValue)
    );
    let mut matrix = Transform::identity().columns();
    matrix[7] = f32::INFINITY;
    assert_eq!(Transform::from_columns(matrix), Err(Error::InvalidValue));
    assert_eq!(
        TextureDesc::new(
            TextureFormat::Depth32Float,
            extent(),
            TextureUsage::RENDER_ATTACHMENT | TextureUsage::SAMPLED
        ),
        Err(Error::InvalidDescriptor)
    );
    assert_eq!(
        VertexBufferLayout::new(
            8,
            vec![
                VertexAttribute::new(0, VertexFormat::Float32x2, 0),
                VertexAttribute::new(0, VertexFormat::Float32x2, 0),
            ]
        ),
        Err(Error::InvalidDescriptor)
    );
}

#[test]
fn invalid_uploads_do_not_record_commands_or_copy_borrowed_bytes() {
    let table = ResourceTable::new();
    let buffer = table
        .define_buffer(BufferDesc::new(4, BufferUsage::COPY_DST).expect("descriptor"))
        .expect("buffer");
    let data = [1, 2, 3, 4];
    let mut encoder = CommandEncoder::new(&table);
    assert_eq!(
        encoder.write_buffer(buffer, 0, &[]),
        Err(Error::InvalidValue)
    );
    assert_eq!(
        encoder.write_buffer(buffer, u64::MAX, &data),
        Err(Error::Overflow)
    );
    assert_eq!(
        encoder.write_buffer(buffer, 1, &data),
        Err(Error::OutOfBounds)
    );
    encoder
        .write_buffer(buffer, 0, &data)
        .expect("valid upload");
    let commands = encoder.finish().expect("finish");
    assert_eq!(commands.command_count(), 1);
    let Command::WriteBuffer { data: recorded, .. } = &commands.commands()[0] else {
        panic!("expected borrowed upload");
    };
    assert!(core::ptr::eq(recorded.as_ptr(), data.as_ptr()));
}

#[test]
fn texture_upload_requires_only_the_last_rows_actual_pixels() {
    let table = ResourceTable::new();
    let texture = table
        .define_texture(
            TextureDesc::new(
                TextureFormat::Rgba8Unorm,
                Extent2D::new(2, 2).expect("extent"),
                TextureUsage::COPY_DST,
            )
            .expect("descriptor"),
        )
        .expect("texture");
    let rect = PixelRect::new(0, 0, 2, 2).expect("rectangle");
    let data = [0; 20]; // First row: 8 pixel bytes + 4 padding bytes; last row: 8 pixel bytes.
    let mut encoder = CommandEncoder::new(&table);
    assert_eq!(
        encoder.write_texture(texture, TextureWrite::new(rect, 7, &data).expect("layout")),
        Err(Error::InvalidValue)
    );
    assert_eq!(
        encoder.write_texture(
            texture,
            TextureWrite::new(rect, 12, &data[..19]).expect("layout")
        ),
        Err(Error::OutOfBounds)
    );
    encoder
        .write_texture(texture, TextureWrite::new(rect, 12, &data).expect("layout"))
        .expect("last row does not require trailing padding");
    assert_eq!(encoder.finish().expect("finish").command_count(), 1);
}

#[test]
fn self_copy_rejects_overlap_but_accepts_disjoint_rectangles() {
    let table = ResourceTable::new();
    let texture = table
        .define_texture(
            TextureDesc::new(
                TextureFormat::Rgba8Unorm,
                extent(),
                TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            )
            .expect("descriptor"),
        )
        .expect("texture");
    let left = PixelRect::new(0, 0, 4, 4).expect("left");
    let right = PixelRect::new(4, 0, 4, 4).expect("right");
    let mut encoder = CommandEncoder::new(&table);
    assert_eq!(
        encoder.copy_texture_to_texture(texture, left, texture, left),
        Err(Error::InvalidValue)
    );
    encoder
        .copy_texture_to_texture(texture, left, texture, right)
        .expect("disjoint copy");
    assert_eq!(encoder.finish().expect("finish").command_count(), 1);
}

#[test]
fn invalid_bindings_preserve_valid_draw_state() {
    let table = ResourceTable::new();
    let target = color_target(&table);
    let good = table
        .define_render_pipeline(pipeline(TextureFormat::Bgra8Unorm).expect("good descriptor"))
        .expect("good pipeline");
    let wrong = table
        .define_render_pipeline(
            pipeline(TextureFormat::Rgba8Unorm).expect("other color descriptor"),
        )
        .expect("wrong pipeline");
    let vertices = table
        .define_buffer(BufferDesc::new(24, BufferUsage::VERTEX).expect("vertex descriptor"))
        .expect("vertices");
    let indices = table
        .define_buffer(BufferDesc::new(6, BufferUsage::INDEX).expect("index descriptor"))
        .expect("indices");
    let desc =
        RenderPassDesc::new(&table, target, area(), LoadOp::Load, StoreOp::Store).expect("pass");
    let mut encoder = CommandEncoder::new(&table);
    let mut pass = encoder.begin_render_pass(desc).expect("begin pass");
    assert_eq!(pass.draw(3, 0), Err(Error::PipelineNotSet));
    pass.set_pipeline(good).expect("valid pipeline");
    assert_eq!(pass.set_pipeline(wrong), Err(Error::PipelineTargetMismatch));
    assert_eq!(pass.draw(3, 0), Err(Error::UniformsNotSet));
    pass.set_uniforms(uniforms()).expect("uniforms");
    assert_eq!(pass.draw(3, 0), Err(Error::VertexBufferNotSet));
    pass.set_vertex_buffer(vertices, 0).expect("vertices");
    assert_eq!(pass.draw(3, 1), Err(Error::OutOfBounds));
    pass.draw(3, 0).expect("retained valid bindings");
    assert_eq!(pass.draw_indexed(3, 0, 0), Err(Error::IndexBufferNotSet));
    assert_eq!(
        pass.set_index_buffer(indices, 1, IndexFormat::Uint16),
        Err(Error::InvalidValue)
    );
    pass.set_index_buffer(indices, 0, IndexFormat::Uint16)
        .expect("indices");
    assert_eq!(pass.draw_indexed(3, 1, 0), Err(Error::OutOfBounds));
    pass.draw_indexed(3, 0, 0).expect("valid indexed draw");
    pass.end().expect("end pass");
    let commands = encoder.finish().expect("finish");
    assert_eq!(
        commands
            .commands()
            .iter()
            .filter(|command| matches!(command, Command::SetPipeline(_)))
            .count(),
        1
    );
}

#[test]
fn command_limit_preserves_the_pass_end_slot() {
    let table = ResourceTable::new();
    let buffer = table
        .define_buffer(BufferDesc::new(1, BufferUsage::COPY_DST).expect("buffer descriptor"))
        .expect("buffer");
    let target = color_target(&table);
    let desc =
        RenderPassDesc::new(&table, target, area(), LoadOp::Load, StoreOp::Store).expect("pass");
    let mut encoder = CommandEncoder::new(&table);
    for _ in 0..MAX_COMMANDS - 2 {
        encoder.write_buffer(buffer, 0, &[0]).expect("fill encoder");
    }
    let mut pass = encoder
        .begin_render_pass(desc)
        .expect("reserved begin and end slots");
    assert_eq!(
        pass.set_uniforms(uniforms()),
        Err(Error::CommandLimitExceeded)
    );
    pass.end().expect("reserved end slot");
    assert_eq!(
        encoder
            .finish()
            .expect("finish full encoder")
            .command_count(),
        MAX_COMMANDS
    );
}

#[test]
fn resource_limit_keeps_existing_descriptors_resolvable() {
    let table = ResourceTable::new();
    let descriptor = BufferDesc::new(4, BufferUsage::COPY_DST).expect("descriptor");
    let first = table.define_buffer(descriptor).expect("first buffer").id();
    for _ in 1..MAX_BUFFERS {
        table.define_buffer(descriptor).expect("fill table");
    }
    assert_eq!(
        table.define_buffer(descriptor),
        Err(Error::ResourceLimitExceeded)
    );
    assert_eq!(
        table.buffer(table.buffer_ref(first).expect("resolve first")),
        Ok(descriptor)
    );
}

#[test]
fn dropping_a_pass_without_end_does_not_finish_the_stream() {
    let table = ResourceTable::new();
    let target = color_target(&table);
    let desc =
        RenderPassDesc::new(&table, target, area(), LoadOp::Load, StoreOp::Store).expect("pass");
    let mut encoder = CommandEncoder::new(&table);
    {
        let _pass = encoder.begin_render_pass(desc).expect("begin pass");
    }
    assert!(matches!(
        encoder.finish(),
        Err(Error::RenderPassStillActive)
    ));
}
