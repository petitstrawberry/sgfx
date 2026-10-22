use sgfx_core::ir::*;

fn record(vertex_bytes: u64, instance_bytes: u64, indexed: bool) -> Result<()> {
    let table = ResourceTable::new();
    let shader = table.define_shader_module(ShaderModuleDesc::wgsl("@vertex fn vs()->@builtin(position) vec4<f32>{return vec4<f32>(0.0);} @fragment fn fs()->@location(0) vec4<f32>{return vec4<f32>(1.0);}".into())?)?;
    let pipeline = table.define_programmable_render_pipeline(
        ProgrammableRenderPipelineDesc::new(
            ShaderEntryPoint::new(shader, ShaderStage::Vertex, "vs".into())?,
            ShaderEntryPoint::new(shader, ShaderStage::Fragment, "fs".into())?,
            PipelineLayoutDesc::new(vec![])?,
            TextureFormat::Rgba8Unorm,
            None,
            PrimitiveTopology::TriangleList,
            BlendState::REPLACE,
            RasterState::new(CullMode::None, FrontFace::CounterClockwise),
        )?
        .with_vertex_buffers(vec![
            VertexBufferLayout::new(8, vec![VertexAttribute::new(0, VertexFormat::Float32x2, 0)])?,
            VertexBufferLayout::new(4, vec![VertexAttribute::new(1, VertexFormat::Float32, 0)])?
                .with_step_mode(VertexStepMode::Instance),
        ])?,
    )?;
    let vertex = table.define_buffer(BufferDesc::new(vertex_bytes, BufferUsage::VERTEX)?)?;
    let instance = table.define_buffer(BufferDesc::new(instance_bytes, BufferUsage::VERTEX)?)?;
    let indices = table.define_buffer(BufferDesc::new(12, BufferUsage::INDEX)?)?;
    let target = table.define_texture(TextureDesc::new(
        TextureFormat::Rgba8Unorm,
        Extent2D::new(8, 8)?,
        TextureUsage::RENDER_ATTACHMENT,
    )?)?;
    let mut commands = CommandEncoder::new(&table);
    let mut pass = commands.begin_render_pass(RenderPassDesc::new(
        &table,
        target,
        PixelRect::new(0, 0, 8, 8)?,
        LoadOp::Load,
        StoreOp::Store,
    )?)?;
    pass.set_programmable_pipeline(pipeline)?;
    pass.set_vertex_buffer(vertex, 0)?;
    pass.set_vertex_buffer_slot(1, instance, 0)?;
    if indexed {
        pass.set_index_buffer(indices, 0, IndexFormat::Uint16)?;
        pass.draw_indexed_instanced(6, 0, 0, 2, 2)?;
    } else {
        pass.draw_instanced(6, 0, 2, 2)?;
    }
    pass.end()?;
    commands.finish()?;
    Ok(())
}

#[test]
fn instance_bounds_use_first_instance_and_instance_count_independently_of_vertices() {
    assert!(record(48, 16, false).is_ok());
    assert!(record(48, 15, false).is_err());
    assert!(record(47, 16, false).is_err());
    // The IR cannot inspect opaque index bytes, but it knows every instance index.
    assert!(record(8, 16, true).is_ok());
    assert!(record(8, 15, true).is_err());
}
