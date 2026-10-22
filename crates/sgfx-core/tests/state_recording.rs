//! Backend-independent state reduction keeps validation and draw/pass boundaries.
use sgfx_core::ir::*;

struct Scene {
    table: ResourceTable,
    target: TextureId,
    texture: TextureId,
    buffer: BufferId,
    pipeline: RenderPipelineId,
    sampler: SamplerId,
}
impl Scene {
    fn new() -> Self {
        let table = ResourceTable::new();
        #[cfg(feature = "backend-abi")]
        table.enable_abi_commands();
        let target = table
            .define_texture(
                TextureDesc::new(
                    TextureFormat::Bgra8Unorm,
                    Extent2D::new(16, 16).unwrap(),
                    TextureUsage::RENDER_ATTACHMENT,
                )
                .unwrap(),
            )
            .unwrap()
            .id();
        let texture = table
            .define_texture(
                TextureDesc::new(
                    TextureFormat::Bgra8Unorm,
                    Extent2D::new(1, 1).unwrap(),
                    TextureUsage::SAMPLED,
                )
                .unwrap(),
            )
            .unwrap()
            .id();
        let buffer = table
            .define_buffer(BufferDesc::new(128, BufferUsage::VERTEX | BufferUsage::INDEX).unwrap())
            .unwrap()
            .id();
        let pipeline = table
            .define_render_pipeline(
                RenderPipelineDesc::new(
                    TextureFormat::Bgra8Unorm,
                    PrimitiveTopology::TriangleList,
                    VertexBufferLayout::new(
                        16,
                        vec![
                            VertexAttribute::new(0, VertexFormat::Float32x2, 0),
                            VertexAttribute::new(1, VertexFormat::Float32x2, 8),
                        ],
                    )
                    .unwrap(),
                    FragmentProgram::Texture(TextureSampleMode::Rgba),
                    BlendState::REPLACE,
                    RasterState::new(CullMode::None, FrontFace::CounterClockwise),
                )
                .unwrap(),
            )
            .unwrap()
            .id();
        let sampler = table
            .define_sampler(SamplerDesc::new(
                FilterMode::Linear,
                FilterMode::Linear,
                AddressMode::ClampToEdge,
                AddressMode::ClampToEdge,
            ))
            .unwrap()
            .id();
        Self {
            table,
            target,
            texture,
            buffer,
            pipeline,
            sampler,
        }
    }
    fn desc(&self) -> RenderPassDesc<'_> {
        RenderPassDesc::new(
            &self.table,
            self.table.texture_ref(self.target).unwrap(),
            area(),
            LoadOp::Load,
            StoreOp::Store,
        )
        .unwrap()
    }
    fn bind<'r>(&'r self, pass: &mut RenderPassEncoder<'_, 'r, '_>) {
        pass.set_pipeline(self.table.render_pipeline_ref(self.pipeline).unwrap())
            .unwrap();
        let buffer = self.table.buffer_ref(self.buffer).unwrap();
        pass.set_vertex_buffer(buffer, 0).unwrap();
        pass.set_vertex_buffer_slot(1, buffer, 0).unwrap();
        pass.set_index_buffer(buffer, 0, IndexFormat::Uint16)
            .unwrap();
        pass.set_texture(self.table.texture_ref(self.texture).unwrap())
            .unwrap();
        pass.set_sampler(self.table.sampler_ref(self.sampler).unwrap())
            .unwrap();
        pass.set_uniforms(uniforms()).unwrap();
        pass.set_scissor(Some(area())).unwrap();
        pass.set_viewport(Viewport::new(0., 0., 16., 16., 0., 1.).unwrap())
            .unwrap();
    }
}
fn area() -> PixelRect {
    PixelRect::new(0, 0, 16, 16).unwrap()
}
fn uniforms() -> DrawUniforms {
    DrawUniforms::new(Transform::identity(), Color::rgba(1., 1., 1., 1.).unwrap())
}

#[test]
fn repeated_bindings_leave_both_draws_and_only_one_copy_of_each_state() {
    let scene = Scene::new();
    let mut encoder = CommandEncoder::new(&scene.table);
    let mut pass = encoder.begin_render_pass(scene.desc()).unwrap();
    for _ in 0..2 {
        scene.bind(&mut pass);
        pass.draw(3, 0).unwrap();
    }
    pass.end().unwrap();
    let commands = encoder.finish().unwrap();
    assert_eq!(commands.command_count(), 13);
    assert_eq!(
        commands
            .commands()
            .iter()
            .filter(|command| matches!(command, Command::Draw { .. }))
            .count(),
        2
    );
}

#[test]
fn each_pass_records_its_own_initial_state() {
    let scene = Scene::new();
    let mut encoder = CommandEncoder::new(&scene.table);
    for _ in 0..2 {
        let mut pass = encoder.begin_render_pass(scene.desc()).unwrap();
        scene.bind(&mut pass);
        scene.bind(&mut pass);
        pass.end().unwrap();
    }
    assert_eq!(encoder.finish().unwrap().command_count(), 22);
}

#[test]
fn changed_offsets_formats_scissors_and_viewports_survive_reduction() {
    let scene = Scene::new();
    let mut encoder = CommandEncoder::new(&scene.table);
    let mut pass = encoder.begin_render_pass(scene.desc()).unwrap();
    let buffer = scene.table.buffer_ref(scene.buffer).unwrap();
    for offset in [0, 0, 16, 16, 0] {
        pass.set_vertex_buffer(buffer, offset).unwrap();
    }
    for format in [
        IndexFormat::Uint16,
        IndexFormat::Uint16,
        IndexFormat::Uint32,
        IndexFormat::Uint32,
    ] {
        pass.set_index_buffer(buffer, 0, format).unwrap();
    }
    for clip in [Some(area()), Some(area()), None, None, Some(area())] {
        pass.set_scissor(clip).unwrap();
    }
    for x in [0., 0., 1., 1.] {
        pass.set_viewport(Viewport::new(x, 0., 8., 8., 0., 1.).unwrap())
            .unwrap();
    }
    assert_eq!(
        pass.set_scissor(Some(PixelRect::new(15, 0, 8, 8).unwrap())),
        Err(Error::OutOfBounds)
    );
    pass.set_scissor(Some(area())).unwrap();
    pass.end().unwrap();
    let commands = encoder.finish().unwrap();
    assert_eq!(commands.command_count(), 12); // begin/end, 3 vertex, 2 index, 3 scissor, 2 viewport.
}

#[test]
fn retired_resources_are_still_rejected_even_when_the_binding_matches() {
    let scene = Scene::new();
    let mut encoder = CommandEncoder::new(&scene.table);
    let mut pass = encoder.begin_render_pass(scene.desc()).unwrap();
    let buffer = scene.table.buffer_ref(scene.buffer).unwrap();
    let texture = scene.table.texture_ref(scene.texture).unwrap();
    scene.bind(&mut pass);
    scene.table.release_buffer(scene.buffer).unwrap();
    scene.table.release_texture(scene.texture).unwrap();
    assert_eq!(
        pass.set_vertex_buffer(buffer, 0),
        Err(Error::InvalidDescriptor)
    );
    assert_eq!(
        pass.set_vertex_buffer_slot(1, buffer, 0),
        Err(Error::InvalidDescriptor)
    );
    assert_eq!(
        pass.set_index_buffer(buffer, 0, IndexFormat::Uint16),
        Err(Error::InvalidDescriptor)
    );
    assert_eq!(pass.set_texture(texture), Err(Error::InvalidDescriptor));
    pass.end().unwrap();
}

#[test]
fn uniform_and_viewport_reduction_preserves_signed_zero_bits() {
    let scene = Scene::new();
    let mut encoder = CommandEncoder::new(&scene.table);
    let mut pass = encoder.begin_render_pass(scene.desc()).unwrap();
    let mut columns = Transform::identity().columns();
    for zero in [0., 0., -0., -0.] {
        columns[1] = zero;
        pass.set_uniforms(DrawUniforms::new(
            Transform::from_columns(columns).unwrap(),
            uniforms().color(),
        ))
        .unwrap();
        pass.set_viewport(Viewport::new(zero, 0., 8., 8., 0., 1.).unwrap())
            .unwrap();
    }
    pass.end().unwrap();
    assert_eq!(encoder.finish().unwrap().command_count(), 6);
}

#[test]
fn compute_state_is_reduced_but_retired_pipelines_and_groups_still_fail() {
    let scene = Scene::new();
    let table = &scene.table;
    let shader = table
        .define_shader_module(
            ShaderModuleDesc::wgsl("@compute @workgroup_size(1) fn main() {}".into()).unwrap(),
        )
        .unwrap();
    let entry = ShaderEntryPoint::new(shader, ShaderStage::Compute, "main".into()).unwrap();
    let pipeline = table
        .define_compute_pipeline(
            ComputePipelineDesc::new(entry, PipelineLayoutDesc::new(vec![]).unwrap()).unwrap(),
        )
        .unwrap();
    let group = table
        .define_bind_group(
            BindGroupDesc::new(table, BindGroupLayoutDesc::new(vec![]).unwrap(), vec![]).unwrap(),
        )
        .unwrap();
    let mut encoder = CommandEncoder::new(table);
    let mut pass = encoder.begin_compute_pass().unwrap();
    for _ in 0..2 {
        pass.set_pipeline(pipeline).unwrap();
        pass.set_bind_group(0, group).unwrap();
        pass.dispatch(1, 1, 1).unwrap();
    }
    table.release_compute_pipeline(pipeline.id()).unwrap();
    table.release_bind_group(group.id()).unwrap();
    assert_eq!(pass.set_pipeline(pipeline), Err(Error::InvalidDescriptor));
    assert_eq!(pass.set_bind_group(0, group), Err(Error::InvalidDescriptor));
    pass.end().unwrap();
    assert_eq!(encoder.finish().unwrap().command_count(), 6);
}

#[test]
fn programmable_and_fixed_pipeline_switches_keep_their_order() {
    let scene = Scene::new();
    let table = &scene.table;
    let module = table.define_shader_module(ShaderModuleDesc::wgsl("@vertex fn vs() -> @builtin(position) vec4f { return vec4f(0.); } @fragment fn fs() -> @location(0) vec4f { return vec4f(1.); }".into()).unwrap()).unwrap();
    let pipeline = table
        .define_programmable_render_pipeline(
            ProgrammableRenderPipelineDesc::new(
                ShaderEntryPoint::new(module, ShaderStage::Vertex, "vs".into()).unwrap(),
                ShaderEntryPoint::new(module, ShaderStage::Fragment, "fs".into()).unwrap(),
                PipelineLayoutDesc::new(vec![]).unwrap(),
                TextureFormat::Bgra8Unorm,
                None,
                PrimitiveTopology::TriangleList,
                BlendState::REPLACE,
                RasterState::new(CullMode::None, FrontFace::CounterClockwise),
            )
            .unwrap(),
        )
        .unwrap();
    let group = table
        .define_bind_group(
            BindGroupDesc::new(table, BindGroupLayoutDesc::new(vec![]).unwrap(), vec![]).unwrap(),
        )
        .unwrap();
    let fixed = table.render_pipeline_ref(scene.pipeline).unwrap();
    let mut encoder = CommandEncoder::new(table);
    let mut pass = encoder.begin_render_pass(scene.desc()).unwrap();
    pass.set_pipeline(fixed).unwrap();
    pass.set_pipeline(fixed).unwrap();
    pass.set_programmable_pipeline(pipeline).unwrap();
    pass.set_programmable_pipeline(pipeline).unwrap();
    pass.set_bind_group(0, group).unwrap();
    pass.set_bind_group(0, group).unwrap();
    pass.set_pipeline(fixed).unwrap();
    pass.set_programmable_pipeline(pipeline).unwrap();
    pass.end().unwrap();
    let commands = encoder.finish().unwrap();
    assert_eq!(commands.command_count(), 7);
    assert!(matches!(commands.commands()[1], Command::SetPipeline(_)));
    assert!(matches!(
        commands.commands()[2],
        Command::SetProgrammablePipeline(_)
    ));
    assert!(matches!(
        commands.commands()[3],
        Command::SetBindGroup { .. }
    ));
    assert!(matches!(commands.commands()[4], Command::SetPipeline(_)));
    assert!(matches!(
        commands.commands()[5],
        Command::SetProgrammablePipeline(_)
    ));
}
