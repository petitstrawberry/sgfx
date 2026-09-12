//! Real-device programmable rendering, compute, bindings and error containment.

use super::execution::headless_device;
use super::*;
use crate::Error;
use ir::*;
use sgfx_core::backend::{Completion, CompletionStatus};

fn shader<'a>(table: &'a ResourceTable, source: &str) -> ShaderModuleRef<'a> {
    table
        .define_shader_module(ShaderModuleDesc::wgsl(source.into()).expect("WGSL descriptor"))
        .expect("shader")
}
fn entry(module: ShaderModuleRef<'_>, stage: ShaderStage, name: &str) -> ShaderEntryPoint {
    ShaderEntryPoint::new(module, stage, name.into()).expect("entry point")
}
fn buffer_group<'a>(
    table: &'a ResourceTable,
    layout: &BindGroupLayoutDesc,
    buffers: &[(BufferRef<'a>, u64)],
) -> BindGroupRef<'a> {
    let entries = buffers
        .iter()
        .enumerate()
        .map(|(binding, (buffer, size))| {
            BindGroupEntry::new(
                binding as u32,
                BindingResource::Buffer {
                    buffer: buffer.id(),
                    offset: 0,
                    size: *size,
                },
            )
        })
        .collect();
    table
        .define_bind_group(
            BindGroupDesc::new(table, layout.clone(), entries).expect("group descriptor"),
        )
        .expect("group")
}

#[test]
fn compute_storage_uniform_bindings_barriers_and_ordered_uploads_produce_readback() {
    let _guard = HEADLESS_WGPU_TEST_LOCK.lock().expect("lock WGPU tests");
    let Some(device) = headless_device() else {
        return;
    };
    let context = device.create_context();
    let table = Rc::new(ResourceTable::new());
    let shader = shader(
        &table,
        r#"
        @group(0) @binding(0) var<uniform> input: vec4<u32>;
        @group(0) @binding(1) var<storage, read_write> output: array<u32>;
        @compute @workgroup_size(4)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) { output[id.x] = input.x + id.x; }
    "#,
    );
    let layout = BindGroupLayoutDesc::new(vec![
        BindGroupLayoutEntry::new(0, ShaderStages::COMPUTE, BindingType::UniformBuffer),
        BindGroupLayoutEntry::new(
            1,
            ShaderStages::COMPUTE,
            BindingType::StorageBuffer { read_only: false },
        ),
    ])
    .unwrap();
    let pipeline = table
        .define_compute_pipeline(
            ComputePipelineDesc::new(
                entry(shader, ShaderStage::Compute, "main"),
                PipelineLayoutDesc::new(vec![layout.clone()]).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    let input = table
        .define_buffer(BufferDesc::new(16, BufferUsage::UNIFORM | BufferUsage::COPY_DST).unwrap())
        .unwrap();
    let output = table
        .define_buffer(BufferDesc::new(32, BufferUsage::STORAGE | BufferUsage::COPY_SRC).unwrap())
        .unwrap();
    let snapshot = table
        .define_buffer(BufferDesc::new(32, BufferUsage::COPY_DST | BufferUsage::COPY_SRC).unwrap())
        .unwrap();
    let group = buffer_group(&table, &layout, &[(input, 16), (output, 32)]);
    let first = [5_u32, 0, 0, 0];
    let second = [91_u32, 0, 0, 0];
    let mut encoder = CommandEncoder::new(&table);
    encoder
        .write_buffer(input, 0, bytemuck::cast_slice(&first))
        .unwrap();
    encoder
        .resource_barrier(ResourceBarrier::Buffer {
            buffer: input,
            before: BufferAccess::CopyDestination,
            after: BufferAccess::Uniform,
        })
        .unwrap();
    let mut pass = encoder.begin_compute_pass().unwrap();
    pass.set_pipeline(pipeline).unwrap();
    pass.set_bind_group(0, group).unwrap();
    pass.dispatch(2, 1, 1).unwrap();
    pass.end().unwrap();
    encoder
        .resource_barrier(ResourceBarrier::Buffer {
            buffer: output,
            before: BufferAccess::StorageReadWrite,
            after: BufferAccess::CopySource,
        })
        .unwrap();
    encoder
        .copy_buffer_to_buffer(output, 0, snapshot, 0, 32)
        .unwrap();
    encoder
        .write_buffer(input, 0, bytemuck::cast_slice(&second))
        .unwrap();
    let mut pass = encoder.begin_compute_pass().unwrap();
    pass.set_pipeline(pipeline).unwrap();
    pass.set_bind_group(0, group).unwrap();
    pass.dispatch(2, 1, 1).unwrap();
    pass.end().unwrap();
    let commands = encoder.finish().unwrap();
    let mut cache = context.create_resources(Rc::clone(&table));
    let receipt = context
        .create_queue()
        .submit_tracked(&mut cache, &commands)
        .unwrap();
    drop(commands);
    assert_eq!(receipt.wait(None), Ok(CompletionStatus::Complete));
    assert_eq!(
        cache.read_buffer(snapshot.id(), 0, 32).unwrap(),
        bytemuck::cast_slice(&(5_u32..13).collect::<Vec<_>>())
    );
    assert_eq!(
        cache.read_buffer(output.id(), 0, 32).unwrap(),
        bytemuck::cast_slice(&(91_u32..99).collect::<Vec<_>>())
    );
}

const FULLSCREEN_VERTEX: &str = r#"
    @vertex fn vertex(@builtin(vertex_index) id: u32) -> @builtin(position) vec4<f32> {
        var positions = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
        return vec4(positions[id], 0.0, 1.0);
    }
"#;

#[test]
fn compute_storage_texture_then_programmable_draw_samples_the_result() {
    let _guard = HEADLESS_WGPU_TEST_LOCK.lock().expect("lock WGPU tests");
    let Some(device) = headless_device() else {
        return;
    };
    let context = device.create_context();
    let table = Rc::new(ResourceTable::new());
    let texture = table
        .define_texture(
            TextureDesc::new(
                TextureFormat::Rgba8Unorm,
                Extent2D::new(4, 4).unwrap(),
                TextureUsage::STORAGE | TextureUsage::SAMPLED,
            )
            .unwrap(),
        )
        .unwrap();
    let target = table
        .define_texture(
            TextureDesc::new(
                TextureFormat::Rgba8Unorm,
                Extent2D::new(4, 4).unwrap(),
                TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            )
            .unwrap(),
        )
        .unwrap();
    let compute_shader = shader(
        &table,
        r#"
        @group(0) @binding(0) var destination: texture_storage_2d<rgba8unorm, write>;
        @compute @workgroup_size(1) fn compute(@builtin(global_invocation_id) id: vec3<u32>) {
            textureStore(destination, vec2<i32>(id.xy), vec4(1.0, 0.0, 0.0, 1.0));
        }
    "#,
    );
    let compute_layout = BindGroupLayoutDesc::new(vec![BindGroupLayoutEntry::new(
        0,
        ShaderStages::COMPUTE,
        BindingType::StorageTexture {
            format: TextureFormat::Rgba8Unorm,
            access: StorageTextureAccess::WriteOnly,
        },
    )])
    .unwrap();
    let compute_group = table
        .define_bind_group(
            BindGroupDesc::new(
                &table,
                compute_layout.clone(),
                vec![BindGroupEntry::new(
                    0,
                    BindingResource::Texture(texture.id()),
                )],
            )
            .unwrap(),
        )
        .unwrap();
    let compute_pipeline = table
        .define_compute_pipeline(
            ComputePipelineDesc::new(
                entry(compute_shader, ShaderStage::Compute, "compute"),
                PipelineLayoutDesc::new(vec![compute_layout]).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    let render_shader = shader(
        &table,
        &format!(
            "{FULLSCREEN_VERTEX}\n{}",
            r#"
        @group(0) @binding(0) var source: texture_2d<f32>;
        @group(0) @binding(1) var filtering: sampler;
        @fragment fn fragment() -> @location(0) vec4<f32> { return textureSample(source, filtering, vec2(0.5, 0.5)); }
    "#
        ),
    );
    let sampler = table
        .define_sampler(SamplerDesc::new(
            FilterMode::Nearest,
            FilterMode::Nearest,
            AddressMode::ClampToEdge,
            AddressMode::ClampToEdge,
        ))
        .unwrap();
    let render_layout = BindGroupLayoutDesc::new(vec![
        BindGroupLayoutEntry::new(0, ShaderStages::FRAGMENT, BindingType::SampledTexture),
        BindGroupLayoutEntry::new(1, ShaderStages::FRAGMENT, BindingType::Sampler),
    ])
    .unwrap();
    let render_group = table
        .define_bind_group(
            BindGroupDesc::new(
                &table,
                render_layout.clone(),
                vec![
                    BindGroupEntry::new(0, BindingResource::Texture(texture.id())),
                    BindGroupEntry::new(1, BindingResource::Sampler(sampler.id())),
                ],
            )
            .unwrap(),
        )
        .unwrap();
    let pipeline = table
        .define_programmable_render_pipeline(
            ProgrammableRenderPipelineDesc::new(
                entry(render_shader, ShaderStage::Vertex, "vertex"),
                entry(render_shader, ShaderStage::Fragment, "fragment"),
                PipelineLayoutDesc::new(vec![render_layout]).unwrap(),
                TextureFormat::Rgba8Unorm,
                None,
                PrimitiveTopology::TriangleList,
                BlendState::SOURCE_OVER_STRAIGHT_ALPHA,
                RasterState::new(CullMode::None, FrontFace::CounterClockwise),
            )
            .unwrap(),
        )
        .unwrap();
    let mut encoder = CommandEncoder::new(&table);
    let mut pass = encoder.begin_compute_pass().unwrap();
    pass.set_pipeline(compute_pipeline).unwrap();
    pass.set_bind_group(0, compute_group).unwrap();
    pass.dispatch(4, 4, 1).unwrap();
    pass.end().unwrap();
    encoder
        .resource_barrier(ResourceBarrier::Texture {
            texture,
            before: TextureAccess::StorageWrite,
            after: TextureAccess::Sampled,
        })
        .unwrap();
    let desc = RenderPassDesc::new(
        &table,
        target,
        PixelRect::new(0, 0, 4, 4).unwrap(),
        LoadOp::Clear(Color::rgba(0.0, 0.0, 1.0, 1.0).unwrap()),
        StoreOp::Store,
    )
    .unwrap();
    let mut pass = encoder.begin_render_pass(desc).unwrap();
    pass.set_programmable_pipeline(pipeline).unwrap();
    pass.set_bind_group(0, render_group).unwrap();
    pass.draw(3, 0).unwrap();
    pass.end().unwrap();
    let mut cache = context.create_resources(Rc::clone(&table));
    context
        .create_queue()
        .submit(&mut cache, &encoder.finish().unwrap())
        .unwrap();
    assert_eq!(
        cache.read_texture(target.id()).unwrap(),
        [255, 0, 0, 255].repeat(16)
    );
}

#[test]
fn shader_errors_and_pipeline_layout_mismatches_return_errors_without_uncaptured_failures() {
    let _guard = HEADLESS_WGPU_TEST_LOCK.lock().expect("lock WGPU tests");
    let Some(device) = headless_device() else {
        return;
    };
    let context = device.create_context();
    let table = Rc::new(ResourceTable::new());
    let mut cache = context.create_resources(Rc::clone(&table));
    for source in [
        "@compute @workgroup_size(1) fn main() { invalid wgsl; }",
        "@group(0) @binding(0) var<storage, read_write> output: array<u32>; @compute @workgroup_size(1) fn main() { output[0] = 1u; }",
    ] {
        let module = shader(&table, source);
        let pipeline = table
            .define_compute_pipeline(
                ComputePipelineDesc::new(
                    entry(module, ShaderStage::Compute, "main"),
                    PipelineLayoutDesc::new(vec![]).unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
        let mut encoder = CommandEncoder::new(&table);
        let mut pass = encoder.begin_compute_pass().unwrap();
        pass.set_pipeline(pipeline).unwrap();
        pass.dispatch(1, 1, 1).unwrap();
        pass.end().unwrap();
        device
            .raw_device()
            .push_error_scope(raw::ErrorFilter::Validation);
        let result = context
            .create_queue()
            .submit(&mut cache, &encoder.finish().unwrap());
        assert!(matches!(result, Err(Error::Validation(_))), "{result:?}");
        assert!(pollster::block_on(device.raw_device().pop_error_scope()).is_none());
    }
}

#[test]
fn shader_minimum_binding_size_is_validated_before_submission() {
    let _guard = HEADLESS_WGPU_TEST_LOCK.lock().expect("lock WGPU tests");
    let Some(device) = headless_device() else {
        return;
    };
    let context = device.create_context();
    let table = Rc::new(ResourceTable::new());
    let module = shader(
        &table,
        "@group(0) @binding(0) var<storage, read_write> output: vec4<u32>; @compute @workgroup_size(1) fn main() { output = vec4(1u); }",
    );
    let layout = BindGroupLayoutDesc::new(vec![BindGroupLayoutEntry::new(
        0,
        ShaderStages::COMPUTE,
        BindingType::StorageBuffer { read_only: false },
    )])
    .unwrap();
    let pipeline = table
        .define_compute_pipeline(
            ComputePipelineDesc::new(
                entry(module, ShaderStage::Compute, "main"),
                PipelineLayoutDesc::new(vec![layout.clone()]).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    let small = table
        .define_buffer(BufferDesc::new(4, BufferUsage::STORAGE).unwrap())
        .unwrap();
    let group = buffer_group(&table, &layout, &[(small, 4)]);
    let mut encoder = CommandEncoder::new(&table);
    let mut pass = encoder.begin_compute_pass().unwrap();
    pass.set_pipeline(pipeline).unwrap();
    pass.set_bind_group(0, group).unwrap();
    pass.dispatch(1, 1, 1).unwrap();
    pass.end().unwrap();
    let mut cache = context.create_resources(Rc::clone(&table));
    let result = context
        .create_queue()
        .submit(&mut cache, &encoder.finish().unwrap());
    assert!(matches!(result, Err(Error::Validation(_))), "{result:?}");
}

#[test]
fn spirv_compute_modules_are_validated_translated_and_executed() {
    let _guard = HEADLESS_WGPU_TEST_LOCK.lock().expect("lock WGPU tests");
    let Some(device) = headless_device() else {
        return;
    };
    let context = device.create_context();
    let table = Rc::new(ResourceTable::new());
    let source = "@group(1) @binding(0) var<storage, read_write> output: array<u32>; @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) { output[id.x] = 37u + id.x; }";
    let module = naga::front::wgsl::parse_str(source).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    let words = naga::back::spv::write_vec(
        &module,
        &info,
        &naga::back::spv::Options::default(),
        Some(&naga::back::spv::PipelineOptions {
            shader_stage: naga::ShaderStage::Compute,
            entry_point: "main".into(),
        }),
    )
    .unwrap();
    let module = table
        .define_shader_module(ShaderModuleDesc::spirv(words).unwrap())
        .unwrap();
    let layout = BindGroupLayoutDesc::new(vec![BindGroupLayoutEntry::new(
        0,
        ShaderStages::COMPUTE,
        BindingType::StorageBuffer { read_only: false },
    )])
    .unwrap();
    let pipeline = table
        .define_compute_pipeline(
            ComputePipelineDesc::new(
                entry(module, ShaderStage::Compute, "main"),
                PipelineLayoutDesc::new(vec![
                    BindGroupLayoutDesc::new(vec![]).unwrap(),
                    layout.clone(),
                ])
                .unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    let output = table
        .define_buffer(BufferDesc::new(16, BufferUsage::STORAGE | BufferUsage::COPY_SRC).unwrap())
        .unwrap();
    let group = buffer_group(&table, &layout, &[(output, 16)]);
    let mut encoder = CommandEncoder::new(&table);
    let mut pass = encoder.begin_compute_pass().unwrap();
    pass.set_pipeline(pipeline).unwrap();
    pass.set_bind_group(1, group).unwrap();
    pass.dispatch(4, 1, 1).unwrap();
    pass.end().unwrap();
    let mut cache = context.create_resources(Rc::clone(&table));
    context
        .create_queue()
        .submit(&mut cache, &encoder.finish().unwrap())
        .unwrap();
    assert_eq!(
        cache.read_buffer(output.id(), 0, 16).unwrap(),
        bytemuck::cast_slice(&[37_u32, 38, 39, 40])
    );
}

#[test]
fn indexed_draw_ignores_unused_groups_and_allows_depth_disabled_with_depth_attachment() {
    let _guard = HEADLESS_WGPU_TEST_LOCK.lock().expect("lock WGPU tests");
    let Some(device) = headless_device() else {
        return;
    };
    let context = device.create_context();
    let table = Rc::new(ResourceTable::new());
    let module = shader(
        &table,
        r#"
        @vertex fn vertex(@location(0) position: vec2<f32>) -> @builtin(position) vec4<f32> { return vec4(position, 0.0, 1.0); }
        @fragment fn fragment() -> @location(0) vec4<f32> { return vec4(0.0, 1.0, 0.0, 1.0); }
    "#,
    );
    let extent = Extent2D::new(4, 4).unwrap();
    let target = table
        .define_texture(
            TextureDesc::new(
                TextureFormat::Rgba8Unorm,
                extent,
                TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC | TextureUsage::SAMPLED,
            )
            .unwrap(),
        )
        .unwrap();
    let depth = table
        .define_texture(
            TextureDesc::new(
                TextureFormat::Depth32Float,
                extent,
                TextureUsage::RENDER_ATTACHMENT,
            )
            .unwrap(),
        )
        .unwrap();
    let unused_layout = BindGroupLayoutDesc::new(vec![BindGroupLayoutEntry::new(
        0,
        ShaderStages::FRAGMENT,
        BindingType::SampledTexture,
    )])
    .unwrap();
    let unused_group = table
        .define_bind_group(
            BindGroupDesc::new(
                &table,
                unused_layout,
                vec![BindGroupEntry::new(
                    0,
                    BindingResource::Texture(target.id()),
                )],
            )
            .unwrap(),
        )
        .unwrap();
    let vertices = [[-1_f32, -1.0], [3.0, -1.0], [-1.0, 3.0]];
    let indices = [0_u16, 1, 2, 0];
    let vertex = table
        .define_buffer(BufferDesc::new(24, BufferUsage::VERTEX | BufferUsage::COPY_DST).unwrap())
        .unwrap();
    let index = table
        .define_buffer(BufferDesc::new(8, BufferUsage::INDEX | BufferUsage::COPY_DST).unwrap())
        .unwrap();
    let pipeline = table
        .define_programmable_render_pipeline(
            ProgrammableRenderPipelineDesc::new(
                entry(module, ShaderStage::Vertex, "vertex"),
                entry(module, ShaderStage::Fragment, "fragment"),
                PipelineLayoutDesc::new(vec![BindGroupLayoutDesc::new(vec![]).unwrap()]).unwrap(),
                TextureFormat::Rgba8Unorm,
                Some(
                    VertexBufferLayout::new(
                        8,
                        vec![VertexAttribute::new(0, VertexFormat::Float32x2, 0)],
                    )
                    .unwrap(),
                ),
                PrimitiveTopology::TriangleList,
                BlendState::SOURCE_OVER_STRAIGHT_ALPHA,
                RasterState::new(CullMode::None, FrontFace::CounterClockwise),
            )
            .unwrap(),
        )
        .unwrap();
    let mut cache = context.create_resources(Rc::clone(&table));
    cache.validate_shader_module(module.id()).unwrap();
    cache
        .validate_programmable_render_pipeline(pipeline.id())
        .unwrap();
    let mut encoder = CommandEncoder::new(&table);
    encoder
        .write_buffer(vertex, 0, bytemuck::cast_slice(&vertices))
        .unwrap();
    encoder
        .write_buffer(index, 0, bytemuck::cast_slice(&indices))
        .unwrap();
    let desc = RenderPassDesc::new(
        &table,
        target,
        PixelRect::new(0, 0, 4, 4).unwrap(),
        LoadOp::Clear(Color::rgba(0.0, 0.0, 0.0, 1.0).unwrap()),
        StoreOp::Store,
    )
    .unwrap()
    .with_depth_attachment(&table, depth, DepthLoadOp::Clear(0.0), StoreOp::Store)
    .unwrap();
    let mut pass = encoder.begin_render_pass(desc).unwrap();
    pass.set_programmable_pipeline(pipeline).unwrap();
    pass.set_bind_group(3, unused_group).unwrap();
    pass.set_vertex_buffer(vertex, 0).unwrap();
    pass.set_index_buffer(index, 0, IndexFormat::Uint16)
        .unwrap();
    pass.draw_indexed(3, 0, 0).unwrap();
    pass.end().unwrap();
    context
        .create_queue()
        .submit(&mut cache, &encoder.finish().unwrap())
        .unwrap();
    assert_eq!(
        cache.read_texture(target.id()).unwrap(),
        [0, 255, 0, 255].repeat(16)
    );
}
