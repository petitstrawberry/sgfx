//! GPU readback check for G-buffer outputs and depth-tested deferred lighting.

use sgfx::driver::Instance;
use sgfx::{
    backend::{Completion, CompletionStatus},
    ir::*,
};
use std::{rc::Rc, time::Duration};

fn shader<'a>(table: &'a ResourceTable, source: &str) -> ShaderModuleRef<'a> {
    table
        .define_shader_module(ShaderModuleDesc::wgsl(source.into()).unwrap())
        .unwrap()
}
fn entry(module: ShaderModuleRef<'_>, stage: ShaderStage, name: &str) -> ShaderEntryPoint {
    ShaderEntryPoint::new(module, stage, name.into()).unwrap()
}

fn run() -> std::result::Result<(), String> {
    let instance = Instance::new().map_err(|e| format!("instance: {e:?}"))?;
    let adapter = instance.adapters().first().ok_or("no SGFX adapter")?;
    println!("backend: {}", adapter.info().backend());
    if adapter.capabilities().limits().max_color_attachments < 2 {
        return Err("adapter reports fewer than two color attachments".into());
    }
    let device = adapter
        .create_device()
        .map_err(|e| format!("device: {e:?}"))?;
    let table = Rc::new(ResourceTable::new());
    let extent = Extent2D::new(8, 8).unwrap();
    let make_texture = |format| {
        table
            .define_texture(
                TextureDesc::new(
                    format,
                    extent,
                    TextureUsage::RENDER_ATTACHMENT
                        | TextureUsage::SAMPLED
                        | TextureUsage::COPY_SRC,
                )
                .unwrap(),
            )
            .unwrap()
    };
    let color = make_texture(TextureFormat::Rgba8Unorm);
    let normal = make_texture(TextureFormat::Bgra8Unorm);
    let output = make_texture(TextureFormat::Rgba8Unorm);
    let shadow_output = make_texture(TextureFormat::Rgba8Unorm);
    let depth = table
        .define_texture(
            TextureDesc::new(
                TextureFormat::Depth32Float,
                extent,
                TextureUsage::RENDER_ATTACHMENT | TextureUsage::SAMPLED,
            )
            .unwrap(),
        )
        .unwrap();
    let module = shader(
        &table,
        r#"
        @vertex fn vertex(@builtin(vertex_index) i:u32) -> @builtin(position) vec4<f32> {
            var p = array<vec2<f32>,3>(vec2(-1.0,-1.0),vec2(3.0,-1.0),vec2(-1.0,3.0));
            return vec4(p[i],0.25,1.0);
        }
        struct GBuffer { @location(0) color:vec4<f32>, @location(1) normal:vec4<f32> }
        @fragment fn geometry() -> GBuffer {
            return GBuffer(vec4(0.25,0.5,0.75,1.0),vec4(0.8,0.5,0.9,0.2));
        }
        @group(0) @binding(0) var color:texture_2d<f32>;
        @group(0) @binding(1) var normal:texture_2d<f32>;
        @group(0) @binding(2) var depth:texture_depth_2d;
        @group(0) @binding(3) var comparison:sampler_comparison;
        @fragment fn lighting(@builtin(position) p:vec4<f32>) -> @location(0) vec4<f32> {
            let xy = vec2<i32>(p.xy);
            let albedo = textureLoad(color,xy,0).rgb;
            let encoded = textureLoad(normal,xy,0).rg * 2.0 - vec2<f32>(1.0);
            let surface_normal = normalize(vec3<f32>(encoded,
                sqrt(max(0.0, 1.0 - dot(encoded, encoded)))));
            let light = vec3<f32>(0.0, 0.0, 1.0);
            let diffuse = max(dot(surface_normal, light), 0.0);
            let specular = pow(max(dot(surface_normal, light), 0.0), 4.0);
            let attenuation = 1.0 / (1.0 + textureLoad(depth,xy,0));
            let shaded = (albedo * (0.2 + 0.7 * diffuse) + vec3<f32>(0.1 * specular)) * attenuation;
            var factor = 0.0;
            for (var i = 0; i < 8; i = i + 1) {
                if (i == 5) { break; }
                if (i == 2) { continue; }
                factor += 0.25;
            }
            return vec4(shaded * factor, 1.0);
        }
        @fragment fn shadow() -> @location(0) vec4<f32> {
            let lit = textureSampleCompareLevel(depth, comparison, vec2<f32>(0.5), 0.2);
            let dark = textureSampleCompareLevel(depth, comparison, vec2<f32>(0.5), 0.75);
            return vec4(lit, dark, 0.0, 1.0);
        }
    "#,
    );
    let make_pipeline = |fragment, layout| {
        ProgrammableRenderPipelineDesc::new(
            entry(module, ShaderStage::Vertex, "vertex"),
            entry(module, ShaderStage::Fragment, fragment),
            layout,
            TextureFormat::Rgba8Unorm,
            None,
            PrimitiveTopology::TriangleList,
            BlendState::REPLACE,
            RasterState::new(CullMode::None, FrontFace::CounterClockwise),
        )
        .unwrap()
    };
    let geometry = table
        .define_programmable_render_pipeline(
            make_pipeline("geometry", PipelineLayoutDesc::new(vec![]).unwrap())
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
                .unwrap()
                .with_depth_state(DepthState::new(
                    TextureFormat::Depth32Float,
                    CompareFunction::Always,
                    true,
                ))
                .unwrap(),
        )
        .unwrap();
    let layout = BindGroupLayoutDesc::new(
        (0..3)
            .map(|binding| {
                BindGroupLayoutEntry::new(
                    binding,
                    ShaderStages::FRAGMENT,
                    BindingType::SampledTextureView {
                        dimension: TextureViewDimension::D2,
                        depth: binding == 2,
                    },
                )
            })
            .collect(),
    )
    .unwrap();
    let lighting = table
        .define_programmable_render_pipeline(
            make_pipeline(
                "lighting",
                PipelineLayoutDesc::new(vec![layout.clone()]).unwrap(),
            )
            .with_depth_state(DepthState::new(
                TextureFormat::Depth32Float,
                CompareFunction::LessEqual,
                false,
            ))
            .unwrap(),
        )
        .unwrap();
    let group = table
        .define_bind_group(
            BindGroupDesc::new(
                &table,
                layout,
                [color, normal, depth]
                    .into_iter()
                    .enumerate()
                    .map(|(slot, t)| {
                        let desc = table.texture(t).unwrap();
                        BindGroupEntry::new(
                            slot as u32,
                            BindingResource::TextureView {
                                texture: t.id(),
                                view: TextureViewDesc::new(
                                    desc,
                                    desc.format(),
                                    TextureViewDimension::D2,
                                    0,
                                    1,
                                    0,
                                    1,
                                )
                                .unwrap(),
                            },
                        )
                    })
                    .collect(),
            )
            .unwrap(),
        )
        .unwrap();
    let shadow_sampler = table
        .define_sampler(
            SamplerDesc::new(
                FilterMode::Nearest,
                FilterMode::Nearest,
                AddressMode::ClampToEdge,
                AddressMode::ClampToEdge,
            )
            .with_compare(Some(CompareFunction::LessEqual)),
        )
        .unwrap();
    let shadow_layout = BindGroupLayoutDesc::new(vec![
        BindGroupLayoutEntry::new(
            2,
            ShaderStages::FRAGMENT,
            BindingType::SampledTextureView {
                dimension: TextureViewDimension::D2,
                depth: true,
            },
        ),
        BindGroupLayoutEntry::new(3, ShaderStages::FRAGMENT, BindingType::ComparisonSampler),
    ])
    .unwrap();
    let shadow_pipeline = table
        .define_programmable_render_pipeline(make_pipeline(
            "shadow",
            PipelineLayoutDesc::new(vec![shadow_layout.clone()]).unwrap(),
        ))
        .unwrap();
    let depth_desc = table.texture(depth).unwrap();
    let shadow_group = table
        .define_bind_group(
            BindGroupDesc::new(
                &table,
                shadow_layout,
                vec![
                    BindGroupEntry::new(
                        2,
                        BindingResource::TextureView {
                            texture: depth.id(),
                            view: TextureViewDesc::new(
                                depth_desc,
                                TextureFormat::Depth32Float,
                                TextureViewDimension::D2,
                                0,
                                1,
                                0,
                                1,
                            )
                            .unwrap(),
                        },
                    ),
                    BindGroupEntry::new(3, BindingResource::Sampler(shadow_sampler.id())),
                ],
            )
            .unwrap(),
        )
        .unwrap();
    let area = PixelRect::new(0, 0, 8, 8).unwrap();
    let mut encoder = CommandEncoder::new(&table);
    let mut pass = encoder
        .begin_render_pass(
            RenderPassDesc::new(&table, color, area, LoadOp::DontCare, StoreOp::Store)
                .unwrap()
                .with_color_attachment(
                    &table,
                    normal,
                    LoadOp::Clear(Color::rgba(0.0, 0.0, 0.2, 1.0).unwrap()),
                    StoreOp::Store,
                )
                .unwrap()
                .with_depth_attachment(&table, depth, DepthLoadOp::Clear(1.0), StoreOp::Store)
                .unwrap(),
        )
        .unwrap();
    pass.set_programmable_pipeline(geometry).unwrap();
    pass.draw(3, 0).unwrap();
    pass.end().unwrap();
    let mut pass = encoder
        .begin_render_pass(
            RenderPassDesc::new(
                &table,
                shadow_output,
                area,
                LoadOp::DontCare,
                StoreOp::Store,
            )
            .unwrap(),
        )
        .unwrap();
    pass.set_programmable_pipeline(shadow_pipeline).unwrap();
    pass.set_bind_group(0, shadow_group).unwrap();
    pass.draw(3, 0).unwrap();
    pass.end().unwrap();
    let mut pass = encoder
        .begin_render_pass(
            RenderPassDesc::new(&table, output, area, LoadOp::DontCare, StoreOp::Store)
                .unwrap()
                .with_depth_attachment(&table, depth, DepthLoadOp::Load, StoreOp::Store)
                .unwrap()
                .with_read_only_depth()
                .unwrap(),
        )
        .unwrap();
    pass.set_programmable_pipeline(lighting).unwrap();
    pass.set_bind_group(0, group).unwrap();
    pass.draw(3, 0).unwrap();
    pass.end().unwrap();
    let commands = encoder.finish().unwrap();
    let mut resources = device
        .create_resources(Rc::clone(&table))
        .map_err(|e| format!("resources: {e:?}"))?;
    let queue = device.create_queue().map_err(|e| format!("queue: {e:?}"))?;
    let submission = queue
        .submit(&mut resources, &commands)
        .map_err(|e| format!("submit: {e:?}"))?;
    let status = submission
        .wait(Some(Duration::from_secs(10)))
        .map_err(|e| format!("wait: {e:?}"))?;
    if status != CompletionStatus::Complete {
        return Err(format!("not complete: {status:?}"));
    }
    let pixels = queue
        .read_texture(&mut resources, output.id())
        .map_err(|e| format!("read: {e:?}"))?;
    println!("lighting first pixel: {:?}", &pixels[..4]);
    if !pixels.chunks_exact(4).all(|pixel| {
        pixel
            .iter()
            .zip([47, 86, 125, 255])
            .all(|(&actual, expected)| i32::from(actual).abs_diff(expected) <= 2)
    }) {
        return Err(format!("lighting mismatch: {:?}", &pixels[..16]));
    }
    let normals = queue
        .read_texture(&mut resources, normal.id())
        .map_err(|e| format!("normal read: {e:?}"))?;
    println!("normal first pixel (BGRA): {:?}", &normals[..4]);
    if !normals
        .chunks_exact(4)
        .all(|pixel| pixel == [51, 128, 204, 255])
    {
        return Err(format!("normal write-mask mismatch: {:?}", &normals[..16]));
    }
    let shadow_pixels = queue
        .read_texture(&mut resources, shadow_output.id())
        .map_err(|e| format!("shadow read: {e:?}"))?;
    println!("shadow first pixel: {:?}", &shadow_pixels[..4]);
    if !shadow_pixels
        .chunks_exact(4)
        .all(|pixel| pixel == [255, 0, 0, 255])
    {
        return Err(format!("shadow mismatch: {:?}", &shadow_pixels[..16]));
    }
    println!("SGFX deferred lighting pass OK");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("SGFX deferred lighting pass FAILED: {error}");
        std::process::exit(1);
    }
}
