//! CPU-only recording/walking workload; no backend, GPU, or presentation.
//! cargo run -p sgfx-core --release --features backend-abi --example ir_recording_bench -- [--abi]
use sgfx_core::ir::*;
use std::{hint::black_box, time::Instant};

fn main() -> Result<()> {
    let table = ResourceTable::new();
    #[cfg(feature = "backend-abi")]
    if std::env::args().any(|arg| arg == "--abi") {
        table.enable_abi_commands();
    }
    let target = table.define_texture(TextureDesc::new(
        TextureFormat::Bgra8Unorm,
        Extent2D::new(64, 64)?,
        TextureUsage::RENDER_ATTACHMENT,
    )?)?;
    let texture = table.define_texture(TextureDesc::new(
        TextureFormat::Bgra8Unorm,
        Extent2D::new(1, 1)?,
        TextureUsage::SAMPLED,
    )?)?;
    let buffer = table.define_buffer(BufferDesc::new(48, BufferUsage::VERTEX)?)?;
    let sampler = table.define_sampler(SamplerDesc::new(
        FilterMode::Linear,
        FilterMode::Linear,
        AddressMode::ClampToEdge,
        AddressMode::ClampToEdge,
    ))?;
    let pipeline = table.define_render_pipeline(RenderPipelineDesc::new(
        TextureFormat::Bgra8Unorm,
        PrimitiveTopology::TriangleList,
        VertexBufferLayout::new(
            16,
            vec![
                VertexAttribute::new(0, VertexFormat::Float32x2, 0),
                VertexAttribute::new(1, VertexFormat::Float32x2, 8),
            ],
        )?,
        FragmentProgram::Texture(TextureSampleMode::Rgba),
        BlendState::REPLACE,
        RasterState::new(CullMode::None, FrontFace::CounterClockwise),
    )?)?;
    let area = PixelRect::new(0, 0, 64, 64)?;
    let uniforms = DrawUniforms::new(Transform::identity(), Color::rgba(1., 1., 1., 1.)?);
    let record = || -> Result<CommandBuffer<'_, '_>> {
        let mut encoder = CommandEncoder::new(&table);
        let mut pass = encoder.begin_render_pass(RenderPassDesc::new(
            &table,
            target,
            area,
            LoadOp::Load,
            StoreOp::Store,
        )?)?;
        for _ in 0..1200 {
            pass.set_pipeline(pipeline)?;
            pass.set_vertex_buffer(buffer, 0)?;
            pass.set_texture(texture)?;
            pass.set_sampler(sampler)?;
            pass.set_uniforms(uniforms)?;
            pass.set_scissor(Some(area))?;
            pass.draw(3, 0)?;
        }
        pass.end()?;
        encoder.finish()
    };
    for _ in 0..100 {
        black_box(record()?);
    }
    let mut record_ns = 0;
    let mut walk_ns = 0;
    let mut count = 0;
    for _ in 0..4000 {
        let start = Instant::now();
        let commands = black_box(record()?);
        record_ns += start.elapsed().as_nanos();
        let start = Instant::now();
        count = 0;
        #[cfg(feature = "backend-abi")]
        {
            let mut reader = commands.command_reader();
            while let Some(command) = reader.next_command()? {
                black_box(command);
                count += 1;
            }
        }
        #[cfg(not(feature = "backend-abi"))]
        for command in commands.commands() {
            black_box(command);
            count += 1;
        }
        walk_ns += start.elapsed().as_nanos();
        black_box(commands);
    }
    println!(
        "draws=1200 commands={count} record_us={:.3} walk_us={:.3}",
        record_ns as f64 / 4_000_000.,
        walk_ns as f64 / 4_000_000.
    );
    Ok(())
}
