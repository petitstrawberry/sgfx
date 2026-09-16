//! The SGFX IR side of the Linux Mesa/VirGL vs Scarlet/VirGL workload.

use std::{
    error::Error,
    fmt, io,
    num::ParseIntError,
    rc::Rc,
    time::{Duration, Instant},
};

use sgfx_core::{
    backend::{Completion, CompletionStatus},
    ir::{
        BlendState, BufferDesc, BufferId, BufferUsage, Color, CommandBuffer, CommandEncoder,
        CullMode, DrawUniforms, Extent2D, FragmentProgram, FrontFace, LoadOp, PixelRect,
        PrimitiveTopology, RasterState, RenderPassDesc, RenderPipelineDesc, RenderPipelineId,
        ResourceTable, StoreOp, TextureDesc, TextureFormat, TextureId, TextureUsage, Transform,
        VertexAttribute, VertexBufferLayout, VertexFormat,
    },
};

type Result<T> = std::result::Result<T, BenchError>;

#[derive(Debug)]
struct BenchError(String);

impl fmt::Display for BenchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for BenchError {}

impl From<io::Error> for BenchError {
    fn from(error: io::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<ParseIntError> for BenchError {
    fn from(error: ParseIntError) -> Self {
        Self(error.to_string())
    }
}

impl From<sgfx_core::ir::Error> for BenchError {
    fn from(error: sgfx_core::ir::Error) -> Self {
        Self(format!("{error:?}"))
    }
}

fn backend_error(error: impl std::fmt::Debug) -> io::Error {
    io::Error::other(format!("{error:?}"))
}

#[derive(Clone, Copy, Debug)]
struct Config {
    width: u32,
    height: u32,
    draws: u32,
    frames: u32,
    warmup: u32,
    uniform_every: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 800,
            draws: 1200,
            frames: 120,
            warmup: 20,
            uniform_every: 20,
        }
    }
}

impl Config {
    fn parse() -> Result<Self> {
        let mut config = Self::default();
        let mut args = std::env::args().skip(1);
        while let Some(name) = args.next() {
            let value = args
                .next()
                .ok_or_else(|| io::Error::other(format!("missing value for {name}")))?;
            let parsed: u32 = value.parse()?;
            match name.as_str() {
                "--width" => config.width = parsed,
                "--height" => config.height = parsed,
                "--draws" => config.draws = parsed,
                "--frames" => config.frames = parsed,
                "--warmup" => config.warmup = parsed,
                "--uniform-every" => config.uniform_every = parsed,
                _ => return Err(io::Error::other(format!("unknown option {name}")).into()),
            }
        }
        if config.width == 0
            || config.height == 0
            || config.width > 8192
            || config.height > 8192
            || config.draws > 20_000
            || config.frames == 0
            || config.frames > 10_000
            || config.warmup > 10_000
        {
            return Err(io::Error::other("invalid benchmark dimensions or counts").into());
        }
        Ok(config)
    }
}

struct Scene {
    table: Rc<ResourceTable>,
    target: TextureId,
    vertices: BufferId,
    pipeline: RenderPipelineId,
    config: Config,
}

impl Scene {
    fn new(config: Config) -> Result<Self> {
        let table = Rc::new(ResourceTable::new());
        let target = table
            .define_texture(TextureDesc::new(
                TextureFormat::Bgra8Unorm,
                Extent2D::new(config.width, config.height)?,
                TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            )?)?
            .id();
        let vertices = table
            .define_buffer(BufferDesc::new(
                24,
                BufferUsage::VERTEX | BufferUsage::COPY_DST,
            )?)?
            .id();
        let pipeline = table
            .define_render_pipeline(RenderPipelineDesc::new(
                TextureFormat::Bgra8Unorm,
                PrimitiveTopology::TriangleList,
                VertexBufferLayout::new(
                    8,
                    vec![VertexAttribute::new(0, VertexFormat::Float32x2, 0)],
                )?,
                FragmentProgram::Solid,
                BlendState::REPLACE,
                RasterState::new(CullMode::None, FrontFace::CounterClockwise),
            )?)?
            .id();
        Ok(Self {
            table,
            target,
            vertices,
            pipeline,
            config,
        })
    }

    fn upload_vertices<'a>(&'a self, bytes: &'a [u8]) -> Result<CommandBuffer<'a, 'a>> {
        let mut encoder = CommandEncoder::new(&self.table);
        encoder.write_buffer(self.table.buffer_ref(self.vertices)?, 0, bytes)?;
        Ok(encoder.finish()?)
    }

    fn record(&self) -> Result<CommandBuffer<'_, '_>> {
        let mut encoder = CommandEncoder::new(&self.table);
        let mut pass = encoder.begin_render_pass(RenderPassDesc::new(
            &self.table,
            self.table.texture_ref(self.target)?,
            PixelRect::new(0, 0, self.config.width, self.config.height)?,
            LoadOp::Clear(Color::rgba(0.05, 0.08, 0.12, 1.0)?),
            StoreOp::Store,
        )?)?;
        if self.config.draws > 0 {
            pass.set_pipeline(self.table.render_pipeline_ref(self.pipeline)?)?;
            pass.set_vertex_buffer(self.table.buffer_ref(self.vertices)?, 0)?;
            pass.set_uniforms(uniforms(1.0)?)?;
            for draw in 0..self.config.draws {
                if self.config.uniform_every != 0
                    && draw != 0
                    && draw % self.config.uniform_every == 0
                {
                    let brightness = if (draw / self.config.uniform_every).is_multiple_of(2) {
                        1.0
                    } else {
                        0.6
                    };
                    pass.set_uniforms(uniforms(brightness)?)?;
                }
                pass.draw(3, 0)?;
            }
        }
        pass.end()?;
        Ok(encoder.finish()?)
    }
}

fn uniforms(brightness: f32) -> Result<DrawUniforms> {
    Ok(DrawUniforms::new(
        Transform::identity(),
        Color::rgba(brightness, brightness, brightness, 1.0)?,
    ))
}

#[derive(Clone, Copy)]
struct Timings {
    admission: Duration,
    completion: Duration,
}

trait Runner {
    fn name(&self) -> &str;
    fn execute(&mut self, commands: &CommandBuffer<'_, '_>) -> Result<Timings>;
    fn read_texture(&mut self, target: TextureId) -> Result<Vec<u8>>;
}

struct ScarletRunner {
    context: sgfx_backend_scarlet_virgl::Context,
    resources: sgfx_backend_scarlet_virgl::IrResources,
    queue: sgfx_backend_scarlet_virgl::Queue,
}

impl ScarletRunner {
    fn new(table: Rc<ResourceTable>) -> Result<Self> {
        let device =
            sgfx_backend_scarlet_virgl::Device::open("/dev/gpu0").map_err(backend_error)?;
        let context = device.create_context().map_err(backend_error)?;
        let resources = context.create_ir_resources(table).map_err(backend_error)?;
        let queue = context.create_queue().map_err(backend_error)?;
        Ok(Self {
            context,
            resources,
            queue,
        })
    }
}

impl Runner for ScarletRunner {
    fn name(&self) -> &str {
        "scarlet-sgfx-virgl"
    }

    fn execute(&mut self, commands: &CommandBuffer<'_, '_>) -> Result<Timings> {
        let admitted = Instant::now();
        let receipt = self
            .queue
            .submit_ir_async(&self.context, &mut self.resources, commands)
            .map_err(backend_error)?;
        let admission = admitted.elapsed();
        let completed = Instant::now();
        if receipt.wait(None).map_err(backend_error)? != CompletionStatus::Complete {
            return Err(io::Error::other("Scarlet submission did not complete").into());
        }
        Ok(Timings {
            admission,
            completion: completed.elapsed(),
        })
    }

    fn read_texture(&mut self, target: TextureId) -> Result<Vec<u8>> {
        self.context
            .read_texture(&mut self.resources, target)
            .map_err(backend_error)
            .map_err(Into::into)
    }
}

fn percentile(samples: &[Duration], numerator: usize, denominator: usize) -> f64 {
    let mut ordered: Vec<_> = samples.iter().map(Duration::as_secs_f64).collect();
    ordered.sort_by(f64::total_cmp);
    let index = (ordered.len() - 1) * numerator / denominator;
    ordered[index] * 1000.0
}

fn main() -> Result<()> {
    let config = Config::parse()?;
    eprintln!("stage: scene");
    let scene = Scene::new(config)?;
    eprintln!("stage: device");
    let mut runner = ScarletRunner::new(Rc::clone(&scene.table))?;
    eprintln!("stage: vertex upload");
    let vertex_bytes: Vec<u8> = [-1.0f32, -1.0, 1.0, -1.0, 0.0, 1.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    runner.execute(&scene.upload_vertices(&vertex_bytes)?)?;
    eprintln!("stage: warmup");
    let mut records = Vec::with_capacity(config.frames as usize);
    let mut admissions = Vec::with_capacity(config.frames as usize);
    let mut completions = Vec::with_capacity(config.frames as usize);
    let mut totals = Vec::with_capacity(config.frames as usize);
    println!(
        "RUN backend={} width={} height={} draws={} uniform_every={} warmup={} frames={}",
        runner.name(),
        config.width,
        config.height,
        config.draws,
        config.uniform_every,
        config.warmup,
        config.frames,
    );
    for frame in 0..config.warmup + config.frames {
        let started = Instant::now();
        let commands = scene.record()?;
        let record = started.elapsed();
        let timings = runner.execute(&commands)?;
        if frame >= config.warmup {
            records.push(record);
            admissions.push(timings.admission);
            completions.push(timings.completion);
            totals.push(record + timings.admission + timings.completion);
        }
    }
    let pixels = runner.read_texture(scene.target)?;
    let expected_len = config.width as usize * config.height as usize * 4;
    if pixels.len() != expected_len {
        return Err(
            io::Error::other(format!("readback size {} != {expected_len}", pixels.len())).into(),
        );
    }
    let edge_at = (config.height as usize / 2) * config.width as usize * 4;
    let edge = &pixels[edge_at..edge_at + 4];
    let center_at =
        ((config.height as usize / 2) * config.width as usize + config.width as usize / 2) * 4;
    let center = &pixels[center_at..center_at + 4];
    if config.draws > 0 && center == edge {
        return Err(io::Error::other("draw did not change the center pixel").into());
    }
    let hash = pixels.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    });
    let median = percentile(&totals, 1, 2);
    let total_seconds: f64 = totals.iter().map(Duration::as_secs_f64).sum();
    println!(
        "RESULT backend={} draws={} frames={} fps={:.2} median_ms={:.3} p95_ms={:.3} record_ms={:.3} admission_ms={:.3} completion_ms={:.3} center={center:?} edge={edge:?} fnv64={hash:016x}",
        runner.name(),
        config.draws,
        config.frames,
        config.frames as f64 / total_seconds,
        median,
        percentile(&totals, 95, 100),
        percentile(&records, 1, 2),
        percentile(&admissions, 1, 2),
        percentile(&completions, 1, 2),
    );
    Ok(())
}
