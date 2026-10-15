//! Build this same scene with backend-dynamic and backend-scarlet-virgl to
//! compare the dynamic boundary against the static implementation in one guest.
//! `--copy-only` runs the async vertex-copy pixel contract directly, skipping
//! the parent coordinator, low-level fixture, and timing benchmarks.
//! `--import-overlap-only` checks same-context resize/session lifetime with a retained old-size image.
//! `--program-churn-only` checks 512 native shader/pipeline generations and retained receipts.
//! `--narrow-only` checks native R8/RG8 upload, copies, sampling, mips and RT blending.
// /init has no inherited stdio handles. Use the native bootstrap console for
// diagnostics, including panics, just like Scarlet's loader smoke fixtures.
#[cfg(target_os = "scarlet")]
fn console(args: std::fmt::Arguments<'_>) {
    struct Console;
    impl std::fmt::Write for Console {
        fn write_str(&mut self, text: &str) -> std::fmt::Result {
            for byte in text.bytes() {
                unsafe {
                    #[cfg(target_arch = "aarch64")]
                    core::arch::asm!("svc #0", in("x8") 16usize, inout("x0") byte as usize => _, options(nostack));
                    #[cfg(target_arch = "riscv64")]
                    core::arch::asm!("ecall", in("a7") 16usize, inout("a0") byte as usize => _, options(nostack));
                }
            }
            Ok(())
        }
    }
    use std::fmt::Write;
    Console.write_fmt(args).unwrap();
}
#[cfg(target_os = "scarlet")]
macro_rules! println {
    ($($arg:tt)*) => { crate::console(format_args!("{}\n", format_args!($($arg)*))) };
}
#[cfg(any(target_os = "scarlet", test))]
mod vertex_copy_scene {
    use sgfx::ir::*;

    pub const SIDE: u32 = 64;
    pub const BANDS: u32 = 4;
    pub const PREFIX: u64 = 96;
    pub const STRIDE: u64 = 12;
    pub const VERTICES: u32 = BANDS * 6;
    #[cfg(target_os = "scarlet")]
    pub const UPDATES: u32 = 10;

    pub fn define(table: &ResourceTable) -> (BufferId, RenderPipelineId) {
        let vertices = table
            .define_buffer(
                BufferDesc::new(
                    PREFIX + u64::from(VERTICES) * STRIDE,
                    BufferUsage::VERTEX | BufferUsage::COPY_DST,
                )
                .unwrap(),
            )
            .unwrap()
            .id();
        let pipeline = table
            .define_render_pipeline(
                RenderPipelineDesc::new(
                    TextureFormat::Bgra8Unorm,
                    PrimitiveTopology::TriangleList,
                    VertexBufferLayout::new(
                        STRIDE as u32,
                        vec![
                            VertexAttribute::new(0, VertexFormat::Float32x2, 0),
                            VertexAttribute::new(1, VertexFormat::Unorm8x4, 8),
                        ],
                    )
                    .unwrap(),
                    FragmentProgram::VertexColor,
                    BlendState::REPLACE,
                    RasterState::new(CullMode::None, FrontFace::CounterClockwise),
                )
                .unwrap(),
            )
            .unwrap()
            .id();
        (vertices, pipeline)
    }

    pub fn record_passes<'r, 'data>(
        encoder: &mut CommandEncoder<'r, 'data>,
        table: &'r ResourceTable,
        target: TextureId,
        vertices: BufferId,
        pipeline: RenderPipelineId,
    ) {
        for first_vertex in [0, VERTICES / 2] {
            let desc = RenderPassDesc::new(
                table,
                table.texture_ref(target).unwrap(),
                PixelRect::new(0, 0, SIDE, SIDE).unwrap(),
                if first_vertex == 0 {
                    LoadOp::Clear(Color::rgba(0.0, 0.0, 0.0, 0.0).unwrap())
                } else {
                    LoadOp::Load
                },
                StoreOp::Store,
            )
            .unwrap();
            let mut pass = encoder.begin_render_pass(desc).unwrap();
            pass.set_pipeline(table.render_pipeline_ref(pipeline).unwrap())
                .unwrap();
            pass.set_vertex_buffer(table.buffer_ref(vertices).unwrap(), PREFIX)
                .unwrap();
            pass.set_uniforms(DrawUniforms::new(
                Transform::identity(),
                Color::rgba(1.0, 1.0, 1.0, 1.0).unwrap(),
            ))
            .unwrap();
            pass.draw(VERTICES / 2, first_vertex).unwrap();
            pass.end().unwrap();
        }
    }
}

// Shared by the real guest contract and host-side IR validation.
#[cfg(any(target_os = "scarlet", test))]
mod narrow_texture_scene {
    use sgfx::ir::*;
    pub const WIDTH: u32 = 13;
    pub const HEIGHT: u32 = 7;
    pub const OUT_WIDTH: u32 = 15;
    pub const OUT_HEIGHT: u32 = 9;

    pub struct Scene {
        pub source: TextureId,
        pub copied: TextureId,
        pub bgra: TextureId,
        pub mipped: TextureId,
        pub mip_read: TextureId,
        pub blended: TextureId,
        vertices: BufferId,
        pipeline: RenderPipelineId,
    }
    pub fn define(table: &ResourceTable, format: TextureFormat) -> Scene {
        let texture = |format, width, height, mips| {
            table
                .define_texture(
                    TextureDesc::new(
                        format,
                        Extent2D::new(width, height).unwrap(),
                        TextureUsage::COPY_SRC
                            | TextureUsage::COPY_DST
                            | TextureUsage::SAMPLED
                            | TextureUsage::RENDER_ATTACHMENT,
                    )
                    .unwrap()
                    .with_mip_level_count(mips)
                    .unwrap(),
                )
                .unwrap()
                .id()
        };
        let vertices = table
            .define_buffer(
                BufferDesc::new(24, BufferUsage::VERTEX | BufferUsage::COPY_DST).unwrap(),
            )
            .unwrap()
            .id();
        let blend = BlendComponent::new(
            BlendFactor::DestinationAlpha,
            BlendFactor::OneMinusDestinationAlpha,
            BlendOp::Add,
        );
        let pipeline = table
            .define_render_pipeline(
                RenderPipelineDesc::new(
                    format,
                    PrimitiveTopology::TriangleList,
                    VertexBufferLayout::new(
                        8,
                        vec![VertexAttribute::new(0, VertexFormat::Float32x2, 0)],
                    )
                    .unwrap(),
                    FragmentProgram::Solid,
                    BlendState::new(blend, blend),
                    RasterState::new(CullMode::None, FrontFace::CounterClockwise),
                )
                .unwrap(),
            )
            .unwrap()
            .id();
        Scene {
            source: texture(format, WIDTH, HEIGHT, 1),
            copied: texture(format, OUT_WIDTH, OUT_HEIGHT, 1),
            bgra: texture(TextureFormat::Bgra8Unorm, WIDTH, HEIGHT, 1),
            mipped: texture(format, 14, 10, 2),
            mip_read: texture(format, 7, 5, 1),
            blended: texture(format, WIDTH, HEIGHT, 1),
            vertices,
            pipeline,
        }
    }
    pub fn pixels(bpp: usize, width: u32, height: u32) -> (Vec<u8>, Vec<u8>, u32) {
        let stride = width as usize * bpp + 5;
        let mut padded = vec![0xa5; stride * height as usize];
        let mut tight = Vec::new();
        for y in 0..height as usize {
            for x in 0..width as usize {
                for c in 0..bpp {
                    let value = (19 + x * 23 + y * 37 + c * 79) as u8;
                    padded[y * stride + x * bpp + c] = value;
                    tight.push(value);
                }
            }
        }
        (padded, tight, stride as u32)
    }
    pub fn record_copy<'r, 'data>(
        table: &'r ResourceTable,
        scene: &Scene,
        data: &'data [u8],
        stride: u32,
    ) -> CommandBuffer<'r, 'data> {
        let mut encoder = CommandEncoder::new(table);
        let full = PixelRect::new(0, 0, WIDTH, HEIGHT).unwrap();
        encoder
            .write_texture(
                table.texture_ref(scene.source).unwrap(),
                TextureWrite::new(full, stride, data).unwrap(),
            )
            .unwrap();
        encoder
            .begin_render_pass(
                RenderPassDesc::new(
                    table,
                    table.texture_ref(scene.copied).unwrap(),
                    PixelRect::new(0, 0, OUT_WIDTH, OUT_HEIGHT).unwrap(),
                    LoadOp::Clear(Color::rgba(0.25, 0.5, 0.75, 0.125).unwrap()),
                    StoreOp::Store,
                )
                .unwrap(),
            )
            .unwrap()
            .end()
            .unwrap();
        encoder
            .copy_texture_to_texture(
                table.texture_ref(scene.source).unwrap(),
                PixelRect::new(3, 2, 7, 3).unwrap(),
                table.texture_ref(scene.copied).unwrap(),
                PixelRect::new(1, 1, 7, 3).unwrap(),
            )
            .unwrap();
        encoder
            .blit_texture(
                table.texture_ref(scene.source).unwrap(),
                0,
                table.texture_ref(scene.bgra).unwrap(),
                0,
                FilterMode::Nearest,
            )
            .unwrap();
        encoder.finish().unwrap()
    }
    pub fn record_mip<'r, 'data>(
        table: &'r ResourceTable,
        scene: &Scene,
        data: &'data [u8],
        stride: u32,
    ) -> CommandBuffer<'r, 'data> {
        let mut encoder = CommandEncoder::new(table);
        encoder
            .write_texture(
                table.texture_ref(scene.mipped).unwrap(),
                TextureWrite::new(PixelRect::new(0, 0, 7, 5).unwrap(), stride, data)
                    .unwrap()
                    .with_mip_level(1),
            )
            .unwrap();
        encoder
            .blit_texture(
                table.texture_ref(scene.mipped).unwrap(),
                1,
                table.texture_ref(scene.mip_read).unwrap(),
                0,
                FilterMode::Linear,
            )
            .unwrap();
        encoder.finish().unwrap()
    }
    pub fn record_blend<'r, 'data>(
        table: &'r ResourceTable,
        scene: &Scene,
        vertices: &'data [u8],
    ) -> CommandBuffer<'r, 'data> {
        let mut encoder = CommandEncoder::new(table);
        encoder
            .write_buffer(table.buffer_ref(scene.vertices).unwrap(), 0, vertices)
            .unwrap();
        let mut pass = encoder
            .begin_render_pass(
                RenderPassDesc::new(
                    table,
                    table.texture_ref(scene.blended).unwrap(),
                    PixelRect::new(0, 0, WIDTH, HEIGHT).unwrap(),
                    LoadOp::Clear(Color::rgba(0.25, 0.5, 0.75, 0.125).unwrap()),
                    StoreOp::Store,
                )
                .unwrap(),
            )
            .unwrap();
        pass.set_pipeline(table.render_pipeline_ref(scene.pipeline).unwrap())
            .unwrap();
        pass.set_vertex_buffer(table.buffer_ref(scene.vertices).unwrap(), 0)
            .unwrap();
        pass.set_uniforms(DrawUniforms::new(
            Transform::identity(),
            Color::rgba(0.5, 0.25, 0.75, 0.5).unwrap(),
        ))
        .unwrap();
        pass.draw(3, 0).unwrap();
        pass.end().unwrap();
        encoder.finish().unwrap()
    }
    pub fn vertices() -> Vec<u8> {
        [-1.0f32, -1.0, 3.0, -1.0, -1.0, 3.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect()
    }
}

#[cfg(any(target_os = "scarlet", test))]
mod import_overlap_scene {
    use sgfx::ir::*;
    pub const WIDTH: u32 = 13;
    pub const HEIGHT: u32 = 7;
    pub const SPLIT: u32 = 6;
    pub const LEFT: [u8; 4] = [51, 179, 26, 255];
    pub const RIGHT: [u8; 4] = [153, 26, 204, 255];
    pub struct Scene {
        pub target: TextureId,
        pub source: TextureId,
        vertices: BufferId,
        pipeline: RenderPipelineId,
        sampler: SamplerId,
        pub width: u32,
        pub height: u32,
    }
    pub fn target(table: &ResourceTable, width: u32, height: u32) -> TextureId {
        table
            .define_texture(
                TextureDesc::new(
                    TextureFormat::Bgra8Unorm,
                    Extent2D::new(width, height).unwrap(),
                    TextureUsage::PRESENT
                        | TextureUsage::RENDER_ATTACHMENT
                        | TextureUsage::COPY_SRC,
                )
                .unwrap(),
            )
            .unwrap()
            .id()
    }
    pub fn define(table: &ResourceTable, width: u32, height: u32) -> Scene {
        let target = target(table, width, height);
        let source = table
            .define_texture(
                TextureDesc::new(
                    TextureFormat::Bgra8Unorm,
                    Extent2D::new(WIDTH, HEIGHT).unwrap(),
                    TextureUsage::SAMPLED | TextureUsage::COPY_DST,
                )
                .unwrap(),
            )
            .unwrap()
            .id();
        let vertices = table
            .define_buffer(
                BufferDesc::new(96, BufferUsage::VERTEX | BufferUsage::COPY_DST).unwrap(),
            )
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
                FilterMode::Nearest,
                FilterMode::Nearest,
                AddressMode::ClampToEdge,
                AddressMode::ClampToEdge,
            ))
            .unwrap()
            .id();
        Scene {
            target,
            source,
            vertices,
            pipeline,
            sampler,
            width,
            height,
        }
    }
    pub fn producer<'r>(table: &'r ResourceTable, target: TextureId) -> CommandBuffer<'r, 'static> {
        let mut encoder = CommandEncoder::new(table);
        for (x, width, bgra) in [(0, WIDTH, LEFT), (SPLIT, WIDTH - SPLIT, RIGHT)] {
            encoder
                .begin_render_pass(
                    RenderPassDesc::new(
                        table,
                        table.texture_ref(target).unwrap(),
                        PixelRect::new(x, 0, width, HEIGHT).unwrap(),
                        LoadOp::Clear(
                            Color::rgba(
                                bgra[2] as f32 / 255.0,
                                bgra[1] as f32 / 255.0,
                                bgra[0] as f32 / 255.0,
                                1.0,
                            )
                            .unwrap(),
                        ),
                        StoreOp::Store,
                    )
                    .unwrap(),
                )
                .unwrap()
                .end()
                .unwrap();
        }
        encoder.finish().unwrap()
    }
    pub fn vertices() -> Vec<u8> {
        [
            [-1.0f32, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0, 1.0],
            [-1.0, -1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, -1.0, 1.0, 1.0],
        ]
        .into_iter()
        .flatten()
        .flat_map(f32::to_le_bytes)
        .collect()
    }
    pub fn draw<'r, 'data>(
        table: &'r ResourceTable,
        scene: &Scene,
        vertices: &'data [u8],
    ) -> CommandBuffer<'r, 'data> {
        let mut encoder = CommandEncoder::new(table);
        encoder
            .write_buffer(table.buffer_ref(scene.vertices).unwrap(), 0, vertices)
            .unwrap();
        let mut pass = encoder
            .begin_render_pass(
                RenderPassDesc::new(
                    table,
                    table.texture_ref(scene.target).unwrap(),
                    PixelRect::new(0, 0, scene.width, scene.height).unwrap(),
                    LoadOp::Clear(Color::rgba(0.0, 0.0, 0.0, 0.0).unwrap()),
                    StoreOp::Store,
                )
                .unwrap(),
            )
            .unwrap();
        pass.set_pipeline(table.render_pipeline_ref(scene.pipeline).unwrap())
            .unwrap();
        pass.set_vertex_buffer(table.buffer_ref(scene.vertices).unwrap(), 0)
            .unwrap();
        pass.set_texture(table.texture_ref(scene.source).unwrap())
            .unwrap();
        pass.set_sampler(table.sampler_ref(scene.sampler).unwrap())
            .unwrap();
        pass.set_uniforms(DrawUniforms::new(
            Transform::identity(),
            Color::rgba(1.0, 1.0, 1.0, 1.0).unwrap(),
        ))
        .unwrap();
        pass.draw(6, 0).unwrap();
        pass.end().unwrap();
        encoder.finish().unwrap()
    }
}

#[cfg(any(target_os = "scarlet", test))]
mod program_churn {
    use sgfx::{
        backend::{Completion, CompletionStatus},
        driver,
        ir::*,
    };
    use std::{rc::Rc, time::Duration};

    fn define(table: &ResourceTable, blue: bool) -> (ShaderModuleId, ProgrammableRenderPipelineId) {
        let colour = if blue {
            "vec4<f32>(0.0,0.0,1.0,1.0)"
        } else {
            "vec4<f32>(1.0,0.0,0.0,1.0)"
        };
        let module = table.define_shader_module(ShaderModuleDesc::wgsl(format!(r#"
            @vertex fn vs(@location(0) p:vec2<f32>) -> @builtin(position) vec4<f32> {{ return vec4<f32>(p,0.0,1.0); }}
            @fragment fn fs() -> @location(0) vec4<f32> {{ return {colour}; }}
        "#)).unwrap()).unwrap();
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
                .unwrap()
                .with_vertex_buffers(vec![
                    VertexBufferLayout::new(
                        8,
                        vec![VertexAttribute::new(0, VertexFormat::Float32x2, 0)],
                    )
                    .unwrap(),
                ])
                .unwrap(),
            )
            .unwrap()
            .id();
        (module.id(), pipeline)
    }

    fn draw(
        table: &ResourceTable,
        queue: &driver::Queue,
        resources: &mut driver::Resources,
        target: TextureId,
        vertices: BufferId,
        pipeline: ProgrammableRenderPipelineId,
    ) -> driver::Submission {
        let mut encoder = CommandEncoder::new(table);
        let mut pass = encoder
            .begin_render_pass(
                RenderPassDesc::new(
                    table,
                    table.texture_ref(target).unwrap(),
                    PixelRect::new(0, 0, 8, 8).unwrap(),
                    LoadOp::Clear(Color::rgba(0.0, 1.0, 0.0, 1.0).unwrap()),
                    StoreOp::Store,
                )
                .unwrap(),
            )
            .unwrap();
        pass.set_programmable_pipeline(table.programmable_render_pipeline_ref(pipeline).unwrap())
            .unwrap();
        pass.set_vertex_buffer(table.buffer_ref(vertices).unwrap(), 0)
            .unwrap();
        pass.draw(3, 0).unwrap();
        pass.end().unwrap();
        let commands = encoder.finish().unwrap();
        queue.submit(resources, &commands).unwrap()
    }

    pub fn run(preference: sgfx::BackendPreference) {
        let instance = driver::Instance::with_preference(preference).unwrap();
        let device = instance.adapters()[0].create_device().unwrap();
        let queue = device.create_queue().unwrap();
        let table = Rc::new(ResourceTable::new());
        let mut resources = device.create_resources(table.clone()).unwrap();
        let target = || {
            table
                .define_texture(
                    TextureDesc::new(
                        TextureFormat::Bgra8Unorm,
                        Extent2D::new(8, 8).unwrap(),
                        TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
                    )
                    .unwrap(),
                )
                .unwrap()
                .id()
        };
        let preserved = target();
        let current = target();
        let vertices = table
            .define_buffer(
                BufferDesc::new(24, BufferUsage::VERTEX | BufferUsage::COPY_DST).unwrap(),
            )
            .unwrap()
            .id();
        let data: Vec<_> = [-1.0f32, -1.0, 3.0, -1.0, -1.0, 3.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let mut encoder = CommandEncoder::new(&table);
        encoder
            .write_buffer(table.buffer_ref(vertices).unwrap(), 0, &data)
            .unwrap();
        let upload = queue
            .submit(&mut resources, &encoder.finish().unwrap())
            .unwrap();
        assert_eq!(
            upload.wait(Some(Duration::from_secs(10))).unwrap(),
            CompletionStatus::Complete
        );
        let mut old_receipt = None;
        let mut old_pixels = Vec::new();
        let mut old_ids = None;
        for generation in 0..512 {
            let (shader, pipeline) = define(&table, generation % 2 != 0);
            if let Some((old_shader, old_pipeline)) = old_ids {
                assert!(
                    table.shader_module_ref(old_shader).is_err(),
                    "old shader revived at {generation}"
                );
                assert!(
                    table
                        .programmable_render_pipeline_ref(old_pipeline)
                        .is_err(),
                    "old pipeline revived at {generation}"
                );
            }
            let target = if generation == 0 { preserved } else { current };
            let receipt = draw(&table, &queue, &mut resources, target, vertices, pipeline);
            assert_eq!(
                receipt.wait(Some(Duration::from_secs(10))).unwrap(),
                CompletionStatus::Complete
            );
            let pixels = queue.read_texture(&mut resources, target).unwrap();
            let expected = if generation % 2 == 0 {
                [0, 0, 255, 255]
            } else {
                [255, 0, 0, 255]
            };
            assert_eq!(pixels.len(), 8 * 8 * 4);
            for pixel in pixels.chunks_exact(4) {
                assert_eq!(pixel, expected, "native program generation {generation}");
            }
            resources
                .release_programmable_render_pipeline(pipeline)
                .unwrap();
            table
                .release_programmable_render_pipeline(pipeline)
                .unwrap();
            resources.release_shader_module(shader).unwrap();
            table.release_shader_module(shader).unwrap();
            assert!(table.programmable_render_pipeline_ref(pipeline).is_err());
            assert!(table.shader_module_ref(shader).is_err());
            if generation == 0 {
                old_ids = Some((shader, pipeline));
                old_receipt = Some(receipt);
                old_pixels = pixels;
            } else {
                assert_eq!(
                    old_receipt
                        .as_ref()
                        .unwrap()
                        .wait(Some(Duration::ZERO))
                        .unwrap(),
                    CompletionStatus::Complete
                );
                assert_eq!(
                    queue.read_texture(&mut resources, preserved).unwrap(),
                    old_pixels,
                    "old generation image changed at {generation}"
                );
            }
        }
        println!(
            "SGFX_PROGRAM_CHURN_OK generations=512 stale_ids=rejected retained_receipt=complete retained_pixels=unchanged"
        );
    }

    #[cfg(all(test, feature = "backend-wgpu"))]
    #[test]
    #[ignore = "requires a native GPU"]
    fn real_gpu_program_generation_churn() {
        run(sgfx::BackendPreference::Wgpu);
    }
}

#[cfg(target_os = "scarlet")]
mod native {
    use sgfx::{
        backend::{CommandExecutor, CommandSubmitter, Completion, CompletionStatus, SubmitError},
        ir::*,
    };
    use std::{
        rc::Rc,
        time::{Duration, Instant},
    };

    // Scarlet enables EL0 access to the AArch64 architectural counter. Reading
    // it avoids a clock syscall (and its scheduling point) inside each sample.
    // This is elapsed time, including preemption, not thread CPU time.
    #[cfg(target_arch = "aarch64")]
    struct BenchClock {
        frequency: u64,
    }
    #[cfg(target_arch = "aarch64")]
    impl BenchClock {
        fn new() -> Self {
            let frequency: u64;
            unsafe {
                core::arch::asm!("mrs {}, cntfrq_el0", out(reg) frequency, options(nostack, preserves_flags));
            }
            assert_ne!(frequency, 0);
            Self { frequency }
        }
        fn now(&self) -> u64 {
            let ticks: u64;
            // ISB and the compiler memory clobber keep the sample after the
            // preceding work. No `nomem`/`pure`: measurements must not move.
            unsafe {
                core::arch::asm!("isb", "mrs {}, cntvct_el0", out(reg) ticks, options(nostack, preserves_flags));
            }
            ticks
        }
        fn elapsed(&self, start: u64) -> u128 {
            u128::from(self.now().wrapping_sub(start)) * 1_000_000_000 / u128::from(self.frequency)
        }
        const NAME: &str = "cntvct";
    }
    #[cfg(not(target_arch = "aarch64"))]
    struct BenchClock;
    #[cfg(not(target_arch = "aarch64"))]
    impl BenchClock {
        fn new() -> Self {
            Self
        }
        fn now(&self) -> Instant {
            Instant::now()
        }
        fn elapsed(&self, start: Instant) -> u128 {
            start.elapsed().as_nanos()
        }
        const NAME: &str = "instant";
    }

    fn complete(receipt: &sgfx::Submission) {
        assert_eq!(
            receipt.wait(Some(Duration::from_secs(10))).unwrap(),
            CompletionStatus::Complete
        );
    }
    fn median(values: &mut [u128]) -> u128 {
        values.sort_unstable();
        values[values.len() / 2]
    }

    fn retry_submit(
        session: &mut sgfx::MappedTargetSession,
        commands: &CommandBuffer<'_, '_>,
    ) -> sgfx::Submission {
        let started = Instant::now();
        loop {
            match session.executor().submit(commands) {
                Ok(receipt) => return receipt,
                Err(SubmitError::Busy) if started.elapsed() < Duration::from_secs(10) => {
                    std::thread::sleep(Duration::from_millis(1));
                }
                result => panic!("submission failed: {result:?}"),
            }
        }
    }

    fn async_vertex_copy_contract(
        table: &ResourceTable,
        session: &mut sgfx::MappedTargetSession,
        target: TextureId,
        mode: &str,
    ) {
        use super::vertex_copy_scene::{BANDS, PREFIX, SIDE, STRIDE, UPDATES, VERTICES};
        assert!(session.executor().supports_async_submission());
        let (vertices, pipeline) = super::vertex_copy_scene::define(table);
        let palette = |update: u32, band: u32| -> [u8; 4] {
            [
                (17 + update * 29 + band * 47) as u8,
                (53 + update * 11 + band * 71) as u8,
                (97 + update * 43 + band * 23) as u8,
                (64 + (update * 17 + band * 19) % 192) as u8,
            ]
        };
        let mut pixels = vec![0; (SIDE * SIDE * 4) as usize];
        for update in 0..UPDATES {
            let mut data = Vec::with_capacity((VERTICES as u64 * STRIDE) as usize);
            for band in 0..BANDS {
                let left = -1.0 + 2.0 * band as f32 / BANDS as f32;
                let right = -1.0 + 2.0 * (band + 1) as f32 / BANDS as f32;
                for [x, y] in [
                    [left, -1.0f32],
                    [right, -1.0],
                    [left, 1.0],
                    [left, 1.0],
                    [right, -1.0],
                    [right, 1.0],
                ] {
                    data.extend_from_slice(&x.to_le_bytes());
                    data.extend_from_slice(&y.to_le_bytes());
                    data.extend_from_slice(&palette(update, band));
                }
            }
            let mut encoder = CommandEncoder::new(table);
            if update == 0 {
                // Ignoring the bound byte offset must not accidentally draw
                // valid vertices from a zero-initialized allocation prefix.
                encoder
                    .write_buffer(
                        table.buffer_ref(vertices).unwrap(),
                        0,
                        &[0xcd; PREFIX as usize],
                    )
                    .unwrap();
            }
            encoder
                .write_buffer(table.buffer_ref(vertices).unwrap(), PREFIX, &data)
                .unwrap();
            super::vertex_copy_scene::record_passes(
                &mut encoder,
                table,
                target,
                vertices,
                pipeline,
            );
            // Both passes share one tracked logical submission. The backend
            // may split them into native packets; ten changing generations
            // reuse its four upload arenas without accepting stale A/B data.
            let commands = encoder.finish().unwrap();
            let receipt = retry_submit(session, &commands);
            drop(commands);
            drop(data);
            complete(&receipt);
            pixels.fill(0xa5);
            session
                .readback_bgra(
                    target,
                    &mut pixels,
                    SIDE * 4,
                    PixelRect::new(0, 0, SIDE, SIDE).unwrap(),
                )
                .unwrap();
            for (index, actual) in pixels.chunks_exact(4).enumerate() {
                let x = index as u32 % SIDE;
                let band = x / (SIDE / BANDS);
                let [r, g, b, a] = palette(update, band);
                assert_eq!(
                    actual,
                    [b, g, r, a],
                    "native vertex copy mismatch: mode={mode} update={update} x={x} y={}",
                    index as u32 / SIDE
                );
            }
        }
        // VertexColor interprets source attributes on the CPU, then stages the
        // resulting native VBO. Actual GPU pixels certify that final VBO copy;
        // this is not an arbitrary packed-source-buffer byte readback test.
        println!(
            "SGFX_ASYNC_VERTEX_COPY_OK mode={mode} updates={UPDATES} offset={PREFIX} pixels_per_update={}",
            SIDE * SIDE
        );
    }

    fn import_overlap_contract() {
        use super::import_overlap_scene as scene;
        let device = sgfx::Device::open("/dev/gpu0").unwrap();
        let producer_context = device.create_context().unwrap();
        let shared_context = device.create_context().unwrap();
        let producer_table = Rc::new(ResourceTable::new());
        let producer_target = scene::target(&producer_table, scene::WIDTH, scene::HEIGHT);
        let mut producer = producer_context
            .create_mapped_target_session(producer_table.clone(), &[producer_target])
            .unwrap();
        let commands = scene::producer(&producer_table, producer_target);
        complete(&retry_submit(&mut producer, &commands));
        let lease = producer
            .image(producer_target)
            .unwrap()
            .shared_handle()
            .duplicate()
            .unwrap();
        let vertices = scene::vertices();
        let paint = |session: &mut sgfx::MappedTargetSession,
                     table: &ResourceTable,
                     objects: &scene::Scene| {
            let commands = scene::draw(table, objects, &vertices);
            complete(&retry_submit(session, &commands));
            let mut pixels = vec![0xa5; (objects.width * objects.height * 4) as usize];
            session
                .readback_bgra(
                    objects.target,
                    &mut pixels,
                    objects.width * 4,
                    PixelRect::new(0, 0, objects.width, objects.height).unwrap(),
                )
                .unwrap();
            for (index, actual) in pixels.chunks_exact(4).enumerate() {
                let x = index as u32 % objects.width;
                let expected = if (2 * x + 1) * scene::WIDTH < 2 * scene::SPLIT * objects.width {
                    scene::LEFT
                } else {
                    scene::RIGHT
                };
                for (a, e) in actual.iter().copied().zip(expected) {
                    assert!(
                        a.abs_diff(e) <= 1,
                        "import overlap pixel mismatch at {index}: {actual:?} != {expected:?}"
                    );
                }
            }
        };
        for generation in 0..6 {
            let old_table = Rc::new(ResourceTable::new());
            let old_scene = scene::define(&old_table, scene::WIDTH, scene::HEIGHT);
            let new_table = Rc::new(ResourceTable::new());
            let new_scene = scene::define(&new_table, 7 + generation % 3, 5 + generation % 2);
            let mut old = shared_context
                .create_mapped_target_session(old_table.clone(), &[old_scene.target])
                .unwrap();
            let mut new = shared_context
                .create_mapped_target_session(new_table.clone(), &[new_scene.target])
                .unwrap();
            old.import_shared_bgra_texture(old_scene.source, lease.duplicate().unwrap())
                .unwrap();
            new.import_shared_bgra_texture(new_scene.source, lease.duplicate().unwrap())
                .unwrap();
            paint(&mut old, &old_table, &old_scene);
            paint(&mut new, &new_table, &new_scene);
            old.release_imported_texture(old_scene.source).unwrap();
            paint(&mut new, &new_table, &new_scene);
            old.import_shared_bgra_texture(old_scene.source, lease.duplicate().unwrap())
                .unwrap();
            let pending_commands = scene::draw(&old_table, &old_scene, &vertices);
            let pending = retry_submit(&mut old, &pending_commands);
            drop(pending_commands);
            drop(old); // Normal resize retirement, without explicit import release.
            assert_eq!(
                pending.poll().unwrap(),
                CompletionStatus::Complete,
                "session Drop must settle accepted GPU work before destroying its resources"
            );
            paint(&mut new, &new_table, &new_scene);
            new.release_imported_texture(new_scene.source).unwrap();
            new.import_shared_bgra_texture(new_scene.source, lease.duplicate().unwrap())
                .unwrap();
            paint(&mut new, &new_table, &new_scene);
            drop(new);
            println!("SGFX_IMPORT_OVERLAP_GENERATION_OK generation={generation}");
        }
        // The producer owner can retire while a transferred old-size lease
        // remains valid; it is not tied to the consumer viewport extent.
        drop(producer);
        let table = Rc::new(ResourceTable::new());
        let objects = scene::define(&table, 7, 5);
        let mut session = shared_context
            .create_mapped_target_session(table.clone(), &[objects.target])
            .unwrap();
        session
            .import_shared_bgra_texture(objects.source, lease)
            .unwrap();
        paint(&mut session, &table, &objects);
        drop(session);
        println!(
            "SGFX_IMPORT_OVERLAP_ONLY_OK generations=6 old_drop_new_draw=6 producer_drop_lease=1"
        );
    }

    fn narrow_texture_contract() {
        use super::narrow_texture_scene as scene;
        let info = gpu_raw::Gpu::open("/dev/gpu0")
            .unwrap()
            .query_info()
            .unwrap();
        let native =
            info.execution_support & gpu_raw::GPU_EXECUTION_SUPPORT_NATIVE_NARROW_COLOR != 0;
        println!("SGFX_NARROW_CAP native={native}");
        assert!(
            native,
            "--narrow-only requires native narrow kernel storage, not emulation"
        );
        let instance =
            sgfx::driver::Instance::with_preference(sgfx::BackendPreference::ScarletVirgl).unwrap();
        let device = instance.adapters()[0].create_device().unwrap();
        let queue = device.create_queue().unwrap();
        let close = |actual: u8, expected: u8| {
            assert!(
                actual.abs_diff(expected) <= 1,
                "native narrow channel mismatch: {actual} != {expected}"
            )
        };
        for (format, bpp) in [(TextureFormat::R8Unorm, 1), (TextureFormat::Rg8Unorm, 2)] {
            let table = Rc::new(ResourceTable::new());
            let objects = scene::define(&table, format);
            let mut resources = device.create_resources(table.clone()).unwrap();
            let (padded, tight, stride) = scene::pixels(bpp, scene::WIDTH, scene::HEIGHT);
            let commands = scene::record_copy(&table, &objects, &padded, stride);
            let receipt = queue.submit(&mut resources, &commands).unwrap();
            assert_eq!(
                receipt.wait(Some(Duration::from_secs(10))).unwrap(),
                CompletionStatus::Complete
            );
            assert_eq!(
                queue.read_texture(&mut resources, objects.source).unwrap(),
                tight,
                "raw native upload/readback"
            );
            let copied = queue.read_texture(&mut resources, objects.copied).unwrap();
            for y in 0..scene::OUT_HEIGHT as usize {
                for x in 0..scene::OUT_WIDTH as usize {
                    for c in 0..bpp {
                        let actual = copied[(y * scene::OUT_WIDTH as usize + x) * bpp + c];
                        if (1..8).contains(&x) && (1..4).contains(&y) {
                            assert_eq!(
                                actual,
                                tight[((y + 1) * scene::WIDTH as usize + x + 2) * bpp + c],
                                "native region copy"
                            );
                        } else {
                            close(actual, if c == 0 { 64 } else { 128 });
                        }
                    }
                }
            }
            let bgra = queue.read_texture(&mut resources, objects.bgra).unwrap();
            for (index, actual) in bgra.chunks_exact(4).enumerate() {
                for (a, e) in actual.iter().copied().zip([
                    0,
                    if bpp == 2 { tight[index * bpp + 1] } else { 0 },
                    tight[index * bpp],
                    255,
                ]) {
                    close(a, e);
                }
            }
            println!("SGFX_NARROW_UPLOAD_COPY_SAMPLE_OK format={format:?}");
            let (mip_data, mip_tight, mip_stride) = scene::pixels(bpp, 7, 5);
            let commands = scene::record_mip(&table, &objects, &mip_data, mip_stride);
            let receipt = queue.submit(&mut resources, &commands).unwrap();
            assert_eq!(
                receipt.wait(Some(Duration::from_secs(10))).unwrap(),
                CompletionStatus::Complete
            );
            for (actual, expected) in queue
                .read_texture(&mut resources, objects.mip_read)
                .unwrap()
                .into_iter()
                .zip(mip_tight)
            {
                close(actual, expected);
            }
            println!("SGFX_NARROW_MIP_OK format={format:?}");
            let vertices = scene::vertices();
            let commands = scene::record_blend(&table, &objects, &vertices);
            let receipt = queue.submit(&mut resources, &commands).unwrap();
            assert_eq!(
                receipt.wait(Some(Duration::from_secs(10))).unwrap(),
                CompletionStatus::Complete
            );
            for pixel in queue
                .read_texture(&mut resources, objects.blended)
                .unwrap()
                .chunks_exact(bpp)
            {
                for (c, actual) in pixel.iter().copied().enumerate() {
                    close(actual, if c == 0 { 128 } else { 64 });
                }
            }
            println!(
                "SGFX_NARROW_TEXTURE_OK format={format:?} bpp={bpp} upload_pixels={} copy_pixels=21 mip_pixels=35 blend_pixels={}",
                scene::WIDTH * scene::HEIGHT,
                scene::WIDTH * scene::HEIGHT
            );
        }
        println!("SGFX_NARROW_ONLY_OK");
    }

    fn low_level() {
        let instance =
            sgfx::driver::Instance::with_preference(sgfx::BackendPreference::ScarletVirgl).unwrap();
        let device = instance.adapters()[0].create_device().unwrap();
        let table = Rc::new(ResourceTable::new());
        let target = table
            .define_texture(
                TextureDesc::new(
                    TextureFormat::Bgra8Unorm,
                    Extent2D::new(16, 16).unwrap(),
                    TextureUsage::RENDER_ATTACHMENT
                        | TextureUsage::COPY_SRC
                        | TextureUsage::PRESENT,
                )
                .unwrap(),
            )
            .unwrap()
            .id();
        let mut resources = device.create_resources(table.clone()).unwrap();
        let unused = table
            .define_buffer(BufferDesc::new(8, BufferUsage::COPY_DST).unwrap())
            .unwrap()
            .id();
        resources.release_buffer(unused).unwrap();
        table.release_buffer(unused).unwrap();
        // Late definitions exercise the low-level metadata synchronization path.
        let buffer = table
            .define_buffer(
                BufferDesc::new(16, BufferUsage::COPY_DST | BufferUsage::COPY_SRC).unwrap(),
            )
            .unwrap()
            .id();
        let image = device
            .create_presentation_image(16, 16, TextureFormat::Bgra8Unorm)
            .unwrap();
        drop(image.duplicate_shared_handle().unwrap());
        resources.map_presentation_image(target, &image).unwrap();
        let queue = device.create_queue().unwrap();
        let data = vec![0x5a; 16];
        let mut encoder = CommandEncoder::new(&table);
        encoder
            .write_buffer(table.buffer_ref(buffer).unwrap(), 0, &data)
            .unwrap();
        encoder
            .begin_render_pass(
                RenderPassDesc::new(
                    &table,
                    table.texture_ref(target).unwrap(),
                    PixelRect::new(0, 0, 16, 16).unwrap(),
                    LoadOp::Clear(Color::rgba(0.0, 1.0, 0.0, 1.0).unwrap()),
                    StoreOp::Store,
                )
                .unwrap(),
            )
            .unwrap()
            .end()
            .unwrap();
        let commands = encoder.finish().unwrap();
        let receipt = queue.submit(&mut resources, &commands).unwrap();
        let retained = receipt.clone();
        drop(receipt);
        drop(commands);
        drop(data);
        assert_eq!(
            retained.wait(Some(Duration::from_secs(10))).unwrap(),
            CompletionStatus::Complete
        );
        assert_eq!(resources.read_buffer(buffer, 0, 16).unwrap(), [0x5a; 16]);
        let pixels = queue.read_texture(&mut resources, target).unwrap();
        assert!(pixels.chunks_exact(4).all(|p| p == [0, 255, 0, 255]));
        resources.release_buffer(buffer).unwrap();
        resources.unmap_presentation_image(target);
        drop(resources);
        drop(queue);
        drop(image);
        drop(device);
        drop(instance);
        std::thread::spawn(move || {
            assert_eq!(retained.wait(None).unwrap(), CompletionStatus::Complete)
        })
        .join()
        .unwrap();
        println!("SGFX_LOW_LEVEL_OK");
    }

    pub fn run() {
        if std::env::args().any(|arg| arg == "--program-churn-only") {
            crate::program_churn::run(sgfx::BackendPreference::ScarletVirgl);
            return;
        }
        if std::env::args().any(|arg| arg == "--import-overlap-only") {
            import_overlap_contract();
            return;
        }
        if std::env::args().any(|arg| arg == "--narrow-only") {
            narrow_texture_contract();
            return;
        }
        let copy_only = std::env::args().any(|arg| arg == "--copy-only");
        if !copy_only {
            low_level();
        }
        let mode = if cfg!(all(
            target_pointer_width = "64",
            feature = "backend-dynamic"
        )) {
            "dynamic"
        } else {
            "static"
        };
        println!("SGFX_SMOKE_BEGIN mode={mode}");
        let device = sgfx::Device::open("/dev/gpu0").unwrap();
        println!(
            "SGFX_DRIVER mode={mode} backend={} library={:?}",
            device.backend(),
            device.backend_library()
        );
        #[cfg(target_arch = "aarch64")]
        println!(
            "SGFX_CPU mode={mode} lse={}",
            std::arch::is_aarch64_feature_detected!("lse")
        );
        assert!(device.capabilities().supports_image_readback());
        let context = device.create_context().unwrap();
        let table = Rc::new(ResourceTable::new());
        let extent = Extent2D::new(64, 64).unwrap();
        let target = table
            .define_texture(
                TextureDesc::new(
                    TextureFormat::Bgra8Unorm,
                    extent,
                    TextureUsage::RENDER_ATTACHMENT
                        | TextureUsage::COPY_SRC
                        | TextureUsage::COPY_DST
                        | TextureUsage::PRESENT,
                )
                .unwrap(),
            )
            .unwrap()
            .id();
        if copy_only {
            let mut session = context
                .create_mapped_target_session(table.clone(), &[target])
                .unwrap();
            async_vertex_copy_contract(&table, &mut session, target, mode);
            println!("SGFX_COPY_ONLY_OK mode={mode}");
            return;
        }
        let vertices = table
            .define_buffer(
                BufferDesc::new(24, BufferUsage::VERTEX | BufferUsage::COPY_DST).unwrap(),
            )
            .unwrap()
            .id();
        let pipeline = table
            .define_render_pipeline(
                RenderPipelineDesc::new(
                    TextureFormat::Bgra8Unorm,
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
            .unwrap()
            .id();
        let mut session = context
            .create_mapped_target_session(table.clone(), &[target])
            .unwrap();
        assert!(session.executor().supports_async_submission());
        let vertex_data: Vec<u8> = [-1.0f32, -1.0, 3.0, -1.0, -1.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut upload = CommandEncoder::new(&table);
        upload
            .write_buffer(table.buffer_ref(vertices).unwrap(), 0, &vertex_data)
            .unwrap();
        let upload = upload.finish().unwrap();
        let receipt = session.executor().submit(&upload).unwrap();
        drop(upload);
        drop(vertex_data);
        complete(&receipt);
        let record = |draws: u32| {
            let mut encoder = CommandEncoder::new(&table);
            let desc = RenderPassDesc::new(
                &table,
                table.texture_ref(target).unwrap(),
                PixelRect::new(0, 0, 64, 64).unwrap(),
                LoadOp::Clear(Color::rgba(0.0, 0.0, 0.0, 1.0).unwrap()),
                StoreOp::Store,
            )
            .unwrap();
            let mut pass = encoder.begin_render_pass(desc).unwrap();
            pass.set_pipeline(table.render_pipeline_ref(pipeline).unwrap())
                .unwrap();
            pass.set_vertex_buffer(table.buffer_ref(vertices).unwrap(), 0)
                .unwrap();
            pass.set_uniforms(DrawUniforms::new(
                Transform::identity(),
                Color::rgba(1.0, 0.0, 0.0, 1.0).unwrap(),
            ))
            .unwrap();
            for _ in 0..draws {
                pass.draw(3, 0).unwrap();
            }
            pass.end().unwrap();
            encoder.finish().unwrap()
        };
        let commands = record(1);
        complete(&session.executor().submit(&commands).unwrap());
        let mut pixels = vec![0; 64 * 64 * 4];
        session
            .readback_bgra(
                target,
                &mut pixels,
                64 * 4,
                PixelRect::new(0, 0, 64, 64).unwrap(),
            )
            .unwrap();
        assert!(
            pixels.chunks_exact(4).all(|p| p == [0, 0, 255, 255]),
            "red triangle readback differs: {:?}",
            &pixels[..16]
        );
        println!("SGFX_PIXELS_OK mode={mode}");
        async_vertex_copy_contract(&table, &mut session, target, mode);
        // Synchronous execution uses the same borrowed boundary.
        session.executor().execute(&commands).unwrap();
        // Resource definitions added after session creation are synchronized once.
        let texture = table
            .define_texture(
                TextureDesc::new(
                    TextureFormat::Bgra8Unorm,
                    Extent2D::new(1024, 1024).unwrap(),
                    TextureUsage::SAMPLED | TextureUsage::COPY_DST | TextureUsage::COPY_SRC,
                )
                .unwrap(),
            )
            .unwrap();
        let data = vec![0x7f; 1024 * 1024 * 4];
        let mut large = CommandEncoder::new(&table);
        large
            .write_texture(
                texture,
                TextureWrite::new(PixelRect::new(0, 0, 1024, 1024).unwrap(), 4096, &data).unwrap(),
            )
            .unwrap();
        let large = large.finish().unwrap();
        let receipt = session.executor().submit(&large).unwrap();
        drop(large);
        drop(data);
        complete(&receipt);
        let mut copy = CommandEncoder::new(&table);
        copy.copy_texture_to_texture(
            texture,
            PixelRect::new(0, 0, 64, 64).unwrap(),
            table.texture_ref(target).unwrap(),
            PixelRect::new(0, 0, 64, 64).unwrap(),
        )
        .unwrap();
        complete(&session.executor().submit(&copy.finish().unwrap()).unwrap());
        session
            .readback_bgra(
                target,
                &mut pixels,
                256,
                PixelRect::new(0, 0, 64, 64).unwrap(),
            )
            .unwrap();
        assert!(
            pixels.iter().all(|byte| *byte == 0x7f),
            "large upload contents differ"
        );
        println!("SGFX_LARGE_UPLOAD_OK mode={mode}");
        let clock = BenchClock::new();
        let mut overhead = [0; 80];
        for sample in &mut overhead {
            let start = clock.now();
            *sample = clock.elapsed(start);
        }
        println!(
            "SGFX_CLOCK mode={mode} source={} overhead_ns={}",
            BenchClock::NAME,
            median(&mut overhead)
        );
        for draws in [1, 200] {
            let recorded = record(draws);
            for _ in 0..30 {
                complete(&session.executor().submit(&recorded).unwrap());
            }
            let mut submit_times = Vec::with_capacity(80);
            let mut frame_times = Vec::with_capacity(80);
            let mut record_times = Vec::with_capacity(80);
            let mut total_times = Vec::with_capacity(80);
            for _ in 0..80 {
                let frame_start = clock.now();
                let frame = record(draws);
                record_times.push(clock.elapsed(frame_start));
                let start = clock.now();
                let receipt = session.executor().submit(&frame).unwrap();
                submit_times.push(clock.elapsed(start));
                total_times.push(clock.elapsed(frame_start));
                complete(&receipt);
                frame_times.push(clock.elapsed(frame_start));
            }
            println!(
                "SGFX_BENCH mode={mode} draws={draws} samples=80 record_ns={} submit_ns={} total_ns={} frame_ns={}",
                median(&mut record_times),
                median(&mut submit_times),
                median(&mut total_times),
                median(&mut frame_times)
            );
        }
        for _ in 0..16 {
            drop(retry_submit(&mut session, &commands));
        }
        let final_receipt = retry_submit(&mut session, &commands);
        drop(commands);
        drop(session);
        drop(context);
        drop(device);
        drop(table);
        std::thread::spawn(move || complete(&final_receipt))
            .join()
            .unwrap();
        println!("SGFX_LIFETIME_OK mode={mode}");
        println!("SGFX_SMOKE_OK mode={mode}");
    }
}

fn main() {
    #[cfg(target_os = "scarlet")]
    {
        std::panic::set_hook(Box::new(|info| {
            println!("SGFX_PANIC {info}\nSGFX_PANIC_END")
        }));
        println!("SGFX_MAIN");
        #[cfg(all(target_pointer_width = "64", feature = "backend-dynamic"))]
        if !std::env::args().any(|arg| {
            arg == "--child"
                || arg == "--copy-only"
                || arg == "--narrow-only"
                || arg == "--import-overlap-only"
                || arg == "--program-churn-only"
        }) {
            for program in ["/bin/sgfx-probe", "/bin/sgfx-legacy-smoke"] {
                if std::path::Path::new(program).is_file() {
                    assert!(
                        std::process::Command::new(program)
                            .status()
                            .unwrap()
                            .success(),
                        "{program} failed"
                    );
                }
            }
            if std::path::Path::new("/bin/sgfx-probe").is_file() {
                assert!(
                    !std::process::Command::new("/bin/sgfx-probe")
                        .env("SGFX_DRIVER_PATH", "/tmp/sgfx-driver-does-not-exist")
                        .status()
                        .unwrap()
                        .success(),
                    "missing driver silently fell back"
                );
                println!("SGFX_MISSING_DRIVER_OK");
            }
            // Both modes run as fresh sibling children. Alternating their order
            // avoids comparing a child against its long-lived coordinator and
            // exposes warm-up/frequency/order effects in the per-round results.
            for round in 0..4 {
                let modes = if round % 2 == 0 {
                    ["static", "dynamic"]
                } else {
                    ["dynamic", "static"]
                };
                for mode in modes {
                    println!("SGFX_ROUND round={round} mode={mode}");
                    let status = std::process::Command::new(format!("/bin/sgfx-{mode}-smoke"))
                        .arg("--child")
                        .status()
                        .unwrap();
                    assert!(
                        status.success(),
                        "{mode} comparison failed in round {round}"
                    );
                }
            }
            println!("SGFX_DYNAMIC_ALL_OK");
            return;
        }
        native::run();
    }
    #[cfg(not(target_os = "scarlet"))]
    eprintln!("This fixture executes Scarlet GPU syscalls; run it in the isolated guest.");
}

#[cfg(test)]
mod tests {
    use super::vertex_copy_scene;
    use sgfx::ir::*;

    #[test]
    fn import_overlap_scene_passes_real_ir_validation() {
        use super::import_overlap_scene as scene;
        let producer = ResourceTable::new();
        let target = scene::target(&producer, scene::WIDTH, scene::HEIGHT);
        scene::producer(&producer, target);
        for (width, height) in [(13, 7), (7, 5), (8, 6), (9, 5)] {
            let table = ResourceTable::new();
            let objects = scene::define(&table, width, height);
            scene::draw(&table, &objects, &scene::vertices());
        }
    }

    #[test]
    fn native_narrow_scene_passes_real_ir_validation() {
        use super::narrow_texture_scene as scene;
        for (format, bpp) in [(TextureFormat::R8Unorm, 1), (TextureFormat::Rg8Unorm, 2)] {
            let table = ResourceTable::new();
            let objects = scene::define(&table, format);
            let (data, _, stride) = scene::pixels(bpp, scene::WIDTH, scene::HEIGHT);
            scene::record_copy(&table, &objects, &data, stride);
            let (data, _, stride) = scene::pixels(bpp, 7, 5);
            scene::record_mip(&table, &objects, &data, stride);
            scene::record_blend(&table, &objects, &scene::vertices());
        }
    }

    #[test]
    fn copy_scene_passes_real_ir_validation_with_nonzero_buffer_offsets() {
        let table = ResourceTable::new();
        let target = table
            .define_texture(
                TextureDesc::new(
                    TextureFormat::Bgra8Unorm,
                    Extent2D::new(vertex_copy_scene::SIDE, vertex_copy_scene::SIDE).unwrap(),
                    TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
                )
                .unwrap(),
            )
            .unwrap()
            .id();
        let (vertices, pipeline) = vertex_copy_scene::define(&table);
        let data =
            vec![0; (u64::from(vertex_copy_scene::VERTICES) * vertex_copy_scene::STRIDE) as usize];
        let mut encoder = CommandEncoder::new(&table);
        encoder
            .write_buffer(
                table.buffer_ref(vertices).unwrap(),
                vertex_copy_scene::PREFIX,
                &data,
            )
            .unwrap();
        vertex_copy_scene::record_passes(&mut encoder, &table, target, vertices, pipeline);
        encoder.finish().unwrap();
    }
}
