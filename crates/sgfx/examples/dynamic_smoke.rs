//! Build this same scene with backend-dynamic and backend-scarlet-virgl to
//! compare the dynamic boundary against the static implementation in one guest.
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
        low_level();
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
        if !std::env::args().any(|arg| arg == "--child") {
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
