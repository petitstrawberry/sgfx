//! Opt-in Scarlet/VirGL diagnostic through the public SGFX facade.
//!
//! Build/install `sgfx-native-completion-smoke` through a normal cargo-scarlet
//! bundle, then run it inside Scarlet. Missing GPU support is a failure, not a
//! skipped success. The same scenarios also remain native Rust unit tests.

#[cfg(target_os = "scarlet")]
mod native {

    use std::rc::Rc;
    use std::time::{Duration, Instant};

    use sgfx::backend::{
        CommandExecutor, CommandSubmitter, Completion, CompletionStatus, SubmitError,
    };
    use sgfx::ir::{
        BlendState, BufferDesc, BufferUsage, Color, CommandBuffer, CommandEncoder, CullMode,
        DrawUniforms, Extent2D, FragmentProgram, FrontFace, LoadOp, PixelRect, PrimitiveTopology,
        RasterState, RenderPassDesc, RenderPipelineDesc, ResourceTable, StoreOp, TextureDesc,
        TextureFormat, TextureId, TextureUsage, TextureWrite, Transform, VertexAttribute,
        VertexBufferLayout, VertexFormat,
    };
    use sgfx::{BackendKind, Device, MappedTargetSession, Submission};

    fn target(table: &ResourceTable, width: u32, height: u32) -> TextureId {
        table
            .define_texture(
                TextureDesc::new(
                    TextureFormat::Bgra8Unorm,
                    Extent2D::new(width, height).expect("extent"),
                    TextureUsage::PRESENT
                        | TextureUsage::RENDER_ATTACHMENT
                        | TextureUsage::COPY_DST
                        | TextureUsage::COPY_SRC,
                )
                .expect("target descriptor"),
            )
            .expect("target")
            .id()
    }

    fn session(table: &Rc<ResourceTable>, targets: &[TextureId]) -> MappedTargetSession {
        let device = Device::open("/dev/gpu0").expect("open native GPU");
        assert_eq!(device.backend(), BackendKind::ScarletVirgl);
        device
            .create_context()
            .expect("context")
            .create_mapped_target_session(Rc::clone(table), targets)
            .expect("mapped session")
    }

    fn submit(session: &mut MappedTargetSession, commands: &CommandBuffer<'_, '_>) -> Submission {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            match session.executor().submit(commands) {
                Ok(receipt) => return receipt,
                Err(SubmitError::Busy) if Instant::now() < deadline => {
                    // Retry only proven non-acceptance, outside the implementation.
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(error) => panic!("tracked submit: {error:?}"),
            }
        }
    }

    fn complete(receipt: &Submission) {
        assert!(matches!(
            receipt.wait(Some(Duration::ZERO)),
            Ok(CompletionStatus::Pending | CompletionStatus::Complete)
        ));
        assert_eq!(
            receipt.wait(Some(Duration::from_secs(10))).expect("wait"),
            CompletionStatus::Complete
        );
        assert_eq!(receipt.poll().expect("poll"), CompletionStatus::Complete);
        assert_eq!(
            receipt.wait(None).expect("unbounded wait after completion"),
            CompletionStatus::Complete
        );
    }

    fn clear<'r>(
        table: &'r ResourceTable,
        target: TextureId,
        color: [f32; 4],
    ) -> CommandBuffer<'r, 'static> {
        let target = table.texture_ref(target).expect("target ref");
        let extent = table.texture(target).expect("target descriptor").extent();
        let area = PixelRect::new(0, 0, extent.width(), extent.height()).expect("area");
        let mut encoder = CommandEncoder::new(table);
        encoder
            .begin_render_pass(
                RenderPassDesc::new(
                    table,
                    target,
                    area,
                    LoadOp::Clear(
                        Color::rgba(color[0], color[1], color[2], color[3]).expect("color"),
                    ),
                    StoreOp::Store,
                )
                .expect("pass"),
            )
            .expect("begin")
            .end()
            .expect("end");
        encoder.finish().expect("clear commands")
    }

    fn pixels(session: &MappedTargetSession, target: TextureId) -> Vec<u8> {
        let image = session.image(target).expect("mapped target");
        let (width, height) = (image.width(), image.height());
        let mut pixels = vec![0; width as usize * height as usize * 4];
        session
            .readback_bgra(
                target,
                &mut pixels,
                width * 4,
                PixelRect::new(0, 0, width, height).expect("readback area"),
            )
            .expect("readback");
        pixels
    }

    #[cfg_attr(test, test)]
    fn split_upload_and_copy_snapshot_borrowed_bytes() {
        let table = Rc::new(ResourceTable::new());
        // An odd row width and a >64 KiB upload exercise multiple native chunks.
        let (width, height) = (257, 129);
        let output = target(&table, width, height);
        let source = table
            .define_texture(
                TextureDesc::new(
                    TextureFormat::Bgra8Unorm,
                    Extent2D::new(width, height).expect("extent"),
                    TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
                )
                .expect("source descriptor"),
            )
            .expect("source");
        let mut session = session(&table, &[output]);
        let area = PixelRect::new(0, 0, width, height).expect("area");
        let expected: Vec<u8> = (0..width * height)
            .flat_map(|pixel| [(pixel % width) as u8, (pixel / width) as u8, 197, 255])
            .collect();
        let mut upload = expected.clone();
        let mut encoder = CommandEncoder::new(&table);
        encoder
            .write_texture(
                source,
                TextureWrite::new(area, width * 4, &upload).expect("upload layout"),
            )
            .expect("write");
        encoder
            .copy_texture_to_texture(
                source,
                area,
                table.texture_ref(output).expect("output"),
                area,
            )
            .expect("copy");
        let commands = encoder.finish().expect("commands");
        let receipt = submit(&mut session, &commands);
        drop(commands);
        upload.fill(0);
        drop(upload);
        complete(&receipt);
        assert_eq!(pixels(&session, output), expected);
    }

    #[cfg_attr(test, test)]
    fn oversized_stream_is_rejected_before_initializing_gpu_state() {
        let table = Rc::new(ResourceTable::new());
        let output = target(&table, 32, 32);
        let source = table
            .define_texture(
                TextureDesc::new(
                    TextureFormat::Bgra8Unorm,
                    Extent2D::new(1024, 513).expect("extent"),
                    TextureUsage::COPY_DST,
                )
                .expect("source descriptor"),
            )
            .expect("source");
        let mut session = session(&table, &[output]);
        let bytes = vec![0; 1024 * 513 * 4];
        let mut encoder = CommandEncoder::new(&table);
        encoder
            .begin_render_pass(
                RenderPassDesc::new(
                    &table,
                    table.texture_ref(output).expect("output"),
                    PixelRect::new(0, 0, 32, 32).expect("area"),
                    LoadOp::Clear(Color::rgba(1.0, 0.0, 0.0, 1.0).expect("color")),
                    StoreOp::Store,
                )
                .expect("pass"),
            )
            .expect("begin")
            .end()
            .expect("end");
        encoder
            .write_texture(
                source,
                TextureWrite::new(
                    PixelRect::new(0, 0, 1024, 513).expect("upload area"),
                    4096,
                    &bytes,
                )
                .expect("upload"),
            )
            .expect("write");
        let commands = encoder.finish().expect("oversized commands");
        assert!(
            matches!(
                session.executor().submit(&commands),
                Err(SubmitError::Rejected(_))
            ),
            "oversized staging must reject without accepting the preceding clear"
        );
        drop(commands);
        let commands = clear(&table, output, [0.0, 1.0, 0.0, 1.0]);
        complete(&submit(&mut session, &commands));
        assert!(
            pixels(&session, output)
                .chunks_exact(4)
                .all(|pixel| pixel == [0, 255, 0, 255])
        );
    }

    #[cfg_attr(test, test)]
    fn several_submissions_share_a_queue_without_an_intermediate_wait() {
        let table = Rc::new(ResourceTable::new());
        let targets: Vec<_> = (0..4).map(|_| target(&table, 32, 32)).collect();
        let mut session = session(&table, &targets);
        let colors = [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0; 4],
        ];
        let expected = [
            [0, 0, 255, 255],
            [0, 255, 0, 255],
            [255, 0, 0, 255],
            [255; 4],
        ];
        let receipts: Vec<_> = targets
            .iter()
            .zip(colors)
            .map(|(&target, color)| submit(&mut session, &clear(&table, target, color)))
            .collect();
        for receipt in receipts.iter().rev() {
            complete(receipt);
        }
        for (&target, color) in targets.iter().zip(expected) {
            assert!(
                pixels(&session, target)
                    .chunks_exact(4)
                    .all(|pixel| pixel == color)
            );
        }
    }

    #[cfg_attr(test, test)]
    fn dropped_receipts_do_not_cancel_work_and_last_receipt_outlives_every_owner() {
        let table = Rc::new(ResourceTable::new());
        let target = target(&table, 32, 32);
        let mut session = session(&table, &[target]);
        let commands = clear(&table, target, [0.0, 1.0, 0.0, 1.0]);
        for _ in 0..32 {
            drop(submit(&mut session, &commands));
        }
        let checkpoint = CommandEncoder::new(&table)
            .finish()
            .expect("empty checkpoint");
        let receipt = submit(&mut session, &checkpoint);
        drop(checkpoint);
        drop(commands);
        drop(session);
        drop(table);
        complete(&receipt);
    }

    #[cfg_attr(test, test)]
    fn rejection_leaves_the_session_usable_and_legacy_execute_orders_after_async() {
        let table = Rc::new(ResourceTable::new());
        let target = target(&table, 32, 32);
        let mut session = session(&table, &[target]);
        let foreign = ResourceTable::new();
        let invalid = CommandEncoder::new(&foreign)
            .finish()
            .expect("foreign commands");
        assert!(matches!(
            session.executor().submit(&invalid),
            Err(SubmitError::Rejected(_))
        ));
        let red = submit(&mut session, &clear(&table, target, [1.0, 0.0, 0.0, 1.0]));
        session
            .executor()
            .execute(&clear(&table, target, [0.0, 1.0, 0.0, 1.0]))
            .expect("legacy execute");
        complete(&red);
        assert!(
            pixels(&session, target)
                .chunks_exact(4)
                .all(|pixel| pixel == [0, 255, 0, 255])
        );
    }

    #[cfg_attr(test, test)]
    fn persistent_vertex_upload_is_ordered_and_owned_by_the_queue() {
        let table = Rc::new(ResourceTable::new());
        let target = target(&table, 32, 32);
        let pipeline = table
            .define_render_pipeline(
                RenderPipelineDesc::new(
                    TextureFormat::Bgra8Unorm,
                    PrimitiveTopology::TriangleList,
                    VertexBufferLayout::new(
                        40,
                        vec![
                            VertexAttribute::new(0, VertexFormat::Float32x4, 0),
                            VertexAttribute::new(1, VertexFormat::Float32x4, 16),
                            VertexAttribute::new(2, VertexFormat::Float32x2, 32),
                        ],
                    )
                    .expect("canonical layout"),
                    FragmentProgram::VertexColor,
                    BlendState::REPLACE,
                    RasterState::new(CullMode::None, FrontFace::CounterClockwise),
                )
                .expect("pipeline descriptor"),
            )
            .expect("pipeline");
        let buffer = table
            .define_buffer(
                BufferDesc::new(120, BufferUsage::VERTEX | BufferUsage::COPY_DST)
                    .expect("buffer descriptor"),
            )
            .expect("buffer");
        let mut session = session(&table, &[target]);
        let mut receipts = Vec::new();
        for color in [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]] {
            let mut bytes = Vec::new();
            for position in [
                [-1.0_f32, -1.0, 0.0, 1.0],
                [3.0, -1.0, 0.0, 1.0],
                [-1.0, 3.0, 0.0, 1.0],
            ] {
                for component in position.into_iter().chain(color).chain([0.0, 0.0]) {
                    bytes.extend_from_slice(&component.to_le_bytes());
                }
            }
            let mut encoder = CommandEncoder::new(&table);
            encoder
                .write_buffer(buffer, 0, &bytes)
                .expect("upload vertices");
            let mut pass = encoder
                .begin_render_pass(
                    RenderPassDesc::new(
                        &table,
                        table.texture_ref(target).expect("target"),
                        PixelRect::new(0, 0, 32, 32).expect("area"),
                        LoadOp::Clear(Color::rgba(0.0, 0.0, 0.0, 1.0).expect("black")),
                        StoreOp::Store,
                    )
                    .expect("pass"),
                )
                .expect("begin");
            pass.set_pipeline(pipeline).expect("bind pipeline");
            pass.set_vertex_buffer(buffer, 0).expect("bind vertices");
            pass.set_uniforms(DrawUniforms::new(
                Transform::identity(),
                Color::rgba(1.0, 1.0, 1.0, 1.0).expect("white"),
            ))
            .expect("uniforms");
            pass.draw(3, 0).expect("triangle");
            pass.end().expect("end");
            let commands = encoder.finish().expect("commands");
            receipts.push(submit(&mut session, &commands));
            drop(commands);
            bytes.fill(0);
        }
        for receipt in &receipts {
            complete(receipt);
        }
        assert!(
            pixels(&session, target)
                .chunks_exact(4)
                .all(|pixel| pixel == [0, 255, 0, 255])
        );
    }

    #[cfg(not(test))]
    pub(super) fn run() {
        println!("[sgfx-native-completion-smoke] starting");
        split_upload_and_copy_snapshot_borrowed_bytes();
        println!("[sgfx-native-completion-smoke] split upload + copy + owned bytes PASS");
        oversized_stream_is_rejected_before_initializing_gpu_state();
        println!(
            "[sgfx-native-completion-smoke] oversized rejection + initialization rollback PASS"
        );
        several_submissions_share_a_queue_without_an_intermediate_wait();
        println!("[sgfx-native-completion-smoke] multiple ordered submissions PASS");
        persistent_vertex_upload_is_ordered_and_owned_by_the_queue();
        println!("[sgfx-native-completion-smoke] persistent vertex upload + drawing PASS");
        reused_vertex_storage_preserves_every_earlier_draw();
        println!("[sgfx-native-completion-smoke] reused vertex storage + every earlier draw PASS");
        dropped_receipts_do_not_cancel_work_and_last_receipt_outlives_every_owner();
        println!("[sgfx-native-completion-smoke] dropped receipts + closed owners PASS");
        rejection_leaves_the_session_usable_and_legacy_execute_orders_after_async();
        println!("[sgfx-native-completion-smoke] rejection + legacy ordering PASS");
        println!("[sgfx-native-completion-smoke] ALL PASS");
    }

    #[cfg_attr(test, test)]
    fn reused_vertex_storage_preserves_every_earlier_draw() {
        const ROWS: u32 = 32;
        const WIDTH: u32 = 32;
        const STRIP_HEIGHT: u32 = 3;
        let color = |row: u32, round: u32| {
            let bits = (row + round) % 7 + 1;
            [
                f32::from((bits & 1 != 0) as u8),
                f32::from((bits & 2 != 0) as u8),
                f32::from((bits & 4 != 0) as u8),
                1.0,
            ]
        };
        for persistent in [false, true] {
            let table = Rc::new(ResourceTable::new());
            let output = target(&table, WIDTH, ROWS * STRIP_HEIGHT);
            let stride = if persistent { 40 } else { 24 };
            let attributes = if persistent {
                vec![
                    VertexAttribute::new(0, VertexFormat::Float32x4, 0),
                    VertexAttribute::new(1, VertexFormat::Float32x4, 16),
                    VertexAttribute::new(2, VertexFormat::Float32x2, 32),
                ]
            } else {
                // Deliberately noncanonical: every pass lowers through the
                // shared inline scratch buffer used by noncanonical UI draws.
                vec![
                    VertexAttribute::new(0, VertexFormat::Float32x2, 0),
                    VertexAttribute::new(1, VertexFormat::Float32x4, 8),
                ]
            };
            let pipeline = table
                .define_render_pipeline(
                    RenderPipelineDesc::new(
                        TextureFormat::Bgra8Unorm,
                        PrimitiveTopology::TriangleList,
                        VertexBufferLayout::new(stride, attributes).expect("layout"),
                        FragmentProgram::VertexColor,
                        BlendState::REPLACE,
                        RasterState::new(CullMode::None, FrontFace::CounterClockwise),
                    )
                    .expect("pipeline"),
                )
                .expect("define pipeline");
            let buffer = table
                .define_buffer(
                    BufferDesc::new(
                        u64::from(ROWS * 3 * stride),
                        BufferUsage::VERTEX | BufferUsage::COPY_DST,
                    )
                    .expect("buffer"),
                )
                .expect("define buffer");
            let mut session = session(&table, &[output]);
            let full = PixelRect::new(0, 0, WIDTH, ROWS * STRIP_HEIGHT).expect("full area");
            for round in 0..8 {
                let mut bytes = Vec::new();
                for row in 0..ROWS {
                    for position in [[-1.0_f32, -1.0], [3.0, -1.0], [-1.0, 3.0]] {
                        for value in position {
                            bytes.extend_from_slice(&value.to_le_bytes());
                        }
                        if persistent {
                            for value in [0.0_f32, 1.0] {
                                bytes.extend_from_slice(&value.to_le_bytes());
                            }
                        }
                        for value in color(row, round) {
                            bytes.extend_from_slice(&value.to_le_bytes());
                        }
                        if persistent {
                            bytes.extend_from_slice(&[0; 8]);
                        }
                    }
                }
                let mut encoder = CommandEncoder::new(&table);
                if !persistent {
                    encoder
                        .write_buffer(buffer, 0, &bytes)
                        .expect("all inline vertices");
                }
                for row in 0..ROWS {
                    if persistent {
                        let start = (row * 3 * stride) as usize;
                        // Overwrite the same persistent buffer for each queued
                        // strip, without waiting after the previous draw.
                        encoder
                            .write_buffer(buffer, 0, &bytes[start..start + (3 * stride) as usize])
                            .expect("persistent vertex update");
                    }
                    let mut pass = encoder
                        .begin_render_pass(
                            RenderPassDesc::new(
                                &table,
                                table.texture_ref(output).expect("output"),
                                full,
                                if row == 0 {
                                    LoadOp::Clear(Color::rgba(0.0, 0.0, 0.0, 1.0).expect("black"))
                                } else {
                                    LoadOp::Load
                                },
                                StoreOp::Store,
                            )
                            .expect("pass"),
                        )
                        .expect("begin pass");
                    pass.set_pipeline(pipeline).expect("pipeline");
                    pass.set_vertex_buffer(buffer, 0).expect("vertices");
                    pass.set_uniforms(DrawUniforms::new(
                        Transform::identity(),
                        Color::rgba(1.0, 1.0, 1.0, 1.0).expect("white"),
                    ))
                    .expect("uniforms");
                    pass.set_scissor(Some(
                        PixelRect::new(0, row * STRIP_HEIGHT, WIDTH, STRIP_HEIGHT).expect("strip"),
                    ))
                    .expect("scissor");
                    pass.draw(3, if persistent { 0 } else { row * 3 })
                        .expect("draw");
                    pass.end().expect("end pass");
                    if persistent {
                        // Dropping the receipt must not recycle its upload arena.
                        drop(submit(
                            &mut session,
                            &encoder.finish().expect("strip commands"),
                        ));
                        encoder = CommandEncoder::new(&table);
                    }
                }
                // Inline mode puts all 32 scratch-buffer writes and draws in
                // ONE native admission. A final-only color check cannot catch
                // earlier draws reading the last upload, so verify every pixel.
                complete(&submit(
                    &mut session,
                    &encoder.finish().expect("frame checkpoint"),
                ));
                for (index, pixel) in pixels(&session, output).chunks_exact(4).enumerate() {
                    let row = index as u32 / WIDTH / STRIP_HEIGHT;
                    let [r, g, b, a] = color(row, round);
                    assert_eq!(
                        pixel,
                        [
                            (b * 255.0) as u8,
                            (g * 255.0) as u8,
                            (r * 255.0) as u8,
                            (a * 255.0) as u8
                        ],
                        "vertex reuse: persistent={persistent}, round={round}, row={row}"
                    );
                }
                bytes.fill(0);
            }
        }
    }
}

#[cfg(not(test))]
fn main() {
    #[cfg(target_os = "scarlet")]
    native::run();
    #[cfg(not(target_os = "scarlet"))]
    {
        eprintln!("sgfx-native-completion-smoke requires a Scarlet/VirGL guest");
        std::process::exit(1);
    }
}
