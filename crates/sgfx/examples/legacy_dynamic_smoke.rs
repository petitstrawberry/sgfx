//! Exercise the installed backend from Scarlet's legacy no_std runtime.
#![no_std]
#![no_main]
extern crate scarlet_std as std;
use sgfx::{
    backend::{CommandSubmitter, Completion, CompletionStatus},
    ir::*,
};
use std::{println, rc::Rc, vec};

#[unsafe(no_mangle)]
fn main() -> i32 {
    let device = sgfx::Device::open("/dev/gpu0").unwrap();
    assert!(device.backend_library().is_some());
    println!(
        "SGFX_LEGACY_DRIVER library={}",
        device.backend_library().unwrap()
    );
    let context = device.create_context().unwrap();
    let table = Rc::new(ResourceTable::new());
    let target = table
        .define_texture(
            TextureDesc::new(
                TextureFormat::Bgra8Unorm,
                Extent2D::new(16, 16).unwrap(),
                TextureUsage::RENDER_ATTACHMENT | TextureUsage::PRESENT | TextureUsage::COPY_SRC,
            )
            .unwrap(),
        )
        .unwrap()
        .id();
    let mut session = context
        .create_mapped_target_session(table.clone(), &[target])
        .unwrap();
    let mut encoder = CommandEncoder::new(&table);
    encoder
        .begin_render_pass(
            RenderPassDesc::new(
                &table,
                table.texture_ref(target).unwrap(),
                PixelRect::new(0, 0, 16, 16).unwrap(),
                LoadOp::Clear(Color::rgba(0.0, 0.0, 1.0, 1.0).unwrap()),
                StoreOp::Store,
            )
            .unwrap(),
        )
        .unwrap()
        .end()
        .unwrap();
    let commands = encoder.finish().unwrap();
    let receipt = session.executor().submit(&commands).unwrap();
    assert_eq!(receipt.wait(None).unwrap(), CompletionStatus::Complete);
    let mut pixels = vec![0; 16 * 16 * 4];
    session
        .readback_bgra(
            target,
            &mut pixels,
            16 * 4,
            PixelRect::new(0, 0, 16, 16).unwrap(),
        )
        .unwrap();
    assert!(pixels.chunks_exact(4).all(|p| p == [255, 0, 0, 255]));
    drop(commands);
    drop(session);
    drop(context);
    drop(device);
    std::thread::spawn(move || assert_eq!(receipt.wait(None).unwrap(), CompletionStatus::Complete))
        .join()
        .unwrap();
    println!("SGFX_LEGACY_OK");
    0
}
