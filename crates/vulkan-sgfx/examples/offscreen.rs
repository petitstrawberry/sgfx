//! Draw and read back a headless Vulkan triangle through an ICD or Vulkan loader.
//!
//! Build first: `cargo build -p vulkan-sgfx`.
//! Run: `cargo run -p vulkan-sgfx --example offscreen -- [ICD_LIBRARY]`.
//! `SGFX_ICD_LIBRARY` can also select the library. No surface is created.
//! To use a Vulkan loader instead, set `SGFX_VULKAN_LOADER=/path/to/libvulkan`
//! and `VK_DRIVER_FILES=/path/to/sgfx_icd.json`; the loader setting takes priority.

use ash::{Entry, vk};
use std::{error::Error, path::PathBuf};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;
#[path = "support/offscreen.rs"]
mod render;

fn default_library() -> Result<PathBuf, Box<dyn Error>> {
    let executable = std::env::current_exe()?;
    let profile_dir = executable
        .parent()
        .and_then(|examples| examples.parent())
        .ok_or("cannot locate the Cargo profile directory")?;
    Ok(profile_dir.join(format!(
        "{}vulkan_sgfx{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX,
    )))
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let icd_path = args.next().or_else(|| std::env::var_os("SGFX_ICD_LIBRARY"));
    if args.next().is_some() {
        return Err("usage: offscreen [ICD_LIBRARY]".into());
    }
    let loader_path = std::env::var_os("SGFX_VULKAN_LOADER");
    let using_loader = loader_path.is_some();
    let library_path = loader_path
        .or(icd_path)
        .map(PathBuf::from)
        .map(Ok)
        .unwrap_or_else(default_library)?;
    let vertex_words = render::shader_words(
        include_str!("../tests/assets/triangle.wgsl"),
        naga::ShaderStage::Vertex,
        "vs_main",
    )?;
    let fragment_words = render::shader_words(
        include_str!("../tests/assets/triangle.wgsl"),
        naga::ShaderStage::Fragment,
        "fs_main",
    )?;

    // Keep the library alive until all handles and dispatch tables are gone.
    let library = unsafe { libloading::Library::new(&library_path)? };
    let entrypoint: &[u8] = if using_loader {
        b"vkGetInstanceProcAddr\0"
    } else {
        b"vk_icdGetInstanceProcAddr\0"
    };
    let get_instance_proc_addr =
        unsafe { *library.get::<vk::PFN_vkGetInstanceProcAddr>(entrypoint)? };
    let entry = unsafe {
        Entry::from_static_fn(ash::StaticFn {
            get_instance_proc_addr,
        })
    };
    let pixels = render::render(
        &entry,
        &vertex_words,
        &fragment_words,
        [WIDTH, HEIGHT],
        [0.0, 0.0, 1.0, 1.0],
    )?;
    validate_pixels(&pixels)?;
    let library_kind = if using_loader { "Vulkan loader" } else { "ICD" };
    println!("{library_kind}: {}", library_path.display());
    Ok(())
}

fn validate_pixels(pixels: &[u8]) -> Result<(), Box<dyn Error>> {
    // Check interior/exterior patches and every corner. The asymmetric
    // upper/lower patches also catch incorrect SPIR-V Y normalization.
    let mut mismatch = None;
    for (name, start_x, start_y, expected) in [
        ("triangle", WIDTH / 2 - 2, HEIGHT / 2 - 2, [255, 0, 0, 255]),
        ("top-left clear", 1, 1, [0, 0, 255, 255]),
        ("top-right clear", WIDTH - 5, 1, [0, 0, 255, 255]),
        ("bottom-left clear", 1, HEIGHT - 5, [0, 0, 255, 255]),
        (
            "bottom-right clear",
            WIDTH - 5,
            HEIGHT - 5,
            [0, 0, 255, 255],
        ),
        ("outside upper apex", 42, 14, [0, 0, 255, 255]),
        ("inside lower base", 42, 46, [255, 0, 0, 255]),
    ] {
        for y in start_y..start_y + 4 {
            for x in start_x..start_x + 4 {
                let offset = ((y * WIDTH + x) * 4) as usize;
                let observed = &pixels[offset..offset + 4];
                if observed != expected && mismatch.is_none() {
                    mismatch = Some(format!(
                        "{name} pixel ({x}, {y}): got {observed:?}, expected {expected:?}",
                    ));
                }
            }
        }
    }
    if let Some(mismatch) = mismatch {
        return Err(format!("offscreen readback mismatch: {mismatch}").into());
    }
    println!(
        "PASS: {WIDTH}x{HEIGHT} SPIR-V triangle; 112 center, corner, and asymmetric orientation pixels verified through image-to-buffer readback"
    );
    Ok(())
}
