//! Save actual SGFX Vulkan rendering to an explicitly selected PNG path.
//!
//! `scripts/run-vulkan-demo.sh target/vulkan-demo.png` builds a fresh ICD,
//! discovers a real Khronos loader and configures its driver manifest.
use ash::{Entry, vk};
use std::{collections::HashSet, error::Error, fs::File, io::BufWriter, path::PathBuf};

#[path = "support/offscreen.rs"]
mod render;

const WIDTH: u32 = 1024;
const HEIGHT: u32 = 768;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let output = PathBuf::from(args.next().ok_or("usage: render_demo OUTPUT.png")?);
    if args.next().is_some() {
        return Err("usage: render_demo OUTPUT.png".into());
    }
    let output = if output.is_absolute() {
        output
    } else {
        std::env::current_dir()?.join(output)
    };
    let loader = PathBuf::from(std::env::var_os("SGFX_VULKAN_LOADER").ok_or(
        "SGFX_VULKAN_LOADER must select a real Vulkan loader; use scripts/run-vulkan-demo.sh",
    )?);
    let manifest = std::env::var_os("VK_DRIVER_FILES")
        .ok_or("VK_DRIVER_FILES must select the freshly built SGFX ICD manifest")?;
    let vertex = render::shader_words(
        include_str!("assets/demo.wgsl"),
        naga::ShaderStage::Vertex,
        "vs_main",
    )?;
    let fragment = render::shader_words(
        include_str!("assets/demo.wgsl"),
        naga::ShaderStage::Fragment,
        "fs_main",
    )?;

    // The loader and its dispatch tables outlive every Vulkan object below.
    let library = unsafe { libloading::Library::new(&loader)? };
    let get_instance_proc_addr =
        unsafe { *library.get::<vk::PFN_vkGetInstanceProcAddr>(b"vkGetInstanceProcAddr\0")? };
    let entry = unsafe {
        Entry::from_static_fn(ash::StaticFn {
            get_instance_proc_addr,
        })
    };
    println!("Vulkan loader: {}", loader.display());
    println!("ICD manifest: {}", PathBuf::from(manifest).display());
    let pixels = render::render(
        &entry,
        &vertex,
        &fragment,
        [WIDTH, HEIGHT],
        [0.0, 0.0, 0.0, 1.0],
    )?;
    validate_readback(&pixels)?;

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = BufWriter::new(File::create(&output)?);
    let mut encoder = png::Encoder::new(file, WIDTH, HEIGHT);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    encoder.add_text_chunk(
        "Software".into(),
        "vulkan-sgfx / actual GPU readback".into(),
    )?;
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&pixels)?;
    writer.finish()?;
    println!(
        "PASS: {WIDTH}x{HEIGHT} RGBA8, {} GPU-readback bytes saved losslessly",
        pixels.len()
    );
    println!("PNG: {}", output.canonicalize()?.display());
    Ok(())
}

fn validate_readback(pixels: &[u8]) -> Result<(), Box<dyn Error>> {
    if pixels.len() != WIDTH as usize * HEIGHT as usize * 4
        || pixels.chunks_exact(4).any(|p| p[3] != 255)
    {
        return Err("GPU readback has incorrect dimensions or non-opaque pixels".into());
    }
    let colors: HashSet<[u8; 3]> = pixels
        .chunks_exact(4)
        .step_by(31)
        .map(|p| [p[0], p[1], p[2]])
        .collect();
    if colors.len() < 256 {
        return Err("GPU output lacks the expected procedural color variation".into());
    }
    let bright = pixels
        .chunks_exact(4)
        .filter(|p| p[..3].iter().copied().max().unwrap_or(0) > 128)
        .count();
    let dark = pixels
        .chunks_exact(4)
        .filter(|p| p[..3].iter().copied().max().unwrap_or(0) < 48)
        .count();
    if bright < 10_000 || dark < 10_000 {
        return Err("GPU output lacks the expected bright triangle and dark background".into());
    }
    println!(
        "Readback checks: {} sampled colors, {bright} bright pixels, {dark} dark pixels",
        colors.len()
    );
    Ok(())
}
