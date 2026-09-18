//! Render an indexed Vulkan cube with D32 depth, optionally textured.
//! Host builds use the installed Vulkan loader; driver selection is external.
//! Scarlet currently packages Vulkan entry points statically.
#[cfg(not(target_os = "scarlet"))]
use ash::Entry;
use std::{error::Error, fs::File, io::BufWriter, path::PathBuf};

#[path = "../../examples/support/cube.rs"]
mod render;

fn main() -> Result<(), Box<dyn Error>> {
    let mut output = None;
    let mut display_requested = false;
    let mut frames = 1usize;
    let mut verify = false;
    let mut options = render::Options::default();
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            "--linked" if cfg!(target_os = "scarlet") => {}
            "--textured" => options.textured = true,
            "--dynamic-viewport" => options.dynamic_viewport = true,
            "--dynamic-uniform" => options.dynamic_uniform = true,
            "--push-constants" => options.push_constants = true,
            "--verify" => verify = true,
            "--display" => display_requested = true,
            "--output" => {
                output = Some(PathBuf::from(
                    arguments.next().ok_or("--output requires a path")?,
                ))
            }
            "--frames" => {
                frames = arguments
                    .next()
                    .ok_or("--frames requires a count")?
                    .parse()?
            }
            "--angle" => {
                options.angle = arguments
                    .next()
                    .ok_or("--angle requires radians")?
                    .parse()?
            }
            _ if !argument.starts_with('-') && output.is_none() => {
                output = Some(PathBuf::from(argument))
            }
            _ => {
                return Err(format!("unknown argument {argument:?}; use --help for usage").into());
            }
        }
    }
    if output.is_none() && !display_requested {
        return Err("specify --output PATH or --display; use --help for usage".into());
    }
    if frames == 0 || frames > 3600 {
        return Err("--frames must be between 1 and 3600".into());
    }
    if !options.angle.is_finite() {
        return Err("--angle must be finite".into());
    }
    #[cfg(not(target_os = "scarlet"))]
    if display_requested {
        return Err("--display requires Scarlet; use --output PATH for host PNG rendering".into());
    }
    #[cfg(target_os = "scarlet")]
    let mut display = if display_requested {
        Some(CubeDisplay::open(&mut options)?)
    } else {
        None
    };
    #[cfg(target_os = "scarlet")]
    let entry = vulkan_sgfx::linked_entry();
    #[cfg(not(target_os = "scarlet"))]
    let entry = unsafe { Entry::load()? };
    if verify {
        render::verify(&entry)?;
        if options.textured {
            render::verify_textured(&entry)?;
        }
        if options.push_constants {
            render::verify_push_constants(&entry)?;
        }
    }
    for frame in 0..frames {
        let frame_options = render::Options {
            angle: options.angle + frame as f32 * 0.045,
            ..options
        };
        let pixels = render::render(&entry, frame_options)?;
        render::validate_image(&pixels, options.size)?;
        #[cfg(target_os = "scarlet")]
        if let Some(display) = display.as_mut() {
            display.present(&pixels, options.size)?;
            println!(
                "PASS: cube frame {}/{} presented by GPU readback + DisplaySurface ({}x{})",
                frame + 1,
                frames,
                options.size[0],
                options.size[1]
            );
        }
        if let Some(output) = &output {
            let output = if frames == 1 {
                output.clone()
            } else {
                let stem = output
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .ok_or("output needs a valid UTF-8 filename")?;
                output.with_file_name(format!("{stem}-{frame:04}.png"))
            };
            write_png(&output, &pixels, options.size)?;
            println!(
                "PASS: indexed cube, 24 vertices / 36 indices, MVP transform, D32 LESS depth; {}x{} PNG: {}",
                options.size[0],
                options.size[1],
                output.canonicalize()?.display()
            );
        }
    }
    Ok(())
}

fn print_help() {
    println!(
        "Usage: vulkan-cube [OUTPUT.png | --output PATH] [--display] [OPTIONS]\n\
         \n\
         Render an indexed, depth-tested Vulkan cube.\n\
         \n\
         --output PATH     Write RGBA GPU readback to PNG.\n\
         --display         Present GPU readback through Scarlet DisplaySurface.\n\
                           Run with SWS stopped; this is direct display output,\n\
                           not a Vulkan swapchain. Host builds do not support it.\n\
                           Centers a region up to 1024x768 within the display.\n\
         --frames COUNT    Render 1..3600 frames, rotating 0.045 radians per frame.\n\
                           PNG sequences use NAME-0000.png, NAME-0001.png, ...\n\
         --angle RADIANS   Set the initial rotation (default: 0.58).\n\
         --textured        Sample a staged checkerboard texture.\n\
         --dynamic-viewport Use Vulkan dynamic viewport and scissor.\n\
         --dynamic-uniform Use a dynamic uniform descriptor with offset 256.\n\
         --push-constants  Use Vulkan push constants for the MVP transform.\n\
         --linked          Scarlet compatibility option (already automatic).\n\
         --verify          Check indexed rendering and depth before rendering.\n\
         --help, -h        Show this help.\n\
         \n\
         --display and --output can be combined. PNG-only output is 512x512.\n\
         Host driver selection: VK_DRIVER_FILES points to an ICD manifest."
    );
}

fn write_png(
    output: &std::path::Path,
    pixels: &[u8],
    size: [u32; 2],
) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut encoder = png::Encoder::new(BufWriter::new(File::create(output)?), size[0], size[1]);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    encoder.add_text_chunk(
        "Software".into(),
        "vulkan-sgfx indexed cube / GPU readback".into(),
    )?;
    let mut writer = encoder.write_header()?;
    writer.write_image_data(pixels)?;
    writer.finish()?;
    Ok(())
}

#[cfg(target_os = "scarlet")]
struct CubeDisplay {
    surface: framebuffer::DisplaySurface,
    origin: [u32; 2],
    bgra: Vec<u8>,
}

#[cfg(target_os = "scarlet")]
impl CubeDisplay {
    fn open(options: &mut render::Options) -> Result<Self, Box<dyn Error>> {
        let surface = framebuffer::DisplaySurface::open_primary()
            .map_err(|error| format!("cannot open primary display: {error:?}"))?;
        let info = surface
            .get_info()
            .map_err(|error| format!("cannot query primary display: {error:?}"))?;
        if info.width == 0 || info.height == 0 {
            return Err("primary display has an empty extent".into());
        }
        // Keep the readback reasonably sized, within the ICD's 2048 limit, and
        // entirely inside the display. Do not silently crop a larger render.
        options.size = [info.width.min(1024), info.height.min(768)];
        let origin = [
            (info.width - options.size[0]) / 2,
            (info.height - options.size[1]) / 2,
        ];
        println!(
            "DisplaySurface: {}x{}, centered {}x{} at ({}, {}); direct display output",
            info.width, info.height, options.size[0], options.size[1], origin[0], origin[1]
        );
        Ok(Self {
            surface,
            origin,
            bgra: Vec::new(),
        })
    }

    fn present(&mut self, rgba: &[u8], size: [u32; 2]) -> Result<(), Box<dyn Error>> {
        // Preserve the original RGBA readback for PNG output and validation.
        self.bgra.clear();
        self.bgra.extend_from_slice(rgba);
        for pixel in self.bgra.chunks_exact_mut(4) {
            pixel.swap(0, 2);
        }
        self.surface
            .write_bgra_strided(
                self.origin[0],
                self.origin[1],
                size[0],
                size[1],
                &self.bgra,
                size[0] as usize * 4,
            )
            .map_err(|error| format!("cannot write cube to display: {error:?}"))?;
        self.surface
            .present()
            .map_err(|error| format!("cannot present cube display frame: {error:?}"))?;
        Ok(())
    }
}
