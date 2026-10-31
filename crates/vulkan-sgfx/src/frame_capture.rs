//! Opt-in GPU frame readback for presentation diagnostics.
use ash::vk;
use sgfx::{driver, ir};
use std::{path::PathBuf, sync::OnceLock};

pub(crate) fn enabled() -> bool {
    directory().is_some()
}
fn directory() -> &'static Option<PathBuf> {
    static DIRECTORY: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIRECTORY.get_or_init(|| {
        crate::api::process_env("SGFX_VULKAN_CAPTURE_DIR")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    })
}
#[derive(Default)]
pub(crate) struct Capture {
    frames: u64,
    captures: u8,
    last: Option<std::time::Instant>,
}
impl Capture {
    pub(crate) fn presented(
        &mut self,
        queue: &driver::Queue,
        cache: &mut driver::Resources,
        image: ir::TextureId,
        format: vk::Format,
        extent: vk::Extent2D,
        chain: u64,
    ) {
        let Some(directory) = directory() else {
            return;
        };
        let frame = self.frames;
        self.frames = self.frames.saturating_add(1);
        if self.captures >= 12
            || self
                .last
                .is_some_and(|last| last.elapsed() < std::time::Duration::from_secs(30))
        {
            return;
        }
        self.captures += 1;
        self.last = Some(std::time::Instant::now());
        // The caller has retired rendering and waited on presentation's
        // semaphores. Capture the actual GPU image before handing it to WSI.
        let result = (|| -> Result<(), Box<dyn std::error::Error>> {
            let mut bytes = queue
                .read_texture(cache, image)
                .map_err(|e| format!("GPU readback: {e:?}"))?;
            let length = (extent.width as usize)
                .checked_mul(extent.height as usize)
                .and_then(|n| n.checked_mul(4))
                .ok_or("capture extent overflow")?;
            if bytes.len() != length {
                return Err("capture byte layout mismatch".into());
            }
            match format {
                vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB => {
                    for pixel in bytes.chunks_exact_mut(4) {
                        pixel.swap(0, 2);
                    }
                }
                vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB => {}
                _ => return Err("capture format unsupported".into()),
            }
            std::fs::create_dir_all(directory)?;
            let path = directory.join(format!("swapchain-{chain:016x}-frame-{frame:06}.png"));
            let file = std::fs::File::create_new(&path)?;
            let mut encoder = png::Encoder::new(file, extent.width, extent.height);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header()?;
            writer.write_image_data(&bytes)?;
            writer.finish()?;
            eprintln!(
                "[SGFX Vulkan] frame capture: frame={frame} size={}x{} path={}",
                extent.width,
                extent.height,
                path.display()
            );
            Ok(())
        })();
        if let Err(error) = result {
            eprintln!("[SGFX Vulkan] frame capture failed: {error}");
        }
    }
}
