//! Scarlet GPU-image sharing exposed as a Vulkan device extension.
//!
//! Rendering remains ordinary Vulkan. This extension only transfers ownership
//! of a duplicated Scarlet GPU image capability to another SGFX context, such
//! as ScarletUI's compositor renderer. The application must call it before the
//! image is first submitted and must synchronize later writes with consumers.

use ash::vk;
use std::ffi::CStr;

use crate::api::with_device;

/// Experimental device extension used for Scarlet SGFX image interop.
pub const DEVICE_EXTENSION_NAME: &CStr = c"VK_SGFX_scarlet_image";

/// Function signature returned for `vkGetImageScarletHandleSGFX`.
#[allow(non_camel_case_types)]
pub type PFN_vkGetImageScarletHandleSGFX =
    unsafe extern "system" fn(device: vk::Device, image: vk::Image, handle: *mut i32) -> vk::Result;

/// Owned exported image capability and immutable image metadata.
#[derive(Debug)]
pub struct ExportedImage {
    handle: sgfx::Handle,
    width: u32,
    height: u32,
}

impl ExportedImage {
    /// Consume the export and transfer its Scarlet handle to an importer.
    pub fn into_handle(self) -> sgfx::Handle {
        self.handle
    }

    /// Image width in physical pixels.
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Image height in physical pixels.
    pub const fn height(&self) -> u32 {
        self.height
    }
}

/// Map a bound BGRA8 Vulkan image to a shareable SGFX image and duplicate its
/// Scarlet capability.
///
/// Call this after `vkBindImageMemory` and before submitting any command that
/// references the image. Repeated calls return independent handle ownership for
/// the same physical image. The device must have no outstanding submissions.
pub fn export_image(device: vk::Device, image: vk::Image) -> Result<ExportedImage, vk::Result> {
    with_device(device, move |runtime| {
        if !runtime.in_flight.is_empty() {
            return Err(vk::Result::NOT_READY);
        }
        let (id, width, height, needs_mapping) = {
            let data = runtime
                .resources
                .images
                .get(&image)
                .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
            if data.swapchain.is_some()
                || data.bound.is_none()
                || data.format != vk::Format::B8G8R8A8_UNORM
                || !data.usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            {
                return Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED);
            }
            (
                data.id,
                data.extent.width,
                data.extent.height,
                data.shared.is_none(),
            )
        };
        if needs_mapping {
            let physical = runtime
                .device
                .create_presentation_image(width, height, sgfx::ir::TextureFormat::Bgra8Unorm)
                .map_err(crate::runtime::backend_failure)?;
            runtime
                .cache
                .map_presentation_image(id, &physical)
                .map_err(crate::runtime::backend_failure)?;
            runtime
                .resources
                .images
                .get_mut(&image)
                .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?
                .shared = Some(physical);
        }
        let shared = runtime
            .resources
            .images
            .get(&image)
            .and_then(|image| image.shared.as_ref())
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let handle = shared
            .duplicate_shared_handle()
            .map_err(crate::runtime::backend_failure)?;
        Ok(ExportedImage {
            handle,
            width,
            height,
        })
    })
}

/// Return an owning raw Scarlet GPU image capability for a Vulkan image.
///
/// The caller must adopt the returned handle exactly once and eventually close
/// it. This is the C ABI entry point advertised by `VK_SGFX_scarlet_image`.
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "system" fn vkGetImageScarletHandleSGFX(
    device: vk::Device,
    image: vk::Image,
    output: *mut i32,
) -> vk::Result {
    if output.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    unsafe { *output = -1 };
    match export_image(device, image) {
        Ok(exported) => {
            let handle = exported.into_handle();
            let raw = handle.as_raw();
            std::mem::forget(handle);
            unsafe { *output = raw };
            vk::Result::SUCCESS
        }
        Err(error) => error,
    }
}
