//! Experimental, non-conformant Vulkan frontend and ICD for SGFX.
//!
//! This crate implements a deliberately bounded development subset. It is not
//! a Vulkan-conformant implementation. See `docs/vulkan-sgfx.md` for the supported
//! contracts, device selection, and explicit limitations.
#![allow(unsafe_op_in_unsafe_fn)]

mod api;
mod images;
mod instance;
mod resources;
mod runtime;
#[cfg(target_os = "scarlet")]
pub mod scarlet_image;
#[cfg(target_os = "macos")]
mod wsi;

/// Construct a Vulkan entry table for the ICD linked into this executable.
///
/// This supports native Scarlet applications before a dynamic Vulkan loader is
/// available. It dispatches the same Vulkan ABI as the exported ICD, but does
/// not discover other drivers or exercise a separate Khronos loader.
pub fn linked_entry() -> ash::Entry {
    // The entry point and all functions it returns live in this executable for
    // the full lifetime of the Entry and its Vulkan objects.
    unsafe {
        ash::Entry::from_static_fn(ash::StaticFn {
            get_instance_proc_addr: instance::vk_icdGetInstanceProcAddr,
        })
    }
}
