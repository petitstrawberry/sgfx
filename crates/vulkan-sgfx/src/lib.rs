//! Experimental, non-conformant Vulkan frontend and ICD for SGFX.
//!
//! This crate implements a deliberately bounded development subset. It is not
//! a Vulkan-conformant implementation. See `docs/vulkan-sgfx.md` for the supported
//! contracts, device selection, and explicit limitations.
#![allow(unsafe_op_in_unsafe_fn)]

mod api;
#[cfg(all(target_os = "linux", feature = "scarlet-wsi"))]
mod display;
mod images;
mod instance;
mod push_constants;
mod resources;
mod runtime;
#[cfg(target_os = "scarlet")]
pub mod scarlet_image;
mod spirv;
mod transfer;
#[cfg(any(target_os = "macos", all(target_os = "linux", feature = "scarlet-wsi")))]
mod wsi;

// Keep the Linux-ABI ICD's short-lived command and draw allocations in a
// reusable heap. Passing these buffers through musl on every submit dominated
// CPU time on Scarlet, including cleanup after native queue submission.
// This allocator belongs to the Rust driver; the Vulkan application's libc
// allocation entry points and the standard loader are unchanged.
#[cfg(all(target_os = "linux", feature = "scarlet-wsi"))]
#[global_allocator]
static ICD_ALLOCATOR: dlmalloc::GlobalDlmalloc = dlmalloc::GlobalDlmalloc;

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
