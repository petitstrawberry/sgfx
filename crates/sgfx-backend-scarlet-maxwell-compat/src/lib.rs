//! Static Maxwell compatibility API for Scarlet ELF32 clients.
#![no_std]

#[cfg(all(target_os = "scarlet", target_pointer_width = "32"))]
pub use sgfx_backend_scarlet_maxwell::*;
