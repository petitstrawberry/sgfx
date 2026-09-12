//! Experimental, non-conformant headless Vulkan frontend for SGFX.
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
