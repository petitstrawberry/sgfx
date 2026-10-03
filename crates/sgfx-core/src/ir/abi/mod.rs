//! Canonical in-process SGFX command storage and cold-path resource exchange.
//!
//! ABI commands are written once, at recording time. Upload operands contain
//! the original borrowed address and length. A backend walks the same words
//! directly; it does not build a second command buffer. Resource descriptors
//! are synchronized only when a table changes, outside steady-state submit.

use alloc::vec::Vec;
use core::cell::{Cell, RefCell};

use super::*;

mod codec;
pub(crate) mod commands;
mod resources;

pub use commands::{CommandIter, CommandReader};
pub use sgfx_backend_abi::{Batch, Span};

pub(crate) struct State {
    pub enabled: Cell<bool>,
    pub revision: Cell<u64>,
    pub snapshot: RefCell<(u64, Vec<u64>)>,
}

impl State {
    pub const fn new() -> Self {
        Self {
            enabled: Cell::new(false),
            revision: Cell::new(1),
            snapshot: RefCell::new((0, Vec::new())),
        }
    }
}

impl ResourceTable {
    /// Record subsequent command encoders directly into the backend ABI.
    /// Existing Rust command recordings remain usable by static executors.
    pub fn enable_abi_commands(&self) {
        self.abi.enabled.set(true);
    }

    /// Identity carried by batches; meaningful only in this process.
    pub fn abi_identity(&self) -> u64 {
        self.id as u64
    }

    /// Changes only when logical resources are defined or retired.
    pub fn abi_revision(&self) -> u64 {
        self.abi.revision.get()
    }

    pub(crate) fn abi_changed(&self) {
        self.abi.revision.set(
            self.abi
                .revision
                .get()
                .checked_add(1)
                .expect("resource revision exhausted"),
        );
    }

    /// Resolve a texture slot in a backend's mirrored resource table.
    pub fn abi_texture(&self, slot: u32) -> Result<TextureRef<'_>> {
        let generation = self
            .textures
            .borrow()
            .get(slot as usize)
            .ok_or(Error::InvalidDescriptor)?
            .generation;
        self.texture_ref(TextureId {
            owner: self.id,
            index: slot as usize,
            generation,
        })
    }
}

impl ResourceTable {
    /// Resolve the current generation of a mirrored buffer slot. The host must
    /// validate the original branded ID before passing a slot across the ABI.
    pub fn abi_buffer(&self, slot: u32) -> Result<BufferRef<'_>> {
        let generation = self
            .buffers
            .borrow()
            .get(slot as usize)
            .ok_or(Error::InvalidDescriptor)?
            .generation;
        self.buffer_ref(BufferId {
            owner: self.id,
            index: slot as usize,
            generation,
        })
    }
    /// Resolve the current generation of a mirrored bind-group slot.
    pub fn abi_bind_group(&self, slot: u32) -> Result<BindGroupRef<'_>> {
        let generation = self
            .bind_groups
            .borrow()
            .get(slot as usize)
            .ok_or(Error::InvalidDescriptor)?
            .generation;
        self.bind_group_ref(BindGroupId {
            owner: self.id,
            index: slot as usize,
            generation,
        })
    }
    /// Resolve an immutable definition in a backend-owned mirrored table.
    pub fn abi_shader_module(&self, slot: u32) -> Result<ShaderModuleRef<'_>> {
        self.shader_module_ref(ShaderModuleId {
            owner: self.id,
            index: slot as usize,
        })
    }
    /// Resolve an immutable definition in a backend-owned mirrored table.
    pub fn abi_programmable_render_pipeline(
        &self,
        slot: u32,
    ) -> Result<ProgrammableRenderPipelineRef<'_>> {
        self.programmable_render_pipeline_ref(ProgrammableRenderPipelineId {
            owner: self.id,
            index: slot as usize,
        })
    }
    /// Resolve an immutable definition in a backend-owned mirrored table.
    pub fn abi_compute_pipeline(&self, slot: u32) -> Result<ComputePipelineRef<'_>> {
        self.compute_pipeline_ref(ComputePipelineId {
            owner: self.id,
            index: slot as usize,
        })
    }
}
