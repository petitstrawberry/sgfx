//! Color outputs shared by render-pass and programmable pipeline descriptors.

use super::{BlendState, Error, LoadOp, Result, StoreOp, TextureFormat, TextureRef};

/// Maximum color outputs in one logical render pass.
pub const MAX_COLOR_ATTACHMENTS: usize = 8;

/// Channels written by a fragment output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorWriteMask(u8);
impl ColorWriteMask {
    /// Write every RGBA channel.
    pub const ALL: Self = Self(15);
    /// Disable color writes.
    pub const NONE: Self = Self(0);
    /// Select RGBA channels with bits zero through three.
    pub const fn from_bits(bits: u8) -> Result<Self> {
        if bits & !15 != 0 {
            Err(Error::InvalidValue)
        } else {
            Ok(Self(bits))
        }
    }
    /// Return the channel bits.
    pub const fn bits(self) -> u8 {
        self.0
    }
}

/// Format, blending and write mask for one fragment output location.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorTargetState {
    format: TextureFormat,
    blend: BlendState,
    write_mask: ColorWriteMask,
}
impl ColorTargetState {
    /// Create a color output; depth formats are not color outputs.
    pub const fn new(
        format: TextureFormat,
        blend: BlendState,
        write_mask: ColorWriteMask,
    ) -> Result<Self> {
        if matches!(format, TextureFormat::Depth32Float) {
            return Err(Error::InvalidDescriptor);
        }
        Ok(Self {
            format,
            blend,
            write_mask,
        })
    }
    /// Return the output format.
    pub const fn format(self) -> TextureFormat {
        self.format
    }
    /// Return the blend operation.
    pub const fn blend(self) -> BlendState {
        self.blend
    }
    /// Return the enabled channels.
    pub const fn write_mask(self) -> ColorWriteMask {
        self.write_mask
    }
}

/// One color image attached to a render pass.
#[derive(Clone, Copy)]
pub struct ColorAttachment<'r> {
    pub(crate) target: TextureRef<'r>,
    pub(crate) load: LoadOp,
    pub(crate) store: StoreOp,
}
impl<'r> ColorAttachment<'r> {
    /// Return the allocation to render into.
    pub const fn target(self) -> TextureRef<'r> {
        self.target
    }
    /// Return the initial contents operation.
    pub const fn load(self) -> LoadOp {
        self.load
    }
    /// Return the final contents operation.
    pub const fn store(self) -> StoreOp {
        self.store
    }
}
