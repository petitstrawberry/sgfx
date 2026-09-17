//! Typed views into a texture allocation.

use super::{Error, Result, TextureDesc, TextureFormat};

/// Coordinate interpretation used by a sampled texture binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureViewDimension {
    /// One two-dimensional layer.
    D2,
    /// An array of two-dimensional layers.
    D2Array,
    /// Six square faces addressed by a three-dimensional direction.
    Cube,
}

/// A validated format and subresource selection, independent of allocation identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureViewDesc {
    format: TextureFormat,
    dimension: TextureViewDimension,
    base_mip_level: u32,
    mip_level_count: u32,
    base_array_layer: u32,
    array_layer_count: u32,
}

impl TextureViewDesc {
    /// Select a nonempty range of mip levels and layers in an allocation.
    pub fn new(
        texture: TextureDesc,
        format: TextureFormat,
        dimension: TextureViewDimension,
        base_mip_level: u32,
        mip_level_count: u32,
        base_array_layer: u32,
        array_layer_count: u32,
    ) -> Result<Self> {
        let view = Self {
            format,
            dimension,
            base_mip_level,
            mip_level_count,
            base_array_layer,
            array_layer_count,
        };
        view.validate(texture)?;
        Ok(view)
    }

    /// Validate a view against the allocation being bound.
    pub fn validate(self, texture: TextureDesc) -> Result<()> {
        if !texture.format().view_compatible(self.format) {
            return Err(Error::InvalidDescriptor);
        }
        if self.mip_level_count == 0
            || self.array_layer_count == 0
            || self
                .base_mip_level
                .checked_add(self.mip_level_count)
                .is_none_or(|end| end > texture.mip_level_count())
            || self
                .base_array_layer
                .checked_add(self.array_layer_count)
                .is_none_or(|end| end > texture.array_layer_count())
        {
            return Err(Error::OutOfBounds);
        }
        match self.dimension {
            TextureViewDimension::D2 if self.array_layer_count != 1 => {
                Err(Error::InvalidDescriptor)
            }
            TextureViewDimension::Cube
                if self.array_layer_count != 6
                    || texture.extent().width() != texture.extent().height() =>
            {
                Err(Error::InvalidDescriptor)
            }
            _ => Ok(()),
        }
    }

    /// Return the texel interpretation.
    pub const fn format(self) -> TextureFormat {
        self.format
    }
    /// Return the shader coordinate interpretation.
    pub const fn dimension(self) -> TextureViewDimension {
        self.dimension
    }
    /// Return the first mip level.
    pub const fn base_mip_level(self) -> u32 {
        self.base_mip_level
    }
    /// Return the number of mip levels.
    pub const fn mip_level_count(self) -> u32 {
        self.mip_level_count
    }
    /// Return the first array layer.
    pub const fn base_array_layer(self) -> u32 {
        self.base_array_layer
    }
    /// Return the number of array layers.
    pub const fn array_layer_count(self) -> u32 {
        self.array_layer_count
    }
}
