//! Shader sources and their structural, backend-independent validation.

use alloc::string::String;
use alloc::vec::Vec;
use core::ops::{BitOr, BitOrAssign};

use super::{Error, Result};

/// Maximum number of words in a portable SPIR-V module.
pub const MAX_SPIRV_WORDS: usize = 1 << 20;
/// Maximum number of UTF-8 bytes in a portable WGSL module.
pub const MAX_WGSL_BYTES: usize = 4 << 20;

/// A programmable pipeline stage supported by the portable IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderStage {
    /// Per-vertex graphics processing.
    Vertex,
    /// Per-fragment graphics processing.
    Fragment,
    /// Compute workgroup processing.
    Compute,
}

/// Shader stages allowed to access a resource binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShaderStages(u8);

impl ShaderStages {
    /// Allow access from vertex shaders.
    pub const VERTEX: Self = Self(1 << 0);
    /// Allow access from fragment shaders.
    pub const FRAGMENT: Self = Self(1 << 1);
    /// Allow access from compute shaders.
    pub const COMPUTE: Self = Self(1 << 2);

    /// Return an empty stage set.
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Return whether this set contains no stages.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Return whether all stages in `other` are present.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Return the union of this stage set and `other`.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl From<ShaderStage> for ShaderStages {
    fn from(stage: ShaderStage) -> Self {
        match stage {
            ShaderStage::Vertex => Self::VERTEX,
            ShaderStage::Fragment => Self::FRAGMENT,
            ShaderStage::Compute => Self::COMPUTE,
        }
    }
}

impl BitOr for ShaderStages {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

impl BitOrAssign for ShaderStages {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = self.union(rhs);
    }
}

/// Owned shader source submitted to a backend compiler and validator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShaderSource {
    /// SPIR-V represented as decoded words, independent of host byte order.
    SpirV(Vec<u32>),
    /// UTF-8 WGSL source text.
    Wgsl(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DeclaredEntryPoint {
    stage: ShaderStage,
    name: String,
}

/// An immutable, structurally checked shader module descriptor.
///
/// SPIR-V validation here covers the header, instruction framing, and entry-point
/// metadata. WGSL receives only source-size and basic text checks. Backends must
/// parse and semantically validate the complete module before executing it;
/// successful construction does not establish shader validity or device support.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderModuleDesc {
    source: ShaderSource,
    entry_points: Vec<DeclaredEntryPoint>,
}

impl ShaderModuleDesc {
    /// Check and own a SPIR-V 1.0 through 1.6 module.
    ///
    /// The header must use the decoded SPIR-V magic, a non-zero ID bound no
    /// greater than `0x3fffff`, and schema zero. Entry points must use a portable
    /// vertex, fragment, or compute stage and contain a non-empty UTF-8 name.
    ///
    /// Returns [`Error::InvalidDescriptor`] for malformed structure,
    /// [`Error::ResourceLimitExceeded`] above [`MAX_SPIRV_WORDS`], or
    /// [`Error::OutOfMemory`] if metadata allocation fails. Opcode semantics,
    /// definitions of referenced IDs, and complete shader validity are checked
    /// by the backend.
    pub fn spirv(words: Vec<u32>) -> Result<Self> {
        if words.len() > MAX_SPIRV_WORDS {
            return Err(Error::ResourceLimitExceeded);
        }
        if words.len() < 5
            || words[0] != 0x0723_0203
            || words[1] & 0xff00_00ff != 0
            || words[1] >> 16 != 1
            || (words[1] >> 8) & 0xff > 6
            || words[3] == 0
            || words[3] > 0x003f_ffff
            || words[4] != 0
        {
            return Err(Error::InvalidDescriptor);
        }

        let bound = words[3];
        let mut entry_points = Vec::new();
        let mut offset = 5;
        while offset < words.len() {
            let word_count = (words[offset] >> 16) as usize;
            if word_count == 0 || word_count > words.len() - offset {
                return Err(Error::InvalidDescriptor);
            }
            if words[offset] & 0xffff == 15 {
                let entry_point = parse_entry_point(&words[offset..offset + word_count], bound)?;
                entry_points
                    .try_reserve(1)
                    .map_err(|_| Error::OutOfMemory)?;
                entry_points.push(entry_point);
            }
            offset += word_count;
        }

        Ok(Self {
            source: ShaderSource::SpirV(words),
            entry_points,
        })
    }

    /// Own non-empty WGSL text for subsequent backend parsing and validation.
    ///
    /// Returns [`Error::InvalidDescriptor`] for blank text or embedded NUL,
    /// or [`Error::ResourceLimitExceeded`] above [`MAX_WGSL_BYTES`]. This method
    /// does not parse WGSL or prove that any entry point exists.
    pub fn wgsl(source: String) -> Result<Self> {
        if source.len() > MAX_WGSL_BYTES {
            return Err(Error::ResourceLimitExceeded);
        }
        if source.trim().is_empty() || source.contains('\0') {
            return Err(Error::InvalidDescriptor);
        }
        Ok(Self {
            source: ShaderSource::Wgsl(source),
            entry_points: Vec::new(),
        })
    }

    /// Return the immutable source passed to the backend shader compiler.
    pub const fn source(&self) -> &ShaderSource {
        &self.source
    }

    /// Check whether a stage and name can be used as a module entry point.
    ///
    /// For SPIR-V this checks declared `OpEntryPoint` metadata. For WGSL it only
    /// checks for a non-empty name without NUL; the backend must validate the
    /// full identifier syntax, declaration, and shader stage when compiling.
    pub fn has_entry_point(&self, stage: ShaderStage, name: &str) -> bool {
        match self.source {
            ShaderSource::SpirV(_) => self
                .entry_points
                .iter()
                .any(|entry| entry.stage == stage && entry.name == name),
            ShaderSource::Wgsl(_) => !name.is_empty() && !name.contains('\0'),
        }
    }
}

fn parse_entry_point(instruction: &[u32], bound: u32) -> Result<DeclaredEntryPoint> {
    if instruction.len() < 4 || instruction[2] == 0 || instruction[2] >= bound {
        return Err(Error::InvalidDescriptor);
    }
    let stage = match instruction[1] {
        0 => ShaderStage::Vertex,
        4 => ShaderStage::Fragment,
        5 => ShaderStage::Compute,
        _ => return Err(Error::InvalidDescriptor),
    };
    let mut name_bytes = Vec::new();
    name_bytes
        .try_reserve((instruction.len() - 3) * 4)
        .map_err(|_| Error::OutOfMemory)?;
    for (word_index, word) in instruction.iter().enumerate().skip(3) {
        let bytes = word.to_le_bytes();
        for (byte_index, byte) in bytes.iter().copied().enumerate() {
            if byte == 0 {
                if name_bytes.is_empty()
                    || bytes[byte_index + 1..].iter().any(|padding| *padding != 0)
                    || instruction[word_index + 1..]
                        .iter()
                        .any(|id| *id == 0 || *id >= bound)
                {
                    return Err(Error::InvalidDescriptor);
                }
                let name = String::from_utf8(name_bytes).map_err(|_| Error::InvalidDescriptor)?;
                return Ok(DeclaredEntryPoint { stage, name });
            }
            name_bytes.push(byte);
        }
    }
    Err(Error::InvalidDescriptor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn module() -> Vec<u32> {
        // Only structural metadata; intentionally not a semantically valid shader.
        vec![
            0x0723_0203,
            0x0001_0000,
            0,
            8,
            0,
            (6 << 16) | 15,
            5,
            1,
            u32::from_le_bytes(*b"main"),
            0,
            2,
        ]
    }

    #[test]
    fn records_entry_point_name_and_stage_for_all_supported_versions() {
        for minor in 0..=6 {
            let mut words = module();
            words[1] |= minor << 8;
            let desc = ShaderModuleDesc::spirv(words.clone()).unwrap();
            assert_eq!(desc.source(), &ShaderSource::SpirV(words));
            assert!(desc.has_entry_point(ShaderStage::Compute, "main"));
            assert!(!desc.has_entry_point(ShaderStage::Vertex, "main"));
            assert!(!desc.has_entry_point(ShaderStage::Compute, "missing"));
        }
    }

    #[test]
    fn rejects_invalid_header_fields() {
        for (index, value) in [
            (0, 0x0302_2307),
            (1, 0x0000_0000),
            (1, 0x0001_0700),
            (1, 0x0002_0000),
            (1, 0x0001_0001),
            (1, 0x0101_0000),
            (3, 0),
            (3, 0x0040_0000),
            (4, 1),
        ] {
            let mut words = module();
            words[index] = value;
            assert_eq!(
                ShaderModuleDesc::spirv(words),
                Err(Error::InvalidDescriptor)
            );
        }
        for length in 0..5 {
            assert_eq!(
                ShaderModuleDesc::spirv(module()[..length].to_vec()),
                Err(Error::InvalidDescriptor)
            );
        }
    }

    #[test]
    fn rejects_zero_length_and_truncated_instructions() {
        for header in [15, (7 << 16) | 15] {
            let mut words = module();
            words[5] = header;
            assert_eq!(
                ShaderModuleDesc::spirv(words),
                Err(Error::InvalidDescriptor)
            );
        }
        let mut words = module();
        words.push(2 << 16);
        assert_eq!(
            ShaderModuleDesc::spirv(words),
            Err(Error::InvalidDescriptor)
        );
    }

    #[test]
    fn rejects_malformed_entry_point_metadata() {
        for (index, value) in [
            (6, 3), // Non-portable geometry stage.
            (7, 0),
            (7, 8),
            (8, 0),           // Empty name.
            (8, 0x0000_00ff), // Invalid UTF-8.
            (8, 0x0062_0061), // Non-zero NUL padding.
            (10, 0),          // Invalid interface IDs.
            (10, 8),
        ] {
            let mut words = module();
            words[index] = value;
            assert_eq!(
                ShaderModuleDesc::spirv(words),
                Err(Error::InvalidDescriptor)
            );
        }
        let mut words = module();
        words[5] = (4 << 16) | 15;
        words.truncate(9); // Name fills its word but has no NUL terminator.
        assert_eq!(
            ShaderModuleDesc::spirv(words),
            Err(Error::InvalidDescriptor)
        );
        let mut words = module();
        words[5] = (3 << 16) | 15;
        words.truncate(8);
        assert_eq!(
            ShaderModuleDesc::spirv(words),
            Err(Error::InvalidDescriptor)
        );
    }

    #[test]
    fn parses_utf8_entry_names_and_multiple_stages() {
        let mut words = module();
        words[8] = u32::from_le_bytes([0xe7, 0x82, 0xb9, 0]); // 点
        words.remove(9);
        words[5] = (5 << 16) | 15;
        words.extend([(4 << 16) | 15, 0, 3, u32::from_le_bytes(*b"vs\0\0")]);
        let desc = ShaderModuleDesc::spirv(words).unwrap();
        assert!(desc.has_entry_point(ShaderStage::Compute, "点"));
        assert!(desc.has_entry_point(ShaderStage::Vertex, "vs"));
    }

    #[test]
    fn bounds_owned_source_sizes() {
        assert_eq!(
            ShaderModuleDesc::spirv(vec![0; MAX_SPIRV_WORDS + 1]),
            Err(Error::ResourceLimitExceeded)
        );
        assert_eq!(
            ShaderModuleDesc::wgsl(" ".repeat(MAX_WGSL_BYTES + 1)),
            Err(Error::ResourceLimitExceeded)
        );
    }

    #[test]
    fn wgsl_entry_point_existence_is_left_to_the_backend() {
        for source in ["", " \n\t", "fn main() {}\0"] {
            assert_eq!(
                ShaderModuleDesc::wgsl(String::from(source)),
                Err(Error::InvalidDescriptor)
            );
        }
        let desc = ShaderModuleDesc::wgsl(String::from("backend validates this text")).unwrap();
        assert!(desc.has_entry_point(ShaderStage::Compute, "main"));
        assert!(desc.has_entry_point(ShaderStage::Fragment, "main"));
        assert!(desc.has_entry_point(ShaderStage::Compute, "e\u{301}"));
        for name in ["", "a\0b"] {
            assert!(!desc.has_entry_point(ShaderStage::Compute, name));
        }
    }

    #[test]
    fn stage_flags_compose() {
        let mut stages = ShaderStages::VERTEX | ShaderStages::FRAGMENT;
        assert!(stages.contains(ShaderStage::Vertex.into()));
        assert!(!stages.contains(ShaderStages::COMPUTE));
        stages |= ShaderStages::COMPUTE;
        assert!(stages.contains(ShaderStages::COMPUTE));
        assert!(ShaderStages::empty().is_empty());
    }
}
