//! SGFX backend ABI v2. Only scalars, C records, pointers and function pointers
//! cross this boundary. No Rust allocator, layout, trait object or unwind ABI is
//! shared. This is an in-process ABI for 64-bit little-endian targets.
//!
//! Command words and upload spans are borrowed for the duration of submit.
//! A backend consumes them before returning, including on partial failure.
//! Pending GPU work must own its transport allocations and imported resources.
//! Libraries stay resident until process exit; dropping an object calls the
//! destructor in the library that created it.

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]

use core::ffi::c_void;

#[cfg(not(all(target_pointer_width = "64", target_endian = "little")))]
compile_error!("SGFX backend ABI v2 requires a 64-bit little-endian target");

pub const VERSION: u32 = 2;
pub const ENTRY: &[u8] = b"sgfx_backend_get_api_v2\0";
pub const OK: i32 = 0;
pub const INVALID: i32 = 1;
pub const UNSUPPORTED: i32 = 2;
pub const OUT_OF_MEMORY: i32 = 3;
pub const DEVICE_LOST: i32 = 4;
pub const INITIALIZATION_FAILED: i32 = 5;
pub const BUSY: i32 = 6;
pub const ABI_MISMATCH: i32 = 7;
pub const PENDING: u32 = 0;
pub const COMPLETE: u32 = 1;
pub const REJECTED: u32 = 0;
pub const ACCEPTED: u32 = 1;
pub const PARTIAL: u32 = 2;
pub const RENDERING: u64 = 1;
pub const PRESENTATION: u64 = 2;
pub const IMAGE_UPLOAD: u64 = 4;
pub const IMAGE_READBACK: u64 = 8;
pub const DEPTH: u64 = 16;
pub const ASYNC: u64 = 32;
/// Confirmed process-wide CPU support, detected by the initialized host runtime.
pub const CPU_AARCH64_LSE: u64 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct HostInfo {
    pub size: u32,
    pub reserved: u32,
    pub cpu_features: u64,
}

pub type Object = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Borrowed span. A zero length permits a null pointer. Nonempty spans must be
/// aligned, initialized, immutable and valid for the entire consuming call.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Span<T> {
    pub data: *const T,
    pub len: usize,
}

impl<T> Span<T> {
    pub const fn from_slice(value: &[T]) -> Self {
        Self {
            data: value.as_ptr(),
            len: value.len(),
        }
    }

    /// # Safety
    /// The span must satisfy the documented allocation and lifetime contract.
    pub unsafe fn as_slice<'a>(self) -> &'a [T] {
        if self.len == 0 {
            &[]
        } else {
            // SAFETY: guaranteed by the caller; empty null spans are handled above.
            unsafe { core::slice::from_raw_parts(self.data, self.len) }
        }
    }
}

/// One borrowed canonical command stream. Words are recorded directly by the
/// frontend; submitting this record never serializes or copies the stream.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Batch {
    pub table: u64,
    pub words: Span<u64>,
    pub count: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    /// An owned duplicate. On success the caller closes this Scarlet handle.
    pub handle: i32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SubmitResult {
    pub disposition: u32,
    pub error: i32,
    /// Non-null exactly for ACCEPTED and PARTIAL. Independent of session life.
    pub receipt: Object,
}

impl Default for SubmitResult {
    fn default() -> Self {
        Self {
            disposition: REJECTED,
            error: INVALID,
            receipt: core::ptr::null_mut(),
        }
    }
}

/// All objects and spans are process-local. Calls on a device/context/session
/// are externally serialized. Receipts support concurrent observation from any
/// thread; destruction runs after the caller's last observer has returned.
/// Functions never unwind. Output records are initialized on every return path.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BackendApi {
    pub version: u32,
    pub size: u32,
    /// NUL-terminated ASCII identifiers, without path separators.
    pub name: [u8; 64],
    pub gpu_backend: [u8; 32],
    pub open: unsafe extern "C" fn(Span<u8>, *mut Object, *mut u64) -> i32,
    pub drop_device: unsafe extern "C" fn(Object),
    pub create_context: unsafe extern "C" fn(Object, *mut Object) -> i32,
    pub drop_context: unsafe extern "C" fn(Object),
    pub create_session:
        unsafe extern "C" fn(Object, Span<u64>, Span<u32>, *mut Object, *mut u64) -> i32,
    pub drop_session: unsafe extern "C" fn(Object),
    /// Cold path only: synchronize new/retired logical resource descriptors.
    pub sync_resources: unsafe extern "C" fn(Object, Span<u64>) -> i32,
    pub image: unsafe extern "C" fn(Object, u32, *mut ImageInfo) -> i32,
    pub readback: unsafe extern "C" fn(Object, u32, *mut u8, usize, u32, Rect) -> i32,
    /// Consumes the owned handle on every return path.
    pub import_bgra: unsafe extern "C" fn(Object, u32, i32) -> i32,
    pub release_import: unsafe extern "C" fn(Object, u32) -> i32,
    pub execute: unsafe extern "C" fn(Object, *const Batch) -> i32,
    pub submit: unsafe extern "C" fn(Object, *const Batch, *mut SubmitResult),
    /// timeout_ns == u64::MAX means no deadline; zero polls.
    pub wait: unsafe extern "C" fn(Object, u64, *mut u32) -> i32,
    pub drop_receipt: unsafe extern "C" fn(Object),
}

/// Writes exactly one v2 table only when version and size are compatible.
pub type GetApi = unsafe extern "C" fn(u32, usize, *const HostInfo, *mut BackendApi) -> i32;

const _: () = {
    assert!(core::mem::size_of::<Span<u8>>() == 16);
    assert!(core::mem::size_of::<HostInfo>() == 16);
    assert!(core::mem::size_of::<Batch>() == 32);
    assert!(core::mem::size_of::<Rect>() == 16);
    assert!(core::mem::size_of::<ImageInfo>() == 16);
    assert!(core::mem::size_of::<SubmitResult>() == 16);
    assert!(core::mem::size_of::<BackendApi>() == 224);
    assert!(core::mem::offset_of!(BackendApi, open) == 104);
    assert!(core::mem::offset_of!(BackendApi, drop_receipt) == 216);
};

pub const PROGRAMMABLE_GRAPHICS: u64 = 64;
pub const TEXTURE_ARRAYS: u64 = 128;
pub const DEPTH_SAMPLING: u64 = 256;
pub const IMAGE_MIPS: u64 = 512;
/// Read-only storage buffers: four bindings per stage, up to 256 KiB each.
/// This does not advertise storage writes or compute support.
pub const READ_ONLY_STORAGE_BUFFERS: u64 = 1024;
/// Typed sampled views, including array, cube and depth views. The applicable
/// texture-array and depth-sampling flags must also be advertised.
pub const TYPED_TEXTURE_VIEWS: u64 = 2048;
/// Sampled sRGB views with correct decoding; no sRGB attachment guarantee.
pub const SRGB_TEXTURE_VIEWS: u64 = 4096;
/// All portable SGFX vertex formats, including narrow and packed formats.
pub const EXTENDED_VERTEX_FORMATS: u64 = 8192;
/// RGBA8 UNORM color attachments.
pub const RGBA8_COLOR_ATTACHMENT: u64 = 16384;
/// Color image blits with the portable filtering and subresource rules.
pub const IMAGE_BLITS: u64 = 32768;
/// Up to 128 bytes of push constants for programmable graphics.
pub const PUSH_CONSTANTS_128: u64 = 65536;
/// Up to eight simultaneous programmable color attachments.
pub const COLOR_ATTACHMENTS_8: u64 = 131072;

/// Optional v2 extension for low-level API frontends, including Vulkan. The
/// original mapped-session table remains binary compatible. Resolve once at
/// load time; all objects below belong to the same loaded backend.
pub const DRIVER_ENTRY: &[u8] = b"sgfx_backend_get_driver_api_v2\0";
pub const VALIDATE_SHADER: u32 = 1;
pub const VALIDATE_RENDER_PIPELINE: u32 = 2;
pub const VALIDATE_COMPUTE_PIPELINE: u32 = 3;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DriverApi {
    pub version: u32,
    pub size: u32,
    pub create_resources: unsafe extern "C" fn(Object, Span<u64>, *mut Object) -> i32,
    pub drop_resources: unsafe extern "C" fn(Object),
    pub sync_resources: unsafe extern "C" fn(Object, Span<u64>) -> i32,
    pub release_buffer: unsafe extern "C" fn(Object, u32) -> i32,
    pub validate: unsafe extern "C" fn(Object, u32, u32) -> i32,
    /// Read directly into caller-owned output storage; no allocator crosses ABI.
    pub read_buffer: unsafe extern "C" fn(Object, u32, u64, *mut u8, usize) -> i32,
    pub create_image: unsafe extern "C" fn(Object, u32, u32, *mut Object, *mut ImageInfo) -> i32,
    pub drop_image: unsafe extern "C" fn(Object),
    pub map_image: unsafe extern "C" fn(Object, u32, Object) -> i32,
    pub unmap_image: unsafe extern "C" fn(Object, u32) -> i32,
    pub create_queue: unsafe extern "C" fn(Object, *mut Object) -> i32,
    pub drop_queue: unsafe extern "C" fn(Object),
    pub submit: unsafe extern "C" fn(Object, Object, *const Batch, *mut SubmitResult),
    pub read_texture: unsafe extern "C" fn(Object, Object, u32, *mut u8, usize) -> i32,
    /// Retain one additional strong receipt owner. Thread-safe; never allocates.
    pub clone_receipt: unsafe extern "C" fn(Object),
    pub release_texture: unsafe extern "C" fn(Object, u32) -> i32,
    pub release_bind_group: unsafe extern "C" fn(Object, u32) -> i32,
}
pub type GetDriverApi = unsafe extern "C" fn(u32, usize, *mut DriverApi) -> i32;
const _: () = {
    assert!(core::mem::size_of::<DriverApi>() == 144);
    assert!(core::mem::offset_of!(DriverApi, create_resources) == 8);
};

/// Optional v2 extension for importing YCbCr images into a mapped session.
/// The original backend and low-level driver tables remain binary compatible.
pub const YCBCR_ENTRY: &[u8] = b"sgfx_backend_get_ycbcr_api_v2\0";
pub const YCBCR_MATRIX_BT601: u32 = 1;
pub const YCBCR_MATRIX_BT709: u32 = 2;
pub const YCBCR_RANGE_LIMITED: u32 = 1;
pub const YCBCR_RANGE_FULL: u32 = 2;
pub const YCBCR_CHROMA_COSITED: u32 = 1;
pub const YCBCR_CHROMA_MIDPOINT: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct YcbcrConversion {
    pub size: u32,
    pub reserved: u32,
    pub matrix: u32,
    pub range: u32,
    pub chroma_x: u32,
    pub chroma_y: u32,
}

impl YcbcrConversion {
    pub const fn is_valid(self) -> bool {
        self.size as usize == core::mem::size_of::<Self>()
            && self.reserved == 0
            && matches!(self.matrix, YCBCR_MATRIX_BT601 | YCBCR_MATRIX_BT709)
            && matches!(self.range, YCBCR_RANGE_LIMITED | YCBCR_RANGE_FULL)
            && matches!(self.chroma_x, YCBCR_CHROMA_COSITED | YCBCR_CHROMA_MIDPOINT)
            && matches!(self.chroma_y, YCBCR_CHROMA_COSITED | YCBCR_CHROMA_MIDPOINT)
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct YcbcrApi {
    pub version: u32,
    pub size: u32,
    /// Imports into a session from the same backend. Consumes the owned Scarlet
    /// handle on every return path, including invalid conversion or session.
    pub import_ycbcr: unsafe extern "C" fn(Object, u32, i32, YcbcrConversion) -> i32,
}

/// Writes the complete extension table only when version and size are compatible.
pub type GetYcbcrApi = unsafe extern "C" fn(u32, usize, *mut YcbcrApi) -> i32;

const _: () = {
    assert!(core::mem::size_of::<YcbcrConversion>() == 24);
    assert!(core::mem::offset_of!(YcbcrConversion, matrix) == 8);
    assert!(core::mem::offset_of!(YcbcrConversion, chroma_y) == 20);
    assert!(core::mem::size_of::<YcbcrApi>() == 16);
    assert!(core::mem::offset_of!(YcbcrApi, import_ycbcr) == 8);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ycbcr_extension_preserves_existing_table_layouts() {
        assert_eq!(core::mem::size_of::<BackendApi>(), 224);
        assert_eq!(core::mem::size_of::<DriverApi>(), 144);
        assert_eq!(core::mem::size_of::<YcbcrConversion>(), 24);
        assert_eq!(core::mem::align_of::<YcbcrConversion>(), 4);
        assert_eq!(core::mem::size_of::<YcbcrApi>(), 16);
        assert_eq!(core::mem::align_of::<YcbcrApi>(), 8);
    }

    #[test]
    fn ycbcr_conversion_accepts_all_known_combinations() {
        for matrix in [YCBCR_MATRIX_BT601, YCBCR_MATRIX_BT709] {
            for range in [YCBCR_RANGE_LIMITED, YCBCR_RANGE_FULL] {
                for chroma_x in [YCBCR_CHROMA_COSITED, YCBCR_CHROMA_MIDPOINT] {
                    for chroma_y in [YCBCR_CHROMA_COSITED, YCBCR_CHROMA_MIDPOINT] {
                        assert!(
                            YcbcrConversion {
                                size: 24,
                                reserved: 0,
                                matrix,
                                range,
                                chroma_x,
                                chroma_y,
                            }
                            .is_valid()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn ycbcr_conversion_rejects_invalid_fields() {
        let conversion = YcbcrConversion {
            size: 24,
            reserved: 0,
            matrix: YCBCR_MATRIX_BT709,
            range: YCBCR_RANGE_LIMITED,
            chroma_x: YCBCR_CHROMA_COSITED,
            chroma_y: YCBCR_CHROMA_MIDPOINT,
        };
        for size in [0, 23, 25, u32::MAX] {
            assert!(!YcbcrConversion { size, ..conversion }.is_valid());
        }
        assert!(
            !YcbcrConversion {
                reserved: 1,
                ..conversion
            }
            .is_valid()
        );
        for invalid in [0, 3, u32::MAX] {
            assert!(
                !YcbcrConversion {
                    matrix: invalid,
                    ..conversion
                }
                .is_valid()
            );
            assert!(
                !YcbcrConversion {
                    range: invalid,
                    ..conversion
                }
                .is_valid()
            );
            assert!(
                !YcbcrConversion {
                    chroma_x: invalid,
                    ..conversion
                }
                .is_valid()
            );
            assert!(
                !YcbcrConversion {
                    chroma_y: invalid,
                    ..conversion
                }
                .is_valid()
            );
        }
    }
}
