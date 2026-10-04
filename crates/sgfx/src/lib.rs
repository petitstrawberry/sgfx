//! Cross-platform SGFX execution facade and backend selector.
//!
//! Portable renderers record resources and command buffers through
//! [`sgfx_core`]. Applications and platform composition roots use this crate
//! to select one complete execution backend. A selected backend continues to
//! own physical resources, command lowering, transport limits, and submission.
//! Renderer/API frontends lower into the common IR; this facade delegates
//! execution and does not introduce a separate application command model.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[cfg(all(
    not(feature = "std"),
    target_os = "scarlet",
    feature = "legacy-scarlet-std"
))]
extern crate scarlet_std as std;

use core::fmt;

pub use sgfx_core::{backend, ir};

#[cfg(all(target_os = "linux", feature = "scarlet-native-api"))]
pub use virgl::Handle;

/// Backend-neutral adapter, device, resource, queue, and completion facade.
#[cfg(any(
    sgfx_dynamic,
    all(not(target_os = "scarlet"), feature = "backend-wgpu"),
    all(
        any(
            target_os = "scarlet",
            all(target_os = "linux", feature = "scarlet-native-api")
        ),
        any(
            feature = "backend-scarlet-virgl",
            feature = "backend-scarlet-virgl-static",
            sgfx_dynamic_virgl
        )
    )
))]
pub mod driver;

#[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
mod host;
#[cfg(all(
    target_os = "scarlet",
    any(
        feature = "backend-scarlet-virgl",
        feature = "backend-scarlet-virgl-static",
        feature = "backend-scarlet-adreno",
        sgfx_dynamic
    )
))]
mod scarlet;

#[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
pub use host::{Executor, MappedTargetSession, Submission, WindowContext};
#[cfg(all(
    target_os = "scarlet",
    any(
        feature = "backend-scarlet-virgl",
        feature = "backend-scarlet-virgl-static",
        feature = "backend-scarlet-adreno",
        sgfx_dynamic
    )
))]
pub use scarlet::{
    Capabilities, Context, Device, Executor, Handle, ImageRef, MappedTargetSession, Submission,
};

#[cfg(sgfx_dynamic)]
pub mod dynamic;
#[cfg(sgfx_dynamic_virgl)]
use dynamic as virgl;
#[cfg(all(
    not(sgfx_dynamic_virgl),
    target_os = "scarlet",
    target_pointer_width = "64",
    feature = "backend-scarlet-virgl-static"
))]
use sgfx_backend_scarlet_virgl as virgl;
#[cfg(all(
    any(
        all(target_os = "scarlet", target_pointer_width = "32"),
        all(target_os = "linux", feature = "scarlet-native-api")
    ),
    any(
        feature = "backend-scarlet-virgl",
        feature = "backend-scarlet-virgl-static",
        sgfx_dynamic_virgl
    )
))]
use sgfx_backend_scarlet_virgl_compat as virgl;

/// Environment variable used to override automatic SGFX backend selection.
pub const BACKEND_ENV: &str = "SGFX_BACKEND";

/// Fixed-capacity driver identifier, independent of the known-backend enum.
#[cfg(feature = "backend-dynamic")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendName {
    bytes: [u8; 32],
    len: u8,
}
#[cfg(feature = "backend-dynamic")]
impl BackendName {
    pub fn new(name: &str) -> Option<Self> {
        if name.is_empty()
            || name.len() >= 32
            || !name
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.'))
        {
            return None;
        }
        let mut bytes = [0; 32];
        bytes[..name.len()].copy_from_slice(name.as_bytes());
        Some(Self {
            bytes,
            len: name.len() as u8,
        })
    }
    pub fn as_str(&self) -> &str {
        core::str::from_utf8(&self.bytes[..self.len as usize]).expect("ASCII driver identifier")
    }
}

/// A complete execution backend that may be selected by the SGFX frontend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// An installed backend whose name is not compiled into the facade.
    #[cfg(feature = "backend-dynamic")]
    Other(BackendName),
    /// SGFX execution through WGPU.
    Wgpu,
    /// Reserved for direct SGFX execution through Metal; currently unavailable.
    /// WGPU using Metal is still [`BackendKind::Wgpu`].
    Metal,
    /// SGFX VirGL execution through the Scarlet GPU ABI.
    ScarletVirgl,
    /// SGFX native Qualcomm Adreno execution through the Scarlet GPU ABI.
    ScarletAdreno,
}

impl BackendKind {
    /// Return the stable configuration name for this backend.
    ///
    /// # Returns
    ///
    /// A value accepted by [`BACKEND_ENV`].
    pub fn as_str(&self) -> &str {
        match self {
            #[cfg(feature = "backend-dynamic")]
            Self::Other(name) => name.as_str(),
            Self::Wgpu => "wgpu",
            Self::Metal => "metal",
            Self::ScarletVirgl => "scarlet-virgl",
            Self::ScarletAdreno => "scarlet-adreno",
        }
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Backend selection requested by an application or the process environment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BackendPreference {
    /// Require an independently installed backend by its manifest name.
    #[cfg(feature = "backend-dynamic")]
    Other(BackendName),
    /// Select the best available backend for the active target and device.
    #[default]
    Auto,
    /// Require WGPU execution.
    Wgpu,
    /// Require direct Metal execution (currently unavailable).
    Metal,
    /// Require Scarlet/VirGL execution.
    ScarletVirgl,
    /// Require native Scarlet/Adreno execution.
    ScarletAdreno,
}

impl BackendPreference {
    /// Parse one backend preference name.
    ///
    /// # Arguments
    ///
    /// * `value` - `auto` or one stable [`BackendKind`] name; `virgl` and
    ///   `adreno` remain aliases for their `scarlet-` names. Parsing is
    ///   case-sensitive and does not trim whitespace.
    ///
    /// # Returns
    ///
    /// The parsed preference, or [`Error::InvalidBackendPreference`].
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "auto" => Ok(Self::Auto),
            "wgpu" => Ok(Self::Wgpu),
            "metal" => Ok(Self::Metal),
            "scarlet-virgl" | "virgl" => Ok(Self::ScarletVirgl),
            "scarlet-adreno" | "adreno" => Ok(Self::ScarletAdreno),
            #[cfg(feature = "backend-dynamic")]
            name => BackendName::new(name)
                .map(Self::Other)
                .ok_or(Error::InvalidBackendPreference),
            #[cfg(not(feature = "backend-dynamic"))]
            _ => Err(Error::InvalidBackendPreference),
        }
    }

    /// Read the process backend preference.
    ///
    /// # Returns
    ///
    /// The parsed [`BACKEND_ENV`] value, or [`BackendPreference::Auto`] when
    /// the variable is absent. Without `std` or Scarlet's legacy runtime
    /// feature, environment lookup is unavailable and this returns `Auto`.
    pub fn from_environment() -> Result<Self> {
        match backend_environment_value() {
            Some(value) => Self::parse(value.as_str()),
            None => Ok(Self::Auto),
        }
    }
}

/// Failure returned by the cross-platform SGFX frontend.
#[derive(Debug)]
pub enum Error {
    /// Dynamic driver discovery, ABI, or execution failure.
    #[cfg(sgfx_dynamic)]
    Dynamic(dynamic::DynamicError),
    /// The requested backend name is invalid.
    InvalidBackendPreference,
    /// The requested backend was not compiled for this target.
    BackendUnavailable(BackendKind),
    /// A resource cache and queue were created by different devices.
    ResourceDeviceMismatch,
    /// WGPU initialization, materialization, execution, or presentation failed.
    #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
    Wgpu(sgfx_backend_wgpu::Error),
    /// A Scarlet/VirGL device or context operation failed.
    #[cfg(all(
        any(
            target_os = "scarlet",
            all(target_os = "linux", feature = "scarlet-native-api")
        ),
        any(
            feature = "backend-scarlet-virgl",
            feature = "backend-scarlet-virgl-static",
            sgfx_dynamic_virgl
        )
    ))]
    ScarletVirglHandle(virgl::HandleError),
    /// A Scarlet/VirGL IR materialization or execution operation failed.
    #[cfg(all(
        any(
            target_os = "scarlet",
            all(target_os = "linux", feature = "scarlet-native-api")
        ),
        any(
            feature = "backend-scarlet-virgl",
            feature = "backend-scarlet-virgl-static",
            sgfx_dynamic_virgl
        )
    ))]
    #[cfg(not(sgfx_dynamic_virgl))]
    ScarletVirglIr(virgl::IrSubmitError),
    /// A Scarlet GPU control connection could not be opened or queried.
    #[cfg(all(
        any(
            target_os = "scarlet",
            all(target_os = "linux", feature = "scarlet-native-api")
        ),
        any(
            feature = "backend-scarlet-virgl",
            feature = "backend-scarlet-virgl-static",
            feature = "backend-scarlet-adreno",
            sgfx_dynamic
        )
    ))]
    ScarletGpu,

    /// The opened Scarlet GPU does not match an explicitly requested backend.
    #[cfg(all(
        any(
            target_os = "scarlet",
            all(target_os = "linux", feature = "scarlet-native-api")
        ),
        any(
            feature = "backend-scarlet-virgl",
            feature = "backend-scarlet-virgl-static",
            feature = "backend-scarlet-adreno",
            sgfx_dynamic
        )
    ))]
    BackendDeviceMismatch(BackendKind),

    /// No available Scarlet backend supports the opened GPU.
    #[cfg(all(
        any(
            target_os = "scarlet",
            all(target_os = "linux", feature = "scarlet-native-api")
        ),
        any(
            feature = "backend-scarlet-virgl",
            feature = "backend-scarlet-virgl-static",
            feature = "backend-scarlet-adreno",
            sgfx_dynamic
        )
    ))]
    ScarletBackendUnsupported,

    /// A Scarlet/Adreno device or context operation failed.
    #[cfg(all(target_os = "scarlet", feature = "backend-scarlet-adreno"))]
    ScarletAdrenoHandle(sgfx_backend_scarlet_adreno::HandleError),
    /// A Scarlet/Adreno IR materialization or execution operation failed.
    #[cfg(all(target_os = "scarlet", feature = "backend-scarlet-adreno"))]
    ScarletAdrenoIr(sgfx_backend_scarlet_adreno::IrSubmitError),
}

/// Backend-neutral failure category for API frontends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    /// A descriptor, handle relationship, or command was invalid.
    InvalidInput,
    /// The selected complete backend cannot represent the requested feature.
    Unsupported,
    /// Host-side allocation failed.
    OutOfHostMemory,
    /// A device limit or device-side allocation was exhausted.
    OutOfDeviceMemory,
    /// Adapter or device initialization could not be completed.
    InitializationFailed,
    /// Accepted work or completion can no longer be trusted.
    DeviceLost,
}

impl Error {
    /// Classify this failure without exposing a concrete backend to a frontend.
    pub fn kind(&self) -> ErrorKind {
        match self {
            #[cfg(sgfx_dynamic)]
            Self::Dynamic(error) => match error {
                dynamic::DynamicError::Loader(_) => ErrorKind::InitializationFailed,
                dynamic::DynamicError::RecordingMode => ErrorKind::InvalidInput,
                dynamic::DynamicError::Status(code) => match *code {
                    sgfx_backend_abi::INVALID => ErrorKind::InvalidInput,
                    sgfx_backend_abi::UNSUPPORTED => ErrorKind::Unsupported,
                    sgfx_backend_abi::OUT_OF_MEMORY => ErrorKind::OutOfHostMemory,
                    sgfx_backend_abi::INITIALIZATION_FAILED | sgfx_backend_abi::ABI_MISMATCH => {
                        ErrorKind::InitializationFailed
                    }
                    _ => ErrorKind::DeviceLost,
                },
            },
            Self::InvalidBackendPreference | Self::ResourceDeviceMismatch => {
                ErrorKind::InvalidInput
            }
            Self::BackendUnavailable(_) => ErrorKind::InitializationFailed,
            #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
            Self::Wgpu(error) => {
                use sgfx_backend_wgpu::Error as E;
                match error {
                    E::InvalidIr(error) => ir_error_kind(*error),
                    E::ResourceTableMismatch | E::ImageNotMapped | E::ImageAlreadyMapped => {
                        ErrorKind::InvalidInput
                    }
                    E::Unsupported(_) | E::Validation(_) => ErrorKind::Unsupported,
                    E::AdapterUnavailable | E::DeviceRequest | E::SurfaceCreation => {
                        ErrorKind::InitializationFailed
                    }
                    E::InvalidState
                    | E::SurfaceAcquire
                    | E::DeviceLost
                    | E::CompletionObservation => ErrorKind::DeviceLost,
                }
            }
            #[cfg(all(
                any(
                    target_os = "scarlet",
                    all(target_os = "linux", feature = "scarlet-native-api")
                ),
                any(
                    feature = "backend-scarlet-virgl",
                    feature = "backend-scarlet-virgl-static",
                    sgfx_dynamic_virgl
                )
            ))]
            Self::ScarletVirglHandle(error) => virgl_handle_error_kind(*error),
            #[cfg(all(
                any(
                    target_os = "scarlet",
                    all(target_os = "linux", feature = "scarlet-native-api")
                ),
                any(
                    feature = "backend-scarlet-virgl",
                    feature = "backend-scarlet-virgl-static",
                    sgfx_dynamic_virgl
                )
            ))]
            #[cfg(not(sgfx_dynamic_virgl))]
            Self::ScarletVirglIr(error) => {
                use virgl::IrSubmitError as E;
                match error {
                    E::InvalidIr(error) => ir_error_kind(*error),
                    E::ResourceTableMismatch
                    | E::ContextMismatch
                    | E::TargetExtentMismatch
                    | E::ImageNotMapped
                    | E::TextureAlreadyMapped
                    | E::ImageAlreadyMapped
                    | E::InvalidVertexData => ErrorKind::InvalidInput,
                    E::OutOfMemory => ErrorKind::OutOfHostMemory,
                    E::SubmissionTooLarge => ErrorKind::OutOfDeviceMemory,
                    E::Unsupported(_) | E::ShaderCompile(_) => ErrorKind::Unsupported,
                    E::Backend(error) => virgl_handle_error_kind(*error),
                    E::CompletionFailed(_) | E::CompletionUnavailable | E::SubmissionFailed => {
                        ErrorKind::DeviceLost
                    }
                }
            }
            #[cfg(all(
                any(
                    target_os = "scarlet",
                    all(target_os = "linux", feature = "scarlet-native-api")
                ),
                any(
                    feature = "backend-scarlet-virgl",
                    feature = "backend-scarlet-virgl-static",
                    feature = "backend-scarlet-adreno",
                    sgfx_dynamic
                )
            ))]
            Self::ScarletGpu | Self::BackendDeviceMismatch(_) => ErrorKind::InitializationFailed,

            #[cfg(all(
                any(
                    target_os = "scarlet",
                    all(target_os = "linux", feature = "scarlet-native-api")
                ),
                any(
                    feature = "backend-scarlet-virgl",
                    feature = "backend-scarlet-virgl-static",
                    feature = "backend-scarlet-adreno",
                    sgfx_dynamic
                )
            ))]
            Self::ScarletBackendUnsupported => ErrorKind::Unsupported,

            #[cfg(all(target_os = "scarlet", feature = "backend-scarlet-adreno"))]
            Self::ScarletAdrenoHandle(error) => adreno_handle_error_kind(*error),
            #[cfg(all(target_os = "scarlet", feature = "backend-scarlet-adreno"))]
            Self::ScarletAdrenoIr(error) => {
                use sgfx_backend_scarlet_adreno::IrSubmitError as E;
                match error {
                    E::InvalidIr(error) => ir_error_kind(*error),
                    E::ResourceTableMismatch
                    | E::ContextMismatch
                    | E::TargetExtentMismatch
                    | E::ImageNotMapped
                    | E::TextureAlreadyMapped
                    | E::ImageAlreadyMapped => ErrorKind::InvalidInput,
                    E::OutOfMemory => ErrorKind::OutOfHostMemory,
                    E::SubmissionTooLarge => ErrorKind::OutOfDeviceMemory,
                    E::Unsupported(_) | E::Codegen(_) | E::SubmitWire(_) | E::AsyncUnsupported => {
                        ErrorKind::Unsupported
                    }
                    E::Backend(error) => adreno_handle_error_kind(*error),
                    E::CompletionUnavailable | E::CompletionFailed(_) => ErrorKind::DeviceLost,
                }
            }
        }
    }

    /// Whether a side-effect-free rejection permits continued use of the executor.
    ///
    /// # Returns
    ///
    /// `true` for input, support, or allocation limits that do not poison the
    /// executor. This classification applies **only** to
    /// [`backend::SubmitError::Rejected`], never to a failed-prefix or completion
    /// error. Earlier accepted work must still retire successfully before its
    /// images or storage are reused. Retrying unchanged oversized/invalid input
    /// is not guaranteed to succeed. Unknown transport/device errors return false.
    pub fn is_recoverable_rejection(&self) -> bool {
        match self {
            #[cfg(sgfx_dynamic)]
            Self::Dynamic(dynamic::DynamicError::Status(code)) => matches!(
                *code,
                sgfx_backend_abi::INVALID
                    | sgfx_backend_abi::UNSUPPORTED
                    | sgfx_backend_abi::OUT_OF_MEMORY
                    | sgfx_backend_abi::BUSY
            ),
            #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
            Self::Wgpu(error) => {
                use sgfx_backend_wgpu::Error as E;
                matches!(
                    error,
                    E::InvalidIr(_)
                        | E::ResourceTableMismatch
                        | E::ImageNotMapped
                        | E::ImageAlreadyMapped
                        | E::Unsupported(_)
                )
            }
            #[cfg(all(
                any(
                    target_os = "scarlet",
                    all(target_os = "linux", feature = "scarlet-native-api")
                ),
                any(
                    feature = "backend-scarlet-virgl",
                    feature = "backend-scarlet-virgl-static",
                    sgfx_dynamic_virgl
                )
            ))]
            #[cfg(not(sgfx_dynamic_virgl))]
            Self::ScarletVirglIr(error) => {
                use virgl::{HandleError, IrSubmitError as E};
                matches!(
                    error,
                    E::InvalidIr(_)
                        | E::ResourceTableMismatch
                        | E::ContextMismatch
                        | E::TargetExtentMismatch
                        | E::ImageNotMapped
                        | E::TextureAlreadyMapped
                        | E::ImageAlreadyMapped
                        | E::Unsupported(_)
                        | E::InvalidVertexData
                        | E::OutOfMemory
                        | E::SubmissionTooLarge
                        | E::Backend(
                            HandleError::InvalidParameter
                                | HandleError::Unsupported
                                | HandleError::OutOfResources
                        )
                )
            }
            #[cfg(all(target_os = "scarlet", feature = "backend-scarlet-adreno"))]
            Self::ScarletAdrenoIr(error) => {
                use sgfx_backend_scarlet_adreno::{HandleError, IrSubmitError as E};
                matches!(
                    error,
                    E::InvalidIr(_)
                        | E::ResourceTableMismatch
                        | E::ContextMismatch
                        | E::TargetExtentMismatch
                        | E::ImageNotMapped
                        | E::TextureAlreadyMapped
                        | E::ImageAlreadyMapped
                        | E::Unsupported(_)
                        | E::OutOfMemory
                        | E::SubmissionTooLarge
                        | E::AsyncUnsupported
                        | E::Backend(
                            HandleError::InvalidParameter
                                | HandleError::Unsupported
                                | HandleError::OutOfResources
                        )
                )
            }
            _ => false,
        }
    }
}

#[cfg(any(
    all(not(target_os = "scarlet"), feature = "backend-wgpu"),
    all(
        any(
            target_os = "scarlet",
            all(target_os = "linux", feature = "scarlet-native-api")
        ),
        any(
            feature = "backend-scarlet-virgl",
            feature = "backend-scarlet-virgl-static",
            sgfx_dynamic_virgl
        )
    ),
    all(target_os = "scarlet", feature = "backend-scarlet-adreno")
))]
#[allow(dead_code)]
fn ir_error_kind(error: ir::Error) -> ErrorKind {
    match error {
        ir::Error::OutOfMemory => ErrorKind::OutOfHostMemory,
        ir::Error::ResourceLimitExceeded | ir::Error::CommandLimitExceeded => {
            ErrorKind::OutOfDeviceMemory
        }
        _ => ErrorKind::InvalidInput,
    }
}

#[cfg(all(
    any(
        target_os = "scarlet",
        all(target_os = "linux", feature = "scarlet-native-api")
    ),
    any(
        feature = "backend-scarlet-virgl",
        feature = "backend-scarlet-virgl-static",
        sgfx_dynamic_virgl
    )
))]
fn virgl_handle_error_kind(error: virgl::HandleError) -> ErrorKind {
    use virgl::HandleError as E;
    match error {
        E::InvalidParameter | E::InvalidHandle => ErrorKind::InvalidInput,
        E::Unsupported => ErrorKind::Unsupported,
        E::OutOfResources => ErrorKind::OutOfDeviceMemory,
        E::NotFound => ErrorKind::InitializationFailed,
        E::PermissionDenied | E::SystemError(_) => ErrorKind::DeviceLost,
    }
}

#[cfg(all(target_os = "scarlet", feature = "backend-scarlet-adreno"))]
fn adreno_handle_error_kind(error: sgfx_backend_scarlet_adreno::HandleError) -> ErrorKind {
    use sgfx_backend_scarlet_adreno::HandleError as E;
    match error {
        E::InvalidParameter | E::InvalidHandle => ErrorKind::InvalidInput,
        E::Unsupported => ErrorKind::Unsupported,
        E::OutOfResources => ErrorKind::OutOfDeviceMemory,
        E::NotFound => ErrorKind::InitializationFailed,
        E::PermissionDenied | E::SystemError(_) => ErrorKind::DeviceLost,
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(sgfx_dynamic)]
            Self::Dynamic(error) => write!(formatter, "SGFX dynamic backend failed: {error}"),
            Self::InvalidBackendPreference => {
                formatter.write_str("invalid SGFX backend preference")
            }
            Self::BackendUnavailable(backend) => {
                write!(formatter, "SGFX backend {backend} is unavailable")
            }
            Self::ResourceDeviceMismatch => {
                formatter.write_str("SGFX resource cache and queue belong to different devices")
            }
            #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
            Self::Wgpu(error) => write!(formatter, "SGFX WGPU backend failed: {error}"),
            #[cfg(all(
                any(
                    target_os = "scarlet",
                    all(target_os = "linux", feature = "scarlet-native-api")
                ),
                any(
                    feature = "backend-scarlet-virgl",
                    feature = "backend-scarlet-virgl-static",
                    sgfx_dynamic_virgl
                )
            ))]
            Self::ScarletVirglHandle(error) => {
                write!(formatter, "SGFX Scarlet/VirGL device failed: {error:?}")
            }
            #[cfg(all(
                any(
                    target_os = "scarlet",
                    all(target_os = "linux", feature = "scarlet-native-api")
                ),
                any(
                    feature = "backend-scarlet-virgl",
                    feature = "backend-scarlet-virgl-static",
                    sgfx_dynamic_virgl
                )
            ))]
            #[cfg(not(sgfx_dynamic_virgl))]
            Self::ScarletVirglIr(error) => {
                write!(formatter, "SGFX Scarlet/VirGL execution failed: {error:?}")
            }
            #[cfg(all(
                any(
                    target_os = "scarlet",
                    all(target_os = "linux", feature = "scarlet-native-api")
                ),
                any(
                    feature = "backend-scarlet-virgl",
                    feature = "backend-scarlet-virgl-static",
                    feature = "backend-scarlet-adreno",
                    sgfx_dynamic
                )
            ))]
            Self::ScarletGpu => formatter.write_str("SGFX Scarlet GPU control operation failed"),

            #[cfg(all(
                any(
                    target_os = "scarlet",
                    all(target_os = "linux", feature = "scarlet-native-api")
                ),
                any(
                    feature = "backend-scarlet-virgl",
                    feature = "backend-scarlet-virgl-static",
                    feature = "backend-scarlet-adreno",
                    sgfx_dynamic
                )
            ))]
            Self::BackendDeviceMismatch(backend) => {
                write!(
                    formatter,
                    "opened Scarlet GPU does not support SGFX backend {backend}"
                )
            }

            #[cfg(all(
                any(
                    target_os = "scarlet",
                    all(target_os = "linux", feature = "scarlet-native-api")
                ),
                any(
                    feature = "backend-scarlet-virgl",
                    feature = "backend-scarlet-virgl-static",
                    feature = "backend-scarlet-adreno",
                    sgfx_dynamic
                )
            ))]
            Self::ScarletBackendUnsupported => {
                formatter.write_str("no SGFX Scarlet backend supports the opened GPU")
            }

            #[cfg(all(target_os = "scarlet", feature = "backend-scarlet-adreno"))]
            Self::ScarletAdrenoHandle(error) => {
                write!(formatter, "SGFX Scarlet/Adreno device failed: {error:?}")
            }
            #[cfg(all(target_os = "scarlet", feature = "backend-scarlet-adreno"))]
            Self::ScarletAdrenoIr(error) => {
                write!(formatter, "SGFX Scarlet/Adreno execution failed: {error:?}")
            }
        }
    }
}

/// Result returned by the SGFX frontend.
pub type Result<T> = core::result::Result<T, Error>;

/// Configured SGFX execution environment.
pub struct Instance {
    backend: BackendKind,
    preference: BackendPreference,
}

impl Instance {
    /// Configure an SGFX backend using [`BACKEND_ENV`] and target defaults.
    ///
    /// # Returns
    ///
    /// A selected instance, or a configuration error.
    pub fn new() -> Result<Self> {
        Self::with_preference(BackendPreference::from_environment()?)
    }

    /// Configure an SGFX backend from an explicit preference.
    ///
    /// # Arguments
    ///
    /// * `preference` - Automatic or required backend selection.
    ///
    /// # Returns
    ///
    /// A selected instance, or [`Error::BackendUnavailable`].
    pub fn with_preference(preference: BackendPreference) -> Result<Self> {
        Ok(Self {
            backend: resolve_backend(preference)?,
            preference,
        })
    }

    /// Return the configured backend's default identity.
    ///
    /// # Returns
    ///
    /// The stable configured backend identity. Native dynamic discovery returns
    /// the pending `auto` identity for [`BackendPreference::Auto`]; use
    /// `Device::backend` after `Device::open` or `Instance::open_device` to obtain
    /// the installed backend selected from the GPU's identifier.
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Return the unresolved preference used when opening a device.
    ///
    /// # Returns
    ///
    /// The requested preference. In particular, [`BackendPreference::Auto`]
    /// remains unresolved until a Scarlet GPU has been opened and queried.
    pub const fn preference(&self) -> BackendPreference {
        self.preference
    }
}

fn resolve_backend(preference: BackendPreference) -> Result<BackendKind> {
    match preference {
        #[cfg(feature = "backend-dynamic")]
        BackendPreference::Other(name) => require_backend(BackendKind::Other(name)),
        BackendPreference::Auto => default_backend(),
        BackendPreference::Wgpu => require_backend(BackendKind::Wgpu),
        BackendPreference::Metal => require_backend(BackendKind::Metal),
        BackendPreference::ScarletVirgl => require_backend(BackendKind::ScarletVirgl),
        BackendPreference::ScarletAdreno => require_backend(BackendKind::ScarletAdreno),
    }
}

#[allow(unreachable_code)] // Feature combinations select one target-specific return.
fn default_backend() -> Result<BackendKind> {
    #[cfg(sgfx_dynamic)]
    {
        return Ok(BackendKind::Other(
            BackendName::new("auto").expect("constant name"),
        ));
    }
    #[cfg(all(
        any(
            target_os = "scarlet",
            all(target_os = "linux", feature = "scarlet-native-api")
        ),
        any(
            feature = "backend-scarlet-virgl",
            feature = "backend-scarlet-virgl-static",
            sgfx_dynamic_virgl
        )
    ))]
    {
        return Ok(BackendKind::ScarletVirgl);
    }
    #[cfg(all(
        target_os = "scarlet",
        not(any(
            feature = "backend-scarlet-virgl",
            feature = "backend-scarlet-virgl-static",
            sgfx_dynamic_virgl
        )),
        feature = "backend-scarlet-adreno"
    ))]
    {
        return Ok(BackendKind::ScarletAdreno);
    }
    #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
    {
        return Ok(BackendKind::Wgpu);
    }
    #[allow(unreachable_code)]
    Err(Error::BackendUnavailable(default_backend_kind()))
}

const fn default_backend_kind() -> BackendKind {
    if cfg!(all(
        target_os = "scarlet",
        not(any(
            feature = "backend-scarlet-virgl",
            feature = "backend-scarlet-virgl-static",
            sgfx_dynamic_virgl
        )),
        feature = "backend-scarlet-adreno"
    )) {
        BackendKind::ScarletAdreno
    } else if cfg!(any(
        target_os = "scarlet",
        all(target_os = "linux", feature = "scarlet-native-api")
    )) {
        BackendKind::ScarletVirgl
    } else {
        BackendKind::Wgpu
    }
}

fn require_backend(backend: BackendKind) -> Result<BackendKind> {
    let available = match backend {
        #[cfg(feature = "backend-dynamic")]
        BackendKind::Other(_) => cfg!(sgfx_dynamic),
        BackendKind::Wgpu => cfg!(all(not(target_os = "scarlet"), feature = "backend-wgpu")),
        BackendKind::Metal => false,
        BackendKind::ScarletVirgl => cfg!(all(
            any(
                target_os = "scarlet",
                all(target_os = "linux", feature = "scarlet-native-api")
            ),
            any(
                feature = "backend-scarlet-virgl",
                feature = "backend-scarlet-virgl-static",
                sgfx_dynamic_virgl
            )
        )),
        BackendKind::ScarletAdreno => cfg!(all(
            target_os = "scarlet",
            feature = "backend-scarlet-adreno"
        )),
    };
    if available
        || (cfg!(sgfx_dynamic)
            && matches!(
                backend,
                BackendKind::ScarletVirgl | BackendKind::ScarletAdreno
            ))
    {
        Ok(backend)
    } else {
        Err(Error::BackendUnavailable(backend))
    }
}

#[cfg(feature = "std")]
fn backend_environment_value() -> Option<std::string::String> {
    std::env::var(BACKEND_ENV).ok()
}

#[cfg(all(
    not(feature = "std"),
    target_os = "scarlet",
    feature = "legacy-scarlet-std"
))]
fn backend_environment_value() -> Option<std::string::String> {
    std::env::var(BACKEND_ENV)
}

#[cfg(not(any(
    feature = "std",
    all(target_os = "scarlet", feature = "legacy-scarlet-std")
)))]
fn backend_environment_value() -> Option<alloc::string::String> {
    None
}

#[cfg(test)]
mod tests {
    use super::{BackendKind, BackendPreference, Error, Instance};

    #[test]
    fn rejection_classification_never_assumes_unknown_errors_are_recoverable() {
        assert!(!Error::InvalidBackendPreference.is_recoverable_rejection());
        assert!(!Error::BackendUnavailable(BackendKind::Metal).is_recoverable_rejection());
    }

    #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
    #[test]
    fn rejection_classification_separates_host_limits_from_device_failure() {
        use sgfx_backend_wgpu::{Error as E, UnsupportedFeature};
        assert!(
            Error::Wgpu(E::Unsupported(UnsupportedFeature::ResourceSize))
                .is_recoverable_rejection()
        );
        for error in [E::DeviceLost, E::CompletionObservation, E::InvalidState] {
            assert!(!Error::Wgpu(error).is_recoverable_rejection());
        }
    }

    #[cfg(all(
        any(
            target_os = "scarlet",
            all(target_os = "linux", feature = "scarlet-native-api")
        ),
        any(
            feature = "backend-scarlet-virgl",
            feature = "backend-scarlet-virgl-static",
            sgfx_dynamic_virgl
        )
    ))]
    #[cfg(not(sgfx_dynamic_virgl))]
    #[test]
    fn rejection_classification_separates_native_limits_from_device_failure() {
        use virgl::{HandleError, IrSubmitError as E};
        for error in [
            E::SubmissionTooLarge,
            E::OutOfMemory,
            E::Backend(HandleError::OutOfResources),
        ] {
            assert!(Error::ScarletVirglIr(error).is_recoverable_rejection());
        }
        for error in [
            E::SubmissionFailed,
            E::CompletionUnavailable,
            E::CompletionFailed(1),
            E::Backend(HandleError::SystemError(-1)),
        ] {
            assert!(!Error::ScarletVirglIr(error).is_recoverable_rejection());
        }
    }

    #[test]
    fn parses_stable_backend_names() {
        for (kind, preference) in [
            (BackendKind::Wgpu, BackendPreference::Wgpu),
            (BackendKind::Metal, BackendPreference::Metal),
            (BackendKind::ScarletVirgl, BackendPreference::ScarletVirgl),
            (BackendKind::ScarletAdreno, BackendPreference::ScarletAdreno),
        ] {
            assert_eq!(
                BackendPreference::parse(kind.as_str()).expect("stable name"),
                preference
            );
        }
        assert_eq!(
            BackendPreference::parse("auto").expect("auto"),
            BackendPreference::Auto
        );
        assert_eq!(
            BackendPreference::parse("virgl").unwrap(),
            BackendPreference::ScarletVirgl
        );
        assert_eq!(
            BackendPreference::parse("adreno").unwrap(),
            BackendPreference::ScarletAdreno
        );
        for invalid in ["", " wgpu", "wgpu ", "../driver", "driver/name"] {
            assert!(matches!(
                BackendPreference::parse(invalid),
                Err(Error::InvalidBackendPreference)
            ));
        }
        #[cfg(not(feature = "backend-dynamic"))]
        for invalid in ["unknown", "WGPU"] {
            assert!(matches!(
                BackendPreference::parse(invalid),
                Err(Error::InvalidBackendPreference)
            ));
        }
        #[cfg(feature = "backend-dynamic")]
        for name in ["new-vendor", "WGPU"] {
            let BackendPreference::Other(parsed) = BackendPreference::parse(name).unwrap() else {
                panic!("installed name");
            };
            assert_eq!(parsed.as_str(), name);
        }
    }

    #[cfg(not(target_os = "scarlet"))]
    #[test]
    fn host_auto_respects_the_compiled_backend() {
        let result = Instance::with_preference(BackendPreference::Auto);
        if cfg!(feature = "backend-wgpu") {
            assert_eq!(
                result.expect("compiled WGPU backend").backend(),
                BackendKind::Wgpu
            );
        } else {
            assert!(matches!(
                result,
                Err(Error::BackendUnavailable(BackendKind::Wgpu))
            ));
        }
    }

    #[cfg(all(not(target_os = "scarlet"), not(feature = "std")))]
    #[test]
    fn host_facade_without_std_does_not_read_environment_overrides() {
        assert_eq!(
            BackendPreference::from_environment().expect("no environment lookup"),
            BackendPreference::Auto
        );
    }

    #[test]
    fn unavailable_backend_is_not_silently_substituted() {
        assert!(matches!(
            Instance::with_preference(BackendPreference::Metal),
            Err(Error::BackendUnavailable(BackendKind::Metal))
        ));
    }
}

#[cfg(all(
    any(
        target_os = "scarlet",
        all(target_os = "linux", feature = "scarlet-native-api")
    ),
    any(
        feature = "backend-scarlet-virgl",
        feature = "backend-scarlet-virgl-static",
        sgfx_dynamic_virgl
    )
))]
impl From<virgl::HandleError> for Error {
    fn from(error: virgl::HandleError) -> Self {
        Self::ScarletVirglHandle(error)
    }
}
#[cfg(all(
    not(sgfx_dynamic_virgl),
    any(
        target_os = "scarlet",
        all(target_os = "linux", feature = "scarlet-native-api")
    ),
    any(
        feature = "backend-scarlet-virgl",
        feature = "backend-scarlet-virgl-static",
        sgfx_dynamic_virgl
    )
))]
impl From<virgl::IrSubmitError> for Error {
    fn from(error: virgl::IrSubmitError) -> Self {
        Self::ScarletVirglIr(error)
    }
}

#[cfg(test)]
#[path = "../build_policy.rs"]
mod build_policy;
