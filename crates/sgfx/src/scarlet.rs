//! Scarlet device and mapped-target integration for the SGFX frontend.

// The same dispatch code wraps native errors and passes through dynamic errors.
#![cfg_attr(sgfx_dynamic, allow(clippy::useless_conversion))]

use alloc::rc::Rc;
use core::time::Duration;
use gpu_raw::Gpu;
use sgfx_core::backend::{
    CommandExecutor, CommandSubmitter, Completion, CompletionStatus, SubmitError,
};

use crate::{BackendKind, BackendPreference, Error, Instance, Result, ir};

#[cfg(sgfx_dynamic)]
pub use crate::dynamic::Handle;
#[cfg(all(
    not(sgfx_dynamic),
    any(
        feature = "backend-scarlet-virgl",
        feature = "backend-scarlet-virgl-static"
    )
))]
pub use crate::virgl::Handle;
#[cfg(all(
    not(any(
        feature = "backend-scarlet-virgl",
        feature = "backend-scarlet-virgl-static",
        sgfx_dynamic
    )),
    feature = "backend-scarlet-adreno"
))]
pub use sgfx_backend_scarlet_adreno::Handle;

#[cfg(all(
    not(any(
        feature = "backend-scarlet-virgl",
        feature = "backend-scarlet-virgl-static",
        sgfx_dynamic
    )),
    not(feature = "backend-scarlet-adreno"),
    sgfx_static_maxwell
))]
pub use crate::maxwell::Handle;

/// Backend-neutral Scarlet rendering capabilities.
#[derive(Clone, Copy, Debug)]
pub struct Capabilities {
    rendering: bool,
    presentation: bool,
    image_upload: bool,
    image_readback: bool,
    depth: bool,
}

impl Capabilities {
    /// Return whether SGFX command execution is available.
    ///
    /// # Returns
    ///
    /// `true` when rendering is supported.
    pub const fn supports_rendering(&self) -> bool {
        self.rendering
    }

    /// Return whether mapped images may be presented.
    ///
    /// # Returns
    ///
    /// `true` when presentation is supported.
    pub const fn supports_presentation(&self) -> bool {
        self.presentation
    }

    /// Return whether sampled image upload is available.
    ///
    /// # Returns
    ///
    /// `true` when image upload is supported.
    pub const fn supports_image_upload(&self) -> bool {
        self.image_upload
    }

    /// Return whether rendered BGRA images can be read back synchronously.
    ///
    /// # Returns
    ///
    /// `true` when image-to-CPU transfer is available.
    pub const fn supports_image_readback(&self) -> bool {
        self.image_readback
    }

    /// Return whether depth attachments are available.
    ///
    /// # Returns
    ///
    /// `true` when depth-enabled passes are supported.
    pub const fn supports_depth(&self) -> bool {
        self.depth
    }
}

/// Scarlet graphics device selected by the SGFX frontend.
pub enum Device {
    /// VirGL execution through Scarlet's VirtIO GPU ABI.
    #[cfg(any(
        feature = "backend-scarlet-virgl",
        feature = "backend-scarlet-virgl-static",
        sgfx_dynamic_virgl
    ))]
    Virgl(crate::virgl::Device),
    #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
    Dynamic(crate::dynamic::Device),
    /// Native Qualcomm Adreno execution through Scarlet's GPU ABI.
    #[cfg(feature = "backend-scarlet-adreno")]
    Adreno(sgfx_backend_scarlet_adreno::Device),
    #[cfg(sgfx_static_maxwell)]
    Maxwell(crate::maxwell::Device),
}

impl Device {
    /// Path of the installed driver used by this device, when dynamically loaded.
    pub fn backend_library(&self) -> Option<&str> {
        match self {
            #[cfg(sgfx_dynamic_virgl)]
            Self::Virgl(device) => device.backend_library(),
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            Self::Dynamic(device) => device.backend_library(),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }
    /// Open a Scarlet device using process backend selection policy.
    ///
    /// # Arguments
    ///
    /// * `path` - Scarlet GPU device path.
    ///
    /// # Returns
    ///
    /// A selected device or frontend/backend error.
    pub fn open(path: &str) -> Result<Self> {
        Instance::new()?.open_device(path)
    }

    /// Return the complete backend selected for this device.
    ///
    /// # Returns
    ///
    /// The stable Scarlet backend identity.
    pub fn backend(&self) -> BackendKind {
        match self {
            #[cfg(any(
                feature = "backend-scarlet-virgl",
                feature = "backend-scarlet-virgl-static",
                sgfx_dynamic_virgl
            ))]
            Self::Virgl(device) => {
                #[cfg(sgfx_dynamic_virgl)]
                {
                    device.backend()
                }
                #[cfg(not(sgfx_dynamic_virgl))]
                {
                    let _ = device;
                    BackendKind::ScarletVirgl
                }
            }
            #[cfg(feature = "backend-scarlet-adreno")]
            Self::Adreno(_) => BackendKind::ScarletAdreno,
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            Self::Dynamic(device) => device.backend(),
            #[cfg(sgfx_static_maxwell)]
            Self::Maxwell(_) => BackendKind::ScarletMaxwell,
        }
    }

    /// Return portable capabilities for the selected Scarlet backend.
    ///
    /// # Returns
    ///
    /// Backend-neutral rendering capabilities.
    pub fn capabilities(&self) -> Capabilities {
        match self {
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            Self::Dynamic(device) => {
                let capabilities = device.capabilities();
                Capabilities {
                    rendering: capabilities.supports_rendering(),
                    presentation: capabilities.supports_presentation(),
                    image_upload: capabilities.supports_image_upload(),
                    image_readback: capabilities.supports_image_readback(),
                    depth: capabilities.supports_depth(),
                }
            }
            #[cfg(any(
                feature = "backend-scarlet-virgl",
                feature = "backend-scarlet-virgl-static",
                sgfx_dynamic_virgl
            ))]
            Self::Virgl(device) => {
                let capabilities = device.capabilities();
                Capabilities {
                    rendering: capabilities.supports_rendering(),
                    presentation: capabilities.supports_presentation(),
                    image_upload: capabilities.supports_image_upload(),
                    image_readback: capabilities.supports_image_readback(),
                    depth: capabilities.supports_depth(),
                }
            }
            #[cfg(feature = "backend-scarlet-adreno")]
            Self::Adreno(device) => {
                let capabilities = device.capabilities();
                Capabilities {
                    rendering: capabilities.supports_rendering(),
                    presentation: capabilities.supports_presentation(),
                    image_upload: capabilities.supports_image_upload(),
                    image_readback: capabilities.supports_image_readback(),
                    depth: capabilities.supports_depth(),
                }
            }
            #[cfg(sgfx_static_maxwell)]
            Self::Maxwell(device) => {
                let capabilities = device.capabilities();
                Capabilities {
                    rendering: capabilities.supports_rendering(),
                    presentation: capabilities.supports_presentation(),
                    image_upload: capabilities.supports_image_upload(),
                    image_readback: capabilities.supports_image_readback(),
                    depth: capabilities.supports_depth(),
                }
            }
        }
    }

    /// Create a context through the selected backend.
    ///
    /// # Returns
    ///
    /// A frontend context or device error.
    pub fn create_context(&self) -> Result<Context> {
        match self {
            #[cfg(any(
                feature = "backend-scarlet-virgl",
                feature = "backend-scarlet-virgl-static",
                sgfx_dynamic_virgl
            ))]
            Self::Virgl(device) => device
                .create_context()
                .map(Context::Virgl)
                .map_err(Error::from),
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            Self::Dynamic(device) => device
                .create_context()
                .map(Context::Dynamic)
                .map_err(Error::from),
            #[cfg(feature = "backend-scarlet-adreno")]
            Self::Adreno(device) => device
                .create_context()
                .map(Context::Adreno)
                .map_err(Error::ScarletAdrenoHandle),
            #[cfg(sgfx_static_maxwell)]
            Self::Maxwell(device) => device
                .create_context()
                .map(Context::Maxwell)
                .map_err(Error::ScarletMaxwellHandle),
        }
    }
}

impl Instance {
    /// Open a Scarlet graphics device through the selected backend.
    ///
    /// Automatic selection opens the GPU once, queries its backend identifier,
    /// and selects its installed driver. Static comparison and compatibility
    /// backends take ownership of that connection; plugins open their own.
    ///
    /// # Arguments
    ///
    /// * `path` - Scarlet GPU device path.
    ///
    /// # Returns
    ///
    /// A frontend device or backend error.
    pub fn open_device(&self, path: &str) -> Result<Device> {
        let gpu = Gpu::open(path).map_err(|_| Error::ScarletGpu)?;
        let info = gpu.query_info().map_err(|_| Error::ScarletGpu)?;
        match self.preference() {
            #[cfg(feature = "backend-dynamic")]
            BackendPreference::Other(name) => {
                #[cfg(sgfx_dynamic)]
                {
                    drop(gpu);
                    open_dynamic(path, Some(name.as_str()))
                }
                #[cfg(not(sgfx_dynamic))]
                {
                    Err(Error::BackendUnavailable(BackendKind::Other(name)))
                }
            }
            BackendPreference::Auto => open_auto(path, gpu, info),
            BackendPreference::ScarletVirgl => open_virgl(path, gpu, info),
            BackendPreference::ScarletAdreno => open_adreno(path, gpu, info),
            BackendPreference::ScarletMaxwell => open_maxwell(path, gpu, info),
            BackendPreference::Wgpu => Err(Error::BackendUnavailable(BackendKind::Wgpu)),
            BackendPreference::Metal => Err(Error::BackendUnavailable(BackendKind::Metal)),
        }
    }
}

fn open_auto(path: &str, gpu: Gpu, info: gpu_raw::GpuQueryInfo) -> Result<Device> {
    let selected = match select_auto_backend(&info) {
        Ok(backend) => backend,
        Err(error) => {
            #[cfg(sgfx_dynamic)]
            {
                let _ = error;
                drop(gpu);
                return open_dynamic(path, None);
            }
            #[cfg(not(sgfx_dynamic))]
            {
                return Err(error);
            }
        }
    };
    match selected {
        BackendKind::ScarletVirgl => open_virgl(path, gpu, info),
        BackendKind::ScarletAdreno => open_adreno(path, gpu, info),
        BackendKind::ScarletMaxwell => open_maxwell(path, gpu, info),
        #[cfg(feature = "backend-dynamic")]
        BackendKind::Other(_) => Err(Error::ScarletBackendUnsupported),
        BackendKind::Wgpu | BackendKind::Metal => Err(Error::ScarletBackendUnsupported),
    }
}

fn select_auto_backend(info: &gpu_raw::GpuQueryInfo) -> Result<BackendKind> {
    #[cfg(any(
        feature = "backend-scarlet-virgl",
        feature = "backend-scarlet-virgl-static",
        sgfx_dynamic_virgl
    ))]
    if crate::virgl::Device::supports(info) {
        return Ok(BackendKind::ScarletVirgl);
    }
    #[cfg(feature = "backend-scarlet-adreno")]
    if sgfx_backend_scarlet_adreno::Device::supports(info) {
        return Ok(BackendKind::ScarletAdreno);
    }
    #[cfg(sgfx_static_maxwell)]
    if crate::maxwell::Device::supports(info) {
        return Ok(BackendKind::ScarletMaxwell);
    }
    #[cfg(all(sgfx_dynamic, not(sgfx_static_maxwell)))]
    if supports_gpu(info, b"nvidia-gm20b") {
        return Ok(BackendKind::ScarletMaxwell);
    }
    Err(Error::ScarletBackendUnsupported)
}

#[cfg(sgfx_dynamic)]
fn supports_gpu(info: &gpu_raw::GpuQueryInfo, backend_id: &[u8]) -> bool {
    info.backend_id_bytes() == backend_id
        && info.device_state == gpu_raw::GPU_DEVICE_STATE_READY
        && info.execution_support
            & (gpu_raw::GPU_EXECUTION_SUPPORT_QUEUE | gpu_raw::GPU_EXECUTION_SUPPORT_MEMORY)
            == gpu_raw::GPU_EXECUTION_SUPPORT_QUEUE | gpu_raw::GPU_EXECUTION_SUPPORT_MEMORY
}

#[cfg(sgfx_dynamic)]
fn open_dynamic(path: &str, preference: Option<&str>) -> Result<Device> {
    let device = crate::dynamic::Device::open_with_backend(path, preference)?;
    #[cfg(sgfx_dynamic_virgl)]
    return Ok(Device::Virgl(device));
    #[cfg(not(sgfx_dynamic_virgl))]
    return Ok(Device::Dynamic(device));
}

fn open_virgl(path: &str, gpu: Gpu, info: gpu_raw::GpuQueryInfo) -> Result<Device> {
    #[cfg(any(
        feature = "backend-scarlet-virgl",
        feature = "backend-scarlet-virgl-static",
        sgfx_dynamic_virgl
    ))]
    {
        if !crate::virgl::Device::supports(&info) {
            return Err(Error::BackendDeviceMismatch(BackendKind::ScarletVirgl));
        }
        #[cfg(sgfx_dynamic_virgl)]
        {
            drop(gpu);
            open_dynamic(path, Some("scarlet-virgl"))
        }
        #[cfg(not(sgfx_dynamic_virgl))]
        {
            let _ = path;
            crate::virgl::Device::from_gpu(gpu, info)
                .map(Device::Virgl)
                .map_err(Error::from)
        }
    }
    #[cfg(not(any(
        feature = "backend-scarlet-virgl",
        feature = "backend-scarlet-virgl-static",
        sgfx_dynamic_virgl
    )))]
    {
        let _ = (path, gpu, info);
        Err(Error::BackendUnavailable(BackendKind::ScarletVirgl))
    }
}

fn open_adreno(path: &str, gpu: Gpu, info: gpu_raw::GpuQueryInfo) -> Result<Device> {
    #[cfg(feature = "backend-scarlet-adreno")]
    {
        let _ = path;
        if !sgfx_backend_scarlet_adreno::Device::supports(&info) {
            return Err(Error::BackendDeviceMismatch(BackendKind::ScarletAdreno));
        }
        sgfx_backend_scarlet_adreno::Device::from_gpu(gpu, info)
            .map(Device::Adreno)
            .map_err(Error::ScarletAdrenoHandle)
    }
    #[cfg(all(not(feature = "backend-scarlet-adreno"), sgfx_dynamic))]
    {
        if !supports_gpu(&info, b"qcom-adreno") {
            return Err(Error::BackendDeviceMismatch(BackendKind::ScarletAdreno));
        }
        drop(gpu);
        open_dynamic(path, Some("scarlet-adreno"))
    }
    #[cfg(all(not(feature = "backend-scarlet-adreno"), not(sgfx_dynamic)))]
    {
        let _ = (path, gpu, info);
        Err(Error::BackendUnavailable(BackendKind::ScarletAdreno))
    }
}

/// Scarlet context selected by the SGFX frontend.
pub enum Context {
    /// A VirGL rendering context.
    #[cfg(any(
        feature = "backend-scarlet-virgl",
        feature = "backend-scarlet-virgl-static",
        sgfx_dynamic_virgl
    ))]
    Virgl(crate::virgl::Context),
    #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
    Dynamic(crate::dynamic::Context),
    /// A native Adreno rendering context.
    #[cfg(feature = "backend-scarlet-adreno")]
    Adreno(sgfx_backend_scarlet_adreno::Context),
    #[cfg(sgfx_static_maxwell)]
    Maxwell(crate::maxwell::Context),
}

impl Context {
    /// Create and map physical images for logical presentation targets.
    ///
    /// # Arguments
    ///
    /// * `resources` - Logical SGFX resource table.
    /// * `targets` - Presentation texture identities to materialize.
    ///
    /// # Returns
    ///
    /// A backend-owned mapped session or execution error.
    pub fn create_mapped_target_session(
        &self,
        resources: Rc<ir::ResourceTable>,
        targets: &[ir::TextureId],
    ) -> Result<MappedTargetSession> {
        match self {
            #[cfg(any(
                feature = "backend-scarlet-virgl",
                feature = "backend-scarlet-virgl-static",
                sgfx_dynamic_virgl
            ))]
            Self::Virgl(context) => context
                .create_mapped_target_session(resources, targets)
                .map(MappedTargetSession::Virgl)
                .map_err(Error::from),
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            Self::Dynamic(context) => context
                .create_mapped_target_session(resources, targets)
                .map(MappedTargetSession::Dynamic)
                .map_err(Error::from),
            #[cfg(feature = "backend-scarlet-adreno")]
            Self::Adreno(context) => context
                .create_mapped_target_session(resources, targets)
                .map(MappedTargetSession::Adreno)
                .map_err(Error::ScarletAdrenoIr),
            #[cfg(sgfx_static_maxwell)]
            Self::Maxwell(context) => context
                .create_mapped_target_session(resources, targets)
                .map(MappedTargetSession::Maxwell)
                .map_err(Error::ScarletMaxwellIr),
        }
    }
}

/// Scarlet mapped-target session selected by the SGFX frontend.
#[allow(clippy::large_enum_variant)]
pub enum MappedTargetSession {
    /// A VirGL mapped-target session.
    #[cfg(any(
        feature = "backend-scarlet-virgl",
        feature = "backend-scarlet-virgl-static",
        sgfx_dynamic_virgl
    ))]
    Virgl(crate::virgl::MappedTargetSession),
    #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
    Dynamic(crate::dynamic::MappedTargetSession),
    /// A native Adreno mapped-target session.
    #[cfg(feature = "backend-scarlet-adreno")]
    Adreno(sgfx_backend_scarlet_adreno::MappedTargetSession),
    #[cfg(sgfx_static_maxwell)]
    Maxwell(crate::maxwell::MappedTargetSession),
}

impl MappedTargetSession {
    /// Import a transferred shared BGRA image into a logical sampled texture.
    pub fn import_shared_bgra_texture(
        &mut self,
        texture: ir::TextureId,
        handle: Handle,
    ) -> Result<()> {
        match self {
            #[cfg(any(
                feature = "backend-scarlet-virgl",
                feature = "backend-scarlet-virgl-static",
                sgfx_dynamic_virgl
            ))]
            Self::Virgl(session) => session
                .import_shared_bgra_texture(texture, handle)
                .map_err(Error::from),
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            Self::Dynamic(session) => session
                .import_shared_bgra_texture(texture, handle)
                .map_err(Error::from),
            #[cfg(feature = "backend-scarlet-adreno")]
            Self::Adreno(session) => session
                .import_shared_bgra_texture(texture, handle)
                .map_err(Error::ScarletAdrenoIr),
            #[cfg(sgfx_static_maxwell)]
            Self::Maxwell(session) => session
                .import_shared_bgra_texture(texture, handle)
                .map_err(Error::ScarletMaxwellIr),
        }
    }

    /// Import a ready shared YCbCr image with an explicit RGB sampling conversion.
    /// The session retains the backing lease through GPU completion.
    pub fn import_ycbcr_texture(
        &mut self,
        texture: ir::TextureId,
        handle: Handle,
        conversion: ir::YcbcrConversion,
    ) -> Result<()> {
        #[cfg(not(any(sgfx_static_maxwell, sgfx_dynamic)))]
        let _ = (texture, handle, conversion);
        match self {
            #[cfg(sgfx_dynamic_virgl)]
            Self::Virgl(session) => session.import_ycbcr_texture(texture, handle, conversion),
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            Self::Dynamic(session) => session.import_ycbcr_texture(texture, handle, conversion),
            #[cfg(sgfx_static_maxwell)]
            Self::Maxwell(session) => session
                .import_ycbcr_texture(texture, handle, conversion)
                .map_err(Error::ScarletMaxwellIr),
            #[allow(unreachable_patterns)]
            _ => Err(Error::ScarletBackendUnsupported),
        }
    }

    /// Detach and release a previously imported sampled texture.
    pub fn release_imported_texture(&mut self, texture: ir::TextureId) -> Result<()> {
        match self {
            #[cfg(any(
                feature = "backend-scarlet-virgl",
                feature = "backend-scarlet-virgl-static",
                sgfx_dynamic_virgl
            ))]
            Self::Virgl(session) => session
                .release_imported_texture(texture)
                .map_err(Error::from),
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            Self::Dynamic(session) => session
                .release_imported_texture(texture)
                .map_err(Error::from),
            #[cfg(feature = "backend-scarlet-adreno")]
            Self::Adreno(session) => session
                .release_imported_texture(texture)
                .map_err(Error::ScarletAdrenoIr),
            #[cfg(sgfx_static_maxwell)]
            Self::Maxwell(session) => session
                .release_imported_texture(texture)
                .map_err(Error::ScarletMaxwellIr),
        }
    }

    /// Borrow a mapped presentation image without exposing its backend type.
    ///
    /// # Arguments
    ///
    /// * `target` - Logical presentation texture identity.
    ///
    /// # Returns
    ///
    /// A borrowed image view or mapping error.
    pub fn image(&self, target: ir::TextureId) -> Result<ImageRef<'_>> {
        match self {
            #[cfg(any(
                feature = "backend-scarlet-virgl",
                feature = "backend-scarlet-virgl-static",
                sgfx_dynamic_virgl
            ))]
            Self::Virgl(session) => session
                .image(target)
                .map(|image| ImageRef {
                    backend: Image::Virgl(image),
                })
                .map_err(Error::from),
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            Self::Dynamic(session) => session
                .image(target)
                .map(|image| ImageRef {
                    backend: Image::Dynamic(image),
                })
                .map_err(Error::from),
            #[cfg(feature = "backend-scarlet-adreno")]
            Self::Adreno(session) => session
                .image(target)
                .map(|image| ImageRef {
                    backend: Image::Adreno(image),
                })
                .map_err(Error::ScarletAdrenoIr),
            #[cfg(sgfx_static_maxwell)]
            Self::Maxwell(session) => session
                .image(target)
                .map(|image| ImageRef {
                    backend: Image::Maxwell(image),
                })
                .map_err(Error::ScarletMaxwellIr),
        }
    }

    /// Read one mapped presentation-target rectangle into a BGRA buffer.
    ///
    /// # Arguments
    ///
    /// * `target` - Logical presentation texture identity.
    /// * `destination` - Complete writable BGRA destination buffer.
    /// * `destination_stride` - Bytes between destination rows.
    /// * `rect` - Source target rectangle written at identical destination coordinates.
    ///
    /// # Returns
    ///
    /// Success after synchronous readback, or an error when the selected
    /// backend does not expose image-to-CPU transfer.
    pub fn readback_bgra(
        &self,
        target: ir::TextureId,
        destination: &mut [u8],
        destination_stride: u32,
        rect: ir::PixelRect,
    ) -> Result<()> {
        match self {
            #[cfg(any(
                feature = "backend-scarlet-virgl",
                feature = "backend-scarlet-virgl-static",
                sgfx_dynamic_virgl
            ))]
            Self::Virgl(session) => session
                .readback_bgra(target, destination, destination_stride, rect)
                .map_err(Error::from),
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            Self::Dynamic(session) => session
                .readback_bgra(target, destination, destination_stride, rect)
                .map_err(Error::from),
            #[cfg(feature = "backend-scarlet-adreno")]
            Self::Adreno(session) => session
                .readback_bgra(target, destination, destination_stride, rect)
                .map_err(Error::ScarletAdrenoIr),
            #[cfg(sgfx_static_maxwell)]
            Self::Maxwell(session) => session
                .readback_bgra(target, destination, destination_stride, rect)
                .map_err(Error::ScarletMaxwellIr),
        }
    }

    /// Bind the selected backend queue and resources for command execution.
    ///
    /// # Returns
    ///
    /// A frontend executor delegating complete command buffers to its backend.
    pub fn executor(&mut self) -> Executor<'_> {
        match self {
            #[cfg(any(
                feature = "backend-scarlet-virgl",
                feature = "backend-scarlet-virgl-static",
                sgfx_dynamic_virgl
            ))]
            Self::Virgl(session) => Executor::Virgl(session.executor()),
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            Self::Dynamic(session) => Executor::Dynamic(session.executor()),
            #[cfg(feature = "backend-scarlet-adreno")]
            Self::Adreno(session) => Executor::Adreno(session.executor()),
            #[cfg(sgfx_static_maxwell)]
            Self::Maxwell(session) => Executor::Maxwell(session.executor()),
        }
    }
}

enum Image<'a> {
    #[cfg(any(
        feature = "backend-scarlet-virgl",
        feature = "backend-scarlet-virgl-static",
        sgfx_dynamic_virgl
    ))]
    #[cfg(not(sgfx_dynamic_virgl))]
    Virgl(&'a crate::virgl::Image),
    #[cfg(sgfx_dynamic_virgl)]
    Virgl(crate::dynamic::ImageRef<'a>),
    #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
    Dynamic(crate::dynamic::ImageRef<'a>),
    #[cfg(feature = "backend-scarlet-adreno")]
    Adreno(&'a sgfx_backend_scarlet_adreno::Image),
    #[cfg(sgfx_static_maxwell)]
    Maxwell(&'a crate::maxwell::Image),
}

/// Borrowed Scarlet presentation image exposed by the SGFX frontend.
pub struct ImageRef<'a> {
    backend: Image<'a>,
}

impl ImageRef<'_> {
    /// Return the image width in pixels.
    ///
    /// # Returns
    ///
    /// Physical image width.
    pub fn width(&self) -> u32 {
        match &self.backend {
            #[cfg(any(
                feature = "backend-scarlet-virgl",
                feature = "backend-scarlet-virgl-static",
                sgfx_dynamic_virgl
            ))]
            Image::Virgl(image) => image.width(),
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            Image::Dynamic(image) => image.width(),
            #[cfg(feature = "backend-scarlet-adreno")]
            Image::Adreno(image) => image.width(),
            #[cfg(sgfx_static_maxwell)]
            Image::Maxwell(image) => image.width(),
        }
    }

    /// Return the image height in pixels.
    ///
    /// # Returns
    ///
    /// Physical image height.
    pub fn height(&self) -> u32 {
        match &self.backend {
            #[cfg(any(
                feature = "backend-scarlet-virgl",
                feature = "backend-scarlet-virgl-static",
                sgfx_dynamic_virgl
            ))]
            Image::Virgl(image) => image.height(),
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            Image::Dynamic(image) => image.height(),
            #[cfg(feature = "backend-scarlet-adreno")]
            Image::Adreno(image) => image.height(),
            #[cfg(sgfx_static_maxwell)]
            Image::Maxwell(image) => image.height(),
        }
    }

    /// Borrow the Scarlet shared-image capability.
    ///
    /// # Returns
    ///
    /// Handle retained by the selected backend session.
    pub fn shared_handle(&self) -> &Handle {
        match &self.backend {
            #[cfg(any(
                feature = "backend-scarlet-virgl",
                feature = "backend-scarlet-virgl-static",
                sgfx_dynamic_virgl
            ))]
            Image::Virgl(image) => image.shared_handle(),
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            Image::Dynamic(image) => image.shared_handle(),
            #[cfg(feature = "backend-scarlet-adreno")]
            Image::Adreno(image) => image.shared_handle(),
            #[cfg(sgfx_static_maxwell)]
            Image::Maxwell(image) => image.shared_handle(),
        }
    }
}

/// Scarlet command executor selected by the SGFX frontend.
pub enum Executor<'a> {
    /// A VirGL command executor.
    #[cfg(any(
        feature = "backend-scarlet-virgl",
        feature = "backend-scarlet-virgl-static",
        sgfx_dynamic_virgl
    ))]
    Virgl(crate::virgl::Executor<'a>),
    #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
    Dynamic(crate::dynamic::Executor<'a>),
    /// A native Adreno command executor.
    #[cfg(feature = "backend-scarlet-adreno")]
    Adreno(sgfx_backend_scarlet_adreno::Executor<'a>),
    #[cfg(sgfx_static_maxwell)]
    Maxwell(crate::maxwell::Executor<'a>),
}

impl CommandExecutor for Executor<'_> {
    type Error = Error;

    fn execute<'r, 'data>(&mut self, commands: &ir::CommandBuffer<'r, 'data>) -> Result<()> {
        match self {
            #[cfg(any(
                feature = "backend-scarlet-virgl",
                feature = "backend-scarlet-virgl-static",
                sgfx_dynamic_virgl
            ))]
            Self::Virgl(executor) => executor.execute(commands).map_err(Error::from),
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            Self::Dynamic(executor) => executor.execute(commands).map_err(Error::from),
            #[cfg(feature = "backend-scarlet-adreno")]
            Self::Adreno(executor) => executor.execute(commands).map_err(Error::ScarletAdrenoIr),
            #[cfg(sgfx_static_maxwell)]
            Self::Maxwell(executor) => executor.execute(commands).map_err(Error::ScarletMaxwellIr),
        }
    }
}

/// Owned completion receipt from the selected Scarlet execution backend.
///
/// It does not borrow its session or command data. Dropping it neither waits
/// nor cancels work; kernel-owned retention is independent of this receipt.
/// GPU completion is separate from presentation and SWS buffer release.
#[derive(Debug)]
pub enum Submission {
    /// Completion of all VirGL chunks in one logical submission.
    #[cfg(any(
        feature = "backend-scarlet-virgl",
        feature = "backend-scarlet-virgl-static",
        sgfx_dynamic_virgl
    ))]
    Virgl(crate::virgl::Submission),
    #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
    Dynamic(crate::dynamic::Submission),
    /// Completion of every Adreno chunk and its ordered queue prefix.
    #[cfg(feature = "backend-scarlet-adreno")]
    Adreno(sgfx_backend_scarlet_adreno::Submission),
    #[cfg(sgfx_static_maxwell)]
    Maxwell(crate::maxwell::Submission),
}

impl Completion for Submission {
    type Error = Error;

    /// Observe completion without waiting for the GPU.
    ///
    /// # Returns
    ///
    /// Pending, complete, or the selected backend's observation/execution error.
    fn poll(&self) -> Result<CompletionStatus> {
        match *self {
            #[cfg(any(
                feature = "backend-scarlet-virgl",
                feature = "backend-scarlet-virgl-static",
                sgfx_dynamic_virgl
            ))]
            Self::Virgl(ref receipt) => receipt.poll().map_err(Error::from),
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            Self::Dynamic(ref receipt) => receipt.poll().map_err(Error::from),
            #[cfg(feature = "backend-scarlet-adreno")]
            Self::Adreno(ref receipt) => receipt.poll().map_err(Error::ScarletAdrenoIr),
            #[cfg(sgfx_static_maxwell)]
            Self::Maxwell(ref receipt) => receipt.poll().map_err(Error::ScarletMaxwellIr),
        }
    }

    /// Wait for completion with an optional deadline.
    ///
    /// # Arguments
    ///
    /// * `timeout` - Zero polls, `None` waits without a caller deadline.
    ///
    /// # Returns
    ///
    /// Complete, pending on timeout, or a backend error. Timeout never cancels
    /// work or grants permission to recycle an externally shared buffer.
    fn wait(&self, timeout: Option<Duration>) -> Result<CompletionStatus> {
        #[cfg(not(any(
            any(
                feature = "backend-scarlet-virgl",
                feature = "backend-scarlet-virgl-static",
                sgfx_dynamic_virgl
            ),
            feature = "backend-scarlet-adreno",
            sgfx_static_maxwell
        )))]
        let _ = timeout;
        match *self {
            #[cfg(any(
                feature = "backend-scarlet-virgl",
                feature = "backend-scarlet-virgl-static",
                sgfx_dynamic_virgl
            ))]
            Self::Virgl(ref receipt) => receipt.wait(timeout).map_err(Error::from),
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            Self::Dynamic(ref receipt) => receipt.wait(timeout).map_err(Error::from),
            #[cfg(feature = "backend-scarlet-adreno")]
            Self::Adreno(ref receipt) => receipt.wait(timeout).map_err(Error::ScarletAdrenoIr),
            #[cfg(sgfx_static_maxwell)]
            Self::Maxwell(ref receipt) => receipt.wait(timeout).map_err(Error::ScarletMaxwellIr),
        }
    }
}

impl CommandSubmitter for Executor<'_> {
    type Submission = Submission;

    fn supports_async_submission(&self) -> bool {
        match self {
            #[cfg(any(
                feature = "backend-scarlet-virgl",
                feature = "backend-scarlet-virgl-static",
                sgfx_dynamic_virgl
            ))]
            Self::Virgl(executor) => executor.supports_async_submission(),
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            Self::Dynamic(executor) => executor.supports_async_submission(),
            #[cfg(feature = "backend-scarlet-adreno")]
            Self::Adreno(executor) => executor.supports_async_submission(),
            #[cfg(sgfx_static_maxwell)]
            Self::Maxwell(executor) => executor.supports_async_submission(),
        }
    }

    /// Submit a portable command stream with owned completion observation.
    ///
    /// # Arguments
    ///
    /// * `commands` - Finished logical stream and borrowed upload bytes.
    ///
    /// # Returns
    ///
    /// An owned receipt or a classified rejection/partial failure. Native backends may
    /// synchronize during first-use resource creation, but uploads, copies,
    /// and drawing use the async transport without waiting for completion or
    /// capacity. The receipt covers all native chunks and their ordered queue
    /// prefix, independently of the lifetime of the executor or upload data.
    fn submit<'r, 'data>(
        &mut self,
        commands: &ir::CommandBuffer<'r, 'data>,
    ) -> core::result::Result<Submission, SubmitError<Error, Submission>> {
        #[cfg(not(any(
            any(
                feature = "backend-scarlet-virgl",
                feature = "backend-scarlet-virgl-static",
                sgfx_dynamic_virgl
            ),
            feature = "backend-scarlet-adreno",
            sgfx_static_maxwell
        )))]
        let _ = commands;
        match self {
            #[cfg(any(
                feature = "backend-scarlet-virgl",
                feature = "backend-scarlet-virgl-static",
                sgfx_dynamic_virgl
            ))]
            Self::Virgl(executor) => executor
                .submit(commands)
                .map(Submission::Virgl)
                .map_err(|error| error.map(Error::from, Submission::Virgl)),
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            Self::Dynamic(executor) => executor
                .submit(commands)
                .map(Submission::Dynamic)
                .map_err(|error| error.map(Error::from, Submission::Dynamic)),
            #[cfg(feature = "backend-scarlet-adreno")]
            Self::Adreno(executor) => executor
                .submit(commands)
                .map(Submission::Adreno)
                .map_err(|error| error.map(Error::ScarletAdrenoIr, Submission::Adreno)),
            #[cfg(sgfx_static_maxwell)]
            Self::Maxwell(executor) => executor
                .submit(commands)
                .map(Submission::Maxwell)
                .map_err(|error| error.map(Error::ScarletMaxwellIr, Submission::Maxwell)),
        }
    }
}

#[cfg(test)]
mod tests {
    use gpu_raw::{
        GPU_DEVICE_STATE_READY, GPU_EXECUTION_SUPPORT_MEMORY, GPU_EXECUTION_SUPPORT_QUEUE,
        GPU_RESULT_SUCCESS, GpuQueryInfo,
    };

    use super::select_auto_backend;
    use crate::BackendKind;

    fn ready_info(backend_id: &[u8]) -> GpuQueryInfo {
        let mut info = GpuQueryInfo::new();
        info.result = GPU_RESULT_SUCCESS;
        info.device_state = GPU_DEVICE_STATE_READY;
        info.execution_support = GPU_EXECUTION_SUPPORT_QUEUE | GPU_EXECUTION_SUPPORT_MEMORY;
        info.max_opaque_command_size = 64 * 1024;
        info.backend_id_len = backend_id.len() as u32;
        info.backend_id[..backend_id.len()].copy_from_slice(backend_id);
        info
    }

    #[cfg(any(
        feature = "backend-scarlet-virgl",
        feature = "backend-scarlet-virgl-static",
        sgfx_dynamic_virgl
    ))]
    #[test]
    fn auto_selects_virgl_for_the_virtio_gpu_id() {
        let info = ready_info(crate::virgl::BACKEND_ID);
        assert_eq!(
            select_auto_backend(&info).unwrap(),
            BackendKind::ScarletVirgl
        );
    }

    #[cfg(sgfx_dynamic)]
    #[test]
    fn auto_selects_maxwell_for_ready_gm20b_only() {
        let info = ready_info(b"nvidia-gm20b");
        assert_eq!(
            select_auto_backend(&info).unwrap(),
            BackendKind::ScarletMaxwell
        );
        for id in [b"nvidia-gm20b-next".as_slice(), b"unknown-gpu".as_slice()] {
            assert!(!super::supports_gpu(&ready_info(id), b"nvidia-gm20b"));
        }
        let mut unready = ready_info(b"nvidia-gm20b");
        unready.device_state = 0;
        assert!(!super::supports_gpu(&unready, b"nvidia-gm20b"));
        let mut no_queue = ready_info(b"nvidia-gm20b");
        no_queue.execution_support = GPU_EXECUTION_SUPPORT_MEMORY;
        assert!(!super::supports_gpu(&no_queue, b"nvidia-gm20b"));
    }

    #[cfg(feature = "backend-scarlet-adreno")]
    #[test]
    fn auto_selects_adreno_for_the_qcom_adreno_id() {
        let info = ready_info(sgfx_backend_scarlet_adreno::BACKEND_ID);
        assert_eq!(
            select_auto_backend(&info).unwrap(),
            BackendKind::ScarletAdreno
        );
    }
}

fn open_maxwell(path: &str, gpu: Gpu, info: gpu_raw::GpuQueryInfo) -> Result<Device> {
    #[cfg(sgfx_static_maxwell)]
    {
        let _ = path;
        if !crate::maxwell::Device::supports(&info) {
            return Err(Error::BackendDeviceMismatch(BackendKind::ScarletMaxwell));
        }
        crate::maxwell::Device::from_gpu(gpu, info)
            .map(Device::Maxwell)
            .map_err(Error::ScarletMaxwellHandle)
    }
    #[cfg(all(not(sgfx_static_maxwell), sgfx_dynamic))]
    {
        if !supports_gpu(&info, b"nvidia-gm20b") {
            return Err(Error::BackendDeviceMismatch(BackendKind::ScarletMaxwell));
        }
        drop(gpu);
        open_dynamic(path, Some("scarlet-maxwell"))
    }
    #[cfg(all(not(sgfx_static_maxwell), not(sgfx_dynamic)))]
    {
        let _ = (path, gpu, info);
        Err(Error::BackendUnavailable(BackendKind::ScarletMaxwell))
    }
}
