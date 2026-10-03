//! Scarlet facade backed exclusively by installed dynamic drivers.
use crate::{BackendKind, Error, Result, ir};
use alloc::{rc::Rc, sync::Arc, vec::Vec};
use core::{fmt, marker::PhantomData, time::Duration};
pub use scarlet_os::handle::{Handle, HandleError};
use sgfx_backend_abi as abi;
use sgfx_backend_loader::{LoadedBackend, discover, driver_directories};
use sgfx_core::backend::{
    CommandExecutor, CommandSubmitter, Completion, CompletionStatus, SubmitError,
};

#[derive(Debug)]
pub enum DynamicError {
    Loader(sgfx_backend_loader::Error),
    Status(i32),
    RecordingMode,
}
impl fmt::Display for DynamicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Loader(e) => write!(f, "{e}"),
            Self::Status(s) => write!(f, "backend status {s}"),
            Self::RecordingMode => f.write_str(concat!(
                "create the dynamic session before recording commands, ",
                "or enable ResourceTable::enable_abi_commands"
            )),
        }
    }
}
fn status(code: i32) -> Result<()> {
    if code == abi::OK {
        Ok(())
    } else {
        Err(Error::Dynamic(DynamicError::Status(code)))
    }
}
fn invalid(_: ir::Error) -> Error {
    Error::Dynamic(DynamicError::Status(abi::INVALID))
}

struct Owner {
    library: Arc<LoadedBackend>,
    raw: abi::Object,
    destroy: unsafe extern "C" fn(abi::Object),
    _local: PhantomData<Rc<()>>,
}
impl Owner {
    fn new(
        library: Arc<LoadedBackend>,
        raw: abi::Object,
        destroy: unsafe extern "C" fn(abi::Object),
    ) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::Dynamic(DynamicError::Status(abi::INVALID)));
        }
        Ok(Self {
            library,
            raw,
            destroy,
            _local: PhantomData,
        })
    }
}
impl Drop for Owner {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.raw) }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Capabilities(u64);
impl Capabilities {
    pub fn supports_programmable_graphics(self) -> bool {
        self.0 & abi::PROGRAMMABLE_GRAPHICS != 0
    }
    pub fn supports_texture_arrays(self) -> bool {
        self.0 & abi::TEXTURE_ARRAYS != 0
    }
    pub fn supports_depth_sampling(self) -> bool {
        self.0 & abi::DEPTH_SAMPLING != 0
    }
    pub fn supports_image_mips(self) -> bool {
        self.0 & abi::IMAGE_MIPS != 0
    }

    pub fn supports_rendering(self) -> bool {
        self.0 & abi::RENDERING != 0
    }
    pub fn supports_presentation(self) -> bool {
        self.0 & abi::PRESENTATION != 0
    }
    pub fn supports_image_upload(self) -> bool {
        self.0 & abi::IMAGE_UPLOAD != 0
    }
    pub fn supports_image_readback(self) -> bool {
        self.0 & abi::IMAGE_READBACK != 0
    }
    pub fn supports_depth(self) -> bool {
        self.0 & abi::DEPTH != 0
    }
}
pub const BACKEND_ID: &[u8] = b"virtio-gpu";
pub struct Device {
    owner: Owner,
    capabilities: Capabilities,
}
impl Device {
    pub fn supports(info: &gpu_raw::GpuQueryInfo) -> bool {
        info.backend_id_bytes() == BACKEND_ID
            && info.device_state == gpu_raw::GPU_DEVICE_STATE_READY
            && info.execution_support
                & (gpu_raw::GPU_EXECUTION_SUPPORT_QUEUE | gpu_raw::GPU_EXECUTION_SUPPORT_MEMORY)
                == gpu_raw::GPU_EXECUTION_SUPPORT_QUEUE | gpu_raw::GPU_EXECUTION_SUPPORT_MEMORY
    }
    pub fn open(path: &str) -> Result<Self> {
        Self::open_with_backend(path, None)
    }
    /// Select an installed driver by manifest name, independent of compiled enums.
    pub fn open_with_backend(path: &str, preference: Option<&str>) -> Result<Self> {
        let gpu = gpu_raw::Gpu::open(path)
            .map_err(|_| Error::Dynamic(DynamicError::Status(abi::INITIALIZATION_FAILED)))?;
        let info = gpu
            .query_info()
            .map_err(|_| Error::Dynamic(DynamicError::Status(abi::INITIALIZATION_FAILED)))?;
        let id = core::str::from_utf8(info.backend_id_bytes())
            .map_err(|_| Error::Dynamic(DynamicError::Status(abi::INVALID)))?;
        let manifest = discover(&driver_directories(), id, preference)
            .map_err(|e| Error::Dynamic(DynamicError::Loader(e)))?;
        let library =
            LoadedBackend::load(&manifest).map_err(|e| Error::Dynamic(DynamicError::Loader(e)))?;
        let mut raw = core::ptr::null_mut();
        let mut flags = 0;
        status(unsafe {
            (library.api.open)(abi::Span::from_slice(path.as_bytes()), &mut raw, &mut flags)
        })?;
        let destroy = library.api.drop_device;
        Ok(Self {
            owner: Owner::new(library, raw, destroy)?,
            capabilities: Capabilities(flags),
        })
    }
    pub fn backend(&self) -> BackendKind {
        match self.backend_name() {
            "scarlet-virgl" => BackendKind::ScarletVirgl,
            "scarlet-adreno" => BackendKind::ScarletAdreno,
            "scarlet-maxwell" => BackendKind::ScarletMaxwell,
            name => {
                BackendKind::Other(crate::BackendName::new(name).expect("validated driver name"))
            }
        }
    }
    pub fn backend_name(&self) -> &str {
        &self.owner.library.name
    }
    /// Path of the library actually loaded for this device.
    pub fn backend_library(&self) -> Option<&str> {
        Some(self.owner.library.library.as_str())
    }
    pub fn capabilities(&self) -> Capabilities {
        self.capabilities
    }
    pub fn create_context(&self) -> Result<Context> {
        let library = self.owner.library.clone();
        let mut raw = core::ptr::null_mut();
        status(unsafe { (library.api.create_context)(self.owner.raw, &mut raw) })?;
        let destroy = library.api.drop_context;
        Ok(Context {
            owner: Owner::new(library, raw, destroy)?,
        })
    }
}
pub struct Context {
    owner: Owner,
}
impl Context {
    pub fn create_mapped_target_session(
        &self,
        resources: Rc<ir::ResourceTable>,
        targets: &[ir::TextureId],
    ) -> Result<MappedTargetSession> {
        let slots = targets
            .iter()
            .map(|id| resources.texture_ref(*id).map(|r| r.slot() as u32))
            .collect::<ir::Result<Vec<_>>>()
            .map_err(invalid)?;
        let library = self.owner.library.clone();
        let mut raw = core::ptr::null_mut();
        let mut capabilities = 0;
        {
            let metadata = resources.abi_snapshot().map_err(invalid)?;
            status(unsafe {
                (library.api.create_session)(
                    self.owner.raw,
                    abi::Span::from_slice(&metadata),
                    abi::Span::from_slice(&slots),
                    &mut raw,
                    &mut capabilities,
                )
            })?;
        }
        let destroy = library.api.drop_session;
        let owner = Owner::new(library, raw, destroy)?;
        let mut images = Vec::new();
        for (id, slot) in targets.iter().zip(slots) {
            let mut info = abi::ImageInfo::default();
            status(unsafe { (owner.library.api.image)(owner.raw, slot, &mut info) })?;
            let handle = unsafe { Handle::from_raw(info.handle) }
                .map_err(|_| Error::Dynamic(DynamicError::Status(abi::INVALID)))?;
            if info.reserved != 0 {
                return Err(Error::Dynamic(DynamicError::Status(abi::ABI_MISMATCH)));
            }
            images.push(Image {
                target: *id,
                width: info.width,
                height: info.height,
                handle,
            });
        }
        resources.enable_abi_commands();
        let revision = resources.abi_revision();
        Ok(MappedTargetSession {
            images,
            owner,
            resources,
            revision,
            capabilities,
        })
    }
}
struct Image {
    target: ir::TextureId,
    width: u32,
    height: u32,
    handle: Handle,
}
pub struct ImageRef<'a>(&'a Image);
impl ImageRef<'_> {
    pub fn width(&self) -> u32 {
        self.0.width
    }
    pub fn height(&self) -> u32 {
        self.0.height
    }
    pub fn shared_handle(&self) -> &Handle {
        &self.0.handle
    }
}
pub struct MappedTargetSession {
    images: Vec<Image>,
    owner: Owner,
    resources: Rc<ir::ResourceTable>,
    revision: u64,
    capabilities: u64,
}
impl MappedTargetSession {
    fn sync(&mut self) -> Result<()> {
        let revision = self.resources.abi_revision();
        if revision != self.revision {
            let metadata = self.resources.abi_snapshot().map_err(invalid)?;
            status(unsafe {
                (self.owner.library.api.sync_resources)(
                    self.owner.raw,
                    abi::Span::from_slice(&metadata),
                )
            })?;
            self.revision = revision;
        }
        Ok(())
    }
    fn batch(&mut self, commands: &ir::CommandBuffer<'_, '_>) -> Result<abi::Batch> {
        if !core::ptr::eq(commands.resources(), self.resources.as_ref()) {
            return Err(Error::ResourceDeviceMismatch);
        }
        let batch = commands
            .abi_batch()
            .ok_or(Error::Dynamic(DynamicError::RecordingMode))?;
        self.sync()?;
        Ok(batch)
    }
    pub fn image(&self, target: ir::TextureId) -> Result<ImageRef<'_>> {
        self.images
            .iter()
            .find(|image| image.target == target)
            .map(ImageRef)
            .ok_or(Error::Dynamic(DynamicError::Status(abi::INVALID)))
    }
    pub fn readback_bgra(
        &self,
        target: ir::TextureId,
        destination: &mut [u8],
        stride: u32,
        rect: ir::PixelRect,
    ) -> Result<()> {
        let slot = self.resources.texture_ref(target).map_err(invalid)?.slot() as u32;
        status(unsafe {
            (self.owner.library.api.readback)(
                self.owner.raw,
                slot,
                destination.as_mut_ptr(),
                destination.len(),
                stride,
                abi::Rect {
                    x: rect.x(),
                    y: rect.y(),
                    width: rect.width(),
                    height: rect.height(),
                },
            )
        })
    }
    pub fn import_shared_bgra_texture(
        &mut self,
        texture: ir::TextureId,
        handle: Handle,
    ) -> Result<()> {
        let slot = self.resources.texture_ref(texture).map_err(invalid)?.slot() as u32;
        self.sync()?;
        let raw = handle.as_raw();
        core::mem::forget(handle);
        status(unsafe { (self.owner.library.api.import_bgra)(self.owner.raw, slot, raw) })
    }
    pub fn import_ycbcr_texture(
        &mut self,
        _texture: ir::TextureId,
        _handle: Handle,
        _conversion: ir::YcbcrConversion,
    ) -> Result<()> {
        Err(Error::Dynamic(DynamicError::Status(abi::UNSUPPORTED)))
    }
    pub fn release_imported_texture(&mut self, texture: ir::TextureId) -> Result<()> {
        let slot = self.resources.texture_ref(texture).map_err(invalid)?.slot() as u32;
        status(unsafe { (self.owner.library.api.release_import)(self.owner.raw, slot) })
    }
    pub fn executor(&mut self) -> Executor<'_> {
        Executor(self)
    }
}
pub struct Executor<'a>(&'a mut MappedTargetSession);
impl CommandExecutor for Executor<'_> {
    type Error = Error;
    fn execute<'r, 'data>(&mut self, commands: &ir::CommandBuffer<'r, 'data>) -> Result<()> {
        let batch = self.0.batch(commands)?;
        status(unsafe { (self.0.owner.library.api.execute)(self.0.owner.raw, &batch) })
    }
}
impl CommandSubmitter for Executor<'_> {
    type Submission = Submission;
    fn supports_async_submission(&self) -> bool {
        self.0.capabilities & abi::ASYNC != 0
    }
    fn submit<'r, 'data>(
        &mut self,
        commands: &ir::CommandBuffer<'r, 'data>,
    ) -> core::result::Result<Submission, SubmitError<Error, Submission>> {
        let batch = self.0.batch(commands).map_err(|e| match e {
            Error::Dynamic(DynamicError::Status(abi::BUSY)) => SubmitError::Busy,
            e => SubmitError::Rejected(e),
        })?;
        let library = self.0.owner.library.clone();
        let mut result = abi::SubmitResult::default();
        unsafe {
            (library.api.submit)(self.0.owner.raw, &batch, &mut result);
        }
        decode_submission(library, result)
    }
}
fn decode_submission(
    library: Arc<LoadedBackend>,
    result: abi::SubmitResult,
) -> core::result::Result<Submission, SubmitError<Error, Submission>> {
    if result.disposition == abi::REJECTED {
        return if result.error == abi::BUSY {
            Err(SubmitError::Busy)
        } else {
            Err(SubmitError::Rejected(Error::Dynamic(DynamicError::Status(
                result.error,
            ))))
        };
    }
    let destroy = library.api.drop_receipt;
    let receipt =
        Submission(Owner::new(library, result.receipt, destroy).map_err(SubmitError::Rejected)?);
    match result.disposition {
        abi::ACCEPTED if result.error == abi::OK => Ok(receipt),
        _ => Err(SubmitError::Failed {
            error: Error::Dynamic(DynamicError::Status(result.error)),
            completion: receipt,
        }),
    }
}

pub struct Submission(Owner);
// SAFETY: ABI v1 requires thread-safe receipt observation. The library is pinned
// and Rust ownership prevents destruction while a shared observer is using it.
unsafe impl Send for Submission {}
unsafe impl Sync for Submission {}
impl fmt::Debug for Submission {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Submission")
            .field("backend", &self.0.library.name)
            .finish_non_exhaustive()
    }
}
impl Completion for Submission {
    type Error = Error;
    fn poll(&self) -> Result<CompletionStatus> {
        self.wait(Some(Duration::ZERO))
    }
    fn wait(&self, timeout: Option<Duration>) -> Result<CompletionStatus> {
        let ns = timeout
            .map(|d| d.as_nanos().min(u128::from(u64::MAX - 1)) as u64)
            .unwrap_or(u64::MAX);
        let mut value = abi::PENDING;
        status(unsafe { (self.0.library.api.wait)(self.0.raw, ns, &mut value) })?;
        match value {
            abi::PENDING => Ok(CompletionStatus::Pending),
            abi::COMPLETE => Ok(CompletionStatus::Complete),
            _ => Err(Error::Dynamic(DynamicError::Status(abi::INVALID))),
        }
    }
}

pub(crate) mod driver;
pub(crate) use driver::{IrResources, PresentationImage, Queue};
