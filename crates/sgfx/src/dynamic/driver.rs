//! Low-level API wrapper; resource metadata is synchronized only on mutation.
use super::*;
use core::cell::Cell;

fn api(library: &LoadedBackend) -> Result<&abi::DriverApi> {
    library
        .driver
        .as_ref()
        .ok_or(Error::Dynamic(DynamicError::Status(abi::UNSUPPORTED)))
}
fn buffer(length: u64) -> Result<Vec<u8>> {
    let length =
        usize::try_from(length).map_err(|_| Error::Dynamic(DynamicError::Status(abi::INVALID)))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| Error::Dynamic(DynamicError::Status(abi::OUT_OF_MEMORY)))?;
    bytes.resize(length, 0);
    Ok(bytes)
}
pub(crate) struct IrResources {
    owner: Owner,
    table: Rc<ir::ResourceTable>,
    revision: Cell<u64>,
}
impl IrResources {
    fn sync(&self) -> Result<()> {
        let revision = self.table.abi_revision();
        if revision != self.revision.get() {
            let words = self.table.abi_snapshot().map_err(invalid)?;
            status(unsafe {
                (api(&self.owner.library)?.sync_resources)(
                    self.owner.raw,
                    abi::Span::from_slice(&words),
                )
            })?;
            self.revision.set(revision);
        }
        Ok(())
    }
    pub(crate) fn release_buffer(&mut self, id: ir::BufferId) -> Result<()> {
        let slot = self.table.buffer_ref(id).map_err(invalid)?.slot() as u32;
        self.sync()?;
        status(unsafe { (api(&self.owner.library)?.release_buffer)(self.owner.raw, slot) })
    }
    pub(crate) fn map_image(
        &mut self,
        id: ir::TextureId,
        image: Rc<PresentationImage>,
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.owner.library, &image.owner.library) {
            return Err(Error::ResourceDeviceMismatch);
        }
        let slot = self.table.texture_ref(id).map_err(invalid)?.slot() as u32;
        self.sync()?;
        status(unsafe {
            (api(&self.owner.library)?.map_image)(self.owner.raw, slot, image.owner.raw)
        })
    }
    pub(crate) fn unmap_image(&mut self, id: ir::TextureId) -> Result<()> {
        let slot = self.table.texture_ref(id).map_err(invalid)?.slot() as u32;
        self.sync()?;
        status(unsafe { (api(&self.owner.library)?.unmap_image)(self.owner.raw, slot) })
    }
    fn validate(&self, kind: u32, slot: u32) -> Result<()> {
        self.sync()?;
        status(unsafe { (api(&self.owner.library)?.validate)(self.owner.raw, kind, slot) })
    }
    pub(crate) fn validate_shader_module(&self, id: ir::ShaderModuleId) -> Result<()> {
        self.validate(
            abi::VALIDATE_SHADER,
            self.table.shader_module_ref(id).map_err(invalid)?.slot() as u32,
        )
    }
    pub(crate) fn validate_programmable_render_pipeline(
        &self,
        id: ir::ProgrammableRenderPipelineId,
    ) -> Result<()> {
        self.validate(
            abi::VALIDATE_RENDER_PIPELINE,
            self.table
                .programmable_render_pipeline_ref(id)
                .map_err(invalid)?
                .slot() as u32,
        )
    }
    pub(crate) fn validate_compute_pipeline(&self, id: ir::ComputePipelineId) -> Result<()> {
        self.validate(
            abi::VALIDATE_COMPUTE_PIPELINE,
            self.table.compute_pipeline_ref(id).map_err(invalid)?.slot() as u32,
        )
    }
    pub(crate) fn read_buffer(&self, id: ir::BufferId, offset: u64, size: u64) -> Result<Vec<u8>> {
        let reference = self.table.buffer_ref(id).map_err(invalid)?;
        let descriptor = self.table.buffer(reference).map_err(invalid)?;
        if offset
            .checked_add(size)
            .is_none_or(|end| end > descriptor.size())
        {
            return Err(invalid(ir::Error::OutOfBounds));
        }
        self.sync()?;
        let mut bytes = buffer(size)?;
        status(unsafe {
            (api(&self.owner.library)?.read_buffer)(
                self.owner.raw,
                reference.slot() as u32,
                offset,
                bytes.as_mut_ptr(),
                bytes.len(),
            )
        })?;
        Ok(bytes)
    }
}
pub(crate) struct PresentationImage {
    owner: Owner,
    handle: Handle,
    width: u32,
    height: u32,
}
impl PresentationImage {
    pub(crate) fn width(&self) -> u32 {
        self.width
    }
    pub(crate) fn height(&self) -> u32 {
        self.height
    }
    pub(crate) fn shared_handle(&self) -> &Handle {
        &self.handle
    }
}
impl Context {
    pub(crate) fn create_ir_resources(&self, table: Rc<ir::ResourceTable>) -> Result<IrResources> {
        let library = self.owner.library.clone();
        let driver = api(&library)?;
        let mut raw = core::ptr::null_mut();
        status(unsafe {
            (driver.create_resources)(
                self.owner.raw,
                abi::Span::from_slice(&table.abi_snapshot().map_err(invalid)?),
                &mut raw,
            )
        })?;
        let destroy = driver.drop_resources;
        let owner = Owner::new(library, raw, destroy)?;
        table.enable_abi_commands();
        let revision = Cell::new(table.abi_revision());
        Ok(IrResources {
            owner,
            table,
            revision,
        })
    }
    pub(crate) fn create_queue(&self) -> Result<Queue> {
        let library = self.owner.library.clone();
        let driver = api(&library)?;
        let mut raw = core::ptr::null_mut();
        status(unsafe { (driver.create_queue)(self.owner.raw, &mut raw) })?;
        let destroy = driver.drop_queue;
        Ok(Queue {
            owner: Owner::new(library, raw, destroy)?,
        })
    }
    pub(crate) fn create_shared_image(&self, width: u32, height: u32) -> Result<PresentationImage> {
        let library = self.owner.library.clone();
        let driver = api(&library)?;
        let mut raw = core::ptr::null_mut();
        let mut info = abi::ImageInfo::default();
        status(unsafe {
            (driver.create_image)(self.owner.raw, width, height, &mut raw, &mut info)
        })?;
        let destroy = driver.drop_image;
        let owner = Owner::new(library, raw, destroy)?;
        let handle = unsafe { Handle::from_raw(info.handle) }
            .map_err(|_| invalid(ir::Error::InvalidDescriptor))?;
        if info.reserved != 0 {
            return Err(Error::Dynamic(DynamicError::Status(abi::ABI_MISMATCH)));
        }
        Ok(PresentationImage {
            owner,
            handle,
            width: info.width,
            height: info.height,
        })
    }
    pub(crate) fn read_texture(
        &self,
        resources: &mut IrResources,
        id: ir::TextureId,
    ) -> Result<Vec<u8>> {
        if !Arc::ptr_eq(&self.owner.library, &resources.owner.library) {
            return Err(Error::ResourceDeviceMismatch);
        }
        let reference = resources.table.texture_ref(id).map_err(invalid)?;
        let descriptor = resources.table.texture(reference).map_err(invalid)?;
        let mut bytes = buffer(descriptor.byte_size().map_err(invalid)?)?;
        resources.sync()?;
        status(unsafe {
            (api(&self.owner.library)?.read_texture)(
                self.owner.raw,
                resources.owner.raw,
                reference.slot() as u32,
                bytes.as_mut_ptr(),
                bytes.len(),
            )
        })?;
        Ok(bytes)
    }
}
pub(crate) struct Queue {
    owner: Owner,
}
impl Queue {
    pub(crate) fn submit_ir_async(
        &self,
        _context: &Context,
        resources: &mut IrResources,
        commands: &ir::CommandBuffer<'_, '_>,
    ) -> core::result::Result<Submission, SubmitError<Error, Submission>> {
        let prepare = || -> Result<abi::Batch> {
            if !Arc::ptr_eq(&self.owner.library, &resources.owner.library)
                || !core::ptr::eq(commands.resources(), resources.table.as_ref())
            {
                return Err(Error::ResourceDeviceMismatch);
            }
            resources.sync()?;
            commands
                .abi_batch()
                .ok_or(Error::Dynamic(DynamicError::RecordingMode))
        };
        let batch = prepare().map_err(|e| match e {
            Error::Dynamic(DynamicError::Status(abi::BUSY)) => SubmitError::Busy,
            e => SubmitError::Rejected(e),
        })?;
        let driver = api(&self.owner.library).map_err(SubmitError::Rejected)?;
        let mut result = abi::SubmitResult::default();
        unsafe {
            (driver.submit)(self.owner.raw, resources.owner.raw, &batch, &mut result);
        }
        decode_submission(self.owner.library.clone(), result)
            .map(Submission)
            .map_err(|e| e.map(core::convert::identity, Submission))
    }
}
/// Receipts made through the low-level extension can retain the native Arc
/// without a second allocation on submit or clone.
pub(crate) struct Submission(super::Submission);
impl Clone for Submission {
    fn clone(&self) -> Self {
        let owner = &self.0.0;
        // Creation of low-level queues already negotiated this extension.
        unsafe {
            (owner
                .library
                .driver
                .as_ref()
                .expect("negotiated driver ABI")
                .clone_receipt)(owner.raw);
        }
        Self(super::Submission(Owner {
            library: owner.library.clone(),
            raw: owner.raw,
            destroy: owner.destroy,
            _local: PhantomData,
        }))
    }
}
impl fmt::Debug for Submission {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl Completion for Submission {
    type Error = Error;
    fn poll(&self) -> Result<CompletionStatus> {
        self.0.poll()
    }
    fn wait(&self, timeout: Option<Duration>) -> Result<CompletionStatus> {
        self.0.wait(timeout)
    }
}
