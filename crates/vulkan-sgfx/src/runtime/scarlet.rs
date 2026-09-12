use ash::vk;
use sgfx_backend_scarlet_virgl as virgl;
use sgfx_core::{backend::SubmitError, ir};
use std::rc::Rc;

pub(crate) type BackendError = virgl::IrSubmitError;

pub(crate) struct Context(Rc<virgl::Context>);

impl Context {
    pub fn new() -> Result<Self, vk::Result> {
        let path = std::env::var("SGFX_VULKAN_GPU").unwrap_or_else(|_| "/dev/gpu0".into());
        let device = virgl::Device::open(&path)
            .map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let context = device.create_context()
            .map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?;
        Ok(Self(Rc::new(context)))
    }

    pub fn create_resources(&self, table: Rc<ir::ResourceTable>) -> Result<Cache, vk::Result> {
        Ok(Cache {
            inner: self.0.create_ir_resources(table).map_err(backend_failure)?,
            context: Rc::clone(&self.0),
        })
    }

    pub fn create_queue(&self) -> Result<Queue, vk::Result> {
        Ok(Queue {
            inner: self.0.create_queue()
                .map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?,
            context: Rc::clone(&self.0),
        })
    }
}

pub(crate) struct Cache {
    inner: virgl::IrResources,
    context: Rc<virgl::Context>,
}

impl Cache {
    pub fn validate_shader_module(&mut self, id: ir::ShaderModuleId) -> Result<(), BackendError> {
        self.inner.validate_shader_module(id)
    }

    pub fn validate_programmable_render_pipeline(
        &mut self,
        id: ir::ProgrammableRenderPipelineId,
    ) -> Result<(), BackendError> {
        self.inner.validate_programmable_render_pipeline(id)
    }

    pub fn validate_compute_pipeline(&mut self, id: ir::ComputePipelineId) -> Result<(), BackendError> {
        self.inner.validate_compute_pipeline(id)
    }

    pub fn read_buffer(
        &mut self,
        id: ir::BufferId,
        offset: u64,
        size: u64,
    ) -> Result<Vec<u8>, BackendError> {
        self.inner.read_buffer(id, offset, size)
    }

    pub fn read_texture(&mut self, id: ir::TextureId) -> Result<Vec<u8>, BackendError> {
        self.context.read_texture(&mut self.inner, id)
    }
}

pub(crate) struct Queue {
    inner: virgl::Queue,
    context: Rc<virgl::Context>,
}

impl Queue {
    pub fn submit_tracked(
        &self,
        cache: &mut Cache,
        commands: &ir::CommandBuffer<'_, '_>,
    ) -> Result<virgl::Submission, SubmitError<BackendError, virgl::Submission>> {
        self.inner.submit_ir_async(&self.context, &mut cache.inner, commands)
    }
}

pub(crate) fn backend_failure(error: BackendError) -> vk::Result {
    match error {
        BackendError::InvalidIr(error) => crate::resources::failure(error),
        BackendError::OutOfMemory => vk::Result::ERROR_OUT_OF_HOST_MEMORY,
        BackendError::CompletionFailed(_)
        | BackendError::CompletionUnavailable
        | BackendError::SubmissionFailed => vk::Result::ERROR_DEVICE_LOST,
        _ => vk::Result::ERROR_FEATURE_NOT_PRESENT,
    }
}
