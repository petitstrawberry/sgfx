use ash::vk;
use sgfx_core::ir;
use std::rc::Rc;

pub(crate) type BackendError = sgfx_backend_wgpu::Error;
pub(crate) type Cache = sgfx_backend_wgpu::Resources;
pub(crate) type Queue = sgfx_backend_wgpu::Queue;

pub(crate) struct Context(sgfx_backend_wgpu::Context);

impl Context {
    pub fn new() -> Result<Self, vk::Result> {
        // Do not select WGPU's Vulkan backend: it can recursively load this ICD.
        let backends = if cfg!(target_os = "macos") {
            wgpu::Backends::METAL
        } else {
            wgpu::Backends::GL
        };
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("SGFX experimental Vulkan ICD"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_compute_workgroup_storage_size: 16384,
                    ..wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits())
                },
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        ))
        .map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?;
        Ok(Self(sgfx_backend_wgpu::Device::new(device, queue).create_context()))
    }

    pub fn create_resources(&self, table: Rc<ir::ResourceTable>) -> Result<Cache, vk::Result> {
        Ok(self.0.create_resources(table))
    }

    pub fn create_queue(&self) -> Result<Queue, vk::Result> {
        Ok(self.0.create_queue())
    }
}

pub(crate) fn backend_failure(error: BackendError) -> vk::Result {
    match error {
        BackendError::DeviceLost => vk::Result::ERROR_DEVICE_LOST,
        BackendError::InvalidIr(error) => crate::resources::failure(error),
        _ => vk::Result::ERROR_FEATURE_NOT_PRESENT,
    }
}
