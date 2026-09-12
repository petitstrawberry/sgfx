use crate::resources::Resources;
use sgfx_core::ir;
use std::rc::Rc;

/// All Rc-backed SGFX state remains on the device worker thread. No unsafe
/// Send/Sync assertion or forged lifetime crosses the FFI/thread boundary.
pub(crate) struct Runtime {
    pub table: Rc<ir::ResourceTable>,
    pub resources: Resources,
    context: sgfx_backend_wgpu::Context,

    pub cache: sgfx_backend_wgpu::Resources,
    pub queue: sgfx_backend_wgpu::Queue,
    pub commands: std::collections::HashMap<u64, crate::api::Recording>,
    pub pools: std::collections::HashMap<u64, ash::vk::CommandPoolCreateFlags>,
    pub fences: crate::api::Fences,
    pub device_lost: std::sync::Arc<std::sync::atomic::AtomicBool>,
    pub lost: bool,
}

impl Runtime {
    pub fn new(
        fences: crate::api::Fences,
        device_lost: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<Self, ash::vk::Result> {
        // Never use Backends::all(), from_env(), or WGPU's Vulkan backend here:
        // recursively loading this ICD would deadlock in the Vulkan loader.
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
        .ok_or(ash::vk::Result::ERROR_INITIALIZATION_FAILED)?;
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
        .map_err(|_| ash::vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let context = sgfx_backend_wgpu::Device::new(device, queue).create_context();
        let table = Rc::new(ir::ResourceTable::new());
        let cache = context.create_resources(Rc::clone(&table));
        let queue = context.create_queue();
        Ok(Self {
            table,
            resources: Resources::new(),
            context,
            cache,
            queue,
            commands: Default::default(),
            pools: Default::default(),
            fences,
            device_lost,
            lost: false,
        })
    }
}

impl Runtime {
    /// Drop an entire idle logical epoch once no live Vulkan object contains
    /// its IDs. Never recycle IDs while any Vulkan resource still owns them.
    pub fn reclaim_idle_resources(&mut self) {
        if self.resources.has_live_ir_objects() {
            return;
        }
        crate::api::invalidate_resource_recordings(self);
        self.resources.clear_ir_cache();
        let table = Rc::new(ir::ResourceTable::new());
        self.cache = self.context.create_resources(Rc::clone(&table));
        self.table = table;
    }
}
