use crate::resources::Resources;
use sgfx::{ErrorKind, driver, ir};
use std::rc::Rc;

pub(crate) type BackendError = sgfx::Error;

pub(crate) fn backend_failure(error: BackendError) -> ash::vk::Result {
    match error.kind() {
        ErrorKind::InvalidInput | ErrorKind::Unsupported => {
            ash::vk::Result::ERROR_FEATURE_NOT_PRESENT
        }
        ErrorKind::OutOfHostMemory => ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY,
        ErrorKind::OutOfDeviceMemory => ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
        ErrorKind::InitializationFailed => ash::vk::Result::ERROR_INITIALIZATION_FAILED,
        ErrorKind::DeviceLost => ash::vk::Result::ERROR_DEVICE_LOST,
    }
}

/// All Rc-backed SGFX state remains on the device worker thread. No unsafe
/// Send/Sync assertion or forged lifetime crosses the FFI/thread boundary.
pub(crate) struct Runtime {
    pub table: Rc<ir::ResourceTable>,
    pub resources: Resources,
    pub(crate) device: driver::Device,

    pub cache: driver::Resources,
    pub queue: driver::Queue,
    pub recordings: crate::api::Recordings,
    pub in_flight: std::sync::Arc<crate::api::InFlight>,
    pub fences: crate::api::Fences,
    pub device_lost: std::sync::Arc<std::sync::atomic::AtomicBool>,
    pub lost: bool,
}

impl Runtime {
    pub fn new(
        adapter: driver::Adapter,
        recordings: crate::api::Recordings,
        in_flight: std::sync::Arc<crate::api::InFlight>,
        fences: crate::api::Fences,
        device_lost: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<Self, ash::vk::Result> {
        let device = adapter.create_device().map_err(backend_failure)?;
        let table = Rc::new(ir::ResourceTable::new());
        let cache = device
            .create_resources(Rc::clone(&table))
            .map_err(backend_failure)?;
        let queue = device.create_queue().map_err(backend_failure)?;
        Ok(Self {
            table,
            resources: Resources::new(),
            device,
            cache,
            queue,
            recordings,
            in_flight,
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
        if self.resources.has_live_ir_objects() || !self.in_flight.is_empty() {
            return;
        }
        crate::api::invalidate_resource_recordings(self);
        self.resources.clear_ir_cache();
        let table = Rc::new(ir::ResourceTable::new());
        let Ok(cache) = self.device.create_resources(Rc::clone(&table)) else {
            self.lost = true;
            return;
        };
        self.cache = cache;
        self.table = table;
    }
}
