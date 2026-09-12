use crate::resources::Resources;
use sgfx_core::ir;
use std::rc::Rc;

#[cfg(not(target_os = "scarlet"))]
#[path = "runtime/host.rs"]
mod backend;
#[cfg(target_os = "scarlet")]
#[path = "runtime/scarlet.rs"]
mod backend;

pub(crate) use backend::{BackendError, backend_failure};

/// All Rc-backed SGFX state remains on the device worker thread. No unsafe
/// Send/Sync assertion or forged lifetime crosses the FFI/thread boundary.
pub(crate) struct Runtime {
    pub table: Rc<ir::ResourceTable>,
    pub resources: Resources,
    context: backend::Context,

    pub cache: backend::Cache,
    pub queue: backend::Queue,
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
        let context = backend::Context::new()?;
        let table = Rc::new(ir::ResourceTable::new());
        let cache = context.create_resources(Rc::clone(&table))?;
        let queue = context.create_queue()?;
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
        let Ok(cache) = self.context.create_resources(Rc::clone(&table)) else {
            self.lost = true;
            return;
        };
        self.cache = cache;
        self.table = table;
    }
}
