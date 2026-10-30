use crate::resources::Resources;
use sgfx::{ErrorKind, driver, ir};
use std::rc::Rc;

pub(crate) type BackendError = sgfx::Error;

// Bound retained native resources while coalescing sequential void destruction
// requests. Observing/materializing operations drain the batch before running.
pub(crate) const IMAGE_RETIREMENT_BATCH: usize = 256;

pub(crate) fn backend_failure(error: BackendError) -> ash::vk::Result {
    eprintln!("[SGFX Vulkan] backend failure: {error:?}");
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
    pub(crate) capabilities: driver::Capabilities,
    pub(crate) native_texture_1d: bool,

    pub cache: driver::Resources,
    pub queue: driver::Queue,
    pub recordings: crate::api::Recordings,
    pub in_flight: std::sync::Arc<crate::api::InFlight>,
    pub device_lost: std::sync::Arc<std::sync::atomic::AtomicBool>,
    pub lost: bool,
    pub(crate) retired_images: Vec<crate::images::Image>,
}

impl Runtime {
    pub fn new(
        adapter: driver::Adapter,
        recordings: crate::api::Recordings,
        in_flight: std::sync::Arc<crate::api::InFlight>,
        device_lost: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<Self, ash::vk::Result> {
        let capabilities = adapter.capabilities();
        let native_texture_1d = adapter.info().backend() == sgfx::BackendKind::ScarletVirgl;
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
            capabilities,
            native_texture_1d,
            cache,
            queue,
            recordings,
            in_flight,
            device_lost,
            lost: false,
            retired_images: Vec::new(),
        })
    }
}

impl Runtime {
    pub(crate) fn retire_image(&mut self, image: ash::vk::Image) -> Result<(), ash::vk::Result> {
        if self
            .resources
            .images
            .get(&image)
            .is_none_or(|data| data.swapchain.is_some())
        {
            return Ok(());
        }
        // Reserve before removing live metadata, including its native lease.
        self.retired_images
            .try_reserve(1)
            .map_err(|_| ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY)?;
        self.retired_images
            .push(self.resources.images.remove(&image).expect("checked image"));
        self.resources.image_view_formats.remove(&image);
        Ok(())
    }

    pub(crate) fn flush_retired_images(&mut self) -> Result<(), ash::vk::Result> {
        if self.retired_images.is_empty() {
            return Ok(());
        }
        // Completion observers progress independently of this device worker.
        // Hold every native image lease until all physical releases succeed.
        self.in_flight.wait();
        let cache = &mut self.cache;
        let result = release_image_batch(&self.table, &self.retired_images, |data| {
            #[cfg(target_os = "scarlet")]
            if data.shared.is_some() {
                cache.unmap_presentation_image(data.id);
            }
            cache.release_texture(data.id).map_err(backend_failure)
        });
        if result.is_err() {
            // A physical release prefix is irreversible. Keep its leases and
            // fail the device instead of retrying or recycling logical IDs.
            self.lost = true;
        } else {
            self.retired_images.clear();
        }
        result
    }

    /// Drop an entire idle logical epoch once no live Vulkan object contains
    /// its IDs. Never recycle IDs while any Vulkan resource still owns them.
    pub fn reclaim_idle_resources(&mut self) {
        let sets = &mut self.resources.descriptor_sets;
        self.resources.pushed_set_owners.retain(|handle, owner| {
            if owner.strong_count() == 0 {
                sets.remove(handle);
                false
            } else {
                true
            }
        });
        if !self.retired_images.is_empty()
            || self.resources.has_live_ir_objects()
            || !self.in_flight.is_empty()
        {
            return;
        }
        crate::api::invalidate_resource_recordings(self);
        self.resources.clear_ir_cache();
        let table = Rc::new(ir::ResourceTable::new());
        let cache = match self.device.create_resources(Rc::clone(&table)) {
            Ok(cache) => cache,
            Err(error) => {
                if std::env::var_os("SGFX_VULKAN_TRACE").is_some() {
                    eprintln!("[SGFX Vulkan] idle resource cache recreation failed: {error:?}");
                }
                self.lost = true;
                return;
            }
        };
        self.cache = cache;
        self.table = table;
    }
}

fn release_image_batch(
    table: &ir::ResourceTable,
    images: &[crate::images::Image],
    mut release_backend: impl FnMut(&crate::images::Image) -> Result<(), ash::vk::Result>,
) -> Result<(), ash::vk::Result> {
    // Native releases synchronize the ABI table. Keep its revision fixed for
    // the entire physical batch; interleaving logical releases used to encode,
    // scan and import the whole table again for every individual image.
    for image in images {
        release_backend(image)?;
    }
    for image in images {
        table
            .release_texture(image.id)
            .map_err(crate::resources::failure)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk;

    fn images(table: &ir::ResourceTable, count: usize) -> Vec<crate::images::Image> {
        (0..count)
            .map(|_| crate::images::Image {
                id: table
                    .define_texture(
                        ir::TextureDesc::new(
                            ir::TextureFormat::Bgra8Unorm,
                            ir::Extent2D::new(1, 1).unwrap(),
                            ir::TextureUsage::SAMPLED,
                        )
                        .unwrap(),
                    )
                    .unwrap()
                    .id(),
                format: vk::Format::B8G8R8A8_UNORM,
                extent: vk::Extent3D {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                flags: vk::ImageCreateFlags::empty(),
                usage: vk::ImageUsageFlags::SAMPLED,
                bound: None,
                swapchain: None,
                #[cfg(target_os = "scarlet")]
                shared: None,
            })
            .collect()
    }

    #[test]
    fn image_batch_keeps_one_metadata_revision_through_all_physical_releases() {
        let table = ir::ResourceTable::new();
        let images = images(&table, IMAGE_RETIREMENT_BATCH);
        let revision = table.abi_revision();
        let mut released = Vec::new();
        release_image_batch(&table, &images, |image| {
            assert_eq!(table.abi_revision(), revision);
            assert!(table.texture_ref(image.id).is_ok());
            released.push(image.id);
            Ok(())
        })
        .unwrap();
        assert_eq!(released.len(), images.len());
        assert!(
            images
                .iter()
                .all(|image| table.texture_ref(image.id).is_err())
        );
    }

    #[test]
    fn failed_physical_release_prefix_preserves_logical_ids() {
        let table = ir::ResourceTable::new();
        let images = images(&table, 3);
        let revision = table.abi_revision();
        let mut calls = 0;
        let result = release_image_batch(&table, &images, |_| {
            calls += 1;
            if calls == 2 {
                Err(vk::Result::ERROR_DEVICE_LOST)
            } else {
                Ok(())
            }
        });
        assert_eq!(result, Err(vk::Result::ERROR_DEVICE_LOST));
        assert_eq!(calls, 2);
        assert_eq!(table.abi_revision(), revision);
        assert!(
            images
                .iter()
                .all(|image| table.texture_ref(image.id).is_ok())
        );
    }
}
