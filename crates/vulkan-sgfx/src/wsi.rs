//! macOS Vulkan WSI backed by the selected SGFX WGPU device.

use ash::vk::{self, Handle};
use sgfx::ir;
use std::{
    collections::HashMap,
    ffi::{CStr, c_void},
    sync::{Mutex, MutexGuard, OnceLock},
};

use crate::api::{next_id, signal_acquire_sync, wait_queue_semaphores, with_device, with_queue};

const INVALID: vk::Result = vk::Result::ERROR_INITIALIZATION_FAILED;
const UNSUPPORTED: vk::Result = vk::Result::ERROR_FEATURE_NOT_PRESENT;
const MAX_SWAPCHAIN_IMAGES: u32 = 8;

#[derive(Clone, Copy)]
struct Surface {
    instance: usize,
    layer: usize,
}

fn surfaces() -> MutexGuard<'static, HashMap<vk::SurfaceKHR, Surface>> {
    static SURFACES: OnceLock<Mutex<HashMap<vk::SurfaceKHR, Surface>>> = OnceLock::new();
    SURFACES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn surface_for_physical(
    physical: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> Result<Surface, vk::Result> {
    let surface = surfaces().get(&surface).copied().ok_or(INVALID)?;
    if crate::instance::physical_instance(physical) != Some(surface.instance) {
        return Err(INVALID);
    }
    Ok(surface)
}

pub(crate) fn destroy_instance_surfaces(instance: vk::Instance) {
    let instance = instance.as_raw() as usize;
    surfaces().retain(|_, surface| surface.instance != instance);
}

pub(crate) unsafe extern "system" fn create_metal_surface(
    instance: vk::Instance,
    info: *const vk::MetalSurfaceCreateInfoEXT<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    output: *mut vk::SurfaceKHR,
) -> vk::Result {
    if output.is_null() {
        return INVALID;
    }
    unsafe { *output = vk::SurfaceKHR::null() };
    if info.is_null() || !allocator.is_null() || !crate::instance::instance_valid(instance) {
        return INVALID;
    }
    let info = unsafe { &*info };
    if info.s_type != vk::StructureType::METAL_SURFACE_CREATE_INFO_EXT
        || !info.p_next.is_null()
        || !info.flags.is_empty()
        || info.p_layer.is_null()
    {
        return UNSUPPORTED;
    }
    let handle = vk::SurfaceKHR::from_raw(next_id());
    surfaces().insert(
        handle,
        Surface {
            instance: instance.as_raw() as usize,
            layer: info.p_layer.cast::<c_void>() as usize,
        },
    );
    unsafe { *output = handle };
    vk::Result::SUCCESS
}

pub(crate) unsafe extern "system" fn destroy_surface(
    instance: vk::Instance,
    surface: vk::SurfaceKHR,
    _allocator: *const vk::AllocationCallbacks<'_>,
) {
    if surface == vk::SurfaceKHR::null() {
        return;
    }
    let instance = instance.as_raw() as usize;
    let mut registry = surfaces();
    if registry
        .get(&surface)
        .is_some_and(|surface| surface.instance == instance)
    {
        registry.remove(&surface);
    }
}

pub(crate) unsafe extern "system" fn get_physical_device_surface_support(
    physical: vk::PhysicalDevice,
    queue_family: u32,
    surface: vk::SurfaceKHR,
    output: *mut vk::Bool32,
) -> vk::Result {
    if output.is_null() {
        return INVALID;
    }
    unsafe { *output = vk::FALSE };
    if surface_for_physical(physical, surface).is_err() || queue_family != 0 {
        return INVALID;
    }
    unsafe { *output = vk::TRUE };
    vk::Result::SUCCESS
}

pub(crate) unsafe extern "system" fn get_physical_device_surface_capabilities(
    physical: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    output: *mut vk::SurfaceCapabilitiesKHR,
) -> vk::Result {
    if output.is_null() {
        return INVALID;
    }
    unsafe { *output = vk::SurfaceCapabilitiesKHR::default() };
    if surface_for_physical(physical, surface).is_err() {
        return INVALID;
    }
    let Some(adapter) = crate::instance::physical_adapter(physical) else {
        return INVALID;
    };
    let maximum = adapter
        .capabilities()
        .limits()
        .max_image_dimension_2d
        .min(2048);
    unsafe {
        *output = vk::SurfaceCapabilitiesKHR {
            min_image_count: 2,
            max_image_count: MAX_SWAPCHAIN_IMAGES,
            current_extent: vk::Extent2D {
                width: u32::MAX,
                height: u32::MAX,
            },
            min_image_extent: vk::Extent2D {
                width: 1,
                height: 1,
            },
            max_image_extent: vk::Extent2D {
                width: maximum,
                height: maximum,
            },
            max_image_array_layers: 1,
            supported_transforms: vk::SurfaceTransformFlagsKHR::IDENTITY,
            current_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
            supported_composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            supported_usage_flags: vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
        };
    }
    vk::Result::SUCCESS
}

unsafe fn enumerate<T: Copy>(values: &[T], count: *mut u32, output: *mut T) -> vk::Result {
    if count.is_null() {
        return INVALID;
    }
    if output.is_null() {
        unsafe { *count = values.len() as u32 };
        return vk::Result::SUCCESS;
    }
    let capacity = unsafe { *count } as usize;
    let written = capacity.min(values.len());
    for (index, value) in values.iter().take(written).enumerate() {
        unsafe { *output.add(index) = *value };
    }
    unsafe { *count = written as u32 };
    if written < values.len() {
        vk::Result::INCOMPLETE
    } else {
        vk::Result::SUCCESS
    }
}

pub(crate) unsafe extern "system" fn get_physical_device_surface_formats(
    physical: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    count: *mut u32,
    output: *mut vk::SurfaceFormatKHR,
) -> vk::Result {
    if surface_for_physical(physical, surface).is_err() {
        if !count.is_null() {
            unsafe { *count = 0 };
        }
        return INVALID;
    }
    let Some(adapter) = crate::instance::physical_adapter(physical) else {
        return INVALID;
    };
    let capabilities = adapter.capabilities();
    let mut formats = Vec::with_capacity(2);
    if capabilities.supports_bgra8_color_attachment() {
        formats.push(vk::SurfaceFormatKHR {
            format: vk::Format::B8G8R8A8_UNORM,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        });
    }
    if capabilities.supports_rgba8_color_attachment() {
        formats.push(vk::SurfaceFormatKHR {
            format: vk::Format::R8G8B8A8_UNORM,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        });
    }
    unsafe { enumerate(&formats, count, output) }
}

pub(crate) unsafe extern "system" fn get_physical_device_surface_present_modes(
    physical: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    count: *mut u32,
    output: *mut vk::PresentModeKHR,
) -> vk::Result {
    if surface_for_physical(physical, surface).is_err() {
        if !count.is_null() {
            unsafe { *count = 0 };
        }
        return INVALID;
    }
    unsafe { enumerate(&[vk::PresentModeKHR::FIFO], count, output) }
}

pub(crate) struct Swapchain {
    surface: vk::SurfaceKHR,
    format: vk::Format,
    extent: vk::Extent2D,
    images: Vec<vk::Image>,
    physical_images: Vec<sgfx::driver::PresentationImage>,
    acquired: Vec<bool>,
    next_image: usize,
    window: sgfx::driver::WindowContext,
}

fn swapchain_usage(usage: vk::ImageUsageFlags) -> ir::TextureUsage {
    let mut result = ir::TextureUsage::RENDER_ATTACHMENT | ir::TextureUsage::PRESENT;
    if usage.contains(vk::ImageUsageFlags::TRANSFER_SRC) {
        result |= ir::TextureUsage::COPY_SRC;
    }
    if usage.contains(vk::ImageUsageFlags::TRANSFER_DST) {
        result |= ir::TextureUsage::COPY_DST;
    }
    result
}

fn remove_swapchain_images(runtime: &mut crate::runtime::Runtime, images: &[vk::Image]) {
    for image in images {
        if let Some(data) = runtime.resources.images.remove(image) {
            runtime.cache.unmap_presentation_image(data.id);
        }
    }
}

pub(crate) unsafe extern "system" fn create_swapchain(
    device: vk::Device,
    info: *const vk::SwapchainCreateInfoKHR<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    output: *mut vk::SwapchainKHR,
) -> vk::Result {
    if output.is_null() {
        return INVALID;
    }
    unsafe { *output = vk::SwapchainKHR::null() };
    if info.is_null() || !allocator.is_null() {
        return INVALID;
    }
    let info = unsafe { &*info };
    let Some(surface) = surfaces().get(&info.surface).copied() else {
        return vk::Result::ERROR_SURFACE_LOST_KHR;
    };
    let supported_usage = crate::images::image_usage(info.image_format);
    if info.s_type != vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR
        || !info.p_next.is_null()
        || !info.flags.is_empty()
        || !(2..=MAX_SWAPCHAIN_IMAGES).contains(&info.min_image_count)
        || !matches!(
            info.image_format,
            vk::Format::B8G8R8A8_UNORM | vk::Format::R8G8B8A8_UNORM
        )
        || info.image_color_space != vk::ColorSpaceKHR::SRGB_NONLINEAR
        || info.image_extent.width == 0
        || info.image_extent.height == 0
        || info.image_extent.width > 2048
        || info.image_extent.height > 2048
        || info.image_array_layers != 1
        || info.image_usage.is_empty()
        || !supported_usage.contains(info.image_usage)
        || info.image_sharing_mode != vk::SharingMode::EXCLUSIVE
        || info.queue_family_index_count != 0
        || !info.p_queue_family_indices.is_null()
        || info.pre_transform != vk::SurfaceTransformFlagsKHR::IDENTITY
        || info.composite_alpha != vk::CompositeAlphaFlagsKHR::OPAQUE
        || info.present_mode != vk::PresentModeKHR::FIFO
    {
        return UNSUPPORTED;
    }
    let image_count = info.min_image_count;
    let format = info.image_format;
    let extent = info.image_extent;
    let surface_handle = info.surface;
    let usage = swapchain_usage(info.image_usage);
    let old_swapchain = info.old_swapchain;
    let swapchain = vk::SwapchainKHR::from_raw(next_id());
    let result = with_device(device, move |runtime| {
        if old_swapchain != vk::SwapchainKHR::null()
            && runtime
                .resources
                .swapchains
                .get(&old_swapchain)
                .is_none_or(|old| old.surface != surface_handle)
        {
            return Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
        }
        let ir_format = crate::images::texture_format(format).ok_or(UNSUPPORTED)?;
        // SAFETY: the Vulkan surface contract keeps the CAMetalLayer alive.
        let window = unsafe {
            runtime.device.create_metal_window_context(
                surface.layer as *mut c_void,
                extent.width,
                extent.height,
                false,
            )
        }
        .map_err(crate::runtime::backend_failure)?;
        let size = ir::Extent2D::new(extent.width, extent.height).map_err(|_| INVALID)?;
        let mut images = Vec::with_capacity(image_count as usize);
        let mut physical_images = Vec::with_capacity(image_count as usize);
        for _ in 0..image_count {
            let id = match runtime
                .table
                .define_texture(ir::TextureDesc::new(ir_format, size, usage).map_err(|_| INVALID)?)
                .map_err(crate::resources::failure)
            {
                Ok(reference) => reference.id(),
                Err(error) => {
                    remove_swapchain_images(runtime, &images);
                    return Err(error);
                }
            };
            let physical = match runtime.device.create_presentation_image(
                extent.width,
                extent.height,
                ir_format,
            ) {
                Ok(image) => image,
                Err(error) => {
                    remove_swapchain_images(runtime, &images);
                    return Err(crate::runtime::backend_failure(error));
                }
            };
            if let Err(error) = runtime.cache.map_presentation_image(id, &physical) {
                remove_swapchain_images(runtime, &images);
                return Err(crate::runtime::backend_failure(error));
            }
            let image = vk::Image::from_raw(next_id());
            runtime.resources.images.insert(
                image,
                crate::images::Image {
                    id,
                    format,
                    extent: vk::Extent3D {
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                    },
                    bound: None,
                    swapchain: Some(swapchain),
                },
            );
            images.push(image);
            physical_images.push(physical);
        }
        runtime.resources.swapchains.insert(
            swapchain,
            Swapchain {
                surface: surface_handle,
                format,
                extent,
                acquired: vec![false; images.len()],
                images,
                physical_images,
                next_image: 0,
                window,
            },
        );
        Ok(())
    });
    match result {
        Ok(()) => {
            unsafe { *output = swapchain };
            vk::Result::SUCCESS
        }
        Err(error) => error,
    }
}

pub(crate) unsafe extern "system" fn destroy_swapchain(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    _allocator: *const vk::AllocationCallbacks<'_>,
) {
    if swapchain == vk::SwapchainKHR::null() {
        return;
    }
    let _ = with_device(device, move |runtime| {
        if let Some(swapchain) = runtime.resources.swapchains.remove(&swapchain) {
            remove_swapchain_images(runtime, &swapchain.images);
        }
        Ok(())
    });
}

pub(crate) unsafe extern "system" fn get_swapchain_images(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    count: *mut u32,
    output: *mut vk::Image,
) -> vk::Result {
    if count.is_null() {
        return INVALID;
    }
    match with_device(device, move |runtime| {
        runtime
            .resources
            .swapchains
            .get(&swapchain)
            .map(|swapchain| swapchain.images.clone())
            .ok_or(INVALID)
    }) {
        Ok(images) => unsafe { enumerate(&images, count, output) },
        Err(error) => {
            unsafe { *count = 0 };
            error
        }
    }
}

pub(crate) unsafe extern "system" fn acquire_next_image(
    device: vk::Device,
    swapchain: vk::SwapchainKHR,
    timeout: u64,
    semaphore: vk::Semaphore,
    fence: vk::Fence,
    output: *mut u32,
) -> vk::Result {
    if output.is_null() {
        return INVALID;
    }
    unsafe { *output = u32::MAX };
    let index = match with_device(device, move |runtime| {
        let swapchain = runtime
            .resources
            .swapchains
            .get_mut(&swapchain)
            .ok_or(vk::Result::ERROR_OUT_OF_DATE_KHR)?;
        for offset in 0..swapchain.images.len() {
            let index = (swapchain.next_image + offset) % swapchain.images.len();
            if !swapchain.acquired[index] {
                swapchain.acquired[index] = true;
                swapchain.next_image = (index + 1) % swapchain.images.len();
                return Ok(index as u32);
            }
        }
        Err(if timeout == 0 {
            vk::Result::NOT_READY
        } else {
            vk::Result::TIMEOUT
        })
    }) {
        Ok(index) => index,
        Err(error) => return error,
    };
    if let Err(error) = signal_acquire_sync(device, semaphore, fence) {
        let _ = with_device(device, move |runtime| {
            if let Some(swapchain) = runtime.resources.swapchains.get_mut(&swapchain) {
                swapchain.acquired[index as usize] = false;
            }
            Ok(())
        });
        return error;
    }
    unsafe { *output = index };
    vk::Result::SUCCESS
}

pub(crate) unsafe extern "system" fn queue_present(
    queue: vk::Queue,
    info: *const vk::PresentInfoKHR<'_>,
) -> vk::Result {
    if info.is_null() {
        return INVALID;
    }
    let info = unsafe { &*info };
    if info.s_type != vk::StructureType::PRESENT_INFO_KHR
        || !info.p_next.is_null()
        || info.swapchain_count == 0
        || info.swapchain_count > MAX_SWAPCHAIN_IMAGES
        || info.wait_semaphore_count > 64
        || info.p_swapchains.is_null()
        || info.p_image_indices.is_null()
        || (info.wait_semaphore_count != 0 && info.p_wait_semaphores.is_null())
    {
        return UNSUPPORTED;
    }
    let waits = unsafe {
        std::slice::from_raw_parts(info.p_wait_semaphores, info.wait_semaphore_count as usize)
    }
    .to_vec();
    let swapchains =
        unsafe { std::slice::from_raw_parts(info.p_swapchains, info.swapchain_count as usize) }
            .to_vec();
    let indices =
        unsafe { std::slice::from_raw_parts(info.p_image_indices, info.swapchain_count as usize) }
            .to_vec();
    if let Err(error) = wait_queue_semaphores(queue, &waits) {
        return error;
    }
    let presented = with_queue(queue, move |runtime| {
        let mut results = Vec::with_capacity(swapchains.len());
        for (&handle, &index) in swapchains.iter().zip(&indices) {
            let result = (|| {
                let swapchain = runtime
                    .resources
                    .swapchains
                    .get_mut(&handle)
                    .ok_or(vk::Result::ERROR_OUT_OF_DATE_KHR)?;
                let index = index as usize;
                if index >= swapchain.images.len() || !swapchain.acquired[index] {
                    return Err(INVALID);
                }
                debug_assert_eq!(
                    crate::images::texture_format(swapchain.format),
                    Some(swapchain.physical_images[index].image_format())
                );
                debug_assert_eq!(
                    swapchain.extent,
                    vk::Extent2D {
                        width: swapchain.physical_images[index].width(),
                        height: swapchain.physical_images[index].height(),
                    }
                );
                let result = swapchain
                    .window
                    .present(&swapchain.physical_images[index])
                    .map_err(present_error);
                if result.is_ok() {
                    swapchain.acquired[index] = false;
                }
                result
            })();
            results.push(result.err().unwrap_or(vk::Result::SUCCESS));
        }
        Ok(results)
    });
    let results = match presented {
        Ok(results) => results,
        Err(error) => return error,
    };
    if !info.p_results.is_null() {
        for (index, result) in results.iter().enumerate() {
            unsafe { *info.p_results.add(index) = *result };
        }
    }
    results
        .into_iter()
        .find(|result| *result != vk::Result::SUCCESS)
        .unwrap_or(vk::Result::SUCCESS)
}

fn present_error(error: sgfx::Error) -> vk::Result {
    match crate::runtime::backend_failure(error) {
        vk::Result::ERROR_OUT_OF_HOST_MEMORY => vk::Result::ERROR_OUT_OF_HOST_MEMORY,
        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
        vk::Result::ERROR_DEVICE_LOST => vk::Result::ERROR_DEVICE_LOST,
        _ => vk::Result::ERROR_OUT_OF_DATE_KHR,
    }
}

pub(crate) fn lookup_physical(name: &CStr) -> vk::PFN_vkVoidFunction {
    macro_rules! entry {
        ($function:ident, $signature:ty) => {{
            let function: $signature = $function;
            Some(unsafe {
                std::mem::transmute::<$signature, unsafe extern "system" fn()>(function)
            })
        }};
    }
    match name.to_bytes() {
        b"vkGetPhysicalDeviceSurfaceSupportKHR" => entry!(
            get_physical_device_surface_support,
            vk::PFN_vkGetPhysicalDeviceSurfaceSupportKHR
        ),
        b"vkGetPhysicalDeviceSurfaceCapabilitiesKHR" => entry!(
            get_physical_device_surface_capabilities,
            vk::PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR
        ),
        b"vkGetPhysicalDeviceSurfaceFormatsKHR" => entry!(
            get_physical_device_surface_formats,
            vk::PFN_vkGetPhysicalDeviceSurfaceFormatsKHR
        ),
        b"vkGetPhysicalDeviceSurfacePresentModesKHR" => entry!(
            get_physical_device_surface_present_modes,
            vk::PFN_vkGetPhysicalDeviceSurfacePresentModesKHR
        ),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_surface_reports_a_bounded_fifo_swapchain_contract() {
        unsafe {
            let extensions = [
                vk::KHR_SURFACE_NAME.as_ptr(),
                vk::EXT_METAL_SURFACE_NAME.as_ptr(),
            ];
            let create = vk::InstanceCreateInfo::default().enabled_extension_names(&extensions);
            let mut instance = vk::Instance::null();
            assert_eq!(
                crate::instance::create_instance(&create, std::ptr::null(), &mut instance,),
                vk::Result::SUCCESS
            );
            let layer = 1usize as *const vk::CAMetalLayer;
            let info = vk::MetalSurfaceCreateInfoEXT::default().layer(layer);
            let mut surface = vk::SurfaceKHR::null();
            assert_eq!(
                create_metal_surface(instance, &info, std::ptr::null(), &mut surface),
                vk::Result::SUCCESS
            );
            let mut count = 0;
            assert_eq!(
                crate::instance::enumerate_physical_devices(
                    instance,
                    &mut count,
                    std::ptr::null_mut(),
                ),
                vk::Result::SUCCESS
            );
            if count != 0 {
                let mut physicals = vec![vk::PhysicalDevice::null(); count as usize];
                assert_eq!(
                    crate::instance::enumerate_physical_devices(
                        instance,
                        &mut count,
                        physicals.as_mut_ptr(),
                    ),
                    vk::Result::SUCCESS
                );
                let physical = physicals[0];
                let mut supported = vk::FALSE;
                assert_eq!(
                    get_physical_device_surface_support(physical, 0, surface, &mut supported),
                    vk::Result::SUCCESS
                );
                assert_eq!(supported, vk::TRUE);
                let mut capabilities = vk::SurfaceCapabilitiesKHR::default();
                assert_eq!(
                    get_physical_device_surface_capabilities(physical, surface, &mut capabilities,),
                    vk::Result::SUCCESS
                );
                assert_eq!(capabilities.min_image_count, 2);
                assert!(
                    capabilities
                        .supported_usage_flags
                        .contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                );
                let mut formats = 0;
                assert_eq!(
                    get_physical_device_surface_formats(
                        physical,
                        surface,
                        &mut formats,
                        std::ptr::null_mut(),
                    ),
                    vk::Result::SUCCESS
                );
                assert_ne!(formats, 0);
                let mut modes = 0;
                assert_eq!(
                    get_physical_device_surface_present_modes(
                        physical,
                        surface,
                        &mut modes,
                        std::ptr::null_mut(),
                    ),
                    vk::Result::SUCCESS
                );
                assert_eq!(modes, 1);
            }
            destroy_surface(instance, surface, std::ptr::null());
            crate::instance::destroy_instance(instance, std::ptr::null());
        }
    }
}
