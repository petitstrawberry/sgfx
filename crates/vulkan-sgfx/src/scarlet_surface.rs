//! Experimental native SWS surface extension, available through Scarlet's Vulkan loader.
//! The application owns the SWS window and keeps it alive through surface destruction.
use crate::{instance, sws_wsi, wsi};
use ash::vk;
use std::ffi::{CStr, c_void};

pub const INSTANCE_EXTENSION_NAME: &CStr = c"VK_SGFX_scarlet_surface";
/// Private fork value, not a Khronos-assigned extension number.
pub const STRUCTURE_TYPE: vk::StructureType = vk::StructureType::from_raw(0x7f530001);
#[repr(C)]
pub struct ScarletSurfaceCreateInfo {
    pub s_type: vk::StructureType,
    pub p_next: *const c_void,
    pub flags: vk::Flags,
    pub window: u32,
}
pub unsafe extern "system" fn create_surface(
    instance: vk::Instance,
    info: *const ScarletSurfaceCreateInfo,
    allocator: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::SurfaceKHR,
) -> vk::Result {
    if out.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    unsafe {
        *out = vk::SurfaceKHR::null();
    }
    if !instance::instance_valid(instance) || info.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let info = unsafe { &*info };
    if info.s_type != STRUCTURE_TYPE
        || !info.p_next.is_null()
        || info.flags != 0
        || !allocator.is_null()
    {
        return vk::Result::ERROR_FEATURE_NOT_PRESENT;
    }
    let window = match sws_wsi::SurfaceWindow::borrow(info.window) {
        Ok(window) => window,
        Err(error) => return error,
    };
    unsafe {
        *out = wsi::insert_platform_surface(instance, window);
    }
    vk::Result::SUCCESS
}
unsafe extern "system" fn presentation_support(
    physical: vk::PhysicalDevice,
    family: u32,
) -> vk::Bool32 {
    if family != 0 || !instance::physical_valid(physical) {
        return vk::FALSE;
    }
    vk::TRUE
}
pub(crate) fn lookup(name: &CStr) -> vk::PFN_vkVoidFunction {
    match name.to_bytes() {
        b"vkCreateScarletSurfaceSGFX" => Some(unsafe {
            std::mem::transmute::<
                unsafe extern "system" fn(
                    vk::Instance,
                    *const ScarletSurfaceCreateInfo,
                    *const vk::AllocationCallbacks<'_>,
                    *mut vk::SurfaceKHR,
                ) -> vk::Result,
                unsafe extern "system" fn(),
            >(create_surface)
        }),
        b"vkGetPhysicalDeviceScarletPresentationSupportSGFX" => Some(unsafe {
            std::mem::transmute::<
                unsafe extern "system" fn(vk::PhysicalDevice, u32) -> vk::Bool32,
                unsafe extern "system" fn(),
            >(presentation_support)
        }),
        _ => None,
    }
}

pub(crate) fn lookup_physical(name: &CStr) -> vk::PFN_vkVoidFunction {
    if name == c"vkGetPhysicalDeviceScarletPresentationSupportSGFX" {
        lookup(name)
    } else {
        None
    }
}
