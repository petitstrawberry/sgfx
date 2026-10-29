//! Standard KHR_display WSI for Scarlet's primary composited output.
//! This is a full-screen display plane, with SWS providing input and sampling
//! registered GPU images. It is not a private Vulkan loader or a Wayland shim.

use crate::{api::next_id, instance, wsi};
use ash::vk::{self, Handle};
use std::ffi::CStr;

const INVALID: vk::Result = vk::Result::ERROR_INITIALIZATION_FAILED;
const LOST: vk::Result = vk::Result::ERROR_SURFACE_LOST_KHR;

#[repr(C)]
#[derive(Default)]
struct SwsDisplay {
    width: u32,
    height: u32,
    compositor_epoch: u32,
    compositor_backend: u32,
    capabilities: u64,
}
#[link(name = "sws_client_c")]
unsafe extern "C" {
    fn sws_get_display(out: *mut SwsDisplay) -> i32;
}
fn display() -> Result<SwsDisplay, vk::Result> {
    let mut display = SwsDisplay::default();
    if unsafe { sws_get_display(&mut display) } < 0 || display.width == 0 || display.height == 0 {
        return Err(LOST);
    }
    Ok(display)
}
pub(crate) fn current_extent() -> Result<vk::Extent2D, vk::Result> {
    display().map(|display| extent(&display))
}
fn extent(display: &SwsDisplay) -> vk::Extent2D {
    vk::Extent2D {
        width: display.width,
        height: display.height,
    }
}
fn mode(physical: vk::PhysicalDevice) -> vk::DisplayModeKHR {
    vk::DisplayModeKHR::from_raw(physical.as_raw())
}
fn output(physical: vk::PhysicalDevice) -> vk::DisplayKHR {
    vk::DisplayKHR::from_raw(physical.as_raw())
}

unsafe extern "system" fn get_display_properties(
    physical: vk::PhysicalDevice,
    count: *mut u32,
    out: *mut vk::DisplayPropertiesKHR<'_>,
) -> vk::Result {
    if !instance::physical_valid(physical) {
        return INVALID;
    }
    let display = match display() {
        Ok(display) => display,
        Err(error) => return error,
    };
    unsafe {
        wsi::enumerate(
            &[vk::DisplayPropertiesKHR::default()
                .display(output(physical))
                .display_name(c"Scarlet SWS primary output")
                .physical_dimensions(vk::Extent2D::default())
                .physical_resolution(extent(&display))
                .supported_transforms(vk::SurfaceTransformFlagsKHR::IDENTITY)
                .plane_reorder_possible(false)
                .persistent_content(false)],
            count,
            out,
        )
    }
}
unsafe extern "system" fn get_plane_properties(
    physical: vk::PhysicalDevice,
    count: *mut u32,
    out: *mut vk::DisplayPlanePropertiesKHR,
) -> vk::Result {
    if !instance::physical_valid(physical) {
        return INVALID;
    }
    unsafe {
        wsi::enumerate(
            &[vk::DisplayPlanePropertiesKHR {
                current_display: output(physical),
                current_stack_index: 0,
            }],
            count,
            out,
        )
    }
}
unsafe extern "system" fn get_plane_displays(
    physical: vk::PhysicalDevice,
    plane: u32,
    count: *mut u32,
    out: *mut vk::DisplayKHR,
) -> vk::Result {
    if !instance::physical_valid(physical) || plane != 0 {
        return INVALID;
    }
    unsafe { wsi::enumerate(&[output(physical)], count, out) }
}
unsafe extern "system" fn get_mode_properties(
    physical: vk::PhysicalDevice,
    handle: vk::DisplayKHR,
    count: *mut u32,
    out: *mut vk::DisplayModePropertiesKHR,
) -> vk::Result {
    if !instance::physical_valid(physical) || handle != output(physical) {
        return INVALID;
    }
    let display = match display() {
        Ok(display) => display,
        Err(error) => return error,
    };
    unsafe {
        wsi::enumerate(
            &[vk::DisplayModePropertiesKHR {
                display_mode: mode(physical),
                parameters: vk::DisplayModeParametersKHR {
                    visible_region: extent(&display),
                    refresh_rate: 60_000,
                },
            }],
            count,
            out,
        )
    }
}
unsafe extern "system" fn create_mode(
    physical: vk::PhysicalDevice,
    handle: vk::DisplayKHR,
    info: *const vk::DisplayModeCreateInfoKHR<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::DisplayModeKHR,
) -> vk::Result {
    if info.is_null()
        || out.is_null()
        || !allocator.is_null()
        || !instance::physical_valid(physical)
        || handle != output(physical)
    {
        return INVALID;
    }
    unsafe {
        out.write(vk::DisplayModeKHR::null());
    }
    let info = unsafe { &*info };
    let display = match display() {
        Ok(display) => display,
        Err(error) => return error,
    };
    if info.s_type != vk::StructureType::DISPLAY_MODE_CREATE_INFO_KHR
        || !info.p_next.is_null()
        || !info.flags.is_empty()
        || info.parameters.visible_region != extent(&display)
        || info.parameters.refresh_rate != 60_000
    {
        return vk::Result::ERROR_FEATURE_NOT_PRESENT;
    }
    unsafe {
        out.write(mode(physical));
    }
    vk::Result::SUCCESS
}
unsafe extern "system" fn get_plane_capabilities(
    physical: vk::PhysicalDevice,
    handle: vk::DisplayModeKHR,
    plane: u32,
    out: *mut vk::DisplayPlaneCapabilitiesKHR,
) -> vk::Result {
    if out.is_null()
        || plane != 0
        || !instance::physical_valid(physical)
        || handle != mode(physical)
    {
        return INVALID;
    }
    let display = match display() {
        Ok(display) => display,
        Err(error) => return error,
    };
    unsafe {
        out.write(vk::DisplayPlaneCapabilitiesKHR {
            supported_alpha: vk::DisplayPlaneAlphaFlagsKHR::OPAQUE,
            min_src_position: vk::Offset2D::default(),
            max_src_position: vk::Offset2D::default(),
            min_src_extent: extent(&display),
            max_src_extent: extent(&display),
            min_dst_position: vk::Offset2D::default(),
            max_dst_position: vk::Offset2D::default(),
            min_dst_extent: extent(&display),
            max_dst_extent: extent(&display),
        });
    }
    vk::Result::SUCCESS
}
unsafe extern "system" fn create_surface(
    instance_handle: vk::Instance,
    info: *const vk::DisplaySurfaceCreateInfoKHR<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::SurfaceKHR,
) -> vk::Result {
    if info.is_null()
        || out.is_null()
        || !allocator.is_null()
        || !instance::instance_valid(instance_handle)
    {
        return INVALID;
    }
    unsafe {
        out.write(vk::SurfaceKHR::null());
    }
    let info = unsafe { &*info };
    let physical = vk::PhysicalDevice::from_raw(info.display_mode.as_raw());
    let display = match display() {
        Ok(display) => display,
        Err(error) => return error,
    };
    if info.s_type != vk::StructureType::DISPLAY_SURFACE_CREATE_INFO_KHR
        || !info.p_next.is_null()
        || !info.flags.is_empty()
        || instance::physical_instance(physical) != Some(instance_handle.as_raw() as usize)
        || info.plane_index != 0
        || info.plane_stack_index != 0
        || info.transform != vk::SurfaceTransformFlagsKHR::IDENTITY
        || info.alpha_mode != vk::DisplayPlaneAlphaFlagsKHR::OPAQUE
        || info.image_extent != extent(&display)
    {
        return vk::Result::ERROR_FEATURE_NOT_PRESENT;
    }
    match wsi::insert_display_surface(instance_handle, extent(&display)) {
        Ok(surface) => {
            unsafe {
                out.write(surface);
            }
            vk::Result::SUCCESS
        }
        Err(error) => error,
    }
}

pub(crate) fn lookup(name: &CStr) -> vk::PFN_vkVoidFunction {
    macro_rules! entry {
        ($function:path, $signature:ty) => {{
            let function: $signature = $function;
            Some(unsafe {
                std::mem::transmute::<$signature, unsafe extern "system" fn()>(function)
            })
        }};
    }
    match name.to_bytes() {
        b"vkGetPhysicalDeviceDisplayPropertiesKHR" => entry!(
            get_display_properties,
            vk::PFN_vkGetPhysicalDeviceDisplayPropertiesKHR
        ),
        b"vkGetPhysicalDeviceDisplayPlanePropertiesKHR" => entry!(
            get_plane_properties,
            vk::PFN_vkGetPhysicalDeviceDisplayPlanePropertiesKHR
        ),
        b"vkGetDisplayPlaneSupportedDisplaysKHR" => entry!(
            get_plane_displays,
            vk::PFN_vkGetDisplayPlaneSupportedDisplaysKHR
        ),
        b"vkGetDisplayModePropertiesKHR" => {
            entry!(get_mode_properties, vk::PFN_vkGetDisplayModePropertiesKHR)
        }
        b"vkCreateDisplayModeKHR" => entry!(create_mode, vk::PFN_vkCreateDisplayModeKHR),
        b"vkGetDisplayPlaneCapabilitiesKHR" => entry!(
            get_plane_capabilities,
            vk::PFN_vkGetDisplayPlaneCapabilitiesKHR
        ),
        b"vkCreateDisplayPlaneSurfaceKHR" => {
            entry!(create_surface, vk::PFN_vkCreateDisplayPlaneSurfaceKHR)
        }
        _ => None,
    }
}

pub(crate) use crate::sws_wsi::{SurfaceWindow, Window};
