//! Standard KHR_display WSI for Scarlet's primary composited output.
//! This is a full-screen display plane, with SWS providing input and sampling
//! registered GPU images. It is not a private Vulkan loader or a Wayland shim.

use crate::{api::next_id, instance, wsi};
use ash::vk::{self, Handle};
use std::{
    ffi::{CStr, c_char},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

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
#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct SwsBuffer {
    window_id: u32,
    buffer_id: u32,
    generation: u32,
    compositor_epoch: u32,
}
#[repr(C)]
#[derive(Default)]
struct SwsGpuEvent {
    buffer: SwsBuffer,
    commit_serial: u64,
    kind: u32,
    code: u32,
}
#[link(name = "sws_client_c")]
unsafe extern "C" {
    fn sws_get_display(out: *mut SwsDisplay) -> i32;
    fn sws_window_create(
        app_id: *const c_char,
        title: *const c_char,
        width: u32,
        height: u32,
        out: *mut u32,
    ) -> i32;
    fn sws_window_destroy(window_id: u32) -> i32;
    fn sws_window_fullscreen(window_id: u32, enabled: u32) -> i32;
    fn sws_gpu_register(buffer: SwsBuffer, width: u32, height: u32, raw_handle: i32) -> i32;
    fn sws_gpu_commit(buffer: SwsBuffer, serial: u64, width: u32, height: u32) -> i32;
    fn sws_gpu_destroy(buffer: SwsBuffer) -> i32;
    fn sws_gpu_poll(window_id: u32, out: *mut SwsGpuEvent) -> i32;
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

// The native window belongs to VkSurfaceKHR, not a swapchain generation.
// Recreating image storage must preserve SDL input routing and fullscreen state.
pub(crate) struct SurfaceWindow {
    id: u32,
    epoch: u32,
    serial: AtomicU64,
    retired: Mutex<Vec<(SwsBuffer, u64)>>,
    pending: Mutex<Vec<SwsGpuEvent>>,
    lost: AtomicBool,
}
impl SurfaceWindow {
    pub fn new(size: vk::Extent2D) -> Result<Arc<Self>, vk::Result> {
        let display = display()?;
        if size != extent(&display) || display.compositor_epoch == 0 {
            return Err(LOST);
        }
        let mut id = 0;
        if unsafe {
            sws_window_create(
                c"org.scarlet.vulkan.display".as_ptr(),
                c"Vulkan display".as_ptr(),
                size.width,
                size.height,
                &mut id,
            )
        } < 0
        {
            return Err(LOST);
        }
        let surface = Arc::new(Self {
            id,
            epoch: display.compositor_epoch,
            serial: AtomicU64::new(0),
            retired: Mutex::new(Vec::new()),
            pending: Mutex::new(Vec::new()),
            lost: AtomicBool::new(false),
        });
        if unsafe { sws_window_fullscreen(id, 1) } < 0 {
            return Err(LOST);
        }
        Ok(surface)
    }
    fn release_retired(&self, event: &SwsGpuEvent) -> bool {
        let mut retired = self.retired.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(index) = retired
            .iter()
            .position(|&(buffer, serial)| buffer == event.buffer && serial == event.commit_serial)
        {
            if event.kind == 1 || event.kind == 2 {
                let (buffer, _) = retired.swap_remove(index);
                unsafe {
                    sws_gpu_destroy(buffer);
                }
                return true;
            }
        }
        false
    }
    fn events(&self, generation: u32) -> Result<Vec<SwsGpuEvent>, vk::Result> {
        // A surface can have two live swapchains during replacement. Preserve
        // release events belonging to the other generation instead of consuming
        // them from the shared SWS queue and leaving its images permanently busy.
        let mut pending = self.pending.lock().unwrap_or_else(|p| p.into_inner());
        if self.lost.load(Ordering::Relaxed) {
            return Err(LOST);
        }
        loop {
            let mut event = SwsGpuEvent::default();
            let result = unsafe { sws_gpu_poll(self.id, &mut event) };
            if result < 0 || event.kind == 3 {
                self.lost.store(true, Ordering::Relaxed);
                return Err(if event.kind == 3 {
                    vk::Result::ERROR_DEVICE_LOST
                } else {
                    LOST
                });
            }
            if result == 0 {
                break;
            }
            if !self.release_retired(&event) {
                pending.push(event);
            }
        }
        let mut output = Vec::new();
        let mut index = 0;
        while index < pending.len() {
            if pending[index].buffer.generation == generation {
                output.push(pending.remove(index));
            } else {
                index += 1;
            }
        }
        Ok(output)
    }
}
impl Drop for SurfaceWindow {
    fn drop(&mut self) {
        unsafe {
            sws_window_destroy(self.id);
        }
    }
}

pub(crate) struct Window {
    surface: Arc<SurfaceWindow>,
    generation: u32,
    extent: vk::Extent2D,
    buffers: Vec<SwsBuffer>,
    queued: Vec<Option<u64>>,
    lost: bool,
}
impl Window {
    pub fn new(surface: Arc<SurfaceWindow>, size: vk::Extent2D) -> Result<Self, vk::Result> {
        let display = display()?;
        if surface.epoch != display.compositor_epoch {
            return Err(LOST);
        }
        if size != extent(&display) {
            return Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
        }
        let generation =
            u32::try_from(next_id()).map_err(|_| vk::Result::ERROR_OUT_OF_HOST_MEMORY)?;
        Ok(Self {
            surface,
            generation,
            extent: size,
            buffers: Vec::new(),
            queued: Vec::new(),
            lost: false,
        })
    }
    pub fn register(&mut self, image: &sgfx::driver::PresentationImage) -> Result<(), vk::Result> {
        let handle = image
            .duplicate_shared_handle()
            .map_err(crate::runtime::backend_failure)?;
        let buffer = SwsBuffer {
            window_id: self.surface.id,
            buffer_id: self.buffers.len() as u32 + 1,
            generation: self.generation,
            compositor_epoch: self.surface.epoch,
        };
        if unsafe { sws_gpu_register(buffer, image.width(), image.height(), handle.as_raw()) } < 0 {
            return Err(LOST);
        }
        self.buffers.push(buffer);
        self.queued.push(None);
        Ok(())
    }
    pub fn dispatch(&mut self) -> Result<(), vk::Result> {
        if self.lost {
            return Err(LOST);
        }
        for event in self.surface.events(self.generation)? {
            let Some(index) = self
                .buffers
                .iter()
                .position(|buffer| *buffer == event.buffer)
            else {
                continue;
            };
            if self.queued[index] != Some(event.commit_serial) {
                continue;
            }
            match event.kind {
                1 => self.queued[index] = None,
                2 => {
                    self.lost = true;
                    return Err(LOST);
                }
                _ => {
                    self.lost = true;
                    return Err(LOST);
                }
            }
        }
        Ok(())
    }
    pub fn available(&self, index: usize) -> bool {
        !self.lost && self.queued[index].is_none()
    }
    pub fn present(&mut self, index: usize) -> Result<(), vk::Result> {
        self.dispatch()?;
        if !self.available(index) {
            return Err(INVALID);
        }
        let serial = self
            .surface
            .serial
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |serial| {
                serial.checked_add(1)
            })
            .map_err(|_| vk::Result::ERROR_OUT_OF_HOST_MEMORY)?
            + 1;
        if unsafe {
            sws_gpu_commit(
                self.buffers[index],
                serial,
                self.extent.width,
                self.extent.height,
            )
        } < 0
        {
            return Err(LOST);
        }
        self.queued[index] = Some(serial);
        Ok(())
    }
}
impl Drop for Window {
    fn drop(&mut self) {
        let _ = self.dispatch();
        let mut retired = self
            .surface
            .retired
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        for (index, &buffer) in self.buffers.iter().enumerate() {
            if let Some(serial) = self.queued[index] {
                // SWS retains the last displayed image until its replacement.
                // The next swapchain destroys it after observing its release.
                retired.push((buffer, serial));
            } else {
                unsafe {
                    sws_gpu_destroy(buffer);
                }
            }
        }
    }
}
