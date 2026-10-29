//! Shared SWS presentation and buffer retirement for native surfaces and KHR_display.

use crate::api::next_id;
use ash::vk;
#[cfg(target_os = "linux")]
use std::ffi::c_char;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
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
#[cfg(target_os = "linux")]
#[link(name = "sws_client_c")]
unsafe extern "C" {
    fn sws_get_display(out: *mut SwsDisplay) -> i32;
    #[cfg(target_os = "linux")]
    fn sws_window_create(
        app_id: *const c_char,
        title: *const c_char,
        width: u32,
        height: u32,
        out: *mut u32,
    ) -> i32;
    fn sws_window_destroy(window_id: u32) -> i32;
    fn sws_window_size(window_id: u32, width: *mut u32, height: *mut u32) -> i32;
    #[cfg(target_os = "linux")]
    fn sws_window_fullscreen(window_id: u32, enabled: u32) -> i32;
    fn sws_gpu_register(buffer: SwsBuffer, width: u32, height: u32, raw_handle: i32) -> i32;
    fn sws_gpu_commit(buffer: SwsBuffer, serial: u64, width: u32, height: u32) -> i32;
    fn sws_gpu_destroy(buffer: SwsBuffer) -> i32;
    fn sws_gpu_poll(window_id: u32, out: *mut SwsGpuEvent) -> i32;
}
#[cfg(target_os = "scarlet")]
mod native {
    use super::*;
    use std::{
        ffi::{CStr, c_char, c_void},
        sync::OnceLock,
    };
    unsafe extern "C" {
        fn dlopen(path: *const c_char, flags: i32) -> *mut c_void;
        fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    }
    fn symbol(name: &CStr) -> Option<usize> {
        static LIBRARY: OnceLock<Option<usize>> = OnceLock::new();
        let handle = (*LIBRARY.get_or_init(|| {
            let handle = unsafe { dlopen(c"/lib/libsws_client_c.so".as_ptr(), 0x102) };
            if handle.is_null() {
                None
            } else {
                Some(handle as usize)
            }
        }))?;
        let address = unsafe { dlsym(handle as *mut c_void, name.as_ptr()) };
        if address.is_null() {
            None
        } else {
            Some(address as usize)
        }
    }
    // Retain one handle for the process. GLFW's DT_NEEDED and this lookup resolve
    // to the same resident library and therefore the same SWS connection/queues.
    macro_rules! import {
        ($name:ident($($arg:ident: $ty:ty),*)) => {
            pub(super) unsafe fn $name($($arg: $ty),*) -> i32 {
                type Function = unsafe extern "C" fn($($ty),*) -> i32;
                static FUNCTION: OnceLock<Option<Function>> = OnceLock::new();
                let function = FUNCTION.get_or_init(|| {
                    let name = CStr::from_bytes_with_nul(concat!(stringify!($name), "\0").as_bytes()).unwrap();
                    symbol(name).map(|address| unsafe { std::mem::transmute::<usize, Function>(address) })
                });
                match function { Some(function) => unsafe { function($($arg),*) }, None => -3 }
            }
        };
    }
    import!(sws_get_display(out: *mut SwsDisplay));
    import!(sws_window_destroy(window: u32));
    import!(sws_window_size(window: u32, width: *mut u32, height: *mut u32));
    import!(sws_gpu_register(buffer: SwsBuffer, width: u32, height: u32, handle: i32));
    import!(sws_gpu_commit(buffer: SwsBuffer, serial: u64, width: u32, height: u32));
    import!(sws_gpu_destroy(buffer: SwsBuffer));
    import!(sws_gpu_poll(window: u32, out: *mut SwsGpuEvent));
}
#[cfg(target_os = "scarlet")]
use native::*;

fn display() -> Result<SwsDisplay, vk::Result> {
    let mut display = SwsDisplay::default();
    if unsafe { sws_get_display(&mut display) } < 0 || display.width == 0 || display.height == 0 {
        return Err(LOST);
    }
    Ok(display)
}

#[cfg(target_os = "linux")]
fn extent(display: &SwsDisplay) -> vk::Extent2D {
    vk::Extent2D {
        width: display.width,
        height: display.height,
    }
}

// The native window belongs to VkSurfaceKHR, not a swapchain generation.
// Recreating image storage must preserve SDL input routing and fullscreen state.
pub(crate) struct SurfaceWindow {
    id: u32,
    epoch: u32,
    owned: bool,
    serial: AtomicU64,
    retired: Mutex<Vec<(SwsBuffer, u64)>>,
    pending: Mutex<Vec<SwsGpuEvent>>,
    lost: AtomicBool,
}
impl SurfaceWindow {
    #[cfg(target_os = "linux")]
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
            owned: true,
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
    #[cfg(target_os = "scarlet")]
    pub fn borrow(id: u32) -> Result<Arc<Self>, vk::Result> {
        let display = display()?;
        if id == 0 || display.compositor_epoch == 0 {
            return Err(LOST);
        }
        let surface = Arc::new(Self {
            id,
            epoch: display.compositor_epoch,
            owned: false,
            serial: AtomicU64::new(0),
            retired: Mutex::new(Vec::new()),
            pending: Mutex::new(Vec::new()),
            lost: AtomicBool::new(false),
        });
        let _ = surface.size()?;
        Ok(surface)
    }
    pub fn size(&self) -> Result<vk::Extent2D, vk::Result> {
        self.events(0)?;
        let mut width = 0;
        let mut height = 0;
        if unsafe { sws_window_size(self.id, &mut width, &mut height) } < 0
            || width == 0
            || height == 0
        {
            return Err(LOST);
        }
        Ok(vk::Extent2D { width, height })
    }
    #[cfg(target_os = "scarlet")]
    pub fn extents(
        &self,
        maximum: u32,
    ) -> Result<(vk::Extent2D, vk::Extent2D, vk::Extent2D), vk::Result> {
        let size = self.size()?;
        if size.width > maximum || size.height > maximum {
            return Err(LOST);
        }
        Ok((size, size, size))
    }
    fn release_retired(&self, event: &SwsGpuEvent) -> bool {
        let mut retired = self.retired.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(index) = retired
            .iter()
            .position(|&(buffer, serial)| buffer == event.buffer && serial == event.commit_serial)
            && (event.kind == 1 || event.kind == 2)
        {
            let (buffer, _) = retired.swap_remove(index);
            unsafe {
                sws_gpu_destroy(buffer);
            }
            return true;
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
        if self.owned {
            unsafe {
                sws_window_destroy(self.id);
            }
        }
        // The compositor owns duplicated capabilities until its pending imports retire.
        for (buffer, _) in self
            .retired
            .get_mut()
            .unwrap_or_else(|p| p.into_inner())
            .drain(..)
        {
            unsafe {
                sws_gpu_destroy(buffer);
            }
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
        if size != surface.size()? {
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
        if self.surface.size()? != self.extent {
            return Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
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
