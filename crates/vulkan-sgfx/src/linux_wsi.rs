//! Wayland uses the caller's wl_surface, including Wine GPU subsurfaces.
//! Each swapchain owns a private queue without changing application proxies.
use ash::vk;
use std::{
    ffi::{CStr, c_void},
    sync::Arc,
};

const LOST: vk::Result = vk::Result::ERROR_SURFACE_LOST_KHR;
unsafe extern "C" {
    fn sgfx_wayland_create(display: *mut c_void, surface: *mut c_void) -> *mut c_void;
    fn sgfx_wayland_destroy(ctx: *mut c_void);
    fn sgfx_wayland_supported(display: *mut c_void) -> i32;
    fn sgfx_wayland_register(ctx: *mut c_void, handle: i32, width: u32, height: u32) -> i32;
    fn sgfx_wayland_dispatch(ctx: *mut c_void) -> i32;
    fn sgfx_wayland_available(ctx: *mut c_void, index: u32) -> i32;
    fn sgfx_wayland_present(ctx: *mut c_void, index: u32, width: u32, height: u32) -> i32;
}

#[derive(Clone)]
pub(crate) enum SurfaceWindow {
    Display(Arc<crate::display::SurfaceWindow>),
    // Vulkan's Wayland surface contract keeps both objects alive until surface
    // destruction. libwayland synchronizes connection I/O across threads.
    Wayland { display: usize, surface: usize },
}
impl SurfaceWindow {
    pub fn extents(
        &self,
        maximum: u32,
    ) -> Result<(vk::Extent2D, vk::Extent2D, vk::Extent2D), vk::Result> {
        match self {
            Self::Display(_) => {
                let extent = crate::display::current_extent()?;
                Ok((extent, extent, extent))
            }
            Self::Wayland { .. } => Ok((
                vk::Extent2D {
                    width: u32::MAX,
                    height: u32::MAX,
                },
                vk::Extent2D {
                    width: 1,
                    height: 1,
                },
                vk::Extent2D {
                    width: maximum,
                    height: maximum,
                },
            )),
        }
    }
}

pub(crate) enum Window {
    Display(crate::display::Window),
    Wayland(WaylandWindow),
}
pub(crate) struct WaylandWindow {
    // Context access is serialized by the Vulkan driver's worker. Foreign
    // application listeners and queues are never dispatched by this context.
    ctx: usize,
    extent: vk::Extent2D,
    count: u32,
}
impl Window {
    pub fn new(surface: SurfaceWindow, size: vk::Extent2D) -> Result<Self, vk::Result> {
        match surface {
            SurfaceWindow::Display(surface) => {
                crate::display::Window::new(surface, size).map(Self::Display)
            }
            SurfaceWindow::Wayland { display, surface } => {
                let ctx =
                    unsafe { sgfx_wayland_create(display as *mut c_void, surface as *mut c_void) };
                if ctx.is_null() {
                    return Err(LOST);
                }
                Ok(Self::Wayland(WaylandWindow {
                    ctx: ctx as usize,
                    extent: size,
                    count: 0,
                }))
            }
        }
    }
    pub fn register(&mut self, image: &sgfx::driver::PresentationImage) -> Result<(), vk::Result> {
        match self {
            Self::Display(window) => window.register(image),
            Self::Wayland(window) => {
                let handle = image
                    .duplicate_shared_handle()
                    .map_err(crate::runtime::backend_failure)?;
                let index = unsafe {
                    sgfx_wayland_register(
                        window.ctx as *mut c_void,
                        handle.as_raw(),
                        image.width(),
                        image.height(),
                    )
                };
                if index < 0 || index as u32 != window.count {
                    return Err(LOST);
                }
                window.count += 1;
                Ok(())
            }
        }
    }
    pub fn dispatch(&mut self) -> Result<(), vk::Result> {
        match self {
            Self::Display(window) => window.dispatch(),
            Self::Wayland(window) => {
                status(unsafe { sgfx_wayland_dispatch(window.ctx as *mut c_void) })
            }
        }
    }
    pub fn available(&self, index: usize) -> bool {
        match self {
            Self::Display(window) => window.available(index),
            Self::Wayland(window) => unsafe {
                sgfx_wayland_available(window.ctx as *mut c_void, index as u32) > 0
            },
        }
    }
    pub fn present(&mut self, index: usize) -> Result<(), vk::Result> {
        match self {
            Self::Display(window) => window.present(index),
            Self::Wayland(window) => match unsafe {
                sgfx_wayland_present(
                    window.ctx as *mut c_void,
                    index as u32,
                    window.extent.width,
                    window.extent.height,
                )
            } {
                0 => Ok(()),
                1 => Err(vk::Result::NOT_READY),
                _ => Err(LOST),
            },
        }
    }
}
fn status(result: i32) -> Result<(), vk::Result> {
    if result < 0 { Err(LOST) } else { Ok(()) }
}
impl Drop for WaylandWindow {
    fn drop(&mut self) {
        unsafe { sgfx_wayland_destroy(self.ctx as *mut c_void) };
    }
}

unsafe extern "system" fn create_surface(
    instance: vk::Instance,
    info: *const vk::WaylandSurfaceCreateInfoKHR<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    output: *mut vk::SurfaceKHR,
) -> vk::Result {
    if output.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    unsafe {
        *output = vk::SurfaceKHR::null();
    }
    if info.is_null() || !allocator.is_null() || !crate::instance::instance_valid(instance) {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let info = unsafe { &*info };
    if info.s_type != vk::StructureType::WAYLAND_SURFACE_CREATE_INFO_KHR
        || !info.p_next.is_null()
        || !info.flags.is_empty()
        || info.display.is_null()
        || info.surface.is_null()
    {
        return vk::Result::ERROR_FEATURE_NOT_PRESENT;
    }
    if unsafe { sgfx_wayland_supported(info.display.cast()) } == 0 {
        return LOST;
    }
    let surface = crate::wsi::insert_linux_surface(
        instance,
        SurfaceWindow::Wayland {
            display: info.display as usize,
            surface: info.surface as usize,
        },
    );
    unsafe {
        *output = surface;
    }
    vk::Result::SUCCESS
}
unsafe extern "system" fn presentation_support(
    physical: vk::PhysicalDevice,
    family: u32,
    display: *mut vk::wl_display,
) -> vk::Bool32 {
    if family != 0 || display.is_null() || !crate::instance::physical_valid(physical) {
        return vk::FALSE;
    }
    if unsafe { sgfx_wayland_supported(display.cast()) } != 0 {
        vk::TRUE
    } else {
        vk::FALSE
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
        b"vkCreateWaylandSurfaceKHR" => entry!(create_surface, vk::PFN_vkCreateWaylandSurfaceKHR),
        b"vkGetPhysicalDeviceWaylandPresentationSupportKHR" => entry!(
            presentation_support,
            vk::PFN_vkGetPhysicalDeviceWaylandPresentationSupportKHR
        ),
        _ => None,
    }
}
