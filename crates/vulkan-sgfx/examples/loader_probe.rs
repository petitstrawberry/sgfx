//! Exercise ICD discovery through a real Vulkan loader, including device creation.
//!
//! Set VK_DRIVER_FILES to the SGFX ICD manifest. Set SGFX_VULKAN_LOADER to a loader
//! library path if the platform's default Vulkan loader is not on the search path.
//! Build the ICD first with `cargo build -p vulkan-sgfx`, then run this example.

use ash::vk::{self, Handle};
use std::ffi::CStr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let default_library = if cfg!(target_os = "macos") {
        "libvulkan.1.dylib"
    } else if cfg!(target_os = "windows") {
        "vulkan-1.dll"
    } else {
        "libvulkan.so.1"
    };
    let library_path =
        std::env::var_os("SGFX_VULKAN_LOADER").unwrap_or_else(|| default_library.into());

    // Keep the library alive until after all objects and the Ash dispatch tables
    // are dropped. This loads the platform loader, never the ICD directly.
    let library = unsafe { libloading::Library::new(&library_path)? };
    let get_instance_proc_addr =
        unsafe { *library.get::<vk::PFN_vkGetInstanceProcAddr>(b"vkGetInstanceProcAddr\0")? };
    let entry = unsafe {
        ash::Entry::from_static_fn(ash::StaticFn {
            get_instance_proc_addr,
        })
    };
    let application = vk::ApplicationInfo::default()
        .application_name(c"SGFX loader probe")
        .api_version(vk::API_VERSION_1_0);
    let info = vk::InstanceCreateInfo::default().application_info(&application);
    let instance = unsafe { entry.create_instance(&info, None)? };

    let probe = unsafe { probe_instance(&instance) };
    unsafe { instance.destroy_instance(None) };
    probe?;
    println!("PASS: real Vulkan loader discovered the SGFX ICD and created/destroyed a device");
    Ok(())
}

unsafe fn probe_instance(instance: &ash::Instance) -> Result<(), Box<dyn std::error::Error>> {
    let physical_devices = unsafe { instance.enumerate_physical_devices()? };
    println!("Physical devices discovered: {}", physical_devices.len());
    let physical = physical_devices
        .into_iter()
        .find(|&physical| {
            let properties = unsafe { instance.get_physical_device_properties(physical) };
            let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) };
            name.to_bytes().starts_with(b"SGFX headless")
        })
        .ok_or("the Vulkan loader did not enumerate the SGFX headless physical device")?;
    let properties = unsafe { instance.get_physical_device_properties(physical) };
    let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) };
    println!("Device: {}", name.to_string_lossy());
    println!(
        "Vulkan API: {}.{}.{}",
        vk::api_version_major(properties.api_version),
        vk::api_version_minor(properties.api_version),
        vk::api_version_patch(properties.api_version)
    );
    let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
    if families.len() != 1 || families[0].queue_count != 1 {
        return Err("unexpected SGFX queue-family capabilities".into());
    }
    let priorities = [1.0];
    let queues = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(0)
        .queue_priorities(&priorities)];
    let device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queues);
    let device = unsafe { instance.create_device(physical, &device_info, None)? };
    let queue = unsafe { device.get_device_queue(0, 0) };
    let result = if queue.as_raw() == 0 {
        Err("SGFX returned a null queue".into())
    } else {
        unsafe { device.device_wait_idle() }.map_err(Into::into)
    };
    unsafe { device.destroy_device(None) };
    result
}
