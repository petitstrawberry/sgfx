//! Load arbitrary SPIR-V through an ordinary Vulkan loader and shader-module API.
use ash::{Entry, vk};
use std::{error::Error, ffi::CString};
fn main() -> Result<(), Box<dyn Error>> {
    let entry = unsafe { Entry::load()? };
    let name = CString::new("SPIR-V module probe")?;
    let app = vk::ApplicationInfo::default()
        .application_name(&name)
        .api_version(vk::API_VERSION_1_0);
    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(&app),
            None,
        )?
    };
    let physical = unsafe { instance.enumerate_physical_devices()? }[0];
    let properties = unsafe { instance.get_physical_device_properties(physical) };
    println!(
        "Device: {}",
        unsafe { std::ffi::CStr::from_ptr(properties.device_name.as_ptr()) }.to_string_lossy()
    );
    let priorities = [1.0];
    let queues = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(0)
        .queue_priorities(&priorities)];
    let device = unsafe {
        instance.create_device(
            physical,
            &vk::DeviceCreateInfo::default().queue_create_infos(&queues),
            None,
        )?
    };
    let mut failed = 0;
    for path in std::env::args().skip(1) {
        let bytes = std::fs::read(&path)?;
        if bytes.len() % 4 != 0 {
            return Err(format!("unaligned SPIR-V file: {path}").into());
        }
        let words = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        match unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
        } {
            Ok(module) => {
                println!("PASS {path}");
                unsafe {
                    device.destroy_shader_module(module, None);
                }
            }
            Err(error) => {
                println!("FAIL {path}: {error:?}");
                failed += 1;
            }
        }
    }
    unsafe {
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
    if failed > 0 {
        return Err(format!("{failed} shader modules were rejected").into());
    }
    Ok(())
}
