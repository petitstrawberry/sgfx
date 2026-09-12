//! Actual GPU checks for indexed input, uniforms, and depth occlusion.
//! Build the ICD, then run `cargo test -p vulkan-sgfx --test cube -- --ignored`.
use ash::{Entry, vk};
use std::path::PathBuf;

#[path = "../examples/support/cube.rs"]
mod render;

#[test]
#[ignore = "requires a native GPU and freshly built SGFX ICD"]
fn indexed_cube_depth_rotation_and_index_formats() {
    let loader = std::env::var_os("SGFX_VULKAN_LOADER");
    let using_loader = loader.is_some();
    let path = loader
        .or_else(|| std::env::var_os("SGFX_ICD_LIBRARY"))
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::current_exe()
                .unwrap()
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .join(format!(
                    "{}vulkan_sgfx{}",
                    std::env::consts::DLL_PREFIX,
                    std::env::consts::DLL_SUFFIX
                ))
        });
    // The library remains loaded until verification destroys all Vulkan objects.
    let library = unsafe { libloading::Library::new(&path).expect("build the ICD first") };
    let symbol: &[u8] = if using_loader {
        b"vkGetInstanceProcAddr\0"
    } else {
        b"vk_icdGetInstanceProcAddr\0"
    };
    let get_instance_proc_addr = unsafe {
        *library
            .get::<vk::PFN_vkGetInstanceProcAddr>(symbol)
            .unwrap()
    };
    let entry = unsafe {
        Entry::from_static_fn(ash::StaticFn {
            get_instance_proc_addr,
        })
    };
    render::verify(&entry).unwrap();
}
