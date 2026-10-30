//! Core 1.2 dispatch for optional device-address operations. No physical-address
//! feature is advertised; those commands cannot be used by a valid application.
use super::*;

unsafe extern "system" fn buffer_address(
    _: vk::Device,
    _: *const vk::BufferDeviceAddressInfo<'_>,
) -> u64 {
    0
}
unsafe extern "system" fn memory_address(
    _: vk::Device,
    _: *const vk::DeviceMemoryOpaqueCaptureAddressInfo<'_>,
) -> u64 {
    0
}

pub(super) fn lookup(name: &CStr) -> vk::PFN_vkVoidFunction {
    match name.to_bytes() {
        b"vkGetBufferDeviceAddress" | b"vkGetBufferOpaqueCaptureAddress" => {
            let f: vk::PFN_vkGetBufferDeviceAddress = buffer_address;
            Some(unsafe {
                std::mem::transmute::<vk::PFN_vkGetBufferDeviceAddress, unsafe extern "system" fn()>(
                    f,
                )
            })
        }
        b"vkGetDeviceMemoryOpaqueCaptureAddress" => {
            let f: vk::PFN_vkGetDeviceMemoryOpaqueCaptureAddress = memory_address;
            Some(unsafe {
                std::mem::transmute::<
                    vk::PFN_vkGetDeviceMemoryOpaqueCaptureAddress,
                    unsafe extern "system" fn(),
                >(f)
            })
        }
        _ => None,
    }
}
