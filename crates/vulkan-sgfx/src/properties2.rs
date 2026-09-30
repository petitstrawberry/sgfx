//! Vulkan 1.0 KHR property queries, including Wine's adapter identification.
//! External memory has no export/import capability; advertising the query
//! extension does not advertise any usable external handle type.
use ash::vk;
use std::ffi::CStr;

unsafe extern "system" fn properties(
    physical: vk::PhysicalDevice,
    out: *mut vk::PhysicalDeviceProperties2<'_>,
) {
    let Some(out) = (unsafe { out.as_mut() }) else {
        return;
    };
    unsafe {
        crate::instance::get_physical_device_properties(physical, &mut out.properties);
    }
    let mut node = out.p_next.cast::<vk::BaseOutStructure<'_>>();
    for _ in 0..64 {
        let Some(base) = (unsafe { node.as_mut() }) else {
            break;
        };
        if base.s_type == vk::StructureType::PHYSICAL_DEVICE_ID_PROPERTIES {
            let id = unsafe { &mut *node.cast::<vk::PhysicalDeviceIDProperties<'_>>() };
            id.device_uuid = device_uuid(&out.properties);
            id.driver_uuid = *b"sgfx-vulkan-v001";
            id.device_luid = [0; vk::LUID_SIZE];
            id.device_node_mask = 0;
            id.device_luid_valid = vk::FALSE;
        }
        node = base.p_next;
    }
}
fn device_uuid(properties: &vk::PhysicalDeviceProperties) -> [u8; vk::UUID_SIZE] {
    let mut a = 0xcbf29ce484222325u64;
    let mut b = 0x84222325cbf29ce4u64;
    for byte in properties
        .vendor_id
        .to_le_bytes()
        .into_iter()
        .chain(properties.device_id.to_le_bytes())
        .chain(properties.device_name.iter().map(|&byte| byte as u8))
    {
        a = (a ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        b = (b ^ u64::from(byte)).wrapping_mul(0x100000001b3);
    }
    let mut uuid = [0; vk::UUID_SIZE];
    uuid[..8].copy_from_slice(&a.to_le_bytes());
    uuid[8..].copy_from_slice(&b.to_le_bytes());
    uuid
}
unsafe extern "system" fn features(
    physical: vk::PhysicalDevice,
    out: *mut vk::PhysicalDeviceFeatures2<'_>,
) {
    if let Some(out) = unsafe { out.as_mut() } {
        unsafe {
            crate::instance::get_physical_device_features(physical, &mut out.features);
        }
    }
}
unsafe extern "system" fn memory(
    physical: vk::PhysicalDevice,
    out: *mut vk::PhysicalDeviceMemoryProperties2<'_>,
) {
    if let Some(out) = unsafe { out.as_mut() } {
        unsafe {
            crate::instance::get_physical_device_memory_properties(
                physical,
                &mut out.memory_properties,
            );
        }
    }
}
unsafe extern "system" fn format(
    physical: vk::PhysicalDevice,
    format: vk::Format,
    out: *mut vk::FormatProperties2<'_>,
) {
    if let Some(out) = unsafe { out.as_mut() } {
        unsafe {
            crate::instance::get_physical_device_format_properties(
                physical,
                format,
                &mut out.format_properties,
            );
        }
    }
}
unsafe extern "system" fn image_format(
    physical: vk::PhysicalDevice,
    info: *const vk::PhysicalDeviceImageFormatInfo2<'_>,
    out: *mut vk::ImageFormatProperties2<'_>,
) -> vk::Result {
    let (Some(info), Some(out)) = (unsafe { info.as_ref() }, unsafe { out.as_mut() }) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    out.image_format_properties = vk::ImageFormatProperties::default();
    let mut node = info.p_next.cast::<vk::BaseInStructure<'_>>();
    for _ in 0..64 {
        let Some(base) = (unsafe { node.as_ref() }) else {
            break;
        };
        if base.s_type == vk::StructureType::PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO {
            let external =
                unsafe { &*node.cast::<vk::PhysicalDeviceExternalImageFormatInfo<'_>>() };
            if !external.handle_type.is_empty() {
                return vk::Result::ERROR_FORMAT_NOT_SUPPORTED;
            }
        }
        node = base.p_next;
    }
    let mut node = out.p_next.cast::<vk::BaseOutStructure<'_>>();
    for _ in 0..64 {
        let Some(base) = (unsafe { node.as_mut() }) else {
            break;
        };
        if base.s_type == vk::StructureType::EXTERNAL_IMAGE_FORMAT_PROPERTIES {
            unsafe {
                (*node.cast::<vk::ExternalImageFormatProperties<'_>>())
                    .external_memory_properties = vk::ExternalMemoryProperties::default();
            }
        }
        node = base.p_next;
    }
    unsafe {
        crate::instance::get_physical_device_image_format_properties(
            physical,
            info.format,
            info.ty,
            info.tiling,
            info.usage,
            info.flags,
            &mut out.image_format_properties,
        )
    }
}
unsafe extern "system" fn queues(
    physical: vk::PhysicalDevice,
    count: *mut u32,
    out: *mut vk::QueueFamilyProperties2<'_>,
) {
    if count.is_null() {
        return;
    }
    if out.is_null() {
        unsafe {
            crate::instance::get_physical_device_queue_family_properties(
                physical,
                count,
                std::ptr::null_mut(),
            );
        }
    } else if unsafe { *count } != 0 {
        let mut capacity = 1;
        unsafe {
            crate::instance::get_physical_device_queue_family_properties(
                physical,
                &mut capacity,
                &mut (*out).queue_family_properties,
            );
            *count = capacity;
        }
    }
}
unsafe extern "system" fn sparse(
    _physical: vk::PhysicalDevice,
    _info: *const vk::PhysicalDeviceSparseImageFormatInfo2<'_>,
    count: *mut u32,
    _out: *mut vk::SparseImageFormatProperties2<'_>,
) {
    if !count.is_null() {
        unsafe {
            *count = 0;
        }
    }
}
unsafe extern "system" fn external_buffer(
    _physical: vk::PhysicalDevice,
    _info: *const vk::PhysicalDeviceExternalBufferInfo<'_>,
    out: *mut vk::ExternalBufferProperties<'_>,
) {
    if let Some(out) = unsafe { out.as_mut() } {
        out.external_memory_properties = vk::ExternalMemoryProperties::default();
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
        b"vkGetPhysicalDeviceProperties2KHR" => {
            entry!(properties, vk::PFN_vkGetPhysicalDeviceProperties2)
        }
        b"vkGetPhysicalDeviceFeatures2KHR" => {
            entry!(features, vk::PFN_vkGetPhysicalDeviceFeatures2)
        }
        b"vkGetPhysicalDeviceMemoryProperties2KHR" => {
            entry!(memory, vk::PFN_vkGetPhysicalDeviceMemoryProperties2)
        }
        b"vkGetPhysicalDeviceFormatProperties2KHR" => {
            entry!(format, vk::PFN_vkGetPhysicalDeviceFormatProperties2)
        }
        b"vkGetPhysicalDeviceImageFormatProperties2KHR" => entry!(
            image_format,
            vk::PFN_vkGetPhysicalDeviceImageFormatProperties2
        ),
        b"vkGetPhysicalDeviceQueueFamilyProperties2KHR" => {
            entry!(queues, vk::PFN_vkGetPhysicalDeviceQueueFamilyProperties2)
        }
        b"vkGetPhysicalDeviceSparseImageFormatProperties2KHR" => entry!(
            sparse,
            vk::PFN_vkGetPhysicalDeviceSparseImageFormatProperties2
        ),
        b"vkGetPhysicalDeviceExternalBufferPropertiesKHR" => entry!(
            external_buffer,
            vk::PFN_vkGetPhysicalDeviceExternalBufferProperties
        ),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn uuid_is_stable_and_does_not_depend_on_instance_handles() {
        let mut properties = vk::PhysicalDeviceProperties::default();
        properties.vendor_id = 0x1234;
        properties.device_id = 7;
        let uuid = device_uuid(&properties);
        assert_ne!(uuid, [0; vk::UUID_SIZE]);
        assert_eq!(uuid, device_uuid(&properties));
        properties.device_id = 8;
        assert_ne!(uuid, device_uuid(&properties));
    }
    #[test]
    fn unsupported_external_memory_is_not_advertised_or_accepted() {
        let input = vk::PhysicalDeviceExternalBufferInfo::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let mut output = vk::ExternalBufferProperties::default();
        output.external_memory_properties.external_memory_features =
            vk::ExternalMemoryFeatureFlags::EXPORTABLE;
        unsafe {
            external_buffer(vk::PhysicalDevice::null(), &input, &mut output);
        }
        assert!(
            output
                .external_memory_properties
                .external_memory_features
                .is_empty()
        );
        assert_eq!(output.s_type, vk::StructureType::EXTERNAL_BUFFER_PROPERTIES);
        let mut external = vk::PhysicalDeviceExternalImageFormatInfo::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let input = vk::PhysicalDeviceImageFormatInfo2::default().push_next(&mut external);
        let mut output = vk::ImageFormatProperties2::default();
        assert_eq!(
            unsafe { image_format(vk::PhysicalDevice::null(), &input, &mut output) },
            vk::Result::ERROR_FORMAT_NOT_SUPPORTED
        );
    }
}
