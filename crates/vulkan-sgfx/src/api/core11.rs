//! Vulkan 1.1 memory/maintenance ABI, backed by the existing resource model.
//! No protected, sparse, YCbCr, descriptor indexing or external-memory capability
//! is implied by these entry points. The ICD remains non-conformant.
use super::*;

unsafe fn dedicated(output: &mut vk::MemoryRequirements2<'_>) {
    let mut node = output.p_next.cast::<vk::BaseOutStructure<'_>>();
    for _ in 0..64 {
        let Some(base) = node.as_mut() else { break };
        if base.s_type == vk::StructureType::MEMORY_DEDICATED_REQUIREMENTS {
            let value = &mut *node.cast::<vk::MemoryDedicatedRequirements<'_>>();
            value.prefers_dedicated_allocation = vk::FALSE;
            value.requires_dedicated_allocation = vk::FALSE;
        }
        node = base.p_next;
    }
}
unsafe extern "system" fn buffer_requirements(
    device: vk::Device,
    input: *const vk::BufferMemoryRequirementsInfo2<'_>,
    output: *mut vk::MemoryRequirements2<'_>,
) {
    let (Some(input), Some(output)) = (input.as_ref(), output.as_mut()) else {
        return;
    };
    output.memory_requirements = vk::MemoryRequirements::default();
    dedicated(output);
    if input.s_type != vk::StructureType::BUFFER_MEMORY_REQUIREMENTS_INFO_2
        || !input.p_next.is_null()
    {
        return;
    }
    let function: vk::PFN_vkGetBufferMemoryRequirements = crate::resources::buffer_requirements;
    function(device, input.buffer, &mut output.memory_requirements);
}
unsafe extern "system" fn image_requirements(
    device: vk::Device,
    input: *const vk::ImageMemoryRequirementsInfo2<'_>,
    output: *mut vk::MemoryRequirements2<'_>,
) {
    let (Some(input), Some(output)) = (input.as_ref(), output.as_mut()) else {
        return;
    };
    output.memory_requirements = vk::MemoryRequirements::default();
    dedicated(output);
    if input.s_type != vk::StructureType::IMAGE_MEMORY_REQUIREMENTS_INFO_2
        || !input.p_next.is_null()
    {
        return;
    }
    let function: vk::PFN_vkGetImageMemoryRequirements =
        crate::images::get_image_memory_requirements;
    function(device, input.image, &mut output.memory_requirements);
}
unsafe extern "system" fn sparse_requirements(
    _: vk::Device,
    _: *const vk::ImageSparseMemoryRequirementsInfo2<'_>,
    count: *mut u32,
    _: *mut vk::SparseImageMemoryRequirements2<'_>,
) {
    if let Some(count) = count.as_mut() {
        *count = 0;
    }
}
unsafe extern "system" fn bind_buffers(
    device: vk::Device,
    count: u32,
    input: *const vk::BindBufferMemoryInfo<'_>,
) -> vk::Result {
    let values = match slice(input, count) {
        Ok(values) => values,
        Err(error) => return error,
    };
    let function: vk::PFN_vkBindBufferMemory = crate::resources::bind_buffer_memory;
    for value in values {
        if value.s_type != vk::StructureType::BIND_BUFFER_MEMORY_INFO || !value.p_next.is_null() {
            return vk::Result::ERROR_FEATURE_NOT_PRESENT;
        }
        let result = function(device, value.buffer, value.memory, value.memory_offset);
        if result != vk::Result::SUCCESS {
            return result;
        }
    }
    vk::Result::SUCCESS
}
unsafe extern "system" fn bind_images(
    device: vk::Device,
    count: u32,
    input: *const vk::BindImageMemoryInfo<'_>,
) -> vk::Result {
    let values = match slice(input, count) {
        Ok(values) => values,
        Err(error) => return error,
    };
    let function: vk::PFN_vkBindImageMemory = crate::images::bind_image_memory;
    for value in values {
        if value.s_type != vk::StructureType::BIND_IMAGE_MEMORY_INFO || !value.p_next.is_null() {
            return vk::Result::ERROR_FEATURE_NOT_PRESENT;
        }
        let result = function(device, value.image, value.memory, value.memory_offset);
        if result != vk::Result::SUCCESS {
            return result;
        }
    }
    vk::Result::SUCCESS
}
unsafe extern "system" fn layout_support(
    device: vk::Device,
    input: *const vk::DescriptorSetLayoutCreateInfo<'_>,
    output: *mut vk::DescriptorSetLayoutSupport<'_>,
) {
    let Some(output) = output.as_mut() else {
        return;
    };
    output.supported = vk::FALSE;
    let Some(input) = input.as_ref() else { return };
    if driver(device.as_raw(), Kind::Device).is_ok()
        && crate::resources::descriptor_layout(input).is_ok()
    {
        output.supported = vk::TRUE;
    }
}
unsafe extern "system" fn queue2(
    device: vk::Device,
    input: *const vk::DeviceQueueInfo2<'_>,
    output: *mut vk::Queue,
) {
    let Some(output) = output.as_mut() else {
        return;
    };
    *output = vk::Queue::null();
    let Some(input) = input.as_ref() else { return };
    if input.s_type == vk::StructureType::DEVICE_QUEUE_INFO_2
        && input.p_next.is_null()
        && input.flags.is_empty()
    {
        get_device_queue(device, input.queue_family_index, input.queue_index, output);
    }
}
unsafe extern "system" fn create_ycbcr(
    _: vk::Device,
    _: *const vk::SamplerYcbcrConversionCreateInfo<'_>,
    _: *const vk::AllocationCallbacks<'_>,
    output: *mut vk::SamplerYcbcrConversion,
) -> vk::Result {
    if let Some(output) = output.as_mut() {
        *output = vk::SamplerYcbcrConversion::null();
    }
    vk::Result::ERROR_FEATURE_NOT_PRESENT
}
unsafe extern "system" fn destroy_ycbcr(
    _: vk::Device,
    _: vk::SamplerYcbcrConversion,
    _: *const vk::AllocationCallbacks<'_>,
) {
}

pub(super) fn lookup(name: &CStr) -> vk::PFN_vkVoidFunction {
    macro_rules! entry {
        ($f:path, $ty:ty) => {{
            let function: $ty = $f;
            Some(std::mem::transmute::<$ty, unsafe extern "system" fn()>(
                function,
            ))
        }};
    }
    unsafe {
        match name.to_bytes() {
            b"vkGetBufferMemoryRequirements2" | b"vkGetBufferMemoryRequirements2KHR" => {
                entry!(buffer_requirements, vk::PFN_vkGetBufferMemoryRequirements2)
            }
            b"vkGetImageMemoryRequirements2" | b"vkGetImageMemoryRequirements2KHR" => {
                entry!(image_requirements, vk::PFN_vkGetImageMemoryRequirements2)
            }
            b"vkGetImageSparseMemoryRequirements2" | b"vkGetImageSparseMemoryRequirements2KHR" => {
                entry!(
                    sparse_requirements,
                    vk::PFN_vkGetImageSparseMemoryRequirements2
                )
            }
            b"vkBindBufferMemory2" | b"vkBindBufferMemory2KHR" => {
                entry!(bind_buffers, vk::PFN_vkBindBufferMemory2)
            }
            b"vkBindImageMemory2" | b"vkBindImageMemory2KHR" => {
                entry!(bind_images, vk::PFN_vkBindImageMemory2)
            }
            b"vkGetDescriptorSetLayoutSupport" | b"vkGetDescriptorSetLayoutSupportKHR" => {
                entry!(layout_support, vk::PFN_vkGetDescriptorSetLayoutSupport)
            }
            b"vkTrimCommandPool" => {
                entry!(maintenance1::trim_command_pool, vk::PFN_vkTrimCommandPool)
            }
            b"vkGetDeviceQueue2" => entry!(queue2, vk::PFN_vkGetDeviceQueue2),
            b"vkCreateSamplerYcbcrConversion" => {
                entry!(create_ycbcr, vk::PFN_vkCreateSamplerYcbcrConversion)
            }
            b"vkDestroySamplerYcbcrConversion" => {
                entry!(destroy_ycbcr, vk::PFN_vkDestroySamplerYcbcrConversion)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unsupported_optional_operations_do_not_publish_handles_or_stale_requirements() {
        unsafe {
            let mut conversion = vk::SamplerYcbcrConversion::from_raw(7);
            assert_eq!(
                create_ycbcr(
                    vk::Device::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    &mut conversion
                ),
                vk::Result::ERROR_FEATURE_NOT_PRESENT
            );
            assert_eq!(conversion, vk::SamplerYcbcrConversion::null());
            let mut dedicated =
                vk::MemoryDedicatedRequirements::default().requires_dedicated_allocation(true);
            let mut out = vk::MemoryRequirements2::default().push_next(&mut dedicated);
            out.memory_requirements.size = 99;
            buffer_requirements(
                vk::Device::null(),
                &vk::BufferMemoryRequirementsInfo2::default(),
                &mut out,
            );
            assert_eq!(out.memory_requirements.size, 0);
            assert_eq!(dedicated.requires_dedicated_allocation, vk::FALSE);
            let mut count = 99;
            sparse_requirements(
                vk::Device::null(),
                std::ptr::null(),
                &mut count,
                std::ptr::null_mut(),
            );
            assert_eq!(count, 0);
            assert_eq!(
                bind_buffers(vk::Device::null(), 1, std::ptr::null()),
                vk::Result::ERROR_INITIALIZATION_FAILED
            );
        }
    }
}
