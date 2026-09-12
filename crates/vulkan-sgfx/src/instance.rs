//! Loader-facing instance and physical-device ABI for the experimental headless ICD.
//!
//! The API version describes the entry-point ABI, not Vulkan conformance. Unsupported
//! features, extensions, formats, and image configurations are never advertised.

use ash::vk::{self, Handle};
use sgfx::driver::{Adapter, DeviceType};
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::ffi::{CStr, c_char};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

const LOADER_MAGIC: usize = 0x01cd_c0de;
const MEMORY_HEAP_SIZE: u64 = 256 * 1024 * 1024;

// The loader replaces this first word with its own dispatch table pointer.
#[repr(C)]
struct InstanceObject {
    loader_word: UnsafeCell<usize>,
}

#[repr(C)]
struct PhysicalObject {
    loader_word: UnsafeCell<usize>,
}

struct InstanceRecord {
    _instance: Box<InstanceObject>,
    physical_devices: Vec<PhysicalRecord>,
}

struct PhysicalRecord {
    object: Box<PhysicalObject>,
}

impl PhysicalRecord {
    fn id(&self) -> usize {
        std::ptr::from_ref(self.object.as_ref()) as usize
    }
}

#[derive(Default)]
struct Instances {
    instances: HashMap<usize, InstanceRecord>,
    physical_devices: HashMap<usize, Adapter>,
}

fn instances() -> MutexGuard<'static, Instances> {
    static INSTANCES: OnceLock<Mutex<Instances>> = OnceLock::new();
    INSTANCES
        .get_or_init(|| Mutex::new(Instances::default()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

// Without negotiation, the legacy loader interface requires API-version checks.
static LOADER_INTERFACE: AtomicU32 = AtomicU32::new(1);

pub(crate) fn physical_valid(physical: vk::PhysicalDevice) -> bool {
    instances()
        .physical_devices
        .contains_key(&(physical.as_raw() as usize))
}

pub(crate) fn physical_adapter(physical: vk::PhysicalDevice) -> Option<Adapter> {
    instances()
        .physical_devices
        .get(&(physical.as_raw() as usize))
        .cloned()
}

/// Negotiate a loader interface with dispatchable-object initialization support.
///
/// # Safety
/// `version` must point to a writable loader interface version.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vk_icdNegotiateLoaderICDInterfaceVersion(
    version: *mut u32,
) -> vk::Result {
    if version.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let supported = unsafe { *version }.min(5);
    if supported < 2 {
        return vk::Result::ERROR_INCOMPATIBLE_DRIVER;
    }
    unsafe { *version = supported };
    LOADER_INTERFACE.store(supported, Ordering::Relaxed);
    vk::Result::SUCCESS
}

/// Return the entry points understood by this ICD.
///
/// # Safety
/// `name` must be null or point to a valid, nul-terminated entry-point name.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vk_icdGetInstanceProcAddr(
    instance: vk::Instance,
    name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    unsafe { get_instance_proc_addr(instance, name) }
}

/// This ICD does not expose physical-device extension entry points.
///
/// # Safety
/// The arguments follow the Vulkan loader's physical-device lookup ABI.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vk_icdGetPhysicalDeviceProcAddr(
    _instance: vk::Instance,
    _name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    None
}

// Assigning the declared Vulkan signature before erasure catches ABI mismatches.
macro_rules! procedure {
    ($function:expr, $signature:ty) => {{
        let function: $signature = $function;
        Some(unsafe { std::mem::transmute::<$signature, unsafe extern "system" fn()>(function) })
    }};
}

pub(crate) unsafe extern "system" fn get_instance_proc_addr(
    instance: vk::Instance,
    name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    if name.is_null() {
        return None;
    }
    let name = unsafe { CStr::from_ptr(name) };
    // Only global commands are visible before instance creation.
    match name.to_bytes() {
        b"vkCreateInstance" => return procedure!(create_instance, vk::PFN_vkCreateInstance),
        b"vkEnumerateInstanceExtensionProperties" => {
            return procedure!(
                enumerate_instance_extension_properties,
                vk::PFN_vkEnumerateInstanceExtensionProperties
            );
        }
        b"vkEnumerateInstanceLayerProperties" => {
            return procedure!(
                enumerate_instance_layer_properties,
                vk::PFN_vkEnumerateInstanceLayerProperties
            );
        }
        b"vkGetInstanceProcAddr" => {
            return procedure!(get_instance_proc_addr, vk::PFN_vkGetInstanceProcAddr);
        }
        _ => {}
    }
    if !instances()
        .instances
        .contains_key(&(instance.as_raw() as usize))
    {
        return None;
    }
    match name.to_bytes() {
        b"vkDestroyInstance" => procedure!(destroy_instance, vk::PFN_vkDestroyInstance),
        b"vkEnumeratePhysicalDevices" => {
            procedure!(
                enumerate_physical_devices,
                vk::PFN_vkEnumeratePhysicalDevices
            )
        }
        b"vkGetPhysicalDeviceFeatures" => procedure!(
            get_physical_device_features,
            vk::PFN_vkGetPhysicalDeviceFeatures
        ),
        b"vkGetPhysicalDeviceProperties" => procedure!(
            get_physical_device_properties,
            vk::PFN_vkGetPhysicalDeviceProperties
        ),
        b"vkGetPhysicalDeviceMemoryProperties" => procedure!(
            get_physical_device_memory_properties,
            vk::PFN_vkGetPhysicalDeviceMemoryProperties
        ),
        b"vkGetPhysicalDeviceQueueFamilyProperties" => procedure!(
            get_physical_device_queue_family_properties,
            vk::PFN_vkGetPhysicalDeviceQueueFamilyProperties
        ),
        b"vkGetPhysicalDeviceFormatProperties" => procedure!(
            get_physical_device_format_properties,
            vk::PFN_vkGetPhysicalDeviceFormatProperties
        ),
        b"vkGetPhysicalDeviceImageFormatProperties" => procedure!(
            get_physical_device_image_format_properties,
            vk::PFN_vkGetPhysicalDeviceImageFormatProperties
        ),
        b"vkGetPhysicalDeviceSparseImageFormatProperties" => procedure!(
            get_physical_device_sparse_image_format_properties,
            vk::PFN_vkGetPhysicalDeviceSparseImageFormatProperties
        ),
        b"vkEnumerateDeviceExtensionProperties" => procedure!(
            enumerate_device_extension_properties,
            vk::PFN_vkEnumerateDeviceExtensionProperties
        ),
        b"vkEnumerateDeviceLayerProperties" => procedure!(
            enumerate_device_layer_properties,
            vk::PFN_vkEnumerateDeviceLayerProperties
        ),
        _ => crate::api::lookup_device(name),
    }
}

pub(crate) unsafe extern "system" fn create_instance(
    info: *const vk::InstanceCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    output: *mut vk::Instance,
) -> vk::Result {
    if output.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    unsafe { *output = vk::Instance::null() };
    if info.is_null() || !allocator.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let info = unsafe { &*info };
    if info.s_type != vk::StructureType::INSTANCE_CREATE_INFO || !info.flags.is_empty() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    if info.enabled_layer_count != 0 {
        return vk::Result::ERROR_LAYER_NOT_PRESENT;
    }
    if info.enabled_extension_count != 0 {
        return vk::Result::ERROR_EXTENSION_NOT_PRESENT;
    }
    // Loader-owned pNext structures may accompany creation. No application
    // extension is enabled, so this ICD does not consume the chain.
    if !info.p_application_info.is_null() && LOADER_INTERFACE.load(Ordering::Relaxed) < 5 {
        let version = unsafe { (*info.p_application_info).api_version };
        if version != 0
            && (vk::api_version_variant(version) != 0
                || vk::api_version_major(version) != 1
                || vk::api_version_minor(version) != 0)
        {
            return vk::Result::ERROR_INCOMPATIBLE_DRIVER;
        }
    }
    let sgfx_instance = match sgfx::driver::Instance::new() {
        Ok(instance) => instance,
        Err(_) => return vk::Result::ERROR_INITIALIZATION_FAILED,
    };
    let instance = Box::new(InstanceObject {
        loader_word: UnsafeCell::new(LOADER_MAGIC),
    });
    let instance_id = std::ptr::from_ref(instance.as_ref()) as usize;
    let physical_devices: Vec<_> = sgfx_instance
        .adapters()
        .iter()
        .map(|_| PhysicalRecord {
            object: Box::new(PhysicalObject {
                loader_word: UnsafeCell::new(LOADER_MAGIC),
            }),
        })
        .collect();
    let record = InstanceRecord {
        _instance: instance,
        physical_devices,
    };
    let mut registry = instances();
    for (physical, adapter) in record
        .physical_devices
        .iter()
        .zip(sgfx_instance.adapters().iter().cloned())
    {
        registry.physical_devices.insert(physical.id(), adapter);
    }
    registry.instances.insert(instance_id, record);
    unsafe { *output = vk::Instance::from_raw(instance_id as u64) };
    vk::Result::SUCCESS
}

pub(crate) unsafe extern "system" fn destroy_instance(
    instance: vk::Instance,
    _allocator: *const vk::AllocationCallbacks<'_>,
) {
    let mut registry = instances();
    if let Some(record) = registry.instances.remove(&(instance.as_raw() as usize)) {
        for physical in &record.physical_devices {
            registry.physical_devices.remove(&physical.id());
        }
    }
}

pub(crate) unsafe extern "system" fn enumerate_physical_devices(
    instance: vk::Instance,
    count: *mut u32,
    output: *mut vk::PhysicalDevice,
) -> vk::Result {
    if count.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let registry = instances();
    let Some(record) = registry.instances.get(&(instance.as_raw() as usize)) else {
        unsafe { *count = 0 };
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let available = record.physical_devices.len();
    if output.is_null() {
        unsafe { *count = available as u32 };
        return vk::Result::SUCCESS;
    }
    let capacity = unsafe { *count } as usize;
    let written = capacity.min(available);
    for (index, physical) in record.physical_devices.iter().take(written).enumerate() {
        unsafe {
            *output.add(index) = vk::PhysicalDevice::from_raw(physical.id() as u64);
        }
    }
    unsafe { *count = written as u32 };
    if written < available {
        vk::Result::INCOMPLETE
    } else {
        vk::Result::SUCCESS
    }
}

unsafe extern "system" fn enumerate_instance_extension_properties(
    layer: *const c_char,
    count: *mut u32,
    _output: *mut vk::ExtensionProperties,
) -> vk::Result {
    if count.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    unsafe { *count = 0 };
    if !layer.is_null() {
        return vk::Result::ERROR_LAYER_NOT_PRESENT;
    }
    vk::Result::SUCCESS
}

unsafe extern "system" fn enumerate_instance_layer_properties(
    count: *mut u32,
    _output: *mut vk::LayerProperties,
) -> vk::Result {
    if count.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    unsafe { *count = 0 };
    vk::Result::SUCCESS
}

unsafe extern "system" fn enumerate_device_extension_properties(
    physical: vk::PhysicalDevice,
    layer: *const c_char,
    count: *mut u32,
    output: *mut vk::ExtensionProperties,
) -> vk::Result {
    if !physical_valid(physical) {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    unsafe { enumerate_instance_extension_properties(layer, count, output) }
}

unsafe extern "system" fn enumerate_device_layer_properties(
    physical: vk::PhysicalDevice,
    count: *mut u32,
    output: *mut vk::LayerProperties,
) -> vk::Result {
    if !physical_valid(physical) {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    unsafe { enumerate_instance_layer_properties(count, output) }
}

unsafe extern "system" fn get_physical_device_features(
    _physical: vk::PhysicalDevice,
    output: *mut vk::PhysicalDeviceFeatures,
) {
    if !output.is_null() {
        unsafe { *output = vk::PhysicalDeviceFeatures::default() };
    }
}

unsafe extern "system" fn get_physical_device_properties(
    physical: vk::PhysicalDevice,
    output: *mut vk::PhysicalDeviceProperties,
) {
    if output.is_null() {
        return;
    }
    let mut properties = vk::PhysicalDeviceProperties::default();
    if let Some(adapter) = physical_adapter(physical) {
        let info = adapter.info();
        let capabilities = adapter.capabilities();
        let limits = capabilities.limits();
        properties.api_version = vk::API_VERSION_1_0;
        properties.driver_version = vk::make_api_version(0, 0, 1, 0);
        properties.vendor_id = info.vendor_id();
        properties.device_id = info.device_id();
        properties.device_type = match info.device_type() {
            DeviceType::Integrated => vk::PhysicalDeviceType::INTEGRATED_GPU,
            DeviceType::Discrete => vk::PhysicalDeviceType::DISCRETE_GPU,
            DeviceType::Cpu => vk::PhysicalDeviceType::CPU,
            DeviceType::Virtual => vk::PhysicalDeviceType::VIRTUAL_GPU,
            DeviceType::Other => vk::PhysicalDeviceType::OTHER,
        };
        let name = format!("SGFX Vulkan ({})", info.name());
        let name_capacity = properties.device_name.len().saturating_sub(1);
        for (slot, byte) in properties
            .device_name
            .iter_mut()
            .take(name_capacity)
            .zip(name.bytes())
        {
            *slot = byte as c_char;
        }
        properties.pipeline_cache_uuid = *b"sgfx-headless-v1";
        let storage_buffers = capabilities.supports_storage_buffers();
        let compute = capabilities.supports_compute();
        let graphics = capabilities.supports_graphics();
        properties.limits = vk::PhysicalDeviceLimits {
            max_image_dimension2_d: limits.max_image_dimension_2d,
            max_image_array_layers: 1,
            max_uniform_buffer_range: limits.max_uniform_buffer_range,
            max_storage_buffer_range: if storage_buffers {
                limits.max_storage_buffer_range
            } else {
                0
            },
            max_memory_allocation_count: 1024,
            buffer_image_granularity: 256,
            max_bound_descriptor_sets: limits.max_bound_descriptor_sets,
            max_per_stage_descriptor_uniform_buffers: limits.max_uniform_buffers_per_stage,
            max_per_stage_descriptor_storage_buffers: if storage_buffers {
                limits.max_storage_buffers_per_stage
            } else {
                0
            },
            max_per_stage_resources: limits
                .max_uniform_buffers_per_stage
                .saturating_add(limits.max_storage_buffers_per_stage)
                .saturating_add(limits.max_color_attachments),
            max_descriptor_set_uniform_buffers: limits.max_uniform_buffers_per_stage,
            max_descriptor_set_storage_buffers: if storage_buffers {
                limits.max_storage_buffers_per_stage
            } else {
                0
            },
            max_vertex_input_attributes: if graphics {
                limits.max_vertex_attributes
            } else {
                0
            },
            max_vertex_input_bindings: if capabilities.supports_vertex_buffers() {
                limits.max_vertex_buffers
            } else {
                0
            },
            max_vertex_input_attribute_offset: limits.max_vertex_buffer_stride.saturating_sub(1),
            max_vertex_input_binding_stride: limits.max_vertex_buffer_stride,
            max_draw_indexed_index_value: u32::MAX,
            max_vertex_output_components: limits.max_inter_stage_components,
            max_fragment_input_components: limits.max_inter_stage_components,
            max_fragment_output_attachments: limits.max_color_attachments,
            max_fragment_combined_output_resources: limits.max_color_attachments.saturating_add(
                if storage_buffers {
                    limits.max_storage_buffers_per_stage
                } else {
                    0
                },
            ),
            max_compute_shared_memory_size: if compute {
                limits.max_compute_shared_memory_size
            } else {
                0
            },
            max_compute_work_group_count: if compute {
                limits.max_compute_work_group_count
            } else {
                [0; 3]
            },
            max_compute_work_group_invocations: if compute {
                limits.max_compute_work_group_invocations
            } else {
                0
            },
            max_compute_work_group_size: if compute {
                limits.max_compute_work_group_size
            } else {
                [0; 3]
            },
            sub_pixel_precision_bits: 4,
            sub_texel_precision_bits: 4,
            mipmap_precision_bits: 4,
            max_viewports: 1,
            max_viewport_dimensions: [limits.max_image_dimension_2d; 2],
            viewport_bounds_range: [-8192.0, 8191.0],
            min_memory_map_alignment: 8,
            min_uniform_buffer_offset_alignment: u64::from(
                limits.min_uniform_buffer_offset_alignment,
            ),
            min_storage_buffer_offset_alignment: u64::from(
                limits.min_storage_buffer_offset_alignment,
            ),
            max_framebuffer_width: limits.max_image_dimension_2d,
            max_framebuffer_height: limits.max_image_dimension_2d,
            max_framebuffer_layers: 1,
            framebuffer_color_sample_counts: vk::SampleCountFlags::TYPE_1,
            framebuffer_depth_sample_counts: vk::SampleCountFlags::TYPE_1,
            max_color_attachments: limits.max_color_attachments,
            max_sample_mask_words: 1,
            discrete_queue_priorities: 1,
            point_size_range: [1.0, 1.0],
            line_width_range: [1.0, 1.0],
            optimal_buffer_copy_offset_alignment: 4,
            optimal_buffer_copy_row_pitch_alignment: 256,
            non_coherent_atom_size: 1,
            ..Default::default()
        };
    }
    unsafe { *output = properties };
}

unsafe extern "system" fn get_physical_device_memory_properties(
    physical: vk::PhysicalDevice,
    output: *mut vk::PhysicalDeviceMemoryProperties,
) {
    if output.is_null() {
        return;
    }
    let mut properties = vk::PhysicalDeviceMemoryProperties::default();
    if physical_valid(physical) {
        properties.memory_type_count = 1;
        properties.memory_types[0] = vk::MemoryType {
            property_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL
                | vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
            heap_index: 0,
        };
        properties.memory_heap_count = 1;
        properties.memory_heaps[0] = vk::MemoryHeap {
            size: MEMORY_HEAP_SIZE,
            flags: vk::MemoryHeapFlags::DEVICE_LOCAL,
        };
    }
    unsafe { *output = properties };
}

unsafe extern "system" fn get_physical_device_queue_family_properties(
    physical: vk::PhysicalDevice,
    count: *mut u32,
    output: *mut vk::QueueFamilyProperties,
) {
    if count.is_null() {
        return;
    }
    let Some(adapter) = physical_adapter(physical) else {
        unsafe { *count = 0 };
        return;
    };
    if output.is_null() {
        unsafe { *count = 1 };
    } else if unsafe { *count } != 0 {
        let capabilities = adapter.capabilities();
        let mut queue_flags = vk::QueueFlags::empty();
        if capabilities.supports_graphics() {
            queue_flags |= vk::QueueFlags::GRAPHICS;
        }
        if capabilities.supports_compute() {
            queue_flags |= vk::QueueFlags::COMPUTE;
        }
        if capabilities.supports_transfer() {
            queue_flags |= vk::QueueFlags::TRANSFER;
        }
        unsafe {
            *output = vk::QueueFamilyProperties {
                queue_flags,
                queue_count: 1,
                timestamp_valid_bits: 0,
                min_image_transfer_granularity: vk::Extent3D {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            };
            *count = 1;
        }
    }
}

unsafe extern "system" fn get_physical_device_format_properties(
    physical: vk::PhysicalDevice,
    format: vk::Format,
    output: *mut vk::FormatProperties,
) {
    if output.is_null() {
        return;
    }
    let mut properties = vk::FormatProperties::default();
    let Some(adapter) = physical_adapter(physical) else {
        unsafe { *output = properties };
        return;
    };
    let capabilities = adapter.capabilities();
    if capabilities.supports_rgba8_color_attachment() && format == vk::Format::R8G8B8A8_UNORM {
        // In Vulkan 1.0, transfer support follows image-format support; the
        // TRANSFER_SRC/DST format-feature bits belong to maintenance1 / 1.1.
        properties.optimal_tiling_features = vk::FormatFeatureFlags::COLOR_ATTACHMENT;
    }
    if capabilities.supports_depth32_attachment() && format == vk::Format::D32_SFLOAT {
        properties.optimal_tiling_features = vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT;
    }
    if capabilities.supports_vertex_buffers()
        && matches!(
            format,
            vk::Format::R32G32_SFLOAT
                | vk::Format::R32G32B32_SFLOAT
                | vk::Format::R32G32B32A32_SFLOAT
                | vk::Format::R8G8B8A8_UNORM
        )
    {
        properties.buffer_features = vk::FormatFeatureFlags::VERTEX_BUFFER;
    }
    unsafe { *output = properties };
}

unsafe extern "system" fn get_physical_device_image_format_properties(
    physical: vk::PhysicalDevice,
    format: vk::Format,
    image_type: vk::ImageType,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    flags: vk::ImageCreateFlags,
    output: *mut vk::ImageFormatProperties,
) -> vk::Result {
    if output.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    unsafe { *output = vk::ImageFormatProperties::default() };
    let Some(adapter) = physical_adapter(physical) else {
        return vk::Result::ERROR_FORMAT_NOT_SUPPORTED;
    };
    let capabilities = adapter.capabilities();
    let mut supported_usage = crate::images::image_usage(format);
    if format == vk::Format::R8G8B8A8_UNORM {
        if !capabilities.supports_rgba8_color_attachment() {
            supported_usage &= !vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        if !capabilities.supports_image_readback() {
            supported_usage &= !vk::ImageUsageFlags::TRANSFER_SRC;
        }
        if !capabilities.supports_transfer() {
            supported_usage &=
                !(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST);
        }
    } else if format == vk::Format::D32_SFLOAT && !capabilities.supports_depth32_attachment() {
        supported_usage = vk::ImageUsageFlags::empty();
    }
    if supported_usage.is_empty()
        || image_type != vk::ImageType::TYPE_2D
        || tiling != vk::ImageTiling::OPTIMAL
        || !flags.is_empty()
        || usage.is_empty()
        || !supported_usage.contains(usage)
    {
        return vk::Result::ERROR_FORMAT_NOT_SUPPORTED;
    }
    unsafe {
        let max_dimension = capabilities.limits().max_image_dimension_2d;
        *output = vk::ImageFormatProperties {
            max_extent: vk::Extent3D {
                width: max_dimension,
                height: max_dimension,
                depth: 1,
            },
            max_mip_levels: 1,
            max_array_layers: 1,
            sample_counts: vk::SampleCountFlags::TYPE_1,
            max_resource_size: u64::from(max_dimension).pow(2) * 4,
        }
    };
    vk::Result::SUCCESS
}

unsafe extern "system" fn get_physical_device_sparse_image_format_properties(
    _physical: vk::PhysicalDevice,
    _format: vk::Format,
    _image_type: vk::ImageType,
    _samples: vk::SampleCountFlags,
    _usage: vk::ImageUsageFlags,
    _tiling: vk::ImageTiling,
    count: *mut u32,
    _output: *mut vk::SparseImageFormatProperties,
) {
    if !count.is_null() {
        unsafe { *count = 0 };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    fn new_instance() -> vk::Instance {
        let mut instance = vk::Instance::null();
        assert_eq!(
            unsafe {
                create_instance(
                    &vk::InstanceCreateInfo::default(),
                    ptr::null(),
                    &mut instance,
                )
            },
            vk::Result::SUCCESS
        );
        instance
    }

    #[test]
    fn lookup_respects_instance_scope_and_unsupported_commands() {
        unsafe {
            assert!(
                get_instance_proc_addr(vk::Instance::null(), c"vkCreateInstance".as_ptr())
                    .is_some()
            );
            assert!(
                get_instance_proc_addr(vk::Instance::null(), c"vkDestroyInstance".as_ptr())
                    .is_none()
            );
            assert!(get_instance_proc_addr(vk::Instance::null(), ptr::null()).is_none());
            let instance = new_instance();
            for name in [
                c"vkEnumeratePhysicalDevices",
                c"vkGetPhysicalDeviceFeatures",
                c"vkGetPhysicalDeviceProperties",
                c"vkGetPhysicalDeviceMemoryProperties",
                c"vkGetPhysicalDeviceQueueFamilyProperties",
                c"vkGetPhysicalDeviceFormatProperties",
                c"vkGetPhysicalDeviceImageFormatProperties",
                c"vkGetPhysicalDeviceSparseImageFormatProperties",
                c"vkEnumerateDeviceExtensionProperties",
            ] {
                assert!(
                    get_instance_proc_addr(instance, name.as_ptr()).is_some(),
                    "{name:?}"
                );
            }
            assert!(get_instance_proc_addr(instance, c"vkCreateSwapchainKHR".as_ptr()).is_none());
            assert!(
                get_instance_proc_addr(instance, c"vkGetPhysicalDeviceFeatures2".as_ptr())
                    .is_none()
            );
            destroy_instance(instance, ptr::null());
            assert!(get_instance_proc_addr(instance, c"vkDestroyInstance".as_ptr()).is_none());
        }
    }

    #[test]
    fn enumeration_obeys_capacity_and_physical_lifetime() {
        unsafe {
            let instance = new_instance();
            assert_eq!(*(instance.as_raw() as *const usize), LOADER_MAGIC);
            let mut count = 0;
            assert_eq!(
                enumerate_physical_devices(instance, &mut count, ptr::null_mut()),
                vk::Result::SUCCESS
            );
            let available = count;
            count = 0;
            let mut physical = vk::PhysicalDevice::null();
            assert_eq!(
                enumerate_physical_devices(instance, &mut count, &mut physical),
                if available == 0 {
                    vk::Result::SUCCESS
                } else {
                    vk::Result::INCOMPLETE
                }
            );
            assert_eq!(physical, vk::PhysicalDevice::null());
            let sentinel = vk::PhysicalDevice::from_raw(0x1234);
            let mut devices = vec![vk::PhysicalDevice::null(); available as usize + 1];
            devices[available as usize] = sentinel;
            count = available + 1;
            assert_eq!(
                enumerate_physical_devices(instance, &mut count, devices.as_mut_ptr()),
                vk::Result::SUCCESS
            );
            assert_eq!(count, available);
            assert_eq!(devices[available as usize], sentinel);
            for &device in &devices[..available as usize] {
                assert!(physical_valid(device));
                assert_eq!(*(device.as_raw() as *const usize), LOADER_MAGIC);
            }
            destroy_instance(instance, ptr::null());
            for &device in &devices[..available as usize] {
                assert!(!physical_valid(device));
            }
            assert!(!physical_valid(sentinel));
            assert_eq!(
                enumerate_physical_devices(instance, &mut count, ptr::null_mut()),
                vk::Result::ERROR_INITIALIZATION_FAILED
            );
            assert_eq!(count, 0);
        }
    }

    #[test]
    fn unsupported_extensions_and_image_configuration_are_rejected() {
        unsafe {
            let extensions = [c"VK_KHR_surface".as_ptr()];
            let info = vk::InstanceCreateInfo::default().enabled_extension_names(&extensions);
            let mut output = vk::Instance::from_raw(0x1234);
            assert_eq!(
                create_instance(&info, ptr::null(), &mut output),
                vk::Result::ERROR_EXTENSION_NOT_PRESENT
            );
            assert_eq!(output, vk::Instance::null());
            let instance = new_instance();
            let mut count = 0;
            assert_eq!(
                enumerate_physical_devices(instance, &mut count, ptr::null_mut()),
                vk::Result::SUCCESS
            );
            if count == 0 {
                destroy_instance(instance, ptr::null());
                return;
            }
            count = 1;
            let mut physical = vk::PhysicalDevice::null();
            assert_eq!(
                enumerate_physical_devices(instance, &mut count, &mut physical),
                vk::Result::SUCCESS
            );
            let mut image = vk::ImageFormatProperties::default();
            assert_eq!(
                get_physical_device_image_format_properties(
                    physical,
                    vk::Format::R8G8B8A8_UNORM,
                    vk::ImageType::TYPE_2D,
                    vk::ImageTiling::OPTIMAL,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT,
                    vk::ImageCreateFlags::empty(),
                    &mut image
                ),
                vk::Result::SUCCESS
            );
            assert_eq!(image.max_mip_levels, 1);
            assert_eq!(
                get_physical_device_image_format_properties(
                    physical,
                    vk::Format::R8G8B8A8_UNORM,
                    vk::ImageType::TYPE_2D,
                    vk::ImageTiling::OPTIMAL,
                    vk::ImageUsageFlags::SAMPLED,
                    vk::ImageCreateFlags::empty(),
                    &mut image
                ),
                vk::Result::ERROR_FORMAT_NOT_SUPPORTED
            );
            assert_eq!(image.max_mip_levels, 0);
            let mut formats = vk::FormatProperties::default();
            get_physical_device_format_properties(physical, vk::Format::D32_SFLOAT, &mut formats);
            assert_eq!(
                formats.optimal_tiling_features,
                vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
            );
            assert_eq!(
                get_physical_device_image_format_properties(
                    physical,
                    vk::Format::D32_SFLOAT,
                    vk::ImageType::TYPE_2D,
                    vk::ImageTiling::OPTIMAL,
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                    vk::ImageCreateFlags::empty(),
                    &mut image,
                ),
                vk::Result::SUCCESS
            );
            assert_eq!(
                get_physical_device_image_format_properties(
                    physical,
                    vk::Format::D32_SFLOAT,
                    vk::ImageType::TYPE_2D,
                    vk::ImageTiling::OPTIMAL,
                    vk::ImageUsageFlags::TRANSFER_SRC,
                    vk::ImageCreateFlags::empty(),
                    &mut image,
                ),
                vk::Result::ERROR_FORMAT_NOT_SUPPORTED
            );
            get_physical_device_format_properties(
                physical,
                vk::Format::R32G32B32_SFLOAT,
                &mut formats,
            );
            assert_eq!(
                formats.buffer_features,
                vk::FormatFeatureFlags::VERTEX_BUFFER
            );
            assert!(formats.optimal_tiling_features.is_empty());
            count = 10;
            assert_eq!(
                enumerate_device_extension_properties(
                    physical,
                    ptr::null(),
                    &mut count,
                    ptr::null_mut()
                ),
                vk::Result::SUCCESS
            );
            assert_eq!(count, 0);
            destroy_instance(instance, ptr::null());
        }
    }
}
