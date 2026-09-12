//! Loader-facing instance and physical-device ABI for the experimental headless ICD.
//!
//! The API version describes the entry-point ABI, not Vulkan conformance. Unsupported
//! features, extensions, formats, and image configurations are never advertised.

use ash::vk::{self, Handle};
use std::cell::UnsafeCell;
use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, c_char};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

const LOADER_MAGIC: usize = 0x01cd_c0de;
const MAX_IMAGE_DIMENSION: u32 = 2048;
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
    physical: Box<PhysicalObject>,
}

impl InstanceRecord {
    fn physical_id(&self) -> usize {
        std::ptr::from_ref(self.physical.as_ref()) as usize
    }
}

#[derive(Default)]
struct Instances {
    instances: HashMap<usize, InstanceRecord>,
    physical_devices: HashSet<usize>,
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
        .contains(&(physical.as_raw() as usize))
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
    let instance = Box::new(InstanceObject {
        loader_word: UnsafeCell::new(LOADER_MAGIC),
    });
    let physical = Box::new(PhysicalObject {
        loader_word: UnsafeCell::new(LOADER_MAGIC),
    });
    let instance_id = std::ptr::from_ref(instance.as_ref()) as usize;
    let record = InstanceRecord {
        _instance: instance,
        physical,
    };
    let mut registry = instances();
    registry.physical_devices.insert(record.physical_id());
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
        registry.physical_devices.remove(&record.physical_id());
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
    if output.is_null() {
        unsafe { *count = 1 };
        return vk::Result::SUCCESS;
    }
    if unsafe { *count } == 0 {
        return vk::Result::INCOMPLETE;
    }
    unsafe {
        *output = vk::PhysicalDevice::from_raw(record.physical_id() as u64);
        *count = 1;
    }
    vk::Result::SUCCESS
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
    if physical_valid(physical) {
        properties.api_version = vk::API_VERSION_1_0;
        properties.driver_version = vk::make_api_version(0, 0, 1, 0);
        properties.device_type = vk::PhysicalDeviceType::VIRTUAL_GPU;
        for (slot, byte) in properties.device_name.iter_mut().zip(
            b"SGFX headless (experimental, non-conformant)\0"
                .iter()
                .copied(),
        ) {
            *slot = byte as c_char;
        }
        properties.pipeline_cache_uuid = *b"sgfx-headless-v1";
        // Limits are for the implemented subset. Unsupported core facilities
        // retain zero limits; this device deliberately does not claim conformance.
        properties.limits = vk::PhysicalDeviceLimits {
            max_image_dimension2_d: MAX_IMAGE_DIMENSION,
            max_image_array_layers: 1,
            max_uniform_buffer_range: 16 * 1024,
            max_storage_buffer_range: 128 * 1024 * 1024,
            max_memory_allocation_count: 1024,
            buffer_image_granularity: 256,
            max_bound_descriptor_sets: 4,
            max_per_stage_descriptor_uniform_buffers: 12,
            max_per_stage_descriptor_storage_buffers: 4,
            max_per_stage_resources: 16,
            max_descriptor_set_uniform_buffers: 12,
            max_descriptor_set_storage_buffers: 4,
            max_vertex_output_components: 60,
            max_fragment_input_components: 60,
            max_fragment_output_attachments: 1,
            max_fragment_combined_output_resources: 4,
            max_compute_shared_memory_size: 16 * 1024,
            max_compute_work_group_count: [65535; 3],
            max_compute_work_group_invocations: 256,
            max_compute_work_group_size: [256, 256, 64],
            sub_pixel_precision_bits: 4,
            sub_texel_precision_bits: 4,
            mipmap_precision_bits: 4,
            max_viewports: 1,
            max_viewport_dimensions: [MAX_IMAGE_DIMENSION; 2],
            viewport_bounds_range: [-8192.0, 8191.0],
            min_memory_map_alignment: 8,
            min_uniform_buffer_offset_alignment: 256,
            min_storage_buffer_offset_alignment: 256,
            max_framebuffer_width: MAX_IMAGE_DIMENSION,
            max_framebuffer_height: MAX_IMAGE_DIMENSION,
            max_framebuffer_layers: 1,
            framebuffer_color_sample_counts: vk::SampleCountFlags::TYPE_1,
            max_color_attachments: 1,
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
    if !physical_valid(physical) {
        unsafe { *count = 0 };
        return;
    }
    if output.is_null() {
        unsafe { *count = 1 };
    } else if unsafe { *count } != 0 {
        unsafe {
            *output = vk::QueueFamilyProperties {
                queue_flags: vk::QueueFlags::GRAPHICS
                    | vk::QueueFlags::COMPUTE
                    | vk::QueueFlags::TRANSFER,
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
    if physical_valid(physical) && format == vk::Format::R8G8B8A8_UNORM {
        // In Vulkan 1.0, transfer support follows image-format support; the
        // TRANSFER_SRC/DST format-feature bits belong to maintenance1 / 1.1.
        properties.optimal_tiling_features = vk::FormatFeatureFlags::COLOR_ATTACHMENT;
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
    let supported_usage = vk::ImageUsageFlags::COLOR_ATTACHMENT
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST;
    if !physical_valid(physical)
        || format != vk::Format::R8G8B8A8_UNORM
        || image_type != vk::ImageType::TYPE_2D
        || tiling != vk::ImageTiling::OPTIMAL
        || !flags.is_empty()
        || usage.is_empty()
        || !supported_usage.contains(usage)
    {
        return vk::Result::ERROR_FORMAT_NOT_SUPPORTED;
    }
    unsafe {
        *output = vk::ImageFormatProperties {
            max_extent: vk::Extent3D {
                width: MAX_IMAGE_DIMENSION,
                height: MAX_IMAGE_DIMENSION,
                depth: 1,
            },
            max_mip_levels: 1,
            max_array_layers: 1,
            sample_counts: vk::SampleCountFlags::TYPE_1,
            max_resource_size: u64::from(MAX_IMAGE_DIMENSION).pow(2) * 4,
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
            assert_eq!(count, 1);
            count = 0;
            let mut physical = vk::PhysicalDevice::null();
            assert_eq!(
                enumerate_physical_devices(instance, &mut count, &mut physical),
                vk::Result::INCOMPLETE
            );
            assert_eq!(physical, vk::PhysicalDevice::null());
            count = 2;
            let sentinel = vk::PhysicalDevice::from_raw(0x1234);
            let mut devices = [vk::PhysicalDevice::null(), sentinel];
            assert_eq!(
                enumerate_physical_devices(instance, &mut count, devices.as_mut_ptr()),
                vk::Result::SUCCESS
            );
            assert_eq!(count, 1);
            assert_eq!(devices[1], sentinel);
            assert!(physical_valid(devices[0]));
            assert_eq!(*(devices[0].as_raw() as *const usize), LOADER_MAGIC);
            destroy_instance(instance, ptr::null());
            assert!(!physical_valid(devices[0]));
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
            let mut count = 1;
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
