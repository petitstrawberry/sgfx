//! Device-owned Vulkan bootstrap objects. Empty pipeline caches are legal:
//! compilation may ignore the cache, but ownership and the binary header matter.
use super::*;
const INVALID: vk::Result = vk::Result::ERROR_INITIALIZATION_FAILED;
const UNSUPPORTED: vk::Result = vk::Result::ERROR_FEATURE_NOT_PRESENT;
const HEADER: [u8; 32] = [
    32, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

pub(crate) fn validate_cache(
    resources: &crate::resources::Resources,
    cache: vk::PipelineCache,
) -> VkResult<()> {
    if cache == vk::PipelineCache::null() || resources.pipeline_caches.contains_key(&cache) {
        Ok(())
    } else {
        Err(INVALID)
    }
}
unsafe extern "system" fn create_cache(
    device: vk::Device,
    info: *const vk::PipelineCacheCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::PipelineCache,
) -> vk::Result {
    if out.is_null() {
        return INVALID;
    }
    *out = vk::PipelineCache::null();
    let Some(info) = info.as_ref() else {
        return INVALID;
    };
    if info.s_type != vk::StructureType::PIPELINE_CACHE_CREATE_INFO
        || !info.p_next.is_null()
        || !info.flags.is_empty()
        || !allocator.is_null()
    {
        return UNSUPPORTED;
    }
    if info.initial_data_size != 0 && info.p_initial_data.is_null() {
        return INVALID;
    }
    // No compiled payload is persisted by this frontend. Incompatible input
    // caches may be ignored, as allowed by the Vulkan pipeline-cache contract.
    match with_device(device, |rt| {
        if rt.resources.pipeline_caches.len() >= 1024 {
            return Err(vk::Result::ERROR_TOO_MANY_OBJECTS);
        }
        let handle = vk::PipelineCache::from_raw(next_id());
        rt.resources.pipeline_caches.insert(handle, HEADER.to_vec());
        Ok(handle)
    }) {
        Ok(handle) => {
            *out = handle;
            vk::Result::SUCCESS
        }
        Err(error) => error,
    }
}
unsafe extern "system" fn destroy_cache(
    device: vk::Device,
    cache: vk::PipelineCache,
    _: *const vk::AllocationCallbacks<'_>,
) {
    let _ = with_device(device, move |rt| {
        rt.resources.pipeline_caches.remove(&cache);
        Ok(())
    });
}
fn copy_cache(data: &[u8], size: &mut usize, output: Option<&mut [u8]>) -> vk::Result {
    let Some(output) = output else {
        *size = data.len();
        return vk::Result::SUCCESS;
    };
    if output.len() < HEADER.len() {
        *size = 0;
        return vk::Result::INCOMPLETE;
    }
    let count = output.len().min(data.len());
    output[..count].copy_from_slice(&data[..count]);
    *size = count;
    if count < data.len() {
        vk::Result::INCOMPLETE
    } else {
        vk::Result::SUCCESS
    }
}
unsafe extern "system" fn get_cache_data(
    device: vk::Device,
    cache: vk::PipelineCache,
    size: *mut usize,
    out: *mut std::ffi::c_void,
) -> vk::Result {
    if size.is_null() {
        return INVALID;
    }
    let data = match with_device(device, move |rt| {
        rt.resources
            .pipeline_caches
            .get(&cache)
            .cloned()
            .ok_or(INVALID)
    }) {
        Ok(data) => data,
        Err(error) => return error,
    };
    let output = if out.is_null() {
        None
    } else {
        Some(std::slice::from_raw_parts_mut(
            out.cast::<u8>(),
            (*size).min(data.len()),
        ))
    };
    copy_cache(&data, &mut *size, output)
}
unsafe extern "system" fn merge_caches(
    device: vk::Device,
    destination: vk::PipelineCache,
    count: u32,
    sources: *const vk::PipelineCache,
) -> vk::Result {
    let sources = match slice(sources, count) {
        Ok(v) => v.to_vec(),
        Err(e) => return e,
    };
    status(with_device(device, move |rt| {
        validate_cache(&rt.resources, destination)?;
        if destination == vk::PipelineCache::null() {
            return Err(INVALID);
        }
        for source in sources {
            if source == destination || source == vk::PipelineCache::null() {
                return Err(INVALID);
            }
            validate_cache(&rt.resources, source)?;
        }
        Ok(()) // All caches have the same empty compiled payload.
    }))
}
unsafe extern "system" fn create_event(
    device: vk::Device,
    info: *const vk::EventCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::Event,
) -> vk::Result {
    if out.is_null() {
        return INVALID;
    }
    *out = vk::Event::null();
    let Some(info) = info.as_ref() else {
        return INVALID;
    };
    if info.s_type != vk::StructureType::EVENT_CREATE_INFO
        || !info.p_next.is_null()
        || !info.flags.is_empty()
        || !allocator.is_null()
    {
        return UNSUPPORTED;
    }
    let d = match driver(device.as_raw(), Kind::Device) {
        Ok(d) => d,
        Err(e) => return e,
    };
    let mut events = d.events.lock().unwrap_or_else(|e| e.into_inner());
    if events.len() >= 4096 {
        return vk::Result::ERROR_TOO_MANY_OBJECTS;
    }
    let handle = vk::Event::from_raw(next_id());
    events.insert(handle.as_raw(), Arc::new(AtomicBool::new(false)));
    *out = handle;
    vk::Result::SUCCESS
}
unsafe extern "system" fn destroy_event(
    device: vk::Device,
    event: vk::Event,
    _: *const vk::AllocationCallbacks<'_>,
) {
    if let Ok(d) = driver(device.as_raw(), Kind::Device) {
        d.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&event.as_raw());
    }
}
pub(super) fn event_state(d: &Driver, event: vk::Event) -> VkResult<Arc<AtomicBool>> {
    d.events
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&event.as_raw())
        .cloned()
        .ok_or(INVALID)
}
unsafe extern "system" fn get_event_status(device: vk::Device, event: vk::Event) -> vk::Result {
    match driver(device.as_raw(), Kind::Device).and_then(|d| event_state(&d, event)) {
        Ok(state) => {
            if state.load(Ordering::Acquire) {
                vk::Result::EVENT_SET
            } else {
                vk::Result::EVENT_RESET
            }
        }
        Err(e) => e,
    }
}
unsafe fn set_event_state(device: vk::Device, event: vk::Event, value: bool) -> vk::Result {
    match driver(device.as_raw(), Kind::Device).and_then(|d| event_state(&d, event)) {
        Ok(state) => {
            state.store(value, Ordering::Release);
            vk::Result::SUCCESS
        }
        Err(e) => e,
    }
}
unsafe extern "system" fn set_event(device: vk::Device, event: vk::Event) -> vk::Result {
    set_event_state(device, event, true)
}
unsafe extern "system" fn reset_event(device: vk::Device, event: vk::Event) -> vk::Result {
    set_event_state(device, event, false)
}
unsafe extern "system" fn sparse_requirements(
    device: vk::Device,
    image: vk::Image,
    count: *mut u32,
    _: *mut vk::SparseImageMemoryRequirements,
) {
    if count.is_null() {
        return;
    }
    // Sparse residency/binding is not exposed and sparse images are rejected.
    *count = 0;
    let _ = with_device(device, move |rt| {
        rt.resources.images.get(&image).ok_or(INVALID)?;
        Ok(())
    });
}
unsafe extern "system" fn render_area_granularity(
    device: vk::Device,
    pass: vk::RenderPass,
    out: *mut vk::Extent2D,
) {
    if out.is_null() {
        return;
    }
    *out = vk::Extent2D::default();
    if with_device(device, move |rt| {
        rt.resources.render_passes.get(&pass).ok_or(INVALID)?;
        Ok(())
    })
    .is_ok()
    {
        *out = vk::Extent2D {
            width: 1,
            height: 1,
        };
    }
}
pub(super) fn lookup(name: &CStr) -> vk::PFN_vkVoidFunction {
    macro_rules! entry {
        ($f:path,$ty:ty) => {{
            let function: $ty = $f;
            Some(unsafe { std::mem::transmute::<$ty, unsafe extern "system" fn()>(function) })
        }};
    }
    match name.to_bytes() {
        b"vkCreatePipelineCache" => entry!(create_cache, vk::PFN_vkCreatePipelineCache),
        b"vkDestroyPipelineCache" => entry!(destroy_cache, vk::PFN_vkDestroyPipelineCache),
        b"vkGetPipelineCacheData" => entry!(get_cache_data, vk::PFN_vkGetPipelineCacheData),
        b"vkMergePipelineCaches" => entry!(merge_caches, vk::PFN_vkMergePipelineCaches),
        b"vkCreateEvent" => entry!(create_event, vk::PFN_vkCreateEvent),
        b"vkDestroyEvent" => entry!(destroy_event, vk::PFN_vkDestroyEvent),
        b"vkGetEventStatus" => entry!(get_event_status, vk::PFN_vkGetEventStatus),
        b"vkSetEvent" => entry!(set_event, vk::PFN_vkSetEvent),
        b"vkResetEvent" => entry!(reset_event, vk::PFN_vkResetEvent),
        b"vkGetImageSparseMemoryRequirements" => entry!(
            sparse_requirements,
            vk::PFN_vkGetImageSparseMemoryRequirements
        ),
        b"vkGetRenderAreaGranularity" => {
            entry!(render_area_granularity, vk::PFN_vkGetRenderAreaGranularity)
        }
        _ => None,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cache_header_survives_small_output_and_round_trip() {
        let mut size = 0;
        assert_eq!(copy_cache(&HEADER, &mut size, None), vk::Result::SUCCESS);
        assert_eq!(size, 32);
        let mut short = [0xa5; 31];
        size = 31;
        assert_eq!(
            copy_cache(&HEADER, &mut size, Some(&mut short)),
            vk::Result::INCOMPLETE
        );
        assert_eq!(size, 0);
        assert_eq!(short, [0xa5; 31]);
        let mut full = [0; 32];
        size = 32;
        assert_eq!(
            copy_cache(&HEADER, &mut size, Some(&mut full)),
            vk::Result::SUCCESS
        );
        assert_eq!(full, HEADER);
        assert_eq!(u32::from_le_bytes(full[0..4].try_into().unwrap()), 32);
    }
    #[test]
    fn caches_validate_device_owned_handles() {
        let mut r = crate::resources::Resources::new();
        let cache = vk::PipelineCache::from_raw(11);
        assert!(validate_cache(&r, cache).is_err());
        r.pipeline_caches.insert(cache, HEADER.to_vec());
        assert!(validate_cache(&r, cache).is_ok());
        r.pipeline_caches.remove(&cache);
        assert!(validate_cache(&r, cache).is_err());
    }
}
