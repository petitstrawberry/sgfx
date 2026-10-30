pub(crate) mod bootstrap;
mod buffer_operations;
mod buffer_readback;
mod buffer_uploads;
mod buffer_views;
mod core11;
mod core12;
pub(crate) mod descriptor_template;
pub(crate) mod dynamic_rendering;
mod dynamic_states;
mod indirect;
mod maintenance1;
mod push_descriptors;
mod queries;
mod queue;
mod render_pass2;
mod render_passes;
mod synchronization2;
mod timeline;
mod unsupported;
use crate::runtime::Runtime;
use ash::vk::{self, Handle};
use sgfx::{
    backend::{Completion, CompletionStatus, SubmitError},
    ir,
};
use std::{
    collections::{BTreeMap, HashMap},
    ffi::{CStr, c_char},
    sync::{
        Arc, Condvar, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering},
        mpsc,
    },
};

type VkResult<T> = Result<T, vk::Result>;
type BufferUploads = (Vec<ir::OwnedCommand>, Vec<(vk::Buffer, vk::DeviceMemory)>);
type Job = Box<dyn FnOnce(&mut Runtime) + Send>;
enum Request {
    Run(Job),
    Stop,
}

enum CompletionRequest {
    Observe {
        submissions: Vec<sgfx::driver::Submission>,
        fence: Option<Arc<AtomicU8>>,
        signals: Vec<Arc<AtomicU8>>,
        timeline_signals: Vec<(Arc<timeline::Timeline>, u64)>,
    },
    Stop,
}
pub(crate) type Fences = Arc<Mutex<HashMap<u64, Arc<AtomicU8>>>>;
pub(crate) type Semaphores = Arc<Mutex<HashMap<u64, Arc<AtomicU8>>>>;
pub(crate) type Recordings = Arc<Mutex<CommandRegistry>>;
type RecordingCell = Arc<Mutex<Recording>>;

#[derive(Default)]
pub(crate) struct InFlight {
    count: Mutex<usize>,
    changed: Condvar,
}

impl InFlight {
    fn reserve_pending(&self, additional: usize) -> VkResult<()> {
        let mut count = self.count.lock().unwrap_or_else(|e| e.into_inner());
        let total = count
            .checked_add(additional)
            .ok_or(vk::Result::ERROR_OUT_OF_HOST_MEMORY)?;
        if total > 4096 {
            return Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY);
        }
        *count = total;
        Ok(())
    }
    fn begin(&self) {
        let mut count = self
            .count
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *count = count.saturating_add(1);
    }

    fn finish(&self) {
        let mut count = self
            .count
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *count = count.saturating_sub(1);
        self.changed.notify_all();
    }

    // State stores happen before acquiring this mutex. Sharing the mutex with
    // wait_until closes the check-to-sleep race without holding runtime locks.
    fn notify(&self) {
        let _count = self.count.lock().unwrap_or_else(|e| e.into_inner());
        self.changed.notify_all();
    }

    fn wait_until(&self, timeout: u64, mut ready: impl FnMut() -> VkResult<bool>) -> VkResult<()> {
        let started = std::time::Instant::now();
        let mut guard = self.count.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            // This predicate must inspect only pre-resolved atomic state. In
            // particular, never resolve handles or enter the device worker here.
            if ready()? {
                return Ok(());
            }
            if timeout == u64::MAX {
                guard = self.changed.wait(guard).unwrap_or_else(|e| e.into_inner());
            } else {
                let remaining =
                    std::time::Duration::from_nanos(timeout).saturating_sub(started.elapsed());
                if remaining.is_zero() {
                    return Err(vk::Result::TIMEOUT);
                }
                let (next, _) = self
                    .changed
                    .wait_timeout(guard, remaining)
                    .unwrap_or_else(|e| e.into_inner());
                guard = next;
            }
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        *self
            .count
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            == 0
    }

    pub(crate) fn wait(&self) {
        let mut count = self
            .count
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        while *count != 0 {
            count = self
                .changed
                .wait(count)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }
}
struct Driver {
    sender: mpsc::Sender<Request>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    completion_sender: mpsc::Sender<CompletionRequest>,
    completion_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    queue: AtomicU64,
    queue_sender: mpsc::Sender<queue::Request>,
    queue_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    timeline_enabled: bool,
    synchronization2_enabled: bool,
    dynamic_rendering_enabled: bool,
    host_query_reset_enabled: bool,
    timestamps: bool,
    query_support: u32,
    occlusion_precise_enabled: bool,
    pipeline_statistics_enabled: bool,
    multi_draw_indirect_enabled: bool,
    indirect_first_instance_enabled: bool,
    indirect_count_enabled: bool,
    queries: Mutex<HashMap<u64, Arc<queries::Pool>>>,
    timelines: Mutex<HashMap<u64, Arc<timeline::Timeline>>>,
    events: Mutex<HashMap<u64, Arc<bootstrap::Event>>>,
    fences: Fences,
    semaphores: Semaphores,
    recordings: Recordings,
    in_flight: Arc<InFlight>,
    pending: Arc<InFlight>,
    lost: Arc<AtomicBool>,
}
impl Driver {
    fn mark_lost(&self) {
        self.lost.store(true, Ordering::Release);
        self.in_flight.notify();
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Device,
    Queue,
    Command,
}
#[repr(C)]
struct DispatchHandle {
    loader: usize,
}
type HandleRegistry = Mutex<HashMap<u64, (Kind, Arc<Driver>)>>;
static HANDLES: OnceLock<HandleRegistry> = OnceLock::new();
fn handles() -> &'static HandleRegistry {
    HANDLES.get_or_init(Default::default)
}
pub(crate) fn next_id() -> u64 {
    static ID: AtomicU64 = AtomicU64::new(1);
    ID.fetch_add(1, Ordering::Relaxed)
}
fn add_handle(kind: Kind, driver: &Arc<Driver>) -> u64 {
    let id = Box::into_raw(Box::new(DispatchHandle { loader: 0x01CDC0DE })) as u64;
    handles()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(id, (kind, Arc::clone(driver)));
    id
}
fn remove_handle(id: u64) {
    if handles()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&id)
        .is_some()
    {
        unsafe {
            drop(Box::from_raw(id as *mut DispatchHandle));
        }
    }
}
fn driver(id: u64, kind: Kind) -> VkResult<Arc<Driver>> {
    handles()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&id)
        .filter(|(k, _)| *k == kind)
        .map(|(_, d)| Arc::clone(d))
        .ok_or(vk::Result::ERROR_DEVICE_LOST)
}

/// Loaded Scarlet DSOs share the loader's process envp, while their private
/// Rust std environment is uninitialized. Copy values before returning them.
pub(crate) fn process_env(name: &str) -> Option<String> {
    #[cfg(target_os = "scarlet")]
    unsafe {
        unsafe extern "C" {
            #[linkage = "extern_weak"]
            static environ: *const *const *const c_char;
        }
        let address = environ;
        if address.is_null() {
            return None;
        }
        let mut cursor = *address;
        if cursor.is_null() {
            return None;
        }
        while !(*cursor).is_null() {
            let entry = CStr::from_ptr(*cursor).to_bytes();
            if let Some(value) = entry
                .strip_prefix(name.as_bytes())
                .and_then(|s| s.strip_prefix(b"="))
            {
                return std::str::from_utf8(value).ok().map(str::to_owned);
            }
            cursor = cursor.add(1);
        }
        None
    }
    #[cfg(not(target_os = "scarlet"))]
    {
        std::env::var(name).ok()
    }
}
fn call_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| process_env("SGFX_VULKAN_CALL_PROFILE").is_some_and(|v| v == "1"))
}

#[derive(Clone, Copy)]
struct CallServiceTiming {
    queued_ns: u128,
    service_ns: u128,
}

#[derive(Default)]
struct CallProfileCounters {
    calls: u64,
    errors: u64,
    services: u64,
    roundtrip_ns: u128,
    queued_ns: u128,
    service_ns: u128,
    max_roundtrip_ns: u128,
    max_service_ns: u128,
}

impl CallProfileCounters {
    fn observe(&mut self, roundtrip_ns: u128, service: Option<CallServiceTiming>, failed: bool) {
        self.calls += 1;
        self.errors += u64::from(failed);
        self.roundtrip_ns += roundtrip_ns;
        self.max_roundtrip_ns = self.max_roundtrip_ns.max(roundtrip_ns);
        if let Some(service) = service {
            self.services += 1;
            self.queued_ns += service.queued_ns;
            self.service_ns += service.service_ns;
            self.max_service_ns = self.max_service_ns.max(service.service_ns);
        }
    }
}

fn call_profile_label(operation: &str) -> &str {
    let operation = operation.strip_prefix("vulkan_sgfx::").unwrap_or(operation);
    let operation = operation.split("::{{closure}}").next().unwrap_or(operation);
    let mut start = operation.len().saturating_sub(120);
    while !operation.is_char_boundary(start) {
        start += 1;
    }
    &operation[start..]
}

fn record_call_profile(
    operation: &'static str,
    started: std::time::Instant,
    service: Option<CallServiceTiming>,
    failed: bool,
) {
    struct Profile {
        period: std::time::Instant,
        operations: HashMap<&'static str, CallProfileCounters>,
    }
    static PROFILE: OnceLock<Mutex<Profile>> = OnceLock::new();
    let finished = std::time::Instant::now();
    let mut profile = PROFILE
        .get_or_init(|| {
            Mutex::new(Profile {
                period: started,
                operations: HashMap::new(),
            })
        })
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    profile.operations.entry(operation).or_default().observe(
        finished.duration_since(started).as_nanos(),
        service,
        failed,
    );
    let period = finished.duration_since(profile.period);
    if period.as_secs() == 0 {
        return;
    }
    let mut top: Vec<_> = profile.operations.iter().collect();
    top.sort_unstable_by(|(a_name, a), (b_name, b)| {
        b.roundtrip_ns
            .cmp(&a.roundtrip_ns)
            .then_with(|| a_name.cmp(b_name))
    });
    let calls: u64 = top.iter().map(|(_, c)| c.calls).sum();
    let errors: u64 = top.iter().map(|(_, c)| c.errors).sum();
    eprintln!(
        "SGFX_VULKAN_CALL_PROFILE period_us={} calls={calls} errors={errors} operations={} top={}",
        period.as_micros(),
        top.len(),
        top.len().min(6),
    );
    for (operation, c) in top.iter().take(6) {
        // Service includes the operation, idle-cache reclamation, and lost-state
        // publication. Queue/wake delays are host thread/channel scheduling,
        // not evidence of GPU execution time. Missing service means channel loss.
        let wake_ns = c.roundtrip_ns.saturating_sub(c.queued_ns + c.service_ns);
        eprintln!(
            "SGFX_VULKAN_CALL_TOP op={} calls={} errors={} missing_service={} roundtrip_us={} queued_us={} service_us={} wake_us={} max_roundtrip_us={} max_service_us={}",
            call_profile_label(operation),
            c.calls,
            c.errors,
            c.calls - c.services,
            c.roundtrip_ns / 1000,
            c.queued_ns / 1000,
            c.service_ns / 1000,
            wake_ns / 1000,
            c.max_roundtrip_ns / 1000,
            c.max_service_ns / 1000,
        );
    }
    profile.operations.clear();
    profile.period = finished;
}

// Nested worker phases share the existing opt-in profiler. Their durations
// overlap the enclosing API call and must not be summed as separate GPU time.
fn profile_work<T>(operation: &'static str, work: impl FnOnce() -> VkResult<T>) -> VkResult<T> {
    let started = call_profile_enabled().then(std::time::Instant::now);
    let result = work();
    if let Some(started) = started {
        record_call_profile(
            operation,
            started,
            Some(CallServiceTiming {
                queued_ns: 0,
                service_ns: started.elapsed().as_nanos(),
            }),
            result.is_err(),
        );
    }
    result
}

// Both synchronous requests and owned void requests use the same panic/lost
// publication path. A void operation has no result channel, so its errors must
// become device loss instead of silently preserving stale state.
fn run_device_operation<T>(
    runtime: &mut Runtime,
    op: impl FnOnce(&mut Runtime) -> VkResult<T>,
    void: bool,
) -> VkResult<T> {
    let result = if runtime.device_lost.load(Ordering::Acquire) {
        runtime.lost = true;
        Err(vk::Result::ERROR_DEVICE_LOST)
    } else {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if !void {
                profile_work("worker::retire_images", || runtime.flush_retired_images())?;
            }
            let result = op(runtime);
            if runtime.retired_images.len() >= crate::runtime::IMAGE_RETIREMENT_BATCH {
                profile_work("worker::retire_images", || runtime.flush_retired_images())?;
            }
            result
        }))
        .unwrap_or_else(|_| {
            if std::env::var_os("SGFX_VULKAN_TRACE").is_some() {
                eprintln!("[SGFX Vulkan] device worker panicked; marking device lost");
            }
            runtime.lost = true;
            Err(vk::Result::ERROR_DEVICE_LOST)
        })
    };
    if void && result.is_err() {
        runtime.lost = true;
    }
    let _ = profile_work("worker::reclaim_idle_resources", || {
        runtime.reclaim_idle_resources();
        Ok(())
    });
    if runtime.lost {
        runtime.device_lost.store(true, Ordering::Release);
        runtime.in_flight.notify();
    }
    result
}

fn call<T: Send + 'static>(
    driver: &Driver,
    op: impl FnOnce(&mut Runtime) -> VkResult<T> + Send + 'static,
) -> VkResult<T> {
    // The disabled path performs no clock reads, type-name formatting, or
    // statistics-map access. The opt-in flag is sampled once per process.
    let profile_start = call_profile_enabled().then(std::time::Instant::now);
    let operation = profile_start.map(|_| std::any::type_name_of_val(&op));
    let mut service = None;
    let result = (|| {
        let (tx, rx) = mpsc::sync_channel(1);
        driver
            .sender
            .send(Request::Run(Box::new(move |runtime| {
                let service_start = profile_start.map(|_| std::time::Instant::now());
                let result = run_device_operation(runtime, op, false);
                let service =
                    profile_start
                        .zip(service_start)
                        .map(|(queued, start)| CallServiceTiming {
                            queued_ns: start.duration_since(queued).as_nanos(),
                            service_ns: start.elapsed().as_nanos(),
                        });
                let _ = tx.send((result, service));
            })))
            .map_err(|_| {
                if std::env::var_os("SGFX_VULKAN_TRACE").is_some() {
                    eprintln!("[SGFX Vulkan] device worker request channel closed");
                }
                driver.mark_lost();
                vk::Result::ERROR_DEVICE_LOST
            })?;
        let (result, timing) = rx.recv().map_err(|_| {
            if std::env::var_os("SGFX_VULKAN_TRACE").is_some() {
                eprintln!("[SGFX Vulkan] device worker response channel closed");
            }
            driver.mark_lost();
            vk::Result::ERROR_DEVICE_LOST
        })?;
        service = timing;
        result
    })();
    if let (Some(started), Some(operation)) = (profile_start, operation) {
        record_call_profile(operation, started, service, result.is_err());
    }
    result
}
pub(crate) fn with_device<T: Send + 'static>(
    device: vk::Device,
    op: impl FnOnce(&mut Runtime) -> VkResult<T> + Send + 'static,
) -> VkResult<T> {
    call(driver(device.as_raw(), Kind::Device)?.as_ref(), op)
}

/// Queue a fully owned void operation without waiting behind GPU submission.
pub(crate) fn with_device_owned(
    device: vk::Device,
    op: impl FnOnce(&mut Runtime) -> VkResult<()> + Send + 'static,
) -> VkResult<()> {
    enqueue_device_owned(driver(device.as_raw(), Kind::Device)?.as_ref(), op)
}

/// Publish fully owned descriptor metadata on the device FIFO without waiting
/// for a reply. Invalidation belongs to the API invocation, before a subsequent
/// command recording can bind the newly updated set. Doing it in the worker
/// would incorrectly invalidate recordings made after this call returns.
pub(crate) fn update_descriptors_owned(
    device: vk::Device,
    sets: &[vk::DescriptorSet],
    op: impl FnOnce(&mut Runtime) -> VkResult<()> + Send + 'static,
) -> VkResult<()> {
    let driver = driver(device.as_raw(), Kind::Device)?;
    invalidate_recorded_descriptor_sets(&driver.recordings, sets);
    enqueue_device_owned(&driver, op)
}

fn enqueue_device_owned(
    driver: &Driver,
    op: impl FnOnce(&mut Runtime) -> VkResult<()> + Send + 'static,
) -> VkResult<()> {
    // The synchronous call profiler intentionally excludes this one-way job:
    // enqueue-to-completion is not an API round trip or response wake delay.
    driver
        .sender
        .send(Request::Run(Box::new(move |runtime| {
            // There is no response receiver to notice a worker unwind. Also
            // publish loss if post-operation reclamation itself panics.
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_device_operation(runtime, op, true)
            }))
            .is_err()
            {
                runtime.lost = true;
                runtime.device_lost.store(true, Ordering::Release);
                runtime.in_flight.notify();
            }
        })))
        .map_err(|_| {
            driver.mark_lost();
            vk::Result::ERROR_DEVICE_LOST
        })
}

pub(crate) fn mark_device_lost(device: vk::Device) {
    if let Ok(driver) = driver(device.as_raw(), Kind::Device) {
        driver.mark_lost();
    }
}

#[cfg(any(
    target_os = "scarlet",
    target_os = "macos",
    all(target_os = "linux", feature = "scarlet-wsi")
))]
pub(crate) fn with_queue<T: Send + 'static>(
    queue: vk::Queue,
    op: impl FnOnce(&mut Runtime) -> VkResult<T> + Send + 'static,
) -> VkResult<T> {
    call(driver(queue.as_raw(), Kind::Queue)?.as_ref(), op)
}
fn status(value: VkResult<()>) -> vk::Result {
    value.map_or_else(|e| e, |_| vk::Result::SUCCESS)
}
pub(crate) unsafe fn slice<'a, T>(ptr: *const T, count: u32) -> VkResult<&'a [T]> {
    if count > 4096 || (count != 0 && ptr.is_null()) {
        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
    }
    Ok(if count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ptr, count as usize)
    })
}

fn core_device_features_supported(
    requested: &vk::PhysicalDeviceFeatures,
    supported: &vk::PhysicalDeviceFeatures,
) -> bool {
    // VkPhysicalDeviceFeatures is a C struct containing only VkBool32 fields.
    // Validate both the boolean representation and the advertised capability.
    let fields = std::mem::size_of::<vk::PhysicalDeviceFeatures>() / 4;
    unsafe {
        std::slice::from_raw_parts(
            (requested as *const vk::PhysicalDeviceFeatures).cast::<u32>(),
            fields,
        )
        .iter()
        .zip(std::slice::from_raw_parts(
            (supported as *const vk::PhysicalDeviceFeatures).cast::<u32>(),
            fields,
        ))
        .all(|(&request, &support)| {
            request == vk::FALSE || (request == vk::TRUE && support == vk::TRUE)
        })
    }
}

unsafe fn device_chain_supported(
    mut next: *const std::ffi::c_void,
    supported_features: &vk::PhysicalDeviceFeatures,
) -> bool {
    // The Khronos loader inserts private device-link records before calling an
    // ICD. They are transport metadata, not enabled application features.
    for _ in 0..32 {
        if next.is_null() {
            return true;
        }
        let header = &*next.cast::<vk::BaseInStructure<'_>>();
        let supported = match header.s_type {
            vk::StructureType::LOADER_DEVICE_CREATE_INFO => true,
            vk::StructureType::PHYSICAL_DEVICE_FEATURES_2 => {
                let f = &*next.cast::<vk::PhysicalDeviceFeatures2<'_>>();
                core_device_features_supported(&f.features, supported_features)
            }
            vk::StructureType::PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT => {
                let f = &*next.cast::<vk::PhysicalDeviceVertexAttributeDivisorFeaturesEXT<'_>>();
                matches!(
                    f.vertex_attribute_instance_rate_divisor,
                    vk::FALSE | vk::TRUE
                ) && f.vertex_attribute_instance_rate_zero_divisor == vk::FALSE
            }
            vk::StructureType::PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES => {
                let f = &*next.cast::<vk::PhysicalDeviceHostQueryResetFeatures<'_>>();
                matches!(f.host_query_reset, vk::FALSE | vk::TRUE)
            }
            vk::StructureType::PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES => {
                let f = &*next.cast::<vk::PhysicalDeviceDynamicRenderingFeatures<'_>>();
                matches!(f.dynamic_rendering, vk::FALSE | vk::TRUE)
            }
            vk::StructureType::PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES => {
                let f = &*next.cast::<vk::PhysicalDeviceSynchronization2Features<'_>>();
                matches!(f.synchronization2, vk::FALSE | vk::TRUE)
            }
            vk::StructureType::PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES => {
                let f = &*next.cast::<vk::PhysicalDeviceTimelineSemaphoreFeatures<'_>>();
                matches!(f.timeline_semaphore, vk::FALSE | vk::TRUE)
            }
            vk::StructureType::PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES => {
                let f = &*next.cast::<vk::PhysicalDeviceImagelessFramebufferFeatures<'_>>();
                matches!(f.imageless_framebuffer, vk::FALSE | vk::TRUE)
            }
            vk::StructureType::PHYSICAL_DEVICE_MULTIVIEW_FEATURES => {
                let f = &*next.cast::<vk::PhysicalDeviceMultiviewFeatures<'_>>();
                f.multiview == 0
                    && f.multiview_geometry_shader == 0
                    && f.multiview_tessellation_shader == 0
            }
            vk::StructureType::PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES => {
                let features = &*next.cast::<vk::PhysicalDeviceDescriptorIndexingFeatures<'_>>();
                // Applications may include an extension's feature record with
                // every feature disabled. This does not enable the extension.
                [
                    features.shader_input_attachment_array_dynamic_indexing,
                    features.shader_uniform_texel_buffer_array_dynamic_indexing,
                    features.shader_storage_texel_buffer_array_dynamic_indexing,
                    features.shader_uniform_buffer_array_non_uniform_indexing,
                    features.shader_sampled_image_array_non_uniform_indexing,
                    features.shader_storage_buffer_array_non_uniform_indexing,
                    features.shader_storage_image_array_non_uniform_indexing,
                    features.shader_input_attachment_array_non_uniform_indexing,
                    features.shader_uniform_texel_buffer_array_non_uniform_indexing,
                    features.shader_storage_texel_buffer_array_non_uniform_indexing,
                    features.descriptor_binding_uniform_buffer_update_after_bind,
                    features.descriptor_binding_sampled_image_update_after_bind,
                    features.descriptor_binding_storage_image_update_after_bind,
                    features.descriptor_binding_storage_buffer_update_after_bind,
                    features.descriptor_binding_uniform_texel_buffer_update_after_bind,
                    features.descriptor_binding_storage_texel_buffer_update_after_bind,
                    features.descriptor_binding_update_unused_while_pending,
                    features.descriptor_binding_partially_bound,
                    features.descriptor_binding_variable_descriptor_count,
                    features.runtime_descriptor_array,
                ]
                .into_iter()
                .all(|feature| feature == vk::FALSE)
            }
            vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_1_FEATURES => {
                let f = &*next.cast::<vk::PhysicalDeviceVulkan11Features<'_>>();
                [
                    f.storage_buffer16_bit_access,
                    f.uniform_and_storage_buffer16_bit_access,
                    f.storage_push_constant16,
                    f.storage_input_output16,
                    f.multiview,
                    f.multiview_geometry_shader,
                    f.multiview_tessellation_shader,
                    f.variable_pointers_storage_buffer,
                    f.variable_pointers,
                    f.protected_memory,
                    f.sampler_ycbcr_conversion,
                ]
                .into_iter()
                .all(|value| value == vk::FALSE)
                    && matches!(f.shader_draw_parameters, vk::FALSE | vk::TRUE)
            }
            vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_2_FEATURES => {
                crate::properties2::core12_request_supported(
                    &*next.cast::<vk::PhysicalDeviceVulkan12Features<'_>>(),
                )
            }
            vk::StructureType::PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES => {
                let f = &*next.cast::<vk::PhysicalDevice16BitStorageFeatures<'_>>();
                [
                    f.storage_buffer16_bit_access,
                    f.uniform_and_storage_buffer16_bit_access,
                    f.storage_push_constant16,
                    f.storage_input_output16,
                ]
                .into_iter()
                .all(|value| value == vk::FALSE)
            }
            vk::StructureType::PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES => {
                let f = &*next.cast::<vk::PhysicalDeviceVariablePointersFeatures<'_>>();
                f.variable_pointers == vk::FALSE && f.variable_pointers_storage_buffer == vk::FALSE
            }
            vk::StructureType::PHYSICAL_DEVICE_PROTECTED_MEMORY_FEATURES => {
                (&*next.cast::<vk::PhysicalDeviceProtectedMemoryFeatures<'_>>()).protected_memory
                    == vk::FALSE
            }
            vk::StructureType::PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES => {
                (&*next.cast::<vk::PhysicalDeviceSamplerYcbcrConversionFeatures<'_>>())
                    .sampler_ycbcr_conversion
                    == vk::FALSE
            }
            vk::StructureType::PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES => {
                let features = &*next.cast::<vk::PhysicalDeviceShaderDrawParametersFeatures<'_>>();
                matches!(features.shader_draw_parameters, vk::FALSE | vk::TRUE)
            }
            _ => false,
        };
        if !supported {
            return false;
        }
        next = header.p_next.cast();
    }
    false
}

unsafe fn device_extensions_supported(info: &vk::DeviceCreateInfo<'_>) -> bool {
    let Ok(names) = slice(
        info.pp_enabled_extension_names,
        info.enabled_extension_count,
    ) else {
        return false;
    };
    names.iter().all(|&name| {
        !name.is_null() && {
            let name = unsafe { CStr::from_ptr(name) };
            if [
                vk::KHR_TIMELINE_SEMAPHORE_NAME,
                vk::KHR_SYNCHRONIZATION2_NAME,
                vk::KHR_DYNAMIC_RENDERING_NAME,
                vk::KHR_DRAW_INDIRECT_COUNT_NAME,
                vk::EXT_VERTEX_ATTRIBUTE_DIVISOR_NAME,
                vk::KHR_PUSH_DESCRIPTOR_NAME,
                vk::EXT_HOST_QUERY_RESET_NAME,
            ]
            .contains(&name)
            {
                return true;
            }
            #[cfg(any(
                target_os = "scarlet",
                all(target_os = "linux", feature = "scarlet-wsi")
            ))]
            if [
                vk::KHR_MAINTENANCE1_NAME,
                vk::KHR_MAINTENANCE2_NAME,
                vk::KHR_MULTIVIEW_NAME,
                vk::KHR_CREATE_RENDERPASS2_NAME,
                vk::KHR_IMAGELESS_FRAMEBUFFER_NAME,
                vk::KHR_IMAGE_FORMAT_LIST_NAME,
                vk::KHR_DESCRIPTOR_UPDATE_TEMPLATE_NAME,
            ]
            .contains(&name)
            {
                return true;
            }
            #[cfg(any(
                target_os = "scarlet",
                target_os = "macos",
                all(target_os = "linux", feature = "scarlet-wsi")
            ))]
            if name == vk::KHR_SWAPCHAIN_NAME {
                return true;
            }
            #[cfg(target_os = "scarlet")]
            if name == crate::scarlet_image::DEVICE_EXTENSION_NAME {
                return true;
            }
            false
        }
    })
}

pub(crate) unsafe extern "system" fn create_device(
    physical: vk::PhysicalDevice,
    info: *const vk::DeviceCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::Device,
) -> vk::Result {
    let Some(adapter) = crate::instance::physical_adapter(physical) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    if info.is_null() || out.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    *out = vk::Device::null();
    let info = &*info;
    let mut supported_features = vk::PhysicalDeviceFeatures::default();
    crate::instance::get_physical_device_features(physical, &mut supported_features);
    if !allocator.is_null()
        || !device_chain_supported(info.p_next, &supported_features)
        || !info.flags.is_empty()
        || !device_extensions_supported(info)
    {
        return vk::Result::ERROR_FEATURE_NOT_PRESENT;
    }
    if !info.p_enabled_features.is_null()
        && !core_device_features_supported(&*info.p_enabled_features, &supported_features)
    {
        return vk::Result::ERROR_FEATURE_NOT_PRESENT;
    }
    let mut timeline_enabled = false;
    let mut synchronization2_enabled = false;
    let mut dynamic_rendering_enabled = false;
    let mut host_query_reset_enabled = false;
    let mut core12 = false;
    let timestamps = adapter.supports_timestamps();
    let query_support = adapter.query_support();
    let mut enabled_core = info
        .p_enabled_features
        .as_ref()
        .copied()
        .unwrap_or_default();
    let mut node = info.p_next.cast::<vk::BaseInStructure<'_>>();
    while let Some(base) = node.as_ref() {
        let draw_requested = match base.s_type {
            vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_1_FEATURES => {
                (*node.cast::<vk::PhysicalDeviceVulkan11Features<'_>>()).shader_draw_parameters
            }
            vk::StructureType::PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES => {
                (*node.cast::<vk::PhysicalDeviceShaderDrawParametersFeatures<'_>>())
                    .shader_draw_parameters
            }
            _ => vk::FALSE,
        };
        if draw_requested != vk::FALSE
            && (draw_requested != vk::TRUE
                || !adapter.capabilities().supports_shader_draw_parameters())
        {
            return vk::Result::ERROR_FEATURE_NOT_PRESENT;
        }

        if base.s_type == vk::StructureType::PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT
            && (*node.cast::<vk::PhysicalDeviceVertexAttributeDivisorFeaturesEXT<'_>>())
                .vertex_attribute_instance_rate_divisor
                == vk::TRUE
            && !adapter.capabilities().supports_vertex_divisors()
        {
            return vk::Result::ERROR_FEATURE_NOT_PRESENT;
        }

        if base.s_type == vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_2_FEATURES {
            let features = &*node.cast::<vk::PhysicalDeviceVulkan12Features<'_>>();
            let supported = crate::properties2::core12_features(physical);
            if (features.host_query_reset != 0 && supported.host_query_reset == 0)
                || (features.timeline_semaphore != 0 && supported.timeline_semaphore == 0)
                || (features.draw_indirect_count != 0 && supported.draw_indirect_count == 0)
            {
                return vk::Result::ERROR_FEATURE_NOT_PRESENT;
            }
            host_query_reset_enabled = features.host_query_reset == vk::TRUE;
            timeline_enabled = features.timeline_semaphore == vk::TRUE;
            core12 = true;
        }
        if base.s_type == vk::StructureType::PHYSICAL_DEVICE_FEATURES_2 {
            enabled_core = (*node.cast::<vk::PhysicalDeviceFeatures2<'_>>()).features;
        }
        if base.s_type == vk::StructureType::PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES {
            host_query_reset_enabled =
                (*node.cast::<vk::PhysicalDeviceHostQueryResetFeatures<'_>>()).host_query_reset
                    == vk::TRUE;
        }
        if base.s_type == vk::StructureType::PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES {
            dynamic_rendering_enabled =
                (*node.cast::<vk::PhysicalDeviceDynamicRenderingFeatures<'_>>()).dynamic_rendering
                    == vk::TRUE;
        }
        if base.s_type == vk::StructureType::PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES {
            synchronization2_enabled =
                (*node.cast::<vk::PhysicalDeviceSynchronization2Features<'_>>()).synchronization2
                    == vk::TRUE;
        }
        if base.s_type == vk::StructureType::PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES {
            timeline_enabled = (*node.cast::<vk::PhysicalDeviceTimelineSemaphoreFeatures<'_>>())
                .timeline_semaphore
                == vk::TRUE;
        }
        node = base.p_next;
    }
    if timeline_enabled
        && !core12
        && !slice(
            info.pp_enabled_extension_names,
            info.enabled_extension_count,
        )
        .unwrap()
        .iter()
        .any(|&n| CStr::from_ptr(n) == vk::KHR_TIMELINE_SEMAPHORE_NAME)
    {
        return vk::Result::ERROR_EXTENSION_NOT_PRESENT;
    }
    if synchronization2_enabled
        && !slice(
            info.pp_enabled_extension_names,
            info.enabled_extension_count,
        )
        .unwrap()
        .iter()
        .any(|&n| CStr::from_ptr(n) == vk::KHR_SYNCHRONIZATION2_NAME)
    {
        return vk::Result::ERROR_EXTENSION_NOT_PRESENT;
    }
    if dynamic_rendering_enabled
        && !slice(
            info.pp_enabled_extension_names,
            info.enabled_extension_count,
        )
        .unwrap()
        .iter()
        .any(|&n| CStr::from_ptr(n) == vk::KHR_DYNAMIC_RENDERING_NAME)
    {
        return vk::Result::ERROR_EXTENSION_NOT_PRESENT;
    }
    if host_query_reset_enabled
        && ((!timestamps && query_support == 0)
            || (!core12
                && !slice(
                    info.pp_enabled_extension_names,
                    info.enabled_extension_count,
                )
                .unwrap()
                .iter()
                .any(|&n| CStr::from_ptr(n) == vk::EXT_HOST_QUERY_RESET_NAME)))
    {
        return vk::Result::ERROR_FEATURE_NOT_PRESENT;
    }
    let qs = match slice(info.p_queue_create_infos, info.queue_create_info_count) {
        Ok(v) => v,
        Err(e) => return e,
    };
    if qs.len() != 1
        || qs[0].queue_family_index != 0
        || qs[0].queue_count != 1
        || !qs[0].flags.is_empty()
        || !qs[0].p_next.is_null()
        || qs[0].p_queue_priorities.is_null()
    {
        return vk::Result::ERROR_FEATURE_NOT_PRESENT;
    }
    let priority = *qs[0].p_queue_priorities;
    if !priority.is_finite() || !(0.0..=1.0).contains(&priority) {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let fences: Fences = Default::default();
    let semaphores: Semaphores = Default::default();
    let recordings: Recordings = Default::default();
    let in_flight = Arc::new(InFlight::default());
    let pending = Arc::new(InFlight::default());
    let lost = Arc::new(AtomicBool::new(false));
    let worker_recordings = Arc::clone(&recordings);
    let worker_in_flight = Arc::clone(&in_flight);
    let worker_lost = Arc::clone(&lost);
    let (tx, rx) = mpsc::channel();
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    let thread = match std::thread::Builder::new()
        .name("sgfx-vulkan-device".into())
        // Command resolution and shader lowering need more than Scarlet
        // native std's 64 KiB default stack, including calls into backend DSOs.
        .stack_size(2 * 1024 * 1024)
        .spawn(move || {
            let init = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                Runtime::new(adapter, worker_recordings, worker_in_flight, worker_lost)
            }));
            let mut runtime = match init {
                Ok(Ok(rt)) => rt,
                Ok(Err(e)) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
                Err(_) => {
                    let _ = ready_tx.send(Err(vk::Result::ERROR_INITIALIZATION_FAILED));
                    return;
                }
            };
            let _ = ready_tx.send(Ok(()));
            while let Ok(request) = rx.recv() {
                match request {
                    Request::Run(job) => job(&mut runtime),
                    Request::Stop => break,
                }
            }
        }) {
        Ok(t) => t,
        Err(_) => return vk::Result::ERROR_OUT_OF_HOST_MEMORY,
    };
    if let Err(e) = ready_rx
        .recv()
        .unwrap_or(Err(vk::Result::ERROR_INITIALIZATION_FAILED))
    {
        let _ = thread.join();
        return e;
    }
    let (completion_tx, completion_rx) = mpsc::channel();
    let completion_in_flight = Arc::clone(&in_flight);
    let completion_pending = Arc::clone(&pending);
    let completion_lost = Arc::clone(&lost);
    let completion_thread = match std::thread::Builder::new()
        .name("sgfx-vulkan-completion".into())
        .spawn(move || {
            while let Ok(request) = completion_rx.recv() {
                match request {
                    CompletionRequest::Observe {
                        submissions,
                        fence,
                        signals,
                        timeline_signals,
                    } => finish_submissions(
                        &submissions,
                        fence.as_ref(),
                        &signals,
                        &timeline_signals,
                        &completion_in_flight,
                        &completion_pending,
                        &completion_lost,
                    ),
                    CompletionRequest::Stop => break,
                }
            }
        }) {
        Ok(thread) => thread,
        Err(_) => {
            let _ = tx.send(Request::Stop);
            let _ = thread.join();
            return vk::Result::ERROR_OUT_OF_HOST_MEMORY;
        }
    };
    let (queue_sender, queue_thread) = match queue::start() {
        Ok(v) => v,
        Err(error) => {
            let _ = completion_tx.send(CompletionRequest::Stop);
            let _ = completion_thread.join();
            let _ = tx.send(Request::Stop);
            let _ = thread.join();
            return error;
        }
    };
    let driver = Arc::new(Driver {
        sender: tx,
        thread: Mutex::new(Some(thread)),
        completion_sender: completion_tx,
        completion_thread: Mutex::new(Some(completion_thread)),
        queue: AtomicU64::new(0),
        queue_sender,
        queue_thread: Mutex::new(Some(queue_thread)),
        timeline_enabled,
        synchronization2_enabled,
        dynamic_rendering_enabled,
        host_query_reset_enabled,
        timestamps,
        query_support,
        occlusion_precise_enabled: enabled_core.occlusion_query_precise == vk::TRUE,
        pipeline_statistics_enabled: enabled_core.pipeline_statistics_query == vk::TRUE,
        multi_draw_indirect_enabled: enabled_core.multi_draw_indirect == vk::TRUE,
        indirect_first_instance_enabled: enabled_core.draw_indirect_first_instance == vk::TRUE,
        indirect_count_enabled: slice(
            info.pp_enabled_extension_names,
            info.enabled_extension_count,
        )
        .unwrap()
        .iter()
        .any(|&n| CStr::from_ptr(n) == vk::KHR_DRAW_INDIRECT_COUNT_NAME),
        queries: Default::default(),
        timelines: Default::default(),
        events: Default::default(),
        fences,
        semaphores,
        recordings,
        in_flight,
        pending,
        lost,
    });
    let device = add_handle(Kind::Device, &driver);
    let queue = add_handle(Kind::Queue, &driver);
    driver.queue.store(queue, Ordering::Relaxed);
    *out = vk::Device::from_raw(device);
    vk::Result::SUCCESS
}
pub(crate) unsafe extern "system" fn destroy_device(
    device: vk::Device,
    _: *const vk::AllocationCallbacks<'_>,
) {
    let Ok(driver) = driver(device.as_raw(), Kind::Device) else {
        return;
    };
    let ids: Vec<_> = handles()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .filter(|(_, (_, d))| Arc::ptr_eq(d, &driver))
        .map(|(&id, _)| id)
        .collect();
    for id in ids {
        remove_handle(id);
    }
    driver.pending.wait();
    let _ = driver.queue_sender.send(queue::Request::Stop);
    if let Some(thread) = driver
        .queue_thread
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take()
    {
        let _ = thread.join();
    }
    let _ = driver.completion_sender.send(CompletionRequest::Stop);
    if let Some(thread) = driver
        .completion_thread
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take()
    {
        let _ = thread.join();
    }
    driver.mark_lost();
    let _ = driver.sender.send(Request::Stop);
    if let Some(thread) = driver
        .thread
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take()
    {
        let _ = thread.join();
    }
}
unsafe extern "system" fn get_device_queue(
    device: vk::Device,
    family: u32,
    index: u32,
    out: *mut vk::Queue,
) {
    if out.is_null() {
        return;
    }
    *out = vk::Queue::null();
    if family == 0
        && index == 0
        && let Ok(d) = driver(device.as_raw(), Kind::Device)
    {
        *out = vk::Queue::from_raw(d.queue.load(Ordering::Relaxed));
    }
}
unsafe extern "system" fn device_wait_idle(device: vk::Device) -> vk::Result {
    let driver = match driver(device.as_raw(), Kind::Device) {
        Ok(driver) => driver,
        Err(error) => return error,
    };
    status(wait_idle(&driver))
}
pub(crate) unsafe extern "system" fn queue_wait_idle(queue: vk::Queue) -> vk::Result {
    let driver = match driver(queue.as_raw(), Kind::Queue) {
        Ok(driver) => driver,
        Err(error) => return error,
    };
    status(wait_idle(&driver))
}

fn wait_idle(driver: &Driver) -> VkResult<()> {
    call(driver, |runtime| {
        if runtime.lost {
            Err(vk::Result::ERROR_DEVICE_LOST)
        } else {
            Ok(())
        }
    })?;
    driver.pending.wait();
    if driver.lost.load(Ordering::Acquire) {
        Err(vk::Result::ERROR_DEVICE_LOST)
    } else {
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RecordingState {
    Initial,
    Recording,
    Executable,
    Invalid,
}
#[derive(Clone)]
struct DescriptorInsertion {
    layout: ir::BindGroupLayoutDesc,
    position: usize,
    index: u32,
    set: vk::DescriptorSet,
    dynamic_offsets: Vec<u32>,
}
#[derive(Clone)]
struct DeferredBarrier {
    position: usize,
    buffers: Vec<vk::Buffer>,
    after: ir::BufferAccess,
}
#[derive(Clone)]
struct ReadImage {
    order: usize,
    mip_level: u32,
    position: usize,
    image: vk::Image,
    buffer: vk::Buffer,
    region: crate::transfer::ImageReadbackRegion,
}

type RecordedMemoryBarrier = (bool, vk::AccessFlags, vk::AccessFlags);
type RecordedBufferBarrier = (
    bool,
    vk::AccessFlags,
    vk::AccessFlags,
    u32,
    u32,
    vk::Buffer,
    u64,
    u64,
);
type RecordedImageBarrier = (
    bool,
    vk::AccessFlags,
    vk::AccessFlags,
    u32,
    u32,
    vk::Image,
    vk::ImageLayout,
    vk::ImageLayout,
    vk::ImageSubresourceRange,
);

#[derive(Clone)]
struct Recording {
    pool: u64,
    one_time: bool,
    state: RecordingState,
    error: Option<vk::Result>,
    render_active: bool,
    render_dynamic: bool,
    active_queries: BTreeMap<u32, (Arc<queries::Pool>, u32, bool)>,
    // Submit snapshots share immutable recorded data across the calling and
    // device threads. Re-recording uses copy-on-write instead of changing an
    // outstanding snapshot's commands.
    commands: Arc<Vec<RecordedCommand>>,
}

#[derive(Default)]
pub(crate) struct CommandRegistry {
    commands: HashMap<u64, RecordingCell>,
    pools: HashMap<u64, vk::CommandPoolCreateFlags>,
    // Immutable metadata permits recording without waiting for the GPU worker.
    pipeline_layouts: HashMap<vk::PipelineLayout, ir::PipelineLayoutDesc>,
}

pub(crate) fn set_pipeline_layout_metadata(
    device: vk::Device,
    handle: vk::PipelineLayout,
    layout: Option<ir::PipelineLayoutDesc>,
) -> VkResult<()> {
    let owner = driver(device.as_raw(), Kind::Device)?;
    let mut recordings = owner
        .recordings
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    match layout {
        Some(layout) => {
            recordings.pipeline_layouts.insert(handle, layout);
        }
        None => {
            recordings.pipeline_layouts.remove(&handle);
        }
    }
    Ok(())
}

#[derive(Clone)]
enum RecordedCommand {
    InertDynamicState,
    Query(queries::Action),
    BeginRendering(dynamic_rendering::Rendering),
    EndRendering,
    DrawIndirect(indirect::Draw),
    Synchronize(synchronization2::Action),
    CopyImage {
        source: vk::Image,
        source_layout: vk::ImageLayout,
        destination: vk::Image,
        destination_layout: vk::ImageLayout,
        regions: Vec<vk::ImageCopy>,
    },
    BlitImage {
        source: vk::Image,
        source_layout: vk::ImageLayout,
        destination: vk::Image,
        destination_layout: vk::ImageLayout,
        regions: Vec<vk::ImageBlit>,
        filter: vk::Filter,
    },
    PushConstants {
        layout: ir::PipelineLayoutDesc,
        stages: ir::ShaderStages,
        offset: u32,
        data: Vec<u8>,
    },
    BindPipeline {
        point: vk::PipelineBindPoint,
        pipeline: vk::Pipeline,
    },
    PushDescriptorWrites {
        point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        set: u32,
        writes: Vec<crate::resources::Write>,
        handle: vk::DescriptorSet,
        owner: std::sync::Weak<()>,
        history: push_descriptors::History,
    },
    BindDescriptorSets {
        point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        first: u32,
        sets: Vec<vk::DescriptorSet>,
        dynamic_offsets: Vec<u32>,
        pushed: Option<Arc<()>>,
    },
    Dispatch {
        x: u32,
        y: u32,
        z: u32,
    },
    BeginRenderPass {
        attachments: Option<Vec<vk::ImageView>>,
        contents: vk::SubpassContents,
        render_pass: vk::RenderPass,
        framebuffer: vk::Framebuffer,
        area: vk::Rect2D,
        clears: Vec<vk::ClearValue>,
    },
    NextSubpass,
    EndRenderPass,
    SetViewport(vk::Viewport),
    SetScissor(vk::Rect2D),
    BindVertexBuffer {
        slot: u32,
        buffer: vk::Buffer,
        offset: u64,
    },
    BindIndexBuffer {
        buffer: vk::Buffer,
        offset: u64,
        index_type: vk::IndexType,
    },
    Draw {
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    },
    DrawIndexed {
        indices: u32,
        instances: u32,
        first: u32,
        base_vertex: i32,
        first_instance: u32,
    },
    CopyImageToBuffer {
        image: vk::Image,
        layout: vk::ImageLayout,
        buffer: vk::Buffer,
        regions: Vec<vk::BufferImageCopy>,
    },
    ClearAttachments {
        attachments: Vec<vk::ClearAttachment>,
        rects: Vec<vk::ClearRect>,
    },
    ClearDepthImage {
        image: vk::Image,
        layout: vk::ImageLayout,
        depth: f32,
        ranges: Vec<vk::ImageSubresourceRange>,
    },
    ClearColorImage {
        image: vk::Image,
        layout: vk::ImageLayout,
        color: [f32; 4],
        ranges: Vec<vk::ImageSubresourceRange>,
    },
    CopyBuffer {
        source: vk::Buffer,
        destination: vk::Buffer,
        regions: Vec<vk::BufferCopy>,
    },
    UpdateBuffer {
        buffer: vk::Buffer,
        offset: u64,
        data: Vec<u8>,
    },
    FillBuffer {
        buffer: vk::Buffer,
        offset: u64,
        size: u64,
        word: u32,
    },
    CopyBufferToImage {
        source: vk::Buffer,
        image: vk::Image,
        layout: vk::ImageLayout,
        regions: Vec<vk::BufferImageCopy>,
    },
    PipelineBarrier {
        full_scope: bool,
        source_stage: vk::PipelineStageFlags,
        destination_stage: vk::PipelineStageFlags,
        flags: vk::DependencyFlags,
        memory: Vec<RecordedMemoryBarrier>,
        buffers: Vec<RecordedBufferBarrier>,
        images: Vec<RecordedImageBarrier>,
    },
}

#[derive(Clone)]
struct ResolvedRecording {
    push_constants: crate::push_constants::PushConstants,
    push_constants_revision: u64,
    emitted_graphics: EmittedGraphics,
    ops: Vec<ir::OwnedCommand>,
    descriptors: Vec<DescriptorInsertion>,
    copies: Vec<ReadImage>,
    barriers: Vec<DeferredBarrier>,
    synchronizations: Vec<synchronization2::Insertion>,
    order: usize,
    written_buffers: Vec<vk::Buffer>,
    compute: Option<vk::Pipeline>,
    graphics: Option<vk::Pipeline>,
    vertex_buffers: [Option<(vk::Buffer, u64)>; ir::MAX_VERTEX_BUFFERS],
    index_buffer: Option<(vk::Buffer, u64, ir::IndexFormat)>,
    bound_sets: Vec<vk::DescriptorSet>,
    compute_sets: BTreeMap<u32, vk::DescriptorSet>,
    graphics_sets: BTreeMap<u32, vk::DescriptorSet>,
    compute_offsets: BTreeMap<u32, Vec<u32>>,
    graphics_offsets: BTreeMap<u32, Vec<u32>>,
    render: Option<(u32, u32)>,
    active_render_pass: Option<render_passes::ActiveRenderPass>,
    viewport: Option<vk::Viewport>,
    scissor: Option<vk::Rect2D>,
    used_buffers: Vec<vk::Buffer>,
    readback_buffers: buffer_readback::Readbacks,
    used_images: Vec<vk::Image>,
    used_pipelines: Vec<vk::Pipeline>,
    used_framebuffers: Vec<vk::Framebuffer>,
    used_render_passes: Vec<vk::RenderPass>,
}

/// SGFX state belongs to one render pass, whereas Vulkan bindings survive
/// pass boundaries. Suppress repeated state only after it was validated and
/// emitted in this pass; every draw still resolves its live Vulkan objects.
#[derive(Clone, Default)]
struct EmittedGraphics {
    viewport: Option<ir::Viewport>,
    scissor: Option<ir::PixelRect>,
    pipeline: Option<ir::ProgrammableRenderPipelineId>,
    push_constants: Option<(ir::ProgrammableRenderPipelineId, u64)>,
    vertex_buffers: [Option<(ir::BufferId, u64)>; ir::MAX_VERTEX_BUFFERS],
    index_buffer: Option<(ir::BufferId, u64, ir::IndexFormat)>,
    descriptors: BTreeMap<u32, (vk::DescriptorSet, Vec<u32>)>,
}

fn emit_changed<T: Copy + PartialEq>(
    previous: &mut Option<T>,
    value: T,
    ops: &mut Vec<ir::OwnedCommand>,
    command: impl FnOnce(T) -> ir::OwnedCommand,
) {
    if *previous != Some(value) {
        ops.push(command(value));
        *previous = Some(value);
    }
}
impl Recording {
    fn new(pool: u64) -> Self {
        Self {
            pool,
            one_time: false,
            state: RecordingState::Initial,
            error: None,
            render_active: false,
            render_dynamic: false,
            active_queries: Default::default(),
            commands: Arc::default(),
        }
    }
    fn fail(&mut self, e: vk::Result) {
        self.error.get_or_insert(e);
    }
    fn push(&mut self, command: RecordedCommand) -> VkResult<()> {
        if self.state != RecordingState::Recording {
            self.fail(vk::Result::ERROR_INITIALIZATION_FAILED);
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        if self.commands.len() >= ir::MAX_COMMANDS {
            self.fail(vk::Result::ERROR_OUT_OF_HOST_MEMORY);
            return Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY);
        }
        match command {
            RecordedCommand::Query(
                queries::Action::Reset(ref pool, first, count)
                | queries::Action::Copy {
                    ref pool,
                    first,
                    count,
                    ..
                },
            ) if self.active_queries.values().any(|(p, i, _)| {
                Arc::ptr_eq(p, pool)
                    && *i >= first
                    && u64::from(*i) < u64::from(first) + u64::from(count)
            }) =>
            {
                self.fail(vk::Result::ERROR_INITIALIZATION_FAILED);
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            }
            RecordedCommand::Query(queries::Action::Begin(ref pool, index)) => {
                if self.active_queries.contains_key(&pool.kind) {
                    self.fail(vk::Result::ERROR_INITIALIZATION_FAILED);
                    return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                }
                self.active_queries
                    .insert(pool.kind, (pool.clone(), index, self.render_active));
            }
            RecordedCommand::Query(queries::Action::End(ref pool, index)) => {
                if !self
                    .active_queries
                    .get(&pool.kind)
                    .is_some_and(|(p, i, inside)| {
                        Arc::ptr_eq(p, pool) && *i == index && *inside == self.render_active
                    })
                {
                    self.fail(vk::Result::ERROR_INITIALIZATION_FAILED);
                    return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                }
                self.active_queries.remove(&pool.kind);
            }
            RecordedCommand::EndRenderPass
            | RecordedCommand::EndRendering
            | RecordedCommand::NextSubpass
                if self.active_queries.values().any(|(_, _, inside)| *inside) =>
            {
                self.fail(vk::Result::ERROR_INITIALIZATION_FAILED);
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            }
            RecordedCommand::BeginRendering(_) if self.render_active => {
                self.fail(vk::Result::ERROR_INITIALIZATION_FAILED);
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            }
            RecordedCommand::BeginRendering(_) => {
                self.render_active = true;
                self.render_dynamic = true;
            }
            RecordedCommand::EndRendering if !self.render_active || !self.render_dynamic => {
                self.fail(vk::Result::ERROR_INITIALIZATION_FAILED);
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            }
            RecordedCommand::EndRendering => {
                self.render_active = false;
                self.render_dynamic = false;
            }
            RecordedCommand::NextSubpass | RecordedCommand::EndRenderPass
                if self.render_dynamic =>
            {
                self.fail(vk::Result::ERROR_INITIALIZATION_FAILED);
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            }
            RecordedCommand::BeginRenderPass { .. } if self.render_active => {
                self.fail(vk::Result::ERROR_INITIALIZATION_FAILED);
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            }
            RecordedCommand::BeginRenderPass { .. } => {
                self.render_active = true;
                self.render_dynamic = false;
            }
            RecordedCommand::EndRenderPass | RecordedCommand::NextSubpass
                if !self.render_active =>
            {
                self.fail(vk::Result::ERROR_INITIALIZATION_FAILED);
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            }
            RecordedCommand::EndRenderPass => self.render_active = false,
            RecordedCommand::Query(queries::Action::Reset(..) | queries::Action::Copy { .. })
                if self.render_active =>
            {
                self.fail(vk::Result::ERROR_INITIALIZATION_FAILED);
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            }
            RecordedCommand::Draw { .. }
            | RecordedCommand::DrawIndexed { .. }
            | RecordedCommand::DrawIndirect(_)
            | RecordedCommand::ClearAttachments { .. }
                if !self.render_active =>
            {
                self.fail(vk::Result::ERROR_INITIALIZATION_FAILED);
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            }
            RecordedCommand::Dispatch { .. }
            | RecordedCommand::PipelineBarrier { .. }
            | RecordedCommand::Synchronize(_)
                if self.render_active =>
            {
                self.fail(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
            }
            RecordedCommand::CopyImageToBuffer { .. }
            | RecordedCommand::ClearColorImage { .. }
            | RecordedCommand::ClearDepthImage { .. }
            | RecordedCommand::BlitImage { .. }
            | RecordedCommand::CopyImage { .. }
            | RecordedCommand::CopyBuffer { .. }
            | RecordedCommand::CopyBufferToImage { .. }
            | RecordedCommand::UpdateBuffer { .. }
            | RecordedCommand::FillBuffer { .. }
                if self.render_active =>
            {
                self.fail(vk::Result::ERROR_INITIALIZATION_FAILED);
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            }
            _ => {}
        }
        Arc::make_mut(&mut self.commands).push(command);
        Ok(())
    }
}

impl ResolvedRecording {
    fn new() -> Self {
        Self {
            push_constants: Default::default(),
            push_constants_revision: 0,
            emitted_graphics: Default::default(),
            ops: Vec::new(),
            descriptors: Vec::new(),
            copies: Vec::new(),
            barriers: Vec::new(),
            synchronizations: Vec::new(),
            order: 0,
            written_buffers: Vec::new(),
            compute: None,
            graphics: None,
            vertex_buffers: [None; ir::MAX_VERTEX_BUFFERS],
            index_buffer: None,
            bound_sets: Vec::new(),
            compute_sets: BTreeMap::new(),
            graphics_sets: BTreeMap::new(),
            compute_offsets: BTreeMap::new(),
            graphics_offsets: BTreeMap::new(),
            render: None,
            active_render_pass: None,
            viewport: None,
            scissor: None,
            used_buffers: Vec::new(),
            readback_buffers: buffer_readback::Readbacks::default(),
            used_images: Vec::new(),
            used_pipelines: Vec::new(),
            used_framebuffers: Vec::new(),
            used_render_passes: Vec::new(),
        }
    }
}

impl RecordedCommand {
    fn apply(&self, rt: &mut Runtime, rec: &mut ResolvedRecording) -> VkResult<()> {
        rec.order += 1;
        match self {
            Self::Query(action) => queries::resolve(rt, rec, action)?,
            Self::BeginRendering(rendering) => dynamic_rendering::begin(rt, rec, rendering)?,
            Self::EndRendering => render_passes::end(rec)?,
            Self::DrawIndirect(draw) => indirect::resolve(rt, rec, draw)?,
            Self::Synchronize(action) => {
                if rec.render.is_some() {
                    return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                }
                rec.synchronizations.push(synchronization2::Insertion {
                    position: rec.ops.len(),
                    order: rec.order,
                    action: action.clone(),
                });
            }
            Self::InertDynamicState => (),
            Self::CopyImage {
                source,
                source_layout,
                destination,
                destination_layout,
                regions,
            } => {
                if rec.render.is_some() {
                    return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                }
                rec.ops.extend(crate::transfer::copy_image(
                    &rt.resources,
                    *source,
                    *source_layout,
                    *destination,
                    *destination_layout,
                    regions,
                )?);
                rec.used_images.extend([*source, *destination]);
            }
            Self::BlitImage {
                source,
                source_layout,
                destination,
                destination_layout,
                regions,
                filter,
            } => {
                if rec.render.is_some() || !rt.capabilities.supports_image_blits() {
                    return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                }
                rec.ops.extend(crate::transfer::blit(
                    &rt.resources,
                    *source,
                    *source_layout,
                    *destination,
                    *destination_layout,
                    regions,
                    *filter,
                )?);
                rec.used_images.extend([*source, *destination]);
            }
            Self::PushConstants {
                layout,
                stages,
                offset,
                data,
            } => {
                if rec.push_constants.update(layout, *stages, *offset, data)? {
                    rec.push_constants_revision += 1;
                }
            }
            Self::BindPipeline { point, pipeline } => {
                match (*point, rt.resources.pipelines.get(pipeline)) {
                    (
                        vk::PipelineBindPoint::COMPUTE,
                        Some(crate::resources::Pipeline::Compute(_)),
                    ) => rec.compute = Some(*pipeline),
                    (
                        vk::PipelineBindPoint::GRAPHICS,
                        Some(crate::resources::Pipeline::Graphics(_)),
                    ) => rec.graphics = Some(*pipeline),
                    _ => return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT),
                }
                rec.used_pipelines.push(*pipeline);
            }
            Self::PushDescriptorWrites {
                point,
                layout,
                set,
                writes,
                handle,
                owner,
                history,
            } => {
                debug_assert!(matches!(
                    *point,
                    vk::PipelineBindPoint::GRAPHICS | vk::PipelineBindPoint::COMPUTE
                ));
                push_descriptors::resolve(rt, *layout, *set, writes, *handle, owner, history)?;
            }
            Self::BindDescriptorSets {
                point,
                layout,
                first,
                sets,
                dynamic_offsets,
                ..
            } => {
                if first
                    .checked_add(sets.len() as u32)
                    .is_none_or(|count| count > 4)
                {
                    return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                }
                let binding_maps = rt
                    .resources
                    .pipeline_binding_maps
                    .get(layout)
                    .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                let binding_stages = rt
                    .resources
                    .pipeline_binding_stages
                    .get(layout)
                    .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                let layout = rt
                    .resources
                    .pipeline_layouts
                    .get(layout)
                    .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                let (target, offsets) = if *point == vk::PipelineBindPoint::COMPUTE {
                    (&mut rec.compute_sets, &mut rec.compute_offsets)
                } else if *point == vk::PipelineBindPoint::GRAPHICS {
                    (&mut rec.graphics_sets, &mut rec.graphics_offsets)
                } else {
                    return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                };
                let mut dynamic_index = 0;
                for (index, set) in sets.iter().enumerate() {
                    let descriptor = rt
                        .resources
                        .descriptor_sets
                        .get(set)
                        .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                    let group_index = *first as usize + index;
                    if layout.bind_groups().get(group_index) != Some(&descriptor.layout)
                        || binding_maps.get(group_index) != Some(descriptor.binding_map())
                        || binding_stages.get(group_index) != Some(&descriptor.stages)
                    {
                        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                    }
                    target.insert(*first + index as u32, *set);
                    let count = descriptor
                        .types
                        .values()
                        .filter(|ty| {
                            matches!(
                                **ty,
                                vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                                    | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
                            )
                        })
                        .count();
                    let end = dynamic_index + count;
                    let values = dynamic_offsets
                        .get(dynamic_index..end)
                        .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                    let dynamic_types = descriptor.types.values().filter(|ty| {
                        matches!(
                            **ty,
                            vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                                | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
                        )
                    });
                    if values.iter().zip(dynamic_types).any(|(offset, ty)| {
                        let limit = if *ty == vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC {
                            rt.capabilities
                                .limits()
                                .min_uniform_buffer_offset_alignment
                                .max(16)
                        } else {
                            rt.capabilities
                                .limits()
                                .min_storage_buffer_offset_alignment
                                .max(4)
                        };
                        !offset.is_multiple_of(limit)
                    }) {
                        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                    }
                    offsets.insert(*first + index as u32, values.to_vec());
                    dynamic_index = end;
                    rec.bound_sets.push(*set);
                }
                if dynamic_index != dynamic_offsets.len() {
                    return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                }
            }
            Self::Dispatch { x, y, z } => {
                if rec.render.is_some() || *x > 65535 || *y > 65535 || *z > 65535 {
                    return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                }
                let pipeline = rec.compute.ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                let id = rt.resources.compute_pipeline_for_views(
                    &rt.table,
                    pipeline,
                    &rec.compute_sets,
                )?;
                rec.ops.push(ir::OwnedCommand::BeginComputePass);
                rec.ops.push(ir::OwnedCommand::SetComputePipeline(id));
                let desc = rt
                    .table
                    .compute_pipeline(
                        rt.table
                            .compute_pipeline_ref(id)
                            .map_err(crate::resources::failure)?,
                    )
                    .map_err(crate::resources::failure)?;
                rec.ops.extend(rec.push_constants.snapshot(desc.layout())?);
                let active_sets = active_sets(&rt.resources, desc.layout(), &rec.compute_sets)?;
                for &(index, set) in &active_sets {
                    rec.descriptors.push(DescriptorInsertion {
                        layout: desc.layout().bind_groups()[index as usize].clone(),
                        position: rec.ops.len(),
                        index,
                        set,
                        dynamic_offsets: rec
                            .compute_offsets
                            .get(&index)
                            .cloned()
                            .unwrap_or_default(),
                    });
                }
                mark_writable_descriptor_buffers(
                    &rt.resources,
                    rec,
                    &active_sets,
                    desc.layout(),
                    vk::PipelineBindPoint::COMPUTE,
                )?;
                if *x != 0 && *y != 0 && *z != 0 {
                    rec.ops.push(ir::OwnedCommand::Dispatch {
                        x: *x,
                        y: *y,
                        z: *z,
                    });
                }
                rec.ops.push(ir::OwnedCommand::EndComputePass);
            }
            Self::BeginRenderPass {
                attachments,
                contents,
                render_pass,
                framebuffer,
                area,
                clears,
            } => {
                if *contents != vk::SubpassContents::INLINE || rec.render.is_some() {
                    return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                }
                render_passes::begin(
                    rt,
                    rec,
                    *render_pass,
                    *framebuffer,
                    *area,
                    clears,
                    attachments.as_deref(),
                )?;
            }
            Self::NextSubpass => render_passes::next(rec)?,
            Self::ClearAttachments { attachments, rects } => {
                render_passes::clear_attachments(rec, attachments, rects)?;
            }
            Self::ClearDepthImage {
                image,
                layout,
                depth,
                ranges,
            } => {
                if rec.render.is_some() {
                    return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                }
                rec.ops.extend(crate::clear::depth_commands(
                    rt, *image, *layout, *depth, ranges,
                )?);
                rec.used_images.push(*image);
            }
            Self::ClearColorImage {
                image,
                layout,
                color,
                ranges,
            } => {
                let data = rt
                    .resources
                    .images
                    .get(image)
                    .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                let commands = crate::clear::color_commands(data, *layout, *color, ranges)?;
                rec.ops.extend(commands);
                rec.used_images.push(*image);
            }
            Self::EndRenderPass => render_passes::end(rec)?,
            Self::SetViewport(viewport) => rec.viewport = Some(*viewport),
            Self::SetScissor(scissor) => rec.scissor = Some(*scissor),
            Self::BindVertexBuffer {
                slot,
                buffer,
                offset,
            } => {
                let data = rt
                    .resources
                    .buffers
                    .get(buffer)
                    .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                if data.bound.is_none()
                    || !data.usage.contains(vk::BufferUsageFlags::VERTEX_BUFFER)
                    || *offset >= data.size
                    || !offset.is_multiple_of(4)
                {
                    return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                }
                rec.vertex_buffers[*slot as usize] = Some((*buffer, *offset));
                rec.used_buffers.push(*buffer);
            }
            Self::BindIndexBuffer {
                buffer,
                offset,
                index_type,
            } => {
                let format = match *index_type {
                    vk::IndexType::UINT16 => ir::IndexFormat::Uint16,
                    vk::IndexType::UINT32 => ir::IndexFormat::Uint32,
                    _ => return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT),
                };
                let data = rt
                    .resources
                    .buffers
                    .get(buffer)
                    .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                if data.bound.is_none()
                    || !data.usage.contains(vk::BufferUsageFlags::INDEX_BUFFER)
                    || *offset >= data.size
                    || !offset.is_multiple_of(format.byte_size())
                {
                    return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                }
                rec.index_buffer = Some((*buffer, *offset, format));
                rec.used_buffers.push(*buffer);
            }
            Self::Draw {
                vertices,
                instances,
                first,
                first_instance,
            } => {
                if rt.capabilities.supports_shader_draw_parameters() {
                    rec.ops.push(ir::OwnedCommand::SetDrawIndex(0));
                }
                if first_instance.checked_add(*instances).is_none() {
                    return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                }
                let visible = graphics_bindings(&rt.table, &mut rt.resources, rec)?;
                if visible && *vertices != 0 && *instances != 0 {
                    rec.ops.push(if *instances == 1 && *first_instance == 0 {
                        ir::OwnedCommand::Draw {
                            vertex_count: *vertices,
                            first_vertex: *first,
                        }
                    } else {
                        ir::OwnedCommand::DrawInstanced {
                            vertex_count: *vertices,
                            first_vertex: *first,
                            instance_count: *instances,
                            first_instance: *first_instance,
                        }
                    });
                }
            }
            Self::DrawIndexed {
                indices,
                instances,
                first,
                base_vertex,
                first_instance,
            } => {
                if rt.capabilities.supports_shader_draw_parameters() {
                    rec.ops.push(ir::OwnedCommand::SetDrawIndex(0));
                }
                if first_instance.checked_add(*instances).is_none() {
                    return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                }
                let visible = graphics_bindings(&rt.table, &mut rt.resources, rec)?;
                let (handle, offset, format) = rec
                    .index_buffer
                    .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                let buffer = rt
                    .resources
                    .buffers
                    .get(&handle)
                    .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                emit_changed(
                    &mut rec.emitted_graphics.index_buffer,
                    (buffer.id, offset, format),
                    &mut rec.ops,
                    |(buffer, offset, format)| ir::OwnedCommand::SetIndexBuffer {
                        buffer,
                        offset,
                        format,
                    },
                );
                if visible && *indices != 0 && *instances != 0 {
                    rec.ops.push(if *instances == 1 && *first_instance == 0 {
                        ir::OwnedCommand::DrawIndexed {
                            index_count: *indices,
                            first_index: *first,
                            base_vertex: *base_vertex,
                        }
                    } else {
                        ir::OwnedCommand::DrawIndexedInstanced {
                            index_count: *indices,
                            first_index: *first,
                            base_vertex: *base_vertex,
                            instance_count: *instances,
                            first_instance: *first_instance,
                        }
                    });
                }
            }
            Self::CopyImageToBuffer {
                image,
                layout,
                buffer,
                regions,
            } => {
                if rec.render.is_some()
                    || !matches!(
                        *layout,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL | vk::ImageLayout::GENERAL
                    )
                {
                    return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                }
                let image_data = rt
                    .resources
                    .images
                    .get(image)
                    .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                let buffer_data = rt
                    .resources
                    .buffers
                    .get(buffer)
                    .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                if !image_data.usable()
                    || buffer_data.bound.is_none()
                    || !buffer_data
                        .usage
                        .contains(vk::BufferUsageFlags::TRANSFER_DST)
                    || !image_data.usage.contains(vk::ImageUsageFlags::TRANSFER_SRC)
                    || !matches!(
                        image_data.format,
                        vk::Format::D32_SFLOAT
                            | vk::Format::R8G8B8A8_UNORM
                            | vk::Format::B8G8R8A8_UNORM
                            | vk::Format::A8B8G8R8_UNORM_PACK32
                            | vk::Format::R8G8B8A8_SRGB
                            | vk::Format::B8G8R8A8_SRGB
                            | vk::Format::A8B8G8R8_SRGB_PACK32
                            | vk::Format::R8_UNORM
                            | vk::Format::R8G8_UNORM
                            | vk::Format::R16G16B16A16_SFLOAT
                            | vk::Format::R32G32B32A32_SFLOAT
                    )
                {
                    return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                }
                for region in regions {
                    let extent = image_data.mip_extent(region.image_subresource.mip_level)?;
                    let bytes_per_pixel = crate::images::texture_format(image_data.format)
                        .and_then(|format| format.bytes_per_pixel())
                        .ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)?;
                    let readback = crate::transfer::ImageReadbackRegion::with_aspect(
                        extent, bytes_per_pixel, buffer_data.size, region, crate::images::image_aspect(image_data.format),
                    ).inspect_err(|error| {
                        if std::env::var_os("SGFX_VULKAN_TRACE").is_some() {
                            eprintln!("[SGFX Vulkan] image readback region rejected: {error:?} format={:?} extent={extent:?} buffer_size={} region={region:?}", image_data.format, buffer_data.size);
                        }
                    })?;
                    for range in readback.destination_ranges() {
                        rec.readback_buffers.insert(
                            *buffer,
                            range.start,
                            range.end - range.start,
                        )?;
                    }
                    rec.copies.push(ReadImage {
                        order: rec.order,
                        mip_level: region.image_subresource.mip_level,
                        position: rec.ops.len(),
                        image: *image,
                        buffer: *buffer,
                        region: readback,
                    });
                }
                rec.used_buffers.push(*buffer);
                rec.used_images.push(*image);
            }
            Self::UpdateBuffer {
                buffer,
                offset,
                data,
            } => {
                buffer_operations::apply_write(&rt.resources, rec, *buffer, *offset, data.clone())?
            }
            Self::FillBuffer {
                buffer,
                offset,
                size,
                word,
            } => {
                let size_bytes = rt
                    .resources
                    .buffers
                    .get(buffer)
                    .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?
                    .size;
                let data = buffer_operations::fill_data(size_bytes, *offset, *size, *word)?;
                buffer_operations::apply_write(&rt.resources, rec, *buffer, *offset, data)?;
            }
            Self::CopyBufferToImage {
                source,
                image,
                layout,
                regions,
            } => {
                if rec.render.is_some() || rec.readback_buffers.contains(source) {
                    return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                }
                rec.ops.extend(crate::transfer::upload(
                    &rt.resources,
                    *source,
                    *image,
                    *layout,
                    regions,
                )?);
                // upload() already snapshots the source into WriteTexture ops.
                // No SGFX command reads this buffer, so staging its entire
                // contents again in execute() is redundant.
                rec.used_images.push(*image);
            }
            Self::CopyBuffer {
                source,
                destination,
                regions,
            } => {
                if rec.render.is_some() {
                    return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                }
                let source_data = rt
                    .resources
                    .buffers
                    .get(source)
                    .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                let destination_data = rt
                    .resources
                    .buffers
                    .get(destination)
                    .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                if source_data.bound.is_none()
                    || destination_data.bound.is_none()
                    || !source_data
                        .usage
                        .contains(vk::BufferUsageFlags::TRANSFER_SRC)
                    || !destination_data
                        .usage
                        .contains(vk::BufferUsageFlags::TRANSFER_DST)
                {
                    return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                }
                for region in regions {
                    if region
                        .dst_offset
                        .checked_add(region.size)
                        .is_none_or(|end| end > destination_data.size)
                        || region
                            .src_offset
                            .checked_add(region.size)
                            .is_none_or(|end| end > source_data.size)
                    {
                        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                    }
                    rec.ops.push(ir::OwnedCommand::CopyBufferToBuffer {
                        source: source_data.id,
                        source_offset: region.src_offset,
                        destination: destination_data.id,
                        destination_offset: region.dst_offset,
                        size: region.size,
                    });
                    rec.readback_buffers
                        .insert(*destination, region.dst_offset, region.size)?;
                }
                rec.used_buffers.extend([*source, *destination]);
            }
            Self::PipelineBarrier {
                full_scope,
                source_stage,
                destination_stage,
                flags,
                memory,
                buffers,
                images,
            } => {
                if rec.render.is_some()
                    || validate_pipeline_barrier_header(*source_stage, *destination_stage, *flags)
                        .is_err()
                {
                    return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                }
                for (extended, source, destination) in memory {
                    if *extended {
                        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                    }
                    if !supported_access(*source) || !supported_access(*destination) {
                        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                    }
                    if source
                        .intersects(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::MEMORY_WRITE)
                    {
                        rec.barriers.push(DeferredBarrier {
                            position: rec.ops.len(),
                            buffers: std::mem::take(&mut rec.written_buffers),
                            after: buffer_access(if *full_scope && destination.is_empty() {
                                vk::AccessFlags::MEMORY_READ
                            } else {
                                *destination
                            })?,
                        });
                    }
                }
                for (
                    extended,
                    source,
                    destination,
                    source_queue,
                    destination_queue,
                    buffer,
                    offset,
                    size,
                ) in buffers
                {
                    if *extended || !same_queue_family(*source_queue, *destination_queue) {
                        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                    }
                    let buffer_data = rt
                        .resources
                        .buffers
                        .get(buffer)
                        .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                    let bytes = if *size == vk::WHOLE_SIZE {
                        buffer_data.size.checked_sub(*offset)
                    } else {
                        Some(*size)
                    };
                    if bytes.is_none_or(|bytes| {
                        bytes == 0
                            || offset
                                .checked_add(bytes)
                                .is_none_or(|end| end > buffer_data.size)
                    }) {
                        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                    }
                    if !*full_scope
                        && (*offset != 0 || (*size != vk::WHOLE_SIZE && *size != buffer_data.size))
                    {
                        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                    }
                    rec.used_buffers.push(*buffer);
                    if *full_scope && source.is_empty() && destination.is_empty() {
                        continue;
                    }
                    let before = if *full_scope {
                        synchronization2::buffer_access(*source, *destination, buffer_data.usage)?
                    } else {
                        buffer_access(*source)?
                    };
                    let after = if *full_scope {
                        synchronization2::buffer_access(*destination, *source, buffer_data.usage)?
                    } else {
                        buffer_access(*destination)?
                    };
                    rec.ops.push(ir::OwnedCommand::ResourceBarrier(
                        ir::OwnedResourceBarrier::Buffer {
                            buffer: buffer_data.id,
                            before,
                            after,
                        },
                    ));
                    rec.used_buffers.push(*buffer);
                }
                for (
                    extended,
                    source_access,
                    destination_access,
                    source_queue,
                    destination_queue,
                    image,
                    old_layout,
                    new_layout,
                    range,
                ) in images
                {
                    if *extended || !same_queue_family(*source_queue, *destination_queue) {
                        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                    }
                    let image_data = rt
                        .resources
                        .images
                        .get(image)
                        .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                    if range.aspect_mask != crate::images::image_aspect(image_data.format) {
                        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                    }
                    let layers = if range.layer_count == vk::REMAINING_ARRAY_LAYERS {
                        image_data.array_layers.checked_sub(range.base_array_layer)
                    } else {
                        Some(range.layer_count)
                    }
                    .filter(|count| *count > 0)
                    .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                    if range
                        .base_array_layer
                        .checked_add(layers)
                        .is_none_or(|end| end > image_data.array_layers)
                    {
                        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                    }
                    let level_count = if range.level_count == vk::REMAINING_MIP_LEVELS {
                        image_data.mip_levels.checked_sub(range.base_mip_level)
                    } else {
                        Some(range.level_count)
                    }
                    .filter(|count| *count > 0)
                    .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                    let end = range
                        .base_mip_level
                        .checked_add(level_count)
                        .filter(|end| *end <= image_data.mip_levels)
                        .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                    // Presentation is outside command IR. These persistent
                    // swapchain images retain color-attachment storage while
                    // presented and can be transitioned back to transfer use.
                    let access = |layout, mask: vk::AccessFlags| {
                        if layout == vk::ImageLayout::PRESENT_SRC_KHR
                            && image_data.swapchain.is_some()
                        {
                            Ok(ir::TextureAccess::RenderAttachment)
                        } else if layout == vk::ImageLayout::GENERAL {
                            if mask.intersects(
                                vk::AccessFlags::COLOR_ATTACHMENT_READ
                                    | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                            ) {
                                Ok(ir::TextureAccess::RenderAttachment)
                            } else if mask.intersects(
                                vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::MEMORY_WRITE,
                            ) {
                                Ok(ir::TextureAccess::StorageWrite)
                            } else if mask.contains(vk::AccessFlags::TRANSFER_WRITE) {
                                Ok(ir::TextureAccess::CopyDestination)
                            } else if mask.contains(vk::AccessFlags::TRANSFER_READ) {
                                Ok(ir::TextureAccess::CopySource)
                            } else {
                                Ok(ir::TextureAccess::Sampled)
                            }
                        } else {
                            texture_access(layout)
                        }
                    };
                    let after = access(*new_layout, *destination_access)?;
                    if *old_layout != vk::ImageLayout::UNDEFINED {
                        for mip_level in range.base_mip_level..end {
                            rec.ops.push(ir::OwnedCommand::ResourceBarrier(
                                ir::OwnedResourceBarrier::TextureMip {
                                    mip_level,
                                    texture: image_data.id,
                                    before: access(*old_layout, *source_access)?,
                                    after,
                                },
                            ));
                        }
                    }
                    rec.used_images.push(*image);
                }
                // Global dependencies cover attachment writes as well as buffers.
                // Retire preceding GPU work at this control point for legacy
                // memory barriers, just as for synchronization2 dependencies.
                if *full_scope || !memory.is_empty() {
                    rec.synchronizations.push(synchronization2::Insertion {
                        position: rec.ops.len(),
                        order: rec.order,
                        action: synchronization2::Action::Barrier,
                    });
                }
            }
        }
        Ok(())
    }

    fn references_descriptor_sets(&self, sets: &[vk::DescriptorSet]) -> bool {
        matches!(
            self,
            Self::BindDescriptorSets {
                sets: recorded, ..
            } if recorded.iter().any(|set| sets.contains(set))
        )
    }
}

fn resolve_recording(rt: &mut Runtime, source: &Recording) -> VkResult<ResolvedRecording> {
    if source.state != RecordingState::Executable {
        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
    }
    let mut resolved = ResolvedRecording::new();
    for (index, command) in source.commands.iter().enumerate() {
        if let Err(error) = command.apply(rt, &mut resolved) {
            {
                eprintln!(
                    "[SGFX Vulkan] resolve command {index} {:?}: {error:?}",
                    std::mem::discriminant(command)
                );
                match command {
                    RecordedCommand::CopyImage {
                        source,
                        source_layout,
                        destination,
                        destination_layout,
                        regions,
                    } => eprintln!(
                        "[SGFX Vulkan] raw image copy: {source:?}/{source_layout:?} {:?} -> {destination:?}/{destination_layout:?} {:?} regions={regions:?}",
                        rt.resources.images.get(source).map(|i| (
                            i.format,
                            i.extent,
                            i.mip_levels,
                            i.array_layers
                        )),
                        rt.resources.images.get(destination).map(|i| (
                            i.format,
                            i.extent,
                            i.mip_levels,
                            i.array_layers
                        ))
                    ),
                    RecordedCommand::BlitImage {
                        source,
                        destination,
                        regions,
                        filter,
                        ..
                    } => eprintln!(
                        "[SGFX Vulkan] blit: {:?} -> {:?} regions={regions:?} filter={filter:?}",
                        rt.resources.images.get(source).map(|i| i.format),
                        rt.resources.images.get(destination).map(|i| i.format)
                    ),
                    RecordedCommand::BeginRenderPass {
                        contents,
                        render_pass,
                        framebuffer,
                        area,
                        attachments,
                        ..
                    } => eprintln!(
                        "[SGFX Vulkan] begin pass: {contents:?} pass={render_pass:?} fb={framebuffer:?} area={area:?} attachments={attachments:?}"
                    ),
                    RecordedCommand::PipelineBarrier {
                        source_stage,
                        destination_stage,
                        flags,
                        images,
                        ..
                    } => {
                        eprintln!(
                            "[SGFX Vulkan] barrier: {source_stage:?} -> {destination_stage:?} flags={flags:?}"
                        );
                        for b in images {
                            eprintln!(
                                "[SGFX Vulkan] image barrier: {:?} {:?} -> {:?}",
                                b.5, b.6, b.7
                            );
                        }
                    }
                    _ => (),
                }
            }
            return Err(error);
        }
    }
    if resolved.render.is_some() {
        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
    }
    Ok(resolved)
}

fn record(command: vk::CommandBuffer, recorded: RecordedCommand) {
    let Ok(driver) = driver(command.as_raw(), Kind::Command) else {
        return;
    };
    let recording = driver
        .recordings
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .commands
        .get(&command.as_raw())
        .cloned();
    if let Some(recording) = recording {
        let _ = recording
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(recorded);
    }
}

fn record_error(command: vk::CommandBuffer, error: vk::Result) {
    let Ok(driver) = driver(command.as_raw(), Kind::Command) else {
        return;
    };
    let recording = driver
        .recordings
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .commands
        .get(&command.as_raw())
        .cloned();
    if let Some(recording) = recording {
        recording
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .fail(error);
    }
}
unsafe extern "system" fn create_command_pool(
    device: vk::Device,
    info: *const vk::CommandPoolCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::CommandPool,
) -> vk::Result {
    if info.is_null() || out.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let info = &*info;
    if !allocator.is_null()
        || !info.p_next.is_null()
        || info.queue_family_index != 0
        || !vk::CommandPoolCreateFlags::from_raw(3).contains(info.flags)
    {
        return vk::Result::ERROR_FEATURE_NOT_PRESENT;
    }
    let flags = info.flags;
    let driver = match driver(device.as_raw(), Kind::Device) {
        Ok(driver) => driver,
        Err(error) => return error,
    };
    let id = next_id();
    driver
        .recordings
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .pools
        .insert(id, flags);
    *out = vk::CommandPool::from_raw(id);
    vk::Result::SUCCESS
}
unsafe extern "system" fn destroy_command_pool(
    device: vk::Device,
    pool: vk::CommandPool,
    _: *const vk::AllocationCallbacks<'_>,
) {
    let Ok(driver) = driver(device.as_raw(), Kind::Device) else {
        return;
    };
    let ids = {
        let mut registry = driver
            .recordings
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if registry.pools.remove(&pool.as_raw()).is_none() {
            return;
        }
        let ids: Vec<_> = registry
            .commands
            .iter()
            .filter(|(_, recording)| {
                recording
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .pool
                    == pool.as_raw()
            })
            .map(|(&i, _)| i)
            .collect();
        for id in &ids {
            registry.commands.remove(id);
        }
        ids
    };
    for id in ids {
        remove_handle(id);
    }
}
unsafe extern "system" fn allocate_command_buffers(
    device: vk::Device,
    info: *const vk::CommandBufferAllocateInfo<'_>,
    out: *mut vk::CommandBuffer,
) -> vk::Result {
    if info.is_null() || out.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let info = &*info;
    if !info.p_next.is_null()
        || info.level != vk::CommandBufferLevel::PRIMARY
        || info.command_buffer_count == 0
        || info.command_buffer_count > 256
    {
        return vk::Result::ERROR_FEATURE_NOT_PRESENT;
    }
    let pool = info.command_pool.as_raw();
    let count = info.command_buffer_count;
    let d = match driver(device.as_raw(), Kind::Device) {
        Ok(d) => d,
        Err(e) => return e,
    };
    if !d
        .recordings
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .pools
        .contains_key(&pool)
    {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let ids: Vec<_> = (0..count).map(|_| add_handle(Kind::Command, &d)).collect();
    {
        let mut registry = d
            .recordings
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for id in &ids {
            registry
                .commands
                .insert(*id, Arc::new(Mutex::new(Recording::new(pool))));
        }
    }
    for (i, id) in ids.into_iter().enumerate() {
        *out.add(i) = vk::CommandBuffer::from_raw(id);
    }
    vk::Result::SUCCESS
}
unsafe extern "system" fn free_command_buffers(
    device: vk::Device,
    _pool: vk::CommandPool,
    count: u32,
    ptr: *const vk::CommandBuffer,
) {
    let Ok(commands) = slice(ptr, count) else {
        return;
    };
    let Ok(driver) = driver(device.as_raw(), Kind::Device) else {
        return;
    };
    let ids: Vec<_> = commands.iter().map(|c| c.as_raw()).collect();
    let removed = {
        let mut registry = driver
            .recordings
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let removed: Vec<_> = ids
            .into_iter()
            .filter(|id| {
                registry.commands.get(id).is_some_and(|recording| {
                    recording
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .pool
                        == _pool.as_raw()
                })
            })
            .collect();
        for id in &removed {
            registry.commands.remove(id);
        }
        removed
    };
    for id in removed {
        remove_handle(id);
    }
}
unsafe extern "system" fn reset_command_pool(
    device: vk::Device,
    pool: vk::CommandPool,
    flags: vk::CommandPoolResetFlags,
) -> vk::Result {
    if !vk::CommandPoolResetFlags::RELEASE_RESOURCES.contains(flags) {
        return vk::Result::ERROR_FEATURE_NOT_PRESENT;
    }
    let driver = match driver(device.as_raw(), Kind::Device) {
        Ok(driver) => driver,
        Err(error) => return error,
    };
    let recordings = {
        let registry = driver
            .recordings
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !registry.pools.contains_key(&pool.as_raw()) {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        }
        registry
            .commands
            .values()
            .filter(|recording| {
                recording
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .pool
                    == pool.as_raw()
            })
            .cloned()
            .collect::<Vec<_>>()
    };
    for recording in recordings {
        *recording
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Recording::new(pool.as_raw());
    }
    vk::Result::SUCCESS
}
unsafe extern "system" fn reset_command_buffer(
    command: vk::CommandBuffer,
    flags: vk::CommandBufferResetFlags,
) -> vk::Result {
    if !vk::CommandBufferResetFlags::RELEASE_RESOURCES.contains(flags) {
        return vk::Result::ERROR_FEATURE_NOT_PRESENT;
    }
    let driver = match driver(command.as_raw(), Kind::Command) {
        Ok(driver) => driver,
        Err(error) => return error,
    };
    let recording = driver
        .recordings
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .commands
        .get(&command.as_raw())
        .cloned();
    let Some(recording) = recording else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let pool = recording
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .pool;
    if !driver
        .recordings
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .pools
        .get(&pool)
        .is_some_and(|flags| flags.contains(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER))
    {
        return vk::Result::ERROR_FEATURE_NOT_PRESENT;
    }
    *recording
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Recording::new(pool);
    vk::Result::SUCCESS
}
unsafe extern "system" fn begin_command_buffer(
    command: vk::CommandBuffer,
    info: *const vk::CommandBufferBeginInfo<'_>,
) -> vk::Result {
    if info.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let info = &*info;
    if !info.p_next.is_null()
        || !info.p_inheritance_info.is_null()
        || !vk::CommandBufferUsageFlags::from_raw(5).contains(info.flags)
    {
        return vk::Result::ERROR_FEATURE_NOT_PRESENT;
    }
    let one_time = info
        .flags
        .contains(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    let driver = match driver(command.as_raw(), Kind::Command) {
        Ok(driver) => driver,
        Err(error) => return error,
    };
    let existing = driver
        .recordings
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .commands
        .get(&command.as_raw())
        .cloned();
    let Some(existing) = existing else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let (pool, state) = {
        let existing = existing
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        (existing.pool, existing.state)
    };
    if state == RecordingState::Recording {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    if state != RecordingState::Initial
        && !driver
            .recordings
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .pools
            .get(&pool)
            .is_some_and(|flags| flags.contains(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER))
    {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let mut recording = Recording::new(pool);
    recording.state = RecordingState::Recording;
    recording.one_time = one_time;
    *existing
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = recording;
    vk::Result::SUCCESS
}
unsafe extern "system" fn end_command_buffer(command: vk::CommandBuffer) -> vk::Result {
    let driver = match driver(command.as_raw(), Kind::Command) {
        Ok(driver) => driver,
        Err(error) => return error,
    };
    let recording = driver
        .recordings
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .commands
        .get(&command.as_raw())
        .cloned();
    let recording = match recording {
        Some(recording) => recording,
        None => return vk::Result::ERROR_INITIALIZATION_FAILED,
    };
    let mut recording = recording
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if recording.state != RecordingState::Recording
        || recording.render_active
        || !recording.active_queries.is_empty()
    {
        recording.fail(vk::Result::ERROR_INITIALIZATION_FAILED)
    }
    if let Some(error) = recording.error {
        recording.state = RecordingState::Invalid;
        error
    } else {
        recording.state = RecordingState::Executable;
        vk::Result::SUCCESS
    }
}
unsafe extern "system" fn cmd_bind_pipeline(
    command: vk::CommandBuffer,
    point: vk::PipelineBindPoint,
    pipeline: vk::Pipeline,
) {
    if !matches!(
        point,
        vk::PipelineBindPoint::GRAPHICS | vk::PipelineBindPoint::COMPUTE
    ) {
        record_error(command, vk::Result::ERROR_FEATURE_NOT_PRESENT);
        return;
    }
    record(command, RecordedCommand::BindPipeline { point, pipeline })
}
unsafe extern "system" fn cmd_bind_descriptor_sets(
    command: vk::CommandBuffer,
    point: vk::PipelineBindPoint,
    layout: vk::PipelineLayout,
    first: u32,
    count: u32,
    sets: *const vk::DescriptorSet,
    dynamic_count: u32,
    dynamic: *const u32,
) {
    if !matches!(
        point,
        vk::PipelineBindPoint::GRAPHICS | vk::PipelineBindPoint::COMPUTE
    ) || first.checked_add(count).is_none_or(|bound| bound > 4)
        || dynamic_count > 64
    {
        record_error(command, vk::Result::ERROR_FEATURE_NOT_PRESENT);
        return;
    }
    let copied = match slice(sets, count) {
        Ok(s) => s.to_vec(),
        Err(e) => {
            record_error(command, e);
            return;
        }
    };
    let dynamic_offsets = match slice(dynamic, dynamic_count) {
        Ok(values) => values.to_vec(),
        Err(error) => {
            record_error(command, error);
            return;
        }
    };
    record(
        command,
        RecordedCommand::BindDescriptorSets {
            point,
            layout,
            first,
            sets: copied,
            dynamic_offsets,
            pushed: None,
        },
    )
}
unsafe extern "system" fn cmd_dispatch(command: vk::CommandBuffer, x: u32, y: u32, z: u32) {
    if x > 65_535 || y > 65_535 || z > 65_535 {
        record_error(command, vk::Result::ERROR_FEATURE_NOT_PRESENT);
        return;
    }
    record(command, RecordedCommand::Dispatch { x, y, z })
}
unsafe extern "system" fn cmd_push_constants(
    command: vk::CommandBuffer,
    layout: vk::PipelineLayout,
    flags: vk::ShaderStageFlags,
    offset: u32,
    size: u32,
    values: *const std::ffi::c_void,
) {
    let result = (|| {
        if size == 0
            || !size.is_multiple_of(4)
            || !offset.is_multiple_of(4)
            || offset
                .checked_add(size)
                .is_none_or(|end| end > ir::MAX_PUSH_CONSTANT_BYTES)
        {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let stages = crate::resources::stages(flags)?;
        let data = slice(values.cast::<u8>(), size)?.to_vec();
        let driver = driver(command.as_raw(), Kind::Command)?;
        let desc = driver
            .recordings
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .pipeline_layouts
            .get(&layout)
            .cloned()
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        desc.validate_push_constants(stages, offset, &data)
            .map_err(crate::resources::failure)?;
        Ok(RecordedCommand::PushConstants {
            layout: desc,
            stages,
            offset,
            data,
        })
    })();
    match result {
        Ok(command_data) => record(command, command_data),
        Err(error) => record_error(command, error),
    }
}
unsafe extern "system" fn cmd_begin_render_pass(
    command: vk::CommandBuffer,
    info: *const vk::RenderPassBeginInfo<'_>,
    contents: vk::SubpassContents,
) {
    if info.is_null() {
        record_error(command, vk::Result::ERROR_INITIALIZATION_FAILED);
        return;
    }
    let info = &*info;
    let clears = match slice(info.p_clear_values, info.clear_value_count) {
        Ok(v) => v.to_vec(),
        Err(e) => {
            record_error(command, e);
            return;
        }
    };
    let render_pass = info.render_pass;
    let framebuffer = info.framebuffer;
    let area = info.render_area;
    let attachments = if info.p_next.is_null() {
        None
    } else {
        let a = &*info.p_next.cast::<vk::RenderPassAttachmentBeginInfo<'_>>();
        if a.s_type != vk::StructureType::RENDER_PASS_ATTACHMENT_BEGIN_INFO || !a.p_next.is_null() {
            record_error(command, vk::Result::ERROR_FEATURE_NOT_PRESENT);
            return;
        }
        match slice(a.p_attachments, a.attachment_count) {
            Ok(v) => Some(v.to_vec()),
            Err(e) => {
                record_error(command, e);
                return;
            }
        }
    };
    if contents != vk::SubpassContents::INLINE {
        record_error(command, vk::Result::ERROR_FEATURE_NOT_PRESENT);
        return;
    }
    record(
        command,
        RecordedCommand::BeginRenderPass {
            attachments,
            contents,
            render_pass,
            framebuffer,
            area,
            clears,
        },
    )
}
unsafe extern "system" fn cmd_next_subpass(
    command: vk::CommandBuffer,
    contents: vk::SubpassContents,
) {
    if contents != vk::SubpassContents::INLINE {
        record_error(command, vk::Result::ERROR_FEATURE_NOT_PRESENT);
        return;
    }
    record(command, RecordedCommand::NextSubpass)
}

unsafe extern "system" fn cmd_clear_attachments(
    command: vk::CommandBuffer,
    attachment_count: u32,
    attachments: *const vk::ClearAttachment,
    rect_count: u32,
    rects: *const vk::ClearRect,
) {
    let result = (|| {
        if attachment_count > 16 || rect_count > 1024 {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        Ok(RecordedCommand::ClearAttachments {
            attachments: slice(attachments, attachment_count)?.to_vec(),
            rects: slice(rects, rect_count)?.to_vec(),
        })
    })();
    match result {
        Ok(command_data) => record(command, command_data),
        Err(error) => record_error(command, error),
    }
}

unsafe extern "system" fn cmd_clear_depth_image(
    command: vk::CommandBuffer,
    image: vk::Image,
    layout: vk::ImageLayout,
    value: *const vk::ClearDepthStencilValue,
    count: u32,
    ranges: *const vk::ImageSubresourceRange,
) {
    let result = (|| {
        let depth = value
            .as_ref()
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?
            .depth;
        let ranges = slice(ranges, count)?.to_vec();
        Ok(RecordedCommand::ClearDepthImage {
            image,
            layout,
            depth,
            ranges,
        })
    })();
    match result {
        Ok(command_data) => record(command, command_data),
        Err(error) => record_error(command, error),
    }
}
unsafe extern "system" fn cmd_clear_color_image(
    command: vk::CommandBuffer,
    image: vk::Image,
    layout: vk::ImageLayout,
    color: *const vk::ClearColorValue,
    count: u32,
    ranges: *const vk::ImageSubresourceRange,
) {
    let result = (|| {
        let color = color
            .as_ref()
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?
            .float32;
        ir::Color::rgba(color[0], color[1], color[2], color[3])
            .map_err(crate::resources::failure)?;
        if count == 0 || count > 64 {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        Ok(RecordedCommand::ClearColorImage {
            image,
            layout,
            color,
            ranges: slice(ranges, count)?.to_vec(),
        })
    })();
    match result {
        Ok(command_data) => record(command, command_data),
        Err(error) => record_error(command, error),
    }
}
unsafe extern "system" fn cmd_end_render_pass(command: vk::CommandBuffer) {
    record(command, RecordedCommand::EndRenderPass)
}
unsafe extern "system" fn cmd_set_viewport(
    command: vk::CommandBuffer,
    first: u32,
    count: u32,
    viewports: *const vk::Viewport,
) {
    if first != 0 || count != 1 || viewports.is_null() {
        record_error(command, vk::Result::ERROR_FEATURE_NOT_PRESENT);
        return;
    }
    let v = *viewports;
    if ir::Viewport::new(v.x, v.y, v.width, v.height, v.min_depth, v.max_depth).is_err() {
        record_error(command, vk::Result::ERROR_FEATURE_NOT_PRESENT);
        return;
    }
    record(command, RecordedCommand::SetViewport(v));
}
unsafe extern "system" fn cmd_set_scissor(
    command: vk::CommandBuffer,
    first: u32,
    count: u32,
    scissors: *const vk::Rect2D,
) {
    if first != 0 || count != 1 || scissors.is_null() {
        record_error(command, vk::Result::ERROR_FEATURE_NOT_PRESENT);
        return;
    }
    let s = *scissors;
    if s.offset.x < 0 || s.offset.y < 0 {
        record_error(command, vk::Result::ERROR_FEATURE_NOT_PRESENT);
        return;
    }
    record(command, RecordedCommand::SetScissor(s));
}

unsafe extern "system" fn cmd_bind_vertex_buffers(
    command: vk::CommandBuffer,
    first: u32,
    count: u32,
    buffers: *const vk::Buffer,
    offsets: *const vk::DeviceSize,
) {
    if count == 0 {
        return;
    }
    if first
        .checked_add(count)
        .is_none_or(|end| end > ir::MAX_VERTEX_BUFFERS as u32)
        || buffers.is_null()
        || offsets.is_null()
    {
        record_error(command, vk::Result::ERROR_FEATURE_NOT_PRESENT);
        return;
    }
    let buffers = std::slice::from_raw_parts(buffers, count as usize);
    let offsets = std::slice::from_raw_parts(offsets, count as usize);
    if offsets.iter().any(|offset| !offset.is_multiple_of(4)) {
        record_error(command, vk::Result::ERROR_INITIALIZATION_FAILED);
        return;
    }
    for (index, (&buffer, &offset)) in buffers.iter().zip(offsets).enumerate() {
        record(
            command,
            RecordedCommand::BindVertexBuffer {
                slot: first + index as u32,
                buffer,
                offset,
            },
        );
    }
}

unsafe extern "system" fn cmd_bind_index_buffer(
    command: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
    index_type: vk::IndexType,
) {
    let format = match index_type {
        vk::IndexType::UINT16 => ir::IndexFormat::Uint16,
        vk::IndexType::UINT32 => ir::IndexFormat::Uint32,
        _ => {
            record_error(command, vk::Result::ERROR_FEATURE_NOT_PRESENT);
            return;
        }
    };
    if !offset.is_multiple_of(format.byte_size()) {
        record_error(command, vk::Result::ERROR_INITIALIZATION_FAILED);
        return;
    }
    record(
        command,
        RecordedCommand::BindIndexBuffer {
            buffer,
            offset,
            index_type,
        },
    )
}

fn graphics_bindings(
    table: &ir::ResourceTable,
    resources: &mut crate::resources::Resources,
    rec: &mut ResolvedRecording,
) -> VkResult<bool> {
    let extent = rec.render.ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
    let pipeline = rec
        .graphics
        .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
    let id = resources.graphics_pipeline_for_views(table, pipeline, &rec.graphics_sets)?;
    let state = resources
        .graphics_state
        .get(&pipeline)
        .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
    let viewport = state
        .viewport
        .or(rec.viewport)
        .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
    let viewport = ir::Viewport::new(
        viewport.x,
        viewport.y,
        viewport.width,
        viewport.height,
        viewport.min_depth,
        viewport.max_depth,
    )
    .map_err(crate::resources::failure)?;
    let scissor = state
        .scissor
        .or(rec.scissor)
        .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
    let area = rec.active_render_pass.as_ref().map_or_else(
        || ir::PixelRect::new(0, 0, extent.0, extent.1).map_err(crate::resources::failure),
        |pass| Ok(pass.area),
    )?;
    let scissor = render_passes::clip_scissor(scissor, area)?;
    let visible = scissor.is_some();
    let desc = table
        .programmable_render_pipeline_shared(
            table
                .programmable_render_pipeline_ref(id)
                .map_err(crate::resources::failure)?,
        )
        .map_err(crate::resources::failure)?;
    render_passes::validate_bindings(resources, rec, pipeline, desc.layout())?;
    if rec
        .active_render_pass
        .as_ref()
        .is_some_and(|a| a.pass.subpasses[a.subpass].read_only_depth)
        && desc.depth_state().is_some_and(|d| d.write_enabled())
    {
        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
    }
    emit_changed(
        &mut rec.emitted_graphics.viewport,
        viewport,
        &mut rec.ops,
        ir::OwnedCommand::SetViewport,
    );
    if let Some(scissor) = scissor {
        emit_changed(
            &mut rec.emitted_graphics.scissor,
            scissor,
            &mut rec.ops,
            |rect| ir::OwnedCommand::SetScissor(Some(rect)),
        );
    }
    if let Some(previous) = rec
        .emitted_graphics
        .pipeline
        .filter(|previous| *previous != id)
    {
        let previous = table
            .programmable_render_pipeline_shared(
                table
                    .programmable_render_pipeline_ref(previous)
                    .map_err(crate::resources::failure)?,
            )
            .map_err(crate::resources::failure)?;
        if previous.layout() != desc.layout() {
            rec.emitted_graphics.descriptors.clear();
        }
    }
    let pipeline_changed = rec.emitted_graphics.pipeline != Some(id);
    emit_changed(
        &mut rec.emitted_graphics.pipeline,
        id,
        &mut rec.ops,
        ir::OwnedCommand::SetProgrammablePipeline,
    );
    if pipeline_changed {
        // Vulkan preserves descriptor bindings that the next shader no longer
        // uses. SGFX's reflected layout has explicit empty slots; replace stale
        // bindings there once per pipeline change, retaining active Vulkan sets.
        for (index, group) in desc.layout().bind_groups().iter().enumerate() {
            if group.entries().is_empty() {
                rec.ops.push(ir::OwnedCommand::SetBindGroup {
                    index: index as u32,
                    bind_group: resources.empty_bind_group(table)?,
                });
            }
        }
    }
    let push_key = (id, rec.push_constants_revision);
    if rec.emitted_graphics.push_constants != Some(push_key) {
        rec.ops.extend(rec.push_constants.snapshot(desc.layout())?);
        rec.emitted_graphics.push_constants = Some(push_key);
    }
    for (slot, _) in desc.vertex_buffers().iter().enumerate() {
        let (handle, offset) =
            rec.vertex_buffers[slot].ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let buffer = resources
            .buffers
            .get(&handle)
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        emit_changed(
            &mut rec.emitted_graphics.vertex_buffers[slot],
            (buffer.id, offset),
            &mut rec.ops,
            |(buffer, offset)| {
                if slot == 0 {
                    ir::OwnedCommand::SetVertexBuffer { buffer, offset }
                } else {
                    ir::OwnedCommand::SetVertexBufferSlot {
                        slot: slot as u32,
                        buffer,
                        offset,
                    }
                }
            },
        );
    }
    let active_sets = active_sets(resources, desc.layout(), &rec.graphics_sets)?;
    for &(index, set) in &active_sets {
        let offsets = rec
            .graphics_offsets
            .get(&index)
            .map_or(&[][..], Vec::as_slice);
        if rec.emitted_graphics.descriptors.get(&index).is_some_and(
            |(previous_set, previous_offsets)| {
                *previous_set == set && previous_offsets.as_slice() == offsets
            },
        ) {
            continue;
        }
        let dynamic_offsets = offsets.to_vec();
        rec.emitted_graphics
            .descriptors
            .insert(index, (set, dynamic_offsets.clone()));
        rec.descriptors.push(DescriptorInsertion {
            layout: desc.layout().bind_groups()[index as usize].clone(),
            position: rec.ops.len(),
            index,
            set,
            dynamic_offsets,
        });
    }
    mark_writable_descriptor_buffers(
        resources,
        rec,
        &active_sets,
        desc.layout(),
        vk::PipelineBindPoint::GRAPHICS,
    )?;
    Ok(visible)
}

fn mark_writable_descriptor_buffers(
    resources: &crate::resources::Resources,
    rec: &mut ResolvedRecording,
    sets: &[(u32, vk::DescriptorSet)],
    layout: &ir::PipelineLayoutDesc,
    point: vk::PipelineBindPoint,
) -> VkResult<()> {
    for (index, handle) in sets {
        let set = resources
            .descriptor_sets
            .get(handle)
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        for (&binding, resource) in &set.bindings {
            let writable = layout.bind_groups()[*index as usize]
                .entries()
                .iter()
                .any(|entry| {
                    Some(entry.binding()) == set.source_to_ir(binding)
                        && matches!(
                            entry.ty(),
                            ir::BindingType::StorageBuffer { read_only: false }
                        )
                });
            if writable {
                let crate::resources::DescriptorBinding::Buffer {
                    buffer,
                    offset,
                    range,
                } = resource
                else {
                    continue;
                };
                if !rec.written_buffers.contains(buffer) {
                    rec.written_buffers.push(*buffer);
                }
                let offsets = if point == vk::PipelineBindPoint::COMPUTE {
                    &rec.compute_offsets
                } else {
                    &rec.graphics_offsets
                };
                let dynamic = buffer_uploads::dynamic_offset(
                    set,
                    binding,
                    offsets.get(index).map_or(&[], Vec::as_slice),
                )?;
                let size = resources
                    .buffers
                    .get(buffer)
                    .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?
                    .size;
                let range = buffer_uploads::descriptor_range(size, *offset, *range, dynamic)?;
                rec.readback_buffers
                    .insert(*buffer, range.start, range.end - range.start)?;
            }
        }
    }
    Ok(())
}

unsafe extern "system" fn cmd_draw(
    command: vk::CommandBuffer,
    vertices: u32,
    instances: u32,
    first: u32,
    first_instance: u32,
) {
    if first_instance.checked_add(instances).is_none() {
        record_error(command, vk::Result::ERROR_FEATURE_NOT_PRESENT);
        return;
    }
    record(
        command,
        RecordedCommand::Draw {
            vertices,
            instances,
            first,
            first_instance,
        },
    )
}

unsafe extern "system" fn cmd_draw_indexed(
    command: vk::CommandBuffer,
    indices: u32,
    instances: u32,
    first: u32,
    base_vertex: i32,
    first_instance: u32,
) {
    if first_instance.checked_add(instances).is_none() {
        record_error(command, vk::Result::ERROR_FEATURE_NOT_PRESENT);
        return;
    }
    record(
        command,
        RecordedCommand::DrawIndexed {
            indices,
            instances,
            first,
            base_vertex,
            first_instance,
        },
    )
}
unsafe extern "system" fn cmd_copy_image_to_buffer(
    command: vk::CommandBuffer,
    image: vk::Image,
    layout: vk::ImageLayout,
    buffer: vk::Buffer,
    count: u32,
    regions: *const vk::BufferImageCopy,
) {
    let regions = match slice(regions, count) {
        Ok(v) => v.to_vec(),
        Err(e) => {
            record_error(command, e);
            return;
        }
    };
    record(
        command,
        RecordedCommand::CopyImageToBuffer {
            image,
            layout,
            buffer,
            regions,
        },
    )
}
unsafe extern "system" fn cmd_copy_buffer(
    command: vk::CommandBuffer,
    source: vk::Buffer,
    destination: vk::Buffer,
    count: u32,
    regions: *const vk::BufferCopy,
) {
    let regions = match slice(regions, count) {
        Ok(v) => v.to_vec(),
        Err(e) => {
            record_error(command, e);
            return;
        }
    };
    record(
        command,
        RecordedCommand::CopyBuffer {
            source,
            destination,
            regions,
        },
    )
}
unsafe extern "system" fn cmd_copy_buffer_to_image(
    command: vk::CommandBuffer,
    source: vk::Buffer,
    image: vk::Image,
    layout: vk::ImageLayout,
    count: u32,
    regions: *const vk::BufferImageCopy,
) {
    let regions = match slice(regions, count) {
        Ok(v) => v.to_vec(),
        Err(e) => {
            record_error(command, e);
            return;
        }
    };
    record(
        command,
        RecordedCommand::CopyBufferToImage {
            source,
            image,
            layout,
            regions,
        },
    );
}

unsafe extern "system" fn cmd_copy_image(
    command: vk::CommandBuffer,
    source: vk::Image,
    source_layout: vk::ImageLayout,
    destination: vk::Image,
    destination_layout: vk::ImageLayout,
    count: u32,
    regions: *const vk::ImageCopy,
) {
    if count == 0 || count > 4000 {
        record_error(command, vk::Result::ERROR_INITIALIZATION_FAILED);
        return;
    }
    let regions = match slice(regions, count) {
        Ok(regions) => regions.to_vec(),
        Err(error) => {
            record_error(command, error);
            return;
        }
    };
    record(
        command,
        RecordedCommand::CopyImage {
            source,
            source_layout,
            destination,
            destination_layout,
            regions,
        },
    );
}

unsafe extern "system" fn cmd_blit_image(
    command: vk::CommandBuffer,
    source: vk::Image,
    source_layout: vk::ImageLayout,
    destination: vk::Image,
    destination_layout: vk::ImageLayout,
    count: u32,
    regions: *const vk::ImageBlit,
    filter: vk::Filter,
) {
    if count == 0 || count > 4000 {
        record_error(command, vk::Result::ERROR_FEATURE_NOT_PRESENT);
        return;
    }
    let regions = match slice(regions, count) {
        Ok(regions) => regions.to_vec(),
        Err(error) => {
            record_error(command, error);
            return;
        }
    };
    record(
        command,
        RecordedCommand::BlitImage {
            source,
            source_layout,
            destination,
            destination_layout,
            regions,
            filter,
        },
    );
}

unsafe extern "system" fn cmd_pipeline_barrier(
    command: vk::CommandBuffer,
    source_stage: vk::PipelineStageFlags,
    destination_stage: vk::PipelineStageFlags,
    flags: vk::DependencyFlags,
    memory_count: u32,
    memory: *const vk::MemoryBarrier<'_>,
    buffer_count: u32,
    buffers: *const vk::BufferMemoryBarrier<'_>,
    image_count: u32,
    images: *const vk::ImageMemoryBarrier<'_>,
) {
    if validate_pipeline_barrier_header(source_stage, destination_stage, flags).is_err() {
        record_error(command, vk::Result::ERROR_FEATURE_NOT_PRESENT);
        return;
    }
    let m = match slice(memory, memory_count) {
        Ok(v) => v
            .iter()
            .map(|x| (!x.p_next.is_null(), x.src_access_mask, x.dst_access_mask))
            .collect::<Vec<_>>(),
        Err(e) => {
            record_error(command, e);
            return;
        }
    };
    let b = match slice(buffers, buffer_count) {
        Ok(v) => v
            .iter()
            .map(|x| {
                (
                    !x.p_next.is_null(),
                    x.src_access_mask,
                    x.dst_access_mask,
                    x.src_queue_family_index,
                    x.dst_queue_family_index,
                    x.buffer,
                    x.offset,
                    x.size,
                )
            })
            .collect::<Vec<_>>(),
        Err(e) => {
            record_error(command, e);
            return;
        }
    };
    let i = match slice(images, image_count) {
        Ok(v) => v
            .iter()
            .map(|x| {
                (
                    !x.p_next.is_null(),
                    x.src_access_mask,
                    x.dst_access_mask,
                    x.src_queue_family_index,
                    x.dst_queue_family_index,
                    x.image,
                    x.old_layout,
                    x.new_layout,
                    x.subresource_range,
                )
            })
            .collect::<Vec<_>>(),
        Err(e) => {
            record_error(command, e);
            return;
        }
    };
    record(
        command,
        RecordedCommand::PipelineBarrier {
            full_scope: false,
            source_stage,
            destination_stage,
            flags,
            memory: m,
            buffers: b,
            images: i,
        },
    )
}

// This ICD exposes queue family 0 only. Equal family indices do not transfer
// ownership; Vulkan treats them like two QUEUE_FAMILY_IGNORED indices.
fn same_queue_family(source: u32, destination: u32) -> bool {
    source == destination && matches!(source, 0 | vk::QUEUE_FAMILY_IGNORED)
}

fn validate_pipeline_barrier_header(
    source_stage: vk::PipelineStageFlags,
    destination_stage: vk::PipelineStageFlags,
    flags: vk::DependencyFlags,
) -> VkResult<()> {
    let allowed = vk::PipelineStageFlags::TOP_OF_PIPE
        | vk::PipelineStageFlags::BOTTOM_OF_PIPE
        | vk::PipelineStageFlags::HOST
        | vk::PipelineStageFlags::DRAW_INDIRECT
        | vk::PipelineStageFlags::TRANSFER
        | vk::PipelineStageFlags::COMPUTE_SHADER
        | vk::PipelineStageFlags::VERTEX_SHADER
        | vk::PipelineStageFlags::FRAGMENT_SHADER
        | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
        | vk::PipelineStageFlags::VERTEX_INPUT
        | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
        | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
        | vk::PipelineStageFlags::ALL_COMMANDS
        | vk::PipelineStageFlags::ALL_GRAPHICS;
    if source_stage.is_empty()
        || destination_stage.is_empty()
        || !allowed.contains(source_stage | destination_stage)
        || !vk::DependencyFlags::BY_REGION.contains(flags)
    {
        Err(vk::Result::ERROR_FEATURE_NOT_PRESENT)
    } else {
        Ok(())
    }
}
fn supported_access(access: vk::AccessFlags) -> bool {
    (vk::AccessFlags::SHADER_READ
        | vk::AccessFlags::SHADER_WRITE
        | vk::AccessFlags::TRANSFER_READ
        | vk::AccessFlags::TRANSFER_WRITE
        | vk::AccessFlags::HOST_READ
        | vk::AccessFlags::HOST_WRITE
        | vk::AccessFlags::UNIFORM_READ
        | vk::AccessFlags::MEMORY_READ
        | vk::AccessFlags::MEMORY_WRITE
        | vk::AccessFlags::COLOR_ATTACHMENT_READ
        | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
        | vk::AccessFlags::VERTEX_ATTRIBUTE_READ
        | vk::AccessFlags::INDEX_READ
        | vk::AccessFlags::INPUT_ATTACHMENT_READ
        | vk::AccessFlags::INDIRECT_COMMAND_READ)
        .contains(access)
}
fn buffer_access(access: vk::AccessFlags) -> VkResult<ir::BufferAccess> {
    if !supported_access(access) {
        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
    }
    if access.intersects(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::MEMORY_WRITE) {
        Ok(ir::BufferAccess::StorageReadWrite)
    } else if access.contains(vk::AccessFlags::SHADER_READ) {
        Ok(ir::BufferAccess::StorageRead)
    } else if access.contains(vk::AccessFlags::VERTEX_ATTRIBUTE_READ) {
        Ok(ir::BufferAccess::Vertex)
    } else if access.contains(vk::AccessFlags::INDEX_READ) {
        Ok(ir::BufferAccess::Index)
    } else if access.contains(vk::AccessFlags::INDIRECT_COMMAND_READ) {
        Ok(ir::BufferAccess::CopySource)
    } else if access.contains(vk::AccessFlags::UNIFORM_READ) {
        Ok(ir::BufferAccess::Uniform)
    } else if access.intersects(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::HOST_WRITE) {
        Ok(ir::BufferAccess::CopyDestination)
    } else if access.intersects(
        vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::HOST_READ | vk::AccessFlags::MEMORY_READ,
    ) {
        Ok(ir::BufferAccess::CopySource)
    } else {
        Err(vk::Result::ERROR_FEATURE_NOT_PRESENT)
    }
}
fn texture_access(layout: vk::ImageLayout) -> VkResult<ir::TextureAccess> {
    let layout = crate::render_pass::normalize_depth_layout(layout);
    match layout {
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        | vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => {
            Ok(ir::TextureAccess::RenderAttachment)
        }
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => Ok(ir::TextureAccess::CopySource),
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => Ok(ir::TextureAccess::CopyDestination),
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        | vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL => Ok(ir::TextureAccess::Sampled),
        _ => Err(vk::Result::ERROR_FEATURE_NOT_PRESENT),
    }
}

unsafe extern "system" fn create_fence(
    device: vk::Device,
    info: *const vk::FenceCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::Fence,
) -> vk::Result {
    if info.is_null() || out.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let info = &*info;
    if info.s_type != vk::StructureType::FENCE_CREATE_INFO
        || !allocator.is_null()
        || !info.p_next.is_null()
        || !vk::FenceCreateFlags::SIGNALED.contains(info.flags)
    {
        return vk::Result::ERROR_FEATURE_NOT_PRESENT;
    }
    let d = match driver(device.as_raw(), Kind::Device) {
        Ok(d) => d,
        Err(e) => return e,
    };
    let mut fences = d.fences.lock().unwrap_or_else(|e| e.into_inner());
    if fences.len() >= 4096 {
        return vk::Result::ERROR_TOO_MANY_OBJECTS;
    }
    let id = next_id();
    fences.insert(
        id,
        Arc::new(AtomicU8::new(u8::from(
            info.flags.contains(vk::FenceCreateFlags::SIGNALED),
        ))),
    );
    *out = vk::Fence::from_raw(id);
    vk::Result::SUCCESS
}
unsafe extern "system" fn destroy_fence(
    device: vk::Device,
    fence: vk::Fence,
    _: *const vk::AllocationCallbacks<'_>,
) {
    if let Ok(d) = driver(device.as_raw(), Kind::Device) {
        d.fences
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&fence.as_raw());
    }
}
fn fence_refs(d: &Driver, handles: &[vk::Fence]) -> VkResult<Vec<Arc<AtomicU8>>> {
    if d.lost.load(Ordering::Acquire) {
        return Err(vk::Result::ERROR_DEVICE_LOST);
    }
    let fences = d.fences.lock().unwrap_or_else(|e| e.into_inner());
    handles
        .iter()
        .map(|f| {
            fences
                .get(&f.as_raw())
                .cloned()
                .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)
        })
        .collect()
}
unsafe extern "system" fn get_fence_status(device: vk::Device, fence: vk::Fence) -> vk::Result {
    let result = driver(device.as_raw(), Kind::Device).and_then(|d| fence_refs(&d, &[fence]));
    match result {
        Ok(fs) if fs[0].load(Ordering::Acquire) == 1 => vk::Result::SUCCESS,
        Ok(_) => vk::Result::NOT_READY,
        Err(e) => e,
    }
}
unsafe extern "system" fn reset_fences(
    device: vk::Device,
    count: u32,
    ptr: *const vk::Fence,
) -> vk::Result {
    let fs = match slice(ptr, count) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let fs = match driver(device.as_raw(), Kind::Device).and_then(|d| fence_refs(&d, fs)) {
        Ok(v) => v,
        Err(e) => return e,
    };
    if fs.iter().any(|f| f.load(Ordering::Acquire) == 2) {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    for f in fs {
        f.store(0, Ordering::Release);
    }
    vk::Result::SUCCESS
}
unsafe extern "system" fn wait_for_fences(
    device: vk::Device,
    count: u32,
    ptr: *const vk::Fence,
    all: vk::Bool32,
    timeout: u64,
) -> vk::Result {
    let handles = match slice(ptr, count) {
        Ok(v) if !v.is_empty() => v,
        _ => return vk::Result::ERROR_INITIALIZATION_FAILED,
    };
    let d = match driver(device.as_raw(), Kind::Device) {
        Ok(d) => d,
        Err(e) => return e,
    };
    let fs = match fence_refs(&d, handles) {
        Ok(v) => v,
        Err(e) => return e,
    };
    // Resolve the selected fence Arcs before taking the notification mutex.
    // Completion updates the atomics and then notifies under the same mutex.
    // The predicate deliberately does not depend on all device work retiring.
    status(d.in_flight.wait_until(timeout, || {
        if d.lost.load(Ordering::Acquire) {
            return Err(vk::Result::ERROR_DEVICE_LOST);
        }
        Ok(if all != 0 {
            fs.iter().all(|f| f.load(Ordering::Acquire) == 1)
        } else {
            fs.iter().any(|f| f.load(Ordering::Acquire) == 1)
        })
    }))
}

unsafe extern "system" fn create_semaphore(
    device: vk::Device,
    info: *const vk::SemaphoreCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::Semaphore,
) -> vk::Result {
    if info.is_null() || out.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    *out = vk::Semaphore::null();
    let info = &*info;
    if info.s_type != vk::StructureType::SEMAPHORE_CREATE_INFO
        || !info.flags.is_empty()
        || !allocator.is_null()
    {
        return vk::Result::ERROR_FEATURE_NOT_PRESENT;
    }
    let d = match driver(device.as_raw(), Kind::Device) {
        Ok(d) => d,
        Err(error) => return error,
    };
    if !info.p_next.is_null() {
        return status(timeline::create(&d, info, out));
    }
    let mut semaphores = d.semaphores.lock().unwrap_or_else(|e| e.into_inner());
    if semaphores.len() >= 4096 {
        return vk::Result::ERROR_TOO_MANY_OBJECTS;
    }
    let id = next_id();
    semaphores.insert(id, Arc::new(AtomicU8::new(0)));
    *out = vk::Semaphore::from_raw(id);
    vk::Result::SUCCESS
}

unsafe extern "system" fn destroy_semaphore(
    device: vk::Device,
    semaphore: vk::Semaphore,
    _: *const vk::AllocationCallbacks<'_>,
) {
    if let Ok(d) = driver(device.as_raw(), Kind::Device) {
        d.timelines
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&semaphore.as_raw());
        d.semaphores
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&semaphore.as_raw());
    }
}

#[cfg(any(
    target_os = "scarlet",
    target_os = "macos",
    all(target_os = "linux", feature = "scarlet-wsi"),
    test
))]
fn semaphore_refs(d: &Driver, handles: &[vk::Semaphore]) -> VkResult<Vec<Arc<AtomicU8>>> {
    if d.lost.load(Ordering::Acquire) {
        return Err(vk::Result::ERROR_DEVICE_LOST);
    }
    let semaphores = d.semaphores.lock().unwrap_or_else(|e| e.into_inner());
    handles
        .iter()
        .map(|semaphore| {
            semaphores
                .get(&semaphore.as_raw())
                .cloned()
                .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)
        })
        .collect()
}

#[cfg(any(
    target_os = "scarlet",
    target_os = "macos",
    all(target_os = "linux", feature = "scarlet-wsi"),
    test
))]
fn wait_and_consume_semaphores(d: &Driver, semaphores: &[Arc<AtomicU8>]) -> VkResult<()> {
    for semaphore in semaphores {
        loop {
            if d.lost.load(Ordering::Acquire) {
                return Err(vk::Result::ERROR_DEVICE_LOST);
            }
            if semaphore
                .compare_exchange(1, 0, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                break;
            }
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
    }
    Ok(())
}

#[cfg(any(
    target_os = "scarlet",
    target_os = "macos",
    all(target_os = "linux", feature = "scarlet-wsi"),
    test
))]
pub(crate) fn signal_acquire_sync(
    device: vk::Device,
    semaphore: vk::Semaphore,
    fence: vk::Fence,
) -> VkResult<()> {
    let d = driver(device.as_raw(), Kind::Device)?;
    if semaphore == vk::Semaphore::null() && fence == vk::Fence::null() {
        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
    }
    let semaphore = if semaphore == vk::Semaphore::null() {
        None
    } else {
        Some(semaphore_refs(&d, &[semaphore])?.remove(0))
    };
    let fence = if fence == vk::Fence::null() {
        None
    } else {
        Some(fence_refs(&d, &[fence])?.remove(0))
    };
    if semaphore
        .as_ref()
        .is_some_and(|state| state.load(Ordering::Acquire) != 0)
        || fence
            .as_ref()
            .is_some_and(|state| state.load(Ordering::Acquire) != 0)
    {
        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
    }
    if let Some(state) = semaphore {
        state.store(1, Ordering::Release);
    }
    if let Some(state) = fence {
        state.store(1, Ordering::Release);
    }
    d.in_flight.notify();
    Ok(())
}

#[cfg(any(
    target_os = "scarlet",
    target_os = "macos",
    all(target_os = "linux", feature = "scarlet-wsi"),
    test
))]
pub(crate) fn wait_queue_semaphores(queue: vk::Queue, handles: &[vk::Semaphore]) -> VkResult<()> {
    let d = driver(queue.as_raw(), Kind::Queue)?;
    let semaphores = semaphore_refs(&d, handles)?;
    wait_and_consume_semaphores(&d, &semaphores)
}

unsafe extern "system" fn queue_submit(
    queue: vk::Queue,
    count: u32,
    submits: *const vk::SubmitInfo<'_>,
    fence: vk::Fence,
) -> vk::Result {
    status(queue::submit(queue, count, submits, fence))
}

fn finish_submissions(
    submissions: &[sgfx::driver::Submission],
    fence: Option<&Arc<AtomicU8>>,
    signals: &[Arc<AtomicU8>],
    timeline_signals: &[(Arc<timeline::Timeline>, u64)],
    in_flight: &InFlight,
    pending: &InFlight,
    lost: &AtomicBool,
) {
    // Drain every receipt even when one fails: failure does not free other GPU work.
    let complete = submissions.iter().enumerate().fold(true, |complete, (index, submission)| {
        let result = submission.wait(None);
        let finished = matches!(result, Ok(CompletionStatus::Complete));
        if !finished {
            eprintln!("[SGFX Vulkan] asynchronous submission completion failed: receipt={index}/{} result={result:?}", submissions.len());
        }
        finished & complete
    });
    if complete && !lost.load(Ordering::Acquire) {
        for (timeline, value) in timeline_signals {
            timeline.complete(*value);
        }
        if let Some(state) = fence {
            state.store(1, Ordering::Release);
        }
        for state in signals {
            state.store(1, Ordering::Release);
        }
    } else {
        lost.store(true, Ordering::Release);
        for (timeline, value) in timeline_signals {
            timeline.cancel(*value);
        }
        if let Some(state) = fence {
            state.store(0, Ordering::Release);
        }
        for state in signals {
            state.store(0, Ordering::Release);
        }
    }
    in_flight.finish();
    pending.finish();
}
fn validate_objects(rt: &Runtime, rec: &ResolvedRecording) -> VkResult<()> {
    if rec
        .bound_sets
        .iter()
        .any(|set| !rt.resources.descriptor_sets.contains_key(set))
    {
        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
    }
    if rec.used_framebuffers.iter().any(|f| {
        rt.resources.framebuffers.get(f).is_none_or(|fb| {
            fb.attachments.iter().any(|view| {
                !rt.resources.views.get(view).is_some_and(|view| {
                    rt.resources
                        .images
                        .get(&view.image)
                        .is_some_and(|image| image.usable())
                })
            })
        })
    }) || rec
        .used_render_passes
        .iter()
        .any(|p| !rt.resources.render_passes.contains_key(p))
        || rec
            .used_pipelines
            .iter()
            .any(|p| !rt.resources.pipelines.contains_key(p))
        || rec.used_buffers.iter().any(|b| {
            rt.resources
                .buffers
                .get(b)
                .is_none_or(|b| b.bound.is_none())
        })
        || rec
            .used_images
            .iter()
            .any(|i| rt.resources.images.get(i).is_none_or(|i| !i.usable()))
    {
        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
    }
    for d in &rec.descriptors {
        let set = rt
            .resources
            .descriptor_sets
            .get(&d.set)
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        if set.invalid {
            if std::env::var_os("SGFX_VULKAN_TRACE").is_some() {
                eprintln!(
                    "[SGFX Vulkan] invalid descriptor set {:?}: layout={:?} bindings={:?}",
                    d.set, set.layout, set.bindings
                );
            }
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        for binding in set.bindings.values() {
            let crate::resources::DescriptorBinding::Buffer { buffer, .. } = binding else {
                continue;
            };
            if !rt.resources.buffers.get(buffer).is_some_and(|b| {
                b.bound
                    .is_some_and(|(m, _)| rt.resources.memories.contains_key(&m))
            }) {
                if std::env::var_os("SGFX_VULKAN_TRACE").is_some() {
                    eprintln!(
                        "[SGFX Vulkan] descriptor buffer missing allocation: set={:?} buffer={buffer:?}",
                        d.set
                    );
                }
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            }
        }
    }
    Ok(())
}
// Vulkan's queue worker may wait for backend capacity. Busy guarantees that
// nothing was accepted, so only that result is safe to retry. The backend's
// completion thread returns capacity independently of this runtime worker.
fn submit_with_backpressure<E, S>(
    mut submit: impl FnMut() -> Result<S, SubmitError<E, S>>,
) -> Result<S, SubmitError<E, S>> {
    loop {
        match submit() {
            Err(SubmitError::Busy) => std::thread::sleep(std::time::Duration::from_millis(1)),
            result => return result,
        }
    }
}

fn submit_owned(
    rt: &mut Runtime,
    ops: Vec<ir::OwnedCommand>,
) -> VkResult<Vec<sgfx::driver::Submission>> {
    let owned = ir::OwnedCommandBuffer::new(ops);
    let operation_count = owned.command_count();
    let batches = profile_work("submit::partition_uploads", || {
        if crate::upload_batches::needs_batches(
            owned.commands(),
            crate::upload_batches::UPLOAD_BATCH_BYTES,
        ) {
            crate::upload_batches::split(owned, &rt.table, crate::upload_batches::UPLOAD_BATCH_BYTES)
                .inspect_err(|e| eprintln!("[SGFX Vulkan] upload batch preflight failed: operations={operation_count} error={e:?}"))
                .map_err(crate::resources::failure)
        } else {
            Ok(vec![owned])
        }
    })?;
    let mut submissions = Vec::new();
    for batch in batches {
        match submit_owned_batch(rt, batch) {
            Ok(Some(receipt)) => submissions.push(receipt),
            Ok(None) => {}
            Err(error) => {
                // Earlier batches are accepted work: retain and drain every
                // receipt, and never make their prefix eligible for replay.
                if !submissions.is_empty() {
                    for receipt in &submissions {
                        let _ = wait_submission(rt, receipt);
                    }
                    rt.lost = true;
                }
                return Err(error);
            }
        }
    }
    Ok(submissions)
}

fn submit_owned_batch(
    rt: &mut Runtime,
    owned: ir::OwnedCommandBuffer,
) -> VkResult<Option<sgfx::driver::Submission>> {
    if owned.is_empty() {
        return Ok(None);
    }
    let commands = profile_work("submit::record_ir", || {
        owned.record(&rt.table).map_err(|error| {
            trace_owned_recording_failure(&rt.table, &owned, error);
            crate::resources::failure(error)
        })
    })?;
    profile_work("submit::backend", || {
        match submit_with_backpressure(|| rt.queue.submit(&mut rt.cache, &commands)) {
            Ok(receipt) => {
                let draws = owned
                    .commands()
                    .iter()
                    .filter(|command| {
                        matches!(
                            command,
                            ir::OwnedCommand::Draw { .. }
                                | ir::OwnedCommand::DrawIndexed { .. }
                                | ir::OwnedCommand::DrawInstanced { .. }
                                | ir::OwnedCommand::DrawIndexedInstanced { .. }
                        )
                    })
                    .count() as u64;
                crate::progress::submitted(draws);
                Ok(Some(receipt))
            }
            Err(SubmitError::Rejected(error)) => {
                trace_owned_recording_failure(&rt.table, &owned, &error);
                let mut buffer_bytes = 0usize;
                let mut texture_bytes = 0usize;
                let mut draws = 0usize;
                let mut passes = 0usize;
                let mut largest_buffer = 0usize;
                for command in owned.commands() {
                    match command {
                        ir::OwnedCommand::WriteBuffer { data, .. } => {
                            buffer_bytes = buffer_bytes.saturating_add(data.len());
                            largest_buffer = largest_buffer.max(data.len());
                        }
                        ir::OwnedCommand::WriteTexture { data, .. }
                        | ir::OwnedCommand::WriteTextureMip { data, .. }
                        | ir::OwnedCommand::WriteTextureLayer { data, .. } => {
                            texture_bytes = texture_bytes.saturating_add(data.len());
                        }
                        ir::OwnedCommand::BeginRenderPass(_)
                        | ir::OwnedCommand::BeginRenderPassWithAttachments { .. } => passes += 1,
                        ir::OwnedCommand::Draw { .. }
                        | ir::OwnedCommand::DrawIndexed { .. }
                        | ir::OwnedCommand::DrawInstanced { .. }
                        | ir::OwnedCommand::DrawIndexedInstanced { .. } => draws += 1,
                        _ => {}
                    }
                }
                eprintln!(
                    "[SGFX Vulkan] rejected stream: operations={} buffer_bytes={buffer_bytes} texture_bytes={texture_bytes} largest_buffer={largest_buffer} passes={passes} draws={draws}",
                    owned.command_count()
                );
                Err(crate::runtime::backend_failure(error))
            }
            Err(SubmitError::Failed { error, completion }) => {
                let result = completion.wait(None);
                if std::env::var_os("SGFX_VULKAN_TRACE").is_some() {
                    eprintln!(
                        "[SGFX Vulkan] submission failed after execution began: error={error:?} completion={result:?}"
                    );
                }
                rt.lost = true;
                Err(vk::Result::ERROR_DEVICE_LOST)
            }
            Err(_) => {
                if std::env::var_os("SGFX_VULKAN_TRACE").is_some() {
                    eprintln!(
                        "[SGFX Vulkan] submission returned an unrecognized error; reporting device lost"
                    );
                }
                Err(vk::Result::ERROR_DEVICE_LOST)
            }
        }
    })
}

/// Failure-only metadata: never print upload or push-constant payloads.
fn trace_owned_recording_failure(
    table: &ir::ResourceTable,
    owned: &ir::OwnedCommandBuffer,
    error: impl std::fmt::Debug,
) {
    eprintln!(
        "[SGFX Vulkan] IR stream rejected: {error:?}, {} operations",
        owned.command_count()
    );
    for (index, command) in owned.commands().iter().enumerate().take(512) {
        match command {
            ir::OwnedCommand::WriteBuffer {
                buffer,
                offset,
                data,
            } => {
                let desc = table.buffer_ref(*buffer).and_then(|id| table.buffer(id));
                eprintln!(
                    "[SGFX Vulkan] IR[{index}] WriteBuffer {buffer:?} offset={offset} bytes={} descriptor={desc:?}",
                    data.len()
                );
            }
            ir::OwnedCommand::WriteTexture {
                texture,
                destination,
                bytes_per_row,
                data,
            }
            | ir::OwnedCommand::WriteTextureMip {
                texture,
                destination,
                bytes_per_row,
                data,
                ..
            }
            | ir::OwnedCommand::WriteTextureLayer {
                texture,
                destination,
                bytes_per_row,
                data,
                ..
            } => {
                let desc = table.texture_ref(*texture).and_then(|id| table.texture(id));
                let (mip, layer) = match command {
                    ir::OwnedCommand::WriteTextureMip { mip_level, .. } => (*mip_level, 0),
                    ir::OwnedCommand::WriteTextureLayer {
                        mip_level,
                        array_layer,
                        ..
                    } => (*mip_level, *array_layer),
                    _ => (0, 0),
                };
                eprintln!(
                    "[SGFX Vulkan] IR[{index}] WriteTexture {texture:?} mip={mip} layer={layer} destination={destination:?} stride={bytes_per_row} bytes={} descriptor={desc:?}",
                    data.len()
                );
            }
            ir::OwnedCommand::SetPushConstants {
                stages,
                offset,
                data,
            } => {
                eprintln!(
                    "[SGFX Vulkan] IR[{index}] SetPushConstants stages={stages:?} offset={offset} bytes={}",
                    data.len()
                );
            }
            ir::OwnedCommand::SetVertexBuffer { buffer, .. }
            | ir::OwnedCommand::SetVertexBufferSlot { buffer, .. }
            | ir::OwnedCommand::SetIndexBuffer { buffer, .. } => {
                let desc = table.buffer_ref(*buffer).and_then(|id| table.buffer(id));
                eprintln!("[SGFX Vulkan] IR[{index}] {command:?} descriptor={desc:?}");
            }
            ir::OwnedCommand::SetProgrammablePipeline(pipeline) => {
                let desc = table
                    .programmable_render_pipeline_ref(*pipeline)
                    .and_then(|id| table.programmable_render_pipeline_shared(id));
                eprintln!(
                    "[SGFX Vulkan] IR[{index}] {command:?} vertex_layout={:?} colors={:?} depth={:?}",
                    desc.as_ref().map(|desc| desc.vertex_buffers()),
                    desc.as_ref()
                        .map(|desc| desc.color_targets().collect::<Vec<_>>()),
                    desc.as_ref().map(|desc| desc.depth_state())
                );
            }
            ir::OwnedCommand::BeginRenderPass(desc)
            | ir::OwnedCommand::BeginRenderPassWithAttachments { desc, .. } => {
                let target = table
                    .texture_ref(desc.target)
                    .and_then(|id| table.texture(id));
                eprintln!("[SGFX Vulkan] IR[{index}] {command:?} target={target:?}");
                if let Some(depth) = desc.depth {
                    let target = table
                        .texture_ref(depth.target)
                        .and_then(|id| table.texture(id));
                    eprintln!("[SGFX Vulkan] IR[{index}] depth target={target:?}");
                }
                if let ir::OwnedCommand::BeginRenderPassWithAttachments { colors, .. } = command {
                    for color in colors {
                        let target = table
                            .texture_ref(color.target)
                            .and_then(|id| table.texture(id));
                        eprintln!("[SGFX Vulkan] IR[{index}] additional color target={target:?}");
                    }
                }
            }
            ir::OwnedCommand::SetBindGroup { bind_group, .. } => {
                let desc = table
                    .bind_group_ref(*bind_group)
                    .and_then(|id| table.bind_group(id));
                eprintln!("[SGFX Vulkan] IR[{index}] {command:?} descriptor={desc:?}");
            }
            _ => eprintln!("[SGFX Vulkan] IR[{index}] {command:?}"),
        }
    }
}

fn wait_submission(rt: &mut Runtime, submission: &sgfx::driver::Submission) -> VkResult<()> {
    match submission.wait(None) {
        Ok(CompletionStatus::Complete) => Ok(()),
        Ok(status) => {
            if std::env::var_os("SGFX_VULKAN_TRACE").is_some() {
                eprintln!(
                    "[SGFX Vulkan] synchronous submission completion did not complete: {status:?}"
                );
            }
            rt.lost = true;
            Err(vk::Result::ERROR_DEVICE_LOST)
        }
        Err(error) => {
            if std::env::var_os("SGFX_VULKAN_TRACE").is_some() {
                eprintln!("[SGFX Vulkan] synchronous submission completion failed: {error:?}");
            }
            let result = crate::runtime::backend_failure(error);
            if result == vk::Result::ERROR_DEVICE_LOST {
                rt.lost = true;
            }
            Err(result)
        }
    }
}

/// Resolution appends insertions at `ops.len()`, in nondecreasing order.
/// Consume each insertion once instead of scanning the entire list per op.
fn take_positioned<'a, T>(
    remaining: &mut &'a [T],
    position: usize,
    key: impl Fn(&T) -> usize,
) -> &'a [T] {
    let count = remaining
        .iter()
        .take_while(|item| key(item) == position)
        .count();
    let (current, later) = remaining.split_at(count);
    *remaining = later;
    current
}

#[cfg(test)]
fn snapshot_buffer_uploads(
    resources: &crate::resources::Resources,
    used: &[vk::Buffer],
) -> VkResult<BufferUploads> {
    buffer_uploads::whole(resources, used)
}

fn execute(
    rt: &mut Runtime,
    rec: ResolvedRecording,
    retained_submissions: &mut Vec<sgfx::driver::Submission>,
) -> VkResult<()> {
    let (mut ops, fresh_uploads) = profile_work("execute::snapshot_buffers", || {
        buffer_uploads::for_recording(&rt.resources, &rec.used_buffers, &rec.descriptors)
    })
    .inspect_err(|error| {
        if std::env::var_os("SGFX_VULKAN_TRACE").is_some() {
            eprintln!(
                "[SGFX Vulkan] buffer snapshot failed: {error:?} used={:?}",
                rec.used_buffers
            );
        }
    })?;
    let mut descriptors = rec.descriptors.as_slice();
    let mut barriers = rec.barriers.as_slice();
    let mut copies = rec.copies.as_slice();
    let mut synchronizations = rec.synchronizations.as_slice();
    // Resolution already owns upload payloads. Transfer them to the final
    // stream instead of cloning every texture/buffer byte a second time.
    // Include the final insertion position after the last recorded command.
    let command_count = rec.ops.len();
    let mut commands = rec.ops.into_iter();
    for position in 0..=command_count {
        for d in take_positioned(&mut descriptors, position, |d| d.position) {
            let group = rt.resources.descriptor_group_with_layout(
                &rt.table,
                d.set,
                &d.dynamic_offsets,
                &d.layout,
                rt.capabilities
                    .limits()
                    .min_uniform_buffer_offset_alignment
                    .max(16),
                rt.capabilities
                    .limits()
                    .min_storage_buffer_offset_alignment
                    .max(4),
            ).inspect_err(|error| {
                {
                    eprintln!("[SGFX Vulkan] descriptor materialization failed: {error:?} set={:?} offsets={:?} layout={:?}", d.set, d.dynamic_offsets, d.layout);
                    if let Some(set) = rt.resources.descriptor_sets.get(&d.set) {
                        eprintln!("[SGFX Vulkan] descriptor contents: invalid={} layout={:?} types={:?} bindings={:?}", set.invalid, set.layout, set.types, set.bindings);
                    }
                }
            })?;
            ops.push(ir::OwnedCommand::SetBindGroup {
                index: d.index,
                bind_group: group,
            });
        }
        for barrier in take_positioned(&mut barriers, position, |b| b.position) {
            let mut ids = Vec::new();
            for buffer in &barrier.buffers {
                let buffer = rt
                    .resources
                    .buffers
                    .get(buffer)
                    .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                if !ids.contains(&buffer.id) {
                    ids.push(buffer.id);
                }
            }
            for buffer in ids {
                ops.push(ir::OwnedCommand::ResourceBarrier(
                    ir::OwnedResourceBarrier::Buffer {
                        buffer,
                        before: ir::BufferAccess::StorageReadWrite,
                        after: barrier.after,
                    },
                ));
            }
        }
        for copy in take_positioned(&mut copies, position, |c| c.position) {
            while synchronizations
                .first()
                .is_some_and(|s| s.position == position && s.order <= copy.order)
            {
                synchronization2::execute(
                    rt,
                    &mut ops,
                    &synchronizations[0].action,
                    retained_submissions,
                )?;
                synchronizations = &synchronizations[1..];
            }
            profile_work("execute::texture_readback", || {
                retained_submissions.extend(submit_owned(rt, std::mem::take(&mut ops))?);
                for submission in retained_submissions.drain(..) {
                    wait_submission(rt, &submission)?;
                }
                let image_id = rt
                    .resources
                    .images
                    .get(&copy.image)
                    .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?
                    .id;
                let bytes = rt
                    .queue
                    .read_texture_mip(&mut rt.cache, image_id, copy.mip_level)
                    .map_err(|_| vk::Result::ERROR_DEVICE_LOST)?;
                let buffer = rt
                    .resources
                    .buffers
                    .get(&copy.buffer)
                    .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                ops.extend(copy.region.writes(&bytes, buffer.id)?);
                Ok(())
            })?;
        }
        while synchronizations
            .first()
            .is_some_and(|s| s.position == position)
        {
            synchronization2::execute(
                rt,
                &mut ops,
                &synchronizations[0].action,
                retained_submissions,
            )?;
            synchronizations = &synchronizations[1..];
        }
        if let Some(op) = commands.next() {
            ops.push(op);
        }
    }
    retained_submissions.extend(submit_owned(rt, ops)?);
    let submissions = retained_submissions;
    let readback = rec.readback_buffers;
    if !readback.is_empty() {
        profile_work("execute::readback_wait", || {
            for submission in submissions.iter() {
                wait_submission(rt, submission)?;
            }
            Ok(())
        })?;
        submissions.clear();
    }
    for (handle, _) in fresh_uploads {
        rt.resources.uploaded_unmapped_buffers.insert(handle);
    }
    // The backend and this buffer's mirror agree after readback. Only aliases
    // overlapping the modified allocation bytes need a later full upload.
    for (handle, ranges) in readback.into_ranges() {
        profile_work("execute::buffer_readback", || {
            let buffer = rt
                .resources
                .buffers
                .get(&handle)
                .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
            let id = buffer.id;
            if let Some((memory_handle, offset)) = buffer.bound {
                for range in ranges {
                    let bytes = rt
                        .cache
                        .read_buffer(id, range.start, range.end - range.start)
                        .map_err(|_| vk::Result::ERROR_DEVICE_LOST)?;
                    let allocation = rt
                        .resources
                        .memories
                        .get_mut(&memory_handle)
                        .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                    allocation.bytes
                        [(offset + range.start) as usize..(offset + range.end) as usize]
                        .copy_from_slice(&bytes);
                    rt.resources.forget_uploads_for_memory_range(
                        memory_handle,
                        offset + range.start..offset + range.end,
                        handle,
                    );
                }
            }
            Ok(())
        })?;
    }
    synchronization2::reap_completed(rt, submissions)?;
    Ok(())
}

pub(crate) unsafe extern "system" fn get_device_proc_addr(
    device: vk::Device,
    name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    if name.is_null() || driver(device.as_raw(), Kind::Device).is_err() {
        return None;
    }
    let name = CStr::from_ptr(name);
    if matches!(
        name.to_bytes(),
        b"vkCreateDevice" | b"vkGetInstanceProcAddr"
    ) {
        return None;
    }
    lookup_device(name)
}
pub(crate) fn lookup_device(name: &CStr) -> vk::PFN_vkVoidFunction {
    if let Some(function) = core11::lookup(name) {
        return Some(function);
    }
    macro_rules! entry {
        ($function:path) => {
            Some(unsafe {
                std::mem::transmute::<*const (), unsafe extern "system" fn()>(
                    $function as *const (),
                )
            })
        };
    }
    match name.to_bytes() {
        b"vkCreateDevice" => entry!(create_device),
        b"vkDestroyDevice" => entry!(destroy_device),
        b"vkGetDeviceProcAddr" => entry!(get_device_proc_addr),
        b"vkGetDeviceQueue" => entry!(get_device_queue),
        b"vkDeviceWaitIdle" => entry!(device_wait_idle),
        b"vkQueueWaitIdle" => entry!(queue_wait_idle),
        b"vkQueueSubmit" => entry!(queue_submit),
        b"vkCreateCommandPool" => entry!(create_command_pool),
        b"vkDestroyCommandPool" => entry!(destroy_command_pool),
        b"vkResetCommandPool" => entry!(reset_command_pool),
        b"vkTrimCommandPoolKHR" => entry!(maintenance1::trim_command_pool),
        b"vkAllocateCommandBuffers" => entry!(allocate_command_buffers),
        b"vkFreeCommandBuffers" => entry!(free_command_buffers),
        b"vkResetCommandBuffer" => entry!(reset_command_buffer),
        b"vkBeginCommandBuffer" => entry!(begin_command_buffer),
        b"vkEndCommandBuffer" => entry!(end_command_buffer),
        b"vkCmdBindPipeline" => entry!(cmd_bind_pipeline),
        b"vkCmdBindDescriptorSets" => entry!(cmd_bind_descriptor_sets),
        b"vkCmdPushConstants" => entry!(cmd_push_constants),
        b"vkCmdBlitImage" => entry!(cmd_blit_image),
        b"vkCmdCopyImage" => entry!(cmd_copy_image),
        b"vkCmdDispatch" => entry!(cmd_dispatch),
        b"vkCmdDraw" => entry!(cmd_draw),
        b"vkCmdDrawIndexed" => entry!(cmd_draw_indexed),
        b"vkCmdBindVertexBuffers" => entry!(cmd_bind_vertex_buffers),
        b"vkCmdBindIndexBuffer" => entry!(cmd_bind_index_buffer),
        b"vkCmdBeginRenderPass" => entry!(cmd_begin_render_pass),
        b"vkCmdNextSubpass" => entry!(cmd_next_subpass),
        b"vkCmdEndRenderPass" => entry!(cmd_end_render_pass),
        b"vkCmdCopyImageToBuffer" => entry!(cmd_copy_image_to_buffer),
        b"vkCmdClearDepthStencilImage" => entry!(cmd_clear_depth_image),
        b"vkCmdClearColorImage" => entry!(cmd_clear_color_image),
        b"vkCmdClearAttachments" => entry!(cmd_clear_attachments),
        b"vkCmdCopyBuffer" => entry!(cmd_copy_buffer),
        b"vkCmdCopyBufferToImage" => entry!(cmd_copy_buffer_to_image),
        b"vkCmdSetViewport" => entry!(cmd_set_viewport),
        b"vkCmdSetScissor" => entry!(cmd_set_scissor),
        b"vkCmdPipelineBarrier" => entry!(cmd_pipeline_barrier),
        b"vkCreateFence" => entry!(create_fence),
        b"vkDestroyFence" => entry!(destroy_fence),
        b"vkResetFences" => entry!(reset_fences),
        b"vkGetFenceStatus" => entry!(get_fence_status),
        b"vkWaitForFences" => entry!(wait_for_fences),
        b"vkCreateSemaphore" => entry!(create_semaphore),
        b"vkDestroySemaphore" => entry!(destroy_semaphore),
        #[cfg(any(
            target_os = "scarlet",
            target_os = "macos",
            all(target_os = "linux", feature = "scarlet-wsi")
        ))]
        b"vkCreateSwapchainKHR" => entry!(crate::wsi::create_swapchain),
        #[cfg(any(
            target_os = "scarlet",
            target_os = "macos",
            all(target_os = "linux", feature = "scarlet-wsi")
        ))]
        b"vkDestroySwapchainKHR" => entry!(crate::wsi::destroy_swapchain),
        #[cfg(any(
            target_os = "scarlet",
            target_os = "macos",
            all(target_os = "linux", feature = "scarlet-wsi")
        ))]
        b"vkGetSwapchainImagesKHR" => entry!(crate::wsi::get_swapchain_images),
        #[cfg(any(
            target_os = "scarlet",
            target_os = "macos",
            all(target_os = "linux", feature = "scarlet-wsi")
        ))]
        b"vkAcquireNextImageKHR" => entry!(crate::wsi::acquire_next_image),
        #[cfg(any(
            target_os = "scarlet",
            target_os = "macos",
            all(target_os = "linux", feature = "scarlet-wsi")
        ))]
        b"vkQueuePresentKHR" => entry!(crate::wsi::queue_present),
        #[cfg(target_os = "scarlet")]
        b"vkAcquireNextImage2KHR" => entry!(crate::wsi::acquire_next_image2),
        #[cfg(target_os = "scarlet")]
        b"vkGetDeviceGroupPresentCapabilitiesKHR" => {
            entry!(crate::wsi::device_group_present_capabilities)
        }
        #[cfg(target_os = "scarlet")]
        b"vkGetDeviceGroupSurfacePresentModesKHR" => {
            entry!(crate::wsi::device_group_surface_present_modes)
        }
        #[cfg(target_os = "scarlet")]
        b"vkGetImageScarletHandleSGFX" => {
            entry!(crate::scarlet_image::vkGetImageScarletHandleSGFX)
        }
        _ => core12::lookup(name)
            .or_else(|| push_descriptors::lookup(name))
            .or_else(|| descriptor_template::lookup(name))
            .or_else(|| dynamic_states::lookup(name))
            .or_else(|| buffer_views::lookup(name))
            .or_else(|| render_pass2::lookup(name))
            .or_else(|| timeline::lookup(name))
            .or_else(|| synchronization2::lookup(name))
            .or_else(|| dynamic_rendering::lookup(name))
            .or_else(|| queries::lookup(name))
            .or_else(|| indirect::lookup(name))
            .or_else(|| bootstrap::lookup(name))
            .or_else(|| buffer_operations::lookup(name))
            .or_else(|| unsupported::lookup(name))
            .or_else(|| crate::resources::lookup(name))
            .or_else(|| crate::images::lookup(name)),
    }
}

/// Descriptor indexing is not exposed. Updating a referenced set invalidates
/// every recording or executable command buffer under Vulkan 1.0 rules.
fn invalidate_recorded_descriptor_sets(recordings: &Recordings, sets: &[vk::DescriptorSet]) {
    let recordings = recordings
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .commands
        .values()
        .cloned()
        .collect::<Vec<_>>();
    for recording in recordings {
        let mut recording = recording
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if recording
            .commands
            .iter()
            .any(|command| command.references_descriptor_sets(sets))
        {
            recording.fail(vk::Result::ERROR_INITIALIZATION_FAILED);
            if recording.state == RecordingState::Executable {
                recording.state = RecordingState::Invalid;
            }
        }
    }
}

fn active_sets(
    resources: &crate::resources::Resources,
    layout: &ir::PipelineLayoutDesc,
    sets: &BTreeMap<u32, vk::DescriptorSet>,
) -> VkResult<Vec<(u32, vk::DescriptorSet)>> {
    let mut active = Vec::new();
    for (index, expected) in layout.bind_groups().iter().enumerate() {
        if expected.entries().is_empty() {
            continue;
        }
        let set = *sets
            .get(&(index as u32))
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        if resources
            .descriptor_sets
            .get(&set)
            .is_none_or(|s| !crate::resources::specialized_layout_matches(&s.layout, expected))
        {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        active.push((index as u32, set));
    }
    Ok(active)
}

pub(crate) fn invalidate_resource_recordings(rt: &mut Runtime) {
    let recordings = rt
        .recordings
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .commands
        .values()
        .cloned()
        .collect::<Vec<_>>();
    for recording in recordings {
        let mut recording = recording
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !recording.commands.is_empty() {
            recording.state = RecordingState::Invalid;
            recording.fail(vk::Result::ERROR_INITIALIZATION_FAILED);
            recording.commands = Arc::default();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_backpressure_waits_for_independent_completion_capacity() {
        let (entered, started) = mpsc::sync_channel(1);
        let (credit, available) = mpsc::sync_channel(1);
        let (finished, result) = mpsc::sync_channel(1);
        let worker = std::thread::spawn(move || {
            let mut first = true;
            let outcome: Result<usize, SubmitError<&str, usize>> = submit_with_backpressure(|| {
                if first {
                    entered.send(()).unwrap();
                    first = false;
                }
                match available.try_recv() {
                    Ok(receipt) => Ok(receipt),
                    Err(mpsc::TryRecvError::Empty) => Err(SubmitError::Busy),
                    Err(mpsc::TryRecvError::Disconnected) => Err(SubmitError::Rejected("closed")),
                }
            });
            finished.send(outcome).unwrap();
        });
        let timeout = std::time::Duration::from_secs(2);
        started.recv_timeout(timeout).unwrap();
        assert!(matches!(result.try_recv(), Err(mpsc::TryRecvError::Empty)));
        credit.send(17).unwrap();
        assert!(matches!(result.recv_timeout(timeout).unwrap(), Ok(17)));
        worker.join().unwrap();
    }

    #[test]
    fn queue_backpressure_never_replays_rejected_or_partially_accepted_work() {
        let mut calls = 0;
        let result = submit_with_backpressure(|| {
            calls += 1;
            Err::<usize, _>(SubmitError::Failed {
                error: "failed prefix",
                completion: 19,
            })
        });
        assert!(matches!(
            result,
            Err(SubmitError::Failed {
                error: "failed prefix",
                completion: 19
            })
        ));
        assert_eq!(calls, 1);
        calls = 0;
        let result = submit_with_backpressure(|| {
            calls += 1;
            Err::<usize, SubmitError<_, usize>>(SubmitError::Rejected("invalid"))
        });
        assert!(matches!(result, Err(SubmitError::Rejected("invalid"))));
        assert_eq!(calls, 1);
    }

    #[test]
    fn call_profile_counts_worker_errors_and_missing_channel_responses_separately() {
        let mut counters = CallProfileCounters::default();
        counters.observe(
            30_500,
            Some(CallServiceTiming {
                queued_ns: 4_000,
                service_ns: 10_500,
            }),
            false,
        );
        counters.observe(
            50_500,
            Some(CallServiceTiming {
                queued_ns: 5_000,
                service_ns: 20_500,
            }),
            true,
        );
        counters.observe(7_500, None, true);
        assert_eq!(
            (counters.calls, counters.errors, counters.services),
            (3, 2, 2)
        );
        assert_eq!(counters.roundtrip_ns, 88_500);
        assert_eq!(counters.queued_ns, 9_000);
        assert_eq!(counters.service_ns, 31_000);
        assert_eq!(counters.max_roundtrip_ns, 50_500);
        assert_eq!(counters.max_service_ns, 20_500);
    }

    #[test]
    fn call_profile_labels_keep_api_names_and_bound_multibyte_type_names() {
        assert_eq!(
            call_profile_label(
                "vulkan_sgfx::resources::update_descriptor_sets::{{closure}}::{{closure}}"
            ),
            "resources::update_descriptor_sets"
        );
        let long_name = "あ".repeat(100);
        let label = call_profile_label(&long_name);
        assert!(label.len() <= 120);
        assert!(long_name.ends_with(label));
    }

    #[test]
    fn copy_image_is_a_transfer_and_cannot_be_recorded_inside_render_pass() {
        let mut recording = Recording::new(0);
        recording.state = RecordingState::Recording;
        recording.render_active = true;
        assert_eq!(
            recording.push(RecordedCommand::CopyImage {
                source: vk::Image::null(),
                source_layout: vk::ImageLayout::GENERAL,
                destination: vk::Image::null(),
                destination_layout: vk::ImageLayout::GENERAL,
                regions: Vec::new(),
            }),
            Err(vk::Result::ERROR_INITIALIZATION_FAILED)
        );
        assert!(recording.commands.is_empty());
    }

    #[test]
    fn clear_attachments_requires_recording_inside_render_pass() {
        let clear = || RecordedCommand::ClearAttachments {
            attachments: Vec::new(),
            rects: Vec::new(),
        };
        let mut outside = Recording::new(0);
        outside.state = RecordingState::Recording;
        assert_eq!(
            outside.push(clear()),
            Err(vk::Result::ERROR_INITIALIZATION_FAILED)
        );
        assert!(outside.commands.is_empty());
        let mut inside = Recording::new(0);
        inside.state = RecordingState::Recording;
        inside.render_active = true;
        inside.push(clear()).unwrap();
        assert_eq!(inside.commands.len(), 1);
        assert!(inside.render_active);
        assert!(inside.error.is_none());
    }

    #[test]
    fn barriers_accept_explicit_same_family_without_ownership_transfer() {
        assert!(same_queue_family(0, 0));
        assert!(same_queue_family(
            vk::QUEUE_FAMILY_IGNORED,
            vk::QUEUE_FAMILY_IGNORED
        ));
        for (source, destination) in [
            (0, 1),
            (1, 0),
            (1, 1),
            (0, vk::QUEUE_FAMILY_IGNORED),
            (vk::QUEUE_FAMILY_EXTERNAL, 0),
            (vk::QUEUE_FAMILY_EXTERNAL, vk::QUEUE_FAMILY_EXTERNAL),
        ] {
            assert!(!same_queue_family(source, destination));
        }
    }

    #[test]
    fn unmapped_buffer_uploads_are_reused_until_host_mapping_changes_them() {
        use crate::resources::{AlignedBytes, Buffer, Memory, Resources};

        let table = ir::ResourceTable::new();
        let id = table
            .define_buffer(ir::BufferDesc::new(8, ir::BufferUsage::COPY_DST).unwrap())
            .unwrap()
            .id();
        let handle = vk::Buffer::from_raw(1);
        let memory = vk::DeviceMemory::from_raw(2);
        let mut resources = Resources::new();
        let mut bytes = AlignedBytes::zeroed(16).unwrap();
        bytes[4..9].copy_from_slice(&[1, 2, 3, 4, 5]);
        resources.memories.insert(
            memory,
            Memory {
                bytes,
                mapped: false,
            },
        );
        resources.buffers.insert(
            handle,
            Buffer {
                id,
                size: 5,
                usage: vk::BufferUsageFlags::TRANSFER_DST,
                bound: Some((memory, 4)),
            },
        );

        let (initial, fresh) = snapshot_buffer_uploads(&resources, &[handle]).unwrap();
        assert!(matches!(initial.as_slice(),
            [ir::OwnedCommand::WriteBuffer { data, .. }] if data == &[1, 2, 3, 4, 5, 0, 0, 0]));
        assert_eq!(fresh, vec![(handle, memory)]);

        resources.uploaded_unmapped_buffers.insert(handle);
        assert!(
            snapshot_buffer_uploads(&resources, &[handle])
                .unwrap()
                .0
                .is_empty()
        );

        resources.forget_uploads_for_memory(memory);
        resources.memories.get_mut(&memory).unwrap().bytes[4] = 9;
        let (updated, fresh) = snapshot_buffer_uploads(&resources, &[handle]).unwrap();
        assert!(matches!(updated.as_slice(),
            [ir::OwnedCommand::WriteBuffer { data, .. }] if data == &[9, 2, 3, 4, 5, 0, 0, 0]));
        assert_eq!(fresh, vec![(handle, memory)]);

        resources.memories.get_mut(&memory).unwrap().mapped = true;
        resources.uploaded_unmapped_buffers.insert(handle);
        let (mapped, fresh) = snapshot_buffer_uploads(&resources, &[handle]).unwrap();
        assert_eq!(mapped.len(), 1);
        assert!(fresh.is_empty());
    }

    #[test]
    fn gpu_readback_invalidates_only_overlapping_aliases() {
        use crate::resources::{Buffer, Resources};
        let table = ir::ResourceTable::new();
        let mut resources = Resources::new();
        let memory = vk::DeviceMemory::from_raw(1);
        let other_memory = vk::DeviceMemory::from_raw(2);
        for (raw, bound, offset) in [
            (1, memory, 0),
            (2, memory, 0),
            (3, memory, 16),
            (4, memory, 32),
            (5, other_memory, 0),
        ] {
            let handle = vk::Buffer::from_raw(raw);
            let id = table
                .define_buffer(ir::BufferDesc::new(16, ir::BufferUsage::COPY_DST).unwrap())
                .unwrap()
                .id();
            resources.buffers.insert(
                handle,
                Buffer {
                    id,
                    size: 16,
                    usage: vk::BufferUsageFlags::TRANSFER_DST,
                    bound: Some((bound, offset)),
                },
            );
            resources.uploaded_unmapped_buffers.insert(handle);
        }
        // Updating the first buffer preserves its synchronized backend copy,
        // invalidates its alias, and leaves adjacent buffers cached.
        resources.forget_uploads_for_memory_range(memory, 4..16, vk::Buffer::from_raw(1));
        assert_eq!(
            (1..=5)
                .filter(|raw| resources
                    .uploaded_unmapped_buffers
                    .contains(&vk::Buffer::from_raw(*raw)))
                .collect::<Vec<_>>(),
            [1, 3, 4, 5]
        );
        // A subsequent write via the alias invalidates the original buffer;
        // it must not be re-marked as fresh at the end of the submission.
        resources.forget_uploads_for_memory_range(memory, 4..8, vk::Buffer::from_raw(2));
        assert!(
            !resources
                .uploaded_unmapped_buffers
                .contains(&vk::Buffer::from_raw(1))
        );
    }

    #[test]
    fn submission_snapshots_share_commands_and_survive_re_recording() {
        let mut source = Recording::new(1);
        source.state = RecordingState::Recording;
        source
            .push(RecordedCommand::SetScissor(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: 16,
                    height: 16,
                },
            }))
            .unwrap();
        let snapshot = source.clone();
        assert!(Arc::ptr_eq(&source.commands, &snapshot.commands));
        source
            .push(RecordedCommand::SetScissor(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: 8,
                    height: 8,
                },
            }))
            .unwrap();
        assert_eq!(snapshot.commands.len(), 1);
        assert_eq!(source.commands.len(), 2);
        assert!(!Arc::ptr_eq(&source.commands, &snapshot.commands));
        source.commands = Arc::default();
        assert!(matches!(snapshot.commands.as_slice(),
            [RecordedCommand::SetScissor(rect)] if rect.extent.width == 16));
    }

    fn graphics_fixture() -> (
        ir::ResourceTable,
        crate::resources::Resources,
        ResolvedRecording,
        ir::PipelineLayoutDesc,
    ) {
        let table = ir::ResourceTable::new();
        let shader = table
            .define_shader_module(
                ir::ShaderModuleDesc::wgsl(
                    "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }
             @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }"
                        .into(),
                )
                .unwrap(),
            )
            .unwrap();
        let group_layout = ir::BindGroupLayoutDesc::new(vec![ir::BindGroupLayoutEntry::new(
            0,
            ir::ShaderStages::VERTEX,
            ir::BindingType::UniformBuffer,
        )])
        .unwrap();
        let layout = ir::PipelineLayoutDesc::new(vec![group_layout.clone()])
            .unwrap()
            .with_push_constant_ranges(vec![
                ir::PushConstantRange::new(ir::ShaderStages::VERTEX, 0, 4).unwrap(),
            ])
            .unwrap();
        let pipeline_desc = ir::ProgrammableRenderPipelineDesc::new(
            ir::ShaderEntryPoint::new(shader, ir::ShaderStage::Vertex, "vs".into()).unwrap(),
            ir::ShaderEntryPoint::new(shader, ir::ShaderStage::Fragment, "fs".into()).unwrap(),
            layout.clone(),
            ir::TextureFormat::Rgba8Unorm,
            Some(
                ir::VertexBufferLayout::new(
                    8,
                    vec![ir::VertexAttribute::new(0, ir::VertexFormat::Float32x2, 0)],
                )
                .unwrap(),
            ),
            ir::PrimitiveTopology::TriangleList,
            ir::BlendState::REPLACE,
            ir::RasterState::new(ir::CullMode::None, ir::FrontFace::CounterClockwise),
        )
        .unwrap();
        let mut resources = crate::resources::Resources::new();
        for handle in [1, 2] {
            let pipeline = vk::Pipeline::from_raw(handle);
            let id = table
                .define_programmable_render_pipeline(pipeline_desc.clone())
                .unwrap()
                .id();
            resources
                .pipelines
                .insert(pipeline, crate::resources::Pipeline::Graphics(id));
            resources.graphics_state.insert(
                pipeline,
                crate::images::GraphicsDynamicState {
                    viewport: None,
                    scissor: None,
                },
            );
        }
        let buffer = vk::Buffer::from_raw(3);
        let id = table
            .define_buffer(
                ir::BufferDesc::new(1024, ir::BufferUsage::VERTEX | ir::BufferUsage::UNIFORM)
                    .unwrap(),
            )
            .unwrap()
            .id();
        resources.buffers.insert(
            buffer,
            crate::resources::Buffer {
                id,
                size: 1024,
                usage: vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::UNIFORM_BUFFER,
                bound: Some((vk::DeviceMemory::from_raw(4), 0)),
            },
        );
        for handle in [5, 6] {
            resources.descriptor_sets.insert(
                vk::DescriptorSet::from_raw(handle),
                crate::resources::DescriptorSet::new(
                    HashMap::new(),
                    vk::DescriptorPool::from_raw(7),
                    group_layout.clone(),
                    BTreeMap::from([(0, vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)]),
                    BTreeMap::from([(0, vk::ShaderStageFlags::VERTEX)]),
                    HashMap::from([(
                        0,
                        crate::resources::DescriptorBinding::Buffer {
                            buffer,
                            offset: 0,
                            range: 16,
                        },
                    )]),
                )
                .unwrap(),
            );
        }
        let mut rec = ResolvedRecording::new();
        rec.render = Some((32, 32));
        rec.graphics = Some(vk::Pipeline::from_raw(1));
        rec.vertex_buffers[0] = Some((buffer, 0));
        rec.viewport = Some(vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: 32.0,
            height: 32.0,
            min_depth: 0.0,
            max_depth: 1.0,
        });
        rec.scissor = Some(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: 32,
                height: 32,
            },
        });
        rec.graphics_sets.insert(0, vk::DescriptorSet::from_raw(5));
        rec.graphics_offsets.insert(0, vec![0]);
        (table, resources, rec, layout)
    }

    #[test]
    fn repeated_graphics_draws_keep_state_and_still_validate_live_bindings() {
        let (table, mut resources, mut rec, _) = graphics_fixture();
        assert!(graphics_bindings(&table, &mut resources, &mut rec).unwrap());
        assert_eq!(rec.ops.len(), 5);
        assert_eq!(rec.descriptors.len(), 1);
        for _ in 0..1000 {
            assert!(graphics_bindings(&table, &mut resources, &mut rec).unwrap());
            rec.ops.push(ir::OwnedCommand::Draw {
                vertex_count: 3,
                first_vertex: 0,
            });
        }
        assert_eq!(rec.ops.len(), 1005);
        assert_eq!(rec.descriptors.len(), 1);
        // Repeating an old binding must not conceal destruction of its object.
        resources
            .descriptor_sets
            .remove(&vk::DescriptorSet::from_raw(5));
        assert_eq!(
            graphics_bindings(&table, &mut resources, &mut rec),
            Err(vk::Result::ERROR_INITIALIZATION_FAILED)
        );
        resources.buffers.remove(&vk::Buffer::from_raw(3));
        assert_eq!(
            graphics_bindings(&table, &mut resources, &mut rec),
            Err(vk::Result::ERROR_INITIALIZATION_FAILED)
        );
    }

    #[test]
    fn pipeline_changes_clear_unused_ir_groups_but_preserve_vulkan_sets() {
        let (table, mut resources, mut rec, layout) = graphics_fixture();
        let crate::resources::Pipeline::Graphics(original) =
            resources.pipelines[&vk::Pipeline::from_raw(1)]
        else {
            panic!("graphics pipeline expected")
        };
        let original = table
            .programmable_render_pipeline_shared(
                table.programmable_render_pipeline_ref(original).unwrap(),
            )
            .unwrap();
        for (handle, second_group) in [
            (10, layout.bind_groups()[0].clone()),
            (11, ir::BindGroupLayoutDesc::new(Vec::new()).unwrap()),
        ] {
            let pipeline_layout =
                ir::PipelineLayoutDesc::new(vec![layout.bind_groups()[0].clone(), second_group])
                    .unwrap()
                    .with_push_constant_ranges(layout.push_constant_ranges().to_vec())
                    .unwrap();
            let desc = ir::ProgrammableRenderPipelineDesc::new(
                original.vertex().clone(),
                original.fragment().clone(),
                pipeline_layout,
                original.target_format(),
                None,
                original.topology(),
                original.blend(),
                original.raster(),
            )
            .unwrap()
            .with_vertex_buffers(original.vertex_buffers().to_vec())
            .unwrap();
            let pipeline = vk::Pipeline::from_raw(handle);
            let id = table
                .define_programmable_render_pipeline(desc)
                .unwrap()
                .id();
            resources
                .pipelines
                .insert(pipeline, crate::resources::Pipeline::Graphics(id));
            resources.graphics_state.insert(
                pipeline,
                crate::images::GraphicsDynamicState {
                    viewport: None,
                    scissor: None,
                },
            );
        }
        rec.graphics_sets.insert(1, vk::DescriptorSet::from_raw(6));
        rec.graphics_offsets.insert(1, vec![0]);
        rec.graphics = Some(vk::Pipeline::from_raw(10));
        graphics_bindings(&table, &mut resources, &mut rec).unwrap();
        assert!(
            rec.descriptors
                .iter()
                .any(|binding| binding.set == vk::DescriptorSet::from_raw(6))
        );
        let old_sets = rec.graphics_sets.clone();
        let previous = rec.ops.len();
        rec.graphics = Some(vk::Pipeline::from_raw(11));
        graphics_bindings(&table, &mut resources, &mut rec).unwrap();
        assert_eq!(rec.graphics_sets, old_sets);
        let empty_group = rec.ops[previous..]
            .iter()
            .find_map(|command| match command {
                ir::OwnedCommand::SetBindGroup {
                    index: 1,
                    bind_group,
                } => Some(*bind_group),
                _ => None,
            })
            .expect("the stale SGFX slot must be replaced on pipeline change");
        assert!(
            table
                .bind_group_shared(table.bind_group_ref(empty_group).unwrap())
                .unwrap()
                .layout()
                .entries()
                .is_empty()
        );
        assert_eq!(resources.empty_bind_group(&table).unwrap(), empty_group);
        let previous = rec.ops.len();
        graphics_bindings(&table, &mut resources, &mut rec).unwrap();
        assert_eq!(
            rec.ops.len(),
            previous,
            "repeated draws must not rebind the empty slot"
        );
        let previous = rec.descriptors.len();
        rec.graphics = Some(vk::Pipeline::from_raw(10));
        graphics_bindings(&table, &mut resources, &mut rec).unwrap();
        assert!(
            rec.descriptors[previous..]
                .iter()
                .any(|binding| binding.set == vk::DescriptorSet::from_raw(6))
        );
    }

    #[test]
    fn graphics_state_changes_emit_selective_updates_and_reseed_new_passes() {
        let (table, mut resources, mut rec, layout) = graphics_fixture();
        graphics_bindings(&table, &mut resources, &mut rec).unwrap();
        rec.graphics_offsets.insert(0, vec![256]);
        graphics_bindings(&table, &mut resources, &mut rec).unwrap();
        assert_eq!(rec.ops.len(), 5);
        assert_eq!(rec.descriptors.len(), 2);
        assert_eq!(rec.descriptors[1].dynamic_offsets, [256]);
        rec.graphics_sets.insert(0, vk::DescriptorSet::from_raw(6));
        graphics_bindings(&table, &mut resources, &mut rec).unwrap();
        assert_eq!(rec.descriptors.len(), 3);
        assert_eq!(rec.descriptors[2].set, vk::DescriptorSet::from_raw(6));
        rec.viewport.as_mut().unwrap().width = 16.0;
        rec.scissor.as_mut().unwrap().extent.width = 16;
        rec.vertex_buffers[0].as_mut().unwrap().1 = 4;
        graphics_bindings(&table, &mut resources, &mut rec).unwrap();
        assert!(matches!(
            &rec.ops[5..],
            [
                ir::OwnedCommand::SetViewport(_),
                ir::OwnedCommand::SetScissor(_),
                ir::OwnedCommand::SetVertexBuffer { offset: 4, .. }
            ]
        ));
        rec.push_constants
            .update(&layout, ir::ShaderStages::VERTEX, 0, &[9; 4])
            .unwrap();
        rec.push_constants_revision += 1;
        graphics_bindings(&table, &mut resources, &mut rec).unwrap();
        assert!(
            matches!(rec.ops.last().unwrap(), ir::OwnedCommand::SetPushConstants { data, .. } if data == &[9; 4])
        );
        rec.graphics = Some(vk::Pipeline::from_raw(2));
        let previous = rec.ops.len();
        graphics_bindings(&table, &mut resources, &mut rec).unwrap();
        assert!(
            matches!(&rec.ops[previous..], [ir::OwnedCommand::SetProgrammablePipeline(_), ir::OwnedCommand::SetPushConstants { data, .. }] if data == &[9; 4])
        );
        rec.emitted_graphics = Default::default();
        let previous = rec.ops.len();
        graphics_bindings(&table, &mut resources, &mut rec).unwrap();
        assert_eq!(rec.ops.len() - previous, 5);
        assert_eq!(rec.descriptors.len(), 4);
    }

    #[test]
    fn cached_graphics_state_preserves_invisible_draw_and_push_layout_errors() {
        let (table, mut resources, mut rec, layout) = graphics_fixture();
        graphics_bindings(&table, &mut resources, &mut rec).unwrap();
        rec.scissor.as_mut().unwrap().extent.width = 0;
        assert!(!graphics_bindings(&table, &mut resources, &mut rec).unwrap());
        rec.scissor.as_mut().unwrap().extent.width = 8;
        assert!(graphics_bindings(&table, &mut resources, &mut rec).unwrap());
        assert!(
            matches!(rec.ops.last().unwrap(), ir::OwnedCommand::SetScissor(Some(rect)) if rect.width() == 8)
        );
        let incompatible = ir::PipelineLayoutDesc::new(vec![])
            .unwrap()
            .with_push_constant_ranges(vec![
                ir::PushConstantRange::new(ir::ShaderStages::VERTEX, 0, 8).unwrap(),
            ])
            .unwrap();
        rec.push_constants
            .update(&incompatible, ir::ShaderStages::VERTEX, 0, &[1; 8])
            .unwrap();
        rec.push_constants_revision += 1;
        assert_ne!(incompatible, layout);
        assert_eq!(
            graphics_bindings(&table, &mut resources, &mut rec),
            Err(vk::Result::ERROR_INITIALIZATION_FAILED)
        );
        resources.pipelines.remove(&vk::Pipeline::from_raw(1));
        assert_eq!(
            graphics_bindings(&table, &mut resources, &mut rec),
            Err(vk::Result::ERROR_INITIALIZATION_FAILED)
        );
    }

    #[test]
    fn deferred_insertions_preserve_order_with_linear_position_checks() {
        const COMMANDS: usize = 10_000;
        let mut entries: Vec<_> = (0..1000).map(|i| (i * 10, i)).collect();
        entries.extend([(COMMANDS, 1000), (COMMANDS, 1001)]);
        let mut remaining = entries.as_slice();
        let checks = std::cell::Cell::new(0usize);
        let mut observed = Vec::new();
        for position in 0..=COMMANDS {
            for entry in take_positioned(&mut remaining, position, |entry| {
                checks.set(checks.get() + 1);
                entry.0
            }) {
                observed.push(*entry);
            }
        }
        assert_eq!(observed, entries);
        assert!(remaining.is_empty());
        assert!(checks.get() <= entries.len() + COMMANDS + 1);
        assert!(take_positioned(&mut remaining, COMMANDS + 1, |entry| entry.0).is_empty());
    }

    #[test]
    fn recording_accepts_application_programs_and_rejects_capacity_overflow() {
        let mut recording = Recording::new(1);
        recording.state = RecordingState::Recording;
        for _ in 0..ir::MAX_COMMANDS {
            recording
                .push(RecordedCommand::Dispatch { x: 1, y: 1, z: 1 })
                .unwrap();
        }
        assert!(recording.error.is_none());
        assert_eq!(recording.commands.len(), ir::MAX_COMMANDS);
        assert_eq!(
            recording.push(RecordedCommand::Dispatch { x: 1, y: 1, z: 1 }),
            Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY)
        );
        assert_eq!(recording.commands.len(), ir::MAX_COMMANDS);
    }

    #[test]
    fn command_recording_stays_on_the_calling_thread() {
        let (sender, _intentionally_unserviced_receiver) = mpsc::channel();
        let (completion_sender, _intentionally_unserviced_completion_receiver) = mpsc::channel();
        let d = Arc::new(Driver {
            sender,
            thread: Mutex::new(None),
            completion_sender,
            completion_thread: Mutex::new(None),
            queue: AtomicU64::new(0),
            queue_sender: mpsc::channel().0,
            queue_thread: Mutex::new(None),
            timeline_enabled: true,
            synchronization2_enabled: true,
            dynamic_rendering_enabled: true,
            host_query_reset_enabled: false,
            timestamps: false,
            query_support: 0,
            occlusion_precise_enabled: false,
            pipeline_statistics_enabled: false,
            multi_draw_indirect_enabled: false,
            indirect_first_instance_enabled: false,
            indirect_count_enabled: false,
            queries: Default::default(),
            timelines: Default::default(),
            events: Default::default(),
            fences: Default::default(),
            semaphores: Default::default(),
            recordings: Default::default(),
            in_flight: Default::default(),
            pending: Default::default(),
            lost: Arc::new(AtomicBool::new(false)),
        });
        let command = vk::CommandBuffer::from_raw(add_handle(Kind::Command, &d));
        let mut recording = Recording::new(1);
        recording.state = RecordingState::Recording;
        d.recordings
            .lock()
            .unwrap()
            .commands
            .insert(command.as_raw(), Arc::new(Mutex::new(recording)));

        record(command, RecordedCommand::Dispatch { x: 1, y: 2, z: 3 });
        let layout = vk::PipelineLayout::from_raw(next_id());
        d.recordings.lock().unwrap().pipeline_layouts.insert(
            layout,
            ir::PipelineLayoutDesc::new(vec![])
                .unwrap()
                .with_push_constant_ranges(vec![
                    ir::PushConstantRange::new(ir::ShaderStages::COMPUTE, 0, 4).unwrap(),
                ])
                .unwrap(),
        );
        let mut values = [7u8; 4];
        unsafe {
            cmd_push_constants(
                command,
                layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                4,
                values.as_ptr().cast(),
            );
        }
        values.fill(9);

        let recording = d
            .recordings
            .lock()
            .unwrap()
            .commands
            .get(&command.as_raw())
            .cloned()
            .unwrap();
        assert!(matches!(
            recording.lock().unwrap().commands.as_slice(),
            [RecordedCommand::Dispatch { x: 1, y: 2, z: 3 }, RecordedCommand::PushConstants { data, offset: 0, .. }] if data == &[7; 4]
        ));
        remove_handle(command.as_raw());
    }

    #[test]
    fn fence_queries_do_not_queue_behind_a_busy_device_worker() {
        let (sender, _intentionally_unserviced_receiver) = mpsc::channel();
        let (completion_sender, _intentionally_unserviced_completion_receiver) = mpsc::channel();
        let d = Arc::new(Driver {
            sender,
            thread: Mutex::new(None),
            completion_sender,
            completion_thread: Mutex::new(None),
            queue: AtomicU64::new(0),
            queue_sender: mpsc::channel().0,
            queue_thread: Mutex::new(None),
            timeline_enabled: true,
            synchronization2_enabled: true,
            dynamic_rendering_enabled: true,
            host_query_reset_enabled: false,
            timestamps: false,
            query_support: 0,
            occlusion_precise_enabled: false,
            pipeline_statistics_enabled: false,
            multi_draw_indirect_enabled: false,
            indirect_first_instance_enabled: false,
            indirect_count_enabled: false,
            queries: Default::default(),
            timelines: Default::default(),
            events: Default::default(),
            fences: Default::default(),
            semaphores: Default::default(),
            recordings: Default::default(),
            in_flight: Default::default(),
            pending: Default::default(),
            lost: Arc::new(AtomicBool::new(false)),
        });
        let device = vk::Device::from_raw(add_handle(Kind::Device, &d));
        let fence = vk::Fence::from_raw(next_id());
        let state = Arc::new(AtomicU8::new(2));
        d.fences
            .lock()
            .unwrap()
            .insert(fence.as_raw(), Arc::clone(&state));
        let start = std::time::Instant::now();
        unsafe {
            assert_eq!(get_fence_status(device, fence), vk::Result::NOT_READY);
            assert_eq!(
                wait_for_fences(device, 1, &fence, vk::TRUE, 0),
                vk::Result::TIMEOUT
            );
            assert_eq!(
                wait_for_fences(device, 1, &fence, vk::TRUE, 1_000_000),
                vk::Result::TIMEOUT
            );
        }
        assert!(start.elapsed() < std::time::Duration::from_millis(100));
        state.store(1, Ordering::Release);
        unsafe { assert_eq!(get_fence_status(device, fence), vk::Result::SUCCESS) }
        d.lost.store(true, Ordering::Release);
        unsafe {
            assert_eq!(
                get_fence_status(device, fence),
                vk::Result::ERROR_DEVICE_LOST
            )
        }
        remove_handle(device.as_raw());
    }

    #[test]
    fn binary_semaphore_can_bridge_acquire_submit_and_present() {
        let (sender, _intentionally_unserviced_receiver) = mpsc::channel();
        let (completion_sender, _intentionally_unserviced_completion_receiver) = mpsc::channel();
        let d = Arc::new(Driver {
            sender,
            thread: Mutex::new(None),
            completion_sender,
            completion_thread: Mutex::new(None),
            queue: AtomicU64::new(0),
            queue_sender: mpsc::channel().0,
            queue_thread: Mutex::new(None),
            timeline_enabled: true,
            synchronization2_enabled: true,
            dynamic_rendering_enabled: true,
            host_query_reset_enabled: false,
            timestamps: false,
            query_support: 0,
            occlusion_precise_enabled: false,
            pipeline_statistics_enabled: false,
            multi_draw_indirect_enabled: false,
            indirect_first_instance_enabled: false,
            indirect_count_enabled: false,
            queries: Default::default(),
            timelines: Default::default(),
            events: Default::default(),
            fences: Default::default(),
            semaphores: Default::default(),
            recordings: Default::default(),
            in_flight: Default::default(),
            pending: Default::default(),
            lost: Arc::new(AtomicBool::new(false)),
        });
        let device = vk::Device::from_raw(add_handle(Kind::Device, &d));
        let queue = vk::Queue::from_raw(add_handle(Kind::Queue, &d));
        let mut semaphore = vk::Semaphore::null();
        unsafe {
            assert_eq!(
                create_semaphore(
                    device,
                    &vk::SemaphoreCreateInfo::default(),
                    std::ptr::null(),
                    &mut semaphore,
                ),
                vk::Result::SUCCESS
            );
        }
        let state = d
            .semaphores
            .lock()
            .unwrap()
            .get(&semaphore.as_raw())
            .cloned()
            .unwrap();
        assert_eq!(state.load(Ordering::Acquire), 0);
        assert_eq!(
            signal_acquire_sync(device, semaphore, vk::Fence::null()),
            Ok(())
        );
        assert_eq!(state.load(Ordering::Acquire), 1);
        assert_eq!(wait_queue_semaphores(queue, &[semaphore]), Ok(()));
        assert_eq!(state.load(Ordering::Acquire), 0);
        unsafe { destroy_semaphore(device, semaphore, std::ptr::null()) };
        remove_handle(queue.as_raw());
        remove_handle(device.as_raw());
    }

    #[test]
    fn loader_device_chain_rejects_unknown_records() {
        let mut base = vk::BaseInStructure {
            s_type: vk::StructureType::LOADER_DEVICE_CREATE_INFO,
            ..Default::default()
        };
        unsafe {
            assert!(device_chain_supported(
                (&base as *const vk::BaseInStructure<'_>).cast(),
                &vk::PhysicalDeviceFeatures::default(),
            ));
        }
        base.s_type = vk::StructureType::DEVICE_GROUP_DEVICE_CREATE_INFO;
        unsafe {
            assert!(!device_chain_supported(
                (&base as *const vk::BaseInStructure<'_>).cast(),
                &vk::PhysicalDeviceFeatures::default(),
            ));
        }
    }

    #[test]
    fn device_chain_accepts_disabled_features_without_enabling_them() {
        fn supported(
            indexing: &mut vk::PhysicalDeviceDescriptorIndexingFeatures<'_>,
            draw: &mut vk::PhysicalDeviceShaderDrawParametersFeatures<'_>,
        ) -> bool {
            indexing.p_next =
                (draw as *mut vk::PhysicalDeviceShaderDrawParametersFeatures<'_>).cast();
            let loader = vk::BaseInStructure {
                s_type: vk::StructureType::LOADER_DEVICE_CREATE_INFO,
                p_next: (indexing as *const vk::PhysicalDeviceDescriptorIndexingFeatures<'_>)
                    .cast(),
                ..Default::default()
            };
            unsafe {
                device_chain_supported(
                    (&loader as *const vk::BaseInStructure<'_>).cast(),
                    &vk::PhysicalDeviceFeatures::default(),
                )
            }
        }
        let mut draw = vk::PhysicalDeviceShaderDrawParametersFeatures::default();
        let mut indexing = vk::PhysicalDeviceDescriptorIndexingFeatures::default();
        assert!(supported(&mut indexing, &mut draw));
        indexing.descriptor_binding_partially_bound = vk::TRUE;
        assert!(!supported(&mut indexing, &mut draw));
        indexing.descriptor_binding_partially_bound = vk::FALSE;
        draw.shader_draw_parameters = vk::TRUE;
        assert!(supported(&mut indexing, &mut draw));
        draw.shader_draw_parameters = 2;
        assert!(!supported(&mut indexing, &mut draw));
    }

    #[test]
    fn device_chain_rejects_cycles() {
        let mut loader = vk::BaseInStructure {
            s_type: vk::StructureType::LOADER_DEVICE_CREATE_INFO,
            ..Default::default()
        };
        loader.p_next = &loader;
        unsafe {
            assert!(!device_chain_supported(
                (&loader as *const vk::BaseInStructure<'_>).cast(),
                &vk::PhysicalDeviceFeatures::default(),
            ));
        }
    }

    #[test]
    fn independent_blend_core_features_reject_unadvertised_and_invalid_requests() {
        let supported = vk::PhysicalDeviceFeatures::default().independent_blend(true);
        let disabled = vk::PhysicalDeviceFeatures::default();
        assert!(core_device_features_supported(&disabled, &supported));
        assert!(core_device_features_supported(&supported, &supported));
        assert!(!core_device_features_supported(&supported, &disabled));

        // Exercise every core feature field, so adding independentBlend cannot
        // accidentally permit an unsupported feature at either end of the ABI.
        let independent_offset =
            std::mem::offset_of!(vk::PhysicalDeviceFeatures, independent_blend) / 4;
        for index in 0..std::mem::size_of::<vk::PhysicalDeviceFeatures>() / 4 {
            let mut requested = vk::PhysicalDeviceFeatures::default();
            unsafe {
                let fields = (&mut requested as *mut vk::PhysicalDeviceFeatures).cast::<u32>();
                *fields.add(index) = vk::TRUE;
                assert_eq!(
                    core_device_features_supported(&requested, &supported),
                    index == independent_offset,
                    "core feature field {index}",
                );
                *fields.add(index) = 2;
                assert!(!core_device_features_supported(&requested, &supported));
            }
        }
    }

    #[test]
    fn independent_blend_features2_matches_legacy_feature_validation() {
        let supported = vk::PhysicalDeviceFeatures::default().independent_blend(true);
        let mut requested = vk::PhysicalDeviceFeatures2::default().features(supported);
        let mut loader = vk::BaseInStructure {
            s_type: vk::StructureType::LOADER_DEVICE_CREATE_INFO,
            ..Default::default()
        };
        requested.p_next = (&mut loader as *mut vk::BaseInStructure<'_>).cast();
        let chain = (&requested as *const vk::PhysicalDeviceFeatures2<'_>).cast();
        unsafe {
            assert!(device_chain_supported(chain, &supported));
            assert!(!device_chain_supported(
                chain,
                &vk::PhysicalDeviceFeatures::default(),
            ));
        }
        requested.features.logic_op = vk::TRUE;
        unsafe {
            assert!(!device_chain_supported(
                (&requested as *const vk::PhysicalDeviceFeatures2<'_>).cast(),
                &supported,
            ));
        }
        requested.features.logic_op = vk::FALSE;
        requested.features.independent_blend = 2;
        unsafe {
            assert!(!device_chain_supported(
                (&requested as *const vk::PhysicalDeviceFeatures2<'_>).cast(),
                &supported,
            ));
        }
    }
}

#[cfg(test)]
mod fence_notification_tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    fn fixture() -> (Arc<Driver>, vk::Device, [vk::Fence; 2], [Arc<AtomicU8>; 2]) {
        // Unserviced/closed channels prove that waiting never enters Runtime.
        let d = Arc::new(Driver {
            sender: mpsc::channel().0,
            thread: Mutex::new(None),
            completion_sender: mpsc::channel().0,
            completion_thread: Mutex::new(None),
            queue: AtomicU64::new(0),
            queue_sender: mpsc::channel().0,
            queue_thread: Mutex::new(None),
            timeline_enabled: true,
            synchronization2_enabled: true,
            dynamic_rendering_enabled: true,
            host_query_reset_enabled: false,
            timestamps: false,
            query_support: 0,
            occlusion_precise_enabled: false,
            pipeline_statistics_enabled: false,
            multi_draw_indirect_enabled: false,
            indirect_first_instance_enabled: false,
            indirect_count_enabled: false,
            queries: Default::default(),
            timelines: Default::default(),
            events: Default::default(),
            fences: Default::default(),
            semaphores: Default::default(),
            recordings: Default::default(),
            in_flight: Default::default(),
            pending: Default::default(),
            lost: Arc::new(AtomicBool::new(false)),
        });
        let states = [Arc::new(AtomicU8::new(0)), Arc::new(AtomicU8::new(0))];
        let fences = states.each_ref().map(|state| {
            let fence = vk::Fence::from_raw(next_id());
            d.fences
                .lock()
                .unwrap()
                .insert(fence.as_raw(), Arc::clone(state));
            fence
        });
        let device = vk::Device::from_raw(add_handle(Kind::Device, &d));
        (d, device, fences, states)
    }

    fn waiter(
        device: vk::Device,
        fences: Vec<vk::Fence>,
        all: bool,
    ) -> std::thread::JoinHandle<vk::Result> {
        std::thread::spawn(move || unsafe {
            wait_for_fences(
                device,
                fences.len() as u32,
                fences.as_ptr(),
                u32::from(all),
                5_000_000_000,
            )
        })
    }

    #[test]
    fn selected_fences_preserve_all_any_zero_timeout_and_presignaled_semantics() {
        let (d, device, fences, states) = fixture();
        unsafe {
            assert_eq!(
                wait_for_fences(device, 2, fences.as_ptr(), vk::FALSE, 0),
                vk::Result::TIMEOUT
            );
            states[0].store(1, Ordering::Release);
            assert_eq!(
                wait_for_fences(device, 2, fences.as_ptr(), vk::FALSE, 0),
                vk::Result::SUCCESS
            );
            assert_eq!(
                wait_for_fences(device, 2, fences.as_ptr(), vk::TRUE, 0),
                vk::Result::TIMEOUT
            );
            states[1].store(1, Ordering::Release);
            assert_eq!(
                wait_for_fences(device, 2, fences.as_ptr(), vk::TRUE, 0),
                vk::Result::SUCCESS
            );
            d.mark_lost();
            assert_eq!(
                wait_for_fences(device, 2, fences.as_ptr(), vk::TRUE, 0),
                vk::Result::ERROR_DEVICE_LOST
            );
        }
        remove_handle(device.as_raw());
    }

    #[test]
    fn selected_fence_wakes_before_unrelated_work_retires() {
        let (d, device, fences, states) = fixture();
        d.in_flight.begin();
        d.in_flight.begin();
        d.pending.begin();
        let thread = waiter(device, vec![fences[0]], true);
        finish_submissions(
            &[],
            Some(&states[0]),
            &[],
            &[],
            &d.in_flight,
            &d.pending,
            &d.lost,
        );
        assert_eq!(thread.join().unwrap(), vk::Result::SUCCESS);
        assert!(!d.in_flight.is_empty());
        assert_eq!(states[1].load(Ordering::Acquire), 0);
        d.in_flight.finish();
        remove_handle(device.as_raw());
    }

    #[test]
    fn notification_before_and_after_wait_cannot_be_lost() {
        for notify_before_wait in [true, false] {
            let state = Arc::new(AtomicBool::new(false));
            let signal = Arc::new(InFlight::default());
            let (checked, first_check) = mpsc::channel();
            if notify_before_wait {
                state.store(true, Ordering::Release);
                signal.notify();
            }
            let worker_state = Arc::clone(&state);
            let worker_signal = Arc::clone(&signal);
            let thread = std::thread::spawn(move || {
                worker_signal.wait_until(u64::MAX, || {
                    let ready = worker_state.load(Ordering::Acquire);
                    checked.send(()).unwrap(); // Executed under the actual wait mutex.
                    Ok(ready)
                })
            });
            first_check.recv().unwrap();
            if !notify_before_wait {
                // notify() cannot pass the waiter's mutex until it atomically
                // enters Condvar::wait. This exercises the check-to-sleep race.
                state.store(true, Ordering::Release);
                signal.notify();
            }
            assert_eq!(thread.join().unwrap(), Ok(()));
        }
    }

    #[test]
    fn spurious_notifications_recheck_predicate_without_restarting_timeout() {
        let signal = Arc::new(InFlight::default());
        let checks = Arc::new(AtomicUsize::new(0));
        let worker_signal = Arc::clone(&signal);
        let worker_checks = Arc::clone(&checks);
        let (tx, rx) = mpsc::channel();
        let waiter = std::thread::spawn(move || {
            tx.send(worker_signal.wait_until(50_000_000, || {
                worker_checks.fetch_add(1, Ordering::Relaxed);
                Ok(false)
            }))
            .unwrap();
        });
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let notifier = std::thread::spawn(move || {
            while !worker_stop.load(Ordering::Acquire) {
                signal.notify();
                std::thread::yield_now();
            }
        });
        let result = rx.recv_timeout(Duration::from_secs(2));
        stop.store(true, Ordering::Release);
        notifier.join().unwrap();
        waiter.join().unwrap();
        assert_eq!(result.unwrap(), Err(vk::Result::TIMEOUT));
        assert!(checks.load(Ordering::Relaxed) > 1);
    }

    #[test]
    fn acquire_sync_wakes_selected_fence_waiter() {
        let (_d, device, fences, _states) = fixture();
        let thread = waiter(device, fences.to_vec(), false);
        signal_acquire_sync(device, vk::Semaphore::null(), fences[1]).unwrap();
        assert_eq!(thread.join().unwrap(), vk::Result::SUCCESS);
        remove_handle(device.as_raw());
    }

    #[test]
    fn queue_failure_worker_channel_loss_and_destroy_wake_fence_waiters() {
        for path in 0..4 {
            let (d, device, fences, _states) = fixture();
            let thread = waiter(device, fences.to_vec(), true);
            match path {
                0 => d.mark_lost(),
                1 => {
                    let queue = vk::Queue::from_raw(add_handle(Kind::Queue, &d));
                    assert_eq!(
                        unsafe { queue::submit(queue, 0, core::ptr::null(), vk::Fence::null()) },
                        Err(vk::Result::ERROR_DEVICE_LOST)
                    );
                    remove_handle(queue.as_raw());
                }
                2 => {
                    let result: VkResult<()> = call(&d, |_| Ok(()));
                    assert_eq!(result, Err(vk::Result::ERROR_DEVICE_LOST));
                }
                3 => unsafe { destroy_device(device, core::ptr::null()) },
                _ => unreachable!(),
            }
            assert_eq!(thread.join().unwrap(), vk::Result::ERROR_DEVICE_LOST);
            remove_handle(device.as_raw());
        }
    }

    #[test]
    fn fence_notifications_are_scoped_to_their_device() {
        let (a, a_device, a_fences, _a_states) = fixture();
        let (b, b_device, _b_fences, b_states) = fixture();
        let thread = waiter(a_device, a_fences.to_vec(), false);
        b.in_flight.begin();
        b.pending.begin();
        finish_submissions(
            &[],
            Some(&b_states[0]),
            &[],
            &[],
            &b.in_flight,
            &b.pending,
            &b.lost,
        );
        unsafe {
            assert_eq!(
                wait_for_fences(a_device, 2, a_fences.as_ptr(), vk::FALSE, 0),
                vk::Result::TIMEOUT
            );
        }
        a.mark_lost();
        assert_eq!(thread.join().unwrap(), vk::Result::ERROR_DEVICE_LOST);
        remove_handle(a_device.as_raw());
        remove_handle(b_device.as_raw());
    }
}

#[cfg(test)]
mod owned_descriptor_tests {
    use super::*;
    use std::time::Duration;

    struct Fixture {
        driver: Arc<Driver>,
        device: vk::Device,
        requests: mpsc::Receiver<Request>,
    }
    impl Fixture {
        fn new() -> Self {
            let (sender, requests) = mpsc::channel();
            let driver = Arc::new(Driver {
                sender,
                thread: Mutex::new(None),
                completion_sender: mpsc::channel().0,
                completion_thread: Mutex::new(None),
                queue: AtomicU64::new(0),
                queue_sender: mpsc::channel().0,
                queue_thread: Mutex::new(None),
                timeline_enabled: false,
                synchronization2_enabled: false,
                dynamic_rendering_enabled: false,
                host_query_reset_enabled: false,
                timestamps: false,
                query_support: 0,
                occlusion_precise_enabled: false,
                pipeline_statistics_enabled: false,
                multi_draw_indirect_enabled: false,
                indirect_first_instance_enabled: false,
                indirect_count_enabled: false,
                queries: Default::default(),
                timelines: Default::default(),
                events: Default::default(),
                fences: Default::default(),
                semaphores: Default::default(),
                recordings: Default::default(),
                in_flight: Default::default(),
                pending: Default::default(),
                lost: Arc::new(AtomicBool::new(false)),
            });
            let device = vk::Device::from_raw(add_handle(Kind::Device, &driver));
            Self {
                driver,
                device,
                requests,
            }
        }
        fn recording(&self, sets: Vec<vk::DescriptorSet>) -> RecordingCell {
            let mut recording = Recording::new(0);
            recording.state = RecordingState::Recording;
            if !sets.is_empty() {
                recording.push(binding(sets)).unwrap();
            }
            let recording = Arc::new(Mutex::new(recording));
            self.driver
                .recordings
                .lock()
                .unwrap()
                .commands
                .insert(next_id(), recording.clone());
            recording
        }
        #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
        fn runtime(&self) -> Runtime {
            let instance = sgfx::driver::Instance::new().unwrap();
            Runtime::new(
                instance.adapters()[0].clone(),
                self.driver.recordings.clone(),
                self.driver.in_flight.clone(),
                self.driver.lost.clone(),
            )
            .unwrap()
        }
        fn service(&self, runtime: &mut Runtime) {
            match self.requests.recv_timeout(Duration::from_secs(5)).unwrap() {
                Request::Run(job) => job(runtime),
                Request::Stop => panic!("unexpected stop"),
            }
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            remove_handle(self.device.as_raw());
        }
    }
    fn binding(sets: Vec<vk::DescriptorSet>) -> RecordedCommand {
        RecordedCommand::BindDescriptorSets {
            point: vk::PipelineBindPoint::GRAPHICS,
            layout: vk::PipelineLayout::null(),
            first: 0,
            sets,
            dynamic_offsets: Vec::new(),
            pushed: None,
        }
    }

    #[test]
    fn owned_update_returns_without_worker_and_invalidates_only_prior_bindings() {
        let fixture = Fixture::new();
        let set = vk::DescriptorSet::from_raw(1);
        let old = fixture.recording(vec![set]);
        old.lock().unwrap().state = RecordingState::Executable;
        let recording = fixture.recording(vec![set]);
        let new = fixture.recording(Vec::new());
        let unrelated = fixture.recording(vec![vk::DescriptorSet::from_raw(2)]);
        let device = fixture.device;
        let (done, received) = mpsc::channel();
        let caller = std::thread::spawn(move || {
            // No worker is running. A synchronous implementation cannot finish.
            let image = [vk::DescriptorImageInfo::default().sampler(vk::Sampler::from_raw(7))];
            let writes = [vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(&image)];
            unsafe {
                crate::resources::update_descriptor_sets(
                    device,
                    1,
                    writes.as_ptr(),
                    0,
                    std::ptr::null(),
                );
            }
            done.send(()).unwrap();
        });
        received.recv_timeout(Duration::from_secs(5)).unwrap();
        caller.join().unwrap();
        assert!(old.lock().unwrap().state == RecordingState::Invalid);
        assert!(old.lock().unwrap().error.is_some());
        assert!(recording.lock().unwrap().error.is_some());
        new.lock().unwrap().push(binding(vec![set])).unwrap();
        assert!(new.lock().unwrap().error.is_none());
        assert!(unrelated.lock().unwrap().error.is_none());
        assert!(matches!(fixture.requests.try_recv(), Ok(Request::Run(_))));
        assert!(fixture.requests.try_recv().is_err());
    }

    #[test]
    fn closed_owned_update_channel_and_malformed_void_batch_mark_device_lost() {
        let mut fixture = Fixture::new();
        drop(std::mem::replace(&mut fixture.requests, mpsc::channel().1));
        assert_eq!(
            update_descriptors_owned(fixture.device, &[], |_| Ok(())),
            Err(vk::Result::ERROR_DEVICE_LOST)
        );
        assert!(fixture.driver.lost.load(Ordering::Acquire));
        let fixture = Fixture::new();
        unsafe {
            crate::resources::update_descriptor_sets(
                fixture.device,
                1,
                std::ptr::null(),
                0,
                std::ptr::null(),
            );
        }
        assert!(fixture.driver.lost.load(Ordering::Acquire));
        assert!(fixture.requests.try_recv().is_err());
    }
    #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
    fn descriptors(
        runtime: &mut Runtime,
    ) -> (vk::DescriptorPool, [vk::DescriptorSet; 2], vk::Buffer) {
        use crate::resources::{Buffer, DescriptorPool, DescriptorSet};
        let pool = vk::DescriptorPool::from_raw(1);
        let sets = [
            vk::DescriptorSet::from_raw(2),
            vk::DescriptorSet::from_raw(3),
        ];
        let buffer = vk::Buffer::from_raw(4);
        let id = runtime
            .table
            .define_buffer(ir::BufferDesc::new(1024, ir::BufferUsage::UNIFORM).unwrap())
            .unwrap()
            .id();
        runtime.resources.buffers.insert(
            buffer,
            Buffer {
                id,
                size: 1024,
                usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                bound: Some((vk::DeviceMemory::from_raw(5), 0)),
            },
        );
        runtime.resources.descriptor_pools.insert(
            pool,
            DescriptorPool {
                max_sets: 2,
                free_sets: true,
                capacity: HashMap::from([(vk::DescriptorType::UNIFORM_BUFFER, 2)]),
                remaining: HashMap::from([(vk::DescriptorType::UNIFORM_BUFFER, 0)]),
            },
        );
        for set in sets {
            runtime.resources.descriptor_sets.insert(
                set,
                DescriptorSet::new(
                    HashMap::new(),
                    pool,
                    ir::BindGroupLayoutDesc::new(vec![ir::BindGroupLayoutEntry::new(
                        0,
                        ir::ShaderStages::VERTEX,
                        ir::BindingType::UniformBuffer,
                    )])
                    .unwrap(),
                    BTreeMap::from([(0, vk::DescriptorType::UNIFORM_BUFFER)]),
                    BTreeMap::from([(0, vk::ShaderStageFlags::VERTEX)]),
                    HashMap::new(),
                )
                .unwrap(),
            );
        }
        (pool, sets, buffer)
    }

    #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
    #[test]
    #[ignore = "requires a native GPU to construct the actual Runtime"]
    fn owned_update_fifo_preserves_copied_input_later_resolution_free_and_pool_destroy() {
        let fixture = Fixture::new();
        let mut runtime = fixture.runtime();
        let (pool, sets, buffer) = descriptors(&mut runtime);
        let after = fixture.recording(Vec::new());
        let mut info = [vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(256)
            .range(16)];
        {
            let writes = [vk::WriteDescriptorSet::default()
                .dst_set(sets[0])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&info)];
            unsafe {
                crate::resources::update_descriptor_sets(
                    fixture.device,
                    1,
                    writes.as_ptr(),
                    0,
                    std::ptr::null(),
                );
            }
        }
        // Application input need only live through vkUpdateDescriptorSets.
        info[0].offset = 512;
        assert_eq!(info[0].offset, 512);
        let copies = [vk::CopyDescriptorSet::default()
            .src_set(sets[0])
            .src_binding(0)
            .dst_set(sets[1])
            .dst_binding(0)
            .descriptor_count(1)];
        unsafe {
            crate::resources::update_descriptor_sets(
                fixture.device,
                0,
                std::ptr::null(),
                1,
                copies.as_ptr(),
            );
        }
        after.lock().unwrap().push(binding(vec![sets[1]])).unwrap();
        let device = fixture.device;
        let resolved = std::thread::spawn(move || {
            with_device(device, move |r| {
                // Descriptor resolution uses the same worker/FIFO as queue submit.
                let group = r.resources.descriptor_group(&r.table, sets[1], &[])?;
                let group = r
                    .table
                    .bind_group(r.table.bind_group_ref(group).unwrap())
                    .unwrap();
                assert!(matches!(
                    group.entries()[0].resource(),
                    ir::BindingResource::Buffer {
                        offset: 256,
                        size: 16,
                        ..
                    }
                ));
                Ok(())
            })
        });
        fixture.service(&mut runtime); // write
        fixture.service(&mut runtime); // copy, seeing preceding write
        fixture.service(&mut runtime); // later descriptor resolution
        resolved.join().unwrap().unwrap();
        assert!(after.lock().unwrap().error.is_none());

        let writes = [vk::WriteDescriptorSet::default()
            .dst_set(sets[0])
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&info)];
        unsafe {
            crate::resources::update_descriptor_sets(
                fixture.device,
                1,
                writes.as_ptr(),
                0,
                std::ptr::null(),
            );
        }
        let free: vk::PFN_vkFreeDescriptorSets = unsafe {
            std::mem::transmute(crate::resources::lookup(c"vkFreeDescriptorSets").unwrap())
        };
        let freed = std::thread::spawn(move || unsafe { free(device, pool, 1, sets.as_ptr()) });
        fixture.service(&mut runtime); // update before free
        fixture.service(&mut runtime);
        assert_eq!(freed.join().unwrap(), vk::Result::SUCCESS);
        assert!(!runtime.resources.descriptor_sets.contains_key(&sets[0]));
        assert!(runtime.resources.descriptor_sets.contains_key(&sets[1]));

        let writes = [vk::WriteDescriptorSet::default()
            .dst_set(sets[1])
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&info)];
        unsafe {
            crate::resources::update_descriptor_sets(
                fixture.device,
                1,
                writes.as_ptr(),
                0,
                std::ptr::null(),
            );
        }
        let destroy: vk::PFN_vkDestroyDescriptorPool = unsafe {
            std::mem::transmute(crate::resources::lookup(c"vkDestroyDescriptorPool").unwrap())
        };
        let destroyed =
            std::thread::spawn(move || unsafe { destroy(device, pool, std::ptr::null()) });
        fixture.service(&mut runtime); // update before destroy
        fixture.service(&mut runtime);
        destroyed.join().unwrap();
        assert!(runtime.resources.descriptor_sets.is_empty());
        assert!(!runtime.resources.descriptor_pools.contains_key(&pool));
        assert!(!runtime.lost);
    }

    #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
    #[test]
    #[ignore = "requires a native GPU to construct the actual Runtime"]
    fn owned_worker_errors_and_panics_publish_loss_and_reject_later_jobs() {
        for panic in [false, true] {
            let fixture = Fixture::new();
            let mut runtime = fixture.runtime();
            let _ = descriptors(&mut runtime);
            // A normal synchronous validation error still does not lose device.
            assert_eq!(
                run_device_operation(
                    &mut runtime,
                    |_| Err::<(), _>(vk::Result::ERROR_UNKNOWN),
                    false
                ),
                Err(vk::Result::ERROR_UNKNOWN)
            );
            assert!(!runtime.lost);
            update_descriptors_owned(fixture.device, &[], move |_| {
                if panic {
                    panic!("injected owned metadata panic");
                }
                Err(vk::Result::ERROR_UNKNOWN)
            })
            .unwrap();
            fixture.service(&mut runtime);
            assert!(runtime.lost && fixture.driver.lost.load(Ordering::Acquire));
            update_descriptors_owned(fixture.device, &[], |_| {
                panic!("lost device must reject job")
            })
            .unwrap();
            fixture.service(&mut runtime);
            assert_eq!(
                run_device_operation(
                    &mut runtime,
                    |_| panic!("lost device must reject sync call"),
                    false
                ),
                Err::<(), _>(vk::Result::ERROR_DEVICE_LOST)
            );
        }
    }
}
