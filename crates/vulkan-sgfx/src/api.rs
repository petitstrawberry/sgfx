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

    pub(crate) fn is_empty(&self) -> bool {
        *self
            .count
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            == 0
    }

    fn wait(&self) {
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
    fences: Fences,
    semaphores: Semaphores,
    recordings: Recordings,
    in_flight: Arc<InFlight>,
    lost: Arc<AtomicBool>,
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
fn call<T: Send + 'static>(
    driver: &Driver,
    op: impl FnOnce(&mut Runtime) -> VkResult<T> + Send + 'static,
) -> VkResult<T> {
    let (tx, rx) = mpsc::sync_channel(1);
    driver
        .sender
        .send(Request::Run(Box::new(move |runtime| {
            let result = if runtime.device_lost.load(Ordering::Acquire) {
                runtime.lost = true;
                Err(vk::Result::ERROR_DEVICE_LOST)
            } else {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| op(runtime)))
                    .unwrap_or_else(|_| {
                        runtime.lost = true;
                        Err(vk::Result::ERROR_DEVICE_LOST)
                    })
            };
            runtime.reclaim_idle_resources();
            if runtime.lost {
                runtime.device_lost.store(true, Ordering::Release);
            }
            let _ = tx.send(result);
        })))
        .map_err(|_| vk::Result::ERROR_DEVICE_LOST)?;
    rx.recv().map_err(|_| vk::Result::ERROR_DEVICE_LOST)?
}
pub(crate) fn with_device<T: Send + 'static>(
    device: vk::Device,
    op: impl FnOnce(&mut Runtime) -> VkResult<T> + Send + 'static,
) -> VkResult<T> {
    call(driver(device.as_raw(), Kind::Device)?.as_ref(), op)
}

#[cfg(any(target_os = "macos", all(target_os = "linux", feature = "scarlet-wsi")))]
pub(crate) fn with_queue<T: Send + 'static>(
    queue: vk::Queue,
    op: impl FnOnce(&mut Runtime) -> VkResult<T> + Send + 'static,
) -> VkResult<T> {
    call(driver(queue.as_raw(), Kind::Queue)?.as_ref(), op)
}
fn status(value: VkResult<()>) -> vk::Result {
    value.map_or_else(|e| e, |_| vk::Result::SUCCESS)
}
unsafe fn slice<'a, T>(ptr: *const T, count: u32) -> VkResult<&'a [T]> {
    if count > 4096 || (count != 0 && ptr.is_null()) {
        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
    }
    Ok(if count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ptr, count as usize)
    })
}

unsafe fn device_chain_supported(mut next: *const std::ffi::c_void) -> bool {
    // The Khronos loader inserts private device-link records before calling an
    // ICD. They are transport metadata, not enabled application features.
    for _ in 0..32 {
        if next.is_null() {
            return true;
        }
        let header = &*next.cast::<vk::BaseInStructure<'_>>();
        let supported = match header.s_type {
            vk::StructureType::LOADER_DEVICE_CREATE_INFO => true,
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
            vk::StructureType::PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES => {
                let features = &*next.cast::<vk::PhysicalDeviceShaderDrawParametersFeatures<'_>>();
                features.shader_draw_parameters == vk::FALSE
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
            #[cfg(any(target_os = "macos", all(target_os = "linux", feature = "scarlet-wsi")))]
            {
                name == vk::KHR_SWAPCHAIN_NAME
            }
            #[cfg(not(any(
                target_os = "macos",
                all(target_os = "linux", feature = "scarlet-wsi")
            )))]
            {
                #[cfg(target_os = "scarlet")]
                {
                    name == crate::scarlet_image::DEVICE_EXTENSION_NAME
                }
                #[cfg(not(target_os = "scarlet"))]
                {
                    let _ = name;
                    false
                }
            }
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
    if !allocator.is_null()
        || !device_chain_supported(info.p_next)
        || !info.flags.is_empty()
        || !device_extensions_supported(info)
    {
        return vk::Result::ERROR_FEATURE_NOT_PRESENT;
    }
    if !info.p_enabled_features.is_null()
        && std::slice::from_raw_parts(
            info.p_enabled_features.cast::<u32>(),
            std::mem::size_of::<vk::PhysicalDeviceFeatures>() / 4,
        )
        .iter()
        .any(|&v| v != 0)
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
    let lost = Arc::new(AtomicBool::new(false));
    let worker_fences = Arc::clone(&fences);
    let worker_recordings = Arc::clone(&recordings);
    let worker_in_flight = Arc::clone(&in_flight);
    let worker_lost = Arc::clone(&lost);
    let (tx, rx) = mpsc::channel();
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    let thread = match std::thread::Builder::new()
        .name("sgfx-vulkan-device".into())
        .spawn(move || {
            let init = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                Runtime::new(
                    adapter,
                    worker_recordings,
                    worker_in_flight,
                    worker_fences,
                    worker_lost,
                )
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
                    } => finish_submissions(
                        &submissions,
                        fence.as_ref(),
                        &signals,
                        &completion_in_flight,
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
    let driver = Arc::new(Driver {
        sender: tx,
        thread: Mutex::new(Some(thread)),
        completion_sender: completion_tx,
        completion_thread: Mutex::new(Some(completion_thread)),
        queue: AtomicU64::new(0),
        fences,
        semaphores,
        recordings,
        in_flight,
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
    driver.in_flight.wait();
    let _ = driver.completion_sender.send(CompletionRequest::Stop);
    if let Some(thread) = driver
        .completion_thread
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take()
    {
        let _ = thread.join();
    }
    driver.lost.store(true, Ordering::Release);
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
    driver.in_flight.wait();
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
    position: usize,
    index: u32,
    set: vk::DescriptorSet,
    dynamic_offsets: Vec<u32>,
}
#[derive(Clone)]
struct DeferredBarrier {
    position: usize,
    sets: Vec<vk::DescriptorSet>,
    after: ir::BufferAccess,
}
#[derive(Clone)]
struct ReadImage {
    mip_level: u32,
    position: usize,
    image: vk::Image,
    buffer: vk::Buffer,
    offset: u64,
    width: u32,
    height: u32,
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
    BindDescriptorSets {
        point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        first: u32,
        sets: Vec<vk::DescriptorSet>,
        dynamic_offsets: Vec<u32>,
    },
    Dispatch {
        x: u32,
        y: u32,
        z: u32,
    },
    BeginRenderPass {
        extended: bool,
        contents: vk::SubpassContents,
        render_pass: vk::RenderPass,
        framebuffer: vk::Framebuffer,
        area: vk::Rect2D,
        clears: Vec<vk::ClearValue>,
    },
    EndRenderPass,
    SetViewport(vk::Viewport),
    SetScissor(vk::Rect2D),
    BindVertexBuffer {
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
    CopyBuffer {
        source: vk::Buffer,
        destination: vk::Buffer,
        regions: Vec<vk::BufferCopy>,
    },
    CopyBufferToImage {
        source: vk::Buffer,
        image: vk::Image,
        layout: vk::ImageLayout,
        regions: Vec<vk::BufferImageCopy>,
    },
    PipelineBarrier {
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
    written_sets: Vec<vk::DescriptorSet>,
    compute: Option<vk::Pipeline>,
    graphics: Option<vk::Pipeline>,
    vertex_buffer: Option<(vk::Buffer, u64)>,
    index_buffer: Option<(vk::Buffer, u64, ir::IndexFormat)>,
    bound_sets: Vec<vk::DescriptorSet>,
    compute_sets: BTreeMap<u32, vk::DescriptorSet>,
    graphics_sets: BTreeMap<u32, vk::DescriptorSet>,
    compute_offsets: BTreeMap<u32, Vec<u32>>,
    graphics_offsets: BTreeMap<u32, Vec<u32>>,
    render: Option<(u32, u32)>,
    viewport: Option<vk::Viewport>,
    scissor: Option<vk::Rect2D>,
    used_buffers: Vec<vk::Buffer>,
    readback_buffers: Vec<vk::Buffer>,
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
    vertex_buffer: Option<(ir::BufferId, u64)>,
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
            RecordedCommand::BeginRenderPass { .. } if self.render_active => {
                self.fail(vk::Result::ERROR_INITIALIZATION_FAILED);
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            }
            RecordedCommand::BeginRenderPass { .. } => self.render_active = true,
            RecordedCommand::EndRenderPass if !self.render_active => {
                self.fail(vk::Result::ERROR_INITIALIZATION_FAILED);
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            }
            RecordedCommand::EndRenderPass => self.render_active = false,
            RecordedCommand::Draw { .. } | RecordedCommand::DrawIndexed { .. }
                if !self.render_active =>
            {
                self.fail(vk::Result::ERROR_INITIALIZATION_FAILED);
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            }
            RecordedCommand::Dispatch { .. } | RecordedCommand::PipelineBarrier { .. }
                if self.render_active =>
            {
                self.fail(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
            }
            RecordedCommand::CopyImageToBuffer { .. }
            | RecordedCommand::BlitImage { .. }
            | RecordedCommand::CopyBuffer { .. }
            | RecordedCommand::CopyBufferToImage { .. }
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
            written_sets: Vec::new(),
            compute: None,
            graphics: None,
            vertex_buffer: None,
            index_buffer: None,
            bound_sets: Vec::new(),
            compute_sets: BTreeMap::new(),
            graphics_sets: BTreeMap::new(),
            compute_offsets: BTreeMap::new(),
            graphics_offsets: BTreeMap::new(),
            render: None,
            viewport: None,
            scissor: None,
            used_buffers: Vec::new(),
            readback_buffers: Vec::new(),
            used_images: Vec::new(),
            used_pipelines: Vec::new(),
            used_framebuffers: Vec::new(),
            used_render_passes: Vec::new(),
        }
    }
}

impl RecordedCommand {
    fn apply(&self, rt: &mut Runtime, rec: &mut ResolvedRecording) -> VkResult<()> {
        match self {
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
            Self::BindDescriptorSets {
                point,
                layout,
                first,
                sets,
                dynamic_offsets,
            } => {
                if first
                    .checked_add(sets.len() as u32)
                    .is_none_or(|count| count > 4)
                {
                    return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                }
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
                    if layout.bind_groups().get(*first as usize + index) != Some(&descriptor.layout)
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
                    if values.iter().any(|offset| !offset.is_multiple_of(256)) {
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
                let Some(crate::resources::Pipeline::Compute(id)) =
                    rt.resources.pipelines.get(&pipeline)
                else {
                    return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                };
                rec.ops.push(ir::OwnedCommand::BeginComputePass);
                rec.ops.push(ir::OwnedCommand::SetComputePipeline(*id));
                let desc = rt
                    .table
                    .compute_pipeline(
                        rt.table
                            .compute_pipeline_ref(*id)
                            .map_err(crate::resources::failure)?,
                    )
                    .map_err(crate::resources::failure)?;
                rec.ops.extend(rec.push_constants.snapshot(desc.layout())?);
                let active_sets = active_sets(&rt.resources, desc.layout(), &rec.compute_sets)?;
                for &(index, set) in &active_sets {
                    rec.descriptors.push(DescriptorInsertion {
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
                mark_writable_descriptor_buffers(&rt.resources, rec, &active_sets)?;
                rec.written_sets
                    .extend(active_sets.into_iter().map(|(_, set)| set));
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
                extended,
                contents,
                render_pass,
                framebuffer,
                area,
                clears,
            } => {
                if *extended || *contents != vk::SubpassContents::INLINE || rec.render.is_some() {
                    return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                }
                let pass = rt
                    .resources
                    .render_passes
                    .get(render_pass)
                    .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                let framebuffer_data = rt
                    .resources
                    .framebuffers
                    .get(framebuffer)
                    .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                if framebuffer_data.render_pass != *render_pass
                    || area.offset.x != 0
                    || area.offset.y != 0
                    || area.extent.width != framebuffer_data.width
                    || area.extent.height != framebuffer_data.height
                {
                    return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                }
                let image = rt
                    .resources
                    .images
                    .get(&framebuffer_data.image)
                    .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                if !image.usable() {
                    return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                }
                let load = match pass.load_op {
                    vk::AttachmentLoadOp::LOAD => ir::LoadOp::Load,
                    vk::AttachmentLoadOp::DONT_CARE => ir::LoadOp::DontCare,
                    vk::AttachmentLoadOp::CLEAR => {
                        let clear = clears
                            .get(pass.color_index as usize)
                            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                        let color = unsafe { clear.color.float32 };
                        ir::LoadOp::Clear(
                            ir::Color::rgba(color[0], color[1], color[2], color[3])
                                .map_err(crate::resources::failure)?,
                        )
                    }
                    _ => return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT),
                };
                let depth = if let Some(load_op) = pass.depth_load_op {
                    let (_, handle) = framebuffer_data
                        .depth
                        .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                    let depth_image = rt
                        .resources
                        .images
                        .get(&handle)
                        .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                    if !depth_image.usable() {
                        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                    }
                    let load = match load_op {
                        vk::AttachmentLoadOp::LOAD => ir::DepthLoadOp::Load,
                        vk::AttachmentLoadOp::DONT_CARE => ir::DepthLoadOp::DontCare,
                        vk::AttachmentLoadOp::CLEAR => {
                            let value = unsafe {
                                clears
                                    .get(
                                        pass.depth_index
                                            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?
                                            as usize,
                                    )
                                    .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?
                                    .depth_stencil
                                    .depth
                            };
                            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                            }
                            ir::DepthLoadOp::Clear(value)
                        }
                        _ => return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT),
                    };
                    rec.used_images.push(handle);
                    Some(ir::OwnedDepthAttachment {
                        target: depth_image.id,
                        load,
                        store: if pass.depth_store_op == vk::AttachmentStoreOp::STORE {
                            ir::StoreOp::Store
                        } else {
                            ir::StoreOp::DontCare
                        },
                    })
                } else {
                    None
                };
                let render_area =
                    ir::PixelRect::new(0, 0, framebuffer_data.width, framebuffer_data.height)
                        .map_err(crate::resources::failure)?;
                rec.ops
                    .push(ir::OwnedCommand::BeginRenderPass(ir::OwnedRenderPassDesc {
                        target: image.id,
                        area: render_area,
                        load,
                        store: if pass.store_op == vk::AttachmentStoreOp::STORE {
                            ir::StoreOp::Store
                        } else {
                            ir::StoreOp::DontCare
                        },
                        depth,
                    }));
                rec.render = Some((framebuffer_data.width, framebuffer_data.height));
                rec.emitted_graphics = Default::default();
                rec.used_images.push(framebuffer_data.image);
                rec.used_framebuffers.push(*framebuffer);
                rec.used_render_passes.push(*render_pass);
            }
            Self::EndRenderPass => {
                if rec.render.take().is_none() {
                    return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                }
                rec.ops.push(ir::OwnedCommand::EndRenderPass);
            }
            Self::SetViewport(viewport) => rec.viewport = Some(*viewport),
            Self::SetScissor(scissor) => rec.scissor = Some(*scissor),
            Self::BindVertexBuffer { buffer, offset } => {
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
                rec.vertex_buffer = Some((*buffer, *offset));
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
                if *instances > 1 || *first_instance != 0 {
                    return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                }
                let visible = graphics_bindings(&rt.table, &rt.resources, rec)?;
                if visible && *vertices != 0 && *instances != 0 {
                    rec.ops.push(ir::OwnedCommand::Draw {
                        vertex_count: *vertices,
                        first_vertex: *first,
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
                if *instances > 1 || *first_instance != 0 {
                    return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                }
                let visible = graphics_bindings(&rt.table, &rt.resources, rec)?;
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
                    rec.ops.push(ir::OwnedCommand::DrawIndexed {
                        index_count: *indices,
                        first_index: *first,
                        base_vertex: *base_vertex,
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
                        vk::Format::R8G8B8A8_UNORM | vk::Format::B8G8R8A8_UNORM
                    )
                {
                    return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                }
                for region in regions {
                    let extent = image_data.mip_extent(region.image_subresource.mip_level)?;
                    if (region.buffer_row_length != 0 && region.buffer_row_length != extent.width)
                        || (region.buffer_image_height != 0
                            && region.buffer_image_height != extent.height)
                        || region.image_offset != vk::Offset3D::default()
                        || region.image_extent != extent
                        || region.image_subresource.aspect_mask != vk::ImageAspectFlags::COLOR
                        || region.image_subresource.base_array_layer != 0
                        || region.image_subresource.layer_count != 1
                    {
                        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                    }
                    let size = u64::from(extent.width) * u64::from(extent.height) * 4;
                    if region.buffer_offset % 4 != 0
                        || region
                            .buffer_offset
                            .checked_add(size)
                            .is_none_or(|end| end > buffer_data.size)
                    {
                        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                    }
                    rec.copies.push(ReadImage {
                        mip_level: region.image_subresource.mip_level,
                        position: rec.ops.len(),
                        image: *image,
                        buffer: *buffer,
                        offset: region.buffer_offset,
                        width: extent.width,
                        height: extent.height,
                    });
                }
                rec.used_buffers.push(*buffer);
                rec.used_images.push(*image);
                rec.readback_buffers.push(*buffer);
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
                rec.used_buffers.push(*source);
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
                    rec.ops.push(ir::OwnedCommand::CopyBufferToBuffer {
                        source: source_data.id,
                        source_offset: region.src_offset,
                        destination: destination_data.id,
                        destination_offset: region.dst_offset,
                        size: region.size,
                    });
                }
                rec.used_buffers.extend([*source, *destination]);
                rec.readback_buffers.push(*destination);
            }
            Self::PipelineBarrier {
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
                    if source
                        .intersects(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::MEMORY_WRITE)
                    {
                        rec.barriers.push(DeferredBarrier {
                            position: rec.ops.len(),
                            sets: rec.written_sets.clone(),
                            after: buffer_access(*destination)?,
                        });
                        rec.written_sets.clear();
                    } else if !(vk::AccessFlags::TRANSFER_WRITE
                        | vk::AccessFlags::HOST_WRITE
                        | vk::AccessFlags::MEMORY_WRITE)
                        .contains(*source)
                        || !supported_access(*destination)
                    {
                        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
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
                    if *extended
                        || !(*source_queue == vk::QUEUE_FAMILY_IGNORED
                            && *destination_queue == vk::QUEUE_FAMILY_IGNORED)
                    {
                        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                    }
                    let buffer_data = rt
                        .resources
                        .buffers
                        .get(buffer)
                        .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                    if *offset != 0 || (*size != vk::WHOLE_SIZE && *size != buffer_data.size) {
                        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                    }
                    rec.ops.push(ir::OwnedCommand::ResourceBarrier(
                        ir::OwnedResourceBarrier::Buffer {
                            buffer: buffer_data.id,
                            before: buffer_access(*source)?,
                            after: buffer_access(*destination)?,
                        },
                    ));
                    rec.used_buffers.push(*buffer);
                }
                for (
                    extended,
                    _source,
                    _destination,
                    source_queue,
                    destination_queue,
                    image,
                    old_layout,
                    new_layout,
                    range,
                ) in images
                {
                    if *extended
                        || !(*source_queue == vk::QUEUE_FAMILY_IGNORED
                            && *destination_queue == vk::QUEUE_FAMILY_IGNORED)
                        || range.base_array_layer != 0
                        || range.layer_count != 1
                    {
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
                    let access = |layout| {
                        if layout == vk::ImageLayout::PRESENT_SRC_KHR
                            && image_data.swapchain.is_some()
                        {
                            Ok(ir::TextureAccess::RenderAttachment)
                        } else {
                            texture_access(layout)
                        }
                    };
                    let after = access(*new_layout)?;
                    if *old_layout != vk::ImageLayout::UNDEFINED {
                        for mip_level in range.base_mip_level..end {
                            rec.ops.push(ir::OwnedCommand::ResourceBarrier(
                                ir::OwnedResourceBarrier::TextureMip {
                                    mip_level,
                                    texture: image_data.id,
                                    before: access(*old_layout)?,
                                    after,
                                },
                            ));
                        }
                    }
                    rec.used_images.push(*image);
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
    for command in source.commands.iter() {
        command.apply(rt, &mut resolved)?;
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
    if recording.state != RecordingState::Recording || recording.render_active {
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
    let extended = !info.p_next.is_null();
    if extended || contents != vk::SubpassContents::INLINE {
        record_error(command, vk::Result::ERROR_FEATURE_NOT_PRESENT);
        return;
    }
    record(
        command,
        RecordedCommand::BeginRenderPass {
            extended,
            contents,
            render_pass,
            framebuffer,
            area,
            clears,
        },
    )
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
    if first != 0 || count != 1 || buffers.is_null() || offsets.is_null() {
        record_error(command, vk::Result::ERROR_FEATURE_NOT_PRESENT);
        return;
    }
    let (buffer, offset) = (*buffers, *offsets);
    if !offset.is_multiple_of(4) {
        record_error(command, vk::Result::ERROR_INITIALIZATION_FAILED);
        return;
    }
    record(
        command,
        RecordedCommand::BindVertexBuffer { buffer, offset },
    )
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
    resources: &crate::resources::Resources,
    rec: &mut ResolvedRecording,
) -> VkResult<bool> {
    let extent = rec.render.ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
    let pipeline = rec
        .graphics
        .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
    let Some(crate::resources::Pipeline::Graphics(id)) = resources.pipelines.get(&pipeline) else {
        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
    };
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
    if scissor.offset.x < 0 || scissor.offset.y < 0 {
        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
    }
    let left = (scissor.offset.x as u32).min(extent.0);
    let top = (scissor.offset.y as u32).min(extent.1);
    let right = (u64::from(left) + u64::from(scissor.extent.width)).min(u64::from(extent.0)) as u32;
    let bottom =
        (u64::from(top) + u64::from(scissor.extent.height)).min(u64::from(extent.1)) as u32;
    let visible = left < right && top < bottom;
    let desc = table
        .programmable_render_pipeline_shared(
            table
                .programmable_render_pipeline_ref(*id)
                .map_err(crate::resources::failure)?,
        )
        .map_err(crate::resources::failure)?;
    emit_changed(
        &mut rec.emitted_graphics.viewport,
        viewport,
        &mut rec.ops,
        ir::OwnedCommand::SetViewport,
    );
    if visible {
        let scissor = ir::PixelRect::new(left, top, right - left, bottom - top)
            .map_err(crate::resources::failure)?;
        emit_changed(
            &mut rec.emitted_graphics.scissor,
            scissor,
            &mut rec.ops,
            |rect| ir::OwnedCommand::SetScissor(Some(rect)),
        );
    }
    emit_changed(
        &mut rec.emitted_graphics.pipeline,
        *id,
        &mut rec.ops,
        ir::OwnedCommand::SetProgrammablePipeline,
    );
    let push_key = (*id, rec.push_constants_revision);
    if rec.emitted_graphics.push_constants != Some(push_key) {
        rec.ops.extend(rec.push_constants.snapshot(desc.layout())?);
        rec.emitted_graphics.push_constants = Some(push_key);
    }
    if desc.vertex_buffer().is_some() {
        let (handle, offset) = rec
            .vertex_buffer
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let buffer = resources
            .buffers
            .get(&handle)
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        emit_changed(
            &mut rec.emitted_graphics.vertex_buffer,
            (buffer.id, offset),
            &mut rec.ops,
            |(buffer, offset)| ir::OwnedCommand::SetVertexBuffer { buffer, offset },
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
            position: rec.ops.len(),
            index,
            set,
            dynamic_offsets,
        });
    }
    mark_writable_descriptor_buffers(resources, rec, &active_sets)?;
    rec.written_sets
        .extend(active_sets.into_iter().map(|(_, set)| set));
    Ok(visible)
}

fn mark_writable_descriptor_buffers(
    resources: &crate::resources::Resources,
    rec: &mut ResolvedRecording,
    sets: &[(u32, vk::DescriptorSet)],
) -> VkResult<()> {
    for (_, handle) in sets {
        let set = resources
            .descriptor_sets
            .get(handle)
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        for (&binding, resource) in &set.bindings {
            let writable = set.layout.entries().iter().any(|entry| {
                entry.binding() == binding * 2
                    && matches!(
                        entry.ty(),
                        ir::BindingType::StorageBuffer { read_only: false }
                    )
            });
            if writable {
                let crate::resources::DescriptorBinding::Buffer { buffer, .. } = resource else {
                    continue;
                };
                if !rec.readback_buffers.contains(buffer) {
                    rec.readback_buffers.push(*buffer);
                }
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
    if instances > 1 || first_instance != 0 {
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
    if instances > 1 || first_instance != 0 {
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
            source_stage,
            destination_stage,
            flags,
            memory: m,
            buffers: b,
            images: i,
        },
    )
}

fn validate_pipeline_barrier_header(
    source_stage: vk::PipelineStageFlags,
    destination_stage: vk::PipelineStageFlags,
    flags: vk::DependencyFlags,
) -> VkResult<()> {
    let allowed = vk::PipelineStageFlags::TOP_OF_PIPE
        | vk::PipelineStageFlags::BOTTOM_OF_PIPE
        | vk::PipelineStageFlags::HOST
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
        | vk::AccessFlags::INDEX_READ)
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
    match layout {
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        | vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => {
            Ok(ir::TextureAccess::RenderAttachment)
        }
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => Ok(ir::TextureAccess::CopySource),
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => Ok(ir::TextureAccess::CopyDestination),
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => Ok(ir::TextureAccess::Sampled),
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
    let start = std::time::Instant::now();
    loop {
        if d.lost.load(Ordering::Acquire) {
            return vk::Result::ERROR_DEVICE_LOST;
        }
        let ready = if all != 0 {
            fs.iter().all(|f| f.load(Ordering::Acquire) == 1)
        } else {
            fs.iter().any(|f| f.load(Ordering::Acquire) == 1)
        };
        if ready {
            return vk::Result::SUCCESS;
        }
        if timeout != u64::MAX && start.elapsed().as_nanos() >= timeout as u128 {
            return vk::Result::TIMEOUT;
        }
        std::thread::sleep(std::time::Duration::from_micros(50));
    }
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
        || !info.p_next.is_null()
        || !info.flags.is_empty()
        || !allocator.is_null()
    {
        return vk::Result::ERROR_FEATURE_NOT_PRESENT;
    }
    let d = match driver(device.as_raw(), Kind::Device) {
        Ok(d) => d,
        Err(error) => return error,
    };
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
        d.semaphores
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&semaphore.as_raw());
    }
}

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
    Ok(())
}

#[cfg(any(
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
    let submits = match slice(submits, count) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let mut commands = Vec::new();
    let mut wait_handles = Vec::new();
    let mut signal_handles = Vec::new();
    for submit in submits {
        if !submit.p_next.is_null() {
            return vk::Result::ERROR_FEATURE_NOT_PRESENT;
        }
        let waits = match slice(submit.p_wait_semaphores, submit.wait_semaphore_count) {
            Ok(v) => v,
            Err(error) => return error,
        };
        let signals = match slice(submit.p_signal_semaphores, submit.signal_semaphore_count) {
            Ok(v) => v,
            Err(error) => return error,
        };
        if !waits.is_empty() {
            let stages = match slice(submit.p_wait_dst_stage_mask, submit.wait_semaphore_count) {
                Ok(v) => v,
                Err(error) => return error,
            };
            if stages.iter().any(|stage| stage.is_empty()) {
                return vk::Result::ERROR_FEATURE_NOT_PRESENT;
            }
        }
        wait_handles.extend_from_slice(waits);
        signal_handles.extend_from_slice(signals);
        let c = match slice(submit.p_command_buffers, submit.command_buffer_count) {
            Ok(v) => v,
            Err(e) => return e,
        };
        commands.extend(c.iter().map(|c| c.as_raw()));
    }
    let d = match driver(queue.as_raw(), Kind::Queue) {
        Ok(d) => d,
        Err(e) => return e,
    };
    let mut unique = wait_handles
        .iter()
        .chain(&signal_handles)
        .map(|handle| handle.as_raw())
        .collect::<Vec<_>>();
    unique.sort_unstable();
    if unique.windows(2).any(|pair| pair[0] == pair[1]) {
        return vk::Result::ERROR_FEATURE_NOT_PRESENT;
    }
    let wait_semaphores = match semaphore_refs(&d, &wait_handles) {
        Ok(semaphores) => semaphores,
        Err(error) => return error,
    };
    let signal_semaphores = match semaphore_refs(&d, &signal_handles) {
        Ok(semaphores) => semaphores,
        Err(error) => return error,
    };
    for &id in &commands {
        if !driver(id, Kind::Command).is_ok_and(|v| Arc::ptr_eq(&v, &d)) {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        }
    }
    let recorded = {
        let registry = d
            .recordings
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match commands
            .iter()
            .map(|id| {
                registry
                    .commands
                    .get(id)
                    .cloned()
                    .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)
            })
            .collect::<VkResult<Vec<RecordingCell>>>()
        {
            Ok(recorded) => recorded,
            Err(error) => return error,
        }
    };
    let recorded = match recorded
        .iter()
        .map(|recording| {
            let recording = recording
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            (recording.state == RecordingState::Executable)
                .then(|| recording.clone())
                .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)
        })
        .collect::<VkResult<Vec<_>>>()
    {
        Ok(recorded) => recorded,
        Err(error) => return error,
    };
    if let Err(error) = wait_and_consume_semaphores(&d, &wait_semaphores) {
        return error;
    }
    let accepted = call(&d, move |rt| {
        if rt.lost {
            return Err(vk::Result::ERROR_DEVICE_LOST);
        }
        let fence_state = if fence == vk::Fence::null() {
            None
        } else {
            let fences = rt.fences.lock().unwrap_or_else(|e| e.into_inner());
            let state = fences
                .get(&fence.as_raw())
                .cloned()
                .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
            if state.load(Ordering::Acquire) != 0 {
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            }
            Some(state)
        };
        let recordings = recorded
            .iter()
            .map(|recording| resolve_recording(rt, recording))
            .collect::<VkResult<Vec<_>>>()?;
        // Validate all object references before any GPU acceptance.
        for rec in &recordings {
            validate_objects(rt, rec)?;
        }
        for state in &signal_semaphores {
            if state
                .compare_exchange(0, 2, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                for reserved in &signal_semaphores {
                    if Arc::ptr_eq(reserved, state) {
                        break;
                    }
                    reserved.store(0, Ordering::Release);
                }
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            }
        }
        if let Some(state) = &fence_state {
            state.store(2, Ordering::Release);
        }
        let mut submissions = Vec::new();
        for recording in &recordings {
            match execute(rt, recording) {
                Ok(mut accepted) => submissions.append(&mut accepted),
                Err(error) => {
                    for submission in &submissions {
                        let _ = wait_submission(rt, submission);
                    }
                    if error == vk::Result::ERROR_DEVICE_LOST {
                        rt.lost = true;
                    }
                    if let Some(state) = &fence_state {
                        state.store(0, Ordering::Release);
                    }
                    for state in &signal_semaphores {
                        state.store(0, Ordering::Release);
                    }
                    return Err(error);
                }
            }
        }
        let submitted = {
            let registry = rt
                .recordings
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            commands
                .iter()
                .filter_map(|id| registry.commands.get(id).cloned())
                .collect::<Vec<_>>()
        };
        for recording in submitted {
            let mut recording = recording
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if recording.one_time {
                recording.state = RecordingState::Invalid;
            }
        }
        if submissions.is_empty() {
            if let Some(state) = &fence_state {
                state.store(1, Ordering::Release);
            }
            for state in &signal_semaphores {
                state.store(1, Ordering::Release);
            }
        } else {
            rt.in_flight.begin();
        }
        Ok((submissions, fence_state, signal_semaphores))
    });
    let (submissions, fence_state, signal_semaphores) = match accepted {
        Ok(accepted) => accepted,
        Err(error) => return error,
    };
    if submissions.is_empty() {
        return vk::Result::SUCCESS;
    }
    if let Err(error) = d.completion_sender.send(CompletionRequest::Observe {
        submissions,
        fence: fence_state,
        signals: signal_semaphores,
    }) {
        let CompletionRequest::Observe {
            submissions,
            fence,
            signals,
        } = error.0
        else {
            unreachable!();
        };
        finish_submissions(
            &submissions,
            fence.as_ref(),
            &signals,
            &d.in_flight,
            &d.lost,
        );
    }
    vk::Result::SUCCESS
}

fn finish_submissions(
    submissions: &[sgfx::driver::Submission],
    fence: Option<&Arc<AtomicU8>>,
    signals: &[Arc<AtomicU8>],
    in_flight: &InFlight,
    lost: &AtomicBool,
) {
    let complete = submissions
        .iter()
        .all(|submission| matches!(submission.wait(None), Ok(CompletionStatus::Complete)));
    if complete {
        if let Some(state) = fence {
            state.store(1, Ordering::Release);
        }
        for state in signals {
            state.store(1, Ordering::Release);
        }
    } else {
        lost.store(true, Ordering::Release);
        if let Some(state) = fence {
            state.store(0, Ordering::Release);
        }
        for state in signals {
            state.store(0, Ordering::Release);
        }
    }
    in_flight.finish();
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
                !rt.resources.views.get(view).is_some_and(|image| {
                    rt.resources
                        .images
                        .get(image)
                        .is_some_and(|image| image.usable())
                })
            }) || fb
                .depth
                .is_some_and(|(view, _)| !rt.resources.views.contains_key(&view))
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
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            }
        }
    }
    Ok(())
}
fn submit_owned(
    rt: &mut Runtime,
    ops: Vec<ir::OwnedCommand>,
) -> VkResult<Option<sgfx::driver::Submission>> {
    if ops.is_empty() {
        return Ok(None);
    }
    let owned = ir::OwnedCommandBuffer::new(ops);
    let commands = owned.record(&rt.table).map_err(crate::resources::failure)?;
    match rt.queue.submit(&mut rt.cache, &commands) {
        Ok(receipt) => Ok(Some(receipt)),
        Err(SubmitError::Busy) => Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY),
        Err(SubmitError::Rejected(error)) => Err(crate::runtime::backend_failure(error)),
        Err(SubmitError::Failed {
            error: _,
            completion,
        }) => {
            let _ = completion.wait(None);
            rt.lost = true;
            Err(vk::Result::ERROR_DEVICE_LOST)
        }
        Err(_) => Err(vk::Result::ERROR_DEVICE_LOST),
    }
}

fn wait_submission(rt: &mut Runtime, submission: &sgfx::driver::Submission) -> VkResult<()> {
    match submission.wait(None) {
        Ok(CompletionStatus::Complete) => Ok(()),
        Ok(CompletionStatus::Pending) | Ok(_) => {
            rt.lost = true;
            Err(vk::Result::ERROR_DEVICE_LOST)
        }
        Err(error) => {
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

fn execute(rt: &mut Runtime, rec: &ResolvedRecording) -> VkResult<Vec<sgfx::driver::Submission>> {
    let mut used = rec.used_buffers.clone();
    for insertion in &rec.descriptors {
        let set = rt
            .resources
            .descriptor_sets
            .get(&insertion.set)
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        for binding in set.bindings.values() {
            let crate::resources::DescriptorBinding::Buffer { buffer, .. } = binding else {
                continue;
            };
            if !used.contains(buffer) {
                used.push(*buffer);
            }
        }
    }
    used.sort_unstable_by_key(|b| b.as_raw());
    used.dedup();
    let mut ops = Vec::new();
    // Every upload is copied into SGFX-owned command storage before submission
    // returns, so application mappings need not remain borrowed by the backend.
    for handle in &used {
        let buffer = rt
            .resources
            .buffers
            .get(handle)
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        if let Some((memory, offset)) = buffer.bound {
            let memory = rt
                .resources
                .memories
                .get(&memory)
                .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
            let bytes = memory
                .bytes
                .get(offset as usize..(offset + buffer.size) as usize)
                .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
            ops.push(ir::OwnedCommand::WriteBuffer {
                buffer: buffer.id,
                offset: 0,
                data: bytes.to_vec(),
            });
        }
    }
    let mut descriptors = rec.descriptors.as_slice();
    let mut barriers = rec.barriers.as_slice();
    let mut copies = rec.copies.as_slice();
    for position in 0..=rec.ops.len() {
        for d in take_positioned(&mut descriptors, position, |d| d.position) {
            let group = rt
                .resources
                .descriptor_group(&rt.table, d.set, &d.dynamic_offsets)?;
            ops.push(ir::OwnedCommand::SetBindGroup {
                index: d.index,
                bind_group: group,
            });
        }
        for barrier in take_positioned(&mut barriers, position, |b| b.position) {
            let mut ids = Vec::new();
            for set in &barrier.sets {
                let set = rt
                    .resources
                    .descriptor_sets
                    .get(set)
                    .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                for binding in set.bindings.values() {
                    let crate::resources::DescriptorBinding::Buffer { buffer, .. } = binding else {
                        continue;
                    };
                    let b = rt
                        .resources
                        .buffers
                        .get(buffer)
                        .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                    if b.usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER)
                        && !ids.contains(&b.id)
                    {
                        ids.push(b.id);
                    }
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
            if let Some(submission) = submit_owned(rt, std::mem::take(&mut ops))? {
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
            if bytes.len() != (copy.width as usize) * (copy.height as usize) * 4 {
                return Err(vk::Result::ERROR_DEVICE_LOST);
            }
            let buffer = rt
                .resources
                .buffers
                .get(&copy.buffer)
                .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
            ops.push(ir::OwnedCommand::WriteBuffer {
                buffer: buffer.id,
                offset: copy.offset,
                data: bytes,
            });
        }
        if let Some(op) = rec.ops.get(position) {
            ops.push(op.clone());
        }
    }
    let mut submissions = submit_owned(rt, ops)?.into_iter().collect::<Vec<_>>();
    let mut readback = rec.readback_buffers.clone();
    readback.sort_unstable_by_key(|buffer| buffer.as_raw());
    readback.dedup();
    if !readback.is_empty() {
        for submission in &submissions {
            wait_submission(rt, submission)?;
        }
        submissions.clear();
    }
    // Stage only GPU-written bytes back into allocations returned by MapMemory.
    for handle in &readback {
        let buffer = rt
            .resources
            .buffers
            .get(handle)
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        if let Some((memory, offset)) = buffer.bound {
            let bytes = rt
                .cache
                .read_buffer(buffer.id, 0, buffer.size)
                .map_err(|_| vk::Result::ERROR_DEVICE_LOST)?;
            let memory = rt
                .resources
                .memories
                .get_mut(&memory)
                .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
            memory.bytes[offset as usize..(offset + buffer.size) as usize].copy_from_slice(&bytes);
        }
    }
    Ok(submissions)
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
        b"vkAllocateCommandBuffers" => entry!(allocate_command_buffers),
        b"vkFreeCommandBuffers" => entry!(free_command_buffers),
        b"vkResetCommandBuffer" => entry!(reset_command_buffer),
        b"vkBeginCommandBuffer" => entry!(begin_command_buffer),
        b"vkEndCommandBuffer" => entry!(end_command_buffer),
        b"vkCmdBindPipeline" => entry!(cmd_bind_pipeline),
        b"vkCmdBindDescriptorSets" => entry!(cmd_bind_descriptor_sets),
        b"vkCmdPushConstants" => entry!(cmd_push_constants),
        b"vkCmdBlitImage" => entry!(cmd_blit_image),
        b"vkCmdDispatch" => entry!(cmd_dispatch),
        b"vkCmdDraw" => entry!(cmd_draw),
        b"vkCmdDrawIndexed" => entry!(cmd_draw_indexed),
        b"vkCmdBindVertexBuffers" => entry!(cmd_bind_vertex_buffers),
        b"vkCmdBindIndexBuffer" => entry!(cmd_bind_index_buffer),
        b"vkCmdBeginRenderPass" => entry!(cmd_begin_render_pass),
        b"vkCmdEndRenderPass" => entry!(cmd_end_render_pass),
        b"vkCmdCopyImageToBuffer" => entry!(cmd_copy_image_to_buffer),
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
        #[cfg(any(target_os = "macos", all(target_os = "linux", feature = "scarlet-wsi")))]
        b"vkCreateSwapchainKHR" => entry!(crate::wsi::create_swapchain),
        #[cfg(any(target_os = "macos", all(target_os = "linux", feature = "scarlet-wsi")))]
        b"vkDestroySwapchainKHR" => entry!(crate::wsi::destroy_swapchain),
        #[cfg(any(target_os = "macos", all(target_os = "linux", feature = "scarlet-wsi")))]
        b"vkGetSwapchainImagesKHR" => entry!(crate::wsi::get_swapchain_images),
        #[cfg(any(target_os = "macos", all(target_os = "linux", feature = "scarlet-wsi")))]
        b"vkAcquireNextImageKHR" => entry!(crate::wsi::acquire_next_image),
        #[cfg(any(target_os = "macos", all(target_os = "linux", feature = "scarlet-wsi")))]
        b"vkQueuePresentKHR" => entry!(crate::wsi::queue_present),
        #[cfg(target_os = "scarlet")]
        b"vkGetImageScarletHandleSGFX" => {
            entry!(crate::scarlet_image::vkGetImageScarletHandleSGFX)
        }
        _ => crate::resources::lookup(name).or_else(|| crate::images::lookup(name)),
    }
}

/// Descriptor indexing is not exposed. Updating a referenced set invalidates
/// every recording or executable command buffer under Vulkan 1.0 rules.
pub(crate) fn invalidate_descriptor_sets(rt: &mut Runtime, sets: &[vk::DescriptorSet]) {
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
            .is_none_or(|s| &s.layout != expected)
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
                crate::resources::DescriptorSet {
                    pool: vk::DescriptorPool::from_raw(7),
                    layout: group_layout.clone(),
                    types: BTreeMap::from([(0, vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)]),
                    bindings: HashMap::from([(
                        0,
                        crate::resources::DescriptorBinding::Buffer {
                            buffer,
                            offset: 0,
                            range: 16,
                        },
                    )]),
                    cached_group: None,
                    invalid: false,
                },
            );
        }
        let mut rec = ResolvedRecording::new();
        rec.render = Some((32, 32));
        rec.graphics = Some(vk::Pipeline::from_raw(1));
        rec.vertex_buffer = Some((buffer, 0));
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
        assert!(graphics_bindings(&table, &resources, &mut rec).unwrap());
        assert_eq!(rec.ops.len(), 5);
        assert_eq!(rec.descriptors.len(), 1);
        for _ in 0..1000 {
            assert!(graphics_bindings(&table, &resources, &mut rec).unwrap());
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
            graphics_bindings(&table, &resources, &mut rec),
            Err(vk::Result::ERROR_INITIALIZATION_FAILED)
        );
        resources.buffers.remove(&vk::Buffer::from_raw(3));
        assert_eq!(
            graphics_bindings(&table, &resources, &mut rec),
            Err(vk::Result::ERROR_INITIALIZATION_FAILED)
        );
    }

    #[test]
    fn graphics_state_changes_emit_selective_updates_and_reseed_new_passes() {
        let (table, resources, mut rec, layout) = graphics_fixture();
        graphics_bindings(&table, &resources, &mut rec).unwrap();
        rec.graphics_offsets.insert(0, vec![256]);
        graphics_bindings(&table, &resources, &mut rec).unwrap();
        assert_eq!(rec.ops.len(), 5);
        assert_eq!(rec.descriptors.len(), 2);
        assert_eq!(rec.descriptors[1].dynamic_offsets, [256]);
        rec.graphics_sets.insert(0, vk::DescriptorSet::from_raw(6));
        graphics_bindings(&table, &resources, &mut rec).unwrap();
        assert_eq!(rec.descriptors.len(), 3);
        assert_eq!(rec.descriptors[2].set, vk::DescriptorSet::from_raw(6));
        rec.viewport.as_mut().unwrap().width = 16.0;
        rec.scissor.as_mut().unwrap().extent.width = 16;
        rec.vertex_buffer.as_mut().unwrap().1 = 4;
        graphics_bindings(&table, &resources, &mut rec).unwrap();
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
        graphics_bindings(&table, &resources, &mut rec).unwrap();
        assert!(
            matches!(rec.ops.last().unwrap(), ir::OwnedCommand::SetPushConstants { data, .. } if data == &[9; 4])
        );
        rec.graphics = Some(vk::Pipeline::from_raw(2));
        let previous = rec.ops.len();
        graphics_bindings(&table, &resources, &mut rec).unwrap();
        assert!(
            matches!(&rec.ops[previous..], [ir::OwnedCommand::SetProgrammablePipeline(_), ir::OwnedCommand::SetPushConstants { data, .. }] if data == &[9; 4])
        );
        rec.emitted_graphics = Default::default();
        let previous = rec.ops.len();
        graphics_bindings(&table, &resources, &mut rec).unwrap();
        assert_eq!(rec.ops.len() - previous, 5);
        assert_eq!(rec.descriptors.len(), 4);
    }

    #[test]
    fn cached_graphics_state_preserves_invisible_draw_and_push_layout_errors() {
        let (table, mut resources, mut rec, layout) = graphics_fixture();
        graphics_bindings(&table, &resources, &mut rec).unwrap();
        rec.scissor.as_mut().unwrap().extent.width = 0;
        assert!(!graphics_bindings(&table, &resources, &mut rec).unwrap());
        rec.scissor.as_mut().unwrap().extent.width = 8;
        assert!(graphics_bindings(&table, &resources, &mut rec).unwrap());
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
            graphics_bindings(&table, &resources, &mut rec),
            Err(vk::Result::ERROR_INITIALIZATION_FAILED)
        );
        resources.pipelines.remove(&vk::Pipeline::from_raw(1));
        assert_eq!(
            graphics_bindings(&table, &resources, &mut rec),
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
            fences: Default::default(),
            semaphores: Default::default(),
            recordings: Default::default(),
            in_flight: Default::default(),
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
            fences: Default::default(),
            semaphores: Default::default(),
            recordings: Default::default(),
            in_flight: Default::default(),
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
            fences: Default::default(),
            semaphores: Default::default(),
            recordings: Default::default(),
            in_flight: Default::default(),
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
                (&base as *const vk::BaseInStructure<'_>).cast()
            ));
        }
        base.s_type = vk::StructureType::PHYSICAL_DEVICE_FEATURES_2;
        unsafe {
            assert!(!device_chain_supported(
                (&base as *const vk::BaseInStructure<'_>).cast()
            ));
        }
    }

    #[test]
    fn device_chain_accepts_disabled_features_without_enabling_them() {
        fn supported(
            indexing: &mut vk::PhysicalDeviceDescriptorIndexingFeatures<'_>,
            draw: &mut vk::PhysicalDeviceShaderDrawParametersFeatures<'_>,
        ) -> bool {
            indexing.p_next = (draw as *mut vk::PhysicalDeviceShaderDrawParametersFeatures<'_>).cast();
            let loader = vk::BaseInStructure {
                s_type: vk::StructureType::LOADER_DEVICE_CREATE_INFO,
                p_next: (indexing as *const vk::PhysicalDeviceDescriptorIndexingFeatures<'_>).cast(),
                ..Default::default()
            };
            unsafe { device_chain_supported((&loader as *const vk::BaseInStructure<'_>).cast()) }
        }
        let mut draw = vk::PhysicalDeviceShaderDrawParametersFeatures::default();
        let mut indexing = vk::PhysicalDeviceDescriptorIndexingFeatures::default();
        assert!(supported(&mut indexing, &mut draw));
        indexing.descriptor_binding_partially_bound = vk::TRUE;
        assert!(!supported(&mut indexing, &mut draw));
        indexing.descriptor_binding_partially_bound = vk::FALSE;
        draw.shader_draw_parameters = vk::TRUE;
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
            assert!(!device_chain_supported((&loader as *const vk::BaseInStructure<'_>).cast()));
        }
    }
}
