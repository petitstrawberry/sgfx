use crate::runtime::Runtime;
use ash::vk::{self, Handle};
use sgfx_core::{
    backend::{Completion, CompletionStatus},
    ir,
};
use std::{
    collections::{BTreeMap, HashMap},
    ffi::{CStr, c_char},
    sync::{
        Arc, Mutex, OnceLock,
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
pub(crate) type Fences = Arc<Mutex<HashMap<u64, Arc<AtomicU8>>>>;
struct Driver {
    sender: mpsc::Sender<Request>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    queue: AtomicU64,
    fences: Fences,
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
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| op(runtime)));
            let result = match result {
                Ok(value) => value,
                Err(_) => {
                    runtime.lost = true;
                    Err(vk::Result::ERROR_DEVICE_LOST)
                }
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
fn with_command<T: Send + 'static>(
    command: vk::CommandBuffer,
    op: impl FnOnce(&mut Runtime, u64) -> VkResult<T> + Send + 'static,
) -> VkResult<T> {
    let id = command.as_raw();
    call(driver(id, Kind::Command)?.as_ref(), move |rt| op(rt, id))
}
fn result<T>(value: VkResult<T>, output: *mut T) -> vk::Result {
    match value {
        Ok(value) => {
            if output.is_null() {
                return vk::Result::ERROR_INITIALIZATION_FAILED;
            }
            unsafe {
                *output = value;
            }
            vk::Result::SUCCESS
        }
        Err(e) => e,
    }
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
        if header.s_type != vk::StructureType::LOADER_DEVICE_CREATE_INFO {
            return false;
        }
        next = header.p_next.cast();
    }
    false
}

pub(crate) unsafe extern "system" fn create_device(
    physical: vk::PhysicalDevice,
    info: *const vk::DeviceCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::Device,
) -> vk::Result {
    if !crate::instance::physical_valid(physical) || info.is_null() || out.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    *out = vk::Device::null();
    let info = &*info;
    if !allocator.is_null()
        || !device_chain_supported(info.p_next)
        || !info.flags.is_empty()
        || info.enabled_extension_count != 0
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
    let lost = Arc::new(AtomicBool::new(false));
    let worker_fences = Arc::clone(&fences);
    let worker_lost = Arc::clone(&lost);
    let (tx, rx) = mpsc::channel();
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    let thread = match std::thread::Builder::new()
        .name("sgfx-vulkan-device".into())
        .spawn(move || {
            let init = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                Runtime::new(worker_fences, worker_lost)
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
    let driver = Arc::new(Driver {
        sender: tx,
        thread: Mutex::new(Some(thread)),
        queue: AtomicU64::new(0),
        fences,
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
    status(with_device(device, |rt| {
        if rt.lost {
            Err(vk::Result::ERROR_DEVICE_LOST)
        } else {
            Ok(())
        }
    }))
}
unsafe extern "system" fn queue_wait_idle(queue: vk::Queue) -> vk::Result {
    status(driver(queue.as_raw(), Kind::Queue).and_then(|d| {
        call(&d, |rt| {
            if rt.lost {
                Err(vk::Result::ERROR_DEVICE_LOST)
            } else {
                Ok(())
            }
        })
    }))
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
}
#[derive(Clone)]
struct DeferredBarrier {
    position: usize,
    sets: Vec<vk::DescriptorSet>,
    after: ir::BufferAccess,
}
#[derive(Clone)]
struct ReadImage {
    position: usize,
    image: vk::Image,
    buffer: vk::Buffer,
    offset: u64,
    width: u32,
    height: u32,
}
#[derive(Clone)]
pub(crate) struct Recording {
    pool: u64,
    one_time: bool,
    state: RecordingState,
    error: Option<vk::Result>,
    ops: Vec<ir::OwnedCommand>,
    descriptors: Vec<DescriptorInsertion>,
    copies: Vec<ReadImage>,
    barriers: Vec<DeferredBarrier>,
    written_sets: Vec<vk::DescriptorSet>,
    compute: Option<vk::Pipeline>,
    graphics: Option<vk::Pipeline>,
    bound_sets: Vec<vk::DescriptorSet>,
    compute_sets: BTreeMap<u32, vk::DescriptorSet>,
    graphics_sets: BTreeMap<u32, vk::DescriptorSet>,
    render: Option<(u32, u32)>,
    used_buffers: Vec<vk::Buffer>,
    used_images: Vec<vk::Image>,
    used_pipelines: Vec<vk::Pipeline>,
    used_framebuffers: Vec<vk::Framebuffer>,
    used_render_passes: Vec<vk::RenderPass>,
}
impl Recording {
    fn new(pool: u64) -> Self {
        Self {
            pool,
            one_time: false,
            state: RecordingState::Initial,
            error: None,
            ops: Vec::new(),
            descriptors: Vec::new(),
            copies: Vec::new(),
            barriers: Vec::new(),
            written_sets: Vec::new(),
            compute: None,
            graphics: None,
            bound_sets: Vec::new(),
            compute_sets: BTreeMap::new(),
            graphics_sets: BTreeMap::new(),
            render: None,
            used_buffers: Vec::new(),
            used_images: Vec::new(),
            used_pipelines: Vec::new(),
            used_framebuffers: Vec::new(),
            used_render_passes: Vec::new(),
        }
    }
    fn fail(&mut self, e: vk::Result) {
        self.error.get_or_insert(e);
    }
    fn ready(&mut self) -> VkResult<()> {
        if self.state != RecordingState::Recording {
            self.fail(vk::Result::ERROR_INITIALIZATION_FAILED);
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        if self.ops.len() > 4000 {
            return Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY);
        }
        Ok(())
    }
}
fn record(
    command: vk::CommandBuffer,
    op: impl FnOnce(&mut Runtime, &mut Recording) -> VkResult<()> + Send + 'static,
) {
    let _ = with_command(command, move |rt, id| {
        let Some(mut rec) = rt.commands.remove(&id) else {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        };
        let res = rec.ready().and_then(|()| op(rt, &mut rec));
        if let Err(e) = res {
            rec.fail(e);
        }
        rt.commands.insert(id, rec);
        Ok(())
    });
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
    result(
        with_device(device, move |rt| {
            let id = next_id();
            rt.pools.insert(id, flags);
            Ok(vk::CommandPool::from_raw(id))
        }),
        out,
    )
}
unsafe extern "system" fn destroy_command_pool(
    device: vk::Device,
    pool: vk::CommandPool,
    _: *const vk::AllocationCallbacks<'_>,
) {
    if let Ok(ids) = with_device(device, move |rt| {
        rt.pools.remove(&pool.as_raw());
        let ids: Vec<_> = rt
            .commands
            .iter()
            .filter(|(_, r)| r.pool == pool.as_raw())
            .map(|(&i, _)| i)
            .collect();
        for id in &ids {
            rt.commands.remove(id);
        }
        Ok(ids)
    }) {
        for id in ids {
            remove_handle(id);
        }
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
    if let Err(e) = call(&d, move |rt| {
        if rt.pools.contains_key(&pool) {
            Ok(())
        } else {
            Err(vk::Result::ERROR_INITIALIZATION_FAILED)
        }
    }) {
        return e;
    }
    let ids: Vec<_> = (0..count).map(|_| add_handle(Kind::Command, &d)).collect();
    let allocated = ids.clone();
    if let Err(e) = call(&d, move |rt| {
        for id in allocated {
            rt.commands.insert(id, Recording::new(pool));
        }
        Ok(())
    }) {
        for id in ids {
            remove_handle(id);
        }
        return e;
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
    let ids: Vec<_> = commands.iter().map(|c| c.as_raw()).collect();
    let removed = ids.clone();
    if with_device(device, move |rt| {
        for id in removed {
            rt.commands.remove(&id);
        }
        Ok(())
    })
    .is_ok()
    {
        for id in ids {
            remove_handle(id);
        }
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
    status(with_device(device, move |rt| {
        if !rt.pools.contains_key(&pool.as_raw()) {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        for rec in rt.commands.values_mut().filter(|r| r.pool == pool.as_raw()) {
            *rec = Recording::new(rec.pool);
        }
        Ok(())
    }))
}
unsafe extern "system" fn reset_command_buffer(
    command: vk::CommandBuffer,
    flags: vk::CommandBufferResetFlags,
) -> vk::Result {
    if !vk::CommandBufferResetFlags::RELEASE_RESOURCES.contains(flags) {
        return vk::Result::ERROR_FEATURE_NOT_PRESENT;
    }
    status(with_command(command, |rt, id| {
        let rec = rt
            .commands
            .get_mut(&id)
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        if !rt
            .pools
            .get(&rec.pool)
            .is_some_and(|f| f.contains(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER))
        {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        *rec = Recording::new(rec.pool);
        Ok(())
    }))
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
    status(with_command(command, move |rt, id| {
        let rec = rt
            .commands
            .get_mut(&id)
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        if rec.state == RecordingState::Recording {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        if rec.state != RecordingState::Initial
            && !rt
                .pools
                .get(&rec.pool)
                .is_some_and(|f| f.contains(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER))
        {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        *rec = Recording::new(rec.pool);
        rec.state = RecordingState::Recording;
        rec.one_time = one_time;
        Ok(())
    }))
}
unsafe extern "system" fn end_command_buffer(command: vk::CommandBuffer) -> vk::Result {
    status(with_command(command, |rt, id| {
        let rec = rt
            .commands
            .get_mut(&id)
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        if rec.state != RecordingState::Recording || rec.render.is_some() {
            rec.fail(vk::Result::ERROR_INITIALIZATION_FAILED)
        }
        if let Some(e) = rec.error {
            rec.state = RecordingState::Invalid;
            Err(e)
        } else {
            rec.state = RecordingState::Executable;
            Ok(())
        }
    }))
}
unsafe extern "system" fn cmd_bind_pipeline(
    command: vk::CommandBuffer,
    point: vk::PipelineBindPoint,
    pipeline: vk::Pipeline,
) {
    record(command, move |rt, rec| {
        match (point, rt.resources.pipelines.get(&pipeline)) {
            (vk::PipelineBindPoint::COMPUTE, Some(crate::resources::Pipeline::Compute(_))) => {
                rec.compute = Some(pipeline)
            }
            (vk::PipelineBindPoint::GRAPHICS, Some(crate::resources::Pipeline::Graphics(_))) => {
                rec.graphics = Some(pipeline)
            }
            _ => return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT),
        }
        rec.used_pipelines.push(pipeline);
        Ok(())
    })
}
unsafe extern "system" fn cmd_bind_descriptor_sets(
    command: vk::CommandBuffer,
    point: vk::PipelineBindPoint,
    layout: vk::PipelineLayout,
    first: u32,
    count: u32,
    sets: *const vk::DescriptorSet,
    dynamic_count: u32,
    _dynamic: *const u32,
) {
    let copied = match slice(sets, count) {
        Ok(s) => s.to_vec(),
        Err(e) => {
            record(command, move |_, _| Err(e));
            return;
        }
    };
    record(command, move |rt, rec| {
        if dynamic_count != 0 || first.checked_add(count).is_none_or(|n| n > 4) {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let l = rt
            .resources
            .pipeline_layouts
            .get(&layout)
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let target = if point == vk::PipelineBindPoint::COMPUTE {
            &mut rec.compute_sets
        } else if point == vk::PipelineBindPoint::GRAPHICS {
            &mut rec.graphics_sets
        } else {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        };
        for (i, set) in copied.iter().enumerate() {
            let d = rt
                .resources
                .descriptor_sets
                .get(set)
                .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
            if l.bind_groups().get(first as usize + i) != Some(&d.layout) {
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            }
            target.insert(first + i as u32, *set);
            rec.bound_sets.push(*set);
        }
        Ok(())
    })
}
unsafe extern "system" fn cmd_dispatch(command: vk::CommandBuffer, x: u32, y: u32, z: u32) {
    record(command, move |rt, rec| {
        if rec.render.is_some() || x > 65535 || y > 65535 || z > 65535 {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let pipeline = rec.compute.ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let Some(crate::resources::Pipeline::Compute(id)) = rt.resources.pipelines.get(&pipeline)
        else {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        };
        rec.ops.push(ir::OwnedCommand::BeginComputePass);
        rec.ops.push(ir::OwnedCommand::SetComputePipeline(*id));
        let active_sets = active_sets(
            rt,
            crate::resources::Pipeline::Compute(*id),
            &rec.compute_sets,
        )?;
        for &(index, set) in &active_sets {
            rec.descriptors.push(DescriptorInsertion {
                position: rec.ops.len(),
                index,
                set,
            });
        }
        rec.written_sets
            .extend(active_sets.into_iter().map(|(_, set)| set));
        if x != 0 && y != 0 && z != 0 {
            rec.ops.push(ir::OwnedCommand::Dispatch { x, y, z });
        }
        rec.ops.push(ir::OwnedCommand::EndComputePass);
        Ok(())
    })
}
unsafe extern "system" fn cmd_begin_render_pass(
    command: vk::CommandBuffer,
    info: *const vk::RenderPassBeginInfo<'_>,
    contents: vk::SubpassContents,
) {
    if info.is_null() {
        record(command, |_, _| Err(vk::Result::ERROR_INITIALIZATION_FAILED));
        return;
    }
    let info = &*info;
    let clears = match slice(info.p_clear_values, info.clear_value_count) {
        Ok(v) => v.to_vec(),
        Err(e) => {
            record(command, move |_, _| Err(e));
            return;
        }
    };
    let render_pass = info.render_pass;
    let framebuffer = info.framebuffer;
    let area = info.render_area;
    let extended = !info.p_next.is_null();
    record(command, move |rt, rec| {
        if extended || contents != vk::SubpassContents::INLINE || rec.render.is_some() {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let pass = rt
            .resources
            .render_passes
            .get(&render_pass)
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let fb = rt
            .resources
            .framebuffers
            .get(&framebuffer)
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        if fb.render_pass != render_pass
            || area.offset.x != 0
            || area.offset.y != 0
            || area.extent.width != fb.width
            || area.extent.height != fb.height
        {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let image = rt
            .resources
            .images
            .get(&fb.image)
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        if image.bound.is_none() {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        let load = match pass.load_op {
            vk::AttachmentLoadOp::LOAD => ir::LoadOp::Load,
            vk::AttachmentLoadOp::DONT_CARE => ir::LoadOp::DontCare,
            vk::AttachmentLoadOp::CLEAR => {
                let clear = clears
                    .first()
                    .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                let c = clear.color.float32;
                ir::LoadOp::Clear(
                    ir::Color::rgba(c[0], c[1], c[2], c[3]).map_err(crate::resources::failure)?,
                )
            }
            _ => return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT),
        };
        let area =
            ir::PixelRect::new(0, 0, fb.width, fb.height).map_err(crate::resources::failure)?;
        rec.ops
            .push(ir::OwnedCommand::BeginRenderPass(ir::OwnedRenderPassDesc {
                target: image.id,
                area,
                load,
                store: ir::StoreOp::Store,
                depth: None,
            }));
        rec.render = Some((fb.width, fb.height));
        rec.used_images.push(fb.image);
        rec.used_framebuffers.push(framebuffer);
        rec.used_render_passes.push(render_pass);
        Ok(())
    })
}
unsafe extern "system" fn cmd_end_render_pass(command: vk::CommandBuffer) {
    record(command, |_, rec| {
        if rec.render.take().is_none() {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        rec.ops.push(ir::OwnedCommand::EndRenderPass);
        Ok(())
    })
}
unsafe extern "system" fn cmd_draw(
    command: vk::CommandBuffer,
    vertices: u32,
    instances: u32,
    first: u32,
    first_instance: u32,
) {
    record(command, move |rt, rec| {
        let extent = rec.render.ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        if instances > 1 || first_instance != 0 {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let pipeline = rec
            .graphics
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let Some(crate::resources::Pipeline::Graphics(id)) = rt.resources.pipelines.get(&pipeline)
        else {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        };
        if rt
            .resources
            .graphics_extents
            .get(&pipeline)
            .map(|s| (s.width, s.height))
            != Some(extent)
        {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        rec.ops.push(ir::OwnedCommand::SetProgrammablePipeline(*id));
        let active_sets = active_sets(
            rt,
            crate::resources::Pipeline::Graphics(*id),
            &rec.graphics_sets,
        )?;
        for &(index, set) in &active_sets {
            rec.descriptors.push(DescriptorInsertion {
                position: rec.ops.len(),
                index,
                set,
            });
        }
        rec.written_sets
            .extend(active_sets.into_iter().map(|(_, set)| set));
        if vertices != 0 && instances != 0 {
            rec.ops.push(ir::OwnedCommand::Draw {
                vertex_count: vertices,
                first_vertex: first,
            });
        }
        Ok(())
    })
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
            record(command, move |_, _| Err(e));
            return;
        }
    };
    record(command, move |rt, rec| {
        if rec.render.is_some()
            || !(layout == vk::ImageLayout::TRANSFER_SRC_OPTIMAL
                || layout == vk::ImageLayout::GENERAL)
        {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let img = rt
            .resources
            .images
            .get(&image)
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let buf = rt
            .resources
            .buffers
            .get(&buffer)
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        if img.bound.is_none()
            || buf.bound.is_none()
            || !buf.usage.contains(vk::BufferUsageFlags::TRANSFER_DST)
        {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        for r in regions {
            if r.buffer_row_length != 0
                || r.buffer_image_height != 0
                || r.image_offset != vk::Offset3D::default()
                || r.image_extent != img.extent
                || r.image_subresource.aspect_mask != vk::ImageAspectFlags::COLOR
                || r.image_subresource.mip_level != 0
                || r.image_subresource.base_array_layer != 0
                || r.image_subresource.layer_count != 1
            {
                return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
            }
            let size = (img.extent.width as u64) * (img.extent.height as u64) * 4;
            if r.buffer_offset % 4 != 0
                || r.buffer_offset
                    .checked_add(size)
                    .is_none_or(|end| end > buf.size)
            {
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            }
            rec.copies.push(ReadImage {
                position: rec.ops.len(),
                image,
                buffer,
                offset: r.buffer_offset,
                width: img.extent.width,
                height: img.extent.height,
            });
        }
        rec.used_buffers.push(buffer);
        rec.used_images.push(image);
        Ok(())
    })
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
            record(command, move |_, _| Err(e));
            return;
        }
    };
    record(command, move |rt, rec| {
        if rec.render.is_some() {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        let src = rt
            .resources
            .buffers
            .get(&source)
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let dst = rt
            .resources
            .buffers
            .get(&destination)
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        if src.bound.is_none()
            || dst.bound.is_none()
            || !src.usage.contains(vk::BufferUsageFlags::TRANSFER_SRC)
            || !dst.usage.contains(vk::BufferUsageFlags::TRANSFER_DST)
        {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        for r in regions {
            rec.ops.push(ir::OwnedCommand::CopyBufferToBuffer {
                source: src.id,
                source_offset: r.src_offset,
                destination: dst.id,
                destination_offset: r.dst_offset,
                size: r.size,
            });
        }
        rec.used_buffers.extend([source, destination]);
        Ok(())
    })
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
    let m = match slice(memory, memory_count) {
        Ok(v) => v
            .iter()
            .map(|x| (!x.p_next.is_null(), x.src_access_mask, x.dst_access_mask))
            .collect::<Vec<_>>(),
        Err(e) => {
            record(command, move |_, _| Err(e));
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
            record(command, move |_, _| Err(e));
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
            record(command, move |_, _| Err(e));
            return;
        }
    };
    record(command, move |rt, rec| {
        let allowed = vk::PipelineStageFlags::TOP_OF_PIPE
            | vk::PipelineStageFlags::BOTTOM_OF_PIPE
            | vk::PipelineStageFlags::HOST
            | vk::PipelineStageFlags::TRANSFER
            | vk::PipelineStageFlags::COMPUTE_SHADER
            | vk::PipelineStageFlags::VERTEX_SHADER
            | vk::PipelineStageFlags::FRAGMENT_SHADER
            | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
            | vk::PipelineStageFlags::ALL_COMMANDS
            | vk::PipelineStageFlags::ALL_GRAPHICS;
        if rec.render.is_some()
            || source_stage.is_empty()
            || destination_stage.is_empty()
            || !allowed.contains(source_stage | destination_stage)
            || !vk::DependencyFlags::BY_REGION.contains(flags)
        {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        for (extended, src, dst) in m {
            if extended {
                return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
            }
            if src.intersects(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::MEMORY_WRITE) {
                let after = buffer_access(dst)?;
                rec.barriers.push(DeferredBarrier {
                    position: rec.ops.len(),
                    sets: rec.written_sets.clone(),
                    after,
                });
                rec.written_sets.clear();
            } else if !(vk::AccessFlags::TRANSFER_WRITE
                | vk::AccessFlags::HOST_WRITE
                | vk::AccessFlags::MEMORY_WRITE)
                .contains(src)
                || !supported_access(dst)
            {
                return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
            }
        }
        for (extended, src, dst, sq, dq, buffer, offset, size) in b {
            if extended || !(sq == vk::QUEUE_FAMILY_IGNORED && dq == vk::QUEUE_FAMILY_IGNORED) {
                return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
            }
            let buf = rt
                .resources
                .buffers
                .get(&buffer)
                .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
            if offset != 0 || (size != vk::WHOLE_SIZE && size != buf.size) {
                return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
            }
            rec.ops.push(ir::OwnedCommand::ResourceBarrier(
                ir::OwnedResourceBarrier::Buffer {
                    buffer: buf.id,
                    before: buffer_access(src)?,
                    after: buffer_access(dst)?,
                },
            ));
            rec.used_buffers.push(buffer);
        }
        for (extended, _src, _dst, sq, dq, image, old, new, range) in i {
            if extended
                || !(sq == vk::QUEUE_FAMILY_IGNORED && dq == vk::QUEUE_FAMILY_IGNORED)
                || range.aspect_mask != vk::ImageAspectFlags::COLOR
                || range.base_mip_level != 0
                || range.level_count != 1
                || range.base_array_layer != 0
                || range.layer_count != 1
            {
                return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
            }
            let image_obj = rt
                .resources
                .images
                .get(&image)
                .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
            let after = texture_access(new)?;
            if old != vk::ImageLayout::UNDEFINED {
                rec.ops.push(ir::OwnedCommand::ResourceBarrier(
                    ir::OwnedResourceBarrier::Texture {
                        texture: image_obj.id,
                        before: texture_access(old)?,
                        after,
                    },
                ));
            }
            rec.used_images.push(image);
        }
        Ok(())
    })
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
        | vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
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
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => Ok(ir::TextureAccess::RenderAttachment),
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => Ok(ir::TextureAccess::CopySource),
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => Ok(ir::TextureAccess::CopyDestination),
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
    for submit in submits {
        if !submit.p_next.is_null()
            || submit.wait_semaphore_count != 0
            || submit.signal_semaphore_count != 0
        {
            return vk::Result::ERROR_FEATURE_NOT_PRESENT;
        }
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
    for &id in &commands {
        if !driver(id, Kind::Command).is_ok_and(|v| Arc::ptr_eq(&v, &d)) {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        }
    }
    status(call(&d, move |rt| {
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
        let recordings = commands
            .iter()
            .map(|id| {
                rt.commands
                    .get(id)
                    .filter(|r| r.state == RecordingState::Executable)
                    .cloned()
                    .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)
            })
            .collect::<VkResult<Vec<_>>>()?;
        // Validate all object references before any GPU acceptance.
        for rec in &recordings {
            validate_objects(rt, rec)?;
        }
        if let Some(state) = &fence_state {
            state.store(2, Ordering::Release);
        }
        for rec in &recordings {
            if let Err(error) = execute(rt, rec) {
                if error == vk::Result::ERROR_DEVICE_LOST {
                    rt.lost = true;
                }
                if let Some(state) = &fence_state {
                    state.store(0, Ordering::Release);
                }
                return Err(error);
            }
        }
        for id in &commands {
            if let Some(rec) = rt.commands.get_mut(id)
                && rec.one_time
            {
                rec.state = RecordingState::Invalid;
            }
        }
        // QueueSubmit is deliberately synchronous. Only actual successful SGFX
        // retirement plus coherent CPU readback permits this transition.
        if let Some(state) = fence_state {
            state.store(1, Ordering::Release);
        }
        Ok(())
    }))
}
fn validate_objects(rt: &Runtime, rec: &Recording) -> VkResult<()> {
    if rec
        .bound_sets
        .iter()
        .any(|set| !rt.resources.descriptor_sets.contains_key(set))
    {
        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
    }
    if rec.used_framebuffers.iter().any(|f| {
        rt.resources
            .framebuffers
            .get(f)
            .is_none_or(|fb| !rt.resources.views.contains_key(&fb.view))
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
            .any(|i| rt.resources.images.get(i).is_none_or(|i| i.bound.is_none()))
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
            let crate::resources::DescriptorBinding::Buffer { buffer, .. } = binding;
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
fn submit_owned(rt: &mut Runtime, ops: Vec<ir::OwnedCommand>) -> VkResult<()> {
    if ops.is_empty() {
        return Ok(());
    }
    let owned = ir::OwnedCommandBuffer::new(ops);
    let commands = owned.record(&rt.table).map_err(crate::resources::failure)?;
    match rt.queue.submit_tracked(&mut rt.cache, &commands) {
        Ok(receipt) => match receipt.wait(None) {
            Ok(CompletionStatus::Complete) => Ok(()),
            _ => {
                rt.lost = true;
                Err(vk::Result::ERROR_DEVICE_LOST)
            }
        },
        Err(_) => {
            rt.lost = true;
            Err(vk::Result::ERROR_DEVICE_LOST)
        }
    }
}
fn execute(rt: &mut Runtime, rec: &Recording) -> VkResult<()> {
    let mut used = rec.used_buffers.clone();
    for insertion in &rec.descriptors {
        let set = rt
            .resources
            .descriptor_sets
            .get(&insertion.set)
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        for binding in set.bindings.values() {
            let crate::resources::DescriptorBinding::Buffer { buffer, .. } = binding;
            if !used.contains(buffer) {
                used.push(*buffer);
            }
        }
    }
    used.sort_unstable_by_key(|b| b.as_raw());
    used.dedup();
    let mut ops = Vec::new();
    // Coherent host memory is stable until this synchronous call returns.
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
    for position in 0..=rec.ops.len() {
        for d in rec.descriptors.iter().filter(|d| d.position == position) {
            let group = rt.resources.descriptor_group(&rt.table, d.set)?;
            ops.push(ir::OwnedCommand::SetBindGroup {
                index: d.index,
                bind_group: group,
            });
        }
        for barrier in rec.barriers.iter().filter(|b| b.position == position) {
            let mut ids = Vec::new();
            for set in &barrier.sets {
                let set = rt
                    .resources
                    .descriptor_sets
                    .get(set)
                    .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                for binding in set.bindings.values() {
                    let crate::resources::DescriptorBinding::Buffer { buffer, .. } = binding;
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
        for copy in rec.copies.iter().filter(|c| c.position == position) {
            submit_owned(rt, std::mem::take(&mut ops))?;
            let image_id = rt
                .resources
                .images
                .get(&copy.image)
                .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?
                .id;
            let bytes = rt
                .cache
                .read_texture(image_id)
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
    submit_owned(rt, ops)?;
    // Stage GPU-written bytes back into the exact allocation returned by MapMemory.
    for handle in &used {
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
    macro_rules! entry {
        ($function:ident) => {
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
        b"vkCmdDispatch" => entry!(cmd_dispatch),
        b"vkCmdDraw" => entry!(cmd_draw),
        b"vkCmdBeginRenderPass" => entry!(cmd_begin_render_pass),
        b"vkCmdEndRenderPass" => entry!(cmd_end_render_pass),
        b"vkCmdCopyImageToBuffer" => entry!(cmd_copy_image_to_buffer),
        b"vkCmdCopyBuffer" => entry!(cmd_copy_buffer),
        b"vkCmdPipelineBarrier" => entry!(cmd_pipeline_barrier),
        b"vkCreateFence" => entry!(create_fence),
        b"vkDestroyFence" => entry!(destroy_fence),
        b"vkResetFences" => entry!(reset_fences),
        b"vkGetFenceStatus" => entry!(get_fence_status),
        b"vkWaitForFences" => entry!(wait_for_fences),
        _ => crate::resources::lookup(name).or_else(|| crate::images::lookup(name)),
    }
}

/// Descriptor indexing is not exposed. Updating a referenced set invalidates
/// every recording or executable command buffer under Vulkan 1.0 rules.
pub(crate) fn invalidate_descriptor_sets(rt: &mut Runtime, sets: &[vk::DescriptorSet]) {
    for rec in rt.commands.values_mut() {
        if sets.iter().any(|s| {
            rec.bound_sets.contains(s)
                || rec
                    .compute_sets
                    .values()
                    .chain(rec.graphics_sets.values())
                    .any(|v| v == s)
                || rec.descriptors.iter().any(|d| d.set == *s)
        }) {
            rec.fail(vk::Result::ERROR_INITIALIZATION_FAILED);
            if rec.state == RecordingState::Executable {
                rec.state = RecordingState::Invalid;
            }
        }
    }
}

fn active_sets(
    rt: &Runtime,
    pipeline: crate::resources::Pipeline,
    sets: &BTreeMap<u32, vk::DescriptorSet>,
) -> VkResult<Vec<(u32, vk::DescriptorSet)>> {
    let layout = match pipeline {
        crate::resources::Pipeline::Compute(id) => rt
            .table
            .compute_pipeline(
                rt.table
                    .compute_pipeline_ref(id)
                    .map_err(crate::resources::failure)?,
            )
            .map_err(crate::resources::failure)?
            .layout()
            .clone(),
        crate::resources::Pipeline::Graphics(id) => rt
            .table
            .programmable_render_pipeline(
                rt.table
                    .programmable_render_pipeline_ref(id)
                    .map_err(crate::resources::failure)?,
            )
            .map_err(crate::resources::failure)?
            .layout()
            .clone(),
    };
    let mut active = Vec::new();
    for (index, expected) in layout.bind_groups().iter().enumerate() {
        if expected.entries().is_empty() {
            continue;
        }
        let set = *sets
            .get(&(index as u32))
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        if rt
            .resources
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
    for rec in rt.commands.values_mut() {
        if !rec.used_buffers.is_empty()
            || !rec.used_images.is_empty()
            || !rec.used_pipelines.is_empty()
            || !rec.descriptors.is_empty()
        {
            rec.state = RecordingState::Invalid;
            rec.fail(vk::Result::ERROR_INITIALIZATION_FAILED);
            rec.ops.clear();
            rec.descriptors.clear();
            rec.copies.clear();
            rec.barriers.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fence_queries_do_not_queue_behind_a_busy_device_worker() {
        let (sender, _intentionally_unserviced_receiver) = mpsc::channel();
        let d = Arc::new(Driver {
            sender,
            thread: Mutex::new(None),
            queue: AtomicU64::new(0),
            fences: Default::default(),
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
    fn loader_device_chain_accepts_only_transport_records() {
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
}
