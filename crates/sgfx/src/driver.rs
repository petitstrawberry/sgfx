//! Common execution facade used by API frontends.
//!
//! Concrete backends are selected and owned here. Frontends see adapters,
//! devices, resource caches, queues, and completion receipts without importing
//! WGPU, Scarlet GPU transport, or a backend crate.

use alloc::{rc::Rc, string::String, vec::Vec};
use core::{
    fmt,
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};

use sgfx_core::backend::{Completion, CompletionStatus, SubmitError};

use crate::{BackendKind, BackendPreference, Error, Result, ir};

/// Stable device category reported by an SGFX adapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceType {
    /// Integrated graphics processor.
    Integrated,
    /// Discrete graphics processor.
    Discrete,
    /// Software rasterizer or compute implementation.
    Cpu,
    /// Virtual graphics processor.
    Virtual,
    /// The backend did not provide a more specific category.
    Other,
}

/// Backend-neutral identity of one discovered adapter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterInfo {
    name: String,
    vendor_id: u32,
    device_id: u32,
    device_type: DeviceType,
    backend: BackendKind,
}

impl AdapterInfo {
    /// Human-readable device name supplied by the selected backend.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// PCI-style vendor identifier when known, otherwise zero.
    pub const fn vendor_id(&self) -> u32 {
        self.vendor_id
    }

    /// PCI-style device identifier when known, otherwise zero.
    pub const fn device_id(&self) -> u32 {
        self.device_id
    }

    /// Portable device category.
    pub const fn device_type(&self) -> DeviceType {
        self.device_type
    }

    /// Complete backend selected for this adapter.
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }
}

/// Resource and shader limits enforced by a complete SGFX backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    pub max_image_dimension_2d: u32,
    pub max_uniform_buffer_range: u32,
    pub max_storage_buffer_range: u32,
    pub max_bound_descriptor_sets: u32,
    pub max_uniform_buffers_per_stage: u32,
    pub max_storage_buffers_per_stage: u32,
    pub max_vertex_attributes: u32,
    pub max_vertex_buffers: u32,
    pub max_vertex_buffer_stride: u32,
    pub max_inter_stage_components: u32,
    pub max_color_attachments: u32,
    pub max_compute_shared_memory_size: u32,
    pub max_compute_work_group_count: [u32; 3],
    pub max_compute_work_group_invocations: u32,
    pub max_compute_work_group_size: [u32; 3],
    pub min_uniform_buffer_offset_alignment: u32,
    pub min_storage_buffer_offset_alignment: u32,
    pub max_buffer_size: u64,
}

/// Capabilities of one adapter after intersecting backend implementation and
/// physical-device support.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capabilities {
    graphics: bool,
    compute: bool,
    transfer: bool,
    programmable_graphics: bool,
    vertex_buffers: bool,
    index_buffers: bool,
    uniform_buffers: bool,
    storage_buffers: bool,
    rgba8_color_attachment: bool,
    bgra8_color_attachment: bool,
    depth32_attachment: bool,
    image_readback: bool,
    limits: Limits,
}

impl Capabilities {
    pub const fn supports_graphics(&self) -> bool {
        self.graphics
    }

    pub const fn supports_compute(&self) -> bool {
        self.compute
    }

    pub const fn supports_transfer(&self) -> bool {
        self.transfer
    }

    pub const fn supports_programmable_graphics(&self) -> bool {
        self.programmable_graphics
    }

    pub const fn supports_vertex_buffers(&self) -> bool {
        self.vertex_buffers
    }

    pub const fn supports_index_buffers(&self) -> bool {
        self.index_buffers
    }

    pub const fn supports_uniform_buffers(&self) -> bool {
        self.uniform_buffers
    }

    pub const fn supports_storage_buffers(&self) -> bool {
        self.storage_buffers
    }

    pub const fn supports_rgba8_color_attachment(&self) -> bool {
        self.rgba8_color_attachment
    }

    pub const fn supports_bgra8_color_attachment(&self) -> bool {
        self.bgra8_color_attachment
    }

    pub const fn supports_depth32_attachment(&self) -> bool {
        self.depth32_attachment
    }

    pub const fn supports_image_readback(&self) -> bool {
        self.image_readback
    }

    pub const fn limits(&self) -> Limits {
        self.limits
    }
}

/// One physical adapter discovered by a compiled complete backend.
#[derive(Clone)]
pub struct Adapter {
    ordinal: u32,
    info: AdapterInfo,
    capabilities: Capabilities,
    backend: AdapterBackend,
}

impl fmt::Debug for Adapter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Adapter")
            .field("ordinal", &self.ordinal)
            .field("info", &self.info)
            .field("capabilities", &self.capabilities)
            .finish_non_exhaustive()
    }
}

impl Adapter {
    /// Ordinal stable for the lifetime of the discovering [`Instance`].
    pub const fn ordinal(&self) -> u32 {
        self.ordinal
    }

    /// Actual backend-provided adapter identity.
    pub const fn info(&self) -> &AdapterInfo {
        &self.info
    }

    /// Capabilities used for frontend feature and limit reporting.
    pub const fn capabilities(&self) -> Capabilities {
        self.capabilities
    }

    /// Open this exact adapter and create its complete execution environment.
    pub fn create_device(&self) -> Result<Device> {
        match &self.backend {
            #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
            AdapterBackend::Wgpu(adapter) => create_wgpu_device(adapter),
            #[cfg(all(target_os = "scarlet", feature = "backend-scarlet-virgl"))]
            AdapterBackend::ScarletVirgl(adapter) => create_virgl_device(adapter),
        }
    }
}

#[derive(Clone)]
enum AdapterBackend {
    #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
    Wgpu(wgpu::Adapter),
    #[cfg(all(target_os = "scarlet", feature = "backend-scarlet-virgl"))]
    ScarletVirgl(ScarletAdapter),
}

/// A snapshot of adapters available to the configured SGFX backend set.
pub struct Instance {
    adapters: Vec<Adapter>,
}

impl Instance {
    /// Discover adapters using the process backend preference.
    pub fn new() -> Result<Self> {
        Self::with_preference(BackendPreference::from_environment()?)
    }

    /// Discover adapters from the compiled backend set.
    pub fn with_preference(preference: BackendPreference) -> Result<Self> {
        let mut adapters = Vec::new();
        #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
        if matches!(
            preference,
            BackendPreference::Auto | BackendPreference::Wgpu
        ) {
            adapters.extend(discover_wgpu_adapters());
        }
        #[cfg(all(target_os = "scarlet", feature = "backend-scarlet-virgl"))]
        if matches!(
            preference,
            BackendPreference::Auto | BackendPreference::ScarletVirgl
        ) {
            adapters.extend(discover_virgl_adapters());
        }
        if !matches!(preference, BackendPreference::Auto) && adapters.is_empty() {
            let kind = match preference {
                BackendPreference::Auto => unreachable!(),
                BackendPreference::Wgpu => BackendKind::Wgpu,
                BackendPreference::Metal => BackendKind::Metal,
                BackendPreference::ScarletVirgl => BackendKind::ScarletVirgl,
                BackendPreference::ScarletAdreno => BackendKind::ScarletAdreno,
            };
            return Err(Error::BackendUnavailable(kind));
        }
        for (ordinal, adapter) in adapters.iter_mut().enumerate() {
            adapter.ordinal = ordinal as u32;
        }
        Ok(Self { adapters })
    }

    /// Actual adapters found during instance construction.
    pub fn adapters(&self) -> &[Adapter] {
        &self.adapters
    }
}

/// Logical device context selected through one [`Adapter`].
pub struct Device {
    id: usize,
    backend: DeviceBackend,
}

enum DeviceBackend {
    #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
    Wgpu(Rc<sgfx_backend_wgpu::Context>),
    #[cfg(all(target_os = "scarlet", feature = "backend-scarlet-virgl"))]
    ScarletVirgl(Rc<sgfx_backend_scarlet_virgl::Context>),
}

impl Device {
    /// Create a backend materialization cache for one logical resource table.
    pub fn create_resources(&self, table: Rc<ir::ResourceTable>) -> Result<Resources> {
        match &self.backend {
            #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
            DeviceBackend::Wgpu(context) => Ok(Resources {
                device_id: self.id,
                backend: ResourcesBackend::Wgpu(context.create_resources(table)),
            }),
            #[cfg(all(target_os = "scarlet", feature = "backend-scarlet-virgl"))]
            DeviceBackend::ScarletVirgl(context) => context
                .create_ir_resources(table)
                .map(|resources| Resources {
                    device_id: self.id,
                    backend: ResourcesBackend::ScarletVirgl(resources),
                })
                .map_err(Error::ScarletVirglIr),
        }
    }

    /// Create a queue tied to this device context.
    pub fn create_queue(&self) -> Result<Queue> {
        match &self.backend {
            #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
            DeviceBackend::Wgpu(context) => Ok(Queue {
                device_id: self.id,
                backend: QueueBackend::Wgpu(context.create_queue()),
            }),
            #[cfg(all(target_os = "scarlet", feature = "backend-scarlet-virgl"))]
            DeviceBackend::ScarletVirgl(context) => context
                .create_queue()
                .map(|queue| Queue {
                    device_id: self.id,
                    backend: QueueBackend::ScarletVirgl {
                        queue,
                        context: Rc::clone(context),
                    },
                })
                .map_err(Error::ScarletVirglHandle),
        }
    }
}

/// Backend-owned materialization cache for one SGFX resource table.
pub struct Resources {
    device_id: usize,
    backend: ResourcesBackend,
}

enum ResourcesBackend {
    #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
    Wgpu(sgfx_backend_wgpu::Resources),
    #[cfg(all(target_os = "scarlet", feature = "backend-scarlet-virgl"))]
    ScarletVirgl(sgfx_backend_scarlet_virgl::IrResources),
}

impl Resources {
    pub fn validate_shader_module(&mut self, id: ir::ShaderModuleId) -> Result<()> {
        match &mut self.backend {
            #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
            ResourcesBackend::Wgpu(resources) => {
                resources.validate_shader_module(id).map_err(Error::Wgpu)
            }
            #[cfg(all(target_os = "scarlet", feature = "backend-scarlet-virgl"))]
            ResourcesBackend::ScarletVirgl(resources) => resources
                .validate_shader_module(id)
                .map_err(Error::ScarletVirglIr),
        }
    }

    pub fn validate_programmable_render_pipeline(
        &mut self,
        id: ir::ProgrammableRenderPipelineId,
    ) -> Result<()> {
        match &mut self.backend {
            #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
            ResourcesBackend::Wgpu(resources) => resources
                .validate_programmable_render_pipeline(id)
                .map_err(Error::Wgpu),
            #[cfg(all(target_os = "scarlet", feature = "backend-scarlet-virgl"))]
            ResourcesBackend::ScarletVirgl(resources) => resources
                .validate_programmable_render_pipeline(id)
                .map_err(Error::ScarletVirglIr),
        }
    }

    pub fn validate_compute_pipeline(&mut self, id: ir::ComputePipelineId) -> Result<()> {
        match &mut self.backend {
            #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
            ResourcesBackend::Wgpu(resources) => {
                resources.validate_compute_pipeline(id).map_err(Error::Wgpu)
            }
            #[cfg(all(target_os = "scarlet", feature = "backend-scarlet-virgl"))]
            ResourcesBackend::ScarletVirgl(resources) => resources
                .validate_compute_pipeline(id)
                .map_err(Error::ScarletVirglIr),
        }
    }

    /// Blocking readback used by explicit frontend map/copy operations.
    pub fn read_buffer(&mut self, id: ir::BufferId, offset: u64, size: u64) -> Result<Vec<u8>> {
        match &mut self.backend {
            #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
            ResourcesBackend::Wgpu(resources) => {
                resources.read_buffer(id, offset, size).map_err(Error::Wgpu)
            }
            #[cfg(all(target_os = "scarlet", feature = "backend-scarlet-virgl"))]
            ResourcesBackend::ScarletVirgl(resources) => resources
                .read_buffer(id, offset, size)
                .map_err(Error::ScarletVirglIr),
        }
    }
}

/// Queue executing canonical SGFX command buffers through its selected backend.
pub struct Queue {
    device_id: usize,
    backend: QueueBackend,
}

enum QueueBackend {
    #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
    Wgpu(sgfx_backend_wgpu::Queue),
    #[cfg(all(target_os = "scarlet", feature = "backend-scarlet-virgl"))]
    ScarletVirgl {
        queue: sgfx_backend_scarlet_virgl::Queue,
        context: Rc<sgfx_backend_scarlet_virgl::Context>,
    },
}

impl Queue {
    /// Submit without waiting for GPU completion.
    pub fn submit<'r, 'data>(
        &self,
        resources: &mut Resources,
        commands: &ir::CommandBuffer<'r, 'data>,
    ) -> core::result::Result<Submission, SubmitError<Error, Submission>> {
        if self.device_id != resources.device_id {
            return Err(SubmitError::Rejected(Error::ResourceDeviceMismatch));
        }
        #[allow(unreachable_patterns)]
        match (&self.backend, &mut resources.backend) {
            #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
            (QueueBackend::Wgpu(queue), ResourcesBackend::Wgpu(resources)) => queue
                .submit_tracked(resources, commands)
                .map(|receipt| Submission {
                    backend: SubmissionBackend::Wgpu(receipt),
                })
                .map_err(|error| {
                    error.map(Error::Wgpu, |receipt| Submission {
                        backend: SubmissionBackend::Wgpu(receipt),
                    })
                }),
            #[cfg(all(target_os = "scarlet", feature = "backend-scarlet-virgl"))]
            (
                QueueBackend::ScarletVirgl { queue, context },
                ResourcesBackend::ScarletVirgl(resources),
            ) => queue
                .submit_ir_async(context, resources, commands)
                .map(|receipt| Submission {
                    backend: SubmissionBackend::ScarletVirgl(receipt),
                })
                .map_err(|error| {
                    error.map(Error::ScarletVirglIr, |receipt| Submission {
                        backend: SubmissionBackend::ScarletVirgl(receipt),
                    })
                }),
            _ => Err(SubmitError::Rejected(Error::ResourceDeviceMismatch)),
        }
    }

    /// Read a color texture after the caller has established completion.
    pub fn read_texture(&self, resources: &mut Resources, id: ir::TextureId) -> Result<Vec<u8>> {
        if self.device_id != resources.device_id {
            return Err(Error::ResourceDeviceMismatch);
        }
        #[allow(unreachable_patterns)]
        match (&self.backend, &mut resources.backend) {
            #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
            (QueueBackend::Wgpu(_), ResourcesBackend::Wgpu(resources)) => {
                resources.read_texture(id).map_err(Error::Wgpu)
            }
            #[cfg(all(target_os = "scarlet", feature = "backend-scarlet-virgl"))]
            (
                QueueBackend::ScarletVirgl { context, .. },
                ResourcesBackend::ScarletVirgl(resources),
            ) => context
                .read_texture(resources, id)
                .map_err(Error::ScarletVirglIr),
            _ => Err(Error::ResourceDeviceMismatch),
        }
    }
}

/// Owned backend-neutral completion receipt.
#[derive(Clone)]
pub struct Submission {
    backend: SubmissionBackend,
}

#[derive(Clone)]
enum SubmissionBackend {
    #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
    Wgpu(sgfx_backend_wgpu::Submission),
    #[cfg(all(target_os = "scarlet", feature = "backend-scarlet-virgl"))]
    ScarletVirgl(sgfx_backend_scarlet_virgl::Submission),
}

impl fmt::Debug for Submission {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.backend {
            #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
            SubmissionBackend::Wgpu(receipt) => receipt.fmt(formatter),
            #[cfg(all(target_os = "scarlet", feature = "backend-scarlet-virgl"))]
            SubmissionBackend::ScarletVirgl(receipt) => receipt.fmt(formatter),
        }
    }
}

impl Completion for Submission {
    type Error = Error;

    fn poll(&self) -> Result<CompletionStatus> {
        match &self.backend {
            #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
            SubmissionBackend::Wgpu(receipt) => receipt.poll().map_err(Error::Wgpu),
            #[cfg(all(target_os = "scarlet", feature = "backend-scarlet-virgl"))]
            SubmissionBackend::ScarletVirgl(receipt) => {
                receipt.poll().map_err(Error::ScarletVirglIr)
            }
        }
    }

    fn wait(&self, timeout: Option<Duration>) -> Result<CompletionStatus> {
        match &self.backend {
            #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
            SubmissionBackend::Wgpu(receipt) => receipt.wait(timeout).map_err(Error::Wgpu),
            #[cfg(all(target_os = "scarlet", feature = "backend-scarlet-virgl"))]
            SubmissionBackend::ScarletVirgl(receipt) => {
                receipt.wait(timeout).map_err(Error::ScarletVirglIr)
            }
        }
    }
}

fn next_device_id() -> usize {
    static NEXT_DEVICE_ID: AtomicUsize = AtomicUsize::new(1);
    NEXT_DEVICE_ID.fetch_add(1, Ordering::Relaxed)
}

#[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
fn discover_wgpu_adapters() -> Vec<Adapter> {
    let backends = if cfg!(target_os = "macos") {
        wgpu::Backends::METAL
    } else {
        // The Vulkan backend is excluded to prevent this ICD from loading itself.
        wgpu::Backends::GL
    };
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends,
        ..Default::default()
    });
    instance
        .enumerate_adapters(backends)
        .into_iter()
        .map(|adapter| {
            let raw_info = adapter.get_info();
            let runtime = wgpu_runtime_limits(adapter.limits());
            let downlevel = adapter.get_downlevel_capabilities();
            let compute = downlevel
                .flags
                .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS);
            let rgba = adapter.get_texture_format_features(wgpu::TextureFormat::Rgba8Unorm);
            let bgra = adapter.get_texture_format_features(wgpu::TextureFormat::Bgra8Unorm);
            let depth = adapter.get_texture_format_features(wgpu::TextureFormat::Depth32Float);
            Adapter {
                ordinal: 0,
                info: AdapterInfo {
                    name: raw_info.name,
                    vendor_id: raw_info.vendor,
                    device_id: raw_info.device,
                    device_type: match raw_info.device_type {
                        wgpu::DeviceType::IntegratedGpu => DeviceType::Integrated,
                        wgpu::DeviceType::DiscreteGpu => DeviceType::Discrete,
                        wgpu::DeviceType::Cpu => DeviceType::Cpu,
                        wgpu::DeviceType::VirtualGpu => DeviceType::Virtual,
                        _ => DeviceType::Other,
                    },
                    backend: BackendKind::Wgpu,
                },
                capabilities: Capabilities {
                    graphics: true,
                    compute,
                    transfer: true,
                    programmable_graphics: true,
                    vertex_buffers: runtime.max_vertex_buffers != 0,
                    index_buffers: true,
                    uniform_buffers: runtime.max_uniform_buffers_per_stage != 0,
                    storage_buffers: compute && runtime.max_storage_buffers_per_stage != 0,
                    rgba8_color_attachment: rgba
                        .allowed_usages
                        .contains(wgpu::TextureUsages::RENDER_ATTACHMENT),
                    bgra8_color_attachment: bgra
                        .allowed_usages
                        .contains(wgpu::TextureUsages::RENDER_ATTACHMENT),
                    depth32_attachment: depth
                        .allowed_usages
                        .contains(wgpu::TextureUsages::RENDER_ATTACHMENT),
                    image_readback: true,
                    limits: runtime,
                },
                backend: AdapterBackend::Wgpu(adapter),
            }
        })
        .collect()
}

#[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
fn wgpu_runtime_limits(adapter: wgpu::Limits) -> Limits {
    let requested = wgpu::Limits::downlevel_defaults().using_resolution(adapter.clone());
    Limits {
        max_image_dimension_2d: requested.max_texture_dimension_2d,
        max_uniform_buffer_range: requested.max_uniform_buffer_binding_size,
        max_storage_buffer_range: requested.max_storage_buffer_binding_size,
        max_bound_descriptor_sets: requested.max_bind_groups,
        max_uniform_buffers_per_stage: requested.max_uniform_buffers_per_shader_stage,
        max_storage_buffers_per_stage: requested.max_storage_buffers_per_shader_stage,
        max_vertex_attributes: requested.max_vertex_attributes,
        max_vertex_buffers: requested.max_vertex_buffers,
        max_vertex_buffer_stride: requested.max_vertex_buffer_array_stride,
        max_inter_stage_components: requested.max_inter_stage_shader_components,
        max_color_attachments: requested.max_color_attachments,
        max_compute_shared_memory_size: requested.max_compute_workgroup_storage_size,
        max_compute_work_group_count: [requested.max_compute_workgroups_per_dimension; 3],
        max_compute_work_group_invocations: requested.max_compute_invocations_per_workgroup,
        max_compute_work_group_size: [
            requested.max_compute_workgroup_size_x,
            requested.max_compute_workgroup_size_y,
            requested.max_compute_workgroup_size_z,
        ],
        min_uniform_buffer_offset_alignment: requested.min_uniform_buffer_offset_alignment,
        min_storage_buffer_offset_alignment: requested.min_storage_buffer_offset_alignment,
        max_buffer_size: requested.max_buffer_size,
    }
}

#[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
fn create_wgpu_device(adapter: &wgpu::Adapter) -> Result<Device> {
    let adapter_limits = adapter.limits();
    let mut required_limits =
        wgpu::Limits::downlevel_defaults().using_resolution(adapter_limits.clone());
    required_limits.max_compute_workgroup_storage_size = adapter_limits
        .max_compute_workgroup_storage_size
        .min(16 * 1024);
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("SGFX common execution device"),
            required_features: wgpu::Features::empty(),
            required_limits,
            memory_hints: wgpu::MemoryHints::MemoryUsage,
        },
        None,
    ))
    .map_err(|_| Error::Wgpu(sgfx_backend_wgpu::Error::DeviceRequest))?;
    let context = sgfx_backend_wgpu::Device::new(device, queue).create_context();
    Ok(Device {
        id: next_device_id(),
        backend: DeviceBackend::Wgpu(Rc::new(context)),
    })
}

#[cfg(all(target_os = "scarlet", feature = "backend-scarlet-virgl"))]
#[derive(Clone)]
struct ScarletAdapter {
    path: String,
    backend_id: Vec<u8>,
}

#[cfg(all(target_os = "scarlet", feature = "backend-scarlet-virgl"))]
fn discover_virgl_adapters() -> Vec<Adapter> {
    use alloc::format;
    use gpu_raw::Gpu;

    let mut adapters = Vec::new();
    for index in 0..16 {
        let path = format!("/dev/gpu{index}");
        let Ok(gpu) = Gpu::open(&path) else {
            continue;
        };
        let Ok(raw) = gpu.query_info() else {
            continue;
        };
        if !sgfx_backend_scarlet_virgl::Device::supports(&raw) {
            continue;
        }
        let capabilities = sgfx_backend_scarlet_virgl::Capabilities::from_query_info(&raw);
        adapters.push(Adapter {
            ordinal: 0,
            info: AdapterInfo {
                name: format!("Scarlet VirGL GPU {index}"),
                vendor_id: 0,
                device_id: 0,
                device_type: DeviceType::Virtual,
                backend: BackendKind::ScarletVirgl,
            },
            capabilities: Capabilities {
                graphics: capabilities.supports_rendering(),
                compute: false,
                transfer: true,
                programmable_graphics: capabilities.supports_programmable_graphics(),
                vertex_buffers: true,
                index_buffers: true,
                uniform_buffers: true,
                storage_buffers: false,
                rgba8_color_attachment: true,
                bgra8_color_attachment: true,
                depth32_attachment: capabilities.supports_depth(),
                image_readback: capabilities.supports_image_readback(),
                limits: Limits {
                    max_image_dimension_2d: 2048,
                    max_uniform_buffer_range: 16 * 1024,
                    max_storage_buffer_range: 0,
                    max_bound_descriptor_sets: 4,
                    max_uniform_buffers_per_stage: 12,
                    max_storage_buffers_per_stage: 0,
                    max_vertex_attributes: 16,
                    max_vertex_buffers: 1,
                    max_vertex_buffer_stride: 2048,
                    max_inter_stage_components: 60,
                    max_color_attachments: 1,
                    max_compute_shared_memory_size: 0,
                    max_compute_work_group_count: [0; 3],
                    max_compute_work_group_invocations: 0,
                    max_compute_work_group_size: [0; 3],
                    min_uniform_buffer_offset_alignment: 16,
                    min_storage_buffer_offset_alignment: 1,
                    max_buffer_size: u64::from(u32::MAX),
                },
            },
            backend: AdapterBackend::ScarletVirgl(ScarletAdapter {
                path,
                backend_id: raw.backend_id_bytes().to_vec(),
            }),
        });
    }
    adapters
}

#[cfg(all(target_os = "scarlet", feature = "backend-scarlet-virgl"))]
fn create_virgl_device(adapter: &ScarletAdapter) -> Result<Device> {
    use gpu_raw::Gpu;

    let gpu = Gpu::open(&adapter.path).map_err(|_| Error::ScarletGpu)?;
    let info = gpu.query_info().map_err(|_| Error::ScarletGpu)?;
    if info.backend_id_bytes() != adapter.backend_id.as_slice() {
        return Err(Error::BackendDeviceMismatch(BackendKind::ScarletVirgl));
    }
    let device = sgfx_backend_scarlet_virgl::Device::from_gpu(gpu, info)
        .map_err(Error::ScarletVirglHandle)?;
    let context = device.create_context().map_err(Error::ScarletVirglHandle)?;
    Ok(Device {
        id: next_device_id(),
        backend: DeviceBackend::ScarletVirgl(Rc::new(context)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_uncompiled_backend_is_rejected() {
        assert!(matches!(
            Instance::with_preference(BackendPreference::Metal),
            Err(Error::BackendUnavailable(BackendKind::Metal))
        ));
    }
}
