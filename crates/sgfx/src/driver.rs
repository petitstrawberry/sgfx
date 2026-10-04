//! Common execution facade used by API frontends.
//!
//! Concrete backends are selected and owned here. Frontends see adapters,
//! devices, resource caches, queues, and completion receipts without importing
//! WGPU, Scarlet GPU transport, or a backend crate.

// The same dispatch code wraps native errors and passes through dynamic errors.
#![cfg_attr(sgfx_dynamic, allow(clippy::useless_conversion))]

use alloc::{rc::Rc, string::String, vec::Vec};
use core::{
    fmt,
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};

use sgfx_core::backend::{Completion, CompletionStatus, SubmitError};

#[cfg(sgfx_dynamic_virgl)]
use crate::dynamic::{PresentationImage as VirglImage, driver::Submission as VirglSubmission};
#[cfg(all(
    not(sgfx_dynamic_virgl),
    any(
        target_os = "scarlet",
        all(target_os = "linux", feature = "scarlet-native-api")
    ),
    any(
        feature = "backend-scarlet-virgl",
        feature = "backend-scarlet-virgl-static",
        sgfx_dynamic_virgl
    )
))]
use crate::virgl::{Image as VirglImage, Submission as VirglSubmission};

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
    pub max_push_constants_size: u32,
    pub max_image_dimension_2d: u32,
    pub max_image_mip_levels: u32,
    pub max_image_array_layers: u32,
    pub max_uniform_buffer_range: u32,
    pub max_storage_buffer_range: u32,
    pub max_bound_descriptor_sets: u32,
    pub max_uniform_buffers_per_stage: u32,
    pub max_storage_buffers_per_stage: u32,
    pub max_storage_images_per_stage: u32,
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
    storage_images: bool,
    typed_texture_views: bool,
    srgb_texture_views: bool,
    srgb_color_attachments: bool,
    extended_vertex_formats: bool,
    rgba8_color_attachment: bool,
    bgra8_color_attachment: bool,
    depth32_attachment: bool,
    image_readback: bool,
    image_blits: bool,
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
    /// Whether RGBA8 write-only storage image bindings execute on this backend.
    pub const fn supports_storage_images(&self) -> bool {
        self.storage_images
    }
    /// Whether typed texture views and sampled depth execute.
    pub const fn supports_typed_texture_views(&self) -> bool {
        self.typed_texture_views
    }
    /// Whether native sRGB sampling and color conversion execute.
    pub const fn supports_srgb_texture_views(&self) -> bool {
        self.srgb_texture_views
    }
    /// Whether sRGB color attachments encode linear shader output on write.
    pub const fn supports_srgb_color_attachments(&self) -> bool {
        self.srgb_color_attachments
    }
    /// Whether integer, half-float and packed signed-normal vertex inputs execute.
    pub const fn supports_extended_vertex_formats(&self) -> bool {
        self.extended_vertex_formats
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
    /// Whether color mip blits execute on this backend.
    pub const fn supports_image_blits(&self) -> bool {
        self.image_blits
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
            #[cfg(all(
                any(
                    target_os = "scarlet",
                    all(target_os = "linux", feature = "scarlet-native-api")
                ),
                any(
                    feature = "backend-scarlet-virgl",
                    feature = "backend-scarlet-virgl-static",
                    sgfx_dynamic_virgl
                )
            ))]
            AdapterBackend::ScarletVirgl(adapter) => create_virgl_device(adapter),
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            AdapterBackend::Dynamic(adapter) => create_dynamic_device(adapter),
        }
    }
}

#[derive(Clone)]
enum AdapterBackend {
    #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
    Wgpu(WgpuAdapter),
    #[cfg(all(
        any(
            target_os = "scarlet",
            all(target_os = "linux", feature = "scarlet-native-api")
        ),
        any(
            feature = "backend-scarlet-virgl",
            feature = "backend-scarlet-virgl-static",
            sgfx_dynamic_virgl
        )
    ))]
    ScarletVirgl(ScarletAdapter),
    #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
    Dynamic(ScarletAdapter),
}

#[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
#[derive(Clone)]
struct WgpuAdapter {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
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
        #[cfg(all(
            not(sgfx_dynamic_virgl),
            any(
                target_os = "scarlet",
                all(target_os = "linux", feature = "scarlet-native-api")
            ),
            any(
                feature = "backend-scarlet-virgl",
                feature = "backend-scarlet-virgl-static"
            )
        ))]
        if matches!(
            preference,
            BackendPreference::Auto | BackendPreference::ScarletVirgl
        ) {
            adapters.extend(discover_virgl_adapters());
        }
        #[cfg(sgfx_dynamic)]
        adapters.extend(discover_dynamic_adapters(preference));
        if !matches!(preference, BackendPreference::Auto) && adapters.is_empty() {
            return Err(Error::BackendUnavailable(
                preference_kind(preference).expect("explicit preference"),
            ));
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
    Wgpu(WgpuDevice),
    #[cfg(all(
        any(
            target_os = "scarlet",
            all(target_os = "linux", feature = "scarlet-native-api")
        ),
        any(
            feature = "backend-scarlet-virgl",
            feature = "backend-scarlet-virgl-static",
            sgfx_dynamic_virgl
        )
    ))]
    ScarletVirgl(Rc<crate::virgl::Context>),
    #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
    Dynamic(Rc<crate::dynamic::Context>),
}

#[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
struct WgpuDevice {
    context: Rc<sgfx_backend_wgpu::Context>,
    _instance: wgpu::Instance,
    _adapter: wgpu::Adapter,
}

impl Device {
    /// Create a backend materialization cache for one logical resource table.
    pub fn create_resources(&self, table: Rc<ir::ResourceTable>) -> Result<Resources> {
        match &self.backend {
            #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
            DeviceBackend::Wgpu(device) => Ok(Resources {
                device_id: self.id,
                backend: ResourcesBackend::Wgpu(device.context.create_resources(table)),
            }),
            #[cfg(all(
                any(
                    target_os = "scarlet",
                    all(target_os = "linux", feature = "scarlet-native-api")
                ),
                any(
                    feature = "backend-scarlet-virgl",
                    feature = "backend-scarlet-virgl-static",
                    sgfx_dynamic_virgl
                )
            ))]
            DeviceBackend::ScarletVirgl(context) => context
                .create_ir_resources(table)
                .map(|resources| Resources {
                    device_id: self.id,
                    backend: ResourcesBackend::ScarletVirgl(resources),
                })
                .map_err(Error::from),
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            DeviceBackend::Dynamic(context) => context
                .create_ir_resources(table)
                .map(|resources| Resources {
                    device_id: self.id,
                    backend: ResourcesBackend::Dynamic(resources),
                })
                .map_err(Error::from),
        }
    }

    /// Create a queue tied to this device context.
    pub fn create_queue(&self) -> Result<Queue> {
        match &self.backend {
            #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
            DeviceBackend::Wgpu(device) => Ok(Queue {
                device_id: self.id,
                backend: QueueBackend::Wgpu(device.context.create_queue()),
            }),
            #[cfg(all(
                any(
                    target_os = "scarlet",
                    all(target_os = "linux", feature = "scarlet-native-api")
                ),
                any(
                    feature = "backend-scarlet-virgl",
                    feature = "backend-scarlet-virgl-static",
                    sgfx_dynamic_virgl
                )
            ))]
            DeviceBackend::ScarletVirgl(context) => context
                .create_queue()
                .map(|queue| Queue {
                    device_id: self.id,
                    backend: QueueBackend::ScarletVirgl {
                        queue,
                        context: Rc::clone(context),
                    },
                })
                .map_err(Error::from),
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            DeviceBackend::Dynamic(context) => context
                .create_queue()
                .map(|queue| Queue {
                    device_id: self.id,
                    backend: QueueBackend::Dynamic {
                        queue,
                        context: Rc::clone(context),
                    },
                })
                .map_err(Error::from),
        }
    }

    /// Create a presentable image on this device.
    #[cfg(any(
        sgfx_dynamic,
        all(target_os = "macos", feature = "backend-wgpu"),
        all(
            any(
                target_os = "scarlet",
                all(target_os = "linux", feature = "scarlet-native-api")
            ),
            any(
                feature = "backend-scarlet-virgl",
                feature = "backend-scarlet-virgl-static",
                sgfx_dynamic_virgl
            )
        )
    ))]
    pub fn create_presentation_image(
        &self,
        width: u32,
        height: u32,
        format: ir::TextureFormat,
    ) -> Result<PresentationImage> {
        match &self.backend {
            #[cfg(all(
                not(any(target_os = "macos", target_os = "scarlet")),
                feature = "backend-wgpu"
            ))]
            DeviceBackend::Wgpu(_) => Err(Error::Wgpu(sgfx_backend_wgpu::Error::Unsupported(
                sgfx_backend_wgpu::UnsupportedFeature::PresentationPlatform,
            ))),
            #[cfg(all(target_os = "macos", feature = "backend-wgpu"))]
            DeviceBackend::Wgpu(device) => device
                .context
                .create_image(width, height, format)
                .map(|image| PresentationImage {
                    device_id: self.id,
                    backend: PresentationImageBackend::Wgpu(image),
                })
                .map_err(Error::Wgpu),
            #[cfg(all(
                any(
                    target_os = "scarlet",
                    all(target_os = "linux", feature = "scarlet-native-api")
                ),
                any(
                    feature = "backend-scarlet-virgl",
                    feature = "backend-scarlet-virgl-static",
                    sgfx_dynamic_virgl
                )
            ))]
            DeviceBackend::ScarletVirgl(context) => {
                if format != ir::TextureFormat::Bgra8Unorm {
                    return Err(Error::ScarletBackendUnsupported);
                }
                context
                    .create_shared_image(width, height)
                    .map(|image| PresentationImage {
                        device_id: self.id,
                        backend: PresentationImageBackend::ScarletVirgl(Rc::new(image)),
                    })
                    .map_err(Error::from)
            }
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            DeviceBackend::Dynamic(context) => {
                if format != ir::TextureFormat::Bgra8Unorm {
                    return Err(Error::ScarletBackendUnsupported);
                }
                context
                    .create_shared_image(width, height)
                    .map(|image| PresentationImage {
                        device_id: self.id,
                        backend: PresentationImageBackend::Dynamic(Rc::new(image)),
                    })
                    .map_err(Error::from)
            }
        }
    }

    /// Bind an existing CAMetalLayer to this Vulkan-selected device.
    ///
    /// # Safety
    ///
    /// `layer` must remain valid until the returned context is dropped.
    #[cfg(all(target_os = "macos", feature = "backend-wgpu"))]
    pub unsafe fn create_metal_window_context(
        &self,
        layer: *mut core::ffi::c_void,
        width: u32,
        height: u32,
        transparent: bool,
    ) -> Result<WindowContext> {
        let DeviceBackend::Wgpu(device) = &self.backend;
        // SAFETY: forwarded from this method's contract.
        unsafe {
            sgfx_backend_wgpu::WindowContext::from_core_animation_layer(
                device._instance.clone(),
                &device._adapter,
                device.context.as_ref().clone(),
                layer,
                width,
                height,
                transparent,
            )
        }
        .map(|context| WindowContext {
            device_id: self.id,
            context,
        })
        .map_err(Error::Wgpu)
    }
}

/// Device-local image used by a platform presentation context.
#[cfg(any(
    sgfx_dynamic,
    all(target_os = "macos", feature = "backend-wgpu"),
    all(
        any(
            target_os = "scarlet",
            all(target_os = "linux", feature = "scarlet-native-api")
        ),
        any(
            feature = "backend-scarlet-virgl",
            feature = "backend-scarlet-virgl-static",
            sgfx_dynamic_virgl
        )
    )
))]
pub struct PresentationImage {
    device_id: usize,
    backend: PresentationImageBackend,
}

#[cfg(any(
    sgfx_dynamic,
    all(target_os = "macos", feature = "backend-wgpu"),
    all(
        any(
            target_os = "scarlet",
            all(target_os = "linux", feature = "scarlet-native-api")
        ),
        any(
            feature = "backend-scarlet-virgl",
            feature = "backend-scarlet-virgl-static",
            sgfx_dynamic_virgl
        )
    )
))]
enum PresentationImageBackend {
    #[cfg(all(target_os = "macos", feature = "backend-wgpu"))]
    Wgpu(alloc::sync::Arc<sgfx_backend_wgpu::Image>),
    #[cfg(all(
        any(
            target_os = "scarlet",
            all(target_os = "linux", feature = "scarlet-native-api")
        ),
        any(
            feature = "backend-scarlet-virgl",
            feature = "backend-scarlet-virgl-static",
            sgfx_dynamic_virgl
        )
    ))]
    ScarletVirgl(Rc<VirglImage>),
    #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
    Dynamic(Rc<crate::dynamic::PresentationImage>),
}

#[cfg(any(
    sgfx_dynamic,
    all(target_os = "macos", feature = "backend-wgpu"),
    all(
        any(
            target_os = "scarlet",
            all(target_os = "linux", feature = "scarlet-native-api")
        ),
        any(
            feature = "backend-scarlet-virgl",
            feature = "backend-scarlet-virgl-static",
            sgfx_dynamic_virgl
        )
    )
))]
impl PresentationImage {
    pub fn width(&self) -> u32 {
        match &self.backend {
            #[cfg(all(target_os = "macos", feature = "backend-wgpu"))]
            PresentationImageBackend::Wgpu(image) => image.width(),
            #[cfg(all(
                any(
                    target_os = "scarlet",
                    all(target_os = "linux", feature = "scarlet-native-api")
                ),
                any(
                    feature = "backend-scarlet-virgl",
                    feature = "backend-scarlet-virgl-static",
                    sgfx_dynamic_virgl
                )
            ))]
            PresentationImageBackend::ScarletVirgl(image) => image.width(),
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            PresentationImageBackend::Dynamic(image) => image.width(),
        }
    }

    pub fn height(&self) -> u32 {
        match &self.backend {
            #[cfg(all(target_os = "macos", feature = "backend-wgpu"))]
            PresentationImageBackend::Wgpu(image) => image.height(),
            #[cfg(all(
                any(
                    target_os = "scarlet",
                    all(target_os = "linux", feature = "scarlet-native-api")
                ),
                any(
                    feature = "backend-scarlet-virgl",
                    feature = "backend-scarlet-virgl-static",
                    sgfx_dynamic_virgl
                )
            ))]
            PresentationImageBackend::ScarletVirgl(image) => image.height(),
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            PresentationImageBackend::Dynamic(image) => image.height(),
        }
    }

    pub fn image_format(&self) -> ir::TextureFormat {
        match &self.backend {
            #[cfg(all(target_os = "macos", feature = "backend-wgpu"))]
            PresentationImageBackend::Wgpu(image) => image.format(),
            #[cfg(all(
                any(
                    target_os = "scarlet",
                    all(target_os = "linux", feature = "scarlet-native-api")
                ),
                any(
                    feature = "backend-scarlet-virgl",
                    feature = "backend-scarlet-virgl-static",
                    sgfx_dynamic_virgl
                )
            ))]
            PresentationImageBackend::ScarletVirgl(_) => ir::TextureFormat::Bgra8Unorm,
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            PresentationImageBackend::Dynamic(_) => ir::TextureFormat::Bgra8Unorm,
        }
    }

    /// Duplicate the Scarlet GPU image capability for another context or process.
    #[cfg(all(
        any(
            target_os = "scarlet",
            all(target_os = "linux", feature = "scarlet-native-api")
        ),
        any(
            feature = "backend-scarlet-virgl",
            feature = "backend-scarlet-virgl-static",
            sgfx_dynamic
        )
    ))]
    pub fn duplicate_shared_handle(&self) -> Result<crate::Handle> {
        match &self.backend {
            #[cfg(any(
                feature = "backend-scarlet-virgl",
                feature = "backend-scarlet-virgl-static",
                sgfx_dynamic_virgl
            ))]
            PresentationImageBackend::ScarletVirgl(image) => {
                image.shared_handle().duplicate().map_err(Error::from)
            }
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            PresentationImageBackend::Dynamic(image) => {
                image.shared_handle().duplicate().map_err(Error::from)
            }
        }
    }
}

/// Backend-neutral owner of a native presentation surface.
#[cfg(all(target_os = "macos", feature = "backend-wgpu"))]
pub struct WindowContext {
    device_id: usize,
    context: sgfx_backend_wgpu::WindowContext,
}

#[cfg(all(target_os = "macos", feature = "backend-wgpu"))]
impl WindowContext {
    /// Present one image created by the same logical device.
    pub fn present(&mut self, image: &PresentationImage) -> Result<()> {
        if self.device_id != image.device_id {
            return Err(Error::ResourceDeviceMismatch);
        }
        let PresentationImageBackend::Wgpu(image) = &image.backend;
        self.context
            .present_image(image.as_ref())
            .map_err(Error::Wgpu)
    }

    /// Reconfigure the drawable extent.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.context.resize(width, height);
    }
}

/// Backend-owned materialization cache for one SGFX resource table.
pub struct Resources {
    device_id: usize,
    backend: ResourcesBackend,
}

// The explicit static comparison retains the native cache inline. Production
// dynamic builds store only the small opaque resource owner.
#[cfg_attr(
    all(sgfx_dynamic, not(sgfx_dynamic_virgl)),
    allow(clippy::large_enum_variant)
)]
enum ResourcesBackend {
    #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
    Wgpu(sgfx_backend_wgpu::Resources),
    #[cfg(all(
        any(
            target_os = "scarlet",
            all(target_os = "linux", feature = "scarlet-native-api")
        ),
        any(
            feature = "backend-scarlet-virgl",
            feature = "backend-scarlet-virgl-static",
            sgfx_dynamic_virgl
        )
    ))]
    ScarletVirgl(crate::virgl::IrResources),
    #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
    Dynamic(crate::dynamic::IrResources),
}

impl Resources {
    /// Release backend materialization for a retired logical texture.
    /// The caller must first wait for submissions using it to complete.
    pub fn release_texture(&mut self, id: ir::TextureId) -> Result<()> {
        match &mut self.backend {
            #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
            ResourcesBackend::Wgpu(resources) => resources.release_texture(id).map_err(Error::Wgpu),
            #[cfg(all(
                any(
                    target_os = "scarlet",
                    all(target_os = "linux", feature = "scarlet-native-api")
                ),
                any(
                    feature = "backend-scarlet-virgl",
                    feature = "backend-scarlet-virgl-static",
                    sgfx_dynamic_virgl
                )
            ))]
            ResourcesBackend::ScarletVirgl(resources) => {
                resources.release_texture(id).map_err(Error::from)
            }
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            ResourcesBackend::Dynamic(resources) => {
                resources.release_texture(id).map_err(Error::from)
            }
        }
    }

    /// Release backend materialization for a retired logical buffer.
    /// The caller must first wait for submissions using it to complete.
    pub fn release_buffer(&mut self, id: ir::BufferId) -> Result<()> {
        match &mut self.backend {
            #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
            ResourcesBackend::Wgpu(resources) => resources.release_buffer(id).map_err(Error::Wgpu),
            #[cfg(all(
                any(
                    target_os = "scarlet",
                    all(target_os = "linux", feature = "scarlet-native-api")
                ),
                any(
                    feature = "backend-scarlet-virgl",
                    feature = "backend-scarlet-virgl-static",
                    sgfx_dynamic_virgl
                )
            ))]
            ResourcesBackend::ScarletVirgl(resources) => {
                resources.release_buffer(id).map_err(Error::from)
            }
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            ResourcesBackend::Dynamic(resources) => {
                resources.release_buffer(id).map_err(Error::from)
            }
        }
    }

    /// Release cached materialization after recorded commands using this bind
    /// group have been consumed. Submitted work retains its own resources, so
    /// no GPU completion wait is required. Retire the table identity afterward.
    pub fn release_bind_group(&mut self, id: ir::BindGroupId) -> Result<()> {
        match &mut self.backend {
            #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
            ResourcesBackend::Wgpu(resources) => {
                resources.release_bind_group(id).map_err(Error::Wgpu)
            }
            #[cfg(all(
                any(
                    target_os = "scarlet",
                    all(target_os = "linux", feature = "scarlet-native-api")
                ),
                any(
                    feature = "backend-scarlet-virgl",
                    feature = "backend-scarlet-virgl-static",
                    sgfx_dynamic_virgl
                )
            ))]
            ResourcesBackend::ScarletVirgl(resources) => {
                resources.release_bind_group(id).map_err(Error::from)
            }
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            ResourcesBackend::Dynamic(resources) => {
                resources.release_bind_group(id).map_err(Error::from)
            }
        }
    }

    /// Map a logical render target to a device-local shareable image.
    #[cfg(any(
        sgfx_dynamic,
        all(target_os = "macos", feature = "backend-wgpu"),
        all(
            any(
                target_os = "scarlet",
                all(target_os = "linux", feature = "scarlet-native-api")
            ),
            any(
                feature = "backend-scarlet-virgl",
                feature = "backend-scarlet-virgl-static",
                sgfx_dynamic_virgl
            )
        )
    ))]
    pub fn map_presentation_image(
        &mut self,
        texture: ir::TextureId,
        image: &PresentationImage,
    ) -> Result<()> {
        if self.device_id != image.device_id {
            return Err(Error::ResourceDeviceMismatch);
        }
        #[allow(unreachable_patterns)]
        match (&mut self.backend, &image.backend) {
            #[cfg(all(
                not(any(target_os = "macos", target_os = "scarlet")),
                feature = "backend-wgpu"
            ))]
            (ResourcesBackend::Wgpu(_), _) => Err(Error::ResourceDeviceMismatch),
            #[cfg(all(target_os = "macos", feature = "backend-wgpu"))]
            (ResourcesBackend::Wgpu(resources), PresentationImageBackend::Wgpu(image)) => resources
                .map_image(texture, alloc::sync::Arc::clone(image))
                .map_err(Error::Wgpu),
            #[cfg(all(
                any(
                    target_os = "scarlet",
                    all(target_os = "linux", feature = "scarlet-native-api")
                ),
                any(
                    feature = "backend-scarlet-virgl",
                    feature = "backend-scarlet-virgl-static",
                    sgfx_dynamic_virgl
                )
            ))]
            (
                ResourcesBackend::ScarletVirgl(resources),
                PresentationImageBackend::ScarletVirgl(image),
            ) => resources
                .map_image(texture, Rc::clone(image))
                .map_err(Error::from),
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            (ResourcesBackend::Dynamic(resources), PresentationImageBackend::Dynamic(image)) => {
                resources
                    .map_image(texture, Rc::clone(image))
                    .map_err(Error::from)
            }
            _ => Err(Error::ResourceDeviceMismatch),
        }
    }

    /// Remove a logical PRESENT texture mapping.
    #[cfg(any(
        sgfx_dynamic,
        all(target_os = "macos", feature = "backend-wgpu"),
        all(
            any(
                target_os = "scarlet",
                all(target_os = "linux", feature = "scarlet-native-api")
            ),
            any(
                feature = "backend-scarlet-virgl",
                feature = "backend-scarlet-virgl-static",
                sgfx_dynamic_virgl
            )
        )
    ))]
    pub fn unmap_presentation_image(&mut self, texture: ir::TextureId) {
        match &mut self.backend {
            #[cfg(all(
                not(any(target_os = "macos", target_os = "scarlet")),
                feature = "backend-wgpu"
            ))]
            ResourcesBackend::Wgpu(_) => {}
            #[cfg(all(target_os = "macos", feature = "backend-wgpu"))]
            ResourcesBackend::Wgpu(resources) => resources.unmap_image(texture),
            #[cfg(all(
                any(
                    target_os = "scarlet",
                    all(target_os = "linux", feature = "scarlet-native-api")
                ),
                any(
                    feature = "backend-scarlet-virgl",
                    feature = "backend-scarlet-virgl-static",
                    sgfx_dynamic_virgl
                )
            ))]
            ResourcesBackend::ScarletVirgl(resources) => {
                let _ = resources.unmap_image(texture);
            }
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            ResourcesBackend::Dynamic(resources) => {
                let _ = resources.unmap_image(texture);
            }
        }
    }

    pub fn validate_shader_module(&mut self, id: ir::ShaderModuleId) -> Result<()> {
        match &mut self.backend {
            #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
            ResourcesBackend::Wgpu(resources) => {
                resources.validate_shader_module(id).map_err(Error::Wgpu)
            }
            #[cfg(all(
                any(
                    target_os = "scarlet",
                    all(target_os = "linux", feature = "scarlet-native-api")
                ),
                any(
                    feature = "backend-scarlet-virgl",
                    feature = "backend-scarlet-virgl-static",
                    sgfx_dynamic_virgl
                )
            ))]
            ResourcesBackend::ScarletVirgl(resources) => {
                resources.validate_shader_module(id).map_err(Error::from)
            }
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            ResourcesBackend::Dynamic(resources) => {
                resources.validate_shader_module(id).map_err(Error::from)
            }
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
            #[cfg(all(
                any(
                    target_os = "scarlet",
                    all(target_os = "linux", feature = "scarlet-native-api")
                ),
                any(
                    feature = "backend-scarlet-virgl",
                    feature = "backend-scarlet-virgl-static",
                    sgfx_dynamic_virgl
                )
            ))]
            ResourcesBackend::ScarletVirgl(resources) => resources
                .validate_programmable_render_pipeline(id)
                .map_err(Error::from),
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            ResourcesBackend::Dynamic(resources) => resources
                .validate_programmable_render_pipeline(id)
                .map_err(Error::from),
        }
    }

    pub fn validate_compute_pipeline(&mut self, id: ir::ComputePipelineId) -> Result<()> {
        match &mut self.backend {
            #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
            ResourcesBackend::Wgpu(resources) => {
                resources.validate_compute_pipeline(id).map_err(Error::Wgpu)
            }
            #[cfg(all(
                any(
                    target_os = "scarlet",
                    all(target_os = "linux", feature = "scarlet-native-api")
                ),
                any(
                    feature = "backend-scarlet-virgl",
                    feature = "backend-scarlet-virgl-static",
                    sgfx_dynamic_virgl
                )
            ))]
            ResourcesBackend::ScarletVirgl(resources) => {
                resources.validate_compute_pipeline(id).map_err(Error::from)
            }
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            ResourcesBackend::Dynamic(resources) => {
                resources.validate_compute_pipeline(id).map_err(Error::from)
            }
        }
    }

    /// Blocking readback used by explicit frontend map/copy operations.
    pub fn read_buffer(&mut self, id: ir::BufferId, offset: u64, size: u64) -> Result<Vec<u8>> {
        match &mut self.backend {
            #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
            ResourcesBackend::Wgpu(resources) => {
                resources.read_buffer(id, offset, size).map_err(Error::Wgpu)
            }
            #[cfg(all(
                any(
                    target_os = "scarlet",
                    all(target_os = "linux", feature = "scarlet-native-api")
                ),
                any(
                    feature = "backend-scarlet-virgl",
                    feature = "backend-scarlet-virgl-static",
                    sgfx_dynamic_virgl
                )
            ))]
            ResourcesBackend::ScarletVirgl(resources) => {
                resources.read_buffer(id, offset, size).map_err(Error::from)
            }
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            ResourcesBackend::Dynamic(resources) => {
                resources.read_buffer(id, offset, size).map_err(Error::from)
            }
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
    #[cfg(all(
        any(
            target_os = "scarlet",
            all(target_os = "linux", feature = "scarlet-native-api")
        ),
        any(
            feature = "backend-scarlet-virgl",
            feature = "backend-scarlet-virgl-static",
            sgfx_dynamic_virgl
        )
    ))]
    ScarletVirgl {
        queue: crate::virgl::Queue,
        context: Rc<crate::virgl::Context>,
    },
    #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
    Dynamic {
        queue: crate::dynamic::Queue,
        context: Rc<crate::dynamic::Context>,
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
            #[cfg(all(
                any(
                    target_os = "scarlet",
                    all(target_os = "linux", feature = "scarlet-native-api")
                ),
                any(
                    feature = "backend-scarlet-virgl",
                    feature = "backend-scarlet-virgl-static",
                    sgfx_dynamic_virgl
                )
            ))]
            (
                QueueBackend::ScarletVirgl { queue, context },
                ResourcesBackend::ScarletVirgl(resources),
            ) => queue
                .submit_ir_async(context, resources, commands)
                .map(|receipt| Submission {
                    backend: SubmissionBackend::ScarletVirgl(receipt),
                })
                .map_err(|error| {
                    error.map(Error::from, |receipt| Submission {
                        backend: SubmissionBackend::ScarletVirgl(receipt),
                    })
                }),
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            (QueueBackend::Dynamic { queue, context }, ResourcesBackend::Dynamic(resources)) => {
                queue
                    .submit_ir_async(context, resources, commands)
                    .map(|receipt| Submission {
                        backend: SubmissionBackend::Dynamic(receipt),
                    })
                    .map_err(|error| {
                        error.map(Error::from, |receipt| Submission {
                            backend: SubmissionBackend::Dynamic(receipt),
                        })
                    })
            }
            _ => Err(SubmitError::Rejected(Error::ResourceDeviceMismatch)),
        }
    }

    /// Read a color texture after the caller has established completion.
    pub fn read_texture(&self, resources: &mut Resources, id: ir::TextureId) -> Result<Vec<u8>> {
        self.read_texture_mip(resources, id, 0)
    }

    /// Read one color mip after establishing completion, through the selected backend.
    pub fn read_texture_mip(
        &self,
        resources: &mut Resources,
        id: ir::TextureId,
        mip_level: u32,
    ) -> Result<Vec<u8>> {
        if self.device_id != resources.device_id {
            return Err(Error::ResourceDeviceMismatch);
        }
        #[allow(unreachable_patterns)]
        match (&self.backend, &mut resources.backend) {
            #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
            (QueueBackend::Wgpu(_), ResourcesBackend::Wgpu(resources)) => resources
                .read_texture_mip(id, mip_level)
                .map_err(Error::Wgpu),
            #[cfg(all(
                any(
                    target_os = "scarlet",
                    all(target_os = "linux", feature = "scarlet-native-api")
                ),
                any(
                    feature = "backend-scarlet-virgl",
                    feature = "backend-scarlet-virgl-static",
                    sgfx_dynamic_virgl
                )
            ))]
            (
                QueueBackend::ScarletVirgl { context, .. },
                ResourcesBackend::ScarletVirgl(resources),
            ) => {
                if mip_level != 0 {
                    return Err(Error::ScarletBackendUnsupported);
                }
                context.read_texture(resources, id).map_err(Error::from)
            }
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            (QueueBackend::Dynamic { context, .. }, ResourcesBackend::Dynamic(resources)) => {
                if mip_level != 0 {
                    return Err(Error::ScarletBackendUnsupported);
                }
                context.read_texture(resources, id).map_err(Error::from)
            }
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
    #[cfg(all(
        any(
            target_os = "scarlet",
            all(target_os = "linux", feature = "scarlet-native-api")
        ),
        any(
            feature = "backend-scarlet-virgl",
            feature = "backend-scarlet-virgl-static",
            sgfx_dynamic_virgl
        )
    ))]
    ScarletVirgl(VirglSubmission),
    #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
    Dynamic(crate::dynamic::driver::Submission),
}

impl fmt::Debug for Submission {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.backend {
            #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
            SubmissionBackend::Wgpu(receipt) => receipt.fmt(formatter),
            #[cfg(all(
                any(
                    target_os = "scarlet",
                    all(target_os = "linux", feature = "scarlet-native-api")
                ),
                any(
                    feature = "backend-scarlet-virgl",
                    feature = "backend-scarlet-virgl-static",
                    sgfx_dynamic_virgl
                )
            ))]
            SubmissionBackend::ScarletVirgl(receipt) => receipt.fmt(formatter),
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            SubmissionBackend::Dynamic(receipt) => receipt.fmt(formatter),
        }
    }
}

impl Completion for Submission {
    type Error = Error;

    fn poll(&self) -> Result<CompletionStatus> {
        match &self.backend {
            #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
            SubmissionBackend::Wgpu(receipt) => receipt.poll().map_err(Error::Wgpu),
            #[cfg(all(
                any(
                    target_os = "scarlet",
                    all(target_os = "linux", feature = "scarlet-native-api")
                ),
                any(
                    feature = "backend-scarlet-virgl",
                    feature = "backend-scarlet-virgl-static",
                    sgfx_dynamic_virgl
                )
            ))]
            SubmissionBackend::ScarletVirgl(receipt) => receipt.poll().map_err(Error::from),
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            SubmissionBackend::Dynamic(receipt) => receipt.poll().map_err(Error::from),
        }
    }

    fn wait(&self, timeout: Option<Duration>) -> Result<CompletionStatus> {
        match &self.backend {
            #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
            SubmissionBackend::Wgpu(receipt) => receipt.wait(timeout).map_err(Error::Wgpu),
            #[cfg(all(
                any(
                    target_os = "scarlet",
                    all(target_os = "linux", feature = "scarlet-native-api")
                ),
                any(
                    feature = "backend-scarlet-virgl",
                    feature = "backend-scarlet-virgl-static",
                    sgfx_dynamic_virgl
                )
            ))]
            SubmissionBackend::ScarletVirgl(receipt) => receipt.wait(timeout).map_err(Error::from),
            #[cfg(all(sgfx_dynamic, not(sgfx_dynamic_virgl)))]
            SubmissionBackend::Dynamic(receipt) => receipt.wait(timeout).map_err(Error::from),
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
            let mut runtime = wgpu_runtime_limits(adapter.limits());
            runtime.max_push_constants_size =
                if adapter.features().contains(wgpu::Features::PUSH_CONSTANTS) {
                    adapter
                        .limits()
                        .max_push_constant_size
                        .min(ir::MAX_PUSH_CONSTANT_BYTES)
                } else {
                    0
                };
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
                    storage_images: compute
                        && rgba
                            .allowed_usages
                            .contains(wgpu::TextureUsages::STORAGE_BINDING),
                    typed_texture_views: true,
                    srgb_texture_views: true,
                    srgb_color_attachments: true,
                    extended_vertex_formats: true,
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
                    image_blits: true,
                    limits: runtime,
                },
                backend: AdapterBackend::Wgpu(WgpuAdapter {
                    instance: instance.clone(),
                    adapter,
                }),
            }
        })
        .collect()
}

#[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
fn wgpu_runtime_limits(adapter: wgpu::Limits) -> Limits {
    let requested = wgpu::Limits::downlevel_defaults().using_resolution(adapter.clone());
    Limits {
        max_push_constants_size: 0,
        max_image_dimension_2d: requested.max_texture_dimension_2d,
        max_image_mip_levels: requested.max_texture_dimension_2d.ilog2() + 1,
        max_image_array_layers: requested.max_texture_array_layers,
        max_uniform_buffer_range: requested.max_uniform_buffer_binding_size,
        max_storage_buffer_range: requested.max_storage_buffer_binding_size,
        max_bound_descriptor_sets: requested.max_bind_groups,
        max_uniform_buffers_per_stage: requested.max_uniform_buffers_per_shader_stage,
        max_storage_buffers_per_stage: requested.max_storage_buffers_per_shader_stage,
        max_storage_images_per_stage: requested.max_storage_textures_per_shader_stage,
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
fn create_wgpu_device(wgpu_adapter: &WgpuAdapter) -> Result<Device> {
    let adapter = &wgpu_adapter.adapter;
    let adapter_limits = adapter.limits();
    let mut required_limits =
        wgpu::Limits::downlevel_defaults().using_resolution(adapter_limits.clone());
    required_limits.max_compute_workgroup_storage_size = adapter_limits
        .max_compute_workgroup_storage_size
        .min(16 * 1024);
    let required_features = adapter.features() & wgpu::Features::PUSH_CONSTANTS;
    required_limits.max_push_constant_size =
        if required_features.contains(wgpu::Features::PUSH_CONSTANTS) {
            adapter_limits
                .max_push_constant_size
                .min(ir::MAX_PUSH_CONSTANT_BYTES)
        } else {
            0
        };
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("SGFX common execution device"),
            required_features,
            required_limits,
            memory_hints: wgpu::MemoryHints::MemoryUsage,
        },
        None,
    ))
    .map_err(|_| Error::Wgpu(sgfx_backend_wgpu::Error::DeviceRequest))?;
    let context = sgfx_backend_wgpu::Device::new(device, queue).create_context();
    Ok(Device {
        id: next_device_id(),
        backend: DeviceBackend::Wgpu(WgpuDevice {
            context: Rc::new(context),
            _instance: wgpu_adapter.instance.clone(),
            _adapter: adapter.clone(),
        }),
    })
}

#[cfg(all(
    any(
        target_os = "scarlet",
        all(target_os = "linux", feature = "scarlet-native-api")
    ),
    any(
        feature = "backend-scarlet-virgl",
        feature = "backend-scarlet-virgl-static",
        sgfx_dynamic_virgl
    )
))]
#[derive(Clone)]
struct ScarletAdapter {
    path: String,
    backend_id: Vec<u8>,
    #[cfg(sgfx_dynamic)]
    backend_name: String,
    #[cfg(sgfx_dynamic)]
    backend_kind: BackendKind,
}

#[cfg(all(
    not(sgfx_dynamic_virgl),
    any(
        target_os = "scarlet",
        all(target_os = "linux", feature = "scarlet-native-api")
    ),
    any(
        feature = "backend-scarlet-virgl",
        feature = "backend-scarlet-virgl-static",
        sgfx_dynamic_virgl
    )
))]
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
        if !crate::virgl::Device::supports(&raw) {
            continue;
        }
        let capabilities = crate::virgl::Capabilities::from_query_info(&raw);
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
                storage_buffers: capabilities.supports_programmable_graphics(),
                storage_images: false,
                typed_texture_views: capabilities.supports_texture_arrays()
                    && capabilities.supports_depth_sampling(),
                // VirGL on macOS cannot reinterpret GL texture storage, so
                // the native backend decodes sRGB sampled views in TGSI.
                srgb_texture_views: capabilities.supports_programmable_graphics(),
                srgb_color_attachments: false,
                extended_vertex_formats: true,
                rgba8_color_attachment: true,
                bgra8_color_attachment: true,
                depth32_attachment: capabilities.supports_depth(),
                image_readback: capabilities.supports_image_readback(),
                image_blits: capabilities.supports_image_mips(),
                limits: Limits {
                    max_push_constants_size: if capabilities.supports_programmable_graphics() {
                        128
                    } else {
                        0
                    },
                    max_image_dimension_2d: 4096,
                    max_image_array_layers: if capabilities.supports_texture_arrays() {
                        2048
                    } else {
                        1
                    },
                    max_image_mip_levels: if capabilities.supports_image_mips() {
                        13
                    } else {
                        1
                    },
                    max_uniform_buffer_range: 16 * 1024,
                    max_storage_buffer_range: 256 * 1024,
                    max_bound_descriptor_sets: 4,
                    max_uniform_buffers_per_stage: 12,
                    max_storage_buffers_per_stage: 4,
                    max_storage_images_per_stage: 0,
                    max_vertex_attributes: 16,
                    max_vertex_buffers: 8,
                    max_vertex_buffer_stride: 2048,
                    max_inter_stage_components: 60,
                    max_color_attachments: if capabilities.supports_programmable_graphics() {
                        ir::MAX_COLOR_ATTACHMENTS as u32
                    } else {
                        1
                    },
                    max_compute_shared_memory_size: 0,
                    max_compute_work_group_count: [0; 3],
                    max_compute_work_group_invocations: 0,
                    max_compute_work_group_size: [0; 3],
                    min_uniform_buffer_offset_alignment: 16,
                    min_storage_buffer_offset_alignment: 4,
                    max_buffer_size: u64::from(u32::MAX),
                },
            },
            backend: AdapterBackend::ScarletVirgl(ScarletAdapter {
                path,
                backend_id: raw.backend_id_bytes().to_vec(),
                #[cfg(sgfx_dynamic)]
                backend_name: String::from("scarlet-virgl"),
                #[cfg(sgfx_dynamic)]
                backend_kind: BackendKind::ScarletVirgl,
            }),
        });
    }
    adapters
}

#[cfg(all(
    any(
        target_os = "scarlet",
        all(target_os = "linux", feature = "scarlet-native-api")
    ),
    any(
        feature = "backend-scarlet-virgl",
        feature = "backend-scarlet-virgl-static",
        sgfx_dynamic_virgl
    )
))]
fn create_virgl_device(adapter: &ScarletAdapter) -> Result<Device> {
    #[cfg(sgfx_dynamic_virgl)]
    return create_dynamic_device(adapter);

    #[cfg(not(sgfx_dynamic_virgl))]
    {
        use gpu_raw::Gpu;
        let gpu = Gpu::open(&adapter.path).map_err(|_| Error::ScarletGpu)?;
        let info = gpu.query_info().map_err(|_| Error::ScarletGpu)?;
        if info.backend_id_bytes() != adapter.backend_id.as_slice() {
            return Err(Error::BackendDeviceMismatch(BackendKind::ScarletVirgl));
        }
        let device = crate::virgl::Device::from_gpu(gpu, info).map_err(Error::from)?;
        let context = device.create_context().map_err(Error::from)?;
        Ok(Device {
            id: next_device_id(),
            backend: DeviceBackend::ScarletVirgl(Rc::new(context)),
        })
    }
}

fn preference_kind(preference: BackendPreference) -> Option<BackendKind> {
    match preference {
        BackendPreference::Auto => None,
        BackendPreference::Wgpu => Some(BackendKind::Wgpu),
        BackendPreference::Metal => Some(BackendKind::Metal),
        BackendPreference::ScarletVirgl => Some(BackendKind::ScarletVirgl),
        BackendPreference::ScarletAdreno => Some(BackendKind::ScarletAdreno),
        #[cfg(feature = "backend-dynamic")]
        BackendPreference::Other(name) => Some(BackendKind::Other(name)),
    }
}

#[cfg(any(sgfx_dynamic, test))]
fn dynamic_selection(preference: BackendPreference) -> Option<Option<BackendKind>> {
    match preference {
        BackendPreference::Auto => Some(None),
        BackendPreference::Wgpu | BackendPreference::Metal => None,
        _ => Some(preference_kind(preference)),
    }
}

#[cfg(sgfx_dynamic)]
fn discover_dynamic_adapters(preference: BackendPreference) -> Vec<Adapter> {
    use alloc::format;
    use gpu_raw::Gpu;

    let mut adapters = Vec::new();
    let Some(selection) = dynamic_selection(preference) else {
        return adapters;
    };
    // The static comparison feature explicitly owns VirGL selection. Other
    // devices still use manifests and never fall back to a static backend.
    #[cfg(all(
        not(sgfx_dynamic_virgl),
        any(
            feature = "backend-scarlet-virgl",
            feature = "backend-scarlet-virgl-static"
        )
    ))]
    if !dynamic_owns_device(preference, true, false) {
        return adapters;
    }
    for index in 0..16 {
        let path = format!("/dev/gpu{index}");
        let Ok(gpu) = Gpu::open(&path) else { continue };
        let Ok(raw) = gpu.query_info() else { continue };
        #[cfg(all(
            not(sgfx_dynamic_virgl),
            any(
                feature = "backend-scarlet-virgl",
                feature = "backend-scarlet-virgl-static"
            )
        ))]
        if !dynamic_owns_device(preference, true, crate::virgl::Device::supports(&raw)) {
            continue;
        }
        let backend_id = raw.backend_id_bytes().to_vec();
        drop(gpu);
        let Ok(device) = crate::dynamic::Device::open_with_backend(
            &path,
            selection.as_ref().map(BackendKind::as_str),
        ) else {
            continue;
        };
        // The high-level session ABI alone does not make a complete low-level
        // frontend backend. Reject it before advertising an adapter.
        if !device.supports_driver_api() {
            continue;
        }
        let backend = device.backend();
        let backend_name = String::from(device.backend_name());
        let capabilities = dynamic_capabilities(DynamicFeatures::from(device.capabilities()));
        let adapter = ScarletAdapter {
            path,
            backend_id,
            backend_name,
            backend_kind: backend,
        };
        adapters.push(Adapter {
            ordinal: 0,
            info: AdapterInfo {
                name: format!("Scarlet {} GPU {index}", device.backend_name()),
                vendor_id: 0,
                device_id: 0,
                device_type: DeviceType::Other,
                backend,
            },
            capabilities,
            #[cfg(sgfx_dynamic_virgl)]
            backend: AdapterBackend::ScarletVirgl(adapter),
            #[cfg(not(sgfx_dynamic_virgl))]
            backend: AdapterBackend::Dynamic(adapter),
        });
    }
    adapters
}

#[cfg(sgfx_dynamic)]
fn create_dynamic_device(adapter: &ScarletAdapter) -> Result<Device> {
    use gpu_raw::Gpu;
    let gpu = Gpu::open(&adapter.path).map_err(|_| Error::ScarletGpu)?;
    let info = gpu.query_info().map_err(|_| Error::ScarletGpu)?;
    let kind = adapter.backend_kind;
    if info.backend_id_bytes() != adapter.backend_id.as_slice() {
        return Err(Error::BackendDeviceMismatch(kind));
    }
    drop(gpu);
    let device =
        crate::dynamic::Device::open_with_backend(&adapter.path, Some(&adapter.backend_name))?;
    if !device.supports_driver_api() {
        return Err(Error::ScarletBackendUnsupported);
    }
    if device.backend_name() != adapter.backend_name {
        return Err(Error::BackendDeviceMismatch(kind));
    }
    let context = Rc::new(device.create_context()?);
    Ok(Device {
        id: next_device_id(),
        #[cfg(sgfx_dynamic_virgl)]
        backend: DeviceBackend::ScarletVirgl(context),
        #[cfg(not(sgfx_dynamic_virgl))]
        backend: DeviceBackend::Dynamic(context),
    })
}

#[cfg(any(all(sgfx_dynamic, not(sgfx_dynamic_virgl)), test))]
fn dynamic_owns_device(
    preference: BackendPreference,
    static_virgl: bool,
    virgl_compatible: bool,
) -> bool {
    !(static_virgl
        && (matches!(preference, BackendPreference::ScarletVirgl)
            || preference == BackendPreference::Auto && virgl_compatible))
}

#[cfg(any(sgfx_dynamic, test))]
#[derive(Clone, Copy, Default)]
struct DynamicFeatures {
    rendering: bool,
    upload: bool,
    readback: bool,
    programmable: bool,
    arrays: bool,
    depth_sampling: bool,
    mips: bool,
    depth: bool,
    read_only_storage: bool,
    typed_views: bool,
    srgb_views: bool,
    extended_vertex_formats: bool,
    rgba8_attachments: bool,
    blits: bool,
    push_constants_128: bool,
    color_attachments_8: bool,
}

#[cfg(sgfx_dynamic)]
impl From<crate::dynamic::Capabilities> for DynamicFeatures {
    fn from(capabilities: crate::dynamic::Capabilities) -> Self {
        Self {
            rendering: capabilities.supports_rendering(),
            upload: capabilities.supports_image_upload(),
            readback: capabilities.supports_image_readback(),
            programmable: capabilities.supports_programmable_graphics(),
            arrays: capabilities.supports_texture_arrays(),
            depth_sampling: capabilities.supports_depth_sampling(),
            mips: capabilities.supports_image_mips(),
            depth: capabilities.supports_depth(),
            read_only_storage: capabilities.supports_read_only_storage_buffers(),
            typed_views: capabilities.supports_typed_texture_views(),
            srgb_views: capabilities.supports_srgb_texture_views(),
            extended_vertex_formats: capabilities.supports_extended_vertex_formats(),
            rgba8_attachments: capabilities.supports_rgba8_color_attachment(),
            blits: capabilities.supports_image_blits(),
            push_constants_128: capabilities.supports_push_constants_128(),
            color_attachments_8: capabilities.supports_color_attachments_8(),
        }
    }
}

#[cfg(any(sgfx_dynamic, test))]
fn dynamic_capabilities(features: DynamicFeatures) -> Capabilities {
    let programmable = features.rendering && features.programmable;
    // Optional properties come exclusively from the negotiated ABI flags.
    // A manifest name never grants execution features or resource limits.
    let read_only_storage = programmable && features.read_only_storage;
    Capabilities {
        graphics: features.rendering,
        compute: false,
        transfer: features.upload || features.readback,
        programmable_graphics: programmable,
        vertex_buffers: programmable,
        index_buffers: programmable,
        uniform_buffers: programmable,
        storage_buffers: read_only_storage,
        storage_images: false,
        typed_texture_views: features.typed_views && features.arrays && features.depth_sampling,
        srgb_texture_views: programmable && features.srgb_views,
        srgb_color_attachments: false,
        extended_vertex_formats: programmable && features.extended_vertex_formats,
        rgba8_color_attachment: features.rendering && features.rgba8_attachments,
        bgra8_color_attachment: features.rendering,
        depth32_attachment: features.depth,
        image_readback: features.readback,
        image_blits: features.blits,
        limits: Limits {
            max_push_constants_size: if programmable && features.push_constants_128 {
                128
            } else {
                0
            },
            max_image_dimension_2d: 4096,
            max_image_mip_levels: if features.mips { 13 } else { 1 },
            max_image_array_layers: if features.arrays { 2048 } else { 1 },
            max_uniform_buffer_range: if programmable { 16 * 1024 } else { 0 },
            max_storage_buffer_range: if read_only_storage { 256 * 1024 } else { 0 },
            max_bound_descriptor_sets: if programmable { 4 } else { 0 },
            max_uniform_buffers_per_stage: if programmable { 12 } else { 0 },
            max_storage_buffers_per_stage: if read_only_storage { 4 } else { 0 },
            max_storage_images_per_stage: 0,
            max_vertex_attributes: if programmable { 16 } else { 0 },
            max_vertex_buffers: if programmable { 8 } else { 0 },
            max_vertex_buffer_stride: if programmable { 2048 } else { 0 },
            max_inter_stage_components: if programmable { 60 } else { 0 },
            max_color_attachments: if programmable && features.color_attachments_8 {
                8
            } else if features.rendering {
                1
            } else {
                0
            },
            max_compute_shared_memory_size: 0,
            max_compute_work_group_count: [0; 3],
            max_compute_work_group_invocations: 0,
            max_compute_work_group_size: [0; 3],
            min_uniform_buffer_offset_alignment: 16,
            min_storage_buffer_offset_alignment: 4,
            max_buffer_size: u64::from(u32::MAX),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_virgl_comparison_owns_only_its_selected_devices() {
        assert!(!dynamic_owns_device(BackendPreference::Auto, true, true));
        assert!(dynamic_owns_device(BackendPreference::Auto, true, false));
        assert!(!dynamic_owns_device(
            BackendPreference::ScarletVirgl,
            true,
            false
        ));
        assert!(dynamic_owns_device(BackendPreference::Auto, false, true));
        assert!(dynamic_owns_device(
            BackendPreference::ScarletVirgl,
            false,
            true
        ));
        assert!(dynamic_owns_device(
            BackendPreference::ScarletAdreno,
            true,
            true
        ));
        #[cfg(feature = "backend-dynamic")]
        assert!(dynamic_owns_device(
            BackendPreference::parse("third-party-gpu").unwrap(),
            true,
            true
        ));
    }

    #[test]
    fn legacy_flags_do_not_grant_optional_execution_features() {
        let features = DynamicFeatures {
            rendering: true,
            programmable: true,
            upload: true,
            readback: true,
            arrays: true,
            depth_sampling: true,
            mips: true,
            depth: true,
            ..Default::default()
        };
        let capabilities = dynamic_capabilities(features);
        assert!(!capabilities.supports_storage_buffers());
        assert!(!capabilities.supports_typed_texture_views());
        assert!(!capabilities.supports_srgb_texture_views());
        assert!(!capabilities.supports_extended_vertex_formats());
        assert!(!capabilities.supports_rgba8_color_attachment());
        assert!(!capabilities.supports_image_blits());
        assert_eq!(capabilities.limits().max_push_constants_size, 0);
        assert_eq!(capabilities.limits().max_color_attachments, 1);
    }

    #[test]
    fn native_dynamic_selection_preserves_known_preferences() {
        assert_eq!(dynamic_selection(BackendPreference::Auto), Some(None));
        assert_eq!(dynamic_selection(BackendPreference::Wgpu), None);
        assert_eq!(dynamic_selection(BackendPreference::Metal), None);
        for (preference, backend) in [
            (BackendPreference::ScarletVirgl, BackendKind::ScarletVirgl),
            (BackendPreference::ScarletAdreno, BackendKind::ScarletAdreno),
        ] {
            assert_eq!(dynamic_selection(preference), Some(Some(backend)));
        }
    }

    #[cfg(feature = "backend-dynamic")]
    #[test]
    fn native_dynamic_selection_preserves_arbitrary_manifest_names() {
        let name = crate::BackendName::new("third-party-gpu").unwrap();
        let preference = BackendPreference::parse(name.as_str()).unwrap();
        let Some(Some(selected)) = dynamic_selection(preference) else {
            panic!("manifest preference must select a dynamic backend");
        };
        assert_eq!(selected.as_str(), name.as_str());
    }

    #[test]
    fn rendering_flags_do_not_claim_programmability() {
        let capabilities = dynamic_capabilities(DynamicFeatures {
            rendering: true,
            upload: true,
            readback: true,
            ..Default::default()
        });
        assert!(capabilities.supports_graphics());
        assert!(capabilities.supports_transfer());
        assert!(capabilities.supports_bgra8_color_attachment());
        assert!(capabilities.supports_image_readback());
        assert!(!capabilities.supports_compute());
        assert!(!capabilities.supports_programmable_graphics());
        assert!(!capabilities.supports_vertex_buffers());
        assert!(!capabilities.supports_index_buffers());
        assert!(!capabilities.supports_uniform_buffers());
        assert!(!capabilities.supports_storage_buffers());
        assert!(!capabilities.supports_extended_vertex_formats());
        assert!(!capabilities.supports_depth32_attachment());
        assert!(!capabilities.supports_image_blits());
        assert_eq!(capabilities.limits().max_push_constants_size, 0);
        assert_eq!(capabilities.limits().max_vertex_buffers, 0);
        assert_eq!(capabilities.limits().max_color_attachments, 1);
        assert_eq!(capabilities.limits().max_compute_work_group_size, [0; 3]);
    }

    #[test]
    fn dynamic_capabilities_require_each_optional_flag() {
        let empty = dynamic_capabilities(DynamicFeatures::default());
        assert!(!empty.supports_graphics());
        assert!(!empty.supports_transfer());
        assert!(!empty.supports_image_readback());
        assert_eq!(empty.limits().max_color_attachments, 0);
        let features = DynamicFeatures {
            programmable: true,
            arrays: true,
            read_only_storage: true,
            typed_views: true,
            srgb_views: true,
            extended_vertex_formats: true,
            rgba8_attachments: true,
            push_constants_128: true,
            color_attachments_8: true,
            ..Default::default()
        };
        let incomplete = dynamic_capabilities(features);
        assert!(!incomplete.supports_programmable_graphics());
        assert!(!incomplete.supports_typed_texture_views());
        assert!(!incomplete.supports_storage_buffers());
        assert!(!incomplete.supports_srgb_texture_views());
        assert!(!incomplete.supports_extended_vertex_formats());
        assert!(!incomplete.supports_rgba8_color_attachment());
        assert_eq!(incomplete.limits().max_storage_buffer_range, 0);
        assert_eq!(incomplete.limits().max_push_constants_size, 0);
        let capabilities = dynamic_capabilities(DynamicFeatures {
            rendering: true,
            programmable: true,
            arrays: true,
            depth_sampling: true,
            mips: true,
            depth: true,
            read_only_storage: true,
            typed_views: true,
            srgb_views: true,
            extended_vertex_formats: true,
            rgba8_attachments: true,
            blits: true,
            push_constants_128: true,
            color_attachments_8: true,
            ..Default::default()
        });
        assert!(capabilities.supports_programmable_graphics());
        assert!(capabilities.supports_storage_buffers());
        assert!(capabilities.supports_typed_texture_views());
        assert!(capabilities.supports_srgb_texture_views());
        assert!(capabilities.supports_extended_vertex_formats());
        assert!(capabilities.supports_rgba8_color_attachment());
        assert!(capabilities.supports_image_blits());
        assert_eq!(capabilities.limits().max_push_constants_size, 128);
        assert_eq!(
            capabilities.limits().max_color_attachments,
            ir::MAX_COLOR_ATTACHMENTS as u32
        );
    }

    #[test]
    fn explicit_uncompiled_backend_is_rejected() {
        assert!(matches!(
            Instance::with_preference(BackendPreference::Metal),
            Err(Error::BackendUnavailable(BackendKind::Metal))
        ));
    }
}
