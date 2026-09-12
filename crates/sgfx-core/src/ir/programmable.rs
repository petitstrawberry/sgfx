//! Programmable pipeline layouts, bindings and explicit resource access declarations.

use super::*;
use alloc::{string::String, vec::Vec};

/// Maximum descriptor sets in a pipeline layout.
pub const MAX_BIND_GROUPS: usize = 4;
/// Maximum bindings in a descriptor set.
pub const MAX_BINDINGS_PER_GROUP: usize = 16;

/// A shader module and a stage-specific entry point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderEntryPoint {
    module: ShaderModuleId,
    stage: ShaderStage,
    entry_point: String,
}
impl ShaderEntryPoint {
    /// Select an entry point; SPIR-V entry point existence and stage are checked here.
    pub fn new(
        module: ShaderModuleRef<'_>,
        stage: ShaderStage,
        entry_point: String,
    ) -> Result<Self> {
        if !module
            .owner
            .shader_module(module)?
            .has_entry_point(stage, &entry_point)
        {
            return Err(Error::InvalidDescriptor);
        }
        Ok(Self {
            module: module.id(),
            stage,
            entry_point,
        })
    }
    /// Return the owning module's persistent identity.
    pub const fn module(&self) -> ShaderModuleId {
        self.module
    }
    /// Return the selected stage.
    pub const fn stage(&self) -> ShaderStage {
        self.stage
    }
    /// Return the shader entry point name.
    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }
}

/// Supported storage texture access. This subset admits write-only RGBA8 images.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageTextureAccess {
    /// Store texels without reading the image.
    WriteOnly,
}
/// Resource type declared at a shader binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingType {
    /// Read-only uniform bytes.
    UniformBuffer,
    /// Structured shader storage bytes.
    StorageBuffer {
        /// Whether writes through this binding are forbidden.
        read_only: bool,
    },
    /// A filterable, single-sampled 2D color texture.
    SampledTexture,
    /// A filtering sampler.
    Sampler,
    /// A single-mip 2D storage texture.
    StorageTexture {
        /// Texel format.
        format: TextureFormat,
        /// Shader access.
        access: StorageTextureAccess,
    },
}
/// One binding number and its stage visibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BindGroupLayoutEntry {
    binding: u32,
    visibility: ShaderStages,
    ty: BindingType,
}
impl BindGroupLayoutEntry {
    /// Describe one shader binding. The enclosing layout validates it.
    pub const fn new(binding: u32, visibility: ShaderStages, ty: BindingType) -> Self {
        Self {
            binding,
            visibility,
            ty,
        }
    }
    /// Return the binding number.
    pub const fn binding(self) -> u32 {
        self.binding
    }
    /// Return shader stage visibility.
    pub const fn visibility(self) -> ShaderStages {
        self.visibility
    }
    /// Return the required resource type.
    pub const fn ty(self) -> BindingType {
        self.ty
    }
}
/// Canonically ordered immutable descriptor-set layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindGroupLayoutDesc {
    entries: Vec<BindGroupLayoutEntry>,
}
impl BindGroupLayoutDesc {
    /// Construct a bounded layout with unique binding numbers and non-empty visibility.
    pub fn new(mut entries: Vec<BindGroupLayoutEntry>) -> Result<Self> {
        if entries.len() > MAX_BINDINGS_PER_GROUP {
            return Err(Error::InvalidDescriptor);
        }
        entries.sort_unstable_by_key(|entry| entry.binding);
        for (index, entry) in entries.iter().enumerate() {
            if entry.binding >= MAX_BINDINGS_PER_GROUP as u32
                || entry.visibility == ShaderStages::empty()
                || (index > 0 && entries[index - 1].binding == entry.binding)
                || matches!(entry.ty, BindingType::StorageTexture { format, .. } if format != TextureFormat::Rgba8Unorm)
            {
                return Err(Error::InvalidDescriptor);
            }
        }
        Ok(Self { entries })
    }
    /// Return bindings ordered by binding number.
    pub fn entries(&self) -> &[BindGroupLayoutEntry] {
        &self.entries
    }
}
/// Ordered descriptor sets shared by pipeline shader stages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineLayoutDesc {
    bind_groups: Vec<BindGroupLayoutDesc>,
}
impl PipelineLayoutDesc {
    /// Construct a layout containing up to four descriptor sets.
    pub fn new(bind_groups: Vec<BindGroupLayoutDesc>) -> Result<Self> {
        if bind_groups.len() > MAX_BIND_GROUPS {
            return Err(Error::InvalidDescriptor);
        }
        Ok(Self { bind_groups })
    }
    /// Return descriptor sets in shader set-number order.
    pub fn bind_groups(&self) -> &[BindGroupLayoutDesc] {
        &self.bind_groups
    }
}
/// An owned resource identity and optional byte range used by one binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingResource {
    /// Uniform or storage buffer range; offsets use portable 256-byte alignment.
    Buffer {
        /// Owning buffer.
        buffer: BufferId,
        /// Byte offset.
        offset: u64,
        /// Non-zero byte length.
        size: u64,
    },
    /// A whole single-mip 2D texture.
    Texture(TextureId),
    /// A sampler.
    Sampler(SamplerId),
}
/// One concrete binding number and resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BindGroupEntry {
    binding: u32,
    resource: BindingResource,
}
impl BindGroupEntry {
    /// Construct a binding; group construction validates ownership, usage and range.
    pub const fn new(binding: u32, resource: BindingResource) -> Self {
        Self { binding, resource }
    }
    /// Return the binding number.
    pub const fn binding(self) -> u32 {
        self.binding
    }
    /// Return the bound resource.
    pub const fn resource(self) -> BindingResource {
        self.resource
    }
}
/// A validated immutable set of resources matching one layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindGroupDesc {
    layout: BindGroupLayoutDesc,
    entries: Vec<BindGroupEntry>,
}
impl BindGroupDesc {
    /// Construct bindings, checking exact layout coverage, ownership, usage and ranges.
    pub fn new(
        resources: &ResourceTable,
        layout: BindGroupLayoutDesc,
        mut entries: Vec<BindGroupEntry>,
    ) -> Result<Self> {
        entries.sort_unstable_by_key(|entry| entry.binding);
        let result = Self { layout, entries };
        result.validate(resources)?;
        Ok(result)
    }
    pub(crate) fn validate(&self, resources: &ResourceTable) -> Result<()> {
        if self.entries.len() != self.layout.entries.len() {
            return Err(Error::BindingLayoutMismatch);
        }
        for (entry, layout) in self.entries.iter().zip(&self.layout.entries) {
            if entry.binding != layout.binding {
                return Err(Error::BindingLayoutMismatch);
            }
            match (entry.resource, layout.ty) {
                (
                    BindingResource::Buffer {
                        buffer,
                        offset,
                        size,
                    },
                    ty @ (BindingType::UniformBuffer | BindingType::StorageBuffer { .. }),
                ) => {
                    let desc = resources.buffer(resources.buffer_ref(buffer)?)?;
                    let required = if ty == BindingType::UniformBuffer {
                        BufferUsage::UNIFORM
                    } else {
                        BufferUsage::STORAGE
                    };
                    if !desc.usage().contains(required) {
                        return Err(Error::InvalidUsage);
                    }
                    if size == 0 || !offset.is_multiple_of(256) || !size.is_multiple_of(4) {
                        return Err(Error::InvalidValue);
                    }
                    if offset.checked_add(size).ok_or(Error::Overflow)? > desc.size() {
                        return Err(Error::OutOfBounds);
                    }
                }
                (BindingResource::Texture(texture), BindingType::SampledTexture) => {
                    let desc = resources.texture(resources.texture_ref(texture)?)?;
                    if !desc.usage().contains(TextureUsage::SAMPLED) {
                        return Err(Error::InvalidUsage);
                    }
                    if desc.format() == TextureFormat::Depth32Float {
                        return Err(Error::InvalidDescriptor);
                    }
                }
                (BindingResource::Texture(texture), BindingType::StorageTexture { format, .. }) => {
                    let desc = resources.texture(resources.texture_ref(texture)?)?;
                    if !desc.usage().contains(TextureUsage::STORAGE) {
                        return Err(Error::InvalidUsage);
                    }
                    if desc.format() != format {
                        return Err(Error::BindingLayoutMismatch);
                    }
                }
                (BindingResource::Sampler(sampler), BindingType::Sampler) => {
                    resources.sampler_ref(sampler)?;
                }
                _ => return Err(Error::BindingLayoutMismatch),
            }
        }
        // Writable aliases are excluded even when their byte ranges do not overlap:
        // WGPU resource tracking and this IR synchronize whole resources.
        for (i, entry) in self.entries.iter().enumerate() {
            for (j, other) in self.entries[..i].iter().enumerate() {
                if resource_alias(entry.resource, other.resource)
                    && (binding_writes(self.layout.entries[i].ty)
                        || binding_writes(self.layout.entries[j].ty))
                {
                    return Err(Error::ResourceAccessConflict);
                }
            }
        }
        Ok(())
    }
    /// Return the layout implemented by this set.
    pub const fn layout(&self) -> &BindGroupLayoutDesc {
        &self.layout
    }
    /// Return bindings in ascending binding order.
    pub fn entries(&self) -> &[BindGroupEntry] {
        &self.entries
    }
}
pub(crate) fn resource_alias(left: BindingResource, right: BindingResource) -> bool {
    match (left, right) {
        (BindingResource::Buffer { buffer: a, .. }, BindingResource::Buffer { buffer: b, .. }) => {
            a == b
        }
        (BindingResource::Texture(a), BindingResource::Texture(b)) => a == b,
        _ => false,
    }
}
pub(crate) fn binding_writes(ty: BindingType) -> bool {
    matches!(
        ty,
        BindingType::StorageBuffer { read_only: false } | BindingType::StorageTexture { .. }
    )
}
/// A compute entry point and its descriptor-set interface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputePipelineDesc {
    shader: ShaderEntryPoint,
    layout: PipelineLayoutDesc,
}
impl ComputePipelineDesc {
    /// Construct a compute pipeline descriptor.
    pub fn new(shader: ShaderEntryPoint, layout: PipelineLayoutDesc) -> Result<Self> {
        if shader.stage != ShaderStage::Compute {
            return Err(Error::InvalidDescriptor);
        }
        Ok(Self { shader, layout })
    }
    /// Return the compute entry point.
    pub const fn shader(&self) -> &ShaderEntryPoint {
        &self.shader
    }
    /// Return the pipeline's descriptor-set layout.
    pub const fn layout(&self) -> &PipelineLayoutDesc {
        &self.layout
    }
}
/// Programmable graphics with one color target and an optional interleaved vertex buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgrammableRenderPipelineDesc {
    vertex: ShaderEntryPoint,
    fragment: ShaderEntryPoint,
    layout: PipelineLayoutDesc,
    target_format: TextureFormat,
    vertex_buffer: Option<VertexBufferLayout>,
    topology: PrimitiveTopology,
    blend: BlendState,
    raster: RasterState,
    depth: Option<DepthState>,
}
impl ProgrammableRenderPipelineDesc {
    /// Construct a programmable graphics pipeline. `None` enables vertex-index-generated geometry.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vertex: ShaderEntryPoint,
        fragment: ShaderEntryPoint,
        layout: PipelineLayoutDesc,
        target_format: TextureFormat,
        vertex_buffer: Option<VertexBufferLayout>,
        topology: PrimitiveTopology,
        blend: BlendState,
        raster: RasterState,
    ) -> Result<Self> {
        if vertex.stage != ShaderStage::Vertex
            || fragment.stage != ShaderStage::Fragment
            || target_format == TextureFormat::Depth32Float
        {
            return Err(Error::InvalidDescriptor);
        }
        Ok(Self {
            vertex,
            fragment,
            layout,
            target_format,
            vertex_buffer,
            topology,
            blend,
            raster,
            depth: None,
        })
    }
    /// Add depth testing and optional depth writes.
    pub fn with_depth_state(mut self, depth: DepthState) -> Result<Self> {
        if depth.format() != TextureFormat::Depth32Float {
            return Err(Error::InvalidDescriptor);
        }
        self.depth = Some(depth);
        Ok(self)
    }
    /// Return the vertex entry point.
    pub const fn vertex(&self) -> &ShaderEntryPoint {
        &self.vertex
    }
    /// Return the fragment entry point.
    pub const fn fragment(&self) -> &ShaderEntryPoint {
        &self.fragment
    }
    /// Return descriptor sets.
    pub const fn layout(&self) -> &PipelineLayoutDesc {
        &self.layout
    }
    /// Return the sole color target format.
    pub const fn target_format(&self) -> TextureFormat {
        self.target_format
    }
    /// Return the optional vertex buffer layout.
    pub const fn vertex_buffer(&self) -> Option<&VertexBufferLayout> {
        self.vertex_buffer.as_ref()
    }
    /// Return triangle topology.
    pub const fn topology(&self) -> PrimitiveTopology {
        self.topology
    }
    /// Return blending state.
    pub const fn blend(&self) -> BlendState {
        self.blend
    }
    /// Return face culling and winding.
    pub const fn raster(&self) -> RasterState {
        self.raster
    }
    /// Return optional depth testing state.
    pub const fn depth_state(&self) -> Option<DepthState> {
        self.depth
    }
}
/// Whole-buffer access scope for an explicit dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferAccess {
    /// Copy/upload writes.
    CopyDestination,
    /// Copy reads.
    CopySource,
    /// Vertex fetch.
    Vertex,
    /// Index fetch.
    Index,
    /// Uniform reads in any shader stage.
    Uniform,
    /// Read-only storage access in any shader stage.
    StorageRead,
    /// Storage reads and writes in any shader stage.
    StorageReadWrite,
}
impl BufferAccess {
    pub(crate) const fn usage(self) -> BufferUsage {
        match self {
            Self::CopyDestination => BufferUsage::COPY_DST,
            Self::CopySource => BufferUsage::COPY_SRC,
            Self::Vertex => BufferUsage::VERTEX,
            Self::Index => BufferUsage::INDEX,
            Self::Uniform => BufferUsage::UNIFORM,
            Self::StorageRead | Self::StorageReadWrite => BufferUsage::STORAGE,
        }
    }
}
/// Whole-texture access scope; images currently have exactly one mip and layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureAccess {
    /// Copy/upload writes.
    CopyDestination,
    /// Copy reads.
    CopySource,
    /// Shader sampling.
    Sampled,
    /// Color or depth attachment operations.
    RenderAttachment,
    /// Shader storage writes.
    StorageWrite,
}
impl TextureAccess {
    pub(crate) const fn usage(self) -> TextureUsage {
        match self {
            Self::CopyDestination => TextureUsage::COPY_DST,
            Self::CopySource => TextureUsage::COPY_SRC,
            Self::Sampled => TextureUsage::SAMPLED,
            Self::RenderAttachment => TextureUsage::RENDER_ATTACHMENT,
            Self::StorageWrite => TextureUsage::STORAGE,
        }
    }
}
/// An explicit execution and memory dependency between whole-resource accesses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceBarrier<'r> {
    /// Synchronize a buffer's accesses.
    Buffer {
        /// Resource.
        buffer: BufferRef<'r>,
        /// Previous access.
        before: BufferAccess,
        /// Subsequent access.
        after: BufferAccess,
    },
    /// Synchronize a texture's accesses.
    Texture {
        /// Resource.
        texture: TextureRef<'r>,
        /// Previous access.
        before: TextureAccess,
        /// Subsequent access.
        after: TextureAccess,
    },
}
