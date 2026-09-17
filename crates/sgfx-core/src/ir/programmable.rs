//! Programmable pipeline layouts, bindings and explicit resource access declarations.

use super::*;
use alloc::{string::String, vec::Vec};

/// Maximum descriptor sets in a pipeline layout.
pub const MAX_BIND_GROUPS: usize = 4;
/// Maximum bindings in a descriptor set.
pub const MAX_BINDINGS_PER_GROUP: usize = 32;
/// Maximum portable push-constant byte range. Backends may expose a lower limit.
pub const MAX_PUSH_CONSTANT_BYTES: u32 = 128;

/// A byte range visible to one or more shader stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PushConstantRange {
    stages: ShaderStages,
    offset: u32,
    size: u32,
}
impl PushConstantRange {
    /// Construct a non-empty, four-byte-aligned range within the portable limit.
    pub fn new(stages: ShaderStages, offset: u32, size: u32) -> Result<Self> {
        if stages.is_empty() || size == 0 || !offset.is_multiple_of(4) || !size.is_multiple_of(4) {
            return Err(Error::InvalidDescriptor);
        }
        if offset.checked_add(size).ok_or(Error::Overflow)? > MAX_PUSH_CONSTANT_BYTES {
            return Err(Error::OutOfBounds);
        }
        Ok(Self {
            stages,
            offset,
            size,
        })
    }
    /// Shader stages consuming the range.
    pub const fn stages(self) -> ShaderStages {
        self.stages
    }
    /// First byte in the stage's push-constant block.
    pub const fn offset(self) -> u32 {
        self.offset
    }
    /// Non-zero number of bytes.
    pub const fn size(self) -> u32 {
        self.size
    }
}

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
    /// A typed, single-sampled view into a texture allocation.
    SampledTextureView {
        /// Coordinate interpretation expected by the shader.
        dimension: TextureViewDimension,
        /// Whether the shader expects depth rather than color samples.
        depth: bool,
    },
    /// A filtering sampler.
    Sampler,
    /// A sampler that compares a supplied reference against sampled depth.
    ComparisonSampler,
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
    push_constant_ranges: Vec<PushConstantRange>,
}
impl PipelineLayoutDesc {
    /// Construct a layout containing up to four descriptor sets.
    pub fn new(bind_groups: Vec<BindGroupLayoutDesc>) -> Result<Self> {
        if bind_groups.len() > MAX_BIND_GROUPS {
            return Err(Error::InvalidDescriptor);
        }
        Ok(Self {
            bind_groups,
            push_constant_ranges: Vec::new(),
        })
    }
    /// Return descriptor sets in shader set-number order.
    pub fn bind_groups(&self) -> &[BindGroupLayoutDesc] {
        &self.bind_groups
    }
    /// Add ranges with each shader stage appearing in at most one range.
    /// Ranges for different stages may overlap in bytes.
    pub fn with_push_constant_ranges(mut self, mut ranges: Vec<PushConstantRange>) -> Result<Self> {
        if ranges.len() > 3 {
            return Err(Error::InvalidDescriptor);
        }
        let mut used = ShaderStages::empty();
        for range in &ranges {
            for stage in [
                ShaderStages::VERTEX,
                ShaderStages::FRAGMENT,
                ShaderStages::COMPUTE,
            ] {
                if range.stages.contains(stage) && used.contains(stage) {
                    return Err(Error::InvalidDescriptor);
                }
            }
            used |= range.stages;
        }
        ranges.sort_unstable_by_key(|range| (range.offset, range.size, range.stages.bits()));
        self.push_constant_ranges = ranges;
        Ok(self)
    }
    /// Declared push-constant ranges.
    pub fn push_constant_ranges(&self) -> &[PushConstantRange] {
        &self.push_constant_ranges
    }
    /// Check that every updated stage declares the complete aligned byte range.
    pub fn validate_push_constants(
        &self,
        stages: ShaderStages,
        offset: u32,
        data: &[u8],
    ) -> Result<()> {
        if stages.is_empty()
            || data.is_empty()
            || !offset.is_multiple_of(4)
            || !data.len().is_multiple_of(4)
        {
            return Err(Error::InvalidValue);
        }
        let size = u32::try_from(data.len()).map_err(|_| Error::Overflow)?;
        let end = offset.checked_add(size).ok_or(Error::Overflow)?;
        if end > MAX_PUSH_CONSTANT_BYTES {
            return Err(Error::OutOfBounds);
        }
        for stage in [
            ShaderStages::VERTEX,
            ShaderStages::FRAGMENT,
            ShaderStages::COMPUTE,
        ] {
            if stages.contains(stage)
                && !self.push_constant_ranges.iter().any(|range| {
                    range.stages.contains(stage)
                        && offset >= range.offset
                        && end <= range.offset + range.size
                })
            {
                return Err(Error::BindingLayoutMismatch);
            }
        }
        if self.push_constant_ranges.iter().any(|range| {
            offset < range.offset + range.size
                && range.offset < end
                && !stages.contains(range.stages)
        }) {
            return Err(Error::BindingLayoutMismatch);
        }
        Ok(())
    }
}
/// An owned resource identity and optional byte range used by one binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingResource {
    /// Uniform or storage buffer range; the backend checks its own alignment limit.
    Buffer {
        /// Owning buffer.
        buffer: BufferId,
        /// Byte offset.
        offset: u64,
        /// Non-zero byte length.
        size: u64,
    },
    /// A whole 2D sampled mip chain, or level zero for a storage binding.
    Texture(TextureId),
    /// A format and subresource selection from one texture allocation.
    TextureView {
        /// Owning allocation.
        texture: TextureId,
        /// Selected subresources and their interpretation.
        view: TextureViewDesc,
    },
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
                    let minimum_alignment = if ty == BindingType::UniformBuffer { 16 } else { 4 };
                    if size == 0
                        || !offset.is_multiple_of(minimum_alignment)
                        || !size.is_multiple_of(4)
                    {
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
                    if desc.format() == TextureFormat::Depth32Float || desc.array_layer_count() != 1 {
                        return Err(Error::InvalidDescriptor);
                    }
                }
                (BindingResource::TextureView { texture, view }, BindingType::SampledTextureView { dimension, depth }) => {
                    let desc = resources.texture(resources.texture_ref(texture)?)?;
                    view.validate(desc)?;
                    if !desc.usage().contains(TextureUsage::SAMPLED) {
                        return Err(Error::InvalidUsage);
                    }
                    if view.dimension() != dimension || (view.format() == TextureFormat::Depth32Float) != depth {
                        return Err(Error::BindingLayoutMismatch);
                    }
                }
                (BindingResource::Texture(texture), BindingType::StorageTexture { format, .. }) => {
                    let desc = resources.texture(resources.texture_ref(texture)?)?;
                    if !desc.usage().contains(TextureUsage::STORAGE) {
                        return Err(Error::InvalidUsage);
                    }
                    if desc.format() != format || desc.array_layer_count() != 1 || desc.mip_level_count() != 1 {
                        return Err(Error::BindingLayoutMismatch);
                    }
                }
                (BindingResource::TextureView { texture, view }, BindingType::StorageTexture { format, .. }) => {
                    let desc = resources.texture(resources.texture_ref(texture)?)?;
                    view.validate(desc)?;
                    if !desc.usage().contains(TextureUsage::STORAGE) {
                        return Err(Error::InvalidUsage);
                    }
                    if view.format() != format || view.dimension() != super::TextureViewDimension::D2
                        || view.mip_level_count() != 1 || view.array_layer_count() != 1 {
                        return Err(Error::BindingLayoutMismatch);
                    }
                }
                (BindingResource::Sampler(sampler), ty @ (BindingType::Sampler | BindingType::ComparisonSampler)) => {
                    let desc = resources.sampler(resources.sampler_ref(sampler)?)?;
                    if desc.compare().is_some() != (ty == BindingType::ComparisonSampler) {
                        return Err(Error::BindingLayoutMismatch);
                    }
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
        (BindingResource::Texture(a) | BindingResource::TextureView { texture: a, .. },
         BindingResource::Texture(b) | BindingResource::TextureView { texture: b, .. }) => a == b,
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
/// Programmable graphics with up to eight color targets and vertex buffers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgrammableRenderPipelineDesc {
    vertex: ShaderEntryPoint,
    fragment: ShaderEntryPoint,
    layout: PipelineLayoutDesc,
    target_format: TextureFormat,
    vertex_buffers: Vec<VertexBufferLayout>,
    topology: PrimitiveTopology,
    blend: BlendState,
    raster: RasterState,
    depth: Option<DepthState>,
    color_write_mask: super::ColorWriteMask,
    additional_targets: Vec<super::ColorTargetState>,
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
            vertex_buffers: vertex_buffer.into_iter().collect(),
            topology,
            blend,
            raster,
            depth: None,
            color_write_mask: super::ColorWriteMask::ALL,
            additional_targets: Vec::new(),
        })
    }
    /// Replace fragment output formats and per-output blend/write state.
    pub fn with_color_targets(mut self, targets: Vec<super::ColorTargetState>) -> Result<Self> {
        if targets.is_empty() || targets.len() > super::MAX_COLOR_ATTACHMENTS { return Err(Error::InvalidDescriptor); }
        self.target_format = targets[0].format();
        self.blend = targets[0].blend();
        self.color_write_mask = targets[0].write_mask();
        self.additional_targets = targets.into_iter().skip(1).collect();
        Ok(self)
    }
    /// Return fragment outputs in location order.
    pub fn color_targets(&self) -> impl Iterator<Item = super::ColorTargetState> + '_ {
        core::iter::once(super::ColorTargetState::new(self.target_format, self.blend, self.color_write_mask).expect("validated color target"))
            .chain(self.additional_targets.iter().copied())
    }
    /// Replace vertex input with consecutive buffer slots. Attribute locations
    /// must be unique across slots and respect the total attribute limit.
    pub fn with_vertex_buffers(mut self, buffers: Vec<VertexBufferLayout>) -> Result<Self> {
        if buffers.len() > MAX_VERTEX_BUFFERS { return Err(Error::ResourceLimitExceeded); }
        let mut locations = Vec::new();
        for buffer in &buffers {
            for attribute in buffer.attributes() {
                if attribute.location() >= MAX_VERTEX_ATTRIBUTES as u32 || locations.contains(&attribute.location()) {
                    return Err(Error::InvalidDescriptor);
                }
                locations.push(attribute.location());
            }
        }
        self.vertex_buffers = buffers;
        Ok(self)
    }
    /// Return the layouts indexed by vertex buffer slot.
    pub fn vertex_buffers(&self) -> &[VertexBufferLayout] { &self.vertex_buffers }
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
    pub fn vertex_buffer(&self) -> Option<&VertexBufferLayout> {
        self.vertex_buffers.first()
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
/// Texture access scope; sampled images may contain multiple mip levels.
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
    /// Synchronize one mip level independently of the other levels.
    TextureMip {
        /// Resource.
        texture: TextureRef<'r>,
        /// Checked mip level.
        mip_level: u32,
        /// Previous access.
        before: TextureAccess,
        /// Subsequent access.
        after: TextureAccess,
    },
}
