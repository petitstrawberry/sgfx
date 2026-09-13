//! Logical resource descriptors, validated resource tables, and branded references.

use alloc::{rc::Rc, vec::Vec};
use core::cell::RefCell;
use core::fmt;
use core::ops::{BitOr, BitOrAssign};
use core::sync::atomic::{AtomicUsize, Ordering};

use super::pipeline::RenderPipelineDesc;
use super::{
    BindGroupDesc, ComputePipelineDesc, Error, Extent2D, PixelRect, ProgrammableRenderPipelineDesc,
    Result, ShaderModuleDesc,
};

/// Maximum textures held by one [`ResourceTable`].
pub const MAX_TEXTURES: usize = 1_024;
/// Maximum buffers held by one [`ResourceTable`].
pub const MAX_BUFFERS: usize = 1_024;
/// Maximum samplers held by one [`ResourceTable`].
pub const MAX_SAMPLERS: usize = 256;
/// Maximum render pipelines held by one [`ResourceTable`].
pub const MAX_RENDER_PIPELINES: usize = 256;
/// Maximum immutable bind-group definitions held by one resource table.
pub const MAX_BIND_GROUP_DEFINITIONS: usize = 4_096;

static NEXT_RESOURCE_TABLE_ID: AtomicUsize = AtomicUsize::new(1);

/// Portable texture pixel format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureFormat {
    /// Eight-bit blue, green, red, and alpha channels.
    Bgra8Unorm,
    /// Eight-bit red, green, blue, and alpha channels.
    Rgba8Unorm,
    /// One eight-bit normalized red channel.
    R8Unorm,
    /// One 32-bit floating-point depth component.
    Depth32Float,
}

impl TextureFormat {
    /// Return the number of bytes per tightly packed pixel.
    ///
    /// # Returns
    ///
    /// The portable byte size for this format.
    pub const fn bytes_per_pixel(self) -> u32 {
        match self {
            Self::Bgra8Unorm | Self::Rgba8Unorm | Self::Depth32Float => 4,
            Self::R8Unorm => 1,
        }
    }
}

/// Bitflag-like allowed operations for a texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureUsage(u8);

impl TextureUsage {
    /// Allow shader sampling.
    pub const SAMPLED: Self = Self(1 << 0);
    /// Allow use as a render-pass color or depth attachment.
    pub const RENDER_ATTACHMENT: Self = Self(1 << 1);
    /// Allow use as a texture copy source.
    pub const COPY_SRC: Self = Self(1 << 2);
    /// Allow use as a texture upload or copy destination.
    pub const COPY_DST: Self = Self(1 << 3);
    /// Allow eventual presentation by a backend.
    pub const PRESENT: Self = Self(1 << 4);
    /// Allow write-only shader storage access to RGBA8 texels.
    pub const STORAGE: Self = Self(1 << 5);

    /// Return no usage flags.
    ///
    /// # Returns
    ///
    /// An empty usage set.
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Return whether all flags in `other` are present.
    ///
    /// # Arguments
    ///
    /// * `other` - Flags to test.
    ///
    /// # Returns
    ///
    /// `true` when every requested flag is set.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Combine this usage set with another.
    ///
    /// # Arguments
    ///
    /// * `other` - Flags to add.
    ///
    /// # Returns
    ///
    /// The union of both usage sets.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl BitOr for TextureUsage {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}
impl BitOrAssign for TextureUsage {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = self.union(rhs);
    }
}

/// Validated descriptor for a logical texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureDesc {
    format: TextureFormat,
    extent: Extent2D,
    usage: TextureUsage,
    mip_level_count: u32,
}

impl TextureDesc {
    /// Construct a texture descriptor.
    ///
    /// # Arguments
    ///
    /// * `format` - Pixel format.
    /// * `extent` - Non-zero texture dimensions.
    /// * `usage` - Allowed texture operations.
    ///
    /// # Returns
    ///
    /// A descriptor, or [`Error::InvalidDescriptor`] for empty usage or a
    /// depth format carrying anything except `RENDER_ATTACHMENT` usage.
    pub const fn new(format: TextureFormat, extent: Extent2D, usage: TextureUsage) -> Result<Self> {
        if usage.0 == 0
            || (usage.contains(TextureUsage::STORAGE)
                && !matches!(format, TextureFormat::Rgba8Unorm))
            || (matches!(format, TextureFormat::Depth32Float)
                && usage.0 != TextureUsage::RENDER_ATTACHMENT.0)
        {
            Err(Error::InvalidDescriptor)
        } else {
            Ok(Self {
                format,
                extent,
                usage,
                mip_level_count: 1,
            })
        }
    }

    /// Return the pixel format.
    ///
    /// # Returns
    /// The texture format.
    pub const fn format(self) -> TextureFormat {
        self.format
    }
    /// Return the dimensions.
    ///
    /// # Returns
    /// The non-zero texture extent.
    pub const fn extent(self) -> Extent2D {
        self.extent
    }
    /// Return allowed operations.
    ///
    /// # Returns
    /// The texture usage flags.
    pub const fn usage(self) -> TextureUsage {
        self.usage
    }

    /// Declare a complete or partial 2D mip chain. Render attachments and
    /// storage bindings continue to use level zero; sampled bindings use the
    /// complete declared chain. Depth, storage and presentation images retain
    /// one level in this portable subset.
    pub fn with_mip_level_count(mut self, count: u32) -> Result<Self> {
        let maximum = self.extent.width().max(self.extent.height()).ilog2() + 1;
        if count == 0
            || count > maximum
            || (count > 1
                && (self.format == TextureFormat::Depth32Float
                    || self.usage.contains(TextureUsage::STORAGE)
                    || self.usage.contains(TextureUsage::PRESENT)))
        {
            return Err(Error::InvalidDescriptor);
        }
        self.mip_level_count = count;
        Ok(self)
    }

    /// Return the number of declared mip levels.
    pub const fn mip_level_count(self) -> u32 {
        self.mip_level_count
    }

    /// Return the checked dimensions of a mip level, including odd-sized
    /// chains and dimensions clamped to one texel.
    pub fn mip_extent(self, level: u32) -> Result<Extent2D> {
        if level >= self.mip_level_count {
            return Err(Error::OutOfBounds);
        }
        Extent2D::new(
            (self.extent.width() >> level).max(1),
            (self.extent.height() >> level).max(1),
        )
    }

    /// Return the tightly packed byte size of every declared mip level.
    pub fn byte_size(self) -> Result<u64> {
        let mut size = 0u64;
        for level in 0..self.mip_level_count {
            let extent = self.mip_extent(level)?;
            let bytes = u64::from(extent.width())
                .checked_mul(u64::from(extent.height()))
                .and_then(|v| v.checked_mul(u64::from(self.format.bytes_per_pixel())))
                .ok_or(Error::Overflow)?;
            size = size.checked_add(bytes).ok_or(Error::Overflow)?;
        }
        Ok(size)
    }
}

/// Bitflag-like allowed operations for a buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferUsage(u8);

impl BufferUsage {
    /// Allow use as a vertex buffer.
    pub const VERTEX: Self = Self(1 << 0);
    /// Allow use as an index buffer.
    pub const INDEX: Self = Self(1 << 1);
    /// Allow use as a copy source.
    pub const COPY_SRC: Self = Self(1 << 2);
    /// Allow writes through upload commands.
    pub const COPY_DST: Self = Self(1 << 3);
    /// Allow shader uniform reads.
    pub const UNIFORM: Self = Self(1 << 4);
    /// Allow shader storage reads and writes.
    pub const STORAGE: Self = Self(1 << 5);
    /// Return no usage flags.
    ///
    /// # Returns
    /// An empty usage set.
    pub const fn empty() -> Self {
        Self(0)
    }
    /// Return whether all flags in `other` are present.
    ///
    /// # Arguments
    ///
    /// * `other` - Flags to test.
    ///
    /// # Returns
    /// `true` when every requested flag is set.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
    /// Combine this usage set with another.
    ///
    /// # Arguments
    ///
    /// * `other` - Flags to add.
    ///
    /// # Returns
    /// The union of both usage sets.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl BitOr for BufferUsage {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}
impl BitOrAssign for BufferUsage {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = self.union(rhs);
    }
}

/// Validated descriptor for a logical buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferDesc {
    size: u64,
    usage: BufferUsage,
}

impl BufferDesc {
    /// Construct a buffer descriptor.
    ///
    /// # Arguments
    ///
    /// * `size` - Non-zero size in bytes.
    /// * `usage` - Allowed buffer operations.
    ///
    /// # Returns
    /// A descriptor, or [`Error::InvalidDescriptor`] for empty size or usage.
    pub const fn new(size: u64, usage: BufferUsage) -> Result<Self> {
        if size == 0 || usage.0 == 0 {
            Err(Error::InvalidDescriptor)
        } else {
            Ok(Self { size, usage })
        }
    }
    /// Return the size in bytes.
    ///
    /// # Returns
    /// The non-zero buffer size.
    pub const fn size(self) -> u64 {
        self.size
    }
    /// Return allowed operations.
    ///
    /// # Returns
    /// The buffer usage flags.
    pub const fn usage(self) -> BufferUsage {
        self.usage
    }
}

/// Texture filtering mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterMode {
    /// Select the nearest texel.
    Nearest,
    /// Linearly filter neighboring texels.
    Linear,
}

/// Texture coordinate addressing mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressMode {
    /// Clamp coordinates to the edge texel.
    ClampToEdge,
    /// Repeat the texture.
    Repeat,
    /// Repeat while mirroring every other copy.
    MirrorRepeat,
}

/// Portable logical sampler descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplerDesc {
    min_filter: FilterMode,
    mag_filter: FilterMode,
    address_u: AddressMode,
    address_v: AddressMode,
    mip_filter: FilterMode,
    min_lod_bits: u32,
    max_lod_bits: u32,
}

impl SamplerDesc {
    /// Construct a sampler descriptor.
    ///
    /// # Arguments
    ///
    /// * `min_filter` - Minification filter.
    /// * `mag_filter` - Magnification filter.
    /// * `address_u` - Horizontal addressing.
    /// * `address_v` - Vertical addressing.
    ///
    /// # Returns
    /// A portable sampler descriptor.
    pub const fn new(
        min_filter: FilterMode,
        mag_filter: FilterMode,
        address_u: AddressMode,
        address_v: AddressMode,
    ) -> Self {
        Self {
            min_filter,
            mag_filter,
            address_u,
            address_v,
            mip_filter: FilterMode::Nearest,
            min_lod_bits: 0,
            max_lod_bits: 0,
        }
    }
    /// Return the minification filter.
    ///
    /// # Returns
    /// The configured filter.
    pub const fn min_filter(self) -> FilterMode {
        self.min_filter
    }
    /// Return the magnification filter.
    ///
    /// # Returns
    /// The configured filter.
    pub const fn mag_filter(self) -> FilterMode {
        self.mag_filter
    }
    /// Return horizontal addressing.
    ///
    /// # Returns
    /// The configured U-coordinate mode.
    pub const fn address_u(self) -> AddressMode {
        self.address_u
    }
    /// Return vertical addressing.
    ///
    /// # Returns
    /// The configured V-coordinate mode.
    pub const fn address_v(self) -> AddressMode {
        self.address_v
    }

    /// Set mip filtering and a finite, nonnegative LOD interval. Keeping
    /// floating values as validated bits preserves descriptor equality.
    pub fn with_mip_filter(
        mut self,
        filter: FilterMode,
        min_lod: f32,
        max_lod: f32,
    ) -> Result<Self> {
        if !min_lod.is_finite() || !max_lod.is_finite() || min_lod < 0.0 || max_lod < min_lod {
            return Err(Error::InvalidValue);
        }
        self.mip_filter = filter;
        self.min_lod_bits = if min_lod == 0.0 { 0 } else { min_lod.to_bits() };
        self.max_lod_bits = if max_lod == 0.0 { 0 } else { max_lod.to_bits() };
        Ok(self)
    }
    /// Return mip-level filtering.
    pub const fn mip_filter(self) -> FilterMode {
        self.mip_filter
    }
    /// Return the minimum LOD.
    pub const fn min_lod(self) -> f32 {
        f32::from_bits(self.min_lod_bits)
    }
    /// Return the maximum LOD.
    pub const fn max_lod(self) -> f32 {
        f32::from_bits(self.max_lod_bits)
    }
}

/// Borrowed pixel data and layout for one texture upload.
#[derive(Debug, Clone, Copy)]
pub struct TextureWrite<'data> {
    destination: PixelRect,
    bytes_per_row: u32,
    data: &'data [u8],
    mip_level: u32,
}

impl<'data> TextureWrite<'data> {
    /// Construct a texture upload description.
    ///
    /// # Arguments
    ///
    /// * `destination` - Non-empty destination rectangle.
    /// * `bytes_per_row` - Source byte stride, validated against the texture format when recorded.
    /// * `data` - Borrowed source bytes.
    ///
    /// # Returns
    /// The upload description, or [`Error::InvalidValue`] for a zero stride.
    pub const fn new(
        destination: PixelRect,
        bytes_per_row: u32,
        data: &'data [u8],
    ) -> Result<Self> {
        if bytes_per_row == 0 {
            Err(Error::InvalidValue)
        } else {
            Ok(Self {
                destination,
                bytes_per_row,
                data,
                mip_level: 0,
            })
        }
    }
    /// Return the destination rectangle.
    ///
    /// # Returns
    /// The upload destination.
    pub const fn destination(self) -> PixelRect {
        self.destination
    }
    /// Return the source row stride.
    ///
    /// # Returns
    /// Bytes between source row starts.
    pub const fn bytes_per_row(self) -> u32 {
        self.bytes_per_row
    }
    /// Return the borrowed source bytes.
    ///
    /// # Returns
    /// The source byte slice.
    pub const fn data(self) -> &'data [u8] {
        self.data
    }
    /// Select a destination mip level; recording validates it against the texture.
    pub const fn with_mip_level(mut self, level: u32) -> Self {
        self.mip_level = level;
        self
    }
    /// Return the destination mip level.
    pub const fn mip_level(self) -> u32 {
        self.mip_level
    }
}

/// Table that owns validated logical resource descriptors.
pub struct ResourceTable {
    id: usize,
    textures: RefCell<Vec<TextureDesc>>,
    buffers: RefCell<Vec<BufferDesc>>,
    samplers: RefCell<Vec<SamplerDesc>>,
    pipelines: RefCell<Vec<RenderPipelineDesc>>,
    shader_modules: RefCell<Vec<ShaderModuleDesc>>,
    bind_groups: RefCell<Vec<Rc<BindGroupDesc>>>,
    compute_pipelines: RefCell<Vec<ComputePipelineDesc>>,
    programmable_pipelines: RefCell<Vec<Rc<ProgrammableRenderPipelineDesc>>>,
}

impl ResourceTable {
    /// Construct an empty resource table.
    ///
    /// # Returns
    /// An empty table with bounded resource categories.
    pub fn new() -> Self {
        Self {
            id: NEXT_RESOURCE_TABLE_ID.fetch_add(1, Ordering::Relaxed),
            textures: RefCell::new(Vec::new()),
            buffers: RefCell::new(Vec::new()),
            samplers: RefCell::new(Vec::new()),
            pipelines: RefCell::new(Vec::new()),
            shader_modules: RefCell::new(Vec::new()),
            bind_groups: RefCell::new(Vec::new()),
            compute_pipelines: RefCell::new(Vec::new()),
            programmable_pipelines: RefCell::new(Vec::new()),
        }
    }

    /// Define a texture descriptor and return its table-branded reference.
    ///
    /// # Arguments
    ///
    /// * `desc` - Validated texture descriptor.
    ///
    /// # Returns
    /// A texture reference, or a bounded-allocation error.
    pub fn define_texture(&self, desc: TextureDesc) -> Result<TextureRef<'_>> {
        let index = Self::push(&self.textures, desc, MAX_TEXTURES)?;
        Ok(TextureRef { owner: self, index })
    }
    /// Define a buffer descriptor and return its table-branded reference.
    ///
    /// # Arguments
    ///
    /// * `desc` - Validated buffer descriptor.
    ///
    /// # Returns
    /// A buffer reference, or a bounded-allocation error.
    pub fn define_buffer(&self, desc: BufferDesc) -> Result<BufferRef<'_>> {
        let index = Self::push(&self.buffers, desc, MAX_BUFFERS)?;
        Ok(BufferRef { owner: self, index })
    }
    /// Define a sampler descriptor and return its table-branded reference.
    ///
    /// # Arguments
    ///
    /// * `desc` - Sampler descriptor.
    ///
    /// # Returns
    /// A sampler reference, or a bounded-allocation error.
    pub fn define_sampler(&self, desc: SamplerDesc) -> Result<SamplerRef<'_>> {
        let index = Self::push(&self.samplers, desc, MAX_SAMPLERS)?;
        Ok(SamplerRef { owner: self, index })
    }
    /// Define a render-pipeline descriptor and return its table-branded reference.
    ///
    /// # Arguments
    ///
    /// * `desc` - Validated owned pipeline descriptor.
    ///
    /// # Returns
    /// A pipeline reference, or a bounded-allocation error.
    pub fn define_render_pipeline(
        &self,
        desc: RenderPipelineDesc,
    ) -> Result<RenderPipelineRef<'_>> {
        let index = Self::push(&self.pipelines, desc, MAX_RENDER_PIPELINES)?;
        Ok(RenderPipelineRef { owner: self, index })
    }

    /// Resolve a persistent texture identity into a borrowed table reference.
    ///
    /// # Arguments
    ///
    /// * `id` - Identity previously obtained from [`TextureRef::id`].
    ///
    /// # Returns
    ///
    /// A reference branded with this borrow of the owning table, or
    /// [`Error::ResourceTableMismatch`] when `id` belongs to another table.
    pub fn texture_ref(&self, id: TextureId) -> Result<TextureRef<'_>> {
        self.validate_id(id.owner, id.index, &self.textures)?;
        Ok(TextureRef {
            owner: self,
            index: id.index,
        })
    }

    /// Resolve a persistent buffer identity into a borrowed table reference.
    ///
    /// # Arguments
    ///
    /// * `id` - Identity previously obtained from [`BufferRef::id`].
    ///
    /// # Returns
    ///
    /// A reference branded with this borrow of the owning table, or
    /// [`Error::ResourceTableMismatch`] when `id` belongs to another table.
    pub fn buffer_ref(&self, id: BufferId) -> Result<BufferRef<'_>> {
        self.validate_id(id.owner, id.index, &self.buffers)?;
        Ok(BufferRef {
            owner: self,
            index: id.index,
        })
    }

    /// Resolve a persistent sampler identity into a borrowed table reference.
    ///
    /// # Arguments
    ///
    /// * `id` - Identity previously obtained from [`SamplerRef::id`].
    ///
    /// # Returns
    ///
    /// A reference branded with this borrow of the owning table, or
    /// [`Error::ResourceTableMismatch`] when `id` belongs to another table.
    pub fn sampler_ref(&self, id: SamplerId) -> Result<SamplerRef<'_>> {
        self.validate_id(id.owner, id.index, &self.samplers)?;
        Ok(SamplerRef {
            owner: self,
            index: id.index,
        })
    }

    /// Resolve a persistent pipeline identity into a borrowed table reference.
    ///
    /// # Arguments
    ///
    /// * `id` - Identity previously obtained from [`RenderPipelineRef::id`].
    ///
    /// # Returns
    ///
    /// A reference branded with this borrow of the owning table, or
    /// [`Error::ResourceTableMismatch`] when `id` belongs to another table.
    pub fn render_pipeline_ref(&self, id: RenderPipelineId) -> Result<RenderPipelineRef<'_>> {
        self.validate_id(id.owner, id.index, &self.pipelines)?;
        Ok(RenderPipelineRef {
            owner: self,
            index: id.index,
        })
    }

    /// Define an immutable shader module and return its branded reference.
    pub fn define_shader_module(&self, desc: ShaderModuleDesc) -> Result<ShaderModuleRef<'_>> {
        let index = Self::push(&self.shader_modules, desc, 256)?;
        Ok(ShaderModuleRef { owner: self, index })
    }
    /// Resolve a persistent shader module identity in its owning table.
    pub fn shader_module_ref(&self, id: ShaderModuleId) -> Result<ShaderModuleRef<'_>> {
        self.validate_id(id.owner, id.index, &self.shader_modules)?;
        Ok(ShaderModuleRef {
            owner: self,
            index: id.index,
        })
    }
    /// Return an owned copy of a validated shader module descriptor.
    pub fn shader_module(&self, reference: ShaderModuleRef<'_>) -> Result<ShaderModuleDesc> {
        if !core::ptr::eq(reference.owner, self) {
            return Err(Error::ResourceTableMismatch);
        }
        self.shader_modules
            .borrow()
            .get(reference.index)
            .cloned()
            .ok_or(Error::InvalidDescriptor)
    }
    /// Define an immutable bind group and return its branded reference.
    pub fn define_bind_group(&self, desc: BindGroupDesc) -> Result<BindGroupRef<'_>> {
        desc.validate(self)?;
        let index = Self::push(&self.bind_groups, Rc::new(desc), MAX_BIND_GROUP_DEFINITIONS)?;
        Ok(BindGroupRef { owner: self, index })
    }
    /// Resolve a persistent bind group identity in its owning table.
    pub fn bind_group_ref(&self, id: BindGroupId) -> Result<BindGroupRef<'_>> {
        self.validate_id(id.owner, id.index, &self.bind_groups)?;
        Ok(BindGroupRef {
            owner: self,
            index: id.index,
        })
    }
    /// Return an owned copy of a validated bind group descriptor.
    pub fn bind_group(&self, reference: BindGroupRef<'_>) -> Result<BindGroupDesc> {
        self.bind_group_shared(reference)
            .map(|desc| (*desc).clone())
    }
    /// Return shared immutable binding metadata without copying its entries.
    /// The table qualification is checked on every call. The returned value
    /// remains valid while further definitions grow the table.
    pub fn bind_group_shared(&self, reference: BindGroupRef<'_>) -> Result<Rc<BindGroupDesc>> {
        if !core::ptr::eq(reference.owner, self) {
            return Err(Error::ResourceTableMismatch);
        }
        self.bind_groups
            .borrow()
            .get(reference.index)
            .cloned()
            .ok_or(Error::InvalidDescriptor)
    }
    /// Define an immutable compute pipeline and return its branded reference.
    pub fn define_compute_pipeline(
        &self,
        desc: ComputePipelineDesc,
    ) -> Result<ComputePipelineRef<'_>> {
        self.shader_module_ref(desc.shader().module())?;
        let index = Self::push(&self.compute_pipelines, desc, 256)?;
        Ok(ComputePipelineRef { owner: self, index })
    }
    /// Resolve a persistent compute pipeline identity in its owning table.
    pub fn compute_pipeline_ref(&self, id: ComputePipelineId) -> Result<ComputePipelineRef<'_>> {
        self.validate_id(id.owner, id.index, &self.compute_pipelines)?;
        Ok(ComputePipelineRef {
            owner: self,
            index: id.index,
        })
    }
    /// Return an owned copy of a validated compute pipeline descriptor.
    pub fn compute_pipeline(
        &self,
        reference: ComputePipelineRef<'_>,
    ) -> Result<ComputePipelineDesc> {
        if !core::ptr::eq(reference.owner, self) {
            return Err(Error::ResourceTableMismatch);
        }
        self.compute_pipelines
            .borrow()
            .get(reference.index)
            .cloned()
            .ok_or(Error::InvalidDescriptor)
    }
    /// Define an immutable programmable render pipeline and return its branded reference.
    pub fn define_programmable_render_pipeline(
        &self,
        desc: ProgrammableRenderPipelineDesc,
    ) -> Result<ProgrammableRenderPipelineRef<'_>> {
        self.shader_module_ref(desc.vertex().module())?;
        self.shader_module_ref(desc.fragment().module())?;
        let index = Self::push(&self.programmable_pipelines, Rc::new(desc), 256)?;
        Ok(ProgrammableRenderPipelineRef { owner: self, index })
    }
    /// Resolve a persistent programmable render pipeline identity in its owning table.
    pub fn programmable_render_pipeline_ref(
        &self,
        id: ProgrammableRenderPipelineId,
    ) -> Result<ProgrammableRenderPipelineRef<'_>> {
        self.validate_id(id.owner, id.index, &self.programmable_pipelines)?;
        Ok(ProgrammableRenderPipelineRef {
            owner: self,
            index: id.index,
        })
    }
    /// Return an owned copy of a validated programmable render pipeline descriptor.
    pub fn programmable_render_pipeline(
        &self,
        reference: ProgrammableRenderPipelineRef<'_>,
    ) -> Result<ProgrammableRenderPipelineDesc> {
        self.programmable_render_pipeline_shared(reference)
            .map(|desc| (*desc).clone())
    }
    /// Return shared immutable pipeline metadata without copying its layout
    /// or vertex attributes. Ownership validation is identical to the copying
    /// getter and no table borrow is retained while the descriptor is used.
    pub fn programmable_render_pipeline_shared(
        &self,
        reference: ProgrammableRenderPipelineRef<'_>,
    ) -> Result<Rc<ProgrammableRenderPipelineDesc>> {
        if !core::ptr::eq(reference.owner, self) {
            return Err(Error::ResourceTableMismatch);
        }
        self.programmable_pipelines
            .borrow()
            .get(reference.index)
            .cloned()
            .ok_or(Error::InvalidDescriptor)
    }

    fn push<T>(items: &RefCell<Vec<T>>, value: T, maximum: usize) -> Result<usize> {
        let mut items = items.borrow_mut();
        if items.len() >= maximum {
            return Err(Error::ResourceLimitExceeded);
        }
        items.try_reserve(1).map_err(|_| Error::OutOfMemory)?;
        let index = items.len();
        items.push(value);
        Ok(index)
    }

    fn validate_id<T>(&self, owner: usize, index: usize, items: &RefCell<Vec<T>>) -> Result<()> {
        if owner != self.id {
            return Err(Error::ResourceTableMismatch);
        }
        if items.borrow().get(index).is_none() {
            return Err(Error::InvalidDescriptor);
        }
        Ok(())
    }
    /// Return the validated descriptor for a texture reference.
    ///
    /// # Arguments
    ///
    /// * `reference` - Texture reference branded by this resource table.
    ///
    /// # Returns
    ///
    /// The texture descriptor, or an error when the reference belongs to a
    /// different table or no longer identifies a defined resource.
    pub fn texture(&self, reference: TextureRef<'_>) -> Result<TextureDesc> {
        if !core::ptr::eq(reference.owner, self) {
            return Err(Error::ResourceTableMismatch);
        }
        self.textures
            .borrow()
            .get(reference.index)
            .copied()
            .ok_or(Error::InvalidDescriptor)
    }
    /// Return the validated descriptor for a buffer reference.
    ///
    /// # Arguments
    ///
    /// * `reference` - Buffer reference branded by this resource table.
    ///
    /// # Returns
    ///
    /// The buffer descriptor, or an error when the reference belongs to a
    /// different table or no longer identifies a defined resource.
    pub fn buffer(&self, reference: BufferRef<'_>) -> Result<BufferDesc> {
        if !core::ptr::eq(reference.owner, self) {
            return Err(Error::ResourceTableMismatch);
        }
        self.buffers
            .borrow()
            .get(reference.index)
            .copied()
            .ok_or(Error::InvalidDescriptor)
    }
    /// Return the validated descriptor for a sampler reference.
    ///
    /// # Arguments
    ///
    /// * `reference` - Sampler reference branded by this resource table.
    ///
    /// # Returns
    ///
    /// The sampler descriptor, or an error when the reference belongs to a
    /// different table or no longer identifies a defined resource.
    pub fn sampler(&self, reference: SamplerRef<'_>) -> Result<SamplerDesc> {
        if !core::ptr::eq(reference.owner, self) {
            return Err(Error::ResourceTableMismatch);
        }
        self.samplers
            .borrow()
            .get(reference.index)
            .copied()
            .ok_or(Error::InvalidDescriptor)
    }
    /// Borrow a validated pipeline descriptor without cloning its owned layout.
    pub(crate) fn with_pipeline<T>(
        &self,
        reference: RenderPipelineRef<'_>,
        access: impl FnOnce(&RenderPipelineDesc) -> T,
    ) -> Result<T> {
        if !core::ptr::eq(reference.owner, self) {
            return Err(Error::ResourceTableMismatch);
        }
        let pipelines = self.pipelines.borrow();
        let descriptor = pipelines
            .get(reference.index)
            .ok_or(Error::InvalidDescriptor)?;
        Ok(access(descriptor))
    }

    /// Return a cloned render-pipeline descriptor for a backend lowerer.
    ///
    /// # Arguments
    ///
    /// * `reference` - Render-pipeline reference branded by this resource table.
    ///
    /// # Returns
    ///
    /// The owned pipeline descriptor, or an error when the reference belongs
    /// to another table or no longer identifies a defined resource.
    pub fn render_pipeline(&self, reference: RenderPipelineRef<'_>) -> Result<RenderPipelineDesc> {
        self.with_pipeline(reference, Clone::clone)
    }
    pub(crate) fn same_texture(&self, left: TextureRef<'_>, right: TextureRef<'_>) -> Result<bool> {
        self.texture(left)?;
        self.texture(right)?;
        Ok(left.index == right.index)
    }
}

impl Default for ResourceTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Reference to a texture retained by one [`ResourceTable`].
#[derive(Clone, Copy)]
pub struct TextureRef<'r> {
    pub(crate) owner: &'r ResourceTable,
    pub(crate) index: usize,
}
/// Reference to a buffer retained by one [`ResourceTable`].
#[derive(Clone, Copy)]
pub struct BufferRef<'r> {
    pub(crate) owner: &'r ResourceTable,
    pub(crate) index: usize,
}
/// Reference to a sampler retained by one [`ResourceTable`].
#[derive(Clone, Copy)]
pub struct SamplerRef<'r> {
    pub(crate) owner: &'r ResourceTable,
    pub(crate) index: usize,
}
/// Reference to a render pipeline retained by one [`ResourceTable`].
#[derive(Clone, Copy)]
pub struct RenderPipelineRef<'r> {
    pub(crate) owner: &'r ResourceTable,
    pub(crate) index: usize,
}

/// Persistent table-qualified identity of a logical texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureId {
    owner: usize,
    index: usize,
}

/// Persistent table-qualified identity of a logical buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId {
    owner: usize,
    index: usize,
}

/// Persistent table-qualified identity of a logical sampler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SamplerId {
    owner: usize,
    index: usize,
}

/// Persistent table-qualified identity of a logical render pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderPipelineId {
    owner: usize,
    index: usize,
}

impl TextureRef<'_> {
    /// Return the backend resource slot assigned by the owning table.
    ///
    /// # Returns
    ///
    /// A stable slot for the lifetime of the resource table.
    pub const fn slot(self) -> usize {
        self.index
    }

    /// Return whether this reference belongs to `resources`.
    ///
    /// # Arguments
    ///
    /// * `resources` - Resource table to compare with this branded reference.
    ///
    /// # Returns
    ///
    /// `true` when the reference was created by `resources`.
    pub fn belongs_to(self, resources: &ResourceTable) -> bool {
        core::ptr::eq(self.owner, resources)
    }

    /// Return an owned identity that may outlive this table borrow.
    ///
    /// # Returns
    ///
    /// The table-qualified texture identity. Resolve it with
    /// [`ResourceTable::texture_ref`] when recording a later command buffer.
    pub const fn id(self) -> TextureId {
        TextureId {
            owner: self.owner.id,
            index: self.index,
        }
    }
}

impl BufferRef<'_> {
    /// Return the backend resource slot assigned by the owning table.
    ///
    /// # Returns
    ///
    /// A stable slot for the lifetime of the resource table.
    pub const fn slot(self) -> usize {
        self.index
    }

    /// Return an owned identity that may outlive this table borrow.
    ///
    /// # Returns
    ///
    /// The table-qualified buffer identity. Resolve it with
    /// [`ResourceTable::buffer_ref`] when recording a later command buffer.
    pub const fn id(self) -> BufferId {
        BufferId {
            owner: self.owner.id,
            index: self.index,
        }
    }
}

impl SamplerRef<'_> {
    /// Return the backend resource slot assigned by the owning table.
    ///
    /// # Returns
    ///
    /// A stable slot for the lifetime of the resource table.
    pub const fn slot(self) -> usize {
        self.index
    }

    /// Return an owned identity that may outlive this table borrow.
    ///
    /// # Returns
    ///
    /// The table-qualified sampler identity. Resolve it with
    /// [`ResourceTable::sampler_ref`] when recording a later command buffer.
    pub const fn id(self) -> SamplerId {
        SamplerId {
            owner: self.owner.id,
            index: self.index,
        }
    }
}

impl RenderPipelineRef<'_> {
    /// Return the backend resource slot assigned by the owning table.
    ///
    /// # Returns
    ///
    /// A stable slot for the lifetime of the resource table.
    pub const fn slot(self) -> usize {
        self.index
    }

    /// Return an owned identity that may outlive this table borrow.
    ///
    /// # Returns
    ///
    /// The table-qualified pipeline identity. Resolve it with
    /// [`ResourceTable::render_pipeline_ref`] when recording a later command
    /// buffer.
    pub const fn id(self) -> RenderPipelineId {
        RenderPipelineId {
            owner: self.owner.id,
            index: self.index,
        }
    }
}

macro_rules! impl_resource_ref_traits {
    ($reference:ident) => {
        impl fmt::Debug for $reference<'_> {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(concat!(stringify!($reference), "(..)"))
            }
        }

        impl PartialEq for $reference<'_> {
            fn eq(&self, other: &Self) -> bool {
                core::ptr::eq(self.owner, other.owner) && self.index == other.index
            }
        }

        impl Eq for $reference<'_> {}
    };
}

impl_resource_ref_traits!(TextureRef);
impl_resource_ref_traits!(BufferRef);
impl_resource_ref_traits!(SamplerRef);
impl_resource_ref_traits!(RenderPipelineRef);

/// Borrowed reference to an immutable shader module.
#[derive(Clone, Copy)]
pub struct ShaderModuleRef<'r> {
    pub(crate) owner: &'r ResourceTable,
    pub(crate) index: usize,
}
/// Persistent table-qualified shader module identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShaderModuleId {
    owner: usize,
    index: usize,
}
impl ShaderModuleRef<'_> {
    /// Return the stable table-local backend slot.
    pub const fn slot(self) -> usize {
        self.index
    }
    /// Return a persistent identity for later resolution by the owning table.
    pub const fn id(self) -> ShaderModuleId {
        ShaderModuleId {
            owner: self.owner.id,
            index: self.index,
        }
    }
}
impl_resource_ref_traits!(ShaderModuleRef);

/// Borrowed reference to an immutable bind group.
#[derive(Clone, Copy)]
pub struct BindGroupRef<'r> {
    pub(crate) owner: &'r ResourceTable,
    pub(crate) index: usize,
}
/// Persistent table-qualified bind group identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BindGroupId {
    owner: usize,
    index: usize,
}
impl BindGroupRef<'_> {
    /// Return the stable table-local backend slot.
    pub const fn slot(self) -> usize {
        self.index
    }
    /// Return a persistent identity for later resolution by the owning table.
    pub const fn id(self) -> BindGroupId {
        BindGroupId {
            owner: self.owner.id,
            index: self.index,
        }
    }
}
impl_resource_ref_traits!(BindGroupRef);

/// Borrowed reference to an immutable compute pipeline.
#[derive(Clone, Copy)]
pub struct ComputePipelineRef<'r> {
    pub(crate) owner: &'r ResourceTable,
    pub(crate) index: usize,
}
/// Persistent table-qualified compute pipeline identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputePipelineId {
    owner: usize,
    index: usize,
}
impl ComputePipelineRef<'_> {
    /// Return the stable table-local backend slot.
    pub const fn slot(self) -> usize {
        self.index
    }
    /// Return a persistent identity for later resolution by the owning table.
    pub const fn id(self) -> ComputePipelineId {
        ComputePipelineId {
            owner: self.owner.id,
            index: self.index,
        }
    }
}
impl_resource_ref_traits!(ComputePipelineRef);

/// Borrowed reference to an immutable programmable render pipeline.
#[derive(Clone, Copy)]
pub struct ProgrammableRenderPipelineRef<'r> {
    pub(crate) owner: &'r ResourceTable,
    pub(crate) index: usize,
}
/// Persistent table-qualified programmable render pipeline identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProgrammableRenderPipelineId {
    owner: usize,
    index: usize,
}
impl ProgrammableRenderPipelineRef<'_> {
    /// Return the stable table-local backend slot.
    pub const fn slot(self) -> usize {
        self.index
    }
    /// Return a persistent identity for later resolution by the owning table.
    pub const fn id(self) -> ProgrammableRenderPipelineId {
        ProgrammableRenderPipelineId {
            owner: self.owner.id,
            index: self.index,
        }
    }
}
impl_resource_ref_traits!(ProgrammableRenderPipelineRef);

#[cfg(test)]
mod shared_metadata_tests {
    use super::*;
    use crate::ir::*;
    use alloc::vec;

    #[test]
    fn shared_bindings_retain_identity_across_table_growth_and_validate_owners() {
        let table = ResourceTable::new();
        let foreign = ResourceTable::new();
        let buffer = table
            .define_buffer(BufferDesc::new(16, BufferUsage::UNIFORM).unwrap())
            .unwrap()
            .id();
        let layout = BindGroupLayoutDesc::new(vec![BindGroupLayoutEntry::new(
            0,
            ShaderStages::VERTEX,
            BindingType::UniformBuffer,
        )])
        .unwrap();
        let desc = BindGroupDesc::new(
            &table,
            layout,
            vec![BindGroupEntry::new(
                0,
                BindingResource::Buffer {
                    buffer,
                    offset: 0,
                    size: 16,
                },
            )],
        )
        .unwrap();
        let reference = table.define_bind_group(desc.clone()).unwrap();
        let shared = table.bind_group_shared(reference).unwrap();
        for _ in 0..32 {
            table.define_bind_group(desc.clone()).unwrap();
            assert!(Rc::ptr_eq(
                &shared,
                &table.bind_group_shared(reference).unwrap()
            ));
        }
        let copied = table.bind_group(reference).unwrap();
        assert_eq!(*shared, copied);
        assert_ne!(shared.entries().as_ptr(), copied.entries().as_ptr());
        assert_eq!(
            foreign.bind_group_shared(reference).unwrap_err(),
            Error::ResourceTableMismatch
        );
        assert_eq!(
            foreign.bind_group(reference).unwrap_err(),
            Error::ResourceTableMismatch
        );
    }

    #[test]
    fn shared_pipelines_preserve_copying_getter_and_survive_new_definitions() {
        let table = ResourceTable::new();
        let foreign = ResourceTable::new();
        let shader = table
            .define_shader_module(
                ShaderModuleDesc::wgsl(
                    "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }
                     @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }"
                        .into(),
                )
                .unwrap(),
            )
            .unwrap();
        let desc = ProgrammableRenderPipelineDesc::new(
            ShaderEntryPoint::new(shader, ShaderStage::Vertex, "vs".into()).unwrap(),
            ShaderEntryPoint::new(shader, ShaderStage::Fragment, "fs".into()).unwrap(),
            PipelineLayoutDesc::new(vec![]).unwrap(),
            TextureFormat::Rgba8Unorm,
            Some(
                VertexBufferLayout::new(
                    8,
                    vec![VertexAttribute::new(0, VertexFormat::Float32x2, 0)],
                )
                .unwrap(),
            ),
            PrimitiveTopology::TriangleList,
            BlendState::REPLACE,
            RasterState::new(CullMode::None, FrontFace::CounterClockwise),
        )
        .unwrap();
        let reference = table
            .define_programmable_render_pipeline(desc.clone())
            .unwrap();
        let shared = table
            .programmable_render_pipeline_shared(reference)
            .unwrap();
        for _ in 0..32 {
            table
                .define_programmable_render_pipeline(desc.clone())
                .unwrap();
            assert!(Rc::ptr_eq(
                &shared,
                &table
                    .programmable_render_pipeline_shared(reference)
                    .unwrap()
            ));
        }
        let copied = table.programmable_render_pipeline(reference).unwrap();
        assert_eq!(*shared, copied);
        assert_ne!(
            shared.vertex_buffer().unwrap().attributes().as_ptr(),
            copied.vertex_buffer().unwrap().attributes().as_ptr()
        );
        assert_eq!(
            foreign
                .programmable_render_pipeline_shared(reference)
                .unwrap_err(),
            Error::ResourceTableMismatch
        );
        assert_eq!(
            foreign.programmable_render_pipeline(reference).unwrap_err(),
            Error::ResourceTableMismatch
        );
    }
}
