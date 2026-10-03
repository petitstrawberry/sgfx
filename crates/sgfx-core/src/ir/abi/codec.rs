//! Frozen v2 scalar/descriptor encoding. Enum numbers here are ABI numbers,
//! independent of Rust discriminants. Existing numbers must never be reordered.

use super::*;
use alloc::{string::String, vec::Vec};

pub(super) trait Codec: Sized {
    fn put(&self, w: &mut Vec<u64>);
    fn get(r: &mut Reader<'_>, table: &ResourceTable) -> Result<Self>;
}

#[derive(Clone)]
pub(super) struct Reader<'a> {
    pub words: &'a [u64],
}
impl<'a> Reader<'a> {
    #[inline(always)]
    pub fn word(&mut self) -> Result<u64> {
        let (first, rest) = self.words.split_first().ok_or(Error::InvalidDescriptor)?;
        self.words = rest;
        Ok(*first)
    }
    #[inline(always)]
    pub fn take(&mut self, count: usize) -> Result<&'a [u64]> {
        if count > self.words.len() {
            return Err(Error::InvalidDescriptor);
        }
        let (value, rest) = self.words.split_at(count);
        self.words = rest;
        Ok(value)
    }
    #[inline(always)]
    pub fn value<T: Codec>(&mut self, t: &ResourceTable) -> Result<T> {
        T::get(self, t)
    }
    #[inline(always)]
    pub fn end(&self) -> Result<()> {
        if self.words.is_empty() {
            Ok(())
        } else {
            Err(Error::InvalidDescriptor)
        }
    }
}

macro_rules! integer {
    ($($ty:ty),*) => { $(impl Codec for $ty {
        #[inline(always)]
        fn put(&self, w: &mut Vec<u64>) { w.push(*self as u64); }
        #[inline(always)]
        fn get(r: &mut Reader<'_>, _: &ResourceTable) -> Result<Self> {
            Self::try_from(r.word()?).map_err(|_| Error::InvalidValue)
        }
    })* };
}
integer!(u8, u32, u64, usize);
impl Codec for i32 {
    fn put(&self, w: &mut Vec<u64>) {
        (*self as u32).put(w);
    }
    fn get(r: &mut Reader<'_>, t: &ResourceTable) -> Result<Self> {
        Ok(u32::get(r, t)? as i32)
    }
}
impl Codec for bool {
    fn put(&self, w: &mut Vec<u64>) {
        w.push(u64::from(*self));
    }
    fn get(r: &mut Reader<'_>, _: &ResourceTable) -> Result<Self> {
        match r.word()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(Error::InvalidValue),
        }
    }
}
impl Codec for f32 {
    fn put(&self, w: &mut Vec<u64>) {
        self.to_bits().put(w);
    }
    fn get(r: &mut Reader<'_>, t: &ResourceTable) -> Result<Self> {
        Ok(Self::from_bits(u32::get(r, t)?))
    }
}
impl<T: Codec> Codec for Option<T> {
    fn put(&self, w: &mut Vec<u64>) {
        self.is_some().put(w);
        if let Some(v) = self {
            v.put(w);
        }
    }
    fn get(r: &mut Reader<'_>, t: &ResourceTable) -> Result<Self> {
        if bool::get(r, t)? {
            Ok(Some(T::get(r, t)?))
        } else {
            Ok(None)
        }
    }
}
impl<T: Codec> Codec for Vec<T> {
    fn put(&self, w: &mut Vec<u64>) {
        self.len().put(w);
        for v in self {
            v.put(w);
        }
    }
    fn get(r: &mut Reader<'_>, t: &ResourceTable) -> Result<Self> {
        let count = usize::get(r, t)?;
        if count > r.words.len() {
            return Err(Error::InvalidDescriptor);
        }
        let mut result = Vec::new();
        result
            .try_reserve_exact(count)
            .map_err(|_| Error::OutOfMemory)?;
        for _ in 0..count {
            result.push(T::get(r, t)?);
        }
        Ok(result)
    }
}
pub(super) fn put_bytes(bytes: &[u8], w: &mut Vec<u64>) {
    bytes.len().put(w);
    for chunk in bytes.chunks(8) {
        let mut value = [0; 8];
        value[..chunk.len()].copy_from_slice(chunk);
        w.push(u64::from_le_bytes(value));
    }
}
pub(super) fn get_bytes(r: &mut Reader<'_>, t: &ResourceTable) -> Result<Vec<u8>> {
    let len = usize::get(r, t)?;
    let words = r.take(len.div_ceil(8))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .map_err(|_| Error::OutOfMemory)?;
    for value in words {
        let chunk = value.to_le_bytes();
        bytes.extend_from_slice(&chunk[..(len - bytes.len()).min(8)]);
    }
    Ok(bytes)
}
impl Codec for String {
    fn put(&self, w: &mut Vec<u64>) {
        put_bytes(self.as_bytes(), w);
    }
    fn get(r: &mut Reader<'_>, t: &ResourceTable) -> Result<Self> {
        String::from_utf8(get_bytes(r, t)?).map_err(|_| Error::InvalidValue)
    }
}

macro_rules! enumeration {
    ($ty:ident {$($variant:ident=$number:literal),* $(,)?}) => {
        impl Codec for $ty {
            fn put(&self,w:&mut Vec<u64>) { w.push(match self {$(Self::$variant=>$number),*}); }
            fn get(r:&mut Reader<'_>,_:&ResourceTable)->Result<Self>{match r.word()?{$($number=>Ok(Self::$variant),)* _=>Err(Error::InvalidValue)}}
        }
    };
}
enumeration!(TextureFormat {Bgra8Unorm=0,Bgra8UnormSrgb=1,Rgba8Unorm=2,Rgba8UnormSrgb=3,R8Unorm=4,Nv12=5,Depth32Float=6,Rg8Unorm=7});
enumeration!(FilterMode {Nearest=0,Linear=1});
enumeration!(AddressMode {ClampToEdge=0,Repeat=1,MirrorRepeat=2});
enumeration!(CompareFunction {Never=0,Less=1,Equal=2,LessEqual=3,Greater=4,NotEqual=5,GreaterEqual=6,Always=7});
enumeration!(VertexFormat {Sint32=0,Uint32=1,Float16x2=2,Float16x4=3,Sint16x4=4,Snorm10_10_10_2=5,Float32x2=6,Float32x3=7,Float32x4=8,Unorm8x4=9});
enumeration!(PrimitiveTopology {TriangleList=0,TriangleStrip=1,TriangleFan=2});
enumeration!(IndexFormat {Uint16=0,Uint32=1});
enumeration!(TextureSampleMode {Rgba=0,RgbIgnoreAlpha=1,AlphaMask=2});
enumeration!(BlendFactor {Zero=0,One=1,SourceAlpha=2,OneMinusSourceAlpha=3,DestinationAlpha=4,OneMinusDestinationAlpha=5});
enumeration!(BlendOp {Add=0,Subtract=1,ReverseSubtract=2});
enumeration!(CullMode {None=0,Front=1,Back=2});
enumeration!(FrontFace {Clockwise=0,CounterClockwise=1});
enumeration!(StorageTextureAccess {WriteOnly=0});
enumeration!(BufferAccess {CopyDestination=0,CopySource=1,Vertex=2,Index=3,Uniform=4,StorageRead=5,StorageReadWrite=6});
enumeration!(TextureAccess {CopyDestination=0,CopySource=1,Sampled=2,RenderAttachment=3,StorageWrite=4});
enumeration!(ShaderStage {Vertex=0,Fragment=1,Compute=2});
enumeration!(TextureViewDimension {D2=0,D2Array=1,Cube=2,D1=3,D1Array=4});
enumeration!(StoreOp {Store=0,DontCare=1});

macro_rules! flags {
    ($ty:ident, $($flag:ident),*) => {
        impl Codec for $ty {
            fn put(&self,w:&mut Vec<u64>){let mut bits=0u64; let mut shift=0; $(if self.contains(Self::$flag){bits|=1<<shift;} shift+=1;)* let _=shift; w.push(bits);}
            fn get(r:&mut Reader<'_>,_:&ResourceTable)->Result<Self>{let mut bits=r.word()?; let mut value=Self::empty(); $(if bits&1!=0{value=value.union(Self::$flag);} bits>>=1;)* if bits!=0{Err(Error::InvalidValue)}else{Ok(value)}}
        }
    };
}
flags!(
    TextureUsage,
    COPY_SRC,
    COPY_DST,
    SAMPLED,
    RENDER_ATTACHMENT,
    PRESENT,
    STORAGE
);
flags!(
    BufferUsage,
    COPY_SRC,
    COPY_DST,
    VERTEX,
    INDEX,
    UNIFORM,
    STORAGE
);
flags!(ShaderStages, VERTEX, FRAGMENT, COMPUTE);

// The following descriptors are exchanged on creation/change, not per draw.
macro_rules! record {
    ($ty:ident, $($get:ident),+ => $new:expr) => {
        impl Codec for $ty {
            fn put(&self,w:&mut Vec<u64>){$(self.$get().put(w);)+}
            fn get(r:&mut Reader<'_>,t:&ResourceTable)->Result<Self>{($new)(r,t)}
        }
    };
}
record!(Extent2D,width,height => |r:&mut Reader<'_>,t| Extent2D::new(r.value(t)?,r.value(t)?));
record!(PixelRect,x,y,width,height => |r:&mut Reader<'_>,t| PixelRect::new(r.value(t)?,r.value(t)?,r.value(t)?,r.value(t)?));
record!(BufferDesc,size,usage => |r:&mut Reader<'_>,t| BufferDesc::new(r.value(t)?,r.value(t)?));
record!(DepthState,format,compare,write_enabled => |r:&mut Reader<'_>,t| Ok(DepthState::new(r.value(t)?,r.value(t)?,r.value(t)?)));
record!(VertexAttribute,location,format,offset => |r:&mut Reader<'_>,t| Ok(VertexAttribute::new(r.value(t)?,r.value(t)?,r.value(t)?)));
record!(BlendComponent,source_factor,destination_factor,operation => |r:&mut Reader<'_>,t| Ok(BlendComponent::new(r.value(t)?,r.value(t)?,r.value(t)?)));
record!(BlendState,color,alpha => |r:&mut Reader<'_>,t| Ok(BlendState::new(r.value(t)?,r.value(t)?)));
record!(RasterState,cull_mode,front_face => |r:&mut Reader<'_>,t| Ok(RasterState::new(r.value(t)?,r.value(t)?)));
record!(DrawUniforms,transform,color => |r:&mut Reader<'_>,t| Ok(DrawUniforms::new(r.value(t)?,r.value(t)?)));
record!(PushConstantRange,stages,offset,size => |r:&mut Reader<'_>,t| PushConstantRange::new(r.value(t)?,r.value(t)?,r.value(t)?));
record!(BindGroupLayoutEntry,binding,visibility,ty => |r:&mut Reader<'_>,t| Ok(BindGroupLayoutEntry::new(r.value(t)?,r.value(t)?,r.value(t)?)));
record!(BindGroupEntry,binding,resource => |r:&mut Reader<'_>,t| Ok(BindGroupEntry::new(r.value(t)?,r.value(t)?)));
record!(ColorTargetState,format,blend,write_mask => |r:&mut Reader<'_>,t| ColorTargetState::new(r.value(t)?,r.value(t)?,r.value(t)?));

impl<T: Codec> Codec for &[T] {
    fn put(&self, w: &mut Vec<u64>) {
        self.len().put(w);
        for v in *self {
            v.put(w);
        }
    }
    fn get(_: &mut Reader<'_>, _: &ResourceTable) -> Result<Self> {
        Err(Error::InvalidDescriptor)
    }
}
impl Codec for ColorWriteMask {
    fn put(&self, w: &mut Vec<u64>) {
        self.bits().put(w);
    }
    fn get(r: &mut Reader<'_>, t: &ResourceTable) -> Result<Self> {
        Self::from_bits(r.value(t)?)
    }
}
// Pack floats in pairs. Large uniform arrays keep their original byte size.
fn put_floats<const N: usize>(v: [f32; N], w: &mut Vec<u64>) {
    for c in v.chunks(2) {
        w.push(
            u64::from(c[0].to_bits())
                | (u64::from(c.get(1).copied().unwrap_or(0.0).to_bits()) << 32),
        );
    }
}
fn get_floats<const N: usize>(r: &mut Reader<'_>) -> Result<[f32; N]> {
    let mut v = [0.0; N];
    for c in v.chunks_mut(2) {
        let w = r.word()?;
        c[0] = f32::from_bits(w as u32);
        if c.len() == 2 {
            c[1] = f32::from_bits((w >> 32) as u32);
        }
    }
    Ok(v)
}
impl Codec for Color {
    fn put(&self, w: &mut Vec<u64>) {
        put_floats(self.components(), w);
    }
    fn get(r: &mut Reader<'_>, _: &ResourceTable) -> Result<Self> {
        let [a, b, c, d] = get_floats(r)?;
        Self::rgba(a, b, c, d)
    }
}
impl Codec for Transform {
    fn put(&self, w: &mut Vec<u64>) {
        put_floats(self.columns(), w);
    }
    fn get(r: &mut Reader<'_>, _: &ResourceTable) -> Result<Self> {
        Self::from_columns(get_floats(r)?)
    }
}
impl Codec for Viewport {
    fn put(&self, w: &mut Vec<u64>) {
        put_floats(self.components(), w);
    }
    fn get(r: &mut Reader<'_>, _: &ResourceTable) -> Result<Self> {
        let [a, b, c, d, e, f] = get_floats(r)?;
        Self::new(a, b, c, d, e, f)
    }
}

impl Codec for LoadOp {
    fn put(&self, w: &mut Vec<u64>) {
        match self {
            Self::Load => w.push(0),
            Self::Clear(c) => {
                w.push(1);
                c.put(w)
            }
            Self::DontCare => w.push(2),
        }
    }
    fn get(r: &mut Reader<'_>, t: &ResourceTable) -> Result<Self> {
        match r.word()? {
            0 => Ok(Self::Load),
            1 => Ok(Self::Clear(r.value(t)?)),
            2 => Ok(Self::DontCare),
            _ => Err(Error::InvalidValue),
        }
    }
}
impl Codec for DepthLoadOp {
    fn put(&self, w: &mut Vec<u64>) {
        match self {
            Self::Load => w.push(0),
            Self::Clear(c) => {
                w.push(1);
                c.put(w)
            }
            Self::DontCare => w.push(2),
        }
    }
    fn get(r: &mut Reader<'_>, t: &ResourceTable) -> Result<Self> {
        match r.word()? {
            0 => Ok(Self::Load),
            1 => Ok(Self::Clear(r.value(t)?)),
            2 => Ok(Self::DontCare),
            _ => Err(Error::InvalidValue),
        }
    }
}
impl Codec for FragmentProgram {
    fn put(&self, w: &mut Vec<u64>) {
        match self {
            Self::Solid => w.push(0),
            Self::VertexColor => w.push(1),
            Self::Texture(m) => {
                w.push(2);
                m.put(w)
            }
            Self::TextureVertexColor(m) => {
                w.push(3);
                m.put(w)
            }
        }
    }
    fn get(r: &mut Reader<'_>, t: &ResourceTable) -> Result<Self> {
        match r.word()? {
            0 => Ok(Self::Solid),
            1 => Ok(Self::VertexColor),
            2 => Ok(Self::Texture(r.value(t)?)),
            3 => Ok(Self::TextureVertexColor(r.value(t)?)),
            _ => Err(Error::InvalidValue),
        }
    }
}

impl Codec for TextureDesc {
    fn put(&self, w: &mut Vec<u64>) {
        self.format().put(w);
        self.extent().put(w);
        self.usage().put(w);
        self.mip_level_count().put(w);
        self.array_layer_count().put(w);
        self.cube_compatible().put(w);
    }
    fn get(r: &mut Reader<'_>, t: &ResourceTable) -> Result<Self> {
        Self::new(r.value(t)?, r.value(t)?, r.value(t)?)?
            .with_mip_level_count(r.value(t)?)?
            .with_array_layer_count(r.value(t)?)?
            .with_cube_compatible(r.value(t)?)
    }
}
impl Codec for SamplerDesc {
    fn put(&self, w: &mut Vec<u64>) {
        self.min_filter().put(w);
        self.mag_filter().put(w);
        self.address_u().put(w);
        self.address_v().put(w);
        self.mip_filter().put(w);
        self.min_lod().put(w);
        self.max_lod().put(w);
        self.compare().put(w);
    }
    fn get(r: &mut Reader<'_>, t: &ResourceTable) -> Result<Self> {
        Ok(
            Self::new(r.value(t)?, r.value(t)?, r.value(t)?, r.value(t)?)
                .with_mip_filter(r.value(t)?, r.value(t)?, r.value(t)?)?
                .with_compare(r.value(t)?),
        )
    }
}
impl Codec for VertexBufferLayout {
    fn put(&self, w: &mut Vec<u64>) {
        self.stride().put(w);
        self.attributes().put(w);
    }
    fn get(r: &mut Reader<'_>, t: &ResourceTable) -> Result<Self> {
        Self::new(r.value(t)?, r.value(t)?)
    }
}
impl Codec for RenderPipelineDesc {
    fn put(&self, w: &mut Vec<u64>) {
        self.target_format().put(w);
        self.topology().put(w);
        self.vertex_buffer().put(w);
        self.fragment().put(w);
        self.blend().put(w);
        self.raster().put(w);
        self.depth_state().put(w);
    }
    fn get(r: &mut Reader<'_>, t: &ResourceTable) -> Result<Self> {
        let d = Self::new(
            r.value(t)?,
            r.value(t)?,
            r.value(t)?,
            r.value(t)?,
            r.value(t)?,
            r.value(t)?,
        )?;
        match r.value(t)? {
            Some(depth) => d.with_depth_state(depth),
            None => Ok(d),
        }
    }
}

macro_rules! identity {
    ($ty:ident,$resolve:ident) => {
        impl Codec for $ty {
            fn put(&self, w: &mut Vec<u64>) {
                self.index.put(w);
            }
            fn get(r: &mut Reader<'_>, t: &ResourceTable) -> Result<Self> {
                let id = Self {
                    owner: t.id,
                    index: r.value(t)?,
                };
                t.$resolve(id)?;
                Ok(id)
            }
        }
    };
}

identity!(SamplerId, sampler_ref);
identity!(RenderPipelineId, render_pipeline_ref);
identity!(ShaderModuleId, shader_module_ref);

identity!(ComputePipelineId, compute_pipeline_ref);
identity!(
    ProgrammableRenderPipelineId,
    programmable_render_pipeline_ref
);
impl Codec for BufferId {
    fn put(&self, w: &mut Vec<u64>) {
        self.index.put(w);
        self.generation.put(w);
    }
    fn get(r: &mut Reader<'_>, t: &ResourceTable) -> Result<Self> {
        let id = Self {
            owner: t.id,
            index: r.value(t)?,
            generation: r.value(t)?,
        };
        // Historical immutable bind groups can retain retired identities.
        // Resolving a command operand (or validating a group for use) checks
        // the generation, rather than reviving an old slot during import.
        Ok(id)
    }
}
impl Codec for TextureId {
    fn put(&self, w: &mut Vec<u64>) {
        self.index.put(w);
        self.generation.put(w);
    }
    fn get(r: &mut Reader<'_>, t: &ResourceTable) -> Result<Self> {
        let id = Self {
            owner: t.id,
            index: r.value(t)?,
            generation: r.value(t)?,
        };
        // Historical immutable bind groups can retain retired identities.
        // Resolving a command operand (or validating a group for use) checks
        // the generation, rather than reviving an old slot during import.
        Ok(id)
    }
}
impl Codec for BindGroupId {
    fn put(&self, w: &mut Vec<u64>) {
        self.index.put(w);
        self.generation.put(w);
    }
    fn get(r: &mut Reader<'_>, t: &ResourceTable) -> Result<Self> {
        let id = Self {
            owner: t.id,
            index: r.value(t)?,
            generation: r.value(t)?,
        };
        // Historical immutable bind groups can retain retired identities.
        // Resolving a command operand (or validating a group for use) checks
        // the generation, rather than reviving an old slot during import.
        Ok(id)
    }
}

impl Codec for ShaderModuleDesc {
    fn put(&self, w: &mut Vec<u64>) {
        match self.source() {
            ShaderSource::SpirV(v) => {
                w.push(0);
                v.put(w)
            }
            ShaderSource::Wgsl(v) => {
                w.push(1);
                v.put(w)
            }
        }
    }
    fn get(r: &mut Reader<'_>, t: &ResourceTable) -> Result<Self> {
        match r.word()? {
            0 => Self::spirv(r.value(t)?),
            1 => Self::wgsl(r.value(t)?),
            _ => Err(Error::InvalidValue),
        }
    }
}
impl Codec for ShaderEntryPoint {
    fn put(&self, w: &mut Vec<u64>) {
        self.module().put(w);
        self.stage().put(w);
        put_bytes(self.entry_point().as_bytes(), w);
    }
    fn get(r: &mut Reader<'_>, t: &ResourceTable) -> Result<Self> {
        Self::new(t.shader_module_ref(r.value(t)?)?, r.value(t)?, r.value(t)?)
    }
}
impl Codec for BindingType {
    fn put(&self, w: &mut Vec<u64>) {
        match self {
            Self::UniformBuffer => w.push(0),
            Self::StorageBuffer { read_only } => {
                w.push(1);
                read_only.put(w)
            }
            Self::SampledTexture => w.push(2),
            Self::SampledTextureView { dimension, depth } => {
                w.push(3);
                dimension.put(w);
                depth.put(w)
            }
            Self::Sampler => w.push(4),
            Self::ComparisonSampler => w.push(5),
            Self::StorageTexture { format, access } => {
                w.push(6);
                format.put(w);
                access.put(w)
            }
        }
    }
    fn get(r: &mut Reader<'_>, t: &ResourceTable) -> Result<Self> {
        Ok(match r.word()? {
            0 => Self::UniformBuffer,
            1 => Self::StorageBuffer {
                read_only: r.value(t)?,
            },
            2 => Self::SampledTexture,
            3 => Self::SampledTextureView {
                dimension: r.value(t)?,
                depth: r.value(t)?,
            },
            4 => Self::Sampler,
            5 => Self::ComparisonSampler,
            6 => Self::StorageTexture {
                format: r.value(t)?,
                access: r.value(t)?,
            },
            _ => return Err(Error::InvalidValue),
        })
    }
}
impl Codec for BindGroupLayoutDesc {
    fn put(&self, w: &mut Vec<u64>) {
        self.entries().put(w);
    }
    fn get(r: &mut Reader<'_>, t: &ResourceTable) -> Result<Self> {
        Self::new(r.value(t)?)
    }
}
impl Codec for PipelineLayoutDesc {
    fn put(&self, w: &mut Vec<u64>) {
        self.bind_groups().put(w);
        self.push_constant_ranges().put(w);
    }
    fn get(r: &mut Reader<'_>, t: &ResourceTable) -> Result<Self> {
        Self::new(r.value(t)?)?.with_push_constant_ranges(r.value(t)?)
    }
}
impl Codec for BindingResource {
    fn put(&self, w: &mut Vec<u64>) {
        match self {
            Self::Buffer {
                buffer,
                offset,
                size,
            } => {
                w.push(0);
                buffer.put(w);
                offset.put(w);
                size.put(w)
            }
            Self::Texture(v) => {
                w.push(1);
                v.put(w)
            }
            Self::Sampler(v) => {
                w.push(2);
                v.put(w)
            }
            Self::TextureView { texture, view } => {
                w.push(3);
                texture.put(w);
                view.format().put(w);
                view.dimension().put(w);
                view.base_mip_level().put(w);
                view.mip_level_count().put(w);
                view.base_array_layer().put(w);
                view.array_layer_count().put(w)
            }
        }
    }
    fn get(r: &mut Reader<'_>, t: &ResourceTable) -> Result<Self> {
        Ok(match r.word()? {
            0 => Self::Buffer {
                buffer: r.value(t)?,
                offset: r.value(t)?,
                size: r.value(t)?,
            },
            1 => Self::Texture(r.value(t)?),
            2 => Self::Sampler(r.value(t)?),
            3 => {
                let texture = r.value(t)?;
                let view = TextureViewDesc::new(
                    t.texture(t.texture_ref(texture)?)?,
                    r.value(t)?,
                    r.value(t)?,
                    r.value(t)?,
                    r.value(t)?,
                    r.value(t)?,
                    r.value(t)?,
                )?;
                Self::TextureView { texture, view }
            }
            _ => return Err(Error::InvalidValue),
        })
    }
}
impl Codec for BindGroupDesc {
    fn put(&self, w: &mut Vec<u64>) {
        self.layout().put(w);
        self.entries().put(w);
    }
    fn get(r: &mut Reader<'_>, t: &ResourceTable) -> Result<Self> {
        Self::from_abi(r.value(t)?, r.value(t)?)
    }
}
impl Codec for ComputePipelineDesc {
    fn put(&self, w: &mut Vec<u64>) {
        self.shader().put(w);
        self.layout().put(w);
    }
    fn get(r: &mut Reader<'_>, t: &ResourceTable) -> Result<Self> {
        Self::new(r.value(t)?, r.value(t)?)
    }
}
impl Codec for ProgrammableRenderPipelineDesc {
    fn put(&self, w: &mut Vec<u64>) {
        self.vertex().put(w);
        self.fragment().put(w);
        self.layout().put(w);
        self.target_format().put(w);
        self.topology().put(w);
        self.blend().put(w);
        self.raster().put(w);
        self.vertex_buffers().put(w);
        self.depth_state().put(w);
        let n = self.color_targets().count();
        n.put(w);
        for target in self.color_targets() {
            target.put(w)
        }
    }
    fn get(r: &mut Reader<'_>, t: &ResourceTable) -> Result<Self> {
        let mut d = Self::new(
            r.value(t)?,
            r.value(t)?,
            r.value(t)?,
            r.value(t)?,
            None,
            r.value(t)?,
            r.value(t)?,
            r.value(t)?,
        )?
        .with_vertex_buffers(r.value(t)?)?;
        if let Some(depth) = r.value(t)? {
            d = d.with_depth_state(depth)?;
        }
        d.with_color_targets(r.value(t)?)
    }
}
