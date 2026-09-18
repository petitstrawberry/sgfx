//! Bounded, semantic Naga IR to TGSI graphics shader lowering.
//!
//! This compiler deliberately rejects operations it cannot lower. It uses no
//! shader-name matching or fixed-shader substitution. The optional feature uses
//! Naga's standard-library frontends; the compatibility encoder remains no_std.

use alloc::{boxed::Box, format, string::String, vec, vec::Vec};
use core::fmt::{self, Write};
use naga::{
    AddressSpace, BinaryOperator as B, Binding, BuiltIn, Expression as E, Handle, Literal,
    ScalarKind as K, Statement as S, TypeInner as T,
};
use sgfx_core::ir::{
    ShaderModuleDesc, ShaderSource, ShaderStage, TextureViewDimension, VertexFormat,
};

type Result<T> = core::result::Result<T, ShaderCompileError>;

/// A parse, validation, unsupported-feature, or bounded-resource failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaderCompileError(pub String);
impl fmt::Display for ShaderCompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
fn unsupported(message: &str) -> ShaderCompileError {
    ShaderCompileError(format!("unsupported TGSI shader: {message}"))
}

/// Uniform-buffer register allocation for a single compiled stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UniformBufferBinding {
    /// SGFX bind-group index.
    pub group: u32,
    /// Binding within that group.
    pub binding: u32,
    /// First vec4 register in the stage's flattened inline constant bank.
    pub first_register: u32,
    /// Required byte span, including the source language's padding.
    pub size: u32,
}

/// A read-only storage buffer lowered to an integer buffer texture. The inline
/// constant register contains the descriptor's byte offset and byte length.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StorageBufferBinding {
    pub group: u32,
    pub binding: u32,
    pub slot: u32,
    pub first_register: u32,
}

/// Push-constant byte layout and its separate inline constant-register span.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PushConstantBinding {
    pub first_register: u32,
    pub size: u32,
}

/// The number of mip levels in a bound image, supplied by the resource
/// descriptor instead of TGSI TXQ (which requires an unavailable GL extension
/// on some VirGL hosts).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageQueryLevelsBinding {
    pub group: u32,
    pub binding: u32,
    pub first_register: u32,
}

/// A shader sampling operation's separate SGFX texture and sampler bindings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextureSamplerBinding {
    /// TGSI sampler/view slot within this stage.
    pub slot: u32,
    /// Texture group.
    pub image_group: u32,
    /// Texture binding.
    pub image_binding: u32,
    /// Sampler group.
    pub sampler_group: u32,
    /// Sampler binding.
    pub sampler_binding: u32,
    /// Texel loads have no sampler descriptor.
    pub uses_sampler: bool,
    pub dimension: TextureViewDimension,
    pub depth: bool,
    pub comparison: bool,
}

fn texture_target(binding: &TextureSamplerBinding) -> &'static str {
    match (binding.dimension, binding.comparison) {
        (TextureViewDimension::D2, false) => "2D",
        (TextureViewDimension::D2Array, false) => "2D_ARRAY",
        (TextureViewDimension::Cube, false) => "CUBE",
        (TextureViewDimension::D2, true) => "SHADOW2D",
        (TextureViewDimension::D2Array, true) => "SHADOW2D_ARRAY",
        (TextureViewDimension::Cube, true) => "SHADOWCUBE",
    }
}

/// Numeric type of a stage interface location.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IoScalar {
    Float,
    Sint,
    Uint,
}
/// Typed stage interface location used to validate pipeline linkage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IoLocation {
    pub location: u32,
    pub components: u8,
    pub scalar: IoScalar,
    pub interpolation: Option<naga::Interpolation>,
}

/// A shader ready for VirGL's TGSI-text shader object command.
#[derive(Clone, Debug)]
pub struct CompiledShader {
    /// Portable shader stage.
    pub stage: ShaderStage,
    /// Complete TGSI text, excluding the transport's terminating NUL.
    pub tgsi: String,
    /// Used uniform buffers, flattened into an inline constant bank per stage.
    pub uniform_buffers: Vec<UniformBufferBinding>,
    pub storage_buffers: Vec<StorageBufferBinding>,
    /// Push constants occupy a distinct span after every uniform buffer.
    pub push_constants: Option<PushConstantBinding>,
    pub image_query_levels: Vec<ImageQueryLevelsBinding>,
    /// Inline constant register added to the vertex instance ID on hosts
    /// without native base-instance draws.
    pub first_instance_register: Option<u32>,
    /// Sampling pairs bound independently for each shader stage.
    pub textures: Vec<TextureSamplerBinding>,
    /// Input locations consumed by the shader, for pipeline validation.
    pub input_locations: Vec<u32>,
    /// Output locations produced by the shader, for pipeline validation.
    pub output_locations: Vec<u32>,
    /// Vertex attributes required by this entry point (normalized formats may supply float4).
    pub vertex_inputs: Vec<(u32, VertexFormat)>,
    /// Typed location inputs, excluding builtins.
    pub inputs: Vec<IoLocation>,
    /// Typed location outputs, excluding builtins.
    pub outputs: Vec<IoLocation>,
}

/// Compile a vertex or fragment entry point, rejecting unsupported semantics.
///
/// SPIR-V is interpreted in SGFX's already-normalized coordinate convention,
/// matching the Vulkan frontend's output. Vertex depth is converted from SGFX's
/// zero-to-one clip range to Gallium's negative-one-to-one range at the output.
/// Uniform buffers keep their declared byte layout and are flattened into each
/// stage's bounded inline constant bank.
/// Compute, storage writes, loops, dynamic indexing stores and early returns
/// are currently rejected. See the tests for executable examples of the subset.
pub fn compile_shader(
    desc: &ShaderModuleDesc,
    stage: ShaderStage,
    entry_point: &str,
) -> Result<CompiledShader> {
    let naga_stage = match stage {
        ShaderStage::Vertex => naga::ShaderStage::Vertex,
        ShaderStage::Fragment => naga::ShaderStage::Fragment,
        ShaderStage::Compute => return Err(unsupported("compute stage")),
    };
    let module = parse_shader_module(desc)?;
    let info = validate_module(&module)?;
    let entry_index = module
        .entry_points
        .iter()
        .position(|entry| entry.stage == naga_stage && entry.name == entry_point)
        .ok_or_else(|| {
            ShaderCompileError(format!(
                "entry point {entry_point:?} is absent for {stage:?}"
            ))
        })?;
    Compiler::new(&module, &info, stage).compile(entry_index)
}

/// Parse and semantically validate a module without selecting a stage subset.
pub fn validate_shader_module(desc: &ShaderModuleDesc) -> Result<()> {
    validate_module(&parse_shader_module(desc)?).map(|_| ())
}
fn parse_shader_module(desc: &ShaderModuleDesc) -> Result<naga::Module> {
    match desc.source() {
        ShaderSource::Wgsl(source) => naga::front::wgsl::parse_str(source)
            .map_err(|error| ShaderCompileError(error.emit_to_string(source))),
        ShaderSource::SpirV(words) => naga::front::spv::Frontend::new(
            words.iter().copied(),
            &naga::front::spv::Options {
                adjust_coordinate_space: false,
                strict_capabilities: true,
                block_ctx_dump_prefix: None,
            },
        )
        .parse()
        .map_err(|error| ShaderCompileError(format!("SPIR-V: {error}"))),
    }
}
fn validate_module(module: &naga::Module) -> Result<naga::valid::ModuleInfo> {
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::PUSH_CONSTANT,
    )
    .validate(module)
    .map_err(|error| ShaderCompileError(format!("shader validation: {error}")))
}

fn uses_instance_index(
    module: &naga::Module,
    ty: Handle<naga::Type>,
    binding: Option<&Binding>,
) -> bool {
    if matches!(binding, Some(Binding::BuiltIn(BuiltIn::InstanceIndex))) {
        return true;
    }
    match &module.types[ty].inner {
        T::Struct { members, .. } => members
            .iter()
            .any(|member| uses_instance_index(module, member.ty, member.binding.as_ref())),
        _ => false,
    }
}

#[derive(Clone, Debug)]
struct Lane {
    register: String,
    component: usize,
}
impl Lane {
    fn src(&self) -> String {
        format!(
            "{}.{}",
            self.register,
            ["xxxx", "yyyy", "zzzz", "wwww"][self.component]
        )
    }
    fn dst(&self) -> String {
        format!("{}.{}", self.register, ["x", "y", "z", "w"][self.component])
    }
}
#[derive(Clone, Debug)]
enum Shape {
    Scalar(K),
    Vector(K, usize),
    Matrix(usize, usize),
    Aggregate(Vec<Shape>),
    Image(u32, u32, TextureViewDimension, bool),
    Sampler(u32, u32),
    StoragePointer {
        inner: Box<T>,
        slot: u32,
        metadata: u32,
        offset: Lane,
    },
}
impl Shape {
    fn len(&self) -> usize {
        match self {
            Self::Scalar(_) => 1,
            Self::Vector(_, n) => *n,
            Self::Matrix(c, r) => c * r,
            Self::Aggregate(items) => items.iter().map(Self::len).sum(),
            Self::Image(..) | Self::Sampler(..) | Self::StoragePointer { .. } => 0,
        }
    }
    fn kind(&self) -> Result<K> {
        match self {
            Self::Scalar(k) | Self::Vector(k, _) => Ok(*k),
            Self::Matrix(..) => Ok(K::Float),
            _ => Err(unsupported("arithmetic on aggregate")),
        }
    }
    fn element(&self, index: usize) -> Result<(usize, Shape)> {
        Ok(match self {
            Self::Vector(k, n) if index < *n => (index, Self::Scalar(*k)),
            Self::Matrix(c, r) if index < *c => (index * r, Self::Vector(K::Float, *r)),
            Self::Aggregate(items) if index < items.len() => (
                items[..index].iter().map(Self::len).sum(),
                items[index].clone(),
            ),
            _ => return Err(unsupported("out-of-range composite index")),
        })
    }
}
#[derive(Clone, Debug)]
struct Value {
    shape: Shape,
    lanes: Vec<Lane>,
    writable: bool,
    indirect: Option<Box<IndirectRead>>,
}

enum LoopControl {
    Body { break_flag: Lane },
    Continuing,
}
#[derive(Clone, Debug)]
struct IndirectRead {
    elements: Vec<Value>,
    index: Lane,
}
impl Value {
    fn element(&self, index: usize) -> Result<Self> {
        let (offset, shape) = self.shape.element(index)?;
        let indirect = self
            .indirect
            .as_ref()
            .map(|read| {
                Ok::<_, ShaderCompileError>(Box::new(IndirectRead {
                    elements: read
                        .elements
                        .iter()
                        .map(|value| value.element(index))
                        .collect::<Result<_>>()?,
                    index: read.index.clone(),
                }))
            })
            .transpose()?;
        Ok(Self {
            lanes: if indirect.is_none() {
                self.lanes[offset..offset + shape.len()].to_vec()
            } else {
                Vec::new()
            },
            shape,
            writable: self.writable,
            indirect,
        })
    }
}
struct Frame<'a> {
    function: &'a naga::Function,
    info: &'a naga::valid::FunctionInfo,
    arguments: Vec<Value>,
    locals: Vec<Value>,
    expressions: Vec<Option<Value>>,
    returned: Option<Value>,
    return_flag: Option<Lane>,
}

fn statement_may_return(statement: &S) -> bool {
    match statement {
        S::Return { .. } => true,
        S::Block(block) => block.iter().any(statement_may_return),
        S::If { accept, reject, .. } => {
            accept.iter().chain(reject.iter()).any(statement_may_return)
        }
        S::Loop {
            body, continuing, ..
        } => body
            .iter()
            .chain(continuing.iter())
            .any(statement_may_return),
        _ => false,
    }
}
struct Compiler<'a> {
    module: &'a naga::Module,
    info: &'a naga::valid::ModuleInfo,
    stage: ShaderStage,
    declarations: Vec<String>,
    immediates: Vec<String>,
    instructions: Vec<String>,
    temp_lanes: usize,
    globals: Vec<Option<Value>>,
    uniforms: Vec<UniformBufferBinding>,
    storage_buffers: Vec<StorageBufferBinding>,
    push_constants: Option<PushConstantBinding>,
    image_query_levels: Vec<ImageQueryLevelsBinding>,
    first_instance_register: Option<u32>,
    textures: Vec<TextureSamplerBinding>,
    input_locations: Vec<u32>,
    output_locations: Vec<u32>,
    call_depth: usize,
    inputs: Vec<IoLocation>,
    outputs: Vec<IoLocation>,
    vertex_inputs: Vec<(u32, VertexFormat)>,
}
impl<'a> Compiler<'a> {
    fn new(
        module: &'a naga::Module,
        info: &'a naga::valid::ModuleInfo,
        stage: ShaderStage,
    ) -> Self {
        Self {
            module,
            info,
            stage,
            declarations: Vec::new(),
            immediates: Vec::new(),
            instructions: Vec::new(),
            temp_lanes: 0,
            globals: vec![None; module.global_variables.len()],
            uniforms: Vec::new(),
            storage_buffers: Vec::new(),
            push_constants: None,
            image_query_levels: Vec::new(),
            first_instance_register: None,
            textures: Vec::new(),
            input_locations: Vec::new(),
            output_locations: Vec::new(),
            call_depth: 0,
            inputs: Vec::new(),
            outputs: Vec::new(),
            vertex_inputs: Vec::new(),
        }
    }
    fn shape(&self, inner: &T) -> Result<Shape> {
        let scalar = |s: naga::Scalar| {
            if s.width == 4 || s.kind == K::Bool {
                Ok(s.kind)
            } else {
                Err(unsupported("non-32-bit scalar"))
            }
        };
        Ok(match inner {
            T::Scalar(s) => Shape::Scalar(scalar(*s)?),
            T::Vector { size, scalar: s } => Shape::Vector(scalar(*s)?, *size as usize),
            T::Matrix {
                columns,
                rows,
                scalar: s,
            } if scalar(*s)? == K::Float => Shape::Matrix(*columns as usize, *rows as usize),
            T::Struct { members, .. } => Shape::Aggregate(
                members
                    .iter()
                    .map(|m| self.shape(&self.module.types[m.ty].inner))
                    .collect::<Result<_>>()?,
            ),
            T::Array {
                base,
                size: naga::ArraySize::Constant(n),
                ..
            } if n.get() <= 64 => {
                Shape::Aggregate(vec![
                    self.shape(&self.module.types[*base].inner)?;
                    n.get() as usize
                ])
            }
            T::Pointer { base, .. } => self.shape(&self.module.types[*base].inner)?,
            T::ValuePointer {
                size, scalar: s, ..
            } => match size {
                Some(n) => Shape::Vector(scalar(*s)?, *n as usize),
                None => Shape::Scalar(scalar(*s)?),
            },
            _ => {
                return Err(unsupported(
                    "type (only 32-bit scalars, vectors, matrices, bounded arrays and structs are supported)",
                ));
            }
        })
    }
    fn allocate(&mut self, shape: Shape, writable: bool) -> Result<Value> {
        if self.temp_lanes + shape.len() > 4096 {
            return Err(unsupported("more than 1024 temporary registers"));
        }
        let lanes = (0..shape.len())
            .map(|_| {
                let n = self.temp_lanes;
                self.temp_lanes += 1;
                Lane {
                    register: format!("TEMP[{}]", n / 4),
                    component: n % 4,
                }
            })
            .collect();
        Ok(Value {
            shape,
            lanes,
            writable,
            indirect: None,
        })
    }
    fn instruction(&mut self, opcode: &str, destination: &Lane, sources: &[&Lane]) -> Result<()> {
        if self.instructions.len() >= 16384 {
            return Err(unsupported("more than 16384 instructions"));
        }
        let mut text = format!("{opcode} {}", destination.dst());
        for source in sources {
            write!(&mut text, ", {}", source.src()).unwrap();
        }
        self.instructions.push(text);
        Ok(())
    }
    fn immediate(&mut self, literal: Literal) -> Result<Value> {
        let (kind, encoding, text) = match literal {
            Literal::F32(v) if v.is_finite() => (K::Float, "FLT32", format!("{v:.9e}")),
            Literal::I32(v) => (K::Sint, "INT32", format!("{v}")),
            Literal::U32(v) => (K::Uint, "UINT32", format!("{v}")),
            Literal::Bool(v) => (
                K::Bool,
                "UINT32",
                format!("{}", if v { u32::MAX } else { 0 }),
            ),
            _ => return Err(unsupported("literal width or non-finite float")),
        };
        let declaration = format!("{encoding} {{ {text}, {text}, {text}, {text} }}");
        let index = match self.immediates.iter().position(|v| v == &declaration) {
            Some(i) => i,
            None => {
                let i = self.immediates.len();
                self.immediates.push(declaration);
                i
            }
        };
        Ok(Value {
            shape: Shape::Scalar(kind),
            lanes: vec![Lane {
                register: format!("IMM[{index}]"),
                component: 0,
            }],
            writable: false,
            indirect: None,
        })
    }
    fn snapshot(&mut self, value: &Value) -> Result<Value> {
        if let Shape::StoragePointer {
            inner,
            slot,
            metadata,
            offset,
        } = &value.shape
        {
            return self.storage_load(inner, *slot, *metadata, offset);
        }
        if let Some(read) = &value.indirect {
            let result = self.allocate(value.shape.clone(), false)?;
            let zero = self.immediate(Literal::U32(0))?;
            for dst in &result.lanes {
                self.instruction("MOV", dst, &[&zero.lanes[0]])?;
            }
            let matches = self.allocate(Shape::Scalar(K::Bool), false)?;
            for (index, element) in read.elements.iter().enumerate() {
                let element = self.snapshot(element)?;
                let expected = self.immediate(Literal::U32(index as u32))?;
                self.instruction(
                    "USEQ",
                    &matches.lanes[0],
                    &[&read.index, &expected.lanes[0]],
                )?;
                for (dst, src) in result.lanes.iter().zip(&element.lanes) {
                    self.instruction("UCMP", dst, &[&matches.lanes[0], src, dst])?;
                }
            }
            return Ok(result);
        }
        let result = self.allocate(value.shape.clone(), false)?;
        for (dst, src) in result.lanes.iter().zip(&value.lanes) {
            self.instruction("MOV", dst, &[src])?;
        }
        Ok(result)
    }
    fn store(&mut self, pointer: &Value, value: &Value) -> Result<()> {
        if !pointer.writable || pointer.lanes.len() != value.lanes.len() {
            return Err(unsupported("store target"));
        }
        for (dst, src) in pointer.lanes.iter().zip(&value.lanes) {
            self.instruction("MOV", dst, &[src])?;
        }
        Ok(())
    }
    fn zero(&mut self, shape: Shape) -> Result<Value> {
        let zero = self.immediate(Literal::U32(0))?;
        Ok(Value {
            lanes: vec![zero.lanes[0].clone(); shape.len()],
            shape,
            writable: false,
            indirect: None,
        })
    }
    fn global_expression(&mut self, handle: Handle<E>) -> Result<Value> {
        match self.module.global_expressions[handle].clone() {
            E::Literal(v) => self.immediate(v),
            E::Constant(c) => self.global_expression(self.module.constants[c].init),
            E::ZeroValue(ty) => self.zero(self.shape(&self.module.types[ty].inner)?),
            E::Compose { ty, components } => {
                let mut lanes = Vec::new();
                for h in components {
                    lanes.extend(self.global_expression(h)?.lanes);
                }
                Ok(Value {
                    shape: self.shape(&self.module.types[ty].inner)?,
                    lanes,
                    writable: false,
                    indirect: None,
                })
            }
            E::Splat { size, value } => {
                let v = self.global_expression(value)?;
                Ok(Value {
                    shape: Shape::Vector(v.shape.kind()?, size as usize),
                    lanes: vec![v.lanes[0].clone(); size as usize],
                    writable: false,
                    indirect: None,
                })
            }
            _ => Err(unsupported("constant expression")),
        }
    }
    fn uniform_value(&self, ty: Handle<naga::Type>, offset: u32) -> Result<Value> {
        let inner = &self.module.types[ty].inner;
        let shape = self.shape(inner)?;
        let mut lanes = Vec::new();
        match inner {
            T::Scalar(_) | T::Vector { .. } => {
                for n in 0..shape.len() {
                    let address = offset + n as u32 * 4;
                    lanes.push(Lane {
                        register: format!("CONST[{}]", address / 16),
                        component: (address % 16 / 4) as usize,
                    });
                }
            }
            T::Matrix { columns, rows, .. } => {
                let stride = if *rows == naga::VectorSize::Bi { 8 } else { 16 };
                for c in 0..*columns as u32 {
                    for r in 0..*rows as u32 {
                        let address = offset + c * stride + r * 4;
                        lanes.push(Lane {
                            register: format!("CONST[{}]", address / 16),
                            component: (address % 16 / 4) as usize,
                        });
                    }
                }
            }
            T::Struct { members, .. } => {
                for member in members {
                    lanes.extend(self.uniform_value(member.ty, offset + member.offset)?.lanes);
                }
            }
            T::Array {
                base,
                size: naga::ArraySize::Constant(n),
                stride,
            } => {
                for i in 0..n.get() {
                    lanes.extend(self.uniform_value(*base, offset + i * stride)?.lanes);
                }
            }
            _ => return Err(unsupported("uniform layout")),
        }
        Ok(Value {
            shape,
            lanes,
            writable: false,
            indirect: None,
        })
    }
    fn io(
        &mut self,
        ty: Handle<naga::Type>,
        binding: Option<&Binding>,
        input: bool,
    ) -> Result<Value> {
        let shape = self.shape(&self.module.types[ty].inner)?;
        if let T::Struct { members, .. } = &self.module.types[ty].inner {
            let mut lanes = Vec::new();
            for member in members {
                lanes.extend(self.io(member.ty, member.binding.as_ref(), input)?.lanes);
            }
            return Ok(Value {
                shape,
                lanes,
                writable: !input,
                indirect: None,
            });
        }
        if !matches!(shape, Shape::Scalar(_) | Shape::Vector(..)) {
            return Err(unsupported("aggregate stage location"));
        }
        let binding = binding.ok_or_else(|| unsupported("unbound stage IO"))?;
        let register = match binding {
            Binding::Location {
                location,
                interpolation,
                sampling,
                second_blend_source,
            } => {
                if *location >= 16 || *second_blend_source {
                    return Err(unsupported("stage location limit or dual-source blending"));
                }
                let scalar = match shape.kind()? {
                    K::Float => IoScalar::Float,
                    K::Sint => IoScalar::Sint,
                    K::Uint => IoScalar::Uint,
                    _ => return Err(unsupported("boolean stage IO")),
                };
                let io = IoLocation {
                    location: *location,
                    components: shape.len() as u8,
                    scalar,
                    interpolation: *interpolation,
                };
                if input {
                    self.input_locations.push(*location);
                    self.inputs.push(io);
                } else {
                    self.output_locations.push(*location);
                    self.outputs.push(io);
                }
                if input && self.stage == ShaderStage::Vertex {
                    let format = match (scalar, shape.len()) {
                        (IoScalar::Sint, 1) => VertexFormat::Sint32,
                        (IoScalar::Uint, 1) => VertexFormat::Uint32,
                        (IoScalar::Sint, 4) => VertexFormat::Sint16x4,
                        (IoScalar::Float, 2) => VertexFormat::Float32x2,
                        (IoScalar::Float, 3) => VertexFormat::Float32x3,
                        (IoScalar::Float, 4) => VertexFormat::Float32x4,
                        _ => return Err(unsupported("vertex input format")),
                    };
                    self.vertex_inputs.push((*location, format));
                }
                let index = if !input && self.stage == ShaderStage::Vertex {
                    *location + 1
                } else {
                    *location
                };
                let reg = format!("{}[{index}]", if input { "IN" } else { "OUT" });
                let suffix = if self.stage == ShaderStage::Vertex && input {
                    String::new()
                } else if self.stage == ShaderStage::Fragment && !input {
                    format!(", COLOR[{location}]")
                } else {
                    let mut suffix = format!(", GENERIC[{location}]");
                    if input {
                        suffix.push_str(
                            match interpolation.unwrap_or(naga::Interpolation::Perspective) {
                                naga::Interpolation::Perspective => ", PERSPECTIVE",
                                naga::Interpolation::Linear => ", LINEAR",
                                naga::Interpolation::Flat => ", CONSTANT",
                            },
                        );
                        if !matches!(sampling, None | Some(naga::Sampling::Center)) {
                            return Err(unsupported("centroid/sample interpolation"));
                        }
                    }
                    suffix
                };
                self.declarations.push(format!("DCL {reg}{suffix}"));
                reg
            }
            Binding::BuiltIn(BuiltIn::Position { .. })
                if !input && self.stage == ShaderStage::Vertex =>
            {
                self.declarations.push("DCL OUT[0], POSITION".into());
                "OUT[0]".into()
            }
            Binding::BuiltIn(BuiltIn::Position { .. })
                if input && self.stage == ShaderStage::Fragment =>
            {
                self.declarations
                    .push("DCL IN[16], POSITION, LINEAR".into());
                "IN[16]".into()
            }
            Binding::BuiltIn(BuiltIn::VertexIndex)
                if input && self.stage == ShaderStage::Vertex =>
            {
                self.declarations.push("DCL SV[0], VERTEXID".into());
                "SV[0]".into()
            }
            Binding::BuiltIn(BuiltIn::InstanceIndex)
                if input && self.stage == ShaderStage::Vertex =>
            {
                if !matches!(&shape, Shape::Scalar(K::Uint)) {
                    return Err(unsupported("instance index type"));
                }
                self.declarations.push("DCL SV[1], INSTANCEID".into());
                let register = self
                    .first_instance_register
                    .ok_or_else(|| unsupported("instance index constant"))?;
                let instance = Lane {
                    register: "SV[1]".into(),
                    component: 0,
                };
                let base = Lane {
                    register: format!("CONST[{register}]"),
                    component: 0,
                };
                let adjusted = self.allocate(shape, false)?;
                self.instruction("UADD", &adjusted.lanes[0], &[&instance, &base])?;
                return Ok(adjusted);
            }
            _ => return Err(unsupported("stage builtin")),
        };
        Ok(Value {
            lanes: (0..shape.len())
                .map(|component| Lane {
                    register: register.clone(),
                    component,
                })
                .collect(),
            shape,
            writable: !input,
            indirect: None,
        })
    }
    fn compile(mut self, index: usize) -> Result<CompiledShader> {
        let entry = &self.module.entry_points[index];
        let entry_info = self.info.get_entry_point(index);
        let uses_query_levels = self.module.functions.iter().any(|(_, function)| {
            function.expressions.iter().any(|(_, expression)| {
                matches!(
                    expression,
                    E::ImageQuery {
                        query: naga::ImageQuery::NumLevels,
                        ..
                    }
                )
            })
        }) || entry.function.expressions.iter().any(|(_, expression)| {
            matches!(
                expression,
                E::ImageQuery {
                    query: naga::ImageQuery::NumLevels,
                    ..
                }
            )
        });
        let mut uniform_handles = Vec::new();
        let mut storage_handles = Vec::new();
        let mut image_bindings = Vec::new();
        let mut push_constant_handle = None;
        for (handle, global) in self.module.global_variables.iter() {
            if entry_info[handle].is_empty() {
                continue;
            }
            match global.space {
                AddressSpace::Handle => {
                    let binding = global
                        .binding
                        .as_ref()
                        .ok_or_else(|| unsupported("unbound texture/sampler"))?;
                    let shape = match self.module.types[global.ty].inner {
                        T::Image {
                            dim,
                            arrayed,
                            class,
                        } => {
                            let dimension = match (dim, arrayed) {
                                (naga::ImageDimension::D2, false) => TextureViewDimension::D2,
                                (naga::ImageDimension::D2, true) => TextureViewDimension::D2Array,
                                (naga::ImageDimension::Cube, false) => TextureViewDimension::Cube,
                                _ => return Err(unsupported("image dimension")),
                            };
                            let depth = match class {
                                naga::ImageClass::Sampled {
                                    kind: K::Float,
                                    multi: false,
                                } => false,
                                naga::ImageClass::Depth { multi: false } => true,
                                _ => return Err(unsupported("image class")),
                            };
                            if uses_query_levels {
                                image_bindings.push((binding.group, binding.binding));
                            }
                            Shape::Image(binding.group, binding.binding, dimension, depth)
                        }
                        T::Sampler { .. } => Shape::Sampler(binding.group, binding.binding),
                        _ => return Err(unsupported("opaque resource type")),
                    };
                    self.globals[handle.index()] = Some(Value {
                        shape,
                        lanes: Vec::new(),
                        writable: false,
                        indirect: None,
                    });
                }
                AddressSpace::Uniform => {
                    let binding = global
                        .binding
                        .as_ref()
                        .ok_or_else(|| unsupported("uniform without resource binding"))?;
                    uniform_handles.push((binding.group, binding.binding, handle));
                }
                AddressSpace::Storage { access }
                    if !access.contains(naga::StorageAccess::STORE) =>
                {
                    let binding = global
                        .binding
                        .as_ref()
                        .ok_or_else(|| unsupported("storage buffer without binding"))?;
                    storage_handles.push((binding.group, binding.binding, handle));
                }
                AddressSpace::PushConstant => {
                    if push_constant_handle.replace(handle).is_some() {
                        return Err(unsupported("multiple push-constant blocks"));
                    }
                }
                AddressSpace::Private => {}
                _ => {
                    return Err(unsupported(
                        "resource address space or writable storage buffer",
                    ));
                }
            }
        }
        uniform_handles.sort_by_key(|v| (v.0, v.1));
        if uniform_handles.len() > 15 {
            return Err(unsupported("more than 15 uniform buffers in a stage"));
        }
        let mut layouter = naga::proc::Layouter::default();
        layouter
            .update(self.module.to_ctx())
            .map_err(|e| ShaderCompileError(format!("uniform layout: {e}")))?;
        let mut next_constant_register = 0u32;
        for (group, binding, handle) in uniform_handles {
            let ty = self.module.global_variables[handle].ty;
            let size = layouter[ty].size;
            if size == 0 || size > 16384 {
                return Err(unsupported("uniform buffer size"));
            }
            let register_count = size.div_ceil(16);
            let first_register = next_constant_register;
            next_constant_register = next_constant_register
                .checked_add(register_count)
                .filter(|&register| register <= 1024)
                .ok_or_else(|| unsupported("more than 1024 inline constant registers"))?;
            self.uniforms.push(UniformBufferBinding {
                group,
                binding,
                first_register,
                size,
            });
            self.globals[handle.index()] = Some(self.uniform_value(ty, first_register * 16)?);
        }
        if let Some(handle) = push_constant_handle {
            let ty = self.module.global_variables[handle].ty;
            let size = layouter[ty].size;
            if size == 0 || size > 128 {
                return Err(unsupported("push-constant block exceeds 128 bytes"));
            }
            let first_register = next_constant_register;
            next_constant_register = next_constant_register
                .checked_add(size.div_ceil(16))
                .filter(|&register| register <= 1024)
                .ok_or_else(|| unsupported("more than 1024 inline constant registers"))?;
            self.push_constants = Some(PushConstantBinding {
                first_register,
                size,
            });
            self.globals[handle.index()] = Some(self.uniform_value(ty, first_register * 16)?);
        }
        if storage_handles.len() > 4 {
            return Err(unsupported("more than four storage buffers in a stage"));
        }
        storage_handles.sort_by_key(|value| (value.0, value.1));
        for (group, binding, handle) in storage_handles {
            let slot = self.storage_buffers.len() as u32;
            let metadata = next_constant_register;
            next_constant_register += 1;
            if next_constant_register > 1024 {
                return Err(unsupported("more than 1024 inline constant registers"));
            }
            self.declarations.push(format!("DCL SAMP[{slot}]"));
            self.declarations
                .push(format!("DCL SVIEW[{slot}], BUFFER, UINT"));
            self.storage_buffers.push(StorageBufferBinding {
                group,
                binding,
                slot,
                first_register: metadata,
            });
            let offset = self.immediate(Literal::U32(0))?.lanes[0].clone();
            self.globals[handle.index()] = Some(Value {
                shape: Shape::StoragePointer {
                    inner: Box::new(
                        self.module.types[self.module.global_variables[handle].ty]
                            .inner
                            .clone(),
                    ),
                    slot,
                    metadata,
                    offset,
                },
                lanes: Vec::new(),
                writable: false,
                indirect: None,
            });
        }
        if self.stage == ShaderStage::Vertex
            && entry.function.arguments.iter().any(|argument| {
                uses_instance_index(self.module, argument.ty, argument.binding.as_ref())
            })
        {
            self.first_instance_register = Some(next_constant_register);
            next_constant_register = next_constant_register
                .checked_add(1)
                .filter(|&register| register <= 1024)
                .ok_or_else(|| unsupported("more than 1024 inline constant registers"))?;
        }
        image_bindings.sort_unstable();
        image_bindings.dedup();
        for (group, binding) in image_bindings {
            let first_register = next_constant_register;
            next_constant_register = next_constant_register
                .checked_add(1)
                .filter(|&register| register <= 1024)
                .ok_or_else(|| unsupported("more than 1024 inline constant registers"))?;
            self.image_query_levels.push(ImageQueryLevelsBinding {
                group,
                binding,
                first_register,
            });
        }
        if next_constant_register == 1 {
            self.declarations.push("DCL CONST[0]".into());
        } else if next_constant_register > 1 {
            self.declarations
                .push(format!("DCL CONST[0..{}]", next_constant_register - 1));
        }
        for (handle, global) in self.module.global_variables.iter() {
            if global.space != AddressSpace::Private || entry_info[handle].is_empty() {
                continue;
            }
            let target = self.allocate(self.shape(&self.module.types[global.ty].inner)?, true)?;
            let init = match global.init {
                Some(h) => self.global_expression(h)?,
                None => self.zero(target.shape.clone())?,
            };
            self.store(&target, &init)?;
            self.globals[handle.index()] = Some(target);
        }
        let mut arguments = Vec::new();
        for arg in &entry.function.arguments {
            arguments.push(self.io(arg.ty, arg.binding.as_ref(), true)?);
        }
        let output = entry
            .function
            .result
            .as_ref()
            .map(|r| self.io(r.ty, r.binding.as_ref(), false))
            .transpose()?;
        let returned = self.function(&entry.function, entry_info, arguments)?;
        if let Some(output) = output {
            self.store(
                &output,
                &returned.ok_or_else(|| unsupported("entry point without result"))?,
            )?;
        }
        if self.stage == ShaderStage::Vertex {
            // Gallium's default clipping is [-w,w], while SGFX uses [0,w].
            let z = Lane {
                register: "OUT[0]".into(),
                component: 2,
            };
            let w = Lane {
                register: "OUT[0]".into(),
                component: 3,
            };
            let two = self.immediate(Literal::F32(2.0))?;
            let temp = self.allocate(Shape::Scalar(K::Float), false)?;
            self.instruction("MUL", &temp.lanes[0], &[&z, &two.lanes[0]])?;
            self.instruction("SUB", &z, &[&temp.lanes[0], &w])?;
        }
        let mut tgsi = String::from(if self.stage == ShaderStage::Vertex {
            "VERT\n"
        } else {
            "FRAG\nPROPERTY FS_COORD_ORIGIN UPPER_LEFT\nPROPERTY FS_COORD_PIXEL_CENTER HALF_INTEGER\n"
        });
        for line in &self.declarations {
            writeln!(&mut tgsi, "{line}").unwrap();
        }
        if self.temp_lanes > 0 {
            writeln!(
                &mut tgsi,
                "DCL TEMP[0..{}]",
                self.temp_lanes.div_ceil(4) - 1
            )
            .unwrap();
        }
        for (index, line) in self.immediates.iter().enumerate() {
            writeln!(&mut tgsi, "IMM[{index}] {line}").unwrap();
        }
        for (index, line) in self.instructions.iter().enumerate() {
            writeln!(&mut tgsi, "{index}: {line}").unwrap();
        }
        writeln!(&mut tgsi, "{}: END", self.instructions.len()).unwrap();
        Ok(CompiledShader {
            stage: self.stage,
            tgsi,
            uniform_buffers: self.uniforms,
            storage_buffers: self.storage_buffers,
            push_constants: self.push_constants,
            image_query_levels: self.image_query_levels,
            first_instance_register: self.first_instance_register,
            textures: self.textures,
            input_locations: self.input_locations,
            output_locations: self.output_locations,
            vertex_inputs: self.vertex_inputs,
            inputs: self.inputs,
            outputs: self.outputs,
        })
    }
    fn function(
        &mut self,
        function: &'a naga::Function,
        info: &'a naga::valid::FunctionInfo,
        arguments: Vec<Value>,
    ) -> Result<Option<Value>> {
        self.call_depth += 1;
        if self.call_depth > 32 {
            return Err(unsupported("call nesting greater than 32"));
        }
        let needs_return_flag =
            function
                .body
                .iter()
                .enumerate()
                .any(|(index, statement)| match statement {
                    S::Return { .. } => index + 1 != function.body.len(),
                    _ => statement_may_return(statement),
                });
        let return_flag = if needs_return_flag {
            let flag = self.allocate(Shape::Scalar(K::Uint), false)?;
            let zero = self.immediate(Literal::U32(0))?;
            self.instruction("MOV", &flag.lanes[0], &[&zero.lanes[0]])?;
            Some(flag.lanes[0].clone())
        } else {
            None
        };
        let returned = if needs_return_flag {
            function
                .result
                .as_ref()
                .map(|result| {
                    let value =
                        self.allocate(self.shape(&self.module.types[result.ty].inner)?, true)?;
                    let zero = self.zero(value.shape.clone())?;
                    self.store(&value, &zero)?;
                    Ok::<_, ShaderCompileError>(value)
                })
                .transpose()?
        } else {
            None
        };
        let mut frame = Frame {
            function,
            info,
            arguments,
            locals: Vec::new(),
            expressions: vec![None; function.expressions.len()],
            returned,
            return_flag,
        };
        for (_, local) in function.local_variables.iter() {
            frame
                .locals
                .push(self.allocate(self.shape(&self.module.types[local.ty].inner)?, true)?);
        }
        for (handle, local) in function.local_variables.iter() {
            let destination = frame.locals[handle.index()].clone();
            let init = match local.init {
                Some(h) => self.expression(&mut frame, h)?,
                None => self.zero(destination.shape.clone())?,
            };
            self.store(&destination, &init)?;
        }
        self.block(&mut frame, &function.body, 0, None)?;
        self.call_depth -= 1;
        Ok(frame.returned)
    }
    fn block(
        &mut self,
        frame: &mut Frame<'a>,
        block: &naga::Block,
        conditional_depth: usize,
        loop_control: Option<&LoopControl>,
    ) -> Result<()> {
        let mut return_guards = 0;
        for (index, statement) in block.iter().enumerate() {
            match statement {
                S::Emit(range) => {
                    for handle in range.clone() {
                        self.expression(frame, handle)?;
                    }
                }
                S::Block(block) => self.block(frame, block, conditional_depth, loop_control)?,
                S::Store { pointer, value } => {
                    let pointer = self.expression(frame, *pointer)?;
                    let value = self.expression(frame, *value)?;
                    self.store(&pointer, &value)?;
                }
                S::Return { value } => {
                    if let Some(flag) = frame.return_flag.clone() {
                        if let Some(value) = value {
                            let value = self.expression(frame, *value)?;
                            let destination = frame
                                .returned
                                .as_ref()
                                .ok_or_else(|| unsupported("return value without result"))?;
                            self.store(destination, &value)?;
                        }
                        let one = self.immediate(Literal::U32(1))?;
                        self.instruction("MOV", &flag, &[&one.lanes[0]])?;
                    } else {
                        if conditional_depth != 0 || index + 1 != block.len() {
                            return Err(unsupported("early/conditional return"));
                        }
                        frame.returned = value.map(|h| self.expression(frame, h)).transpose()?;
                    }
                }
                S::Call {
                    function,
                    arguments,
                    result,
                } => {
                    let mut args = Vec::new();
                    for h in arguments {
                        args.push(self.expression(frame, *h)?);
                    }
                    let returned = self.function(
                        &self.module.functions[*function],
                        &self.info[*function],
                        args,
                    )?;
                    if let Some(h) = result {
                        frame.expressions[h.index()] =
                            Some(returned.ok_or_else(|| unsupported("missing call result"))?);
                    }
                }
                S::If {
                    condition,
                    accept,
                    reject,
                } => {
                    let condition = self.expression(frame, *condition)?;
                    self.instructions
                        .push(format!("UIF {}", condition.lanes[0].src()));
                    self.block(frame, accept, conditional_depth + 1, loop_control)?;
                    if !reject.is_empty() {
                        self.instructions.push("ELSE".into());
                        self.block(frame, reject, conditional_depth + 1, loop_control)?;
                    }
                    self.instructions.push("ENDIF".into());
                }
                S::Loop {
                    body,
                    continuing,
                    break_if,
                } => {
                    let break_flag = self.allocate(Shape::Scalar(K::Uint), false)?;
                    let zero = self.immediate(Literal::U32(0))?;
                    self.instructions.push("BGNLOOP".into());
                    self.instruction("MOV", &break_flag.lanes[0], &[&zero.lanes[0]])?;
                    // The inner loop lets `continue` leave the body and execute
                    // Naga's continuing block before the next outer iteration.
                    self.instructions.push("BGNLOOP".into());
                    let body_control = LoopControl::Body {
                        break_flag: break_flag.lanes[0].clone(),
                    };
                    self.block(frame, body, conditional_depth + 1, Some(&body_control))?;
                    self.instructions.push("BRK".into());
                    self.instructions.push("ENDLOOP".into());
                    if let Some(flag) = &frame.return_flag {
                        self.instructions.push(format!("UIF {}", flag.src()));
                        self.instructions.push("BRK".into());
                        self.instructions.push("ENDIF".into());
                    }
                    self.instructions
                        .push(format!("UIF {}", break_flag.lanes[0].src()));
                    self.instructions.push("BRK".into());
                    self.instructions.push("ENDIF".into());
                    self.block(
                        frame,
                        continuing,
                        conditional_depth + 1,
                        Some(&LoopControl::Continuing),
                    )?;
                    if let Some(condition) = break_if {
                        let condition = self.expression(frame, *condition)?;
                        self.instructions
                            .push(format!("UIF {}", condition.lanes[0].src()));
                        self.instructions.push("BRK".into());
                        self.instructions.push("ENDIF".into());
                    }
                    self.instructions.push("ENDLOOP".into());
                }
                S::Break => match loop_control {
                    Some(LoopControl::Body { break_flag }) => {
                        let one = self.immediate(Literal::U32(1))?;
                        self.instruction("MOV", break_flag, &[&one.lanes[0]])?;
                        self.instructions.push("BRK".into());
                    }
                    Some(LoopControl::Continuing) => self.instructions.push("BRK".into()),
                    None => return Err(unsupported("break outside loop")),
                },
                S::Continue => match loop_control {
                    Some(LoopControl::Body { .. }) => self.instructions.push("BRK".into()),
                    Some(LoopControl::Continuing) => self.instructions.push("CONT".into()),
                    None => return Err(unsupported("continue outside loop")),
                },
                S::Kill if self.stage == ShaderStage::Fragment => {
                    self.instructions.push("KILL".into())
                }
                _ => {
                    return Err(unsupported(
                        "statement (switch, atomics, barriers and image stores are not supported)",
                    ));
                }
            }
            if index + 1 != block.len() && statement_may_return(statement) {
                if let Some(flag) = &frame.return_flag {
                    self.instructions.push(format!("UIF {}", flag.src()));
                    self.instructions.push("ELSE".into());
                    return_guards += 1;
                }
            }
        }
        for _ in 0..return_guards {
            self.instructions.push("ENDIF".into());
        }
        Ok(())
    }
    fn literal_index(literal: Literal) -> Option<u32> {
        match literal {
            Literal::U32(value) => Some(value),
            Literal::I32(value) if value >= 0 => Some(value as u32),
            _ => None,
        }
    }
    fn global_constant_index(&self, handle: Handle<E>, depth: usize) -> Option<u32> {
        if depth >= 16 {
            return None;
        }
        match self.module.global_expressions[handle].clone() {
            E::Literal(literal) => Self::literal_index(literal),
            E::Constant(constant) => {
                self.global_constant_index(self.module.constants[constant].init, depth + 1)
            }
            _ => None,
        }
    }
    fn constant_index(&self, frame: &Frame<'a>, handle: Handle<E>) -> Option<u32> {
        match frame.function.expressions[handle].clone() {
            E::Literal(literal) => Self::literal_index(literal),
            E::Constant(constant) => {
                self.global_constant_index(self.module.constants[constant].init, 0)
            }
            _ => None,
        }
    }
    fn expression(&mut self, frame: &mut Frame<'a>, handle: Handle<E>) -> Result<Value> {
        if let Some(v) = &frame.expressions[handle.index()] {
            return Ok(v.clone());
        }
        // Opaque image/sampler values keep the binding carried by their source.
        // Their Naga type alone cannot identify an SGFX descriptor binding.
        let direct = match frame.function.expressions[handle].clone() {
            E::GlobalVariable(h) => Some(
                self.globals[h.index()]
                    .clone()
                    .ok_or_else(|| unsupported("unused/unsupported global"))?,
            ),
            E::FunctionArgument(i) => Some(frame.arguments[i as usize].clone()),
            E::Load { pointer } => {
                let value = self.expression(frame, pointer)?;
                Some(self.snapshot(&value)?)
            }
            E::AccessIndex { base, index } => {
                let base = self.expression(frame, base)?;
                if matches!(base.shape, Shape::StoragePointer { .. }) {
                    Some(self.storage_access(&base, Some(index), None)?)
                } else {
                    None
                }
            }
            E::Access { base, index } => {
                let base = self.expression(frame, base)?;
                if matches!(base.shape, Shape::StoragePointer { .. }) {
                    let constant = self.constant_index(frame, index);
                    let index = self.expression(frame, index)?;
                    Some(self.storage_access(&base, constant, Some(&index))?)
                } else {
                    None
                }
            }
            _ => None,
        };
        if let Some(value) = direct {
            frame.expressions[handle.index()] = Some(value.clone());
            return Ok(value);
        }
        let shape = self.shape(frame.info[handle].ty.inner_with(&self.module.types))?;
        let value = match frame.function.expressions[handle].clone() {
            E::Literal(v) => self.immediate(v)?,
            E::Constant(c) => self.global_expression(self.module.constants[c].init)?,
            E::ZeroValue(_) => self.zero(shape)?,
            E::FunctionArgument(i) => frame.arguments[i as usize].clone(),
            E::GlobalVariable(h) => self.globals[h.index()]
                .clone()
                .ok_or_else(|| unsupported("unused/unsupported global"))?,
            E::LocalVariable(h) => frame.locals[h.index()].clone(),
            E::Load { pointer } => {
                let v = self.expression(frame, pointer)?;
                self.snapshot(&v)?
            }
            E::AccessIndex { base, index } => {
                self.expression(frame, base)?.element(index as usize)?
            }
            E::Access { base, index } => {
                let base = self.expression(frame, base)?;
                if let Some(index) = self.constant_index(frame, index) {
                    base.element(index as usize)?
                } else {
                    let index = self.expression(frame, index)?;
                    if !matches!(index.shape, Shape::Scalar(K::Sint | K::Uint)) {
                        return Err(unsupported("non-integer composite index"));
                    }
                    let count = match &base.shape {
                        Shape::Vector(_, count) | Shape::Matrix(count, _) => *count,
                        Shape::Aggregate(elements) => elements.len(),
                        _ => return Err(unsupported("dynamic indexing of non-composite")),
                    };
                    let elements = (0..count)
                        .map(|i| base.element(i))
                        .collect::<Result<Vec<_>>>()?;
                    let value = Value {
                        shape: elements
                            .first()
                            .ok_or_else(|| unsupported("empty composite"))?
                            .shape
                            .clone(),
                        lanes: Vec::new(),
                        writable: false,
                        indirect: Some(Box::new(IndirectRead {
                            elements,
                            index: index.lanes[0].clone(),
                        })),
                    };
                    // Pointer indexing must read the registers at Load, after
                    // preceding stores. A value index is evaluated immediately.
                    if matches!(
                        frame.info[handle].ty.inner_with(&self.module.types),
                        T::Pointer { .. } | T::ValuePointer { .. }
                    ) {
                        value
                    } else {
                        self.snapshot(&value)?
                    }
                }
            }
            E::Compose { components, .. } => {
                let mut lanes = Vec::new();
                for h in components {
                    lanes.extend(self.expression(frame, h)?.lanes);
                }
                if lanes.len() != shape.len() {
                    return Err(unsupported("composite component count"));
                }
                Value {
                    shape,
                    lanes,
                    writable: false,
                    indirect: None,
                }
            }
            E::Splat { size, value } => {
                let v = self.expression(frame, value)?;
                Value {
                    shape,
                    lanes: vec![v.lanes[0].clone(); size as usize],
                    writable: false,
                    indirect: None,
                }
            }
            E::Swizzle {
                size,
                vector,
                pattern,
            } => {
                let v = self.expression(frame, vector)?;
                Value {
                    shape,
                    lanes: pattern[..size as usize]
                        .iter()
                        .map(|p| v.lanes[*p as usize].clone())
                        .collect(),
                    writable: false,
                    indirect: None,
                }
            }
            E::Binary { op, left, right } => {
                let a = self.expression(frame, left)?;
                let b = self.expression(frame, right)?;
                self.binary(op, &a, &b, shape)?
            }
            E::Unary { op, expr } => {
                let v = self.expression(frame, expr)?;
                let result = self.allocate(shape, false)?;
                match op {
                    naga::UnaryOperator::Negate if v.shape.kind()? == K::Float => {
                        let zero = self.immediate(Literal::F32(0.0))?;
                        for (dst, src) in result.lanes.iter().zip(&v.lanes) {
                            self.instruction("SUB", dst, &[&zero.lanes[0], src])?;
                        }
                    }
                    naga::UnaryOperator::Negate => {
                        for (dst, src) in result.lanes.iter().zip(&v.lanes) {
                            self.instruction("INEG", dst, &[src])?;
                        }
                    }
                    naga::UnaryOperator::LogicalNot | naga::UnaryOperator::BitwiseNot => {
                        for (dst, src) in result.lanes.iter().zip(&v.lanes) {
                            self.instruction("NOT", dst, &[src])?;
                        }
                    }
                }
                result
            }
            E::As {
                expr,
                kind,
                convert,
            } => {
                let v = self.expression(frame, expr)?;
                if convert.is_none() || v.shape.kind()? == kind {
                    Value {
                        shape,
                        lanes: v.lanes,
                        writable: false,
                        indirect: None,
                    }
                } else {
                    if convert != Some(4) {
                        return Err(unsupported("conversion width"));
                    }
                    let opcode = match (v.shape.kind()?, kind) {
                        (K::Float, K::Sint) => "F2I",
                        (K::Float, K::Uint) => "F2U",
                        (K::Sint, K::Float) => "I2F",
                        (K::Uint, K::Float) => "U2F",
                        (K::Sint, K::Uint) | (K::Uint, K::Sint) => "MOV",
                        _ => return Err(unsupported("scalar conversion")),
                    };
                    self.unary_instruction(opcode, &v, shape)?
                }
            }
            E::Select {
                condition,
                accept,
                reject,
            } => {
                let c = self.expression(frame, condition)?;
                let a = self.expression(frame, accept)?;
                let b = self.expression(frame, reject)?;
                let result = self.allocate(shape, false)?;
                for (i, dst) in result.lanes.iter().enumerate() {
                    self.instruction(
                        "UCMP",
                        dst,
                        &[&c.lanes[i % c.lanes.len()], &a.lanes[i], &b.lanes[i]],
                    )?;
                }
                result
            }
            E::ImageSample {
                image,
                sampler,
                gather: None,
                coordinate,
                array_index,
                offset: None,
                level,
                depth_ref,
            } => {
                let image = self.expression(frame, image)?;
                let sampler = self.expression(frame, sampler)?;
                let (Shape::Image(group, binding, dimension, depth), Shape::Sampler(sg, sb)) =
                    (&image.shape, &sampler.shape)
                else {
                    return Err(unsupported("sampling handles"));
                };
                if depth_ref.is_some() && !depth {
                    return Err(unsupported("comparison image type"));
                }
                let pair = TextureSamplerBinding {
                    slot: 0,
                    image_group: *group,
                    image_binding: *binding,
                    sampler_group: *sg,
                    sampler_binding: *sb,
                    uses_sampler: true,
                    dimension: *dimension,
                    depth: *depth,
                    comparison: depth_ref.is_some(),
                };
                let target = texture_target(&pair);
                let slot = self.texture_slot(pair)?;
                let coordinate = self.expression(frame, coordinate)?;
                let components = if *dimension == TextureViewDimension::Cube {
                    3
                } else {
                    2
                };
                if !matches!(coordinate.shape,Shape::Vector(K::Float,n) if n==components)
                    || array_index.is_some() != (*dimension == TextureViewDimension::D2Array)
                {
                    return Err(unsupported("sampling coordinates"));
                }
                self.temp_lanes = self.temp_lanes.div_ceil(4) * 4;
                let coords = self.allocate(Shape::Vector(K::Float, 4), false)?;
                let zero = self.immediate(Literal::F32(0.0))?;
                for i in 0..4 {
                    self.instruction(
                        "MOV",
                        &coords.lanes[i],
                        &[coordinate.lanes.get(i).unwrap_or(&zero.lanes[0])],
                    )?;
                }
                if let Some(layer) = array_index {
                    let layer = self.expression(frame, layer)?;
                    self.instruction(
                        if layer.shape.kind()? == K::Uint {
                            "U2F"
                        } else {
                            "I2F"
                        },
                        &coords.lanes[2],
                        &[&layer.lanes[0]],
                    )?;
                }
                if let Some(reference) = depth_ref {
                    let reference = self.expression(frame, reference)?;
                    let component = if *dimension == TextureViewDimension::D2 {
                        2
                    } else {
                        3
                    };
                    self.instruction("MOV", &coords.lanes[component], &[&reference.lanes[0]])?;
                    if component == 3 && !matches!(level, naga::SampleLevel::Auto) {
                        return Err(unsupported("array/cube comparison LOD"));
                    }
                }
                let opcode = match level {
                    naga::SampleLevel::Auto if self.stage == ShaderStage::Fragment => "TEX",
                    naga::SampleLevel::Zero => "TXL",
                    naga::SampleLevel::Exact(lod) => {
                        let lod = self.expression(frame, lod)?;
                        self.instruction("MOV", &coords.lanes[3], &[&lod.lanes[0]])?;
                        "TXL"
                    }
                    naga::SampleLevel::Bias(lod) if self.stage == ShaderStage::Fragment => {
                        let lod = self.expression(frame, lod)?;
                        self.instruction("MOV", &coords.lanes[3], &[&lod.lanes[0]])?;
                        "TXB"
                    }
                    _ => return Err(unsupported("sampling level/gradients")),
                };
                self.texture_instruction(opcode, slot, target, &coords, shape)?
            }
            E::ImageLoad {
                image,
                coordinate,
                array_index,
                sample: None,
                level,
            } => {
                let image = self.expression(frame, image)?;
                let Shape::Image(group, binding, dimension, depth) = image.shape else {
                    return Err(unsupported("texel load handle"));
                };
                if dimension == TextureViewDimension::Cube
                    || array_index.is_some() != (dimension == TextureViewDimension::D2Array)
                {
                    return Err(unsupported("texel load dimension"));
                }
                let pair = TextureSamplerBinding {
                    slot: 0,
                    image_group: group,
                    image_binding: binding,
                    sampler_group: 0,
                    sampler_binding: 0,
                    uses_sampler: false,
                    dimension,
                    depth,
                    comparison: false,
                };
                let target = texture_target(&pair);
                let slot = self.texture_slot(pair)?;
                let coordinate = self.expression(frame, coordinate)?;
                if !matches!(coordinate.shape, Shape::Vector(K::Sint | K::Uint, 2)) {
                    return Err(unsupported("texel load coordinates"));
                }
                self.temp_lanes = self.temp_lanes.div_ceil(4) * 4;
                let coords = self.allocate(Shape::Vector(K::Sint, 4), false)?;
                let zero = self.immediate(Literal::I32(0))?;
                for i in 0..4 {
                    self.instruction(
                        "MOV",
                        &coords.lanes[i],
                        &[coordinate.lanes.get(i).unwrap_or(&zero.lanes[0])],
                    )?;
                }
                if let Some(layer) = array_index {
                    let layer = self.expression(frame, layer)?;
                    self.instruction("MOV", &coords.lanes[2], &[&layer.lanes[0]])?;
                }
                if let Some(lod) = level {
                    let lod = self.expression(frame, lod)?;
                    self.instruction("MOV", &coords.lanes[3], &[&lod.lanes[0]])?;
                }
                self.texture_instruction("TXF", slot, target, &coords, shape)?
            }
            E::ImageQuery { image, query } => {
                let image = self.expression(frame, image)?;
                let Shape::Image(group, binding, dimension, depth) = image.shape else {
                    return Err(unsupported("image query handle"));
                };
                if matches!(query, naga::ImageQuery::NumLevels) {
                    let metadata = self
                        .image_query_levels
                        .iter()
                        .find(|metadata| metadata.group == group && metadata.binding == binding)
                        .ok_or_else(|| unsupported("missing image mip-level metadata"))?;
                    Value {
                        shape,
                        lanes: vec![Lane {
                            register: format!("CONST[{}]", metadata.first_register),
                            component: 0,
                        }],
                        writable: false,
                        indirect: None,
                    }
                } else {
                    let pair = TextureSamplerBinding {
                        slot: 0,
                        image_group: group,
                        image_binding: binding,
                        sampler_group: 0,
                        sampler_binding: 0,
                        uses_sampler: false,
                        dimension,
                        depth,
                        comparison: false,
                    };
                    let target = texture_target(&pair);
                    let slot = self.texture_slot(pair)?;
                    self.temp_lanes = self.temp_lanes.div_ceil(4) * 4;
                    let coords = self.allocate(Shape::Vector(K::Sint, 4), false)?;
                    let zero = self.immediate(Literal::I32(0))?;
                    self.instruction("MOV", &coords.lanes[0], &[&zero.lanes[0]])?;
                    let selected = match query {
                        naga::ImageQuery::Size { level } => {
                            if let Some(level) = level {
                                let level = self.expression(frame, level)?;
                                self.instruction("MOV", &coords.lanes[0], &[&level.lanes[0]])?;
                            }
                            (0..shape.len()).collect::<Vec<_>>()
                        }
                        naga::ImageQuery::NumLevels => unreachable!(),
                        naga::ImageQuery::NumLayers => vec![2],
                        naga::ImageQuery::NumSamples => {
                            return Err(unsupported("multisample image query"));
                        }
                    };
                    let query_result = self.allocate(Shape::Vector(K::Uint, 4), false)?;
                    let write_mask = match selected.iter().copied().max() {
                        Some(0) => "x",
                        Some(1) => "xy",
                        Some(2) => "xyz",
                        _ => return Err(unsupported("image query components")),
                    };
                    self.instructions.push(format!(
                        "TXQ {}.{write_mask}, {}, SAMP[{slot}], {target}",
                        query_result.lanes[0].register, coords.lanes[0].register
                    ));
                    Value {
                        shape,
                        lanes: selected
                            .into_iter()
                            .map(|index| query_result.lanes[index].clone())
                            .collect(),
                        writable: false,
                        indirect: None,
                    }
                }
            }
            E::Relational { fun, argument } => {
                let argument = self.expression(frame, argument)?;
                let opcode = match fun {
                    naga::RelationalFunction::Any => "OR",
                    naga::RelationalFunction::All => "AND",
                    _ => return Err(unsupported("relational function")),
                };
                let result = self.allocate(shape, false)?;
                let first = argument
                    .lanes
                    .first()
                    .ok_or_else(|| unsupported("empty relational argument"))?;
                self.instruction("MOV", &result.lanes[0], &[first])?;
                for lane in argument.lanes.iter().skip(1) {
                    self.instruction(opcode, &result.lanes[0], &[&result.lanes[0], lane])?;
                }
                result
            }
            E::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3: _,
            } => {
                let a = self.expression(frame, arg)?;
                let b = arg1.map(|h| self.expression(frame, h)).transpose()?;
                let c = arg2.map(|h| self.expression(frame, h)).transpose()?;
                self.math(fun, a, b, c, shape)?
            }
            _ => {
                return Err(unsupported(
                    "expression (texture, derivative, atomic, override and subgroup operations are not supported)",
                ));
            }
        };
        frame.expressions[handle.index()] = Some(value.clone());
        Ok(value)
    }
    fn storage_access(
        &mut self,
        value: &Value,
        constant: Option<u32>,
        index: Option<&Value>,
    ) -> Result<Value> {
        let Shape::StoragePointer {
            inner,
            slot,
            metadata,
            offset,
        } = &value.shape
        else {
            return Err(unsupported("storage pointer"));
        };
        let (inner, stride, fixed) = match inner.as_ref() {
            T::Struct { members, .. } => {
                let member = constant
                    .and_then(|i| members.get(i as usize))
                    .ok_or_else(|| unsupported("storage structure member"))?;
                (self.module.types[member.ty].inner.clone(), 0, member.offset)
            }
            T::Array { base, stride, .. } => (self.module.types[*base].inner.clone(), *stride, 0),
            T::Vector { scalar, .. } => (T::Scalar(*scalar), u32::from(scalar.width), 0),
            T::Matrix { rows, scalar, .. } => (
                T::Vector {
                    size: *rows,
                    scalar: *scalar,
                },
                if *rows == naga::VectorSize::Bi { 8 } else { 16 },
                0,
            ),
            _ => return Err(unsupported("storage composite indexing")),
        };
        let delta = if let Some(index) = constant {
            self.immediate(Literal::U32(
                index
                    .checked_mul(stride)
                    .and_then(|v| v.checked_add(fixed))
                    .ok_or_else(|| unsupported("storage offset overflow"))?,
            ))?
        } else {
            let index = index.ok_or_else(|| unsupported("storage array index"))?;
            if !matches!(index.shape, Shape::Scalar(K::Sint | K::Uint)) {
                return Err(unsupported("non-integer storage array index"));
            }
            let scale = self.immediate(Literal::U32(stride))?;
            let delta = self.allocate(Shape::Scalar(K::Uint), false)?;
            self.instruction("UMUL", &delta.lanes[0], &[&index.lanes[0], &scale.lanes[0]])?;
            delta
        };
        let address = self.allocate(Shape::Scalar(K::Uint), false)?;
        self.instruction("UADD", &address.lanes[0], &[offset, &delta.lanes[0]])?;
        Ok(Value {
            shape: Shape::StoragePointer {
                inner: Box::new(inner),
                slot: *slot,
                metadata: *metadata,
                offset: address.lanes[0].clone(),
            },
            lanes: Vec::new(),
            writable: false,
            indirect: None,
        })
    }

    fn storage_offsets(&self, inner: &T, base: u32, output: &mut Vec<u32>) -> Result<()> {
        match inner {
            T::Scalar(scalar) if scalar.width == 4 => output.push(base),
            T::Vector { size, scalar } if scalar.width == 4 => {
                output.extend((0..*size as u32).map(|i| base + i * 4));
            }
            T::Matrix {
                columns,
                rows,
                scalar,
            } if scalar.width == 4 => {
                let stride = if *rows == naga::VectorSize::Bi { 8 } else { 16 };
                for column in 0..*columns as u32 {
                    output.extend((0..*rows as u32).map(|row| base + column * stride + row * 4));
                }
            }
            T::Struct { members, .. } => {
                for member in members {
                    self.storage_offsets(
                        &self.module.types[member.ty].inner,
                        base + member.offset,
                        output,
                    )?;
                }
            }
            T::Array {
                base: element,
                size: naga::ArraySize::Constant(count),
                stride,
            } if count.get() <= 64 => {
                for i in 0..count.get() {
                    self.storage_offsets(
                        &self.module.types[*element].inner,
                        base + i * stride,
                        output,
                    )?;
                }
            }
            _ => return Err(unsupported("storage load layout")),
        }
        Ok(())
    }

    fn storage_load(
        &mut self,
        inner: &T,
        slot: u32,
        metadata: u32,
        offset: &Lane,
    ) -> Result<Value> {
        let result = self.allocate(self.shape(inner)?, false)?;
        let mut offsets = Vec::new();
        self.storage_offsets(inner, 0, &mut offsets)?;
        let base = Lane {
            register: format!("CONST[{metadata}]"),
            component: 0,
        };
        let length = Lane {
            register: format!("CONST[{metadata}]"),
            component: 1,
        };
        let shift = self.immediate(Literal::U32(2))?;
        let zero = self.immediate(Literal::U32(0))?;
        self.temp_lanes = self.temp_lanes.div_ceil(4) * 4;
        let coords = self.allocate(Shape::Vector(K::Uint, 4), false)?;
        let texel = self.allocate(Shape::Vector(K::Uint, 4), false)?;
        let relative = self.allocate(Shape::Scalar(K::Uint), false)?;
        let valid = self.allocate(Shape::Scalar(K::Bool), false)?;
        for lane in &coords.lanes {
            self.instruction("MOV", lane, &[&zero.lanes[0]])?;
        }
        for (destination, extra) in result.lanes.iter().zip(offsets) {
            let extra = self.immediate(Literal::U32(extra))?;
            self.instruction("UADD", &relative.lanes[0], &[offset, &extra.lanes[0]])?;
            self.instruction("USLT", &valid.lanes[0], &[&relative.lanes[0], &length])?;
            self.instruction("UADD", &coords.lanes[0], &[&base, &relative.lanes[0]])?;
            self.instruction(
                "USHR",
                &coords.lanes[0],
                &[&coords.lanes[0], &shift.lanes[0]],
            )?;
            self.instructions.push(format!(
                "TXF {}, {}, SAMP[{slot}], BUFFER",
                texel.lanes[0].register, coords.lanes[0].register
            ));
            self.instruction(
                "UCMP",
                destination,
                &[&valid.lanes[0], &texel.lanes[0], &zero.lanes[0]],
            )?;
        }
        Ok(result)
    }

    fn texture_slot(&mut self, mut pair: TextureSamplerBinding) -> Result<u32> {
        if let Some(existing) = self.textures.iter().find(|existing| {
            let mut key = (*existing).clone();
            key.slot = 0;
            key == pair
        }) {
            return Ok(existing.slot);
        }
        if self.textures.len() + self.storage_buffers.len() >= 16 {
            return Err(unsupported("more than 16 texture bindings"));
        }
        let slot = (self.textures.len() + self.storage_buffers.len()) as u32;
        pair.slot = slot;
        self.declarations.push(format!("DCL SAMP[{slot}]"));
        self.declarations.push(format!(
            "DCL SVIEW[{slot}], {}, FLOAT",
            texture_target(&pair)
        ));
        self.textures.push(pair);
        Ok(slot)
    }
    fn texture_instruction(
        &mut self,
        opcode: &str,
        slot: u32,
        target: &str,
        coords: &Value,
        shape: Shape,
    ) -> Result<Value> {
        self.temp_lanes = self.temp_lanes.div_ceil(4) * 4;
        let result = self.allocate(Shape::Vector(K::Float, 4), false)?;
        self.instructions.push(format!(
            "{opcode} {}, {}, SAMP[{slot}], {target}",
            result.lanes[0].register, coords.lanes[0].register
        ));
        if matches!(shape, Shape::Scalar(K::Float)) {
            result.element(0)
        } else {
            Ok(result)
        }
    }
    fn unary_instruction(&mut self, opcode: &str, value: &Value, shape: Shape) -> Result<Value> {
        let result = self.allocate(shape, false)?;
        for (dst, src) in result.lanes.iter().zip(&value.lanes) {
            self.instruction(opcode, dst, &[src])?;
        }
        Ok(result)
    }
    fn binary(&mut self, op: B, a: &Value, b: &Value, shape: Shape) -> Result<Value> {
        if op == B::Multiply {
            match (&a.shape, &b.shape) {
                (Shape::Matrix(columns, rows), Shape::Vector(_, n)) if columns == n => {
                    let result = self.allocate(shape, false)?;
                    for row in 0..*rows {
                        let products = (0..*columns)
                            .map(|col| (a.lanes[col * rows + row].clone(), b.lanes[col].clone()))
                            .collect::<Vec<_>>();
                        self.dot(&result.lanes[row], &products)?;
                    }
                    return Ok(result);
                }
                (Shape::Vector(_, n), Shape::Matrix(columns, rows)) if n == rows => {
                    let result = self.allocate(shape, false)?;
                    for col in 0..*columns {
                        let products = (0..*rows)
                            .map(|row| (a.lanes[row].clone(), b.lanes[col * rows + row].clone()))
                            .collect::<Vec<_>>();
                        self.dot(&result.lanes[col], &products)?;
                    }
                    return Ok(result);
                }
                (Shape::Matrix(ac, ar), Shape::Matrix(bc, br)) if ac == br => {
                    let result = self.allocate(shape, false)?;
                    for col in 0..*bc {
                        for row in 0..*ar {
                            let products = (0..*ac)
                                .map(|k| {
                                    (a.lanes[k * ar + row].clone(), b.lanes[col * br + k].clone())
                                })
                                .collect::<Vec<_>>();
                            self.dot(&result.lanes[col * ar + row], &products)?;
                        }
                    }
                    return Ok(result);
                }
                _ => {}
            }
        }
        let kind = a.shape.kind()?;
        let opcode = match (op, kind) {
            (B::Add, K::Float) => "ADD",
            (B::Subtract, K::Float) => "SUB",
            (B::Multiply, K::Float) => "MUL",
            (B::Divide, K::Float) => "DIV",
            (B::Add, K::Sint | K::Uint) => "UADD",
            (B::Multiply, K::Sint | K::Uint) => "UMUL",
            (B::Divide, K::Sint) => "IDIV",
            (B::Divide, K::Uint) => "UDIV",
            (B::Modulo, K::Sint) => "IMOD",
            (B::Modulo, K::Uint) => "UMOD",
            (B::Equal, K::Float) => "FSEQ",
            (B::NotEqual, K::Float) => "FSNE",
            (B::Less, K::Float) => "FSLT",
            (B::GreaterEqual, K::Float) => "FSGE",
            (B::Equal, _) => "USEQ",
            (B::NotEqual, _) => "USNE",
            (B::Less, K::Sint) => "ISLT",
            (B::GreaterEqual, K::Sint) => "ISGE",
            (B::Less, K::Uint) => "USLT",
            (B::GreaterEqual, K::Uint) => "USGE",
            (B::And | B::LogicalAnd, _) => "AND",
            (B::InclusiveOr | B::LogicalOr, _) => "OR",
            (B::ExclusiveOr, _) => "XOR",
            (B::ShiftLeft, _) => "SHL",
            (B::ShiftRight, K::Uint) => "USHR",
            (B::ShiftRight, K::Sint) => "ISHR",
            (B::Greater, _) => return self.binary(B::Less, b, a, shape),
            (B::LessEqual, _) => return self.binary(B::GreaterEqual, b, a, shape),
            (B::Subtract, K::Sint | K::Uint) => {
                let neg = self.unary_instruction("INEG", b, b.shape.clone())?;
                return self.binary(B::Add, a, &neg, shape);
            }
            _ => return Err(unsupported("binary operator/type")),
        };
        let result = self.allocate(shape, false)?;
        if !(a.lanes.len() == 1 || a.lanes.len() == result.lanes.len())
            || !(b.lanes.len() == 1 || b.lanes.len() == result.lanes.len())
        {
            return Err(unsupported("binary operand shape"));
        }
        for (i, dst) in result.lanes.iter().enumerate() {
            self.instruction(
                opcode,
                dst,
                &[&a.lanes[i % a.lanes.len()], &b.lanes[i % b.lanes.len()]],
            )?;
        }
        Ok(result)
    }
    fn dot(&mut self, destination: &Lane, products: &[(Lane, Lane)]) -> Result<()> {
        for (index, (a, b)) in products.iter().enumerate() {
            if index == 0 {
                self.instruction("MUL", destination, &[a, b])?;
            } else {
                let temp = self.allocate(Shape::Scalar(K::Float), false)?;
                self.instruction("MUL", &temp.lanes[0], &[a, b])?;
                self.instruction("ADD", destination, &[destination, &temp.lanes[0]])?;
            }
        }
        Ok(())
    }
    fn math(
        &mut self,
        fun: naga::MathFunction,
        a: Value,
        b: Option<Value>,
        c: Option<Value>,
        shape: Shape,
    ) -> Result<Value> {
        use naga::MathFunction as M;
        if fun == M::Cross {
            let b = b.ok_or_else(|| unsupported("cross operands"))?;
            if a.lanes.len() != 3 || b.lanes.len() != 3 || a.shape.kind()? != K::Float {
                return Err(unsupported("cross type"));
            }
            let result = self.allocate(shape, false)?;
            let temp = self.allocate(Shape::Vector(K::Float, 2), false)?;
            for i in 0..3 {
                let j = (i + 1) % 3;
                let k = (i + 2) % 3;
                self.instruction("MUL", &temp.lanes[0], &[&a.lanes[j], &b.lanes[k]])?;
                self.instruction("MUL", &temp.lanes[1], &[&a.lanes[k], &b.lanes[j]])?;
                self.instruction("SUB", &result.lanes[i], &[&temp.lanes[0], &temp.lanes[1]])?;
            }
            return Ok(result);
        }
        if fun == M::Mix {
            let b = b.ok_or_else(|| unsupported("mix operands"))?;
            let c = c.ok_or_else(|| unsupported("mix operands"))?;
            let difference = self.binary(B::Subtract, &b, &a, shape.clone())?;
            let weighted = self.binary(B::Multiply, &difference, &c, shape.clone())?;
            return self.binary(B::Add, &a, &weighted, shape);
        }
        if fun == M::SmoothStep {
            let upper = b.ok_or_else(|| unsupported("smoothstep operands"))?;
            let value = c.ok_or_else(|| unsupported("smoothstep operands"))?;
            let offset = self.binary(B::Subtract, &value, &a, shape.clone())?;
            let range = self.binary(B::Subtract, &upper, &a, shape.clone())?;
            let ratio = self.binary(B::Divide, &offset, &range, shape.clone())?;
            let zero = self.immediate(Literal::F32(0.0))?;
            let one = self.immediate(Literal::F32(1.0))?;
            let two = self.immediate(Literal::F32(2.0))?;
            let three = self.immediate(Literal::F32(3.0))?;
            let t = self.math(M::Clamp, ratio, Some(zero), Some(one), shape.clone())?;
            let squared = self.binary(B::Multiply, &t, &t, shape.clone())?;
            let doubled = self.binary(B::Multiply, &two, &t, shape.clone())?;
            let curve = self.binary(B::Subtract, &three, &doubled, shape.clone())?;
            return self.binary(B::Multiply, &squared, &curve, shape);
        }
        if fun == M::Reflect {
            let normal = b.ok_or_else(|| unsupported("reflect operands"))?;
            if a.lanes.len() != normal.lanes.len() || a.shape.kind()? != K::Float {
                return Err(unsupported("reflect type"));
            }
            let dot = self.allocate(Shape::Scalar(K::Float), false)?;
            let products = a
                .lanes
                .iter()
                .cloned()
                .zip(normal.lanes.iter().cloned())
                .collect::<Vec<_>>();
            self.dot(&dot.lanes[0], &products)?;
            let doubled = self.binary(B::Add, &dot, &dot, Shape::Scalar(K::Float))?;
            let scaled = self.binary(B::Multiply, &normal, &doubled, shape.clone())?;
            return self.binary(B::Subtract, &a, &scaled, shape);
        }
        if fun == M::Transpose {
            let Shape::Matrix(columns, rows) = &a.shape else {
                return Err(unsupported("transpose type"));
            };
            if !matches!(&shape, Shape::Matrix(result_columns, result_rows)
                if result_columns == rows && result_rows == columns)
            {
                return Err(unsupported("transpose result type"));
            }
            let lanes = &a.lanes;
            return Ok(Value {
                shape,
                lanes: (0..*rows)
                    .flat_map(|column| {
                        (0..*columns).map(move |row| lanes[row * *rows + column].clone())
                    })
                    .collect(),
                writable: false,
                indirect: None,
            });
        }
        if fun == M::Sign && a.shape.kind()? == K::Float {
            let zero = self.immediate(Literal::F32(0.0))?;
            let result = self.allocate(shape, false)?;
            let positive = self.allocate(result.shape.clone(), false)?;
            let negative = self.allocate(result.shape.clone(), false)?;
            for (index, destination) in result.lanes.iter().enumerate() {
                self.instruction(
                    "SGE",
                    &positive.lanes[index],
                    &[&a.lanes[index], &zero.lanes[0]],
                )?;
                self.instruction(
                    "SGE",
                    &negative.lanes[index],
                    &[&zero.lanes[0], &a.lanes[index]],
                )?;
                self.instruction(
                    "SUB",
                    destination,
                    &[&positive.lanes[index], &negative.lanes[index]],
                )?;
            }
            return Ok(result);
        }
        if fun == M::Step {
            let x = b.ok_or_else(|| unsupported("step operands"))?;
            let result = self.allocate(shape, false)?;
            for (i, destination) in result.lanes.iter().enumerate() {
                self.instruction(
                    "SGE",
                    destination,
                    &[&x.lanes[i % x.lanes.len()], &a.lanes[i % a.lanes.len()]],
                )?;
            }
            return Ok(result);
        }
        if matches!(fun, M::Dot | M::Length | M::Normalize) {
            let rhs = b.as_ref().unwrap_or(&a);
            let dot = self.allocate(Shape::Scalar(K::Float), false)?;
            let pairs = a
                .lanes
                .iter()
                .cloned()
                .zip(rhs.lanes.iter().cloned())
                .collect::<Vec<_>>();
            self.dot(&dot.lanes[0], &pairs)?;
            return match fun {
                M::Dot => Ok(dot),
                M::Length => self.unary_instruction("SQRT", &dot, shape),
                M::Normalize => {
                    let inv = self.unary_instruction("RSQ", &dot, Shape::Scalar(K::Float))?;
                    self.binary(B::Multiply, &a, &inv, shape)
                }
                _ => unreachable!(),
            };
        }
        if fun == M::Clamp {
            let lower = b.ok_or_else(|| unsupported("clamp operands"))?;
            let upper = c.ok_or_else(|| unsupported("clamp operands"))?;
            let limited = self.math(M::Max, a, Some(lower), None, shape.clone())?;
            return self.math(M::Min, limited, Some(upper), None, shape);
        }
        if matches!(a.shape.kind()?, K::Sint | K::Uint) {
            let signed = a.shape.kind()? == K::Sint;
            let opcode = match fun {
                M::Min => {
                    if signed {
                        "IMIN"
                    } else {
                        "UMIN"
                    }
                }
                M::Max => {
                    if signed {
                        "IMAX"
                    } else {
                        "UMAX"
                    }
                }
                M::Abs if signed => "IABS",
                _ => return Err(unsupported(&format!("integer math builtin {fun:?}"))),
            };
            let result = self.allocate(shape, false)?;
            for (i, dst) in result.lanes.iter().enumerate() {
                let mut sources = vec![&a.lanes[i % a.lanes.len()]];
                if let Some(b) = &b {
                    sources.push(&b.lanes[i % b.lanes.len()]);
                }
                self.instruction(opcode, dst, &sources)?;
            }
            return Ok(result);
        }
        let opcode = match fun {
            M::Abs => "ABS",
            M::Floor => "FLR",
            M::Ceil => "CEIL",
            M::Round => "ROUND",
            M::Trunc => "TRUNC",
            M::Fract => "FRC",
            M::Sqrt => "SQRT",
            M::InverseSqrt => "RSQ",
            M::Sin => "SIN",
            M::Cos => "COS",
            M::Exp2 => "EX2",
            M::Log2 => "LG2",
            M::Min => "MIN",
            M::Max => "MAX",
            M::Pow => "POW",
            _ => return Err(unsupported(&format!("math function {fun:?}"))),
        };
        if a.shape.kind()? != K::Float {
            return Err(unsupported("integer math builtin"));
        }
        let result = self.allocate(shape, false)?;
        for (i, dst) in result.lanes.iter().enumerate() {
            let mut sources = vec![&a.lanes[i % a.lanes.len()]];
            if let Some(b) = &b {
                sources.push(&b.lanes[i % b.lanes.len()]);
            }
            self.instruction(opcode, dst, &sources)?;
        }
        Ok(result)
    }
}
