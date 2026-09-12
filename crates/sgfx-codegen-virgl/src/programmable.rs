//! Bounded, semantic Naga IR to TGSI graphics shader lowering.
//!
//! This compiler deliberately rejects operations it cannot lower. It uses no
//! shader-name matching or fixed-shader substitution. The optional feature uses
//! Naga's standard-library frontends; the compatibility encoder remains no_std.

use alloc::{format, string::String, vec, vec::Vec};
use core::fmt::{self, Write};
use naga::{AddressSpace, BinaryOperator as B, Binding, BuiltIn, Expression as E,
    Handle, Literal, ScalarKind as K, Statement as S, TypeInner as T};
use sgfx_core::ir::{ShaderModuleDesc, ShaderSource, ShaderStage, VertexFormat};

type Result<T> = core::result::Result<T, ShaderCompileError>;

/// A parse, validation, unsupported-feature, or bounded-resource failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaderCompileError(pub String);
impl fmt::Display for ShaderCompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(&self.0) }
}
fn unsupported(message: &str) -> ShaderCompileError { ShaderCompileError(format!("unsupported TGSI shader: {message}")) }

/// Uniform-buffer register allocation for a single compiled stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UniformBufferBinding {
    /// SGFX bind-group index.
    pub group: u32,
    /// Binding within that group.
    pub binding: u32,
    /// VirGL constant-buffer slot; zero is reserved for inline constants.
    pub slot: u32,
    /// Required byte span, including the source language's padding.
    pub size: u32,
}

/// Numeric type of a stage interface location.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IoScalar { Float, Sint, Uint }
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
    /// Used uniform buffers, with independent slot allocation per stage.
    pub uniform_buffers: Vec<UniformBufferBinding>,
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
/// Uniform buffers use their declared byte layout and one-based VirGL slots.
/// Compute, textures, storage buffers, loops, dynamic indexing and early returns
/// are currently rejected. See the tests for executable examples of the subset.
pub fn compile_shader(desc: &ShaderModuleDesc, stage: ShaderStage, entry_point: &str) -> Result<CompiledShader> {
    let naga_stage = match stage {
        ShaderStage::Vertex => naga::ShaderStage::Vertex,
        ShaderStage::Fragment => naga::ShaderStage::Fragment,
        ShaderStage::Compute => return Err(unsupported("compute stage")),
    };
    let module = parse_shader_module(desc)?;
    let info = validate_module(&module)?;
    let entry_index = module.entry_points.iter().position(|entry| entry.stage == naga_stage && entry.name == entry_point)
        .ok_or_else(|| ShaderCompileError(format!("entry point {entry_point:?} is absent for {stage:?}")))?;
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
        ShaderSource::SpirV(words) => naga::front::spv::Frontend::new(words.iter().copied(), &naga::front::spv::Options {
            adjust_coordinate_space: false, strict_capabilities: true, block_ctx_dump_prefix: None,
        }).parse().map_err(|error| ShaderCompileError(format!("SPIR-V: {error}"))),
    }
}
fn validate_module(module: &naga::Module) -> Result<naga::valid::ModuleInfo> {
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty())
        .validate(module).map_err(|error| ShaderCompileError(format!("shader validation: {error}")))
}

#[derive(Clone, Debug)]
struct Lane { register: String, component: usize }
impl Lane {
    fn src(&self) -> String { format!("{}.{}", self.register, ["xxxx", "yyyy", "zzzz", "wwww"][self.component]) }
    fn dst(&self) -> String { format!("{}.{}", self.register, ["x", "y", "z", "w"][self.component]) }
}
#[derive(Clone, Debug)]
enum Shape { Scalar(K), Vector(K, usize), Matrix(usize, usize), Aggregate(Vec<Shape>) }
impl Shape {
    fn len(&self) -> usize { match self { Self::Scalar(_) => 1, Self::Vector(_, n) => *n, Self::Matrix(c,r) => c*r, Self::Aggregate(items) => items.iter().map(Self::len).sum() } }
    fn kind(&self) -> Result<K> { match self { Self::Scalar(k) | Self::Vector(k,_) => Ok(*k), Self::Matrix(..) => Ok(K::Float), _ => Err(unsupported("arithmetic on aggregate")) } }
    fn element(&self, index: usize) -> Result<(usize, Shape)> {
        Ok(match self {
            Self::Vector(k,n) if index < *n => (index, Self::Scalar(*k)),
            Self::Matrix(c,r) if index < *c => (index*r, Self::Vector(K::Float,*r)),
            Self::Aggregate(items) if index < items.len() => (items[..index].iter().map(Self::len).sum(), items[index].clone()),
            _ => return Err(unsupported("out-of-range composite index")),
        })
    }
}
#[derive(Clone, Debug)]
struct Value { shape: Shape, lanes: Vec<Lane>, writable: bool }
impl Value {
    fn element(&self, index: usize) -> Result<Self> {
        let (offset, shape) = self.shape.element(index)?;
        Ok(Self { lanes: self.lanes[offset..offset+shape.len()].to_vec(), shape, writable: self.writable })
    }
}
struct Frame<'a> {
    function: &'a naga::Function,
    info: &'a naga::valid::FunctionInfo,
    arguments: Vec<Value>,
    locals: Vec<Value>,
    expressions: Vec<Option<Value>>,
    returned: Option<Value>,
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
    input_locations: Vec<u32>,
    output_locations: Vec<u32>,
    call_depth: usize,
    inputs: Vec<IoLocation>,
    outputs: Vec<IoLocation>,
    vertex_inputs: Vec<(u32,VertexFormat)>,
}
impl<'a> Compiler<'a> {
    fn new(module: &'a naga::Module, info: &'a naga::valid::ModuleInfo, stage: ShaderStage) -> Self {
        Self { module, info, stage, declarations: Vec::new(), immediates: Vec::new(), instructions: Vec::new(), temp_lanes: 0,
            globals: vec![None;module.global_variables.len()], uniforms:Vec::new(), input_locations:Vec::new(),output_locations:Vec::new(), call_depth:0,inputs:Vec::new(),outputs:Vec::new(),vertex_inputs:Vec::new() }
    }
    fn shape(&self, inner: &T) -> Result<Shape> {
        let scalar = |s: naga::Scalar| if s.width == 4 || s.kind == K::Bool { Ok(s.kind) } else { Err(unsupported("non-32-bit scalar")) };
        Ok(match inner {
            T::Scalar(s) => Shape::Scalar(scalar(*s)?),
            T::Vector {size,scalar:s} => Shape::Vector(scalar(*s)?, *size as usize),
            T::Matrix {columns,rows,scalar:s} if scalar(*s)? == K::Float => Shape::Matrix(*columns as usize,*rows as usize),
            T::Struct {members,..} => Shape::Aggregate(members.iter().map(|m| self.shape(&self.module.types[m.ty].inner)).collect::<Result<_>>()?),
            T::Array {base,size:naga::ArraySize::Constant(n),..} if n.get() <= 64 => Shape::Aggregate(vec![self.shape(&self.module.types[*base].inner)?;n.get() as usize]),
            T::Pointer {base,..} => self.shape(&self.module.types[*base].inner)?,
            T::ValuePointer {size,scalar:s,..} => match size { Some(n) => Shape::Vector(scalar(*s)?, *n as usize), None => Shape::Scalar(scalar(*s)?) },
            _ => return Err(unsupported("type (only 32-bit scalars, vectors, matrices, bounded arrays and structs are supported)")),
        })
    }
    fn allocate(&mut self, shape: Shape, writable: bool) -> Result<Value> {
        if self.temp_lanes + shape.len() > 4096 { return Err(unsupported("more than 1024 temporary registers")); }
        let lanes = (0..shape.len()).map(|_| { let n=self.temp_lanes;self.temp_lanes+=1;Lane {register:format!("TEMP[{}]",n/4),component:n%4} }).collect();
        Ok(Value { shape, lanes, writable })
    }
    fn instruction(&mut self, opcode: &str, destination: &Lane, sources: &[&Lane]) -> Result<()> {
        if self.instructions.len() >= 16384 { return Err(unsupported("more than 16384 instructions")); }
        let mut text=format!("{opcode} {}",destination.dst());
        for source in sources { write!(&mut text, ", {}", source.src()).unwrap(); }
        self.instructions.push(text); Ok(())
    }
    fn immediate(&mut self, literal: Literal) -> Result<Value> {
        let (kind, encoding, text) = match literal {
            Literal::F32(v) if v.is_finite() => (K::Float,"FLT32",format!("{v:.9e}")),
            Literal::I32(v) => (K::Sint,"INT32",format!("{v}")),
            Literal::U32(v) => (K::Uint,"UINT32",format!("{v}")),
            Literal::Bool(v) => (K::Bool,"UINT32",format!("{}",if v {u32::MAX} else {0})),
            _ => return Err(unsupported("literal width or non-finite float")),
        };
        let declaration=format!("{encoding} {{ {text}, {text}, {text}, {text} }}");
        let index=match self.immediates.iter().position(|v|v==&declaration) {Some(i)=>i,None=>{let i=self.immediates.len();self.immediates.push(declaration);i}};
        Ok(Value {shape:Shape::Scalar(kind),lanes:vec![Lane {register:format!("IMM[{index}]"),component:0}],writable:false})
    }
    fn snapshot(&mut self, value: &Value) -> Result<Value> {
        let result=self.allocate(value.shape.clone(),false)?;
        for (dst,src) in result.lanes.iter().zip(&value.lanes) { self.instruction("MOV",dst,&[src])?; } Ok(result)
    }
    fn store(&mut self, pointer: &Value, value: &Value) -> Result<()> {
        if !pointer.writable || pointer.lanes.len()!=value.lanes.len() {return Err(unsupported("store target"));}
        for (dst,src) in pointer.lanes.iter().zip(&value.lanes) {self.instruction("MOV",dst,&[src])?;} Ok(())
    }
    fn zero(&mut self, shape: Shape) -> Result<Value> {
        let zero=self.immediate(Literal::U32(0))?;
        Ok(Value {lanes:vec![zero.lanes[0].clone();shape.len()],shape,writable:false})
    }
    fn global_expression(&mut self, handle: Handle<E>) -> Result<Value> {
        match self.module.global_expressions[handle].clone() {
            E::Literal(v)=>self.immediate(v),
            E::Constant(c)=>self.global_expression(self.module.constants[c].init),
            E::ZeroValue(ty)=>self.zero(self.shape(&self.module.types[ty].inner)?),
            E::Compose {ty,components}=>{let mut lanes=Vec::new();for h in components {lanes.extend(self.global_expression(h)?.lanes);} Ok(Value{shape:self.shape(&self.module.types[ty].inner)?,lanes,writable:false})},
            E::Splat {size,value}=>{let v=self.global_expression(value)?;Ok(Value{shape:Shape::Vector(v.shape.kind()?,size as usize),lanes:vec![v.lanes[0].clone();size as usize],writable:false})},
            _=>Err(unsupported("constant expression")),
        }
    }
    fn uniform_value(&self, ty: Handle<naga::Type>, slot:u32, offset:u32) -> Result<Value> {
        let inner=&self.module.types[ty].inner;
        let shape=self.shape(inner)?;
        let mut lanes=Vec::new();
        match inner {
            T::Scalar(_) | T::Vector {..} => for n in 0..shape.len() {let address=offset+n as u32*4;lanes.push(Lane{register:format!("CONST[{slot}][{}]",address/16),component:(address%16/4) as usize});},
            T::Matrix{columns,rows,..}=>{let stride=if *rows==naga::VectorSize::Bi {8}else{16};for c in 0..*columns as u32 {for r in 0..*rows as u32 {let address=offset+c*stride+r*4;lanes.push(Lane{register:format!("CONST[{slot}][{}]",address/16),component:(address%16/4) as usize});}}},
            T::Struct{members,..}=>for member in members {lanes.extend(self.uniform_value(member.ty,slot,offset+member.offset)?.lanes);},
            T::Array{base,size:naga::ArraySize::Constant(n),stride}=>for i in 0..n.get() {lanes.extend(self.uniform_value(*base,slot,offset+i*stride)?.lanes);},
            _=>return Err(unsupported("uniform layout")),
        }
        Ok(Value{shape,lanes,writable:false})
    }
    fn io(&mut self, ty:Handle<naga::Type>, binding:Option<&Binding>, input:bool) -> Result<Value> {
        let shape=self.shape(&self.module.types[ty].inner)?;
        if let T::Struct{members,..}=&self.module.types[ty].inner {
            let mut lanes=Vec::new();for member in members {lanes.extend(self.io(member.ty,member.binding.as_ref(),input)?.lanes);}return Ok(Value{shape,lanes,writable:!input});
        }
        if !matches!(shape,Shape::Scalar(_) | Shape::Vector(..)) {return Err(unsupported("aggregate stage location"));}
        let binding=binding.ok_or_else(||unsupported("unbound stage IO"))?;
        let register=match binding {
            Binding::Location{location,interpolation,sampling,second_blend_source}=>{
                if *location>=16 || *second_blend_source {return Err(unsupported("stage location limit or dual-source blending"));}
                let scalar=match shape.kind()? { K::Float=>IoScalar::Float,K::Sint=>IoScalar::Sint,K::Uint=>IoScalar::Uint,_=>return Err(unsupported("boolean stage IO")) };
                let io=IoLocation{location:*location,components:shape.len() as u8,scalar,interpolation:*interpolation};
                if input {self.input_locations.push(*location);self.inputs.push(io);}else{self.output_locations.push(*location);self.outputs.push(io);}
                if input && self.stage==ShaderStage::Vertex {
                    let format=match (scalar,shape.len()){(IoScalar::Float,2)=>VertexFormat::Float32x2,(IoScalar::Float,3)=>VertexFormat::Float32x3,(IoScalar::Float,4)=>VertexFormat::Float32x4,_=>return Err(unsupported("vertex input format"))};
                    self.vertex_inputs.push((*location,format));
                }
                let index=if !input && self.stage==ShaderStage::Vertex {*location+1}else{*location};
                let reg=format!("{}[{index}]",if input {"IN"}else{"OUT"});
                let suffix=if self.stage==ShaderStage::Vertex && input {String::new()}
                    else if self.stage==ShaderStage::Fragment && !input {format!(", COLOR[{location}]")}
                    else {let mut suffix=format!(", GENERIC[{location}]");if input {
                        suffix.push_str(match interpolation.unwrap_or(naga::Interpolation::Perspective) {naga::Interpolation::Perspective=>", PERSPECTIVE",naga::Interpolation::Linear=>", LINEAR",naga::Interpolation::Flat=>", CONSTANT"});
                        if !matches!(sampling,None|Some(naga::Sampling::Center)) {return Err(unsupported("centroid/sample interpolation"));}
                    }suffix};
                self.declarations.push(format!("DCL {reg}{suffix}"));reg
            },
            Binding::BuiltIn(BuiltIn::Position{..}) if !input && self.stage==ShaderStage::Vertex=>{self.declarations.push("DCL OUT[0], POSITION".into());"OUT[0]".into()},
            Binding::BuiltIn(BuiltIn::Position{..}) if input && self.stage==ShaderStage::Fragment=>{self.declarations.push("DCL IN[16], POSITION, LINEAR".into());"IN[16]".into()},
            Binding::BuiltIn(BuiltIn::VertexIndex) if input && self.stage==ShaderStage::Vertex=>{self.declarations.push("DCL SV[0], VERTEXID".into());"SV[0]".into()},
            Binding::BuiltIn(BuiltIn::InstanceIndex) if input && self.stage==ShaderStage::Vertex=>{self.declarations.push("DCL SV[1], INSTANCEID".into());"SV[1]".into()},
            _=>return Err(unsupported("stage builtin")),
        };
        Ok(Value{lanes:(0..shape.len()).map(|component|Lane{register:register.clone(),component}).collect(),shape,writable:!input})
    }
    fn compile(mut self, index:usize) -> Result<CompiledShader> {
        let entry=&self.module.entry_points[index];
        let entry_info=self.info.get_entry_point(index);
        let mut uniform_handles=Vec::new();
        for (handle,global) in self.module.global_variables.iter() {
            if entry_info[handle].is_empty() {continue;}
            match global.space {
                AddressSpace::Uniform=>{let binding=global.binding.as_ref().ok_or_else(||unsupported("uniform without resource binding"))?;uniform_handles.push((binding.group,binding.binding,handle));},
                AddressSpace::Private=>{},
                _=>return Err(unsupported("resource address space (only uniform and private are supported)")),
            }
        }
        uniform_handles.sort_by_key(|v|(v.0,v.1));
        if uniform_handles.len()>15 {return Err(unsupported("more than 15 uniform buffers in a stage"));}
        let mut layouter=naga::proc::Layouter::default();layouter.update(self.module.to_ctx()).map_err(|e|ShaderCompileError(format!("uniform layout: {e}")))?;
        for (group,binding,handle) in uniform_handles {
            let ty=self.module.global_variables[handle].ty;let size=layouter[ty].size;
            if size==0 || size>16384 {return Err(unsupported("uniform buffer size"));}
            let slot=self.uniforms.len() as u32+1;
            self.uniforms.push(UniformBufferBinding{group,binding,slot,size});
            self.declarations.push(format!("DCL CONST[{slot}][0..{}]",size.div_ceil(16)-1));
            self.globals[handle.index()]=Some(self.uniform_value(ty,slot,0)?);
        }
        for (handle,global) in self.module.global_variables.iter() {
            if global.space!=AddressSpace::Private || entry_info[handle].is_empty(){continue;}
            let target=self.allocate(self.shape(&self.module.types[global.ty].inner)?,true)?;
            let init=match global.init {Some(h)=>self.global_expression(h)?,None=>self.zero(target.shape.clone())?};
            self.store(&target,&init)?;self.globals[handle.index()]=Some(target);
        }
        let mut arguments=Vec::new();for arg in &entry.function.arguments {arguments.push(self.io(arg.ty,arg.binding.as_ref(),true)?);}
        let output=entry.function.result.as_ref().map(|r|self.io(r.ty,r.binding.as_ref(),false)).transpose()?;
        let returned=self.function(&entry.function,entry_info,arguments)?;
        if let Some(output)=output {self.store(&output,&returned.ok_or_else(||unsupported("entry point without result"))?)?;}
        if self.stage==ShaderStage::Vertex {
            // Gallium's default clipping is [-w,w], while SGFX uses [0,w].
            let z=Lane{register:"OUT[0]".into(),component:2};let w=Lane{register:"OUT[0]".into(),component:3};
            let two=self.immediate(Literal::F32(2.0))?;let temp=self.allocate(Shape::Scalar(K::Float),false)?;
            self.instruction("MUL",&temp.lanes[0],&[&z,&two.lanes[0]])?;self.instruction("SUB",&z,&[&temp.lanes[0],&w])?;
        }
        let mut tgsi=String::from(if self.stage==ShaderStage::Vertex {"VERT\n"}else{"FRAG\nPROPERTY FS_COORD_ORIGIN UPPER_LEFT\nPROPERTY FS_COORD_PIXEL_CENTER HALF_INTEGER\n"});
        for line in &self.declarations {writeln!(&mut tgsi,"{line}").unwrap();}
        if self.temp_lanes>0 {writeln!(&mut tgsi,"DCL TEMP[0..{}]",self.temp_lanes.div_ceil(4)-1).unwrap();}
        for (index,line) in self.immediates.iter().enumerate(){writeln!(&mut tgsi,"IMM[{index}] {line}").unwrap();}
        for (index,line) in self.instructions.iter().enumerate(){writeln!(&mut tgsi,"{index}: {line}").unwrap();}
        writeln!(&mut tgsi,"{}: END",self.instructions.len()).unwrap();
        Ok(CompiledShader{stage:self.stage,tgsi,uniform_buffers:self.uniforms,input_locations:self.input_locations,output_locations:self.output_locations,vertex_inputs:self.vertex_inputs,inputs:self.inputs,outputs:self.outputs})
    }
    fn function(&mut self, function:&'a naga::Function, info:&'a naga::valid::FunctionInfo, arguments:Vec<Value>) -> Result<Option<Value>> {
        self.call_depth+=1;if self.call_depth>32{return Err(unsupported("call nesting greater than 32"));}
        let mut frame=Frame{function,info,arguments,locals:Vec::new(),expressions:vec![None;function.expressions.len()],returned:None};
        for (_,local) in function.local_variables.iter(){frame.locals.push(self.allocate(self.shape(&self.module.types[local.ty].inner)?,true)?);}
        for (handle,local) in function.local_variables.iter(){let destination=frame.locals[handle.index()].clone();let init=match local.init{Some(h)=>self.expression(&mut frame,h)?,None=>self.zero(destination.shape.clone())?};self.store(&destination,&init)?;}
        self.block(&mut frame,&function.body,0)?;
        self.call_depth-=1;Ok(frame.returned)
    }
    fn block(&mut self, frame:&mut Frame<'a>, block:&naga::Block, conditional_depth:usize) -> Result<()> {
        for (index,statement) in block.iter().enumerate(){match statement {
            S::Emit(range)=>for handle in range.clone(){self.expression(frame,handle)?;},
            S::Block(block)=>self.block(frame,block,conditional_depth)?,
            S::Store{pointer,value}=>{let pointer=self.expression(frame,*pointer)?;let value=self.expression(frame,*value)?;self.store(&pointer,&value)?;},
            S::Return{value}=>{if conditional_depth!=0 || index+1!=block.len(){return Err(unsupported("early/conditional return"));}frame.returned=value.map(|h|self.expression(frame,h)).transpose()?;},
            S::Call{function,arguments,result}=>{let mut args=Vec::new();for h in arguments {args.push(self.expression(frame,*h)?);}let returned=self.function(&self.module.functions[*function],&self.info[*function],args)?;if let Some(h)=result{frame.expressions[h.index()]=Some(returned.ok_or_else(||unsupported("missing call result"))?);}},
            S::If{condition,accept,reject}=>{let condition=self.expression(frame,*condition)?;self.instructions.push(format!("UIF {}",condition.lanes[0].src()));self.block(frame,accept,conditional_depth+1)?;if !reject.is_empty(){self.instructions.push("ELSE".into());self.block(frame,reject,conditional_depth+1)?;}self.instructions.push("ENDIF".into());},
            S::Kill if self.stage==ShaderStage::Fragment=>self.instructions.push("KILL".into()),
            _=>return Err(unsupported("statement (loops, switch, atomics, barriers and image stores are not supported)")),
        }}Ok(())
    }
    fn expression(&mut self, frame:&mut Frame<'a>, handle:Handle<E>) -> Result<Value> {
        if let Some(v)=&frame.expressions[handle.index()]{return Ok(v.clone());}
        let shape=self.shape(frame.info[handle].ty.inner_with(&self.module.types))?;
        let value=match frame.function.expressions[handle].clone(){
            E::Literal(v)=>self.immediate(v)?,
            E::Constant(c)=>self.global_expression(self.module.constants[c].init)?,
            E::ZeroValue(_)=>self.zero(shape)?,
            E::FunctionArgument(i)=>frame.arguments[i as usize].clone(),
            E::GlobalVariable(h)=>self.globals[h.index()].clone().ok_or_else(||unsupported("unused/unsupported global"))?,
            E::LocalVariable(h)=>frame.locals[h.index()].clone(),
            E::Load{pointer}=>{let v=self.expression(frame,pointer)?;self.snapshot(&v)?},
            E::AccessIndex{base,index}=>self.expression(frame,base)?.element(index as usize)?,
            E::Access{base,index}=>{let index=match frame.function.expressions[index]{E::Literal(Literal::U32(n))=>n,E::Literal(Literal::I32(n)) if n>=0=>n as u32,_=>return Err(unsupported("dynamic indexing"))};self.expression(frame,base)?.element(index as usize)?},
            E::Compose{components,..}=>{let mut lanes=Vec::new();for h in components {lanes.extend(self.expression(frame,h)?.lanes);}if lanes.len()!=shape.len(){return Err(unsupported("composite component count"));}Value{shape,lanes,writable:false}},
            E::Splat{size,value}=>{let v=self.expression(frame,value)?;Value{shape,lanes:vec![v.lanes[0].clone();size as usize],writable:false}},
            E::Swizzle{size,vector,pattern}=>{let v=self.expression(frame,vector)?;Value{shape,lanes:pattern[..size as usize].iter().map(|p|v.lanes[*p as usize].clone()).collect(),writable:false}},
            E::Binary{op,left,right}=>{let a=self.expression(frame,left)?;let b=self.expression(frame,right)?;self.binary(op,&a,&b,shape)?},
            E::Unary{op,expr}=>{let v=self.expression(frame,expr)?;let result=self.allocate(shape,false)?;match op {
                naga::UnaryOperator::Negate if v.shape.kind()?==K::Float=>{let zero=self.immediate(Literal::F32(0.0))?;for(dst,src)in result.lanes.iter().zip(&v.lanes){self.instruction("SUB",dst,&[&zero.lanes[0],src])?;}},
                naga::UnaryOperator::Negate=>for(dst,src)in result.lanes.iter().zip(&v.lanes){self.instruction("INEG",dst,&[src])?;},
                naga::UnaryOperator::LogicalNot | naga::UnaryOperator::BitwiseNot=>for(dst,src)in result.lanes.iter().zip(&v.lanes){self.instruction("NOT",dst,&[src])?;},
            }result},
            E::As{expr,kind,convert}=>{let v=self.expression(frame,expr)?;if convert.is_none()||v.shape.kind()?==kind {Value{shape,lanes:v.lanes,writable:false}}else{if convert!=Some(4){return Err(unsupported("conversion width"));}let opcode=match(v.shape.kind()?,kind){(K::Float,K::Sint)=>"F2I",(K::Float,K::Uint)=>"F2U",(K::Sint,K::Float)=>"I2F",(K::Uint,K::Float)=>"U2F",(K::Sint,K::Uint)|(K::Uint,K::Sint)=>"MOV",_=>return Err(unsupported("scalar conversion"))};self.unary_instruction(opcode,&v,shape)?}},
            E::Select{condition,accept,reject}=>{let c=self.expression(frame,condition)?;let a=self.expression(frame,accept)?;let b=self.expression(frame,reject)?;let result=self.allocate(shape,false)?;for(i,dst)in result.lanes.iter().enumerate(){self.instruction("UCMP",dst,&[&c.lanes[i%c.lanes.len()],&a.lanes[i],&b.lanes[i]])?;}result},
            E::Math{fun,arg,arg1,arg2,arg3:_}=>{let a=self.expression(frame,arg)?;let b=arg1.map(|h|self.expression(frame,h)).transpose()?;let c=arg2.map(|h|self.expression(frame,h)).transpose()?;self.math(fun,a,b,c,shape)?},
            _=>return Err(unsupported("expression (texture, derivative, atomic, override and subgroup operations are not supported)")),
        };
        frame.expressions[handle.index()]=Some(value.clone());Ok(value)
    }
    fn unary_instruction(&mut self,opcode:&str,value:&Value,shape:Shape)->Result<Value>{let result=self.allocate(shape,false)?;for(dst,src)in result.lanes.iter().zip(&value.lanes){self.instruction(opcode,dst,&[src])?;}Ok(result)}
    fn binary(&mut self,op:B,a:&Value,b:&Value,shape:Shape)->Result<Value>{
        if op==B::Multiply {
            match (&a.shape,&b.shape){
                (Shape::Matrix(columns,rows),Shape::Vector(_,n)) if columns==n=>{let result=self.allocate(shape,false)?;for row in 0..*rows{let products=(0..*columns).map(|col|(a.lanes[col*rows+row].clone(),b.lanes[col].clone())).collect::<Vec<_>>();self.dot(&result.lanes[row],&products)?;}return Ok(result);},
                (Shape::Vector(_,n),Shape::Matrix(columns,rows))if n==rows=>{let result=self.allocate(shape,false)?;for col in 0..*columns{let products=(0..*rows).map(|row|(a.lanes[row].clone(),b.lanes[col*rows+row].clone())).collect::<Vec<_>>();self.dot(&result.lanes[col],&products)?;}return Ok(result);},
                (Shape::Matrix(ac,ar),Shape::Matrix(bc,br))if ac==br=>{let result=self.allocate(shape,false)?;for col in 0..*bc{for row in 0..*ar{let products=(0..*ac).map(|k|(a.lanes[k*ar+row].clone(),b.lanes[col*br+k].clone())).collect::<Vec<_>>();self.dot(&result.lanes[col*ar+row],&products)?;}}return Ok(result);},
                _=>{},
            }
        }
        let kind=a.shape.kind()?;
        let opcode=match (op,kind){
            (B::Add,K::Float)=>"ADD",(B::Subtract,K::Float)=>"SUB",(B::Multiply,K::Float)=>"MUL",(B::Divide,K::Float)=>"DIV",
            (B::Add,K::Sint|K::Uint)=>"UADD",(B::Multiply,K::Sint|K::Uint)=>"UMUL",(B::Divide,K::Sint)=>"IDIV",(B::Divide,K::Uint)=>"UDIV",
            (B::Modulo,K::Sint)=>"IMOD",(B::Modulo,K::Uint)=>"UMOD",
            (B::Equal,K::Float)=>"FSEQ",(B::NotEqual,K::Float)=>"FSNE",(B::Less,K::Float)=>"FSLT",(B::GreaterEqual,K::Float)=>"FSGE",
            (B::Equal,_)=>"USEQ",(B::NotEqual,_)=>"USNE",(B::Less,K::Sint)=>"ISLT",(B::GreaterEqual,K::Sint)=>"ISGE",(B::Less,K::Uint)=>"USLT",(B::GreaterEqual,K::Uint)=>"USGE",
            (B::And|B::LogicalAnd,_)=>"AND",(B::InclusiveOr|B::LogicalOr,_)=>"OR",(B::ExclusiveOr,_)=>"XOR",(B::ShiftLeft,_)=>"SHL",(B::ShiftRight,K::Uint)=>"USHR",(B::ShiftRight,K::Sint)=>"ISHR",
            (B::Greater,_)=>return self.binary(B::Less,b,a,shape), (B::LessEqual,_)=>return self.binary(B::GreaterEqual,b,a,shape),
            (B::Subtract,K::Sint|K::Uint)=>{let neg=self.unary_instruction("INEG",b,b.shape.clone())?;return self.binary(B::Add,a,&neg,shape);},
            _=>return Err(unsupported("binary operator/type")),
        };
        let result=self.allocate(shape,false)?;
        if !(a.lanes.len()==1||a.lanes.len()==result.lanes.len())||!(b.lanes.len()==1||b.lanes.len()==result.lanes.len()){return Err(unsupported("binary operand shape"));}
        for(i,dst)in result.lanes.iter().enumerate(){self.instruction(opcode,dst,&[&a.lanes[i%a.lanes.len()],&b.lanes[i%b.lanes.len()]])?;}Ok(result)
    }
    fn dot(&mut self,destination:&Lane,products:&[(Lane,Lane)])->Result<()>{
        for(index,(a,b))in products.iter().enumerate(){if index==0{self.instruction("MUL",destination,&[a,b])?;}else{let temp=self.allocate(Shape::Scalar(K::Float),false)?;self.instruction("MUL",&temp.lanes[0],&[a,b])?;self.instruction("ADD",destination,&[destination,&temp.lanes[0]])?;}}Ok(())
    }
    fn math(&mut self,fun:naga::MathFunction,a:Value,b:Option<Value>,c:Option<Value>,shape:Shape)->Result<Value>{
        use naga::MathFunction as M;
        if matches!(fun,M::Dot|M::Length|M::Normalize){let rhs=b.as_ref().unwrap_or(&a);let dot=self.allocate(Shape::Scalar(K::Float),false)?;let pairs=a.lanes.iter().cloned().zip(rhs.lanes.iter().cloned()).collect::<Vec<_>>();self.dot(&dot.lanes[0],&pairs)?;return match fun{M::Dot=>Ok(dot),M::Length=>self.unary_instruction("SQRT",&dot,shape),M::Normalize=>{let inv=self.unary_instruction("RSQ",&dot,Shape::Scalar(K::Float))?;self.binary(B::Multiply,&a,&inv,shape)},_=>unreachable!()};}
        if fun==M::Clamp{let lower=b.ok_or_else(||unsupported("clamp operands"))?;let upper=c.ok_or_else(||unsupported("clamp operands"))?;let limited=self.math(M::Max,a,Some(lower),None,shape.clone())?;return self.math(M::Min,limited,Some(upper),None,shape);}
        let opcode=match fun{M::Abs=>"ABS",M::Floor=>"FLR",M::Ceil=>"CEIL",M::Round=>"ROUND",M::Trunc=>"TRUNC",M::Fract=>"FRC",M::Sqrt=>"SQRT",M::InverseSqrt=>"RSQ",M::Sin=>"SIN",M::Cos=>"COS",M::Exp2=>"EX2",M::Log2=>"LG2",M::Min=>"MIN",M::Max=>"MAX",M::Pow=>"POW",_=>return Err(unsupported("math function"))};
        if a.shape.kind()?!=K::Float{return Err(unsupported("integer math builtin"));}
        let result=self.allocate(shape,false)?;for(i,dst)in result.lanes.iter().enumerate(){let mut sources=vec![&a.lanes[i%a.lanes.len()]];if let Some(b)=&b{sources.push(&b.lanes[i%b.lanes.len()]);}self.instruction(opcode,dst,&sources)?;}Ok(result)
    }
}
