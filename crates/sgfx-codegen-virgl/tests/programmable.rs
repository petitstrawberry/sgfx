#![cfg(feature = "programmable")]

use sgfx_codegen_virgl::programmable::{compile_shader, validate_shader_module};
use sgfx_core::ir::{ShaderModuleDesc, ShaderStage};

const CUBE: &str = r#"
struct Transform { mvp: mat4x4<f32>, };
@group(0) @binding(0) var<uniform> transform: Transform;
struct Output { @builtin(position) position: vec4<f32>, @location(0) color: vec3<f32> };
@vertex fn vs_main(@location(0) position: vec3<f32>, @location(1) color: vec3<f32>) -> Output {
    var output: Output;
    output.position = transform.mvp * vec4<f32>(position, 1.0);
    output.color = color;
    return output;
}
@fragment fn fs_main(input: Output) -> @location(0) vec4<f32> { return vec4<f32>(input.color, 1.0); }
"#;
fn wgsl(source: &str) -> ShaderModuleDesc { ShaderModuleDesc::wgsl(source.into()).unwrap() }
fn spirv(source: &str) -> ShaderModuleDesc {
    let module = naga::front::wgsl::parse_str(source).unwrap();
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty()).validate(&module).unwrap();
    let mut options = naga::back::spv::Options::default();
    options.flags.remove(naga::back::spv::WriterFlags::ADJUST_COORDINATE_SPACE);
    ShaderModuleDesc::spirv(naga::back::spv::write_vec(&module, &info, &options, None).unwrap()).unwrap()
}

#[test]
fn cube_wgsl_and_spirv_compile_with_uniform_and_typed_interfaces() {
    for shader in [wgsl(CUBE),spirv(CUBE)] {
        validate_shader_module(&shader).unwrap();
        let vertex=compile_shader(&shader,ShaderStage::Vertex,"vs_main").unwrap();
        let fragment=compile_shader(&shader,ShaderStage::Fragment,"fs_main").unwrap();
        assert_eq!(vertex.input_locations,vec![0,1]);
        assert_eq!(vertex.output_locations,vec![0]);
        assert_eq!(fragment.input_locations,vec![0]);
        assert_eq!(vertex.uniform_buffers.len(),1);
        let binding=&vertex.uniform_buffers[0];
        assert_eq!((binding.group,binding.binding,binding.slot,binding.size),(0,0,1,64));
        assert!(fragment.uniform_buffers.is_empty());
        assert!(vertex.tgsi.contains("DCL CONST[1][0..3]"));
        assert!(vertex.tgsi.contains("CONST[1][3]"));
        assert!(vertex.tgsi.contains("MUL"));
        assert!(vertex.tgsi.contains("SUB OUT[0].z"));
        assert_eq!(vertex.outputs[0].components,fragment.inputs[0].components);
    }
}

#[test]
fn uniform_byte_layout_and_stage_slot_mapping_are_reflected() {
    let source=r#"
struct U { a: f32, b: vec3<f32>, c: mat4x4<f32> };
@group(2) @binding(5) var<uniform> right: U;
@group(0) @binding(7) var<uniform> left: U;
@vertex fn vertex(@location(0) p: vec4<f32>) -> @builtin(position) vec4<f32> {
    return right.c * p + vec4<f32>(left.b, left.a);
}"#;
    let shader=compile_shader(&wgsl(source),ShaderStage::Vertex,"vertex").unwrap();
    assert_eq!(shader.uniform_buffers.iter().map(|b|(b.group,b.binding,b.slot,b.size)).collect::<Vec<_>>(),vec![(0,7,1,96),(2,5,2,96)]);
    assert!(shader.tgsi.contains("CONST[1][1].xxxx")); // vec3 starts at byte 16.
    assert!(shader.tgsi.contains("CONST[2][2].xxxx")); // matrix starts at byte 32.
}

#[test]
fn actual_shader_operations_and_constants_change_the_program() {
    let source=r#"@fragment fn fragment(@location(0) color: vec3<f32>) -> @location(0) vec4<f32> {
      let adjusted = color.bgr * 0.25 + vec3<f32>(0.125);
      return vec4<f32>(adjusted, 1.0);
    }"#;
    let a=compile_shader(&wgsl(source),ShaderStage::Fragment,"fragment").unwrap();
    let b=compile_shader(&wgsl(&source.replace("0.25","0.75")),ShaderStage::Fragment,"fragment").unwrap();
    assert_ne!(a.tgsi,b.tgsi);
    assert!(a.tgsi.contains("IN[0].zzzz"));
    assert!(a.tgsi.contains("MUL"));
    assert!(a.tgsi.contains("ADD"));
    assert!(a.tgsi.contains("2.500000000e-1"));
}

#[test]
fn helper_calls_and_value_snapshots_lower_without_reusing_mutated_storage() {
    let source=r#"
fn adjust(v: vec4<f32>) -> vec4<f32> { return v * vec4<f32>(0.5, 1.0, 1.0, 1.0); }
@vertex fn vertex(@location(0) p: vec4<f32>) -> @builtin(position) vec4<f32> {
    var a = p;
    let before = a;
    a.x = 0.25;
    return adjust(before + a);
}"#;
    for shader in [wgsl(source),spirv(source)] { let result=compile_shader(&shader,ShaderStage::Vertex,"vertex").unwrap();assert!(result.tgsi.contains("MUL"));assert!(result.tgsi.contains("ADD")); }
}

#[test]
fn unsupported_shaders_fail_instead_of_emitting_a_compatibility_shader() {
    let cases=[
      ("@compute @workgroup_size(1) fn main() {}",ShaderStage::Compute,"compute"),
      ("@group(0) @binding(0) var<storage,read> x:array<vec4<f32>>; @vertex fn main()->@builtin(position) vec4<f32>{return x[0];}",ShaderStage::Vertex,"resource address space"),
      ("@vertex fn main(@builtin(vertex_index) i:u32)->@builtin(position) vec4<f32>{let a=array<vec4<f32>,2>(vec4<f32>(0.0),vec4<f32>(1.0));return a[i];}",ShaderStage::Vertex,"dynamic indexing"),
      ("@vertex fn main(@location(0) p:vec4<f32>)->@builtin(position) vec4<f32>{var v=p; for(var i=0;i<3;i++){v.x+=1.0;}return v;}",ShaderStage::Vertex,"statement"),
    ];
    for(source,stage,reason)in cases {let error=compile_shader(&wgsl(source),stage,"main").unwrap_err();assert!(error.0.contains(reason),"{error}");}
    assert!(validate_shader_module(&wgsl("this is not WGSL")).is_err());
    assert!(compile_shader(&wgsl(CUBE),ShaderStage::Vertex,"missing").is_err());
}

/// Emit the exact tested compiler output for external Mesa parser/renderer QA.
#[test]
#[ignore = "writes TGSI fixtures for the external virglrenderer acceptance harness"]
fn export_tgsi_acceptance_fixtures() {
    let directory=std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/virgl-compiler-shaders");
    std::fs::create_dir_all(&directory).unwrap();
    for (format,source) in [("wgsl",wgsl(CUBE)),("spirv",spirv(CUBE))] {
        for (stage,entry,suffix) in [(ShaderStage::Vertex,"vs_main","vert"),(ShaderStage::Fragment,"fs_main","frag")] {
            let shader=compile_shader(&source,stage,entry).unwrap();
            std::fs::write(directory.join(format!("cube-{format}.{suffix}.tgsi")),shader.tgsi).unwrap();
        }
    }
}
