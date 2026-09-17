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
const TEXTURED: &str = r#"
@group(1) @binding(4) var image: texture_2d<f32>;
@group(2) @binding(7) var filtering: sampler;
fn sample(uv: vec2<f32>) -> vec4<f32> { return textureSample(image, filtering, uv); }
@fragment fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return sample(uv) * vec4<f32>(0.5, 1.0, 1.0, 1.0);
}
"#;
fn wgsl(source: &str) -> ShaderModuleDesc {
    ShaderModuleDesc::wgsl(source.into()).unwrap()
}

#[test]
fn readonly_storage_arrays_preserve_struct_matrix_layout_and_dynamic_indices() {
    let source = r#"
struct Object { translation: vec3<f32>, joint: u32, transform: mat3x3<f32> };
@group(1) @binding(1) var<storage, read> objects: array<Object>;
@group(1) @binding(2) var<storage, read> joints: array<mat4x4<f32>>;
@vertex fn main(@location(0) position: vec3<f32>, @builtin(instance_index) instance: u32) -> @builtin(position) vec4<f32> {
    let object = objects[instance];
    return joints[object.joint] * vec4<f32>(object.transform * position + object.translation, 1.0);
}"#;
    for module in [wgsl(source), spirv(source)] {
        let shader = compile_shader(&module, ShaderStage::Vertex, "main").unwrap();
        assert_eq!(shader.storage_buffers.len(), 2);
        assert_eq!(
            (
                shader.storage_buffers[0].group,
                shader.storage_buffers[0].binding
            ),
            (1, 1)
        );
        assert_eq!(shader.storage_buffers[1].first_register, 1);
        assert!(shader.tgsi.contains("DCL SVIEW[0], BUFFER, UINT"));
        assert!(shader.tgsi.contains("DCL SVIEW[1], BUFFER, UINT"));
        assert!(shader.tgsi.contains("UINT32 { 64, 64, 64, 64 }"));
        assert!(shader.tgsi.contains("UINT32 { 48, 48, 48, 48 }"));
        assert!(shader.tgsi.contains("USLT"));
        assert!(shader.tgsi.contains("TXF"));
        if let Ok(directory) = std::env::var("SGFX_TGSI_EXPORT_DIR") {
            std::fs::write(
                std::path::Path::new(&directory).join("storage.vert.tgsi"),
                &shader.tgsi,
            )
            .unwrap();
        }
    }
}

#[test]
fn quaternion_color_and_signed_index_math_compile_for_wgsl_and_spirv() {
    let source = r#"
@vertex fn main(@location(0) p: vec3<f32>, @location(1) q: vec4<f32>, @location(2) joint: i32) -> @builtin(position) vec4<f32> {
    let rotated = p + 2.0 * cross(cross(p, q.xyz) + q.w * p, q.xyz);
    let t = step(0.5, p.x);
    return vec4<f32>(mix(p, rotated, t), f32(max(joint, 0) + 1));
}"#;
    for module in [wgsl(source), spirv(source)] {
        let shader = compile_shader(&module, ShaderStage::Vertex, "main").unwrap();
        assert!(shader.tgsi.contains("IMAX"));
        assert!(shader.tgsi.contains("SGE"));
        assert!(shader.tgsi.contains("MUL"));
        assert!(shader.tgsi.contains("SUB"));
    }
}
fn spirv(source: &str) -> ShaderModuleDesc {
    let module = naga::front::wgsl::parse_str(source).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::PUSH_CONSTANT,
    )
    .validate(&module)
    .unwrap();
    let mut options = naga::back::spv::Options::default();
    options
        .flags
        .remove(naga::back::spv::WriterFlags::ADJUST_COORDINATE_SPACE);
    ShaderModuleDesc::spirv(naga::back::spv::write_vec(&module, &info, &options, None).unwrap())
        .unwrap()
}
fn vulkan_normalized_spirv(source: &str) -> ShaderModuleDesc {
    let module = naga::front::wgsl::parse_str(source).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    let words =
        naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
            .unwrap();
    let adjusted = naga::front::spv::Frontend::new(
        words.into_iter(),
        &naga::front::spv::Options {
            adjust_coordinate_space: true,
            strict_capabilities: true,
            block_ctx_dump_prefix: None,
        },
    )
    .parse()
    .unwrap();
    let adjusted_info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&adjusted)
    .unwrap();
    let mut options = naga::back::spv::Options {
        lang_version: (1, 0),
        ..Default::default()
    };
    options
        .flags
        .remove(naga::back::spv::WriterFlags::ADJUST_COORDINATE_SPACE);
    ShaderModuleDesc::spirv(
        naga::back::spv::write_vec(&adjusted, &adjusted_info, &options, None).unwrap(),
    )
    .unwrap()
}

#[test]
fn cube_wgsl_and_spirv_compile_with_uniform_and_typed_interfaces() {
    for shader in [wgsl(CUBE), spirv(CUBE)] {
        validate_shader_module(&shader).unwrap();
        let vertex = compile_shader(&shader, ShaderStage::Vertex, "vs_main").unwrap();
        let fragment = compile_shader(&shader, ShaderStage::Fragment, "fs_main").unwrap();
        assert_eq!(vertex.input_locations, vec![0, 1]);
        assert_eq!(vertex.output_locations, vec![0]);
        assert_eq!(fragment.input_locations, vec![0]);
        assert_eq!(vertex.uniform_buffers.len(), 1);
        let binding = &vertex.uniform_buffers[0];
        assert_eq!(
            (
                binding.group,
                binding.binding,
                binding.first_register,
                binding.size
            ),
            (0, 0, 0, 64)
        );
        assert!(fragment.uniform_buffers.is_empty());
        assert!(vertex.tgsi.contains("DCL CONST[0..3]"));
        assert!(vertex.tgsi.contains("CONST[3]"));
        assert!(vertex.tgsi.contains("MUL"));
        assert!(vertex.tgsi.contains("SUB OUT[0].z"));
        assert_eq!(vertex.outputs[0].components, fragment.inputs[0].components);
    }
}

#[test]
fn vulkan_normalized_cube_spirv_compiles() {
    let shader = vulkan_normalized_spirv(CUBE);
    compile_shader(&shader, ShaderStage::Vertex, "vs_main").unwrap();
    compile_shader(&shader, ShaderStage::Fragment, "fs_main").unwrap();
}

#[test]
fn separate_texture_and_sampler_pairs_lower_for_wgsl_and_spirv() {
    for module in [wgsl(TEXTURED), spirv(TEXTURED)] {
        let shader = compile_shader(&module, ShaderStage::Fragment, "fs_main").unwrap();
        assert_eq!(shader.textures.len(), 1);
        let pair = &shader.textures[0];
        assert_eq!(
            (
                pair.slot,
                pair.image_group,
                pair.image_binding,
                pair.sampler_group,
                pair.sampler_binding
            ),
            (0, 1, 4, 2, 7)
        );
        assert!(shader.tgsi.contains("DCL SAMP[0]"));
        assert!(shader.tgsi.contains("DCL SVIEW[0], 2D, FLOAT"));
        assert!(shader.tgsi.contains("TEX "));
        assert!(shader.tgsi.contains("MUL"));
    }
}

#[test]
fn explicit_lod_sampling_and_multiple_pairs_have_distinct_slots() {
    let source = r#"
@group(0) @binding(0) var a: texture_2d<f32>;
@group(0) @binding(1) var b: texture_2d<f32>;
@group(0) @binding(2) var s: sampler;
@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSampleLevel(a, s, uv, 0.0) + textureSampleLevel(b, s, uv, 0.0);
}"#;
    for module in [wgsl(source), spirv(source)] {
        let shader = compile_shader(&module, ShaderStage::Fragment, "main").unwrap();
        assert_eq!(
            shader
                .textures
                .iter()
                .map(|b| (b.slot, b.image_binding, b.sampler_binding))
                .collect::<Vec<_>>(),
            vec![(0, 0, 2), (1, 1, 2)]
        );
        assert_eq!(shader.tgsi.matches("TXL ").count(), 2);
    }
}

#[test]
fn uniform_byte_layout_and_stage_register_mapping_are_reflected() {
    let source = r#"
struct U { a: f32, b: vec3<f32>, c: mat4x4<f32> };
@group(2) @binding(5) var<uniform> right: U;
@group(0) @binding(7) var<uniform> left: U;
@vertex fn vertex(@location(0) p: vec4<f32>) -> @builtin(position) vec4<f32> {
    return right.c * p + vec4<f32>(left.b, left.a);
}"#;
    let shader = compile_shader(&wgsl(source), ShaderStage::Vertex, "vertex").unwrap();
    assert_eq!(
        shader
            .uniform_buffers
            .iter()
            .map(|b| (b.group, b.binding, b.first_register, b.size))
            .collect::<Vec<_>>(),
        vec![(0, 7, 0, 96), (2, 5, 6, 96)]
    );
    assert!(shader.tgsi.contains("CONST[1].xxxx")); // vec3 starts at byte 16.
    assert!(shader.tgsi.contains("CONST[8].xxxx")); // second buffer + matrix byte offset.
}

#[test]
fn push_constants_follow_uniforms_and_preserve_each_stage_byte_layout() {
    let source = r#"
struct Uniforms { matrix: mat4x4<f32> };
struct Push { bias: vec3<f32>, gain: f32 };
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
var<push_constant> push: Push;
@vertex fn vertex(@location(0) p: vec4<f32>) -> @builtin(position) vec4<f32> {
    return uniforms.matrix * p + vec4<f32>(push.bias, push.gain);
}
@fragment fn fragment() -> @location(0) vec4<f32> {
    return vec4<f32>(push.bias * push.gain, 1.0);
}"#;
    for module in [wgsl(source), spirv(source)] {
        validate_shader_module(&module).unwrap();
        let vertex = compile_shader(&module, ShaderStage::Vertex, "vertex").unwrap();
        let fragment = compile_shader(&module, ShaderStage::Fragment, "fragment").unwrap();
        assert_eq!(vertex.uniform_buffers[0].first_register, 0);
        assert_eq!(vertex.uniform_buffers[0].size, 64);
        let push = vertex.push_constants.unwrap();
        assert_eq!((push.first_register, push.size), (4, 16));
        assert!(vertex.tgsi.contains("DCL CONST[0..4]"));
        assert!(vertex.tgsi.contains("CONST[4].wwww"));
        let push = fragment.push_constants.unwrap();
        assert_eq!((push.first_register, push.size), (0, 16));
        assert!(fragment.uniform_buffers.is_empty());
        assert!(fragment.tgsi.contains("CONST[0].wwww"));
    }
}

#[test]
fn push_constant_limits_are_checked_before_emitting_a_shader() {
    let source = r#"
struct Push { a: mat4x4<f32>, b: mat4x4<f32>, c: vec4<f32> };
var<push_constant> push: Push;
@vertex fn main(@location(0) p: vec4<f32>) -> @builtin(position) vec4<f32> {
    return push.a * p + push.b * p + push.c;
}"#;
    for module in [wgsl(source), spirv(source)] {
        let error = compile_shader(&module, ShaderStage::Vertex, "main").unwrap_err();
        assert!(
            error.0.contains("push-constant block exceeds 128 bytes"),
            "{error}"
        );
    }
}

#[test]
fn actual_shader_operations_and_constants_change_the_program() {
    let source = r#"@fragment fn fragment(@location(0) color: vec3<f32>) -> @location(0) vec4<f32> {
      let adjusted = color.bgr * 0.25 + vec3<f32>(0.125);
      return vec4<f32>(adjusted, 1.0);
    }"#;
    let a = compile_shader(&wgsl(source), ShaderStage::Fragment, "fragment").unwrap();
    let b = compile_shader(
        &wgsl(&source.replace("0.25", "0.75")),
        ShaderStage::Fragment,
        "fragment",
    )
    .unwrap();
    assert_ne!(a.tgsi, b.tgsi);
    assert!(a.tgsi.contains("IN[0].zzzz"));
    assert!(a.tgsi.contains("MUL"));
    assert!(a.tgsi.contains("ADD"));
    assert!(a.tgsi.contains("2.500000000e-1"));
}

#[test]
fn helper_calls_and_value_snapshots_lower_without_reusing_mutated_storage() {
    let source = r#"
fn adjust(v: vec4<f32>) -> vec4<f32> { return v * vec4<f32>(0.5, 1.0, 1.0, 1.0); }
@vertex fn vertex(@location(0) p: vec4<f32>) -> @builtin(position) vec4<f32> {
    var a = p;
    let before = a;
    a.x = 0.25;
    return adjust(before + a);
}"#;
    for shader in [wgsl(source), spirv(source)] {
        let result = compile_shader(&shader, ShaderStage::Vertex, "vertex").unwrap();
        assert!(result.tgsi.contains("MUL"));
        assert!(result.tgsi.contains("ADD"));
    }
}

#[test]
fn bounded_dynamic_reads_compile_for_fullscreen_vertex_arrays_and_components() {
    let source = r#"
@vertex fn main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var positions = array<vec4<f32>, 3>(
        vec4<f32>(-1.0, -1.0, 0.0, 1.0),
        vec4<f32>(3.0, -1.0, 0.0, 1.0),
        vec4<f32>(-1.0, 3.0, 0.0, 1.0));
    let selected = &positions[index % 3u];
    positions[0].z = 0.5;
    let current = *selected;
    let component = current[index % 2u];
    return vec4<f32>(current.xy, component * 0.0 + current.z, current.w);
}"#;
    for module in [wgsl(source), spirv(source)] {
        let shader = compile_shader(&module, ShaderStage::Vertex, "main").unwrap();
        assert!(shader.input_locations.is_empty());
        assert!(shader.tgsi.contains("VERTEXID"));
        assert!(shader.tgsi.contains("USEQ"));
        assert!(shader.tgsi.contains("UCMP"));
    }
}

#[test]
fn unsupported_shaders_fail_instead_of_emitting_a_compatibility_shader() {
    let cases = [
        (
            "@compute @workgroup_size(1) fn main() {}",
            ShaderStage::Compute,
            "compute",
        ),
        (
            "@group(0) @binding(0) var<storage,read_write> x:array<vec4<f32>>; @vertex fn main()->@builtin(position) vec4<f32>{x[0]=vec4<f32>(1.0); return x[0];}",
            ShaderStage::Vertex,
            "resource address space",
        ),
        (
            "@vertex fn main(@builtin(vertex_index) i:u32)->@builtin(position) vec4<f32>{var a=array<vec4<f32>,2>(vec4<f32>(0.0),vec4<f32>(1.0));a[i]=vec4<f32>(2.0);return a[0];}",
            ShaderStage::Vertex,
            "store target",
        ),
        (
            "@vertex fn main(@location(0) p:vec4<f32>)->@builtin(position) vec4<f32>{var v=p; for(var i=0;i<3;i++){v.x+=1.0;}return v;}",
            ShaderStage::Vertex,
            "statement",
        ),
    ];
    for (source, stage, reason) in cases {
        let error = compile_shader(&wgsl(source), stage, "main").unwrap_err();
        assert!(error.0.contains(reason), "{error}");
    }
    assert!(validate_shader_module(&wgsl("this is not WGSL")).is_err());
    assert!(compile_shader(&wgsl(CUBE), ShaderStage::Vertex, "missing").is_err());
}

/// Emit the exact tested compiler output for external Mesa parser/renderer QA.
#[test]
#[ignore = "writes TGSI fixtures for the external virglrenderer acceptance harness"]
fn export_tgsi_acceptance_fixtures() {
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/virgl-compiler-shaders");
    std::fs::create_dir_all(&directory).unwrap();
    for (format, source) in [("wgsl", wgsl(LAYERED)), ("spirv", spirv(LAYERED))] {
        let shader = compile_shader(&source, ShaderStage::Fragment, "main").unwrap();
        std::fs::write(
            directory.join(format!("layered-{format}.frag.tgsi")),
            shader.tgsi,
        )
        .unwrap();
    }
    for (format, source) in [("wgsl", wgsl(TEXTURED)), ("spirv", spirv(TEXTURED))] {
        let shader = compile_shader(&source, ShaderStage::Fragment, "fs_main").unwrap();
        std::fs::write(
            directory.join(format!("textured-{format}.frag.tgsi")),
            shader.tgsi,
        )
        .unwrap();
    }
    for (format, source) in [("wgsl", wgsl(CUBE)), ("spirv", spirv(CUBE))] {
        for (stage, entry, suffix) in [
            (ShaderStage::Vertex, "vs_main", "vert"),
            (ShaderStage::Fragment, "fs_main", "frag"),
        ] {
            let shader = compile_shader(&source, stage, entry).unwrap();
            std::fs::write(
                directory.join(format!("cube-{format}.{suffix}.tgsi")),
                shader.tgsi,
            )
            .unwrap();
        }
    }
}

const LAYERED: &str = r#"
@group(0) @binding(0) var cube: texture_cube<f32>;
@group(0) @binding(1) var layers: texture_2d_array<f32>;
@group(0) @binding(2) var depth: texture_depth_2d;
@group(0) @binding(3) var linear: sampler;
@group(0) @binding(4) var compare: sampler_comparison;
@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let a = textureSample(cube, linear, vec3<f32>(uv, 1.0));
    let b = textureSample(layers, linear, uv, 2);
    let c = textureSampleCompare(depth, compare, uv, 0.5);
    let d = textureLoad(layers, vec2<i32>(uv * 16.0), 1, 0);
    let e = textureLoad(depth, vec2<i32>(uv * 16.0), 0);
    return a + b + d + vec4<f32>(c + e);
}
"#;

#[test]
fn cube_array_depth_comparison_and_texel_loads_preserve_binding_types() {
    use sgfx_core::ir::TextureViewDimension;
    for module in [wgsl(LAYERED), spirv(LAYERED)] {
        let shader = compile_shader(&module, ShaderStage::Fragment, "main").unwrap();
        assert_eq!(shader.textures.len(), 5);
        assert!(
            shader
                .textures
                .iter()
                .any(|b| b.dimension == TextureViewDimension::Cube && b.uses_sampler)
        );
        assert!(
            shader
                .textures
                .iter()
                .any(|b| b.dimension == TextureViewDimension::D2Array && !b.uses_sampler)
        );
        assert!(
            shader
                .textures
                .iter()
                .any(|b| b.depth && b.comparison && b.sampler_binding == 4)
        );
        assert!(shader.textures.iter().any(|b| b.depth && !b.uses_sampler));
        assert_eq!(shader.tgsi.matches("TXF ").count(), 2);
        for target in ["CUBE", "2D_ARRAY", "SHADOW2D"] {
            assert!(shader.tgsi.contains(target));
        }
    }
}
