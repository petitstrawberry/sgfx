#![cfg(feature = "programmable")]

use sgfx_codegen_virgl::programmable::{CompiledShader, compile_shader};
use sgfx_core::ir::{ShaderModuleDesc, ShaderStage, TextureViewDimension};

const SAMPLE: &str = r#"
@group(0) @binding(0) var image: texture_1d<f32>;
@group(0) @binding(1) var filtering: sampler;
@fragment fn main() -> @location(0) vec4<f32> {
    var layer = 2i;
    return textureSampleLevel(image, filtering, 0.3125, 2.0);
}
"#;
const LOAD: &str = r#"
@group(0) @binding(0) var image: texture_1d<f32>;
@fragment fn main() -> @location(0) vec4<f32> {
    var layer = 2i;
    return textureLoad(image, 7, 1);
}
"#;
const SIZE: &str = r#"
@group(0) @binding(0) var image: texture_1d<f32>;
@fragment fn main() -> @location(0) vec4<f32> {
    let width = textureDimensions(image, 2);
    return vec4<f32>(f32(width), 0.0, 0.0, 1.0);
}
"#;

fn spirv(module: &naga::Module) -> ShaderModuleDesc {
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(module)
    .unwrap();
    let mut options = naga::back::spv::Options::default();
    options
        .flags
        .remove(naga::back::spv::WriterFlags::ADJUST_COORDINATE_SPACE);
    ShaderModuleDesc::spirv(naga::back::spv::write_vec(module, &info, &options, None).unwrap())
        .unwrap()
}

fn formats(source: &str) -> [(&'static str, ShaderModuleDesc); 2] {
    [
        ("wgsl", ShaderModuleDesc::wgsl(source.into()).unwrap()),
        (
            "spirv",
            spirv(&naga::front::wgsl::parse_str(source).unwrap()),
        ),
    ]
}

fn export(name: &str, shader: &CompiledShader) {
    if let Ok(directory) = std::env::var("SGFX_TGSI_EXPORT_DIR") {
        std::fs::write(
            std::path::Path::new(&directory).join(format!("{name}.frag.tgsi")),
            &shader.tgsi,
        )
        .unwrap();
    }
}

fn texture_operands<'a>(shader: &'a CompiledShader, opcode: &str) -> Vec<&'a str> {
    shader
        .tgsi
        .lines()
        .find_map(|line| {
            line.split_once(": ")
                .and_then(|(_, instruction)| instruction.strip_prefix(&format!("{opcode} ")))
        })
        .unwrap_or_else(|| panic!("missing {opcode}:\n{}", shader.tgsi))
        .split(", ")
        .collect()
}

// Assert the externally visible texture instruction's coordinates, without
// depending on the compiler's temporary-register allocation order.
fn assert_immediate_move(shader: &CompiledShader, destination: &str, encoding: &str, value: &str) {
    let declaration = format!("{encoding} {{ {value}, {value}, {value}, {value} }}");
    let immediate = shader
        .tgsi
        .lines()
        .find_map(|line| line.strip_suffix(&format!(" {declaration}")))
        .unwrap_or_else(|| panic!("missing {declaration}:\n{}", shader.tgsi));
    let last_write = shader
        .tgsi
        .lines()
        .filter_map(|line| line.split_once(": ").map(|(_, instruction)| instruction))
        .filter(|instruction| {
            instruction
                .split_once(' ')
                .is_some_and(|(_, operands)| operands.starts_with(&format!("{destination}, ")))
        })
        .last();
    assert_eq!(
        last_write,
        Some(format!("MOV {destination}, {immediate}.xxxx").as_str()),
        "wrong final coordinate assignment:\n{}",
        shader.tgsi
    );
}

#[test]
fn d1_sample_uses_row_center_and_preserves_x_and_explicit_lod() {
    for (format, desc) in formats(SAMPLE) {
        let shader = compile_shader(&desc, ShaderStage::Fragment, "main").unwrap();
        export(&format!("texture-1d-sample-{format}"), &shader);
        assert_eq!(shader.textures[0].dimension, TextureViewDimension::D1);
        assert!(shader.tgsi.contains("DCL SVIEW[0], 2D, FLOAT"));
        let op = texture_operands(&shader, "TXL");
        assert_eq!(op[3], "2D");
        assert_immediate_move(&shader, &format!("{}.x", op[1]), "FLT32", "3.125000000e-1");
        assert_immediate_move(&shader, &format!("{}.y", op[1]), "FLT32", "5.000000000e-1");
        assert_immediate_move(&shader, &format!("{}.w", op[1]), "FLT32", "2.000000000e0");
    }
}

#[test]
fn d1_texel_load_uses_integer_row_zero_and_preserves_x_and_lod() {
    for (format, desc) in formats(LOAD) {
        let shader = compile_shader(&desc, ShaderStage::Fragment, "main").unwrap();
        export(&format!("texture-1d-load-{format}"), &shader);
        assert_eq!(shader.textures[0].dimension, TextureViewDimension::D1);
        assert!(!shader.textures[0].uses_sampler);
        let op = texture_operands(&shader, "TXF");
        assert_eq!(op[3], "2D");
        assert_immediate_move(&shader, &format!("{}.x", op[1]), "INT32", "7");
        assert_immediate_move(&shader, &format!("{}.y", op[1]), "INT32", "0");
        assert_immediate_move(&shader, &format!("{}.w", op[1]), "INT32", "1");
    }
}

#[test]
fn d1_dimensions_returns_only_logical_width_at_requested_mip() {
    for (format, desc) in formats(SIZE) {
        let shader = compile_shader(&desc, ShaderStage::Fragment, "main").unwrap();
        export(&format!("texture-1d-size-{format}"), &shader);
        assert_eq!(shader.textures[0].dimension, TextureViewDimension::D1);
        let op = texture_operands(&shader, "TXQ");
        assert!(op[0].ends_with(".x"), "logical size must be scalar: {op:?}");
        assert_eq!(op[3], "2D");
        assert_immediate_move(&shader, &format!("{}.x", op[1]), "INT32", "2");
    }
}

// WGSL has no texture_1d_array spelling. Preserve the valid D1 coordinate
// expressions, change the Naga image type, and supply a separate integer layer.
fn array_module(source: &str) -> naga::Module {
    let mut module = naga::front::wgsl::parse_str(source).unwrap();
    let image_type = module
        .types
        .iter()
        .find_map(|(handle, ty)| {
            matches!(ty.inner, naga::TypeInner::Image { .. }).then_some(handle)
        })
        .unwrap();
    let mut ty = module.types[image_type].clone();
    let naga::TypeInner::Image {
        ref mut arrayed, ..
    } = ty.inner
    else {
        unreachable!();
    };
    *arrayed = true;
    module.types.replace(image_type, ty);
    let function = &mut module.entry_points[0].function;
    let layer = function
        .expressions
        .iter()
        .find_map(|(handle, expression)| {
            matches!(expression, naga::Expression::Literal(naga::Literal::I32(2))).then_some(handle)
        });
    for (_, expression) in function.expressions.iter_mut() {
        match expression {
            naga::Expression::ImageSample { array_index, .. }
            | naga::Expression::ImageLoad { array_index, .. } => {
                *array_index =
                    Some(layer.expect("fixture must declare layer 2 before image operation"));
            }
            _ => {}
        }
    }
    module
}

#[test]
fn d1_array_sample_and_load_keep_layer_in_physical_z() {
    for (name, source, opcode) in [("sample", SAMPLE, "TXL"), ("load", LOAD, "TXF")] {
        let desc = spirv(&array_module(source));
        let shader = compile_shader(&desc, ShaderStage::Fragment, "main").unwrap();
        export(&format!("texture-1d-array-{name}-spirv"), &shader);
        assert_eq!(shader.textures[0].dimension, TextureViewDimension::D1Array);
        assert!(shader.tgsi.contains("DCL SVIEW[0], 2D_ARRAY, FLOAT"));
        let op = texture_operands(&shader, opcode);
        assert_eq!(op[3], "2D_ARRAY");
        if opcode == "TXF" {
            assert_immediate_move(&shader, &format!("{}.x", op[1]), "INT32", "7");
            assert_immediate_move(&shader, &format!("{}.y", op[1]), "INT32", "0");
            assert_immediate_move(&shader, &format!("{}.z", op[1]), "INT32", "2");
        } else {
            assert_immediate_move(&shader, &format!("{}.x", op[1]), "FLT32", "3.125000000e-1");
            assert_immediate_move(&shader, &format!("{}.y", op[1]), "FLT32", "5.000000000e-1");
            assert!(
                shader
                    .tgsi
                    .lines()
                    .any(|line| line.contains(&format!("I2F {}.z,", op[1])))
            );
        }
    }
}

#[test]
fn naga24_d1_array_dimensions_roundtrip_is_rejected_explicitly() {
    // Naga Size is scalar for D1, while SPIR-V's array query returns
    // (width, layers). Naga 24 cannot round-trip this query; accepting sampling
    // and loads must not be mistaken for complete D1Array query support.
    let desc = spirv(&array_module(SIZE));
    let sgfx_core::ir::ShaderSource::SpirV(words) = desc.source() else {
        unreachable!();
    };
    let module = naga::front::spv::Frontend::new(
        words.iter().copied(),
        &naga::front::spv::Options {
            adjust_coordinate_space: false,
            ..Default::default()
        },
    )
    .parse()
    .unwrap();
    let validation = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap_err();
    assert!(
        matches!(
            validation.as_inner(),
            naga::valid::ValidationError::Function {
                source: naga::valid::FunctionError::Expression {
                    source: naga::valid::ExpressionError::Type(
                        naga::proc::ResolveError::InvalidVector(_)
                    ),
                    ..
                },
                ..
            }
        ),
        "unexpected validation failure: {validation:?}"
    );
    eprintln!("known Naga 24 D1Array query validation: {validation:?}");
    let error = compile_shader(&desc, ShaderStage::Fragment, "main").unwrap_err();
    assert!(
        error.0.starts_with("shader validation:"),
        "unexpected rejection: {error}"
    );
    eprintln!("known Naga 24 D1Array dimensions limitation: {error}");
}
