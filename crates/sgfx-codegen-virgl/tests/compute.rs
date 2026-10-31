#![cfg(feature = "programmable")]
use sgfx_codegen_virgl::programmable::{CompiledShader, compile_shader};
use sgfx_core::ir::{ShaderModuleDesc, ShaderStage};
use std::collections::HashMap;

fn modules(source: &str) -> [ShaderModuleDesc; 2] {
    let module = naga::front::wgsl::parse_str(source).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::PUSH_CONSTANT,
    )
    .validate(&module)
    .unwrap();
    let words = naga::back::spv::write_vec(&module, &info, &Default::default(), None).unwrap();
    [
        ShaderModuleDesc::wgsl(source.into()).unwrap(),
        ShaderModuleDesc::spirv(words).unwrap(),
    ]
}

/// Execute the emitted TGSI scalar subset, independently of the Naga lowering.
/// SSBO LOAD/STORE address and mask behavior follows the Gallium instruction
/// contract; every out-of-range physical access panics instead of masking bugs.
fn execute(
    shader: &CompiledShader,
    constants: &[[u32; 4]],
    systems: [[u32; 4]; 3],
    buffers: &mut [&mut [u32]],
) {
    let mut registers = HashMap::<String, [u32; 4]>::new();
    for (i, value) in constants.iter().enumerate() {
        registers.insert(format!("CONST[{i}]"), *value);
    }
    for (i, value) in systems.iter().enumerate() {
        registers.insert(format!("SV[{i}]"), *value);
    }
    fn source(registers: &HashMap<String, [u32; 4]>, operand: &str) -> [u32; 4] {
        let (name, swizzle) = operand.split_once('.').unwrap_or((operand, "xyzw"));
        let value = registers
            .get(name)
            .unwrap_or_else(|| panic!("undefined {name}"));
        let components: Vec<_> = swizzle.chars().map(|c| "xyzw".find(c).unwrap()).collect();
        std::array::from_fn(|i| value[components[i % components.len()]])
    }
    let mut enabled = vec![true];
    for line in shader.tgsi.lines() {
        if let Some(imm) = line.strip_prefix("IMM[") {
            let (index, body) = imm.split_once(']').unwrap();
            let (kind, values) = body.trim().split_once(" {").unwrap();
            let bits: Vec<_> = values
                .trim_end_matches('}')
                .split(',')
                .map(|v| match kind {
                    "UINT32" => v.trim().parse::<u32>().unwrap(),
                    "INT32" => v.trim().parse::<i32>().unwrap() as u32,
                    "FLT32" => v.trim().parse::<f32>().unwrap().to_bits(),
                    _ => panic!("{kind}"),
                })
                .collect();
            registers.insert(format!("IMM[{index}]"), bits.try_into().unwrap());
            continue;
        }
        let Some((_, instruction)) = line.split_once(": ") else {
            continue;
        };
        let (opcode, operands) = instruction.split_once(' ').unwrap_or((instruction, ""));
        if opcode == "UIF" {
            let parent = *enabled.last().unwrap();
            enabled.push(parent && source(&registers, operands)[0] != 0);
            continue;
        }
        if opcode == "ELSE" {
            let i = enabled.len() - 1;
            enabled[i] = enabled[i - 1] && !enabled[i];
            continue;
        }
        if opcode == "ENDIF" {
            assert!(enabled.len() > 1);
            enabled.pop();
            continue;
        }
        if !*enabled.last().unwrap() {
            continue;
        }
        if matches!(opcode, "END" | "BARRIER" | "MEMBAR") {
            continue;
        }
        let operands: Vec<_> = operands.split(", ").collect();
        let (destination, mask) = operands[0].split_once('.').unwrap_or((operands[0], "xyzw"));
        if opcode == "STORE" {
            let index: usize = destination
                .strip_prefix("BUFFER[")
                .unwrap()
                .trim_end_matches(']')
                .parse()
                .unwrap();
            let address = source(&registers, operands[1])[0];
            assert_eq!(address % 4, 0);
            let values = source(&registers, operands[2]);
            for component in mask.chars().map(|c| "xyzw".find(c).unwrap()) {
                buffers[index][address as usize / 4 + component] = values[component];
            }
            continue;
        }
        let sources: Vec<_> = operands[1..]
            .iter()
            .map(|operand| {
                if operand.starts_with("BUFFER[") {
                    [0; 4]
                } else {
                    source(&registers, operand)
                }
            })
            .collect();
        let output = registers.entry(destination.into()).or_insert([0; 4]);
        for component in mask.chars().map(|c| "xyzw".find(c).unwrap()) {
            let s: Vec<_> = sources.iter().map(|v| v[component]).collect();
            output[component] = match opcode {
                "MOV" => s[0],
                "UADD" => s[0].wrapping_add(s[1]),
                "UMUL" => s[0].wrapping_mul(s[1]),
                "UDIV" => s[0] / s[1],
                "UMOD" => s[0] % s[1],
                "USLT" => {
                    if s[0] < s[1] {
                        u32::MAX
                    } else {
                        0
                    }
                }
                "USGE" => {
                    if s[0] >= s[1] {
                        u32::MAX
                    } else {
                        0
                    }
                }
                "USEQ" => {
                    if s[0] == s[1] {
                        u32::MAX
                    } else {
                        0
                    }
                }
                "AND" => s[0] & s[1],
                "UCMP" => {
                    if s[0] != 0 {
                        s[1]
                    } else {
                        s[2]
                    }
                }
                "LOAD" => {
                    let index: usize = operands[1]
                        .strip_prefix("BUFFER[")
                        .unwrap()
                        .trim_end_matches(']')
                        .parse()
                        .unwrap();
                    assert_eq!(sources[1][0] % 4, 0);
                    buffers[index][sources[1][0] as usize / 4 + component]
                }
                _ => panic!("unhandled TGSI: {instruction}"),
            };
        }
    }
    assert_eq!(enabled, [true]);
}

const GRID: &str = r#"
@group(0) @binding(0) var<storage,read_write> output:array<u32>;
@compute @workgroup_size(2,2,2)
fn main(@builtin(global_invocation_id) global:vec3<u32>,
        @builtin(local_invocation_id) local:vec3<u32>,
        @builtin(local_invocation_index) index:u32,
        @builtin(workgroup_id) block:vec3<u32>,
        @builtin(num_workgroups) grid:vec3<u32>) {
    let offset=global.x+global.y*4u+global.z*16u;
    output[offset]=output[offset]*3u+index+block.x*11u+block.y*13u+block.z*17u+grid.x*19u+local.z*23u;
}
"#;

#[test]
fn compute_3d_ids_and_storage_read_modify_write_match_wgsl_and_spirv() {
    for module in modules(GRID) {
        let shader = compile_shader(&module, ShaderStage::Compute, "main").unwrap();
        assert_eq!(shader.workgroup_size, Some([2, 2, 2]));
        assert!(shader.storage_buffers[0].writable);
        assert!(
            shader
                .tgsi
                .starts_with("COMP\nPROPERTY CS_FIXED_BLOCK_WIDTH 2\n")
        );
        let mut buffer: Vec<_> = (0..68).map(|i| 100 + i).collect();
        for bz in 0..2 {
            for by in 0..2 {
                for bx in 0..2 {
                    for z in 0..2 {
                        for y in 0..2 {
                            for x in 0..2 {
                                execute(
                                    &shader,
                                    &[[8, 256, 0, 0]],
                                    [[bx, by, bz, 0], [2, 2, 2, 0], [x, y, z, 0]],
                                    &mut [&mut buffer],
                                );
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(&buffer[..2], &[100, 101]);
        assert_eq!(&buffer[66..], &[166, 167]);
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    let offset = x + y * 4 + z * 16;
                    let index = x % 2 + (y % 2) * 2 + (z % 2) * 4;
                    let expected = (102 + offset) * 3
                        + index
                        + x / 2 * 11
                        + y / 2 * 13
                        + z / 2 * 17
                        + 2 * 19
                        + z % 2 * 23;
                    assert_eq!(buffer[2 + offset as usize], expected, "({x},{y},{z})");
                }
            }
        }
    }
}

#[test]
fn storage_bounds_and_overflow_do_not_touch_adjacent_descriptor_bytes() {
    let text = r#"
    @group(0) @binding(0) var<storage,read_write> data:array<vec4<u32>>;
    @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3<u32>) {
        data[id.x]=vec4<u32>(data[id.x].y+1u,2u,3u,4u);
    }"#;
    for module in modules(text) {
        let shader = compile_shader(&module, ShaderStage::Compute, "main").unwrap();
        for index in [0, 1, 2, 0x10000000, 0x40000000, u32::MAX] {
            let mut data = [99u32; 16];
            execute(
                &shader,
                &[[16, 32, 0, 0]],
                [[index, 0, 0, 0], [1, 1, 1, 0], [0; 4]],
                &mut [&mut data],
            );
            let mut expected = [99; 16];
            if index < 2 {
                expected[4 + index as usize * 4..8 + index as usize * 4]
                    .copy_from_slice(&[100, 2, 3, 4]);
            }
            assert_eq!(data, expected, "index={index}");
        }
    }
}

#[test]
fn storage_fixed_subarray_index_cannot_alias_next_struct_member() {
    let text = r#"
    struct Data { a:array<u32,2>, next:u32, }
    @group(0) @binding(0) var<storage,read_write> data:Data;
    @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3<u32>) { data.a[id.x]=7u; }
    "#;
    for module in modules(text) {
        let shader = compile_shader(&module, ShaderStage::Compute, "main").unwrap();
        for index in [0, 1, 2, u32::MAX] {
            let mut data = [99u32; 4];
            execute(
                &shader,
                &[[0, 16, 0, 0]],
                [[index, 0, 0, 0], [1, 1, 1, 0], [0; 4]],
                &mut [&mut data],
            );
            let mut expected = [99; 4];
            if index < 2 {
                expected[index as usize] = 7;
            }
            assert_eq!(data, expected);
        }
    }
}

#[test]
fn invalid_reads_return_zero_and_uniform_push_storage_registers_stay_distinct() {
    let text = r#"
    struct Params { bias:u32, }
    @group(0) @binding(0) var<uniform> params:Params;
    var<push_constant> pc:vec4<u32>;
    @group(0) @binding(1) var<storage,read> input:array<u32>;
    @group(0) @binding(2) var<storage,read_write> output:array<u32>;
    @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3<u32>) {
        output[0]=input[id.x]+params.bias+pc.x;
    }"#;
    for module in modules(text) {
        let shader = compile_shader(&module, ShaderStage::Compute, "main").unwrap();
        assert_eq!(shader.uniform_buffers[0].first_register, 0);
        assert_eq!(shader.push_constants.as_ref().unwrap().first_register, 1);
        assert_eq!(
            shader
                .storage_buffers
                .iter()
                .map(|b| (b.first_register, b.writable))
                .collect::<Vec<_>>(),
            [(2, false), (3, true)]
        );
        for index in [0, 1, 2, 0x40000000, u32::MAX] {
            let mut input = [33u32, 40, 50, 66];
            let mut output = [0xbadd];
            execute(
                &shader,
                &[[7, 0, 0, 0], [11, 0, 0, 0], [4, 8, 0, 0], [0, 4, 0, 0]],
                [[index, 0, 0, 0], [1, 1, 1, 0], [0; 4]],
                &mut [&mut input, &mut output],
            );
            assert_eq!(input, [33, 40, 50, 66]);
            assert_eq!(
                output,
                [18 + match index {
                    0 => 40,
                    1 => 50,
                    _ => 0,
                }]
            );
        }
    }
}

#[test]
fn storage_barrier_and_compute_profile_rejections_are_explicit() {
    for module in modules("@compute @workgroup_size(1) fn main() { storageBarrier(); }") {
        let shader = compile_shader(&module, ShaderStage::Compute, "main").unwrap();
        assert!(shader.tgsi.contains("MEMBAR IMM["));
        assert!(shader.tgsi.contains("BARRIER"));
    }
    for text in [
        "@compute @workgroup_size(129) fn main() {}",
        "@compute @workgroup_size(8,8,4) fn main() {}",
        "var<workgroup> scratch:array<u32,4>; @compute @workgroup_size(4) fn main() { scratch[0]=1u; }",
        "@group(0) @binding(0) var<storage,read_write> value:atomic<u32>; @compute @workgroup_size(1) fn main() { atomicAdd(&value,1u); }",
    ] {
        for module in modules(text) {
            assert!(
                compile_shader(&module, ShaderStage::Compute, "main").is_err(),
                "{text}"
            );
        }
    }
}

#[test]
#[ignore = "exports TGSI for the pinned virglrenderer parser"]
fn export_compute_tgsi() {
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/compute-validation");
    std::fs::create_dir_all(&root).unwrap();
    for (i, module) in modules(GRID).iter().enumerate() {
        std::fs::write(
            root.join(format!("compute-{i}.tgsi")),
            compile_shader(module, ShaderStage::Compute, "main")
                .unwrap()
                .tgsi,
        )
        .unwrap();
    }
}
