//! Repack scalar uniform arrays into vec4 arrays without changing byte offsets.
//! `float[N]` with stride 4 becomes `vec4[N/4]` with stride 16; accesses use
//! `[index >> 2][index & 3]`. Other scalar-layout structures remain unsupported.
//! This changes shader types and accesses, never the bound buffer's contents.
use ash::vk;
use std::collections::{HashMap, HashSet};
const UNSUPPORTED: vk::Result = vk::Result::ERROR_FEATURE_NOT_PRESENT;
#[derive(Clone)]
enum Ty {
    Scalar,
    Array(u32, u32),
    Struct(Vec<u32>),
    Pointer(u32),
    Other,
}
fn instruction(out: &mut Vec<u32>, op: u32, operands: &[u32]) {
    out.push(((operands.len() as u32 + 1) << 16) | op);
    out.extend_from_slice(operands);
}
pub(crate) fn lower(words: Vec<u32>) -> Result<Vec<u32>, vk::Result> {
    if words.len() < 5 || words[0] != 0x07230203 {
        return Err(UNSUPPORTED);
    }
    // Each original word can require at most six new IDs. Reserve enough
    // headroom before allocating IDs, including for malformed module bounds.
    let extra = u32::try_from(words.len())
        .ok()
        .and_then(|n| n.checked_mul(6))
        .ok_or(UNSUPPORTED)?;
    words[3].checked_add(extra).ok_or(UNSUPPORTED)?;
    let mut instructions = Vec::new();
    let mut offset = 5;
    while offset < words.len() {
        let n = (words[offset] >> 16) as usize;
        if n == 0 || offset.checked_add(n).is_none_or(|end| end > words.len()) {
            return Err(UNSUPPORTED);
        }
        let op = words[offset] & 0xffff;
        let minimum = match op {
            21 | 28 | 32 | 43 | 59 => 4,
            22 | 55 => 3,
            65 | 66 => 5,
            19..=39 => 2,
            _ => 1,
        };
        if n < minimum {
            return Err(UNSUPPORTED);
        }
        instructions.push(&words[offset..offset + n]);
        offset += n;
    }
    let mut types = HashMap::new();
    let mut constants = HashMap::new();
    let mut result_types = HashMap::new();
    let mut strides = HashMap::new();
    let mut uniforms = Vec::new();
    for ins in &instructions {
        match (ins[0] & 0xffff, ins.len()) {
            (21, 4) if ins[2] == 32 => {
                types.insert(ins[1], Ty::Scalar);
            }
            (22, 3) if ins[2] == 32 => {
                types.insert(ins[1], Ty::Scalar);
            }
            (28, 4) => {
                types.insert(ins[1], Ty::Array(ins[2], ins[3]));
            }
            (30, _) => {
                types.insert(ins[1], Ty::Struct(ins[2..].to_vec()));
            }
            (32, 4) => {
                types.insert(ins[1], Ty::Pointer(ins[3]));
            }
            (43, 4) => {
                constants.insert(ins[2], ins[3]);
                result_types.insert(ins[2], ins[1]);
            }
            (59, _) => {
                result_types.insert(ins[2], ins[1]);
                if ins[3] == 2 {
                    uniforms.push(ins[1]);
                }
            }
            (65 | 66 | 55, _) => {
                result_types.insert(ins[2], ins[1]);
            }
            (71, 4) if ins[2] == 6 => {
                strides.insert(ins[1], ins[3]);
            }
            (op, _) if (19..=39).contains(&op) => {
                types.entry(ins[1]).or_insert(Ty::Other);
            }
            _ => (),
        }
    }
    let mut packed = HashSet::new();
    let mut seen = HashSet::new();
    let mut visit = uniforms;
    while let Some(id) = visit.pop() {
        if !seen.insert(id) {
            continue;
        }
        match types.get(&id) {
            Some(Ty::Pointer(base)) => visit.push(*base),
            Some(Ty::Struct(fields)) => visit.extend(fields),
            Some(Ty::Array(base, length)) => {
                visit.push(*base);
                if matches!(types.get(base), Some(Ty::Scalar)) && strides.get(&id) == Some(&4) {
                    let n = constants.get(length).copied().ok_or(UNSUPPORTED)?;
                    // Keep allocation size unchanged; partial vec4 tails need
                    // separate descriptor-range padding, which is not enabled.
                    if n == 0 || n % 4 != 0 {
                        return Err(UNSUPPORTED);
                    }
                    packed.insert(id);
                }
            }
            _ => (),
        }
    }
    if packed.is_empty() {
        return Ok(words);
    }
    let mut next = words[3];
    let mut id = || {
        let current = next;
        next += 1;
        current
    };
    let uint = id();
    let two = id();
    let three = id();
    let mut globals = Vec::new();
    instruction(&mut globals, 21, &[uint, 32, 0]);
    instruction(&mut globals, 43, &[uint, two, 2]);
    instruction(&mut globals, 43, &[uint, three, 3]);
    let mut declarations = HashMap::new();
    // Emit per-array declarations before the original array type, including
    // its length constant and the vector type it now references.
    for ins in &instructions {
        if ins[0] & 0xffff == 28 && packed.contains(&ins[1]) {
            let vector = id();
            let length = id();
            let n = constants[&ins[3]] / 4;
            declarations.insert(ins[1], (vector, length, n));
        }
    }
    let mut out = words[..5].to_vec();
    let mut body = Vec::new();
    let mut functions = false;
    let mut emitted_uint = false;
    for ins in &instructions {
        let op = ins[0] & 0xffff;
        if op == 54 {
            functions = true;
        }
        if op == 71 && ins.len() == 4 && ins[2] == 6 && packed.contains(&ins[1]) {
            instruction(&mut out, op, &[ins[1], 6, 16]);
            continue;
        }
        if op == 28 && packed.contains(&ins[1]) {
            if !emitted_uint {
                out.append(&mut globals);
                emitted_uint = true;
            }
            let (vector, length, n) = declarations[&ins[1]];
            instruction(&mut out, 23, &[vector, ins[2], 4]);
            instruction(&mut out, 43, &[uint, length, n]);
            instruction(&mut out, 28, &[ins[1], vector, length]);
            continue;
        }
        if functions && matches!(op, 65 | 66) {
            let mut ty = match result_types.get(&ins[3]).and_then(|ty| types.get(ty)) {
                Some(Ty::Pointer(base)) => *base,
                _ => {
                    body.extend_from_slice(ins);
                    continue;
                }
            };
            let mut operands = ins[1..4].to_vec();
            for &index in &ins[4..] {
                match types.get(&ty) {
                    Some(Ty::Struct(fields)) => {
                        let member = *constants.get(&index).ok_or(UNSUPPORTED)? as usize;
                        ty = *fields.get(member).ok_or(UNSUPPORTED)?;
                        operands.push(index);
                    }
                    Some(Ty::Array(base, _)) => {
                        if packed.contains(&ty) {
                            if let Some(&n) = constants.get(&index) {
                                let row = id();
                                let lane = id();
                                instruction(&mut globals, 43, &[uint, row, n / 4]);
                                instruction(&mut globals, 43, &[uint, lane, n % 4]);
                                operands.extend([row, lane]);
                            } else {
                                let unsigned = id();
                                let row = id();
                                let lane = id();
                                instruction(&mut body, 124, &[uint, unsigned, index]);
                                instruction(&mut body, 194, &[uint, row, unsigned, two]);
                                instruction(&mut body, 199, &[uint, lane, unsigned, three]);
                                operands.extend([row, lane]);
                            }
                        } else {
                            operands.push(index);
                        }
                        ty = *base;
                    }
                    _ => operands.push(index),
                }
            }
            instruction(&mut body, op, &operands);
        } else if functions {
            body.extend_from_slice(ins);
        } else {
            out.extend_from_slice(ins);
        }
    }
    out.append(&mut globals);
    out.append(&mut body);
    out[3] = next;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Fragment shader reads one scalar from a tightly packed 32-byte UBO.
    // Dynamic indexing comes from a flat integer fragment input.
    fn shader(dynamic: bool, element_count: u32) -> Vec<u32> {
        let mut words = vec![0x07230203, 0x00010000, 0, 25, 0];
        for (op, operands) in [
            (17, vec![1]),
            (14, vec![0, 1]),
            (15, vec![4, 20, u32::from_le_bytes(*b"main"), 0, 14, 16]),
            (16, vec![20, 7]),
            (71, vec![7, 6, 4]),
            (71, vec![8, 2]),
            (72, vec![8, 0, 35, 0]),
            (71, vec![10, 33, 0]),
            (71, vec![10, 34, 0]),
            (71, vec![14, 30, 0]),
            (71, vec![14, 14]),
            (71, vec![16, 30, 0]),
            (19, vec![1]),
            (33, vec![2, 1]),
            (22, vec![3, 32]),
            (21, vec![4, 32, 0]),
            (43, vec![4, 5, 0]),
            (43, vec![4, 6, element_count]),
            (28, vec![7, 3, 6]),
            (30, vec![8, 7]),
            (32, vec![9, 2, 8]),
            (59, vec![9, 10, 2]),
            (32, vec![11, 2, 3]),
            (43, vec![4, 12, 5]),
            (32, vec![13, 1, 4]),
            (59, vec![13, 14, 1]),
            (32, vec![15, 3, 3]),
            (59, vec![15, 16, 3]),
            (54, vec![1, 20, 0, 2]),
            (248, vec![21]),
            (61, vec![4, 22, 14]),
            (65, vec![11, 23, 10, 5, if dynamic { 22 } else { 12 }]),
            (61, vec![3, 24, 23]),
            (62, vec![16, 24]),
            (253, vec![]),
            (56, vec![]),
        ] {
            instruction(&mut words, op, &operands);
        }
        words
    }

    fn parse(words: &[u32]) -> naga::Module {
        naga::front::spv::Frontend::new(words.iter().copied(), &Default::default())
            .parse()
            .unwrap()
    }

    fn validate(
        module: &naga::Module,
    ) -> Result<naga::valid::ModuleInfo, Box<naga::WithSpan<naga::valid::ValidationError>>> {
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(module)
        .map_err(Box::new)
    }

    fn scalar_read_offset(words: &[u32], input: u32) -> u32 {
        // Interpret the rewritten address calculation independently of Naga.
        let mut values = HashMap::new();
        let mut offset = 5;
        while offset < words.len() {
            let n = (words[offset] >> 16) as usize;
            let ins = &words[offset..offset + n];
            match ins[0] & 0xffff {
                43 => {
                    values.insert(ins[2], ins[3]);
                }
                61 if ins[3] == 14 => {
                    values.insert(ins[2], input);
                }
                124 => {
                    values.insert(ins[2], values[&ins[3]]);
                }
                194 => {
                    values.insert(ins[2], values[&ins[3]] >> values[&ins[4]]);
                }
                199 => {
                    values.insert(ins[2], values[&ins[3]] & values[&ins[4]]);
                }
                65 if ins[3] == 10 => {
                    assert_eq!(ins.len(), 7);
                    assert_eq!(values[&ins[4]], 0);
                    return values[&ins[5]] * 16 + values[&ins[6]] * 4;
                }
                134 | 137 => panic!("uniform addressing must not emit division or modulo"),
                _ => (),
            }
            offset += n;
        }
        panic!("missing scalar read")
    }

    #[test]
    fn static_and_dynamic_uniform_reads_preserve_byte_addresses_and_size() {
        for dynamic in [false, true] {
            let original = shader(dynamic, 8);
            assert!(validate(&parse(&original)).is_err());
            let lowered = lower(original).unwrap();
            let module = parse(&lowered);
            validate(&module).unwrap();
            let uniform = module
                .global_variables
                .iter()
                .find(|(_, global)| global.space == naga::AddressSpace::Uniform)
                .unwrap()
                .1;
            let naga::TypeInner::Struct { members, span } = &module.types[uniform.ty].inner else {
                panic!("uniform block expected")
            };
            assert_eq!(*span, 32);
            assert_eq!(members[0].offset, 0);
            for index in 0..8 {
                assert_eq!(
                    scalar_read_offset(&lowered, index),
                    4 * if dynamic { index } else { 5 }
                );
            }
        }
    }

    #[test]
    fn partial_vector_tail_is_rejected_without_padding_the_buffer() {
        assert_eq!(lower(shader(false, 6)), Err(UNSUPPORTED));
    }

    #[test]
    fn malformed_instructions_and_exhausted_ids_return_errors() {
        for opcode in [19, 21, 22, 28, 30, 32, 43, 55, 59, 65, 66] {
            let mut words = vec![0x07230203, 0x00010000, 0, 25, 0];
            instruction(&mut words, opcode, &[]);
            assert_eq!(lower(words), Err(UNSUPPORTED));
        }
        let mut words = shader(true, 8);
        words[3] = u32::MAX;
        assert_eq!(lower(words), Err(UNSUPPORTED));
    }
}
