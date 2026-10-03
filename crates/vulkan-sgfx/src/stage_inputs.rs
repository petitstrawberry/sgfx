//! Preserve packed fragment Location/Component inputs before Naga drops Component.
//! A location becomes one vec4 input; original typed loads extract their lanes.
use crate::spirv::{emit, instructions};
use ash::vk;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

fn unsupported(reason: &str) -> vk::Result {
    if std::env::var_os("SGFX_VULKAN_TRACE").is_some() {
        eprintln!("[SGFX Vulkan] stage input lowering: {reason}");
    }
    vk::Result::ERROR_FEATURE_NOT_PRESENT
}

#[derive(Default)]
struct Decorations {
    location: Option<u32>,
    component: Option<u32>,
    other: BTreeSet<Vec<u32>>,
}

struct Input {
    id: u32,
    ty: u32,
    scalar: u32,
    width: u32,
    component: u32,
}

struct Packed {
    ty: u32,
    id: u32,
}

fn interface_start(inst: &[u32]) -> Result<usize, vk::Result> {
    if inst.len() < 4 {
        return Err(unsupported("malformed OpEntryPoint"));
    }
    inst[3..]
        .iter()
        .position(|word| word.to_le_bytes().contains(&0))
        .map(|offset| offset + 4)
        .ok_or_else(|| unsupported("unterminated entry-point name"))
}

pub(crate) fn lower(words: Vec<u32>) -> Result<Vec<u32>, vk::Result> {
    if words.len() < 5 || words[0] != 0x07230203 {
        return Err(unsupported("invalid SPIR-V header"));
    }
    let code = instructions(&words)?;
    let mut decorations: HashMap<u32, Decorations> = HashMap::new();
    let mut pointers = HashMap::new();
    let mut variables = HashMap::new();
    let mut scalars = HashSet::new();
    let mut vectors = HashMap::new();
    for inst in &code {
        match inst[0] & 0xffff {
            21 if inst.len() == 4 && inst[2] == 32 && inst[3] <= 1 => {
                scalars.insert(inst[1]);
            }
            22 if inst.len() == 3 && inst[2] == 32 => {
                scalars.insert(inst[1]);
            }
            23 if inst.len() == 4 => {
                vectors.insert(inst[1], (inst[2], inst[3]));
            }
            32 if inst.len() == 4 => {
                pointers.insert(inst[1], (inst[2], inst[3]));
            }
            59 if inst.len() >= 4 => {
                variables.insert(inst[2], (inst[1], inst[3], inst.len()));
            }
            71 if inst.len() >= 3 => {
                let target = decorations.entry(inst[1]).or_default();
                match inst[2] {
                    30 | 31 => {
                        if inst.len() != 4 {
                            return Err(unsupported("malformed Location/Component decoration"));
                        }
                        let slot = if inst[2] == 30 {
                            &mut target.location
                        } else {
                            &mut target.component
                        };
                        if slot.replace(inst[3]).is_some_and(|old| old != inst[3]) {
                            return Err(unsupported("conflicting Location/Component decorations"));
                        }
                        if inst[2] == 31 && inst[3] > 3 {
                            return Err(unsupported("Component exceeds a 32-bit location"));
                        }
                    }
                    _ => {
                        target.other.insert(inst[2..].to_vec());
                    }
                }
            }
            72 if inst.len() >= 4 && inst[3] == 31 => {
                if inst.len() != 5 || inst[4] != 0 {
                    return Err(unsupported("nonzero member Component is not supported"));
                }
            }
            _ => {}
        }
    }
    let mut groups: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for (&id, decor) in &decorations {
        let storage = variables.get(&id).map(|v| v.1);
        if decor.component.is_some_and(|component| component != 0)
            && (storage != Some(1) || decor.location.is_none())
        {
            return Err(unsupported(
                "nonzero Component requires a direct located input",
            ));
        }
        if storage == Some(1)
            && let Some(location) = decor.location
        {
            groups.entry(location).or_default().push(id);
        }
    }
    groups.retain(|_, ids| {
        ids.len() > 1
            || decorations[&ids[0]]
                .component
                .is_some_and(|component| component != 0)
    });
    if groups.is_empty() {
        return Ok(words);
    }
    let mut next = words[3];
    let mut fresh = || {
        let id = next;
        next = next
            .checked_add(1)
            .filter(|bound| *bound <= 0x3fffff)
            .ok_or_else(|| unsupported("SPIR-V ID bound exhausted"))?;
        Ok::<_, vk::Result>(id)
    };
    let mut annotations = Vec::new();
    let mut declarations = Vec::new();
    let mut inputs = HashMap::new();
    let mut packed = HashMap::new();
    for (&location, ids) in &mut groups {
        ids.sort_unstable();
        let mut members = Vec::new();
        let mut mask = 0u32;
        let qualifiers = &decorations[&ids[0]].other;
        if qualifiers
            .iter()
            .any(|dec| dec.len() != 1 || !matches!(dec[0], 0 | 13 | 14 | 16 | 17 | 18))
        {
            return Err(unsupported("unsupported decoration on packed input"));
        }
        for &id in ids.iter() {
            let decor = &decorations[&id];
            if &decor.other != qualifiers {
                return Err(unsupported("packed input qualifiers differ"));
            }
            let &(pointer, storage, length) = &variables[&id];
            let &(pointer_storage, ty) = pointers
                .get(&pointer)
                .ok_or_else(|| unsupported("packed input has no pointer type"))?;
            if storage != 1 || pointer_storage != 1 || length != 4 {
                return Err(unsupported("unsupported input storage or initializer"));
            }
            let (scalar, width) = vectors.get(&ty).copied().unwrap_or((ty, 1));
            let component = decor.component.unwrap_or(0);
            if !scalars.contains(&scalar) || !(1..=4).contains(&width) || component + width > 4 {
                return Err(unsupported(
                    "packed inputs require 32-bit scalar/vector lanes",
                ));
            }
            if members
                .first()
                .is_some_and(|first: &Input| first.scalar != scalar)
            {
                return Err(unsupported("packed input scalar types differ"));
            }
            let lanes = ((1 << width) - 1) << component;
            if mask & lanes != 0 {
                return Err(unsupported("packed input Component ranges overlap"));
            }
            mask |= lanes;
            members.push(Input {
                id,
                ty,
                scalar,
                width,
                component,
            });
        }
        let scalar = members[0].scalar;
        let vec4 = fresh()?;
        let pointer = fresh()?;
        let variable = fresh()?;
        emit(&mut declarations, 23, &[vec4, scalar, 4]);
        emit(&mut declarations, 32, &[pointer, 1, vec4]);
        emit(&mut declarations, 59, &[pointer, variable, 1]);
        emit(&mut annotations, 71, &[variable, 30, location]);
        for qualifier in qualifiers {
            let mut operands = vec![variable];
            operands.extend_from_slice(qualifier);
            emit(&mut annotations, 71, &operands);
        }
        for member in members {
            packed.insert(
                member.id,
                Packed {
                    ty: vec4,
                    id: variable,
                },
            );
            inputs.insert(member.id, member);
        }
    }

    // Original pointers must only feed whole-value OpLoad instructions. Check
    // executable operands before deleting variables, so aliases fail closed.
    let mut in_function = false;
    let mut seen_interfaces = HashSet::new();
    for inst in &code {
        let opcode = inst[0] & 0xffff;
        if opcode == 15 {
            let start = interface_start(inst)?;
            for id in &inst[start..] {
                if inputs.contains_key(id) {
                    if inst[1] != 4 {
                        return Err(unsupported("packed inputs are currently fragment-only"));
                    }
                    seen_interfaces.insert(*id);
                }
            }
        }
        if opcode == 74 {
            if inst.len() < 3 {
                return Err(unsupported("malformed OpGroupDecorate"));
            }
            if inst[2..].iter().any(|id| inputs.contains_key(id)) {
                return Err(unsupported("group decoration on packed input"));
            }
        }
        if opcode == 332 && inst.get(1).is_some_and(|id| inputs.contains_key(id)) {
            return Err(unsupported("OpDecorateId on packed input"));
        }
        if opcode == 54 {
            in_function = true;
        }
        if !in_function {
            continue;
        }
        let operands: &[u32] = match opcode {
            61 if inst.len() >= 4 && inputs.contains_key(&inst[3]) => {
                if inst[1] != inputs[&inst[3]].ty {
                    return Err(unsupported("packed input load type mismatch"));
                }
                continue;
            }
            // These positions are IDs; the remaining operands are literals.
            8 | 317 | 54 | 56 | 248 | 249 | 250 | 251 | 252 | 253 => &[],
            59 if inst.len() >= 4 => &inst[4..],
            62..=64 if inst.len() >= 3 => &inst[1..3],
            79 | 82 if inst.len() >= 5 => &inst[3..5],
            81 if inst.len() >= 4 => &inst[3..4],
            12 if inst.len() >= 5 => &inst[5..],
            _ => &inst[1..],
        };
        if operands.iter().any(|id| inputs.contains_key(id)) {
            return Err(unsupported("packed input pointer alias or unsupported use"));
        }
    }
    if inputs.keys().any(|id| !seen_interfaces.contains(id)) {
        return Err(unsupported(
            "packed input is absent from fragment entry interface",
        ));
    }

    let mut result = words[..5].to_vec();
    let mut annotated = false;
    let mut declared = false;
    for inst in code {
        let opcode = inst[0] & 0xffff;
        if !annotated && (19..=39).contains(&opcode) {
            result.extend_from_slice(&annotations);
            annotated = true;
        }
        if !declared && opcode == 54 {
            result.extend_from_slice(&declarations);
            declared = true;
        }
        match opcode {
            5 | 71 if inst.len() >= 2 && inputs.contains_key(&inst[1]) => {}
            59 if inst.len() >= 3 && inputs.contains_key(&inst[2]) => {}
            15 => {
                let start = interface_start(inst)?;
                let mut operands = inst[1..start].to_vec();
                let mut emitted = HashSet::new();
                for &id in &inst[start..] {
                    let id = packed.get(&id).map_or(id, |value| value.id);
                    if emitted.insert(id) {
                        operands.push(id);
                    }
                }
                emit(&mut result, 15, &operands);
            }
            61 if inst.len() >= 4 && inputs.contains_key(&inst[3]) => {
                let input = &inputs[&inst[3]];
                let packed = &packed[&input.id];
                let loaded = fresh()?;
                let mut operands = vec![packed.ty, loaded, packed.id];
                operands.extend_from_slice(&inst[4..]);
                emit(&mut result, 61, &operands);
                if input.width == 1 {
                    emit(
                        &mut result,
                        81,
                        &[input.ty, inst[2], loaded, input.component],
                    );
                } else {
                    let mut operands = vec![input.ty, inst[2], loaded, loaded];
                    operands.extend(input.component..input.component + input.width);
                    emit(&mut result, 79, &operands);
                }
            }
            _ => result.extend_from_slice(inst),
        }
    }
    if !annotated || !declared {
        return Err(unsupported(
            "packed input module has no declaration/function section",
        ));
    }
    result[3] = next;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Vec<u32> {
        let mut words = vec![0x07230203, 0x00010000, 0, 20, 0];
        for (opcode, operands) in [
            (17, vec![1]),
            (14, vec![0, 1]),
            (15, vec![4, 12, u32::from_le_bytes(*b"main"), 0, 9, 10, 11]),
            (16, vec![12, 7]),
            (71, vec![9, 30, 0]),
            (71, vec![10, 30, 0]),
            (71, vec![10, 31, 3]),
            (71, vec![11, 30, 0]),
            (19, vec![1]),
            (33, vec![2, 1]),
            (22, vec![3, 32]),
            (23, vec![4, 3, 2]),
            (23, vec![5, 3, 4]),
            (32, vec![6, 1, 4]),
            (32, vec![7, 1, 3]),
            (32, vec![8, 3, 5]),
            (43, vec![3, 16, 0]),
            (59, vec![6, 9, 1]),
            (59, vec![7, 10, 1]),
            (59, vec![8, 11, 3]),
            (54, vec![1, 12, 0, 2]),
            (248, vec![13]),
            (61, vec![4, 14, 9]),
            (61, vec![3, 15, 10]),
            (81, vec![3, 17, 14, 0]),
            (81, vec![3, 18, 14, 1]),
            (80, vec![5, 19, 17, 18, 16, 15]),
            (62, vec![11, 19]),
            (253, vec![]),
            (56, vec![]),
        ] {
            emit(&mut words, opcode, &operands);
        }
        words
    }
    fn rewrite(mut words: Vec<u32>, f: impl Fn(&mut Vec<u32>)) -> Vec<u32> {
        let mut output = words[..5].to_vec();
        for inst in instructions(&words).unwrap() {
            let mut inst = inst.to_vec();
            f(&mut inst);
            output.extend(inst);
        }
        words.clear();
        output
    }
    #[test]
    fn packed_xy_and_w_preserve_load_ids_and_interface() {
        let lowered = lower(fixture()).unwrap();
        let code = instructions(&lowered).unwrap();
        let shuffle = code
            .iter()
            .find(|i| i[0] & 0xffff == 79 && i[2] == 14)
            .unwrap();
        assert_eq!(shuffle[1], 4);
        assert_eq!(&shuffle[5..], &[0, 1]);
        let extract = code
            .iter()
            .find(|i| i[0] & 0xffff == 81 && i[2] == 15)
            .unwrap();
        assert_eq!(extract[1], 3);
        assert_eq!(extract[4], 3);
        assert!(
            !code
                .iter()
                .any(|i| i[0] & 0xffff == 59 && matches!(i[2], 9 | 10))
        );
        let entry = code.iter().find(|i| i[0] & 0xffff == 15).unwrap();
        assert_eq!(entry[interface_start(entry).unwrap()..].len(), 2);
        let desc = sgfx::ir::ShaderModuleDesc::spirv(lowered).unwrap();
        let shader = sgfx_codegen_virgl::programmable::compile_shader(
            &desc,
            sgfx::ir::ShaderStage::Fragment,
            "main",
        )
        .unwrap();
        assert_eq!(shader.inputs.len(), 1);
        assert_eq!(shader.inputs[0].location, 0);
        assert_eq!(shader.inputs[0].components, 4);
        assert!(shader.tgsi.contains("IN[0].wwww"));
    }
    #[test]
    fn overlapping_component_ranges_are_rejected() {
        let words = rewrite(fixture(), |i| {
            if i[0] & 0xffff == 71 && i[1] == 10 && i[2] == 31 {
                i[3] = 1;
            }
        });
        assert_eq!(lower(words), Err(vk::Result::ERROR_FEATURE_NOT_PRESENT));
    }
    #[test]
    fn differing_interpolation_and_output_components_are_rejected() {
        for extra in [vec![9, 14], vec![11, 31, 1]] {
            let words = rewrite(fixture(), |i| {
                if i[0] & 0xffff == 71 && i[1] == 9 {
                    emit(i, 71, &extra);
                }
            });
            assert_eq!(lower(words), Err(vk::Result::ERROR_FEATURE_NOT_PRESENT));
        }
    }
    #[test]
    fn input_pointer_alias_is_rejected() {
        let mut words = rewrite(fixture(), |i| {
            if i[0] & 0xffff == 248 {
                emit(i, 83, &[6, 20, 9]);
            }
        });
        words[3] = 21;
        assert_eq!(lower(words), Err(vk::Result::ERROR_FEATURE_NOT_PRESENT));
    }
    #[test]
    fn truncated_group_decoration_returns_error_without_panicking() {
        let words = rewrite(fixture(), |i| {
            if i[0] & 0xffff == 71 && i[1] == 9 {
                emit(i, 74, &[]);
            }
        });
        assert_eq!(lower(words), Err(vk::Result::ERROR_FEATURE_NOT_PRESENT));
    }
    #[test]
    fn separate_unpacked_locations_are_unchanged() {
        let words = rewrite(fixture(), |i| {
            if i[0] & 0xffff == 71 && i[1] == 10 {
                match i[2] {
                    30 => i[3] = 1,
                    31 => i.clear(),
                    _ => {}
                }
            }
        });
        assert_eq!(lower(words.clone()).unwrap(), words);
    }
}
