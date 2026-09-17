//! Lower single-sample Vulkan subpass inputs to same-pixel GPU texture loads.
//! Attachment formats specialize depth handles before Naga reflection/validation.
use crate::spirv::{emit, instructions};
use ash::vk;
use std::collections::{HashMap, HashSet};
const UNSUPPORTED: vk::Result = vk::Result::ERROR_FEATURE_NOT_PRESENT;

#[derive(Clone, Copy, Debug)]
pub(crate) struct InputBinding {
    pub group: u32,
    pub binding: u32,
    pub index: usize,
}

pub(crate) fn bindings(words: &[u32]) -> Result<Vec<InputBinding>, vk::Result> {
    let mut groups = HashMap::new();
    let mut bindings = HashMap::new();
    let mut inputs = Vec::new();
    for inst in instructions(words)? {
        if inst[0] & 0xffff == 71 && inst.len() == 4 {
            match inst[2] {
                33 => {
                    bindings.insert(inst[1], inst[3]);
                }
                34 => {
                    groups.insert(inst[1], inst[3]);
                }
                43 => inputs.push((inst[1], inst[3] as usize)),
                _ => {}
            }
        }
    }
    inputs
        .into_iter()
        .map(|(id, index)| {
            Ok(InputBinding {
                group: *groups.get(&id).ok_or(UNSUPPORTED)?,
                binding: *bindings.get(&id).ok_or(UNSUPPORTED)?,
                index,
            })
        })
        .collect()
}

pub(crate) fn lower(words: Vec<u32>, inputs: &[Option<bool>]) -> Result<Vec<u32>, vk::Result> {
    let code = instructions(&words)?;
    for inst in &code {
        let minimum = match inst[0] & 0xffff {
            25 => 9,
            32 | 59 | 61 => 4,
            98 => 5,
            _ => 1,
        };
        if inst.len() < minimum {
            return Err(UNSUPPORTED);
        }
    }
    let subpass_types: HashMap<_, _> = code
        .iter()
        .filter(|i| i[0] & 0xffff == 25 && i.len() >= 9 && i[3] == 6)
        .map(|i| (i[1], *i))
        .collect();
    if subpass_types.is_empty() {
        return Ok(words);
    }
    let mut next = words[3];
    let mut id = || {
        let result = next;
        next += 1;
        result
    };
    let pointers: HashMap<_, _> = code
        .iter()
        .filter(|i| i[0] & 0xffff == 32 && i.len() == 4)
        .map(|i| (i[1], (i[2], i[3])))
        .collect();
    let variables: HashMap<_, _> = code
        .iter()
        .filter(|i| i[0] & 0xffff == 59 && i.len() >= 4)
        .map(|i| (i[2], i[1]))
        .collect();
    let decorations: HashMap<_, _> = code
        .iter()
        .filter(|i| i[0] & 0xffff == 71 && i.len() == 4 && i[2] == 43)
        .map(|i| (i[1], i[3] as usize))
        .collect();
    let mut image_variables = HashMap::new();
    let mut depth_types = HashMap::new();
    let mut depth_pointers = HashMap::new();
    for (&var, &index) in &decorations {
        let depth = inputs.get(index).copied().flatten().ok_or(UNSUPPORTED)?;
        let pointer = *variables.get(&var).ok_or(UNSUPPORTED)?;
        let &(storage, ty) = pointers.get(&pointer).ok_or(UNSUPPORTED)?;
        let image = subpass_types.get(&ty).ok_or(UNSUPPORTED)?;
        // Array and multisample inputs need separate coordinate/sample lowering.
        if storage != 0 || image[5] != 0 || image[6] != 0 {
            return Err(UNSUPPORTED);
        }
        image_variables.insert(var, (ty, depth));
        if depth {
            depth_types.entry(ty).or_insert_with(&mut id);
            depth_pointers.entry(pointer).or_insert_with(&mut id);
        }
    }
    let frag_coord = code
        .iter()
        .find(|i| i[0] & 0xffff == 71 && i.len() == 4 && i[2] == 11 && i[3] == 15)
        .map(|i| i[1]);
    let float = code
        .iter()
        .find(|i| i[0] & 0xffff == 22 && i.len() == 3 && i[2] == 32)
        .map(|i| i[1])
        .ok_or(UNSUPPORTED)?;
    let vec4 = code
        .iter()
        .find(|i| i[0] & 0xffff == 23 && i.len() == 4 && i[2] == float && i[3] == 4)
        .map(|i| i[1])
        .ok_or(UNSUPPORTED)?;
    if let Some(var) = frag_coord {
        let pointer = variables.get(&var).ok_or(UNSUPPORTED)?;
        if pointers.get(pointer) != Some(&(1, vec4)) {
            return Err(UNSUPPORTED);
        }
    }
    let coord_var = frag_coord.unwrap_or_else(&mut id);
    let coord_pointer = id();
    let vec2 = id();
    let int = id();
    let ivec2 = id();
    let lod0 = id();
    let zero = id();
    let one = id();
    let mut declarations = Vec::new();
    emit(&mut declarations, 23, &[vec2, float, 2]);
    emit(&mut declarations, 21, &[int, 32, 1]);
    emit(&mut declarations, 23, &[ivec2, int, 2]);
    emit(&mut declarations, 43, &[int, lod0, 0]);
    emit(&mut declarations, 43, &[float, zero, 0]);
    emit(&mut declarations, 43, &[float, one, 1.0f32.to_bits()]);
    if frag_coord.is_none() {
        emit(&mut declarations, 32, &[coord_pointer, 1, vec4]);
        emit(&mut declarations, 59, &[coord_pointer, coord_var, 1]);
    }
    let mut zeros = HashSet::new();
    let mut loads = HashMap::new();
    let mut result = words[..5].to_vec();
    let mut annotated = frag_coord.is_some();
    let mut declared = false;
    for inst in code {
        let opcode = inst[0] & 0xffff;
        if !annotated && (19..=39).contains(&opcode) {
            emit(&mut result, 71, &[coord_var, 11, 15]);
            annotated = true;
        }
        if !declared && opcode == 54 {
            result.extend_from_slice(&declarations);
            declared = true;
        }
        match opcode {
            17 if inst.len() == 2 && inst[1] == 40 => {} // InputAttachment capability
            71 if inst.len() == 4 && inst[2] == 43 => {} // InputAttachmentIndex
            15 if inst.len() >= 4 && inst[1] == 4 && frag_coord.is_none() => {
                let mut operands = inst[1..].to_vec();
                operands.push(coord_var);
                emit(&mut result, opcode, &operands);
            }
            25 if subpass_types.contains_key(&inst[1]) => {
                let mut operands = inst[1..].to_vec();
                operands[2] = 1;
                operands[3] = 0;
                operands[6] = 1;
                emit(&mut result, opcode, &operands);
                if let Some(&ty) = depth_types.get(&inst[1]) {
                    operands[0] = ty;
                    operands[3] = 1;
                    emit(&mut result, opcode, &operands);
                }
            }
            32 if depth_pointers.contains_key(&inst[1]) => {
                result.extend_from_slice(inst);
                emit(
                    &mut result,
                    opcode,
                    &[depth_pointers[&inst[1]], inst[2], depth_types[&inst[3]]],
                );
            }
            59 if image_variables
                .get(&inst[2])
                .is_some_and(|(_, depth)| *depth) =>
            {
                let mut operands = inst[1..].to_vec();
                operands[0] = depth_pointers[&inst[1]];
                emit(&mut result, opcode, &operands);
            }
            61 if inst.len() >= 4 && subpass_types.contains_key(&inst[1]) => {
                let &(ty, depth) = image_variables.get(&inst[3]).ok_or(UNSUPPORTED)?;
                loads.insert(inst[2], depth);
                let mut operands = inst[1..].to_vec();
                if depth {
                    operands[0] = depth_types[&ty];
                }
                emit(&mut result, opcode, &operands);
            }
            98 if inst.len() >= 5 && loads.contains_key(&inst[3]) => {
                if inst.len() != 5 || !zeros.contains(&inst[4]) || inst[1] != vec4 {
                    return Err(UNSUPPORTED);
                }
                let position = id();
                let xy = id();
                let pixel = id();
                emit(&mut result, 61, &[vec4, position, coord_var]);
                emit(&mut result, 79, &[vec2, xy, position, position, 0, 1]);
                emit(&mut result, 110, &[ivec2, pixel, xy]);
                if loads[&inst[3]] {
                    let fetched = id();
                    let red = id();
                    emit(&mut result, 95, &[vec4, fetched, inst[3], pixel, 2, lod0]);
                    emit(&mut result, 81, &[float, red, fetched, 0]);
                    emit(&mut result, 80, &[vec4, inst[2], red, zero, zero, one]);
                } else {
                    emit(&mut result, 95, &[vec4, inst[2], inst[3], pixel, 2, lod0]);
                }
            }
            43 if inst.len() >= 4 && inst[3..].iter().all(|&word| word == 0) => {
                zeros.insert(inst[2]);
                result.extend_from_slice(inst);
            }
            44 if inst.len() >= 4 && inst[3..].iter().all(|word| zeros.contains(word)) => {
                zeros.insert(inst[2]);
                result.extend_from_slice(inst);
            }
            46 if inst.len() == 3 => {
                zeros.insert(inst[2]);
                result.extend_from_slice(inst);
            }
            _ => result.extend_from_slice(inst),
        }
    }
    result[3] = next;
    Ok(result)
}

#[cfg(test)]
#[path = "../tests/fixtures/input_attachment.rs"]
mod fixture;
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn color_and_depth_inputs_with_shared_spirv_type_lower_to_distinct_gpu_handles() {
        let source = fixture::input_fragment(3);
        let descriptors = bindings(&source).unwrap();
        assert_eq!(
            descriptors
                .iter()
                .map(|b| (b.group, b.binding, b.index))
                .collect::<Vec<_>>(),
            [(0, 0, 0), (0, 1, 1), (0, 2, 2)]
        );
        let words = lower(source, &[Some(false), Some(false), Some(true)]).unwrap();
        let module = naga::front::spv::Frontend::new(words.into_iter(), &Default::default())
            .parse()
            .unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
        let depths = module
            .global_variables
            .iter()
            .filter(|(_, g)| {
                matches!(
                    module.types[g.ty].inner,
                    naga::TypeInner::Image {
                        class: naga::ImageClass::Depth { .. },
                        ..
                    }
                )
            })
            .count();
        assert_eq!(depths, 1);
        assert!(
            module.entry_points[0]
                .function
                .arguments
                .iter()
                .any(|a| matches!(
                    a.binding,
                    Some(naga::Binding::BuiltIn(naga::BuiltIn::Position { .. }))
                ))
        );
    }
    #[test]
    fn input_indices_must_exist_and_multisample_inputs_are_rejected() {
        assert!(lower(fixture::input_fragment(3), &[Some(false), None, Some(true)]).is_err());
        let mut words = fixture::input_fragment(1);
        let mut offset = 5;
        while offset < words.len() {
            let count = (words[offset] >> 16) as usize;
            if words[offset] & 0xffff == 25 {
                words[offset + 6] = 1;
            }
            offset += count;
        }
        assert!(lower(words, &[Some(false)]).is_err());
    }
}
