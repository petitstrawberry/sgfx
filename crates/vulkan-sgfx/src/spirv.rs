//! Structural lowering of Vulkan combined image samplers for Naga's separate handles.
use ash::vk;
use std::collections::HashMap;

const TYPE_SAMPLER: u32 = 26;
const TYPE_SAMPLED_IMAGE: u32 = 27;
const TYPE_POINTER: u32 = 32;
const VARIABLE: u32 = 59;
const LOAD: u32 = 61;
const DECORATE: u32 = 71;
const SAMPLED_IMAGE: u32 = 86;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn combined_sampler_becomes_a_valid_separate_texture_sampler_pair() {
        let mut words = vec![0x07230203, 0x00010000, 0, 18, 0];
        for (opcode, operands) in [
            (17, vec![1]),
            (14, vec![0, 1]),
            (15, vec![4, 14, u32::from_le_bytes(*b"main"), 0, 11]),
            (16, vec![14, 7]),
            (71, vec![9, 33, 2]),
            (71, vec![9, 34, 1]),
            (71, vec![11, 30, 0]),
            (19, vec![1]),
            (33, vec![2, 1]),
            (22, vec![3, 32]),
            (23, vec![4, 3, 2]),
            (23, vec![5, 3, 4]),
            (25, vec![6, 3, 1, 0, 0, 0, 1, 0]),
            (27, vec![7, 6]),
            (32, vec![8, 0, 7]),
            (32, vec![10, 3, 5]),
            (43, vec![3, 12, 0.25f32.to_bits()]),
            (44, vec![4, 13, 12, 12]),
            (59, vec![8, 9, 0]),
            (59, vec![10, 11, 3]),
            (54, vec![1, 14, 0, 2]),
            (248, vec![15]),
            (61, vec![7, 16, 9]),
            (87, vec![5, 17, 16, 13]),
            (62, vec![11, 17]),
            (253, vec![]),
            (56, vec![]),
        ] {
            emit(&mut words, opcode, &operands);
        }
        let normalized = crate::resources::normalize_spirv(words).unwrap();
        let sgfx::ir::ShaderSource::SpirV(transformed) = normalized.source() else {
            panic!("expected SPIR-V");
        };
        let module = naga::front::spv::Frontend::new(
            transformed.iter().copied(),
            &naga::front::spv::Options::default(),
        )
        .parse()
        .unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
        let bound = module
            .global_variables
            .iter()
            .filter_map(|(_, global)| global.binding.as_ref())
            .collect::<Vec<_>>();
        assert_eq!(bound.len(), 2);
        assert!(bound.iter().all(|binding| binding.group == 1));
        assert_eq!(
            bound
                .iter()
                .map(|binding| binding.binding)
                .collect::<std::collections::BTreeSet<_>>(),
            [4, 5].into()
        );
        assert!(module.functions.iter().any(|(_, function)| {
            function
                .expressions
                .iter()
                .any(|(_, expr)| matches!(expr, naga::Expression::ImageSample { .. }))
        }));
    }
    #[test]
    fn malformed_instruction_lengths_are_rejected() {
        for tail in [vec![0], vec![(4 << 16) | LOAD, 1]] {
            let mut words = vec![0x07230203, 0x00010000, 0, 1, 0];
            words.extend(tail);
            assert!(separate_combined_samplers(words).is_err());
        }
    }
}

fn instructions(words: &[u32]) -> Result<Vec<&[u32]>, vk::Result> {
    if words.len() < 5 {
        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
    }
    let mut result = Vec::new();
    let mut offset = 5;
    while offset < words.len() {
        let count = (words[offset] >> 16) as usize;
        if count == 0
            || offset
                .checked_add(count)
                .is_none_or(|end| end > words.len())
        {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        result.push(&words[offset..offset + count]);
        offset += count;
    }
    Ok(result)
}
fn emit(output: &mut Vec<u32>, opcode: u32, operands: &[u32]) {
    output.push(((operands.len() as u32 + 1) << 16) | opcode);
    output.extend_from_slice(operands);
}

/// Split directly bound OpTypeSampledImage globals while retaining their Vulkan bindings.
/// Arrays and pointer aliases are left unsupported by the frontend. No shader identity,
/// application name or replacement shader participates in this transform.
pub(crate) fn separate_combined_samplers(words: Vec<u32>) -> Result<Vec<u32>, vk::Result> {
    let ops = instructions(&words)?;
    let mut combined = HashMap::new();
    let mut pointers = HashMap::new();
    for op in &ops {
        match op[0] & 0xffff {
            TYPE_SAMPLED_IMAGE if op.len() == 3 => {
                combined.insert(op[1], op[2]);
            }
            TYPE_POINTER if op.len() == 4 && op[2] == 0 => {
                pointers.insert(op[1], op[3]);
            }
            _ => {}
        }
    }
    let mut variables = HashMap::new();
    let mut next = words[3];
    let mut id = || {
        let value = next;
        next = next
            .checked_add(1)
            .ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)?;
        Ok::<_, vk::Result>(value)
    };
    let sampler_type = id()?;
    let sampler_pointer = id()?;
    for op in &ops {
        if op[0] & 0xffff == VARIABLE
            && op.len() >= 4
            && op[3] == 0
            && let Some(image) = pointers.get(&op[1]).and_then(|ty| combined.get(ty))
        {
            if op.len() != 4 {
                return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
            }
            variables.insert(op[2], (*image, id()?));
        }
    }
    if variables.is_empty() {
        return Ok(words);
    }
    let mut output = words[..5].to_vec();
    let mut declared = false;
    for op in ops {
        let opcode = op[0] & 0xffff;
        if !declared && opcode == TYPE_SAMPLED_IMAGE {
            emit(&mut output, TYPE_SAMPLER, &[sampler_type]);
            emit(
                &mut output,
                TYPE_POINTER,
                &[sampler_pointer, 0, sampler_type],
            );
            declared = true;
        }
        match opcode {
            TYPE_POINTER if op.len() == 4 && op[2] == 0 && combined.contains_key(&op[3]) => {
                emit(&mut output, TYPE_POINTER, &[op[1], 0, combined[&op[3]]]);
            }
            VARIABLE if op.len() == 4 && variables.contains_key(&op[2]) => {
                output.extend_from_slice(op);
                emit(
                    &mut output,
                    VARIABLE,
                    &[sampler_pointer, variables[&op[2]].1, 0],
                );
            }
            DECORATE
                if op.len() == 4 && matches!(op[2], 33 | 34) && variables.contains_key(&op[1]) =>
            {
                output.extend_from_slice(op);
                emit(&mut output, DECORATE, &[variables[&op[1]].1, op[2], op[3]]);
            }
            LOAD if op.len() >= 4 && combined.contains_key(&op[1]) => {
                let &(image_type, sampler) = variables
                    .get(&op[3])
                    .ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)?;
                let image_value = id()?;
                let sampler_value = id()?;
                let mut load = op[1..].to_vec();
                load[0] = image_type;
                load[1] = image_value;
                emit(&mut output, LOAD, &load);
                emit(&mut output, LOAD, &[sampler_type, sampler_value, sampler]);
                emit(
                    &mut output,
                    SAMPLED_IMAGE,
                    &[op[1], op[2], image_value, sampler_value],
                );
            }
            15 => {
                // Entry-point interface variables follow its NUL-terminated name.
                let mut entry = op[1..].to_vec();
                let end = op
                    .iter()
                    .enumerate()
                    .skip(3)
                    .find(|(_, word)| word.to_le_bytes().contains(&0))
                    .map(|(index, _)| index + 1)
                    .ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)?;
                for variable in &op[end..] {
                    if let Some((_, sampler)) = variables.get(variable) {
                        entry.push(*sampler);
                    }
                }
                emit(&mut output, 15, &entry);
            }
            _ => output.extend_from_slice(op),
        }
    }
    output[3] = next;
    Ok(output)
}
