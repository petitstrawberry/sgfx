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
    fn sampler_function(depth: bool) -> Vec<u32> {
        let mut words = vec![0x07230203, 0x00010000, 0, 26, 0];
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
            (25, vec![6, 3, 1, u32::from(depth), 0, 0, 1, 0]),
            (27, vec![7, 6]),
            (32, vec![8, 0, 7]),
            (32, vec![10, 3, 5]),
            (33, vec![19, 5, 8]),
            (43, vec![3, 12, 0.25f32.to_bits()]),
            (44, vec![4, 13, 12, 12]),
            (59, vec![8, 9, 0]),
            (59, vec![10, 11, 3]),
            (54, vec![1, 14, 0, 2]),
            (248, vec![15]),
            (57, vec![5, 17, 20, 9]),
            (62, vec![11, 17]),
            (253, vec![]),
            (56, vec![]),
            (54, vec![5, 20, 0, 19]),
            (55, vec![8, 21]),
            (248, vec![22]),
            (61, vec![7, 23, 21]),
        ] {
            emit(&mut words, opcode, &operands);
        }
        if depth {
            emit(&mut words, 89, &[3, 25, 23, 13, 12]);
            emit(&mut words, 80, &[5, 24, 25, 25, 25, 25]);
        } else {
            emit(&mut words, 87, &[5, 24, 23, 13]);
        }
        emit(&mut words, 254, &[24]);
        emit(&mut words, 56, &[]);
        words
    }
    #[test]
    fn combined_sampler_helpers_preserve_color_and_comparison_sampling() {
        for depth in [false, true] {
            let normalized = crate::resources::normalize_spirv(sampler_function(depth)).unwrap();
            let sgfx::ir::ShaderSource::SpirV(words) = normalized.source() else {
                panic!("SPIR-V expected")
            };
            let module =
                naga::front::spv::Frontend::new(words.iter().copied(), &Default::default())
                    .parse()
                    .unwrap();
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .unwrap();
            assert!(
                module
                    .functions
                    .iter()
                    .all(|(_, f)| f.arguments.iter().all(|a| !matches!(
                        module.types[a.ty].inner,
                        naga::TypeInner::Image { .. } | naga::TypeInner::Sampler { .. }
                    )))
            );
            let samplers = module
                .global_variables
                .iter()
                .filter_map(|(_, global)| match module.types[global.ty].inner {
                    naga::TypeInner::Sampler { comparison } => Some(comparison),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(samplers, [depth]);
            assert!(module.functions.iter().any(|(_,function)|function.expressions.iter().any(|(_,expr)|matches!(expr,naga::Expression::ImageSample {depth_ref,..} if depth_ref.is_some()==depth))));
        }
    }
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

pub(crate) fn instructions(words: &[u32]) -> Result<Vec<&[u32]>, vk::Result> {
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
pub(crate) fn emit(output: &mut Vec<u32>, opcode: u32, operands: &[u32]) {
    output.push(((operands.len() as u32 + 1) << 16) | opcode);
    output.extend_from_slice(operands);
}

/// Split OpTypeSampledImage globals and UniformConstant function parameters.
/// Arrays and other pointer aliases are left unsupported by the frontend. No shader identity,
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
        if op[0] & 0xffff == 55
            && op.len() == 3
            && let Some(image) = pointers.get(&op[1]).and_then(|ty| combined.get(ty))
        {
            variables.insert(op[2], (*image, id()?));
        }
    }
    if variables.is_empty() {
        return Ok(words);
    }
    let mut output = words[..5].to_vec();
    let signatures: HashMap<_, _> = ops
        .iter()
        .filter(|op| op[0] & 0xffff == 33 && op.len() >= 3)
        .map(|op| (op[1], op[3..].to_vec()))
        .collect();
    let functions: HashMap<_, _> = ops
        .iter()
        .filter(|op| op[0] & 0xffff == 54 && op.len() == 5)
        .map(|op| (op[2], op[4]))
        .collect();
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
            33 if op.len() >= 3 => {
                let mut types = op[1..3].to_vec();
                for ty in &op[3..] {
                    types.push(*ty);
                    if pointers
                        .get(ty)
                        .is_some_and(|base| combined.contains_key(base))
                    {
                        types.push(sampler_pointer);
                    }
                }
                emit(&mut output, opcode, &types);
            }
            55 if op.len() == 3 && variables.contains_key(&op[2]) => {
                output.extend_from_slice(op);
                emit(&mut output, opcode, &[sampler_pointer, variables[&op[2]].1]);
            }
            57 if op.len() >= 4 => {
                let parameters = functions
                    .get(&op[3])
                    .and_then(|ty| signatures.get(ty))
                    .ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)?;
                if parameters.len() != op.len() - 4 {
                    return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                }
                let mut call = op[1..4].to_vec();
                for (ty, argument) in parameters.iter().zip(&op[4..]) {
                    call.push(*argument);
                    if pointers
                        .get(ty)
                        .is_some_and(|base| combined.contains_key(base))
                    {
                        call.push(
                            variables
                                .get(argument)
                                .ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)?
                                .1,
                        );
                    }
                }
                emit(&mut output, opcode, &call);
            }
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

/// Vulkan descriptor layouts do not encode image dimensionality or comparison
/// sampling. Recover these types from the selected SPIR-V entry points.
pub(crate) fn specialize_layout(
    table: &sgfx::ir::ResourceTable,
    layout: &sgfx::ir::PipelineLayoutDesc,
    shaders: &[&sgfx::ir::ShaderEntryPoint],
) -> Result<sgfx::ir::PipelineLayoutDesc, vk::Result> {
    use sgfx::ir;
    let invalid = vk::Result::ERROR_INITIALIZATION_FAILED;
    let unsupported = vk::Result::ERROR_FEATURE_NOT_PRESENT;
    let mut bindings = std::collections::BTreeMap::new();
    for shader in shaders {
        let desc = table
            .shader_module(
                table
                    .shader_module_ref(shader.module())
                    .map_err(|_| invalid)?,
            )
            .map_err(|_| invalid)?;
        let ir::ShaderSource::SpirV(words) = desc.source() else {
            return Err(unsupported);
        };
        let module = naga::front::spv::Frontend::new(
            words.iter().copied(),
            &naga::front::spv::Options {
                adjust_coordinate_space: false,
                ..Default::default()
            },
        )
        .parse()
        .map_err(|_| unsupported)?;
        let stage = match shader.stage() {
            ir::ShaderStage::Vertex => naga::ShaderStage::Vertex,
            ir::ShaderStage::Fragment => naga::ShaderStage::Fragment,
            ir::ShaderStage::Compute => naga::ShaderStage::Compute,
        };
        let entry = module
            .entry_points
            .iter()
            .position(|entry| entry.name == shader.entry_point() && entry.stage == stage)
            .ok_or(invalid)?;
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::PUSH_CONSTANT,
        )
        .validate(&module)
        .map_err(|_| unsupported)?;
        for (handle, global) in module.global_variables.iter() {
            let Some(binding) = &global.binding else {
                continue;
            };
            if info.get_entry_point(entry)[handle].is_empty() {
                continue;
            }
            let original = layout
                .bind_groups()
                .get(binding.group as usize)
                .ok_or(invalid)?
                .entries()
                .iter()
                .find(|entry| entry.binding() == binding.binding)
                .ok_or(invalid)?;
            let ty = if let naga::AddressSpace::Storage { access } = global.space {
                ir::BindingType::StorageBuffer {
                    read_only: !access.contains(naga::StorageAccess::STORE),
                }
            } else {
                match module.types[global.ty].inner {
                    naga::TypeInner::Image {
                        dim,
                        arrayed,
                        class,
                    } => {
                        let depth = match class {
                            naga::ImageClass::Sampled {
                                kind: naga::ScalarKind::Float,
                                multi: false,
                            } => false,
                            naga::ImageClass::Depth { multi: false } => true,
                            naga::ImageClass::Storage {
                                format: naga::StorageFormat::Rgba8Unorm,
                                access,
                            } if dim == naga::ImageDimension::D2
                                && !arrayed
                                && access == naga::StorageAccess::STORE =>
                            {
                                let ty = ir::BindingType::StorageTexture {
                                    format: ir::TextureFormat::Rgba8Unorm,
                                    access: ir::StorageTextureAccess::WriteOnly,
                                };
                                if original.ty() != ty {
                                    return Err(invalid);
                                }
                                bindings.insert((binding.group, binding.binding), ty);
                                continue;
                            }
                            _ => return Err(unsupported),
                        };
                        let dimension = match (dim, arrayed) {
                            (naga::ImageDimension::D2, false) => ir::TextureViewDimension::D2,
                            (naga::ImageDimension::D2, true) => ir::TextureViewDimension::D2Array,
                            (naga::ImageDimension::Cube, false) => ir::TextureViewDimension::Cube,
                            _ => return Err(unsupported),
                        };
                        ir::BindingType::SampledTextureView { dimension, depth }
                    }
                    naga::TypeInner::Sampler { comparison: true } => {
                        ir::BindingType::ComparisonSampler
                    }
                    _ => original.ty(),
                }
            };
            if let Some(previous) = bindings.insert((binding.group, binding.binding), ty) {
                match (previous, ty) {
                    (
                        ir::BindingType::StorageBuffer { read_only: a },
                        ir::BindingType::StorageBuffer { read_only: b },
                    ) => {
                        bindings.insert(
                            (binding.group, binding.binding),
                            ir::BindingType::StorageBuffer { read_only: a && b },
                        );
                    }
                    _ if previous != ty => return Err(invalid),
                    _ => {}
                }
            }
        }
    }
    let groups = layout
        .bind_groups()
        .iter()
        .enumerate()
        .map(|(set, group)| {
            ir::BindGroupLayoutDesc::new(
                group
                    .entries()
                    .iter()
                    .filter_map(|entry| {
                        bindings.get(&(set as u32, entry.binding())).map(|ty| {
                            ir::BindGroupLayoutEntry::new(entry.binding(), entry.visibility(), *ty)
                        })
                    })
                    .collect(),
            )
            .map_err(|_| invalid)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if !layout
        .bind_groups()
        .iter()
        .zip(&groups)
        .all(|(declared, used)| crate::resources::specialized_layout_matches(declared, used))
    {
        return Err(invalid);
    }
    ir::PipelineLayoutDesc::new(groups)
        .and_then(|layout_out| {
            layout_out.with_push_constant_ranges(layout.push_constant_ranges().to_vec())
        })
        .map_err(|_| invalid)
}

/// Non-identity component maps by normalized SGFX group/binding. Components
/// 0..3 select RGBA; 4 and 5 select the constants zero and one.
pub(crate) type TextureSwizzles = Vec<(u32, u32, [u8; 4])>;

/// Lower view component mapping into sampling instructions. The result stays
/// on the GPU and preserves allocation aliases, mip selection and sRGB decode.
pub(crate) fn sampled_image_swizzles(
    words: &[u32],
    views: &TextureSwizzles,
) -> Result<Vec<u32>, vk::Result> {
    let unsupported = vk::Result::ERROR_FEATURE_NOT_PRESENT;
    let ops = instructions(words)?;
    let mut groups = HashMap::new();
    let mut bindings = HashMap::new();
    let mut vectors = HashMap::new();
    let mut floats = Vec::new();
    for op in &ops {
        match op[0] & 0xffff {
            DECORATE if op.len() == 4 && op[2] == 34 => {
                groups.insert(op[1], op[3]);
            }
            DECORATE if op.len() == 4 && op[2] == 33 => {
                bindings.insert(op[1], op[3]);
            }
            22 if op.len() == 3 && op[2] == 32 => floats.push(op[1]),
            23 if op.len() == 4 && op[3] == 4 => {
                vectors.insert(op[1], op[2]);
            }
            _ => {}
        }
    }
    let mut roots = HashMap::new();
    for (&id, &binding) in &bindings {
        if let Some(&group) = groups.get(&id) {
            if let Some((_, _, mapping)) =
                views.iter().find(|(g, b, _)| *g == group && *b == binding)
            {
                if mapping.iter().any(|v| *v > 5) {
                    return Err(unsupported);
                }
                roots.insert(id, *mapping);
            }
        }
    }
    let mut next = words[3];
    let mut id = || {
        let value = next;
        next = next
            .checked_add(1)
            .filter(|v| *v <= 0x3fffff)
            .ok_or(unsupported)?;
        Ok::<u32, vk::Result>(value)
    };
    let mut constants = Vec::new();
    let mut constant_vectors = HashMap::new();
    let mut lowered = Vec::new();
    for op in &ops {
        let opcode = op[0] & 0xffff;
        if matches!(opcode, LOAD | SAMPLED_IMAGE | 83 | 100) && op.len() >= 4 {
            if let Some(mapping) = roots.get(&op[3]).copied() {
                roots.insert(op[2], mapping);
            }
        }
        if (87..=98).contains(&opcode) && op.len() >= 4 {
            if let Some(mapping) = roots.get(&op[3]) {
                if !matches!(opcode, 87 | 88 | 91 | 92 | 95 | 98) {
                    return Err(unsupported);
                }
                let ty = op[1];
                let scalar = *vectors
                    .get(&ty)
                    .filter(|scalar| floats.contains(scalar))
                    .ok_or(unsupported)?;
                let constant = if let Some(value) = constant_vectors.get(&ty) {
                    *value
                } else {
                    let zero = id()?;
                    let one = id()?;
                    let value = id()?;
                    emit(&mut constants, 43, &[scalar, zero, 0]);
                    emit(&mut constants, 43, &[scalar, one, 1.0f32.to_bits()]);
                    emit(&mut constants, 44, &[ty, value, zero, one, zero, one]);
                    constant_vectors.insert(ty, value);
                    value
                };
                let raw = id()?;
                let mut sample = op.to_vec();
                sample[2] = raw;
                lowered.extend(sample);
                emit(
                    &mut lowered,
                    79,
                    &[
                        ty,
                        op[2],
                        raw,
                        constant,
                        u32::from(mapping[0]),
                        u32::from(mapping[1]),
                        u32::from(mapping[2]),
                        u32::from(mapping[3]),
                    ],
                );
                continue;
            }
        }
        lowered.extend_from_slice(op);
    }
    if constants.is_empty() {
        return Ok(words.to_vec());
    }
    let mut result = words[..5].to_vec();
    let mut inserted = false;
    for op in instructions(&[words[..5].to_vec(), lowered].concat())? {
        if !inserted && op[0] & 0xffff == 54 {
            result.extend_from_slice(&constants);
            inserted = true;
        }
        result.extend_from_slice(op);
    }
    result[3] = next;
    Ok(result)
}
