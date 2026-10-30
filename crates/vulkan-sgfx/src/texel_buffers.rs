//! Scalar R32 texel buffers lowered to read-only storage words without copying.
use ash::vk;
use std::collections::{HashMap, HashSet};
const UNSUPPORTED: vk::Result = vk::Result::ERROR_FEATURE_NOT_PRESENT;
fn emit(out: &mut Vec<u32>, op: u32, args: &[u32]) {
    out.push(((args.len() as u32 + 1) << 16) | op);
    out.extend_from_slice(args);
}
pub(crate) fn lower(words: Vec<u32>) -> Result<Vec<u32>, vk::Result> {
    let rows = crate::spirv::instructions(&words)?;
    let images = rows
        .iter()
        .filter(|r| r[0] & 65535 == 25 && r.len() == 9 && r[3] == 5)
        .map(|r| (r[1], r[2]))
        .collect::<HashMap<_, _>>();
    if images.is_empty() {
        return Ok(words);
    }
    let mut wrappers = images.clone();
    for r in &rows {
        if r[0] & 65535 == 27 && r.len() == 3 && images.contains_key(&r[2]) {
            wrappers.insert(r[1], images[&r[2]]);
        }
    }
    let pointers = rows
        .iter()
        .filter(|r| r[0] & 65535 == 32 && r.len() == 4 && r[2] == 0 && wrappers.contains_key(&r[3]))
        .map(|r| (r[1], wrappers[&r[3]]))
        .collect::<HashMap<_, _>>();
    let variables = rows
        .iter()
        .filter(|r| r[0] & 65535 == 59 && r.len() == 4 && pointers.contains_key(&r[1]))
        .map(|r| (r[2], pointers[&r[1]]))
        .collect::<HashMap<_, _>>();
    if variables.is_empty() {
        return Err(UNSUPPORTED);
    }
    let mut aliases = HashMap::new();
    for r in &rows {
        match r[0] & 65535 {
            61 if r.len() >= 4 && variables.contains_key(&r[3]) => {
                aliases.insert(r[2], r[3]);
            }
            100 | 83 if r.len() == 4 && aliases.contains_key(&r[3]) => {
                aliases.insert(r[2], aliases[&r[3]]);
            }
            _ => {}
        }
    }
    let removed = wrappers
        .keys()
        .chain(pointers.keys())
        .chain(aliases.keys())
        .copied()
        .collect::<HashSet<_>>();
    let mut bound = words[3];
    let mut fresh = || {
        let id = bound;
        bound = bound.checked_add(1).ok_or(UNSUPPORTED)?;
        Ok::<_, vk::Result>(id)
    };
    let mut annotations = Vec::new();
    let mut declarations = Vec::new();
    let mut scalars = HashMap::new();
    // A legacy Uniform + BufferBlock is valid in SPIR-V 1.0 as well as 1.3;
    // Naga recognizes it as a storage buffer. External word stride stays four.
    for &scalar in variables.values() {
        if scalars.contains_key(&scalar) {
            continue;
        }
        let definition = rows
            .iter()
            .find(|r| r.len() >= 2 && r[1] == scalar && matches!(r[0] & 65535, 21 | 22))
            .ok_or(UNSUPPORTED)?;
        if definition[2] != 32 {
            return Err(UNSUPPORTED);
        }
        declarations.extend_from_slice(definition);
        let array = fresh()?;
        let structure = fresh()?;
        let pointer = fresh()?;
        let element = fresh()?;
        let zero = fresh()?;
        let one = fresh()?;
        emit(&mut annotations, 71, &[array, 6, 4]);
        emit(&mut annotations, 71, &[structure, 3]);
        emit(&mut annotations, 72, &[structure, 0, 35, 0]);
        emit(&mut annotations, 72, &[structure, 0, 24]);
        emit(&mut declarations, 29, &[array, scalar]);
        emit(&mut declarations, 30, &[structure, array]);
        emit(&mut declarations, 32, &[pointer, 2, structure]);
        emit(&mut declarations, 32, &[element, 2, scalar]);
        emit(&mut declarations, 43, &[scalar, zero, 0]);
        emit(
            &mut declarations,
            43,
            &[
                scalar,
                one,
                if definition[0] & 65535 == 22 {
                    1f32.to_bits()
                } else {
                    1
                },
            ],
        );
        scalars.insert(scalar, (pointer, element, zero, one));
    }
    let uint = fresh()?;
    let index_zero = fresh()?;
    emit(&mut declarations, 21, &[uint, 32, 0]);
    emit(&mut declarations, 43, &[uint, index_zero, 0]);
    let mut out = words[..5].to_vec();
    let mut added_annotations = false;
    let mut added_types = false;
    for r in rows {
        let op = r[0] & 65535;
        if !added_annotations && ((19..=39).contains(&op) || op == 71 || op == 72) {
            out.extend_from_slice(&annotations);
            added_annotations = true;
        }
        if !added_types && (19..=39).contains(&op) {
            out.extend_from_slice(&declarations);
            added_types = true;
        }
        if (op == 17 && r.len() == 2 && matches!(r[1], 46 | 47))
            || (matches!(op, 5 | 71) && r.len() >= 2 && removed.contains(&r[1]))
            || ((19..=39).contains(&op) && r.len() >= 2 && removed.contains(&r[1]))
            || (matches!(op, 21 | 22) && r.len() >= 2 && scalars.contains_key(&r[1]))
            || (matches!(op, 61 | 100 | 83) && r.len() >= 3 && aliases.contains_key(&r[2]))
        {
            continue;
        }
        if op == 59 && r.len() == 4 && variables.contains_key(&r[2]) {
            emit(&mut out, 59, &[scalars[&variables[&r[2]]].0, r[2], 2]);
        } else if op == 95 && r.len() >= 5 && aliases.contains_key(&r[3]) {
            // SPIR-V 1.4+ may request sign/zero extension explicitly. R32
            // storage already has the full scalar width, so these are no-ops.
            if r.len() != 5 && !(r.len() == 6 && matches!(r[5], 0 | 0x1000 | 0x2000)) {
                return Err(UNSUPPORTED);
            }
            let variable = aliases[&r[3]];
            let scalar = variables[&variable];
            let (_, element, zero, one) = scalars[&scalar];
            let pointer = fresh()?;
            let value = fresh()?;
            emit(
                &mut out,
                65,
                &[element, pointer, variable, index_zero, r[4]],
            );
            emit(&mut out, 61, &[scalar, value, pointer]);
            emit(&mut out, 80, &[r[1], r[2], value, zero, zero, one]);
        } else {
            out.extend_from_slice(r);
        }
    }
    out[3] = bound;
    Ok(out)
}

/// Specialize scalar R32 storage loads for signed byte texel views. The same
/// GPU buffer remains bound: address packed words and sign-extend the selected
/// byte in the shader, including high-bit values and nonzero view offsets.
pub(crate) fn narrow_signed_bytes(
    words: Vec<u32>,
    bindings: &[(u32, u32)],
) -> Result<Vec<u32>, vk::Result> {
    if bindings.is_empty() {
        return Ok(words);
    }
    let rows = crate::spirv::instructions(&words)?;
    let mut decorations = HashMap::<u32, (Option<u32>, Option<u32>)>::new();
    for r in &rows {
        if r[0] & 65535 == 71 && r.len() == 4 {
            let pair = decorations.entry(r[1]).or_default();
            match r[2] {
                34 => pair.0 = Some(r[3]),
                33 => pair.1 = Some(r[3]),
                _ => {}
            }
        }
    }
    let globals = decorations
        .into_iter()
        .filter_map(|(id, (set, binding))| bindings.contains(&(set?, binding?)).then_some(id))
        .collect::<HashSet<_>>();
    if globals.is_empty() {
        return Ok(words);
    }
    let signed = rows
        .iter()
        .filter(|r| r[0] & 65535 == 21 && r.len() == 4 && r[2] == 32 && r[3] == 1)
        .map(|r| r[1])
        .collect::<HashSet<_>>();
    let pointers = rows
        .iter()
        .filter(|r| r[0] & 65535 == 32 && r.len() == 4)
        .map(|r| (r[1], r[3]))
        .collect::<HashMap<_, _>>();
    let mut bound = words[3];
    let mut fresh = || {
        let id = bound;
        bound = bound.checked_add(1).ok_or(UNSUPPORTED)?;
        Ok::<_, vk::Result>(id)
    };
    let uint = fresh()?;
    let two = fresh()?;
    let three = fresh()?;
    let twenty_four = fresh()?;
    let mut declarations = Vec::new();
    emit(&mut declarations, 21, &[uint, 32, 0]);
    for (id, value) in [(two, 2), (three, 3), (twenty_four, 24)] {
        emit(&mut declarations, 43, &[uint, id, value]);
    }
    let mut out = words[..5].to_vec();
    let mut added = false;
    let mut accesses = HashMap::new();
    for r in rows {
        let op = r[0] & 65535;
        if op == 54 && !added {
            out.extend_from_slice(&declarations);
            added = true;
        }
        if op == 65 && r.len() >= 4 && globals.contains(&r[3]) {
            if r.len() != 6 || !pointers.get(&r[1]).is_some_and(|ty| signed.contains(ty)) {
                return Err(UNSUPPORTED);
            }
            let coord = fresh()?;
            let word = fresh()?;
            emit(&mut out, 124, &[uint, coord, r[5]]);
            emit(&mut out, 194, &[uint, word, coord, two]);
            let mut access = r.to_vec();
            access[5] = word;
            out.extend_from_slice(&access);
            accesses.insert(r[2], coord);
        } else if op == 61 && r.len() == 4 && accesses.contains_key(&r[3]) {
            if !signed.contains(&r[1]) {
                return Err(UNSUPPORTED);
            }
            let raw = fresh()?;
            let byte = fresh()?;
            let shift = fresh()?;
            let selected = fresh()?;
            let high = fresh()?;
            emit(&mut out, 61, &[r[1], raw, r[3]]);
            emit(&mut out, 199, &[uint, byte, accesses[&r[3]], three]);
            emit(&mut out, 196, &[uint, shift, byte, three]);
            emit(&mut out, 194, &[r[1], selected, raw, shift]);
            emit(&mut out, 196, &[r[1], high, selected, twenty_four]);
            emit(&mut out, 195, &[r[1], r[2], high, twenty_four]);
        } else {
            out.extend_from_slice(r);
        }
    }
    out[3] = bound;
    Ok(out)
}
