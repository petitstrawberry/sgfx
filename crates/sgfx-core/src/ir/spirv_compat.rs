//! SPIR-V interoperability adapters shared by shader consumers.
use super::{Error, Result};
use alloc::vec::Vec;

/// Naga 24 implements the draw-parameter builtins but omits their capability
/// from its strict parser allowlist. Adapt just that declaration (and its
/// pre-1.3 extension spelling); all actual operations still pass its validator
/// and this compiler's explicit builtin lowering.
pub fn naga24_spirv(words: &[u32]) -> Result<Vec<u32>> {
    if words.len() < 5 || words[0] != 0x07230203 || words[3] == 0 {
        return Err(Error::InvalidDescriptor);
    }
    let mut instructions = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        let end = at
            .checked_add(count)
            .filter(|&end| count != 0 && end <= words.len())
            .ok_or(Error::InvalidDescriptor)?;
        instructions.push(&words[at..end]);
        at = end;
    }
    // DrawIndex is signed in GLSL. Unlike the other integer builtins Naga 24
    // does not insert its required unsigned entry-point adapter for DrawID.
    // Re-type only those scalar input variables and bitcast their loads back
    // to the caller's original signed type, preserving every bit and use.
    let draws = instructions
        .iter()
        .filter(|i| i[0] & 65535 == 71 && i.len() == 4 && i[2] == 11 && i[3] == 4426)
        .map(|i| i[1])
        .collect::<Vec<_>>();
    let signed = instructions
        .iter()
        .filter(|i| i[0] & 65535 == 21 && i.len() == 4 && i[2] == 32 && i[3] == 1)
        .map(|i| i[1])
        .collect::<Vec<_>>();
    let signed_pointers = instructions
        .iter()
        .filter(|i| i[0] & 65535 == 32 && i.len() == 4 && i[2] == 1 && signed.contains(&i[3]))
        .map(|i| i[1])
        .collect::<Vec<_>>();
    let variables = instructions
        .iter()
        .filter(|i| {
            i[0] & 65535 == 59
                && i.len() >= 4
                && i[3] == 1
                && draws.contains(&i[2])
                && signed_pointers.contains(&i[1])
        })
        .map(|i| i[2])
        .collect::<Vec<_>>();
    let mut bound = words[3];
    let mut fresh = || {
        let id = bound;
        bound = bound.checked_add(1).ok_or(Error::Overflow)?;
        Ok::<_, Error>(id)
    };
    let mut types = Vec::new();
    let mut pointer = 0;
    let mut uint = 0;
    if !variables.is_empty() {
        uint = if let Some(i) = instructions
            .iter()
            .find(|i| i[0] & 65535 == 21 && i.len() == 4 && i[2] == 32 && i[3] == 0)
        {
            types.extend_from_slice(i);
            i[1]
        } else {
            let id = fresh()?;
            types.extend_from_slice(&[(4u32 << 16) | 21, id, 32, 0]);
            id
        };
        pointer = if let Some(i) = instructions
            .iter()
            .find(|i| i[0] & 65535 == 32 && i.len() == 4 && i[2] == 1 && i[3] == uint)
        {
            types.extend_from_slice(i);
            i[1]
        } else {
            let id = fresh()?;
            types.extend_from_slice(&[(4u32 << 16) | 32, id, 1, uint]);
            id
        };
    }
    let mut result = words[..5].to_vec();
    for inst in instructions {
        let op = inst[0] & 65535;
        let draw_cap = op == 17 && inst.get(1) == Some(&4427);
        let draw_extension = op == 10
            && inst[1..]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .take_while(|&v| v != 0)
                .eq(b"SPV_KHR_shader_draw_parameters".iter().copied());
        // Debug line records may legally appear amongst type declarations.
        // Naga 24 only accepts them inside functions; they have no execution
        // semantics, so discard both line-state instructions everywhere.
        if draw_cap || draw_extension || op == 8 || op == 317 {
            continue;
        }
        if !types.is_empty() && (19..=39).contains(&op) {
            result.extend_from_slice(&types);
            types.clear();
        }
        if !variables.is_empty()
            && ((op == 21 && inst.get(1) == Some(&uint))
                || (op == 32 && inst.get(1) == Some(&pointer)))
        {
            continue;
        }
        if op == 59 && inst.len() < 4 {
            return Err(Error::InvalidDescriptor);
        }
        if op == 59 && variables.contains(&inst[2]) {
            result.extend_from_slice(inst);
            let start = result.len() - inst.len();
            result[start + 1] = pointer;
        } else if op == 61 && inst.len() >= 4 && variables.contains(&inst[3]) {
            let id = fresh()?;
            result.extend_from_slice(inst);
            let start = result.len() - inst.len();
            result[start + 1] = uint;
            result[start + 2] = id;
            result.extend_from_slice(&[(4u32 << 16) | 124, inst[1], inst[2], id]);
        } else {
            result.extend_from_slice(inst);
        }
    }
    result[3] = bound;
    Ok(result)
}
