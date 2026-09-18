//! Resolve scalar SPIR-V specialization expressions before Naga's SPIR-V frontend.
//! Unknown operations are rejected rather than silently using their default value.
use crate::spirv::{emit, instructions};
use ash::vk;
use std::collections::HashMap;
const UNSUPPORTED: vk::Result = vk::Result::ERROR_FEATURE_NOT_PRESENT;

pub(crate) fn freeze(
    words: Vec<u32>,
    values: &crate::resources::Specialization,
) -> Result<Vec<u32>, vk::Result> {
    let code = instructions(&words)?;
    let ids: HashMap<_, _> = code
        .iter()
        .filter(|i| i[0] & 65535 == 71 && i.len() == 4 && i[2] == 1)
        .map(|i| (i[1], i[3]))
        .collect();
    let types: HashMap<_, _> = code
        .iter()
        .filter(|i| matches!(i[0] & 65535, 20..=22))
        .map(|i| (i.get(1).copied().unwrap_or(0), *i))
        .collect();
    let mut constants = HashMap::<u32, u32>::new();
    let mut result = words[..5].to_vec();
    for i in code {
        let op = i[0] & 65535;
        if op == 71 && i.len() == 4 && i[2] == 1 {
            continue;
        }
        if matches!(op, 41 | 42) && i.len() == 3 {
            constants.insert(i[2], u32::from(op == 41));
        }
        if op == 43 && i.len() == 4 {
            constants.insert(i[2], i[3]);
        }
        if !matches!(op, 48..=52) {
            result.extend_from_slice(i);
            continue;
        }
        if i.len() < 3 {
            return Err(UNSUPPORTED);
        }
        if op == 51 {
            emit(&mut result, 44, &i[1..]);
            continue;
        }
        let ty = types.get(&i[1]).ok_or(UNSUPPORTED)?;
        let boolean = ty[0] & 65535 == 20;
        if !boolean && (ty.len() < 3 || ty[2] != 32) {
            return Err(UNSUPPORTED);
        }
        let supplied = ids
            .get(&i[2])
            .and_then(|id| values.iter().find(|(key, _)| key == id))
            .map(|(_, data)| {
                let bytes: [u8; 4] = data
                    .as_slice()
                    .try_into()
                    .map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?;
                Ok::<_, vk::Result>(u32::from_ne_bytes(bytes))
            })
            .transpose()?;
        let value = if let Some(value) = supplied {
            value
        } else {
            match op {
                48 => 1,
                49 => 0,
                50 if i.len() == 4 => i[3],
                52 if i.len() >= 5 => {
                    let args = i[4..]
                        .iter()
                        .map(|id| constants.get(id).copied().ok_or(UNSUPPORTED))
                        .collect::<Result<Vec<_>, _>>()?;
                    fold(i[3], &args).ok_or(UNSUPPORTED)?
                }
                _ => return Err(UNSUPPORTED),
            }
        };
        constants.insert(i[2], value);
        if boolean {
            emit(&mut result, if value != 0 { 41 } else { 42 }, &i[1..3]);
        } else {
            emit(&mut result, 43, &[i[1], i[2], value]);
        }
    }
    Ok(result)
}

fn fold(op: u32, args: &[u32]) -> Option<u32> {
    let truth = |v: bool| u32::from(v);
    Some(match *args {
        [a] => match op {
            113 | 114 | 124 => a,    // 32-bit conversions and bitcast
            126 => a.wrapping_neg(), // SNegate
            168 => truth(a == 0),    // LogicalNot
            200 => !a,               // Not
            _ => return None,
        },
        [a, b] => match op {
            128 => a.wrapping_add(b),
            130 => a.wrapping_sub(b),
            132 => a.wrapping_mul(b),
            134 => a.checked_div(b)?,
            135 => (a as i32).checked_div(b as i32)? as u32,
            137 => a.checked_rem(b)?,
            138 => (a as i32).checked_rem(b as i32)? as u32,
            139 => {
                let r = (a as i32).checked_rem(b as i32)?;
                if r != 0 && (r < 0) != ((b as i32) < 0) {
                    r.wrapping_add(b as i32) as u32
                } else {
                    r as u32
                }
            }
            164 => truth((a != 0) == (b != 0)),
            165 => truth((a != 0) != (b != 0)),
            166 => truth(a != 0 || b != 0),
            167 => truth(a != 0 && b != 0),
            170 => truth(a == b),
            171 => truth(a != b),
            172 => truth(a > b),
            173 => truth((a as i32) > (b as i32)),
            174 => truth(a >= b),
            175 => truth((a as i32) >= (b as i32)),
            176 => truth(a < b),
            177 => truth((a as i32) < (b as i32)),
            178 => truth(a <= b),
            179 => truth((a as i32) <= (b as i32)),
            194 => a.checked_shr(b)?,
            195 => (a as i32).checked_shr(b)? as u32,
            196 => a.checked_shl(b)?,
            197 => a | b,
            198 => a ^ b,
            199 => a & b,
            _ => return None,
        },
        [condition, a, b] if op == 169 => {
            if condition != 0 {
                a
            } else {
                b
            }
        } // Select
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn specialization_expressions_use_the_supplied_values_not_defaults() {
        let mut words = vec![0x07230203, 0x00010000, 0, 9, 0];
        for (op, args) in [
            (71, vec![3, 1, 5003]),
            (20, vec![1]),
            (21, vec![2, 32, 0]),
            (50, vec![2, 3, 0]),
            (43, vec![2, 4, 7]),
            (52, vec![2, 5, 128, 3, 4]),
            (52, vec![1, 6, 170, 5, 4]),
        ] {
            emit(&mut words, op, &args);
        }
        let result = freeze(words, &vec![(5003, 5u32.to_ne_bytes().to_vec())]).unwrap();
        let code = instructions(&result).unwrap();
        assert!(
            code.iter()
                .any(|i| i[0] & 65535 == 43 && i[1..] == [2, 5, 12])
        );
        assert!(code.iter().any(|i| i[0] & 65535 == 42 && i[1..] == [1, 6]));
        assert!(!code.iter().any(|i| matches!(i[0] & 65535, 48..=52)));
    }
    #[test]
    fn scalar_folding_preserves_signed_math_and_rejects_undefined_operations() {
        assert_eq!(fold(135, &[(-7i32) as u32, 3]), Some((-2i32) as u32));
        assert_eq!(fold(139, &[(-7i32) as u32, 3]), Some(2));
        assert_eq!(fold(139, &[7, (-3i32) as u32]), Some((-2i32) as u32));
        assert_eq!(fold(134, &[7, 0]), None);
        assert_eq!(fold(196, &[1, 32]), None);
        assert_eq!(fold(9999, &[1]), None);
    }
}
