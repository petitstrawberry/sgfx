//! Word-based comparison and changed-range discovery without temporary buffers.

/// Avoid the byte-at-a-time generic native memcmp on large unchanged ranges.
/// Conversion from byte arrays permits unaligned buffer offsets without unsafe
/// loads; XOR/OR reduction also prevents lowering this loop back to memcmp.
pub fn equal_buffer_chunk(old: &[u8], new: &[u8]) -> bool {
    if old.len() != new.len() {
        return false;
    }
    let mut old_words = old.chunks_exact(8);
    let mut new_words = new.chunks_exact(8);
    let difference =
        old_words
            .by_ref()
            .zip(new_words.by_ref())
            .fold(0, |difference, (old, new)| {
                difference
                    | (u64::from_ne_bytes(old.try_into().unwrap())
                        ^ u64::from_ne_bytes(new.try_into().unwrap()))
            });
    difference == 0
        && old_words
            .remainder()
            .iter()
            .zip(new_words.remainder())
            .all(|(old, new)| old == new)
}

// Word comparisons preserve unaligned byte-buffer support and avoid the
// byte-at-a-time memcmp used by the native runtime.
fn common_buffer_prefix(old: &[u8], new: &[u8]) -> usize {
    let n = old.len().min(new.len());
    let words = n / 8;
    for i in 0..words {
        let at = i * 8;
        let delta = u64::from_le_bytes(old[at..at + 8].try_into().unwrap())
            ^ u64::from_le_bytes(new[at..at + 8].try_into().unwrap());
        if delta != 0 {
            return at + delta.trailing_zeros() as usize / 8;
        }
    }
    let at = words * 8;
    at + old[at..n]
        .iter()
        .zip(&new[at..n])
        .take_while(|(a, b)| a == b)
        .count()
}

fn common_buffer_suffix(old: &[u8], new: &[u8]) -> usize {
    let n = old.len().min(new.len());
    let mut matched = 0;
    while matched + 8 <= n {
        let a = old.len() - matched - 8;
        let b = new.len() - matched - 8;
        let delta = u64::from_le_bytes(old[a..a + 8].try_into().unwrap())
            ^ u64::from_le_bytes(new[b..b + 8].try_into().unwrap());
        if delta != 0 {
            return matched + delta.leading_zeros() as usize / 8;
        }
        matched += 8;
    }
    matched
        + old[..old.len() - matched]
            .iter()
            .rev()
            .zip(new[..new.len() - matched].iter().rev())
            .take_while(|(a, b)| a == b)
            .count()
}

/// Bound changed bytes without walking large unchanged regions byte by byte.
pub fn changed_buffer_bytes(
    previous: Option<&[u8]>,
    start: usize,
    data: &[u8],
) -> Result<core::ops::Range<usize>, crate::ir::Error> {
    let end = start
        .checked_add(data.len())
        .ok_or(crate::ir::Error::Overflow)?;
    let Some(previous) = previous else {
        return Ok(0..end);
    };
    let old = previous.get(start..).unwrap_or(&[]);
    let overlap = old.len().min(data.len());
    let mut prefix = 0;
    for (old, new) in old[..overlap]
        .chunks(4096)
        .zip(data[..overlap].chunks(4096))
    {
        if !equal_buffer_chunk(old, new) {
            prefix += common_buffer_prefix(old, new);
            break;
        }
        prefix += old.len();
    }
    let mut suffix = 0;
    if data.len() <= old.len() {
        for (old, new) in old[prefix..data.len()]
            .rchunks(4096)
            .zip(data[prefix..].rchunks(4096))
        {
            if !equal_buffer_chunk(old, new) {
                suffix += common_buffer_suffix(old, new);
                break;
            }
            suffix += old.len();
        }
    }
    Ok((start + prefix).min(previous.len())..end - suffix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};
    #[test]
    fn shadow_diff_finds_changes_across_comparison_chunk_boundaries() {
        let previous = vec![0; 1032];
        for position in [0, 255, 256, 257, 1023, 1031] {
            let mut data = previous.clone();
            data[position] = 1;
            assert_eq!(
                changed_buffer_bytes(Some(&previous), 0, &data).unwrap(),
                position..position + 1
            );
        }
        assert_eq!(
            changed_buffer_bytes(Some(&previous), 4, &[0, 1, 0, 0]).unwrap(),
            5..6
        );
    }

    #[test]
    fn shadow_diff_handles_unaligned_words_tails_and_growth() {
        let previous: Vec<u8> = (0..530).map(|n| (n % 251) as u8).collect();
        for start in 0..8 {
            for length in [0, 1, 7, 8, 9, 255, 256, 257, 521] {
                let original = &previous[start..start + length];
                assert!(equal_buffer_chunk(original, original));
                assert!(
                    changed_buffer_bytes(Some(&previous), start, original)
                        .unwrap()
                        .is_empty()
                );
                for position in 0..length {
                    let mut changed = original.to_vec();
                    changed[position] ^= 0x80;
                    assert!(!equal_buffer_chunk(original, &changed));
                    assert_eq!(
                        changed_buffer_bytes(Some(&previous), start, &changed).unwrap(),
                        start + position..start + position + 1
                    );
                }
            }
        }
        assert_eq!(
            changed_buffer_bytes(Some(&previous), 529, &[previous[529], 0, 0]).unwrap(),
            530..532
        );
        assert_eq!(
            changed_buffer_bytes(Some(&previous), 534, &[0, 0]).unwrap(),
            530..536
        );
    }
}
