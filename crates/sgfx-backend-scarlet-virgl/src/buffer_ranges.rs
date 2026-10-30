//! Disjoint last-write-wins views over stable chronological write payloads.

use alloc::vec::Vec;
use core::ops::Range;

use super::IrSubmitError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Segment {
    pub start: usize,
    pub end: usize,
    pub write: usize,
}

#[derive(Default)]
pub(super) struct RangeIndex {
    segments: Vec<Segment>,
    #[cfg(test)]
    pub visits: core::cell::Cell<usize>,
}

/// The input must be sorted and disjoint; both ends are monotonic. Searching
/// both boundaries avoids walking writes before or after the requested range.
pub(crate) fn intersecting<T>(
    items: &[T],
    range: Range<usize>,
    bounds: impl Fn(&T) -> Range<usize>,
) -> &[T] {
    if range.is_empty() {
        return &items[..0];
    }
    let first = items.partition_point(|item| bounds(item).end <= range.start);
    let remaining = &items[first..];
    let count = remaining.partition_point(|item| bounds(item).start < range.end);
    &remaining[..count]
}

impl RangeIndex {
    pub fn all(&self) -> &[Segment] {
        &self.segments
    }

    pub fn query(&self, range: Range<usize>) -> impl Iterator<Item = &Segment> {
        intersecting(&self.segments, range, |segment| segment.start..segment.end)
            .iter()
            .inspect(|_| {
                #[cfg(test)]
                self.visits.set(self.visits.get() + 1);
            })
    }

    /// Reserve before mutation. Splitting an old segment keeps references to
    /// its original payload; overwritten payloads remain owned by the plan.
    pub fn insert(&mut self, range: Range<usize>, write: usize) -> Result<(), IrSubmitError> {
        if range.is_empty() {
            return Ok(());
        }
        let first = self
            .segments
            .partition_point(|segment| segment.end <= range.start);
        let last =
            first + self.segments[first..].partition_point(|segment| segment.start < range.end);
        let prefix = self
            .segments
            .get(first)
            .copied()
            .filter(|segment| segment.start < range.start)
            .map(|segment| Segment {
                end: range.start,
                ..segment
            });
        let suffix = last
            .checked_sub(1)
            .filter(|_| last > first)
            .and_then(|index| self.segments.get(index).copied())
            .filter(|segment| segment.end > range.end)
            .map(|segment| Segment {
                start: range.end,
                ..segment
            });
        let added = 1 + usize::from(prefix.is_some()) + usize::from(suffix.is_some());
        let removed = last - first;
        self.segments
            .try_reserve(added.saturating_sub(removed))
            .map_err(|_| IrSubmitError::OutOfMemory)?;
        let segment = Segment {
            start: range.start,
            end: range.end,
            write,
        };
        let old_len = self.segments.len();
        let new_len = old_len - removed + added;
        if new_len > old_len {
            self.segments.resize(new_len, segment);
        }
        self.segments.copy_within(last..old_len, first + added);
        self.segments.truncate(new_len);
        let mut at = first;
        if let Some(prefix) = prefix {
            self.segments[at] = prefix;
            at += 1;
        }
        self.segments[at] = segment;
        if let Some(suffix) = suffix {
            self.segments[at + 1] = suffix;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use core::cell::Cell;

    #[test]
    fn normalized_segments_split_trim_and_replace_out_of_order_writes() {
        let mut index = RangeIndex::default();
        let mut expected = [None; 128];
        for (write, range) in [10..50, 20..30, 0..12, 25..45, 48..75, 4..6, 0..100]
            .into_iter()
            .enumerate()
        {
            index.insert(range.clone(), write).unwrap();
            expected[range].fill(Some(write));
            for (offset, expected) in expected.iter().enumerate() {
                let found = index
                    .query(offset..offset + 1)
                    .next()
                    .map(|segment| segment.write);
                assert_eq!(&found, expected, "write {write}, byte {offset}");
            }
            assert!(
                index
                    .all()
                    .windows(2)
                    .all(|pair| pair[0].end <= pair[1].start)
            );
        }
        assert_eq!(
            index.all(),
            &[Segment {
                start: 0,
                end: 100,
                write: 6
            }]
        );
        index.insert(70..70, 7).unwrap();
        assert_eq!(index.all().len(), 1);
    }

    #[test]
    fn range_lookup_has_logarithmic_search_and_visits_only_intersections() {
        let mut index = RangeIndex::default();
        // One buffer with thousands of writes, inserted in descending order.
        for write in (0..8192).rev() {
            index.insert(write * 16..write * 16 + 8, write).unwrap();
        }
        index.visits.set(0);
        let selected = index
            .query(65535..65553)
            .map(|segment| segment.write)
            .collect::<Vec<_>>();
        assert_eq!(selected, vec![4096, 4097]);
        assert_eq!(index.visits.get(), 2);
        index.visits.set(0);
        assert!(index.query(65528..65536).next().is_none());
        assert_eq!(index.visits.get(), 0);
        assert!(index.query(0..0).next().is_none());
        let comparisons = Cell::new(0);
        let selected = intersecting(index.all(), 65535..65553, |segment| {
            comparisons.set(comparisons.get() + 1);
            segment.start..segment.end
        });
        assert_eq!(selected.len(), 2);
        assert!(
            comparisons.get() <= 28,
            "{} boundary probes",
            comparisons.get()
        );
    }

    #[test]
    fn random_overwrites_match_an_independent_byte_owner_model() {
        let mut index = RangeIndex::default();
        let mut expected = vec![None; 4096];
        let mut seed = 42_u64;
        for write in 0..1500 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let start = (seed >> 16) as usize % expected.len();
            let end = (start + (seed >> 32) as usize % 500).min(expected.len());
            index.insert(start..end, write).unwrap();
            expected[start..end].fill(Some(write));
        }
        for (offset, expected) in expected.iter().enumerate() {
            let actual = index
                .query(offset..offset + 1)
                .next()
                .map(|segment| segment.write);
            assert_eq!(&actual, expected, "byte {offset}");
        }
    }
}
