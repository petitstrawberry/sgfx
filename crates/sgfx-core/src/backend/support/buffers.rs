//! Borrowed sparse upload sources and bounded revision deltas.
use alloc::vec::Vec;

/// Cumulative byte changes since `previous_revision`, an exact physical base.
/// `range` is the compatible envelope. Optional intervals omit unwritten gaps;
/// an empty interval list requires full repair after tracking overflows.
#[derive(Clone)]
pub struct BufferUpdate<const N: usize, const ALIGN: usize> {
    pub previous_revision: u64,
    pub range: core::ops::Range<usize>,
    pub intervals: Option<BufferIntervals<N, ALIGN>>,
}

/// One disjoint range borrowed from a stable pending write payload. Sparse
/// sources supply these overlays in ascending, nonoverlapping range order.
pub struct BufferOverlay<'a> {
    pub range: core::ops::Range<usize>,
    pub bytes: &'a [u8],
}

/// Read-only source for a single logical buffer preparation. Sparse sources
/// keep the accepted shadow untouched and materialize only the requested
/// upload range into the backend's bounded scratch buffer.
pub enum BufferSource<'a> {
    Contiguous(&'a [u8]),
    Sparse {
        base: Option<&'a [u8]>,
        overlays: &'a [BufferOverlay<'a>],
        len: usize,
    },
}

impl BufferSource<'_> {
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Contiguous(bytes) => bytes.len(),
            Self::Sparse { len, .. } => *len,
        }
    }

    /// Copy an exact logical range from normalized last-write-wins overlays.
    pub fn copy_range(
        &self,
        range: core::ops::Range<usize>,
        destination: &mut [u8],
    ) -> Result<(), crate::ir::Error> {
        self.copy_range_with(range, destination, || {})
    }

    fn copy_range_with(
        &self,
        range: core::ops::Range<usize>,
        destination: &mut [u8],
        mut visit: impl FnMut(),
    ) -> Result<(), crate::ir::Error> {
        if range.start > range.end || range.len() != destination.len() || range.end > self.len() {
            return Err(crate::ir::Error::OutOfBounds);
        }
        destination.fill(0);
        match self {
            Self::Contiguous(bytes) => {
                let source = bytes
                    .get(range.clone())
                    .ok_or(crate::ir::Error::OutOfBounds)?;
                destination.copy_from_slice(source);
            }
            Self::Sparse { base, overlays, .. } => {
                if let Some(base) = base {
                    copy_intersection(base, 0, range.clone(), destination);
                }
                for overlay in super::ranges::intersecting(overlays, range.clone(), |overlay| {
                    overlay.range.clone()
                }) {
                    visit();
                    copy_intersection(
                        overlay.bytes,
                        overlay.range.start,
                        range.clone(),
                        destination,
                    );
                }
            }
        }
        Ok(())
    }

    pub fn direct_range(&self, range: core::ops::Range<usize>) -> Option<&[u8]> {
        let Self::Contiguous(bytes) = self else {
            return None;
        };
        bytes.get(range)
    }
}

fn copy_intersection(
    source: &[u8],
    source_start: usize,
    requested: core::ops::Range<usize>,
    destination: &mut [u8],
) {
    let first = requested.start.max(source_start);
    let last = requested.end.min(source_start.saturating_add(source.len()));
    if first >= last {
        return;
    }
    let destination_start = first - requested.start;
    let source_offset = first - source_start;
    let count = last - first;
    destination[destination_start..destination_start + count]
        .copy_from_slice(&source[source_offset..source_offset + count]);
}

#[cfg(test)]
mod buffer_range_tests {
    use super::*;

    #[test]
    fn sparse_upload_queries_visit_only_intersecting_normalized_overlays() {
        let data = [9; 8];
        let overlays = (0..8192)
            .map(|index| BufferOverlay {
                range: index * 16..index * 16 + 8,
                bytes: &data,
            })
            .collect::<Vec<_>>();
        let source = BufferSource::Sparse {
            base: None,
            overlays: &overlays,
            len: 131072,
        };
        let mut visits = 0;
        let mut bytes = [1; 18];
        source
            .copy_range_with(65535..65553, &mut bytes, || visits += 1)
            .unwrap();
        assert_eq!(visits, 2);
        assert_eq!(
            bytes,
            [0, 9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 9]
        );
        visits = 0;
        let mut gap = [1; 8];
        source
            .copy_range_with(65528..65536, &mut gap, || visits += 1)
            .unwrap();
        assert_eq!(visits, 0);
        assert_eq!(gap, [0; 8]);
        let mut full = [0; 65536];
        visits = 0;
        source
            .copy_range_with(65536..131072, &mut full, || visits += 1)
            .unwrap();
        assert_eq!(visits, 4096);
        assert_eq!(
            &full[..16],
            &[9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0]
        );
        let reversed = core::ops::Range { start: 10, end: 9 };
        assert!(source.copy_range(reversed, &mut []).is_err());
    }
}

/// Nonempty lists always reserve the complete bound before admission. This
/// permits cumulative merging at accepted publication without allocating.
pub struct BufferIntervals<const N: usize, const ALIGN: usize> {
    ranges: Vec<core::ops::Range<usize>>,
}

impl<const N: usize, const ALIGN: usize> Clone for BufferIntervals<N, ALIGN> {
    fn clone(&self) -> Self {
        let mut ranges = Vec::new();
        if !self.ranges.is_empty() && ranges.try_reserve_exact(N).is_ok() {
            ranges.extend_from_slice(&self.ranges);
        }
        // Allocation failure conservatively produces the full-repair marker.
        Self { ranges }
    }
}

impl<const N: usize, const ALIGN: usize> BufferIntervals<N, ALIGN> {
    fn new(range: core::ops::Range<usize>) -> Self {
        assert!(N > 0 && ALIGN.is_power_of_two());
        let mut intervals = Self { ranges: Vec::new() };
        if intervals.ranges.try_reserve_exact(N).is_ok()
            && let Some(range) = Self::aligned(range)
        {
            intervals.ranges.push(range);
        }
        intervals
    }

    fn aligned(range: core::ops::Range<usize>) -> Option<core::ops::Range<usize>> {
        assert!(
            ALIGN.is_power_of_two(),
            "upload alignment must be a power of two"
        );
        (range.start < range.end).then_some(())?;
        Some((range.start & !(ALIGN - 1))..(range.end.checked_add(ALIGN - 1)? & !(ALIGN - 1)))
    }

    pub fn ranges(&self) -> &[core::ops::Range<usize>] {
        &self.ranges
    }

    fn include(&mut self, range: core::ops::Range<usize>) {
        if self.ranges.is_empty() {
            return; // Full repair remains sticky until accepted preparation.
        }
        let Some(mut range) = Self::aligned(range) else {
            self.ranges.clear();
            return;
        };
        let first = self.ranges.partition_point(|old| old.end < range.start);
        let last = first + self.ranges[first..].partition_point(|old| old.start <= range.end);
        if first < last {
            range.start = range.start.min(self.ranges[first].start);
            range.end = range.end.max(self.ranges[last - 1].end);
            self.ranges.drain(first..last);
        } else if self.ranges.len() == N {
            self.ranges.clear();
            return;
        }
        debug_assert!(self.ranges.capacity() >= N);
        self.ranges.insert(first, range);
    }
}

impl<const N: usize, const ALIGN: usize> BufferUpdate<N, ALIGN> {
    pub fn sparse(previous_revision: u64, range: core::ops::Range<usize>) -> Self {
        Self {
            previous_revision,
            intervals: Some(BufferIntervals::new(range.clone())),
            range,
        }
    }

    pub fn include_range(&mut self, range: core::ops::Range<usize>) {
        self.range.start = self.range.start.min(range.start);
        self.range.end = self.range.end.max(range.end);
        if let Some(intervals) = &mut self.intervals {
            intervals.include(range);
        }
    }

    /// Called only when the earlier delta ends at our immediate predecessor.
    /// All storage is already reserved, including on the publication path.
    pub fn merge_previous(&mut self, previous: &Self) {
        self.previous_revision = previous.previous_revision;
        self.range.start = self.range.start.min(previous.range.start);
        self.range.end = self.range.end.max(previous.range.end);
        if previous
            .intervals
            .as_ref()
            .is_some_and(|ranges| ranges.ranges.is_empty())
        {
            if let Some(intervals) = &mut self.intervals {
                intervals.ranges.clear();
            } else {
                self.intervals = Some(BufferIntervals { ranges: Vec::new() });
            }
        } else if let Some(intervals) = &mut self.intervals {
            if let Some(earlier) = &previous.intervals {
                for range in &earlier.ranges {
                    intervals.include(range.clone());
                }
            } else {
                intervals.include(previous.range.clone());
            }
        }
        // A general planner's absent interval list keeps the envelope fallback.
    }
}

#[cfg(test)]
mod delta_tests {
    use super::*;
    #[test]
    fn dirty_intervals_align_coalesce_and_merge_without_allocating() {
        let mut update = BufferUpdate::<128, 4>::sparse(7, 5..6);
        let pointer = update.intervals.as_ref().unwrap().ranges.as_ptr();
        update.include_range(8..9);
        update.include_range(20..21);
        assert_eq!(
            update.intervals.as_ref().unwrap().ranges(),
            &[4..12, 20..24]
        );
        update.include_range(13..17);
        assert_eq!(update.intervals.as_ref().unwrap().ranges().len(), 1);
        assert_eq!(update.intervals.as_ref().unwrap().ranges()[0], 4..24);
        assert_eq!(update.range, 5..21);
        let earlier = BufferUpdate::<128, 4>::sparse(5, 40..44);
        update.merge_previous(&earlier);
        assert_eq!(update.previous_revision, 5);
        assert_eq!(
            update.intervals.as_ref().unwrap().ranges(),
            &[4..24, 40..44]
        );
        assert_eq!(update.intervals.as_ref().unwrap().ranges.as_ptr(), pointer);
        let mut cloned = update.clone();
        let pointer = cloned.intervals.as_ref().unwrap().ranges.as_ptr();
        cloned.include_range(100..104);
        assert_eq!(cloned.intervals.as_ref().unwrap().ranges.as_ptr(), pointer);
        assert_eq!(cloned.intervals.as_ref().unwrap().ranges().len(), 3);
        assert_eq!(update.intervals.as_ref().unwrap().ranges().len(), 2);
    }

    #[test]
    fn dirty_interval_overflow_is_sticky_and_general_plans_keep_the_envelope() {
        let mut update = BufferUpdate::<128, 4>::sparse(7, 0..4);
        for index in 1..128 {
            update.include_range(index * 8..index * 8 + 4);
        }
        assert_eq!(update.intervals.as_ref().unwrap().ranges().len(), 128);
        let pointer = update.intervals.as_ref().unwrap().ranges.as_ptr();
        // A touching interval at capacity still coalesces without overflow.
        update.include_range(1020..1024);
        assert_eq!(update.intervals.as_ref().unwrap().ranges().len(), 128);
        update.include_range(1032..1036);
        assert!(update.intervals.as_ref().unwrap().ranges().is_empty());
        update.include_range(0..2048);
        assert!(update.intervals.as_ref().unwrap().ranges().is_empty());
        assert_eq!(update.intervals.as_ref().unwrap().ranges.as_ptr(), pointer);
        let mut next = BufferUpdate::<128, 4>::sparse(8, 4096..4100);
        next.merge_previous(&update);
        assert_eq!(next.previous_revision, 7);
        assert!(next.intervals.as_ref().unwrap().ranges().is_empty());

        let mut general = BufferUpdate::<128, 4> {
            previous_revision: 9,
            range: 5000..5004,
            intervals: None,
        };
        general.merge_previous(&BufferUpdate::<128, 4>::sparse(8, 4000..4004));
        assert!(general.intervals.is_none());
        assert_eq!(general.range, 4000..5004);
        general.merge_previous(&update);
        assert!(general.intervals.as_ref().unwrap().ranges().is_empty());
    }
    #[test]
    fn another_transport_selects_its_own_alignment_and_capacity() {
        let mut delta = BufferUpdate::<2, 16>::sparse(4, 1..3);
        delta.include_range(33..34);
        assert_eq!(delta.intervals.as_ref().unwrap().ranges(), &[0..16, 32..48]);
        delta.include_range(65..66);
        assert!(delta.intervals.as_ref().unwrap().ranges().is_empty());
        assert_eq!(delta.previous_revision, 4);
        assert_eq!(delta.range, 1..66);
    }
}
