//! Preserve host mirrors without reading unchanged parts of large GPU buffers.
use super::*;
use std::ops::Range;

#[derive(Clone, Default)]
pub(super) struct Readbacks(BTreeMap<vk::Buffer, Vec<Range<u64>>>);

impl Readbacks {
    pub(super) fn insert(&mut self, buffer: vk::Buffer, offset: u64, size: u64) -> VkResult<()> {
        let end = offset
            .checked_add(size)
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        if size == 0 {
            return Ok(());
        }
        let ranges = self.0.entry(buffer).or_default();
        let first = ranges.partition_point(|range| range.end < offset);
        let last = first + ranges[first..].partition_point(|range| range.start <= end);
        let start = ranges
            .get(first)
            .map_or(offset, |range| offset.min(range.start));
        let end = if last > first {
            end.max(ranges[last - 1].end)
        } else {
            end
        };
        ranges.splice(first..last, std::iter::once(start..end));
        Ok(())
    }

    pub(super) fn contains(&self, buffer: &vk::Buffer) -> bool {
        self.0.contains_key(buffer)
    }

    pub(super) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub(super) fn into_ranges(self) -> impl Iterator<Item = (vk::Buffer, Vec<Range<u64>>)> {
        self.0.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coalesces_writes_but_does_not_read_unwritten_gaps() {
        let buffer = vk::Buffer::from_raw(1);
        let mut readbacks = Readbacks::default();
        for (offset, size) in [(32, 8), (8, 8), (12, 4), (16, 4), (4, 4), (80, 4)] {
            readbacks.insert(buffer, offset, size).unwrap();
        }
        assert_eq!(readbacks.0[&buffer], [4..20, 32..40, 80..84]);
        readbacks.insert(buffer, 20, 60).unwrap();
        assert_eq!(readbacks.0[&buffer].len(), 1);
        assert_eq!(readbacks.0[&buffer][0], 4..84);
        assert!(readbacks.insert(buffer, u64::MAX, 4).is_err());
        assert_eq!(readbacks.0[&buffer].len(), 1);
        assert_eq!(readbacks.0[&buffer][0], 4..84);
    }
}
