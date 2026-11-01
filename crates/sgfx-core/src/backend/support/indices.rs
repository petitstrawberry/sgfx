//! Bounded index-extrema caching with caller-owned, revisioned reads.
use crate::ir::{self, IndexFormat};
use alloc::vec::Vec;

const INDEX_BOUNDS_CACHE_SIZE: usize = 32;
const INDEX_BLOCK_WORDS: usize = 1024;
const INDEX_BLOCK_CACHE_SIZE: usize = 512;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct IndexBoundsKey {
    pub buffer: ir::BufferId,
    pub slot: usize,
    pub revision: u64,
    pub start: usize,
    pub end: usize,
    pub format: IndexFormat,
}

#[derive(Clone, Copy)]
struct IndexBoundsEntry {
    key: IndexBoundsKey,
    bounds: Option<(u32, u32)>,
}

/// Exact round-robin and direct-mapped block caches owned by one plan only.
/// Raw bounds are independent of baseVertex; no command bytes are retained.
#[derive(Default)]
pub struct IndexBoundsCache {
    entries: [Option<IndexBoundsEntry>; INDEX_BOUNDS_CACHE_SIZE],
    next: usize,
    // At most 512 tagged metadata entries, allocated only after a full block
    // is consumed. No index bytes or cross-plan state are retained.
    blocks: Vec<Option<IndexBoundsEntry>>,
    block_allocation_failed: bool,
    live: usize,
}

impl IndexBoundsCache {
    fn get(&self, key: IndexBoundsKey) -> Option<Option<(u32, u32)>> {
        self.entries
            .iter()
            .flatten()
            .find(|entry| entry.key == key)
            .map(|entry| entry.bounds)
    }

    fn insert(&mut self, key: IndexBoundsKey, bounds: Option<(u32, u32)>) {
        if self.entries[self.next].is_none() {
            self.live += 1;
        }
        self.entries[self.next] = Some(IndexBoundsEntry { key, bounds });
        self.next = (self.next + 1) % INDEX_BOUNDS_CACHE_SIZE;
    }

    pub fn invalidate(&mut self, slot: usize) {
        // Prefix uploads commonly precede every draw. Do not walk either
        // cache until there is a proof to invalidate.
        if self.live == 0 {
            return;
        }
        for entry in self.entries.iter_mut().chain(self.blocks.iter_mut()) {
            if entry.is_some_and(|entry| entry.key.slot == slot) {
                *entry = None;
                self.live -= 1;
            }
        }
    }

    fn block_bucket(key: IndexBoundsKey) -> usize {
        let block = key.start / (INDEX_BLOCK_WORDS * key.format.byte_size() as usize);
        let salt = key.slot.wrapping_mul(0x9e37_79b1)
            ^ key.revision as usize
            ^ if key.format == IndexFormat::Uint16 {
                0
            } else {
                0x85eb_ca6b
            };
        (block ^ salt) & (INDEX_BLOCK_CACHE_SIZE - 1)
    }

    fn block(&mut self, key: IndexBoundsKey) -> Option<Option<(u32, u32)>> {
        self.blocks
            .get(Self::block_bucket(key))
            .copied()
            .flatten()
            .filter(|entry| entry.key == key)
            .map(|entry| entry.bounds)
    }

    fn insert_block(&mut self, key: IndexBoundsKey, bounds: Option<(u32, u32)>) {
        if self.blocks.is_empty() {
            if self.block_allocation_failed {
                return;
            }
            if self
                .blocks
                .try_reserve_exact(INDEX_BLOCK_CACHE_SIZE)
                .is_err()
            {
                self.block_allocation_failed = true;
                return;
            }
            self.blocks.resize(INDEX_BLOCK_CACHE_SIZE, None);
        }
        let bucket = Self::block_bucket(key);
        if self.blocks[bucket].is_none() {
            self.live += 1;
        }
        self.blocks[bucket] = Some(IndexBoundsEntry { key, bounds });
    }
}

pub fn raw_index_bounds(
    bytes: &[u8],
    format: IndexFormat,
) -> Result<Option<(u32, u32)>, ir::Error> {
    let mut bounds = None;
    for index in bytes.chunks_exact(format.byte_size() as usize) {
        let index = match format {
            IndexFormat::Uint16 => u32::from(u16::from_le_bytes(index.try_into().unwrap())),
            IndexFormat::Uint32 => u32::from_le_bytes(index.try_into().unwrap()),
        };
        bounds = Some(
            bounds.map_or((index, index), |(minimum, maximum): (u32, u32)| {
                (minimum.min(index), maximum.max(index))
            }),
        );
    }
    Ok(bounds)
}

pub fn maximum_with_base(bounds: Option<(u32, u32)>, base_vertex: i32) -> Result<u32, ir::Error> {
    let Some((minimum, maximum)) = bounds else {
        return Ok(0);
    };
    // Signed addition is monotone: validating both extrema proves every index
    // fits, including a negative base underflow hidden by a valid maximum.
    u32::try_from(i64::from(minimum) + i64::from(base_vertex))
        .map_err(|_| ir::Error::OutOfBounds)?;
    u32::try_from(i64::from(maximum) + i64::from(base_vertex)).map_err(|_| ir::Error::OutOfBounds)
}

fn merge_index_bounds(a: Option<(u32, u32)>, b: Option<(u32, u32)>) -> Option<(u32, u32)> {
    match (a, b) {
        (Some((a_min, a_max)), Some((b_min, b_max))) => Some((a_min.min(b_min), a_max.max(b_max))),
        (a, None) | (None, a) => a,
    }
}

/// Authoritative range reader. Callers validate resource identity and range bounds
/// before querying, and invalidate the slot on every pending write or copy.
pub trait IndexReader {
    type Error;
    fn read(&mut self, range: core::ops::Range<usize>) -> Result<Option<(u32, u32)>, Self::Error>;
    /// Optional diagnostics; the default has no runtime work.
    fn block(&mut self, _hit: bool) {}
}

impl IndexBoundsCache {
    /// Reuse exact and aligned-block extrema without retaining buffer bytes.
    /// The key must describe one validated, authoritative buffer version.
    pub fn bounds<R: IndexReader>(
        &mut self,
        key: IndexBoundsKey,
        reader: &mut R,
    ) -> Result<Option<(u32, u32)>, R::Error> {
        if let Some(bounds) = self.get(key) {
            return Ok(bounds);
        }
        let bounds = self.blocked_bounds(key, reader)?;
        self.insert(key, bounds);
        Ok(bounds)
    }

    fn blocked_bounds<R: IndexReader>(
        &mut self,
        key: IndexBoundsKey,
        reader: &mut R,
    ) -> Result<Option<(u32, u32)>, R::Error> {
        let width = key.format.byte_size() as usize;
        let block_bytes = INDEX_BLOCK_WORDS * width;
        let mut cursor = key.start;
        // Match chunks_exact: trailing bytes are not indices, but the entire
        // caller-supplied byte range was validated before looking in either cache.
        let end = key.end - (key.end - key.start) % width;
        let leading_words =
            (INDEX_BLOCK_WORDS - (cursor / width) % INDEX_BLOCK_WORDS) % INDEX_BLOCK_WORDS;
        let head_end = cursor + (end - cursor).min(leading_words * width);
        let mut bounds = None;
        if head_end != cursor {
            bounds = reader.read(cursor..head_end)?;
            cursor = head_end;
        }
        while end - cursor >= block_bytes {
            let block = IndexBoundsKey {
                start: cursor,
                end: cursor + block_bytes,
                ..key
            };
            let cached = self.block(block);
            reader.block(cached.is_some());
            let part = if let Some(bounds) = cached {
                bounds
            } else {
                let bounds = reader.read(block.start..block.end)?;
                self.insert_block(block, bounds);
                bounds
            };
            bounds = merge_index_bounds(bounds, part);
            cursor += block_bytes;
        }
        if cursor != end {
            bounds = merge_index_bounds(bounds, reader.read(cursor..end)?);
        }
        Ok(bounds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BufferUsage, ResourceTable};
    #[test]
    fn block_cache_uses_tagged_single_slots_and_empty_invalidation_allocates_nothing() {
        let table = ResourceTable::new();
        let define = || {
            table
                .define_buffer(ir::BufferDesc::new(8192, BufferUsage::INDEX).unwrap())
                .unwrap()
                .id()
        };
        let buffer = define();
        let key = IndexBoundsKey {
            buffer,
            slot: 0,
            revision: 7,
            start: 0,
            end: INDEX_BLOCK_WORDS * 4,
            format: IndexFormat::Uint32,
        };
        let mut cache = IndexBoundsCache::default();
        for _ in 0..100 {
            cache.invalidate(0);
        }
        assert_eq!(cache.blocks.capacity(), 0);
        assert_eq!(cache.live, 0);
        cache.insert(key, Some((2, 3)));
        cache.insert_block(key, Some((2, 3)));
        assert_eq!(cache.blocks.len(), INDEX_BLOCK_CACHE_SIZE);
        assert_eq!(cache.block(key), Some(Some((2, 3))));
        let collision = IndexBoundsKey {
            start: INDEX_BLOCK_CACHE_SIZE * INDEX_BLOCK_WORDS * 4,
            end: (INDEX_BLOCK_CACHE_SIZE + 1) * INDEX_BLOCK_WORDS * 4,
            ..key
        };
        assert_eq!(
            IndexBoundsCache::block_bucket(key),
            IndexBoundsCache::block_bucket(collision)
        );
        cache.insert_block(collision, Some((7, 9)));
        assert_eq!(cache.block(key), None);
        assert_eq!(cache.block(collision), Some(Some((7, 9))));
        for changed in [
            IndexBoundsKey {
                buffer: define(),
                ..collision
            },
            IndexBoundsKey {
                revision: 8,
                ..collision
            },
            IndexBoundsKey {
                format: IndexFormat::Uint16,
                ..collision
            },
            IndexBoundsKey {
                start: collision.start + 1,
                end: collision.end + 1,
                ..collision
            },
        ] {
            assert_eq!(cache.block(changed), None);
        }
        table.release_buffer(buffer).unwrap();
        let next_generation = define();
        assert_ne!(next_generation, buffer);
        assert_eq!(
            cache.block(IndexBoundsKey {
                buffer: next_generation,
                ..collision
            }),
            None
        );
        cache.invalidate(0);
        assert_eq!(cache.live, 0);
        assert!(
            cache
                .entries
                .iter()
                .chain(cache.blocks.iter())
                .all(Option::is_none)
        );
        cache.invalidate(0); // also O(1) after entries were discarded
    }

    #[test]
    fn raw_index_extrema_preserve_per_index_signed_bounds_checks() {
        for format in [IndexFormat::Uint16, IndexFormat::Uint32] {
            for values in [&[][..], &[0, 2, 7][..], &[2, 9, 3][..], &[u32::MAX, 0][..]] {
                let bytes: Vec<u8> = values
                    .iter()
                    .flat_map(|index| match format {
                        IndexFormat::Uint16 => (*index as u16).to_le_bytes().to_vec(),
                        IndexFormat::Uint32 => index.to_le_bytes().to_vec(),
                    })
                    .collect();
                for base in [i32::MIN, -10, -2, 0, 2, i32::MAX] {
                    // Independent version of the previous per-index validation.
                    let mut expected = Ok(0u32);
                    for item in bytes.chunks_exact(format.byte_size() as usize) {
                        let raw = if format == IndexFormat::Uint16 {
                            u32::from(u16::from_le_bytes(item.try_into().unwrap()))
                        } else {
                            u32::from_le_bytes(item.try_into().unwrap())
                        };
                        match u32::try_from(i64::from(raw) + i64::from(base)) {
                            Ok(index) => expected = expected.map(|maximum| maximum.max(index)),
                            Err(_) => {
                                expected = Err(());
                                break;
                            }
                        }
                    }
                    let actual = maximum_with_base(raw_index_bounds(&bytes, format).unwrap(), base);
                    match expected {
                        Ok(value) => assert_eq!(actual.unwrap(), value),
                        Err(()) => assert!(matches!(actual, Err(ir::Error::OutOfBounds))),
                    }
                }
            }
        }
    }

    #[test]
    fn index_cache_is_bounded_and_keys_identity_revision_exact_range_and_format() {
        let table = ResourceTable::new();
        let foreign = ResourceTable::new();
        let define = |table: &ResourceTable| {
            table
                .define_buffer(ir::BufferDesc::new(256, BufferUsage::INDEX).unwrap())
                .unwrap()
                .id()
        };
        let buffer = define(&table);
        let key = IndexBoundsKey {
            buffer,
            slot: 0,
            revision: 7,
            start: 0,
            end: 4,
            format: IndexFormat::Uint16,
        };
        let mut cache = IndexBoundsCache::default();
        cache.insert(key, Some((1, 9)));
        assert_eq!(cache.get(key), Some(Some((1, 9))));
        for different in [
            IndexBoundsKey {
                buffer: define(&table),
                ..key
            },
            IndexBoundsKey {
                buffer: define(&foreign),
                ..key
            },
            IndexBoundsKey { revision: 8, ..key },
            IndexBoundsKey { start: 2, ..key },
            IndexBoundsKey { end: 6, ..key },
            IndexBoundsKey {
                format: IndexFormat::Uint32,
                ..key
            },
        ] {
            assert_eq!(cache.get(different), None);
        }
        for index in 1..=INDEX_BOUNDS_CACHE_SIZE {
            cache.insert(
                IndexBoundsKey {
                    start: index * 4,
                    end: index * 4 + 4,
                    ..key
                },
                Some((0, index as u32)),
            );
        }
        assert_eq!(
            cache.entries.iter().flatten().count(),
            INDEX_BOUNDS_CACHE_SIZE
        );
        assert_eq!(
            cache.get(key),
            None,
            "round-robin eviction must remove the oldest entry"
        );
        cache.invalidate(0);
        assert!(cache.entries.iter().all(Option::is_none));
    }
    #[test]
    fn shared_reader_reuses_blocks_and_invalidates_same_revision_writes() {
        struct Reader {
            bytes: Vec<u8>,
            reads: usize,
            words: usize,
            hits: usize,
        }
        impl IndexReader for Reader {
            type Error = ir::Error;
            fn read(
                &mut self,
                range: core::ops::Range<usize>,
            ) -> Result<Option<(u32, u32)>, Self::Error> {
                let bytes = self.bytes.get(range).ok_or(ir::Error::OutOfBounds)?;
                self.reads += 1;
                self.words += bytes.len() / 4;
                raw_index_bounds(bytes, IndexFormat::Uint32)
            }
            fn block(&mut self, hit: bool) {
                self.hits += usize::from(hit);
            }
        }
        let table = ResourceTable::new();
        let buffer = table
            .define_buffer(ir::BufferDesc::new(16384, BufferUsage::INDEX).unwrap())
            .unwrap();
        let mut reader = Reader {
            bytes: (0..4096u32).flat_map(u32::to_le_bytes).collect(),
            reads: 0,
            words: 0,
            hits: 0,
        };
        let mut cache = IndexBoundsCache::default();
        let key = IndexBoundsKey {
            buffer: buffer.id(),
            slot: buffer.slot(),
            revision: 1,
            start: 0,
            end: 0,
            format: IndexFormat::Uint32,
        };
        for count in [1025, 1033, 2049, 2057, 1027] {
            assert_eq!(
                cache
                    .bounds(
                        IndexBoundsKey {
                            end: count * 4,
                            ..key
                        },
                        &mut reader
                    )
                    .unwrap(),
                Some((0, count as u32 - 1))
            );
        }
        assert_eq!((reader.reads, reader.words, reader.hits), (7, 2071, 5));
        let exact = IndexBoundsKey {
            end: 1027 * 4,
            ..key
        };
        assert_eq!(
            maximum_with_base(cache.bounds(exact, &mut reader).unwrap(), -1),
            Err(ir::Error::OutOfBounds)
        );
        assert_eq!(reader.reads, 7);
        reader.bytes[..4].copy_from_slice(&9000u32.to_le_bytes());
        cache.invalidate(buffer.slot());
        assert_eq!(cache.bounds(exact, &mut reader).unwrap(), Some((1, 9000)));
        assert_eq!(reader.reads, 9);
    }
}
