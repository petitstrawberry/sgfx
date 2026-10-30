//! Packet-boundary batching and disjoint per-chunk vertex staging ranges.

extern crate alloc;

use alloc::vec::Vec;

pub(crate) const UPLOAD_ARENA_COUNT: usize = 4;
use crate::scheduler::MAX_STREAM_BYTES;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Error {
    TooLarge,
    OutOfMemory,
}

pub(crate) struct Chunk {
    pub(crate) bytes: Vec<u8>,
    pub(crate) arena: Option<usize>,
}

pub(crate) struct Packets {
    chunks: Vec<Chunk>,
    current: Chunk,
    upload_offset: u32,
    bytes: usize,
    limit: usize,
}

impl Packets {
    pub(crate) fn new(limit: usize) -> Self {
        Self {
            chunks: Vec::new(),
            current: Chunk {
                bytes: Vec::new(),
                arena: None,
            },
            upload_offset: 0,
            bytes: 0,
            limit,
        }
    }

    // Call before encoding a packet containing vertex writes: rotating the
    // native chunk must happen before choosing its arena and source offsets.
    pub(crate) fn prepare(&mut self, maximum: usize) -> Result<(), Error> {
        if maximum > self.limit {
            #[cfg(feature = "std")]
            std::eprintln!(
                "[SGFX VirGL] packet exceeds transport: bytes={maximum} limit={}",
                self.limit
            );
            return Err(Error::TooLarge);
        }
        if self.current.bytes.len().saturating_add(maximum) > self.limit {
            self.finish_chunk()?;
        }
        Ok(())
    }

    pub(crate) fn upload_range(&mut self, length: u32) -> Result<(usize, u32), Error> {
        let start = self.upload_offset;
        #[cfg(feature = "std")]
        if u64::from(start) + u64::from(length) > self.limit as u64 {
            std::eprintln!(
                "[SGFX VirGL] upload exceeds arena: offset={start} bytes={length} commands={} limit={}",
                self.current.bytes.len(),
                self.limit
            );
        }
        let end = start
            .checked_add(length)
            .filter(|end| *end as usize <= self.limit)
            .ok_or(Error::TooLarge)?;
        let arena = self.chunks.len() % UPLOAD_ARENA_COUNT;
        self.current.arena = Some(arena);
        self.upload_offset = end;
        Ok((arena, start))
    }

    pub(crate) fn append(&mut self, commands: &[u8]) -> Result<(), Error> {
        #[cfg(feature = "std")]
        if self.bytes.saturating_add(commands.len()) > MAX_STREAM_BYTES {
            std::eprintln!(
                "[SGFX VirGL] stream exceeds admission: staged={} packet={} chunks={} limit={MAX_STREAM_BYTES}",
                self.bytes,
                commands.len(),
                self.chunks.len()
            );
        }
        let bytes = self
            .bytes
            .checked_add(commands.len())
            .filter(|bytes| *bytes <= MAX_STREAM_BYTES)
            .ok_or(Error::TooLarge)?;
        self.prepare(commands.len())?;
        self.current
            .bytes
            .try_reserve(commands.len())
            .map_err(|_| Error::OutOfMemory)?;
        self.current.bytes.extend_from_slice(commands);
        self.bytes = bytes;
        Ok(())
    }

    fn finish_chunk(&mut self) -> Result<(), Error> {
        self.chunks.try_reserve(1).map_err(|_| Error::OutOfMemory)?;
        self.chunks.push(core::mem::replace(
            &mut self.current,
            Chunk {
                bytes: Vec::new(),
                arena: None,
            },
        ));
        self.upload_offset = 0;
        Ok(())
    }

    pub(crate) fn finish(mut self) -> Result<Vec<Chunk>, Error> {
        if !self.current.bytes.is_empty() || self.chunks.is_empty() {
            self.finish_chunk()?;
        }
        Ok(self.chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_stream_splits_at_packet_boundaries_and_reuses_only_between_chunks() {
        let mut stream = Packets::new(128);
        for index in 0..40 {
            stream.prepare(64).unwrap();
            assert_eq!(
                stream.upload_range(60).unwrap(),
                ((index / 2) % UPLOAD_ARENA_COUNT, (index % 2 * 60) as u32)
            );
            stream.append(&[index as u8; 64]).unwrap();
        }
        let chunks = stream.finish().unwrap();
        assert_eq!(chunks.len(), 20);
        for (index, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.bytes.len(), 128);
            assert_eq!(chunk.arena, Some(index % UPLOAD_ARENA_COUNT));
            assert_eq!(&chunk.bytes[..64], &[index as u8 * 2; 64]);
            assert_eq!(&chunk.bytes[64..], &[index as u8 * 2 + 1; 64]);
        }
    }

    #[test]
    fn texture_packets_rotate_before_the_next_vertex_packet_selects_its_arena() {
        let mut stream = Packets::new(128);
        stream.append(&[1; 96]).unwrap();
        stream.prepare(64).unwrap();
        assert_eq!(stream.upload_range(60).unwrap(), (1, 0));
        stream.append(&[2; 64]).unwrap();
        let chunks = stream.finish().unwrap();
        assert_eq!(chunks[0].arena, None);
        assert_eq!(chunks[1].arena, Some(1));
    }

    #[test]
    fn oversized_packet_and_range_are_rejected_without_advancing() {
        let mut stream = Packets::new(128);
        assert_eq!(stream.prepare(129), Err(Error::TooLarge));
        assert_eq!(stream.upload_range(129), Err(Error::TooLarge));
        assert_eq!(stream.upload_range(60).unwrap(), (0, 0));
        assert_eq!(stream.upload_range(u32::MAX), Err(Error::TooLarge));
        assert_eq!(stream.upload_range(60).unwrap(), (0, 60));
    }

    #[test]
    fn full_64_mib_upload_keeps_room_for_following_draw_packets() {
        let mut stream = Packets::new(2 * 1024 * 1024);
        let upload = alloc::vec![7; 2 * 1024 * 1024];
        for _ in 0..32 {
            stream.append(&upload).unwrap();
        }
        stream.append(&[9; 65536]).unwrap();
        let chunks = stream.finish().unwrap();
        assert_eq!(chunks.len(), 33);
        assert!(chunks[..32].iter().all(|chunk| chunk.bytes == upload));
        assert_eq!(chunks[32].bytes, &[9; 65536]);
    }

    #[test]
    fn exclusive_stream_bound_rejects_before_mutating_packet_state() {
        let mut stream = Packets::new(128);
        stream.bytes = MAX_STREAM_BYTES;
        assert_eq!(stream.append(&[1; 4]), Err(Error::TooLarge));
        assert!(stream.current.bytes.is_empty());
        assert!(stream.chunks.is_empty());
    }

    #[test]
    fn empty_stream_keeps_a_native_checkpoint() {
        let chunks = Packets::new(128).finish().unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].bytes.is_empty());
        assert_eq!(chunks[0].arena, None);
    }
}
