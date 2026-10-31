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
        let bytes = self.prepare_append(commands.len())?;
        self.current
            .bytes
            .try_reserve(commands.len())
            .map_err(|_| Error::OutOfMemory)?;
        self.current.bytes.extend_from_slice(commands);
        self.bytes = bytes;
        Ok(())
    }

    /// Transfer the encoding allocation when this packet starts a chunk.
    /// Otherwise keep the caller's buffer reusable and batch through `append`.
    pub(crate) fn append_owned(&mut self, commands: &mut Vec<u8>) -> Result<(), Error> {
        let bytes = self.prepare_append(commands.len())?;
        if self.current.bytes.is_empty() {
            self.current.bytes = core::mem::take(commands);
            self.bytes = bytes;
            Ok(())
        } else {
            self.append(commands)
        }
    }

    fn prepare_append(&mut self, length: usize) -> Result<usize, Error> {
        #[cfg(feature = "std")]
        if self.bytes.saturating_add(length) > MAX_STREAM_BYTES {
            std::eprintln!(
                "[SGFX VirGL] stream exceeds admission: staged={} packet={length} chunks={} limit={MAX_STREAM_BYTES}",
                self.bytes,
                self.chunks.len()
            );
        }
        let bytes = self
            .bytes
            .checked_add(length)
            .filter(|bytes| *bytes <= MAX_STREAM_BYTES)
            .ok_or(Error::TooLarge)?;
        self.prepare(length)?;
        Ok(bytes)
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
    fn owned_packet_adopts_allocation_and_preserves_prepared_upload_range() {
        let mut stream = Packets::new(128);
        stream.prepare(64).unwrap();
        assert_eq!(stream.upload_range(60).unwrap(), (0, 0));
        let mut commands = alloc::vec![7; 64];
        let pointer = commands.as_ptr();
        let capacity = commands.capacity();
        stream.append_owned(&mut commands).unwrap();
        assert_eq!(commands.capacity(), 0);
        assert_eq!(stream.current.bytes.as_ptr(), pointer);
        assert_eq!(stream.current.bytes.capacity(), capacity);
        assert_eq!(stream.upload_range(60).unwrap(), (0, 60));
        let chunks = stream.finish().unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].bytes, &[7; 64]);
        assert_eq!(chunks[0].arena, Some(0));
    }

    #[test]
    fn owned_packet_batches_after_existing_data_and_keeps_encoding_buffer() {
        let mut stream = Packets::new(128);
        stream.append(&[1; 32]).unwrap();
        let mut commands = alloc::vec![2; 64];
        let pointer = commands.as_ptr();
        stream.append_owned(&mut commands).unwrap();
        assert_eq!(commands.as_ptr(), pointer);
        assert_eq!(commands, &[2; 64]);
        stream.append(&[3; 32]).unwrap();
        let chunks = stream.finish().unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(&chunks[0].bytes[..32], &[1; 32]);
        assert_eq!(&chunks[0].bytes[32..96], &[2; 64]);
        assert_eq!(&chunks[0].bytes[96..], &[3; 32]);
    }

    #[test]
    fn owned_packet_rotates_and_adopts_without_reordering_or_losing_arenas() {
        let mut stream = Packets::new(128);
        stream.prepare(96).unwrap();
        assert_eq!(stream.upload_range(60).unwrap(), (0, 0));
        stream.append(&[1; 96]).unwrap();
        let mut commands = alloc::vec![2; 64];
        let pointer = commands.as_ptr();
        stream.append_owned(&mut commands).unwrap();
        assert_eq!(stream.current.bytes.as_ptr(), pointer);
        assert_eq!(stream.upload_range(60).unwrap(), (1, 0));
        stream.append(&[3; 64]).unwrap();
        let chunks = stream.finish().unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].bytes, &[1; 96]);
        assert_eq!(chunks[0].arena, Some(0));
        assert_eq!(&chunks[1].bytes[..64], &[2; 64]);
        assert_eq!(&chunks[1].bytes[64..], &[3; 64]);
        assert_eq!(chunks[1].arena, Some(1));
    }

    #[test]
    fn owned_packet_transport_and_admission_overflow_leave_state_and_input_intact() {
        let mut stream = Packets::new(128);
        stream.append(&[1; 96]).unwrap();
        assert_eq!(stream.upload_range(60).unwrap(), (0, 0));
        let mut oversized = alloc::vec![2; 129];
        assert_eq!(stream.append_owned(&mut oversized), Err(Error::TooLarge));
        assert_eq!(oversized, &[2; 129]);
        let mut commands = alloc::vec![3; 64];
        let pointer = commands.as_ptr();
        for staged in [MAX_STREAM_BYTES - 63, usize::MAX] {
            stream.bytes = staged;
            assert_eq!(stream.append_owned(&mut commands), Err(Error::TooLarge));
            assert_eq!(stream.bytes, staged);
            assert!(stream.chunks.is_empty());
            assert_eq!(stream.current.bytes, &[1; 96]);
            assert_eq!(stream.current.arena, Some(0));
            assert_eq!(stream.upload_offset, 60);
            assert_eq!(commands.as_ptr(), pointer);
            assert_eq!(commands, &[3; 64]);
        }
        stream.bytes = MAX_STREAM_BYTES - commands.len();
        stream.append_owned(&mut commands).unwrap();
        assert_eq!(stream.bytes, MAX_STREAM_BYTES);
        assert_eq!(stream.current.bytes.as_ptr(), pointer);
        let mut extra = alloc::vec![4];
        assert_eq!(stream.append_owned(&mut extra), Err(Error::TooLarge));
        assert_eq!(stream.current.bytes, &[3; 64]);
        assert_eq!(stream.chunks.len(), 1);
        assert_eq!(extra, &[4]);
    }

    #[test]
    fn empty_owned_packets_keep_checkpoint_and_do_not_add_trailing_chunks() {
        let mut stream = Packets::new(128);
        let mut empty = Vec::with_capacity(64);
        let pointer = empty.as_ptr();
        stream.append_owned(&mut empty).unwrap();
        let chunks = stream.finish().unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].bytes.is_empty());
        assert_eq!(chunks[0].bytes.as_ptr(), pointer);
        assert_eq!(chunks[0].arena, None);

        let mut stream = Packets::new(128);
        let mut commands = alloc::vec![1; 128];
        stream.append_owned(&mut commands).unwrap();
        stream.append_owned(&mut Vec::new()).unwrap();
        // Rotation can leave an empty current chunk before the checkpoint.
        stream.prepare(1).unwrap();
        stream.append_owned(&mut Vec::new()).unwrap();
        let chunks = stream.finish().unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].bytes, &[1; 128]);
    }

    #[test]
    fn empty_stream_keeps_a_native_checkpoint() {
        let chunks = Packets::new(128).finish().unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].bytes.is_empty());
        assert_eq!(chunks[0].arena, None);
    }
}
