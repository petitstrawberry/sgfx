use super::{
    codec::{Codec, Reader},
    *,
};
use alloc::{collections::TryReserveError, vec::Vec};
use core::{cell::OnceCell, marker::PhantomData, mem::MaybeUninit};

const MAX_WORDS: usize = 64;

pub(crate) enum Storage<'r, 'data> {
    Rust(Vec<Command<'r, 'data>>),
    Abi {
        words: Vec<u64>,
        count: usize,
        decoded: OnceCell<Vec<Command<'r, 'data>>>,
    },
    Borrowed {
        words: &'data [u64],
        count: usize,
        decoded: OnceCell<Vec<Command<'r, 'data>>>,
    },
}

impl<'r, 'data> Storage<'r, 'data> {
    pub fn new(t: &ResourceTable) -> Self {
        if t.abi.enabled.get() {
            Self::Abi {
                words: Vec::new(),
                count: 0,
                decoded: OnceCell::new(),
            }
        } else {
            Self::Rust(Vec::new())
        }
    }
    pub const fn len(&self) -> usize {
        match self {
            Self::Rust(v) => v.len(),
            Self::Abi { count, .. } | Self::Borrowed { count, .. } => *count,
        }
    }
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn try_reserve(&mut self, n: usize) -> core::result::Result<(), TryReserveError> {
        match self {
            Self::Rust(v) => v.try_reserve(n),
            Self::Abi { words, .. } => words.try_reserve(n * MAX_WORDS),
            Self::Borrowed { .. } => unreachable!("immutable imported recording"),
        }
    }
    pub fn push(&mut self, command: Command<'r, 'data>) {
        match self {
            Self::Rust(v) => v.push(command),
            Self::Abi { words, count, .. } => {
                let start = words.len();
                encode(command, words);
                debug_assert!(words.len() - start <= MAX_WORDS);
                *count += 1;
            }
            Self::Borrowed { .. } => unreachable!("immutable imported recording"),
        }
    }
    pub fn batch(&self, table: u64) -> Option<Batch> {
        match self {
            Self::Rust(_) => None,
            Self::Abi { words, count, .. } => Some(Batch {
                table,
                words: Span::from_slice(words),
                count: *count,
            }),
            Self::Borrowed { words, count, .. } => Some(Batch {
                table,
                words: Span::from_slice(words),
                count: *count,
            }),
        }
    }
    pub fn has_compute_commands(&self) -> Result<bool> {
        match self {
            Self::Rust(commands) => Ok(commands.iter().any(|command| {
                matches!(
                    command,
                    Command::BeginComputePass
                        | Command::EndComputePass
                        | Command::SetComputePipeline(_)
                        | Command::Dispatch { .. }
                )
            })),
            Self::Abi { words, count, .. } => scan_compute(words, *count),
            Self::Borrowed { words, count, .. } => scan_compute(words, *count),
        }
    }
    #[inline]
    pub fn iter<'a>(&'a self, table: &'r ResourceTable) -> CommandIter<'a, 'r, 'data> {
        let inner = match self {
            Self::Rust(v) => Iter::Rust(v.iter()),
            Self::Abi { words, count, .. } => Iter::Abi {
                reader: Reader { words },
                remaining: *count,
                table,
            },
            Self::Borrowed { words, count, .. } => Iter::Abi {
                reader: Reader { words },
                remaining: *count,
                table,
            },
        };
        CommandIter {
            inner,
            _data: PhantomData,
        }
    }
    pub fn as_slice(&self, table: &'r ResourceTable) -> &[Command<'r, 'data>] {
        match self {
            Self::Rust(v) => v,
            Self::Abi { decoded, .. } | Self::Borrowed { decoded, .. } => {
                decoded.get_or_init(|| {
                    self.iter(table)
                        .collect::<Result<Vec<_>>>()
                        .expect("validated ABI recording")
                })
            }
        }
    }
}

fn scan_compute(words: &[u64], count: usize) -> Result<bool> {
    let mut reader = Reader { words };
    let mut found = false;
    for _ in 0..count {
        let header = reader.word()?;
        let length = (header >> 32) as usize;
        let opcode = header as u32;
        if length == 0 || length > MAX_WORDS || !(1..=28).contains(&opcode) {
            return Err(Error::InvalidDescriptor);
        }
        reader.take(length - 1)?;
        found |= (19..=22).contains(&opcode);
    }
    reader.end()?;
    Ok(found)
}

enum Iter<'a, 'r, 'data> {
    Rust(core::slice::Iter<'a, Command<'r, 'data>>),
    Abi {
        reader: Reader<'a>,
        remaining: usize,
        table: &'r ResourceTable,
    },
}

/// Allocation-free command views. Upload slices retain their original address.
/// ABI fields are read as scalars; no command Vec or upload copy is constructed.
pub struct CommandIter<'a, 'r, 'data> {
    inner: Iter<'a, 'r, 'data>,
    _data: PhantomData<&'data [u8]>,
}
/// Lending command reader. Native recordings are borrowed in place; ABI
/// operands are decoded into one reusable stack slot, outside the planner.
pub struct CommandReader<'a, 'r, 'data> {
    inner: Iter<'a, 'r, 'data>,
    scratch: MaybeUninit<Command<'r, 'data>>,
}
impl<'r, 'data> CommandReader<'_, 'r, 'data> {
    /// The returned view remains valid until the next mutable reader call.
    #[inline]
    pub fn next_command(&mut self) -> Result<Option<&Command<'r, 'data>>> {
        match &mut self.inner {
            Iter::Rust(values) => Ok(values.next()),
            Iter::Abi {
                reader,
                remaining,
                table,
            } => {
                if read_next(reader, remaining, table, &mut self.scratch)? {
                    // SAFETY: read_next initializes a complete command on Ok(true).
                    // The exclusive borrow prevents overwriting a live view.
                    Ok(Some(unsafe { self.scratch.assume_init_ref() }))
                } else {
                    Ok(None)
                }
            }
        }
    }
}
impl<'r, 'data> Iterator for CommandIter<'_, 'r, 'data> {
    type Item = Result<Command<'r, 'data>>;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            Iter::Rust(v) => v.next().copied().map(Ok),
            Iter::Abi {
                reader,
                remaining,
                table,
            } => {
                let mut out = MaybeUninit::uninit();
                match read_next(reader, remaining, table, &mut out) {
                    // SAFETY: read_next initialized out on Ok(true).
                    Ok(true) => Some(Ok(unsafe { out.assume_init() })),
                    Ok(false) => None,
                    Err(error) => Some(Err(error)),
                }
            }
        }
    }
}
#[inline]
fn read_next<'r, 'data>(
    reader: &mut Reader<'_>,
    remaining: &mut usize,
    table: &'r ResourceTable,
    out: &mut MaybeUninit<Command<'r, 'data>>,
) -> Result<bool> {
    let result = (|| {
        if *remaining == 0 {
            reader.end()?;
            return Ok(false);
        }
        *remaining -= 1;
        let header = reader.word()?;
        let count = (header >> 32) as usize;
        if count == 0 || count > MAX_WORDS {
            return Err(Error::InvalidDescriptor);
        }
        let mut fields = Reader {
            words: reader.take(count - 1)?,
        };
        // SAFETY: owned recordings retain upload borrows. Imported recordings
        // require valid spans for the entire consuming call.
        unsafe { decode_into(header as u32, &mut fields, table, out) }?;
        fields.end()?;
        Ok(true)
    })();
    if result.is_err() {
        *remaining = 0;
        reader.words = &[];
    }
    result
}

fn bytes(value: &[u8], w: &mut Vec<u64>) {
    (value.as_ptr() as usize).put(w);
    value.len().put(w);
}
unsafe fn borrowed_bytes<'a>(r: &mut Reader<'_>, t: &ResourceTable) -> Result<&'a [u8]> {
    let ptr = usize::get(r, t)? as *const u8;
    let len = usize::get(r, t)?;
    if len > isize::MAX as usize || (len > 0 && ptr.is_null()) {
        return Err(Error::InvalidValue);
    }
    // SAFETY: validity/lifetime are the importing caller's contract. No owner
    // or allocator moves between libraries; the backend only borrows these bytes.
    Ok(unsafe { Span { data: ptr, len }.as_slice() })
}

fn encode(c: Command<'_, '_>, w: &mut Vec<u64>) {
    let start = w.len();
    w.push(0);
    let opcode = match c {
        Command::WriteBuffer {
            buffer,
            offset,
            data,
        } => {
            buffer.id().put(w);
            offset.put(w);
            bytes(data, w);
            1
        }
        Command::WriteTexture { texture, write } => {
            texture.id().put(w);
            write.destination().put(w);
            write.bytes_per_row().put(w);
            write.mip_level().put(w);
            write.array_layer().put(w);
            bytes(write.data(), w);
            2
        }
        Command::CopyTextureToTexture {
            source,
            source_rect,
            destination,
            destination_rect,
        } => {
            source.id().put(w);
            source_rect.put(w);
            destination.id().put(w);
            destination_rect.put(w);
            3
        }
        Command::BeginRenderPass(d) => {
            d.area().put(w);
            d.color_attachments().count().put(w);
            for a in d.color_attachments() {
                a.target().id().put(w);
                a.load().put(w);
                a.store().put(w);
            }
            d.depth_attachment().is_some().put(w);
            if let Some(d) = d.depth_attachment() {
                d.target().id().put(w);
                d.load().put(w);
                d.store().put(w);
                d.read_only().put(w);
            }
            4
        }
        Command::EndRenderPass => 5,
        Command::SetPipeline(v) => {
            v.id().put(w);
            6
        }
        Command::SetVertexBuffer { buffer, offset } => {
            buffer.id().put(w);
            offset.put(w);
            7
        }
        Command::SetIndexBuffer {
            buffer,
            offset,
            format,
        } => {
            buffer.id().put(w);
            offset.put(w);
            format.put(w);
            8
        }
        Command::SetTexture(v) => {
            v.id().put(w);
            9
        }
        Command::SetSampler(v) => {
            v.id().put(w);
            10
        }
        Command::SetUniforms(v) => {
            v.put(w);
            11
        }
        Command::SetScissor(v) => {
            v.put(w);
            12
        }
        Command::Draw {
            vertex_count,
            first_vertex,
        } => {
            vertex_count.put(w);
            first_vertex.put(w);
            13
        }
        Command::DrawIndexed {
            index_count,
            first_index,
            base_vertex,
        } => {
            index_count.put(w);
            first_index.put(w);
            base_vertex.put(w);
            14
        }
        Command::BlitTexture {
            source,
            source_mip,
            destination,
            destination_mip,
            filter,
        } => {
            source.id().put(w);
            source_mip.put(w);
            destination.id().put(w);
            destination_mip.put(w);
            filter.put(w);
            15
        }
        Command::CopyBufferToBuffer {
            source,
            source_offset,
            destination,
            destination_offset,
            size,
        } => {
            source.id().put(w);
            source_offset.put(w);
            destination.id().put(w);
            destination_offset.put(w);
            size.put(w);
            16
        }
        Command::SetProgrammablePipeline(v) => {
            v.id().put(w);
            17
        }
        Command::SetBindGroup { index, bind_group } => {
            index.put(w);
            bind_group.id().put(w);
            18
        }
        Command::BeginComputePass => 19,
        Command::EndComputePass => 20,
        Command::SetComputePipeline(v) => {
            v.id().put(w);
            21
        }
        Command::Dispatch { x, y, z } => {
            x.put(w);
            y.put(w);
            z.put(w);
            22
        }
        Command::ResourceBarrier(v) => {
            match v {
                ResourceBarrier::Buffer {
                    buffer,
                    before,
                    after,
                } => {
                    w.push(0);
                    buffer.id().put(w);
                    before.put(w);
                    after.put(w);
                }
                ResourceBarrier::Texture {
                    texture,
                    before,
                    after,
                } => {
                    w.push(1);
                    texture.id().put(w);
                    before.put(w);
                    after.put(w);
                }
                ResourceBarrier::TextureMip {
                    texture,
                    mip_level,
                    before,
                    after,
                } => {
                    w.push(2);
                    texture.id().put(w);
                    mip_level.put(w);
                    before.put(w);
                    after.put(w);
                }
            }
            23
        }
        Command::SetVertexBufferSlot {
            slot,
            buffer,
            offset,
        } => {
            slot.put(w);
            buffer.id().put(w);
            offset.put(w);
            24
        }
        Command::SetViewport(v) => {
            v.put(w);
            25
        }
        Command::SetPushConstants {
            stages,
            offset,
            data,
        } => {
            stages.put(w);
            offset.put(w);
            bytes(data, w);
            26
        }
        Command::DrawInstanced {
            vertex_count,
            first_vertex,
            instance_count,
            first_instance,
        } => {
            vertex_count.put(w);
            first_vertex.put(w);
            instance_count.put(w);
            first_instance.put(w);
            27
        }
        Command::DrawIndexedInstanced {
            index_count,
            first_index,
            base_vertex,
            instance_count,
            first_instance,
        } => {
            index_count.put(w);
            first_index.put(w);
            base_vertex.put(w);
            instance_count.put(w);
            first_instance.put(w);
            28
        }
    };
    w[start] = opcode | (((w.len() - start) as u64) << 32);
}

// Scalar commands have no resource lookup or variable-sized descriptors. Keep
// their reads in the consuming loop so the planner can fold the temporary
// command into its match. Resource/attachment decoding stays out of line.
#[inline(always)]
unsafe fn decode_into<'r, 'data>(
    opcode: u32,
    r: &mut Reader<'_>,
    t: &'r ResourceTable,
    out: &mut MaybeUninit<Command<'r, 'data>>,
) -> Result<()> {
    match opcode {
        5 => {
            out.write(Command::EndRenderPass);
        }
        11 => {
            out.write(Command::SetUniforms(r.value(t)?));
        }
        12 => {
            out.write(Command::SetScissor(r.value(t)?));
        }
        13 => {
            out.write(Command::Draw {
                vertex_count: r.value(t)?,
                first_vertex: r.value(t)?,
            });
        }
        14 => {
            out.write(Command::DrawIndexed {
                index_count: r.value(t)?,
                first_index: r.value(t)?,
                base_vertex: r.value(t)?,
            });
        }
        19 => {
            out.write(Command::BeginComputePass);
        }
        20 => {
            out.write(Command::EndComputePass);
        }
        22 => {
            out.write(Command::Dispatch {
                x: r.value(t)?,
                y: r.value(t)?,
                z: r.value(t)?,
            });
        }
        25 => {
            out.write(Command::SetViewport(r.value(t)?));
        }
        27 => {
            out.write(Command::DrawInstanced {
                vertex_count: r.value(t)?,
                first_vertex: r.value(t)?,
                instance_count: r.value(t)?,
                first_instance: r.value(t)?,
            });
        }
        28 => {
            out.write(Command::DrawIndexedInstanced {
                index_count: r.value(t)?,
                first_index: r.value(t)?,
                base_vertex: r.value(t)?,
                instance_count: r.value(t)?,
                first_instance: r.value(t)?,
            });
        }
        // SAFETY: both decoders use the same borrowed-recording contract.
        _ => return unsafe { decode_resource_command(opcode, r, t, out) },
    }
    Ok(())
}

#[inline(never)]
unsafe fn decode_resource_command<'r, 'data>(
    opcode: u32,
    r: &mut Reader<'_>,
    t: &'r ResourceTable,
    out: &mut MaybeUninit<Command<'r, 'data>>,
) -> Result<()> {
    match opcode {
        1 => {
            out.write(Command::WriteBuffer {
                buffer: t.buffer_ref(r.value(t)?)?,
                offset: r.value(t)?,
                data: unsafe { borrowed_bytes(r, t) }?,
            });
        }
        2 => {
            out.write({
                let texture = t.texture_ref(r.value(t)?)?;
                let destination = r.value(t)?;
                let stride = r.value(t)?;
                let mip = r.value(t)?;
                let layer = r.value(t)?;
                let data = unsafe { borrowed_bytes(r, t) }?;
                Command::WriteTexture {
                    texture,
                    write: TextureWrite::new(destination, stride, data)?
                        .with_mip_level(mip)
                        .with_array_layer(layer),
                }
            });
        }
        3 => {
            out.write(Command::CopyTextureToTexture {
                source: t.texture_ref(r.value(t)?)?,
                source_rect: r.value(t)?,
                destination: t.texture_ref(r.value(t)?)?,
                destination_rect: r.value(t)?,
            });
        }
        4 => {
            out.write({
                let area = r.value(t)?;
                let count = usize::get(r, t)?;
                if count == 0 || count > MAX_COLOR_ATTACHMENTS {
                    return Err(Error::InvalidDescriptor);
                }
                let mut desc = RenderPassDesc::new(
                    t,
                    t.texture_ref(r.value(t)?)?,
                    area,
                    r.value(t)?,
                    r.value(t)?,
                )?;
                for _ in 1..count {
                    desc = desc.with_color_attachment(
                        t,
                        t.texture_ref(r.value(t)?)?,
                        r.value(t)?,
                        r.value(t)?,
                    )?;
                }
                if bool::get(r, t)? {
                    desc = desc.with_depth_attachment(
                        t,
                        t.texture_ref(r.value(t)?)?,
                        r.value(t)?,
                        r.value(t)?,
                    )?;
                    if bool::get(r, t)? {
                        desc = desc.with_read_only_depth()?;
                    }
                }
                Command::BeginRenderPass(desc)
            });
        }
        6 => {
            out.write(Command::SetPipeline(t.render_pipeline_ref(r.value(t)?)?));
        }
        7 => {
            out.write(Command::SetVertexBuffer {
                buffer: t.buffer_ref(r.value(t)?)?,
                offset: r.value(t)?,
            });
        }
        8 => {
            out.write(Command::SetIndexBuffer {
                buffer: t.buffer_ref(r.value(t)?)?,
                offset: r.value(t)?,
                format: r.value(t)?,
            });
        }
        9 => {
            out.write(Command::SetTexture(t.texture_ref(r.value(t)?)?));
        }
        10 => {
            out.write(Command::SetSampler(t.sampler_ref(r.value(t)?)?));
        }
        15 => {
            out.write(Command::BlitTexture {
                source: t.texture_ref(r.value(t)?)?,
                source_mip: r.value(t)?,
                destination: t.texture_ref(r.value(t)?)?,
                destination_mip: r.value(t)?,
                filter: r.value(t)?,
            });
        }
        16 => {
            out.write(Command::CopyBufferToBuffer {
                source: t.buffer_ref(r.value(t)?)?,
                source_offset: r.value(t)?,
                destination: t.buffer_ref(r.value(t)?)?,
                destination_offset: r.value(t)?,
                size: r.value(t)?,
            });
        }
        17 => {
            out.write(Command::SetProgrammablePipeline(
                t.programmable_render_pipeline_ref(r.value(t)?)?,
            ));
        }
        18 => {
            out.write(Command::SetBindGroup {
                index: r.value(t)?,
                bind_group: t.bind_group_ref(r.value(t)?)?,
            });
        }
        21 => {
            out.write(Command::SetComputePipeline(
                t.compute_pipeline_ref(r.value(t)?)?,
            ));
        }
        23 => {
            out.write(Command::ResourceBarrier(match r.word()? {
                0 => ResourceBarrier::Buffer {
                    buffer: t.buffer_ref(r.value(t)?)?,
                    before: r.value(t)?,
                    after: r.value(t)?,
                },
                1 => ResourceBarrier::Texture {
                    texture: t.texture_ref(r.value(t)?)?,
                    before: r.value(t)?,
                    after: r.value(t)?,
                },
                2 => ResourceBarrier::TextureMip {
                    texture: t.texture_ref(r.value(t)?)?,
                    mip_level: r.value(t)?,
                    before: r.value(t)?,
                    after: r.value(t)?,
                },
                _ => return Err(Error::InvalidValue),
            }));
        }
        24 => {
            out.write(Command::SetVertexBufferSlot {
                slot: r.value(t)?,
                buffer: t.buffer_ref(r.value(t)?)?,
                offset: r.value(t)?,
            });
        }
        26 => {
            out.write(Command::SetPushConstants {
                stages: r.value(t)?,
                offset: r.value(t)?,
                data: unsafe { borrowed_bytes(r, t) }?,
            });
        }
        _ => return Err(Error::InvalidDescriptor),
    }
    Ok(())
}

impl<'r, 'data> CommandBuffer<'r, 'data> {
    /// Inspect only record headers for capability preflight. This neither
    /// resolves resource descriptors nor decodes/copies command operands.
    pub fn has_compute_commands(&self) -> Result<bool> {
        self.commands.has_compute_commands()
    }

    /// Borrow the recording for one C ABI call. Returns None for recordings
    /// created before `enable_abi_commands`; no hidden conversion is performed.
    pub fn abi_batch(&self) -> Option<Batch> {
        self.commands.batch(self.resources.abi_identity())
    }

    /// Visit commands directly, including foreign ABI recordings, without allocation.
    #[inline]
    pub fn iter_commands(&self) -> CommandIter<'_, 'r, 'data> {
        self.commands.iter(self.resources)
    }

    /// Read borrowed command views, reusing a stack slot for ABI scalar operands.
    #[inline]
    pub fn command_reader(&self) -> CommandReader<'_, 'r, 'data> {
        CommandReader {
            inner: self.commands.iter(self.resources).inner,
            scratch: MaybeUninit::uninit(),
        }
    }

    /// Borrow a plugin caller's command storage without copying or rebuilding it.
    /// `source_table` is the caller identity recorded by resource synchronization.
    ///
    /// # Safety
    /// The words and every upload span referenced by them must remain valid and
    /// immutable for `'data`. Upload spans must refer to initialized readable
    /// allocations. This function validates framing during iteration, but cannot
    /// establish the validity of arbitrary process-local pointers.
    pub unsafe fn from_abi(t: &'r ResourceTable, source_table: u64, batch: Batch) -> Result<Self> {
        if batch.table != source_table {
            return Err(Error::ResourceTableMismatch);
        }
        if batch.count > MAX_COMMANDS
            || batch.words.len > MAX_COMMANDS * MAX_WORDS
            || (batch.words.len > 0
                && (batch.words.data.is_null() || !(batch.words.data as usize).is_multiple_of(8)))
        {
            return Err(Error::InvalidDescriptor);
        }
        Ok(Self {
            resources: t,
            commands: Storage::Borrowed {
                words: unsafe { batch.words.as_slice() },
                count: batch.count,
                decoded: OnceCell::new(),
            },
        })
    }
}
