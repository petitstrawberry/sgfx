use super::super::resource::{BindGroupSlot, BufferSlot, TextureSlot};
use super::{
    codec::{Codec, Reader},
    *,
};
use alloc::{rc::Rc, vec::Vec};
use core::cell::Ref;

const MAGIC: u64 = 0x3252_5846_4753; // SGFXR2

fn rows<T: Codec>(items: &[T], out: &mut Vec<u64>) {
    items.len().put(out);
    for item in items {
        let start = out.len();
        out.push(0);
        item.put(out);
        out[start] = (out.len() - start - 1) as u64;
    }
}
impl<T: Codec> Codec for Rc<T> {
    fn put(&self, w: &mut Vec<u64>) {
        self.as_ref().put(w);
    }
    fn get(r: &mut Reader<'_>, t: &ResourceTable) -> Result<Self> {
        Ok(Rc::new(r.value(t)?))
    }
}

impl ResourceTable {
    /// Active texture and buffer identities replaced by a newer snapshot.
    /// Retire their physical caches before applying the new generations.
    pub fn abi_retired_resources(&self, words: &[u64]) -> Result<(Vec<TextureId>, Vec<BufferId>)> {
        let mut r = Reader { words };
        if r.word()? != MAGIC {
            return Err(Error::InvalidDescriptor);
        }
        r.take(2)?;
        let mut textures = Vec::new();
        let mut buffers = Vec::new();
        scan_retired(
            &mut r,
            &self.textures.borrow(),
            MAX_TEXTURES,
            self,
            |index, generation| {
                textures.push(TextureId {
                    owner: self.id,
                    index,
                    generation,
                })
            },
        )?;
        scan_retired(
            &mut r,
            &self.buffers.borrow(),
            MAX_BUFFERS,
            self,
            |index, generation| {
                buffers.push(BufferId {
                    owner: self.id,
                    index,
                    generation,
                })
            },
        )?;
        Ok((textures, buffers))
    }

    /// Buffer-only compatibility view of the retirement scan.
    pub fn abi_retired_buffers(&self, words: &[u64]) -> Result<Vec<BufferId>> {
        self.abi_retired_resources(words)
            .map(|(_, buffers)| buffers)
    }

    /// Cached metadata snapshot. Repeated calls without mutations allocate and
    /// copy nothing. Large upload payloads never appear in this snapshot.
    pub fn abi_snapshot(&self) -> Result<Ref<'_, [u64]>> {
        if self.abi.snapshot.borrow().0 != self.abi_revision() {
            let mut out = Vec::new();
            out.try_reserve(64).map_err(|_| Error::OutOfMemory)?;
            out.extend_from_slice(&[MAGIC, self.abi_identity(), self.abi_revision()]);
            rows(&self.textures.borrow(), &mut out);
            rows(&self.buffers.borrow(), &mut out);
            rows(&self.samplers.borrow(), &mut out);
            rows(&self.pipelines.borrow(), &mut out);
            rows(&self.shader_modules.borrow(), &mut out);
            rows(&self.bind_groups.borrow(), &mut out);
            rows(&self.compute_pipelines.borrow(), &mut out);
            rows(&self.programmable_pipelines.borrow(), &mut out);
            *self.abi.snapshot.borrow_mut() = (self.abi_revision(), out);
        }
        Ok(Ref::map(self.abi.snapshot.borrow(), |v| v.1.as_slice()))
    }

    /// Apply a caller's v2 metadata snapshot to a backend-owned mirror. Existing
    /// immutable definitions remain cached; only changed definitions allocate.
    /// Texture, buffer and bind-group slots carry generations, so retired IDs
    /// cannot alias replacements.
    ///
    /// On error the mirror must be discarded, since a valid prefix may have
    /// been applied. The resource table is never shared across the library ABI.
    pub fn sync_abi_snapshot(&self, words: &[u64]) -> Result<(u64, u64)> {
        let mut r = Reader { words };
        if r.word()? != MAGIC {
            return Err(Error::InvalidDescriptor);
        }
        let source = r.word()?;
        let revision = r.word()?;
        if source == 0 || revision == 0 {
            return Err(Error::InvalidDescriptor);
        }
        sync_slots(&mut r, &mut self.textures.borrow_mut(), MAX_TEXTURES, self)?;
        sync_slots(&mut r, &mut self.buffers.borrow_mut(), MAX_BUFFERS, self)?;
        let existing = self.samplers.borrow().len();
        self.append_abi_rows::<SamplerDesc>(&mut r, existing, MAX_SAMPLERS, |d| {
            self.define_sampler(d).map(|_| ())
        })?;
        let existing = self.pipelines.borrow().len();
        self.append_abi_rows::<RenderPipelineDesc>(&mut r, existing, MAX_RENDER_PIPELINES, |d| {
            self.define_render_pipeline(d).map(|_| ())
        })?;
        let existing = self.shader_modules.borrow().len();
        self.append_abi_rows::<ShaderModuleDesc>(&mut r, existing, 256, |d| {
            self.define_shader_module(d).map(|_| ())
        })?;
        // Historical descriptors can reference retired resources. Their IDs
        // are checked when used, without reviving old generations on import.
        sync_slots(
            &mut r,
            &mut self.bind_groups.borrow_mut(),
            MAX_BIND_GROUP_DEFINITIONS,
            self,
        )?;
        let existing = self.compute_pipelines.borrow().len();
        self.append_abi_rows::<ComputePipelineDesc>(&mut r, existing, 256, |d| {
            self.define_compute_pipeline(d).map(|_| ())
        })?;
        let existing = self.programmable_pipelines.borrow().len();
        self.append_abi_rows::<ProgrammableRenderPipelineDesc>(&mut r, existing, 256, |d| {
            self.define_programmable_render_pipeline(d).map(|_| ())
        })?;
        r.end()?;
        Ok((source, revision))
    }

    fn append_abi_rows<T: Codec>(
        &self,
        r: &mut Reader<'_>,
        existing: usize,
        maximum: usize,
        mut append: impl FnMut(T) -> Result<()>,
    ) -> Result<()> {
        let count = usize::get(r, self)?;
        if count < existing || count > maximum {
            return Err(Error::ResourceLimitExceeded);
        }
        for index in 0..count {
            let len = usize::get(r, self)?;
            let mut fields = Reader {
                words: r.take(len)?,
            };
            if index >= existing {
                let value = T::get(&mut fields, self)?;
                fields.end()?;
                append(value)?;
            }
        }
        Ok(())
    }
}

trait Slot: Codec {
    fn generation(&self) -> u64;
    fn active(&self) -> bool;
}
macro_rules! slot {
    ($ty:ty) => {
        impl Codec for $ty {
            fn put(&self, words: &mut Vec<u64>) {
                self.generation.put(words);
                self.descriptor.put(words);
            }
            fn get(r: &mut Reader<'_>, t: &ResourceTable) -> Result<Self> {
                Ok(Self {
                    generation: r.value(t)?,
                    descriptor: r.value(t)?,
                    next_free: None,
                })
            }
        }
        impl Slot for $ty {
            fn generation(&self) -> u64 {
                self.generation
            }
            fn active(&self) -> bool {
                self.descriptor.is_some()
            }
        }
    };
}
slot!(TextureSlot);
slot!(BufferSlot);
slot!(BindGroupSlot);

fn scan_retired<T: Slot>(
    r: &mut Reader<'_>,
    current: &[T],
    maximum: usize,
    table: &ResourceTable,
    mut retire: impl FnMut(usize, u64),
) -> Result<()> {
    let count = usize::get(r, table)?;
    if count < current.len() || count > maximum {
        return Err(Error::ResourceLimitExceeded);
    }
    for index in 0..count {
        let len = usize::get(r, table)?;
        let mut fields = Reader {
            words: r.take(len)?,
        };
        let generation = u64::get(&mut fields, table)?;
        if let Some(old) = current.get(index) {
            if generation < old.generation() {
                return Err(Error::InvalidDescriptor);
            }
            if old.active() && generation != old.generation() {
                retire(index, old.generation());
            }
        }
    }
    Ok(())
}

fn sync_slots<T: Slot>(
    r: &mut Reader<'_>,
    current: &mut Vec<T>,
    maximum: usize,
    table: &ResourceTable,
) -> Result<()> {
    let count = usize::get(r, table)?;
    if count < current.len() || count > maximum {
        return Err(Error::ResourceLimitExceeded);
    }
    for index in 0..count {
        let len = usize::get(r, table)?;
        let words = r.take(len)?;
        let generation = *words.first().ok_or(Error::InvalidDescriptor)?;
        if let Some(old) = current.get(index) {
            if generation < old.generation() {
                return Err(Error::InvalidDescriptor);
            }
            if generation == old.generation() {
                // Definitions are immutable within a generation. Comparing
                // scalar metadata avoids recreating shared bind-group owners.
                let mut expected = Vec::new();
                old.put(&mut expected);
                if expected == words {
                    continue;
                }
                if old.active() {
                    return Err(Error::InvalidDescriptor);
                }
            }
        }
        let mut fields = Reader { words };
        let descriptor = T::get(&mut fields, table)?;
        fields.end()?;
        if let Some(old) = current.get_mut(index) {
            *old = descriptor;
        } else {
            current.try_reserve(1).map_err(|_| Error::OutOfMemory)?;
            current.push(descriptor);
        }
    }
    Ok(())
}
