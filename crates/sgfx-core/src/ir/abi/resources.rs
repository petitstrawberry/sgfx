use super::super::resource::BufferSlot;
use super::{
    codec::{Codec, Reader},
    *,
};
use alloc::{rc::Rc, vec::Vec};
use core::cell::Ref;

const MAGIC: u64 = 0x3152_5846_4753; // SGFXR1

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
    /// Old active buffer identities retired by a newer snapshot. Backends must
    /// retire their corresponding physical caches before applying the snapshot.
    pub fn abi_retired_buffers(&self, words: &[u64]) -> Result<Vec<BufferId>> {
        let mut r = Reader { words };
        if r.word()? != MAGIC {
            return Err(Error::InvalidDescriptor);
        }
        r.take(2)?;
        let textures = usize::get(&mut r, self)?;
        if textures > MAX_TEXTURES {
            return Err(Error::ResourceLimitExceeded);
        }
        for _ in 0..textures {
            let n = usize::get(&mut r, self)?;
            r.take(n)?;
        }
        let count = usize::get(&mut r, self)?;
        if count > MAX_BUFFERS {
            return Err(Error::ResourceLimitExceeded);
        }
        let current = self.buffers.borrow();
        let mut retired = Vec::new();
        for index in 0..count {
            let generation = r.value::<u64>(self)?;
            let _ = r.value::<Option<BufferDesc>>(self)?;
            if let Some(slot) = current.get(index)
                && slot.descriptor.is_some()
                && generation != slot.generation
            {
                retired.push(BufferId {
                    owner: self.id,
                    index,
                    generation: slot.generation,
                });
            }
        }
        Ok(retired)
    }

    /// Cached metadata snapshot. Repeated calls without mutations allocate and
    /// copy nothing. Large upload payloads never appear in this snapshot.
    pub fn abi_snapshot(&self) -> Result<Ref<'_, [u64]>> {
        if self.abi.snapshot.borrow().0 != self.abi_revision() {
            let mut out = Vec::new();
            out.try_reserve(64).map_err(|_| Error::OutOfMemory)?;
            out.extend_from_slice(&[MAGIC, self.abi_identity(), self.abi_revision()]);
            rows(&self.textures.borrow(), &mut out);
            let buffers = self.buffers.borrow();
            buffers.len().put(&mut out);
            for slot in buffers.iter() {
                slot.generation.put(&mut out);
                slot.descriptor.put(&mut out);
            }
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

    /// Apply a caller's v1 metadata snapshot to a backend-owned mirror. Existing
    /// immutable definitions remain cached; only appended definitions allocate.
    /// Buffer slots carry generations, so retired IDs cannot alias new buffers.
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
        let existing = self.textures.borrow().len();
        self.append_abi_rows::<TextureDesc>(&mut r, existing, MAX_TEXTURES, |d| {
            self.define_texture(d).map(|_| ())
        })?;
        let count = usize::get(&mut r, self)?;
        if count > MAX_BUFFERS || count < self.buffers.borrow().len() {
            return Err(Error::ResourceLimitExceeded);
        }
        for index in 0..count {
            let generation = r.value::<u64>(self)?;
            let descriptor = r.value::<Option<BufferDesc>>(self)?;
            let mut buffers = self.buffers.borrow_mut();
            if let Some(slot) = buffers.get_mut(index) {
                if generation < slot.generation
                    || (generation == slot.generation
                        && slot.descriptor.is_some()
                        && descriptor != slot.descriptor)
                {
                    return Err(Error::InvalidDescriptor);
                }
                slot.generation = generation;
                slot.descriptor = descriptor;
                slot.next_free = None;
            } else {
                buffers.try_reserve(1).map_err(|_| Error::OutOfMemory)?;
                buffers.push(BufferSlot {
                    generation,
                    descriptor,
                    next_free: None,
                });
            }
        }
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
        let existing = self.bind_groups.borrow().len();
        self.append_abi_rows::<BindGroupDesc>(&mut r, existing, MAX_BIND_GROUP_DEFINITIONS, |d| {
            // Immutable historical groups can contain since-retired buffer IDs.
            // Keep their slots, without reviving those buffers or rejecting an
            // otherwise valid table because an unused old group remains in it.
            self.push(&self.bind_groups, Rc::new(d), MAX_BIND_GROUP_DEFINITIONS)
                .map(|_| ())
        })?;
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
