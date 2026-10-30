use super::super::resource::{BindGroupSlot, BufferSlot, ProgramSlot, TextureSlot};
use super::{
    codec::{Codec, Reader},
    *,
};
use alloc::{rc::Rc, vec::Vec};
use core::cell::Ref;

const MAGIC: u64 = 0x3352_5846_4753; // SGFXR3
const LEGACY_MAGIC: u64 = 0x3252_5846_4753; // SGFXR2

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

struct AcceptedRow {
    generation: u64,
    words: Vec<u64>,
}

/// Encoded copies of rows already accepted by this backend-owned mirror.
/// Every inner vector is bounded by its resource table category's slot count.
pub(super) struct RowCache {
    rows: Vec<Option<AcceptedRow>>,
    #[cfg(test)]
    hits: core::cell::Cell<u64>,
    #[cfg(test)]
    fallback_encodes: core::cell::Cell<u64>,
}

impl RowCache {
    const fn new() -> Self {
        Self {
            rows: Vec::new(),
            #[cfg(test)]
            hits: core::cell::Cell::new(0),
            #[cfg(test)]
            fallback_encodes: core::cell::Cell::new(0),
        }
    }

    /// `None` means this slot has no accepted cache for the generation;
    /// `Some(false)` is an immutable-row mismatch and must be rejected.
    fn compare(&self, index: usize, generation: u64, words: &[u64]) -> Option<bool> {
        let row = self.rows.get(index)?.as_ref()?;
        if row.generation != generation {
            return None;
        }
        let equal = row.words == words;
        #[cfg(test)]
        if equal {
            self.hits.set(self.hits.get() + 1);
        }
        Some(equal)
    }

    fn accept(&mut self, index: usize, generation: u64, words: &[u64]) -> Result<()> {
        let len = index.checked_add(1).ok_or(Error::Overflow)?;
        if self.rows.len() < len {
            self.rows
                .try_reserve(len - self.rows.len())
                .map_err(|_| Error::OutOfMemory)?;
            self.rows.resize_with(len, || None);
        }
        let mut encoded = Vec::new();
        encoded
            .try_reserve_exact(words.len())
            .map_err(|_| Error::OutOfMemory)?;
        encoded.extend_from_slice(words);
        self.rows[index] = Some(AcceptedRow {
            generation,
            words: encoded,
        });
        Ok(())
    }

    #[cfg(test)]
    fn note_fallback_encode(&self) {
        self.fallback_encodes.set(self.fallback_encodes.get() + 1);
    }

    #[cfg(test)]
    fn hits(&self) -> u64 {
        self.hits.get()
    }

    #[cfg(test)]
    fn fallback_encodes(&self) -> u64 {
        self.fallback_encodes.get()
    }
}

/// Bounded consumer-side encoded-row caches, one per generated slot category.
pub(crate) struct AcceptedRows {
    textures: RowCache,
    buffers: RowCache,
    shader_modules: RowCache,
    bind_groups: RowCache,
    compute_pipelines: RowCache,
    programmable_pipelines: RowCache,
}

impl AcceptedRows {
    pub(super) const fn new() -> Self {
        Self {
            textures: RowCache::new(),
            buffers: RowCache::new(),
            shader_modules: RowCache::new(),
            bind_groups: RowCache::new(),
            compute_pipelines: RowCache::new(),
            programmable_pipelines: RowCache::new(),
        }
    }
}

impl ResourceTable {
    /// Active texture and buffer identities replaced by a newer snapshot.
    /// Retire their physical caches before applying the new generations.
    pub fn abi_retired_resources(&self, words: &[u64]) -> Result<(Vec<TextureId>, Vec<BufferId>)> {
        let mut r = Reader { words };
        if !matches!(r.word()?, MAGIC | LEGACY_MAGIC) {
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

    /// Program identities replaced by newer metadata generations. Retire
    /// pipeline caches before shader caches, and all caches before importing.
    /// R2 clients have immutable append-only program rows and cannot retire them.
    pub fn abi_retired_programs(
        &self,
        words: &[u64],
    ) -> Result<(
        Vec<ShaderModuleId>,
        Vec<ComputePipelineId>,
        Vec<ProgrammableRenderPipelineId>,
    )> {
        let mut r = Reader { words };
        let magic = r.word()?;
        if magic == LEGACY_MAGIC {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }
        if magic != MAGIC {
            return Err(Error::InvalidDescriptor);
        }
        r.take(2)?;
        for maximum in [
            MAX_TEXTURES,
            MAX_BUFFERS,
            MAX_SAMPLERS,
            MAX_RENDER_PIPELINES,
        ] {
            skip_rows(&mut r, maximum, self)?;
        }
        let mut shaders = Vec::new();
        scan_retired(
            &mut r,
            &self.shader_modules.borrow(),
            MAX_SHADER_MODULES,
            self,
            |index, generation| {
                shaders.push(ShaderModuleId {
                    owner: self.id,
                    index,
                    generation,
                });
            },
        )?;
        skip_rows(&mut r, MAX_BIND_GROUP_DEFINITIONS, self)?;
        let mut compute = Vec::new();
        scan_retired(
            &mut r,
            &self.compute_pipelines.borrow(),
            MAX_COMPUTE_PIPELINES,
            self,
            |index, generation| {
                compute.push(ComputePipelineId {
                    owner: self.id,
                    index,
                    generation,
                });
            },
        )?;
        let mut programmable = Vec::new();
        scan_retired(
            &mut r,
            &self.programmable_pipelines.borrow(),
            MAX_PROGRAMMABLE_RENDER_PIPELINES,
            self,
            |index, generation| {
                programmable.push(ProgrammableRenderPipelineId {
                    owner: self.id,
                    index,
                    generation,
                });
            },
        )?;
        r.end()?;
        Ok((shaders, compute, programmable))
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
            let mut snapshot = self.abi.snapshot.borrow_mut();
            let out = &mut snapshot.1;
            out.clear();
            out.try_reserve(64).map_err(|_| Error::OutOfMemory)?;
            out.extend_from_slice(&[MAGIC, self.abi_identity(), self.abi_revision()]);
            rows(&self.textures.borrow(), out);
            rows(&self.buffers.borrow(), out);
            rows(&self.samplers.borrow(), out);
            rows(&self.pipelines.borrow(), out);
            rows(&self.shader_modules.borrow(), out);
            rows(&self.bind_groups.borrow(), out);
            rows(&self.compute_pipelines.borrow(), out);
            rows(&self.programmable_pipelines.borrow(), out);
            snapshot.0 = self.abi_revision();
        }
        Ok(Ref::map(self.abi.snapshot.borrow(), |v| v.1.as_slice()))
    }

    /// Apply a caller's R2 or R3 metadata snapshot to a backend-owned mirror. Existing
    /// immutable definitions remain cached; only changed definitions allocate.
    /// R3 program rows and texture, buffer and bind-group slots carry generations, so retired IDs
    /// cannot alias replacements.
    ///
    /// On error the mirror must be discarded, since a valid prefix may have
    /// been applied. The resource table is never shared across the library ABI.
    pub fn sync_abi_snapshot(&self, words: &[u64]) -> Result<(u64, u64)> {
        let mut r = Reader { words };
        let magic = r.word()?;
        if !matches!(magic, MAGIC | LEGACY_MAGIC) {
            return Err(Error::InvalidDescriptor);
        }
        self.abi.legacy_program_ids.set(magic == LEGACY_MAGIC);
        let source = r.word()?;
        let revision = r.word()?;
        if source == 0 || revision == 0 {
            return Err(Error::InvalidDescriptor);
        }
        let mut accepted = self.abi.accepted_rows.borrow_mut();
        sync_slots(
            &mut r,
            &mut self.textures.borrow_mut(),
            MAX_TEXTURES,
            self,
            &mut accepted.textures,
        )?;
        sync_slots(
            &mut r,
            &mut self.buffers.borrow_mut(),
            MAX_BUFFERS,
            self,
            &mut accepted.buffers,
        )?;
        let existing = self.samplers.borrow().len();
        self.append_abi_rows::<SamplerDesc>(&mut r, existing, MAX_SAMPLERS, |d| {
            self.define_sampler(d).map(|_| ())
        })?;
        let existing = self.pipelines.borrow().len();
        self.append_abi_rows::<RenderPipelineDesc>(&mut r, existing, MAX_RENDER_PIPELINES, |d| {
            self.define_render_pipeline(d).map(|_| ())
        })?;
        sync_program_slots(
            &mut r,
            &mut self.shader_modules.borrow_mut(),
            MAX_SHADER_MODULES,
            self,
            &mut accepted.shader_modules,
        )?;
        // Historical descriptors can reference retired resources. Their IDs
        // are checked when used, without reviving old generations on import.
        sync_slots(
            &mut r,
            &mut self.bind_groups.borrow_mut(),
            MAX_BIND_GROUP_DEFINITIONS,
            self,
            &mut accepted.bind_groups,
        )?;
        sync_program_slots(
            &mut r,
            &mut self.compute_pipelines.borrow_mut(),
            MAX_COMPUTE_PIPELINES,
            self,
            &mut accepted.compute_pipelines,
        )?;
        sync_program_slots(
            &mut r,
            &mut self.programmable_pipelines.borrow_mut(),
            MAX_PROGRAMMABLE_RENDER_PIPELINES,
            self,
            &mut accepted.programmable_pipelines,
        )?;
        r.end()?;
        // A cached unchanged pipeline row may not outlive the shader generation
        // it names. Check this even when sync_slots reused its Rc metadata.
        for slot in self.compute_pipelines.borrow().iter() {
            if let Some(pipeline) = &slot.descriptor {
                self.shader_module_ref(pipeline.shader().module())?;
            }
        }
        for slot in self.programmable_pipelines.borrow().iter() {
            if let Some(pipeline) = &slot.descriptor {
                self.shader_module_ref(pipeline.vertex().module())?;
                self.shader_module_ref(pipeline.fragment().module())?;
            }
        }
        rebuild_program_free_list(
            &mut self.shader_modules.borrow_mut(),
            &self.free_shader_module,
        );
        rebuild_program_free_list(
            &mut self.compute_pipelines.borrow_mut(),
            &self.free_compute_pipeline,
        );
        rebuild_program_free_list(
            &mut self.programmable_pipelines.borrow_mut(),
            &self.free_programmable_pipeline,
        );
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
    cache: &mut RowCache,
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
            if generation == old.generation() && old.active() {
                match cache.compare(index, generation, words) {
                    Some(true) => continue,
                    Some(false) => return Err(Error::InvalidDescriptor),
                    None => {
                        // A cache populated by an older implementation or
                        // lost after mirror mutation is re-established once.
                        #[cfg(test)]
                        cache.note_fallback_encode();
                        let mut expected = Vec::new();
                        old.put(&mut expected);
                        if expected != words {
                            return Err(Error::InvalidDescriptor);
                        }
                        cache.accept(index, generation, words)?;
                        continue;
                    }
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
        cache.accept(index, generation, words)?;
    }
    Ok(())
}

fn skip_rows(r: &mut Reader<'_>, maximum: usize, table: &ResourceTable) -> Result<()> {
    let count = usize::get(r, table)?;
    if count > maximum {
        return Err(Error::ResourceLimitExceeded);
    }
    for _ in 0..count {
        let len = usize::get(r, table)?;
        r.take(len)?;
    }
    Ok(())
}

impl<T: Codec> Codec for ProgramSlot<T> {
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
impl<T: Codec> Slot for ProgramSlot<T> {
    fn generation(&self) -> u64 {
        self.generation
    }
    fn active(&self) -> bool {
        self.descriptor.is_some()
    }
}

fn sync_program_slots<T: Codec + PartialEq>(
    r: &mut Reader<'_>,
    current: &mut Vec<ProgramSlot<T>>,
    maximum: usize,
    table: &ResourceTable,
    cache: &mut RowCache,
) -> Result<()> {
    if !table.abi.legacy_program_ids.get() {
        return sync_slots(r, current, maximum, table, cache);
    }
    let count = usize::get(r, table)?;
    if count < current.len() || count > maximum.min(256) {
        return Err(Error::ResourceLimitExceeded);
    }
    for index in 0..count {
        let len = usize::get(r, table)?;
        let mut fields = Reader {
            words: r.take(len)?,
        };
        let descriptor = T::get(&mut fields, table)?;
        fields.end()?;
        if let Some(old) = current.get(index) {
            if old.generation != 0 || old.descriptor.as_ref() != Some(&descriptor) {
                return Err(Error::InvalidDescriptor);
            }
        } else {
            current.try_reserve(1).map_err(|_| Error::OutOfMemory)?;
            current.push(ProgramSlot {
                descriptor: Some(descriptor),
                generation: 0,
                next_free: None,
            });
        }
    }
    Ok(())
}

fn rebuild_program_free_list<T>(
    slots: &mut [ProgramSlot<T>],
    free: &core::cell::Cell<Option<usize>>,
) {
    let mut head = None;
    for (index, slot) in slots.iter_mut().enumerate().rev() {
        slot.next_free = None;
        if slot.descriptor.is_none() {
            slot.next_free = head;
            head = Some(index);
        }
    }
    free.set(head);
}

#[cfg(test)]
mod capacity_tests {
    use super::*;

    fn cache_counters(table: &ResourceTable) -> (u64, u64) {
        let caches = table.abi.accepted_rows.borrow();
        let hits = [
            &caches.textures,
            &caches.buffers,
            &caches.shader_modules,
            &caches.bind_groups,
            &caches.compute_pipelines,
            &caches.programmable_pipelines,
        ]
        .into_iter()
        .map(RowCache::hits)
        .sum();
        let fallbacks = [
            &caches.textures,
            &caches.buffers,
            &caches.shader_modules,
            &caches.bind_groups,
            &caches.compute_pipelines,
            &caches.programmable_pipelines,
        ]
        .into_iter()
        .map(RowCache::fallback_encodes)
        .sum();
        (hits, fallbacks)
    }

    #[test]
    fn shared_shader_capacity_roundtrips_and_recycles_beyond_256_slots() {
        let producer = ResourceTable::new();
        let consumer = ResourceTable::new();
        let descriptor =
            ShaderModuleDesc::wgsl("@compute @workgroup_size(1) fn main() {}".into()).unwrap();
        let mut ids = Vec::new();
        for _ in 0..MAX_SHADER_MODULES {
            ids.push(
                producer
                    .define_shader_module(descriptor.clone())
                    .unwrap()
                    .id(),
            );
        }
        assert_eq!(
            producer
                .define_shader_module(descriptor.clone())
                .unwrap_err(),
            Error::ResourceLimitExceeded
        );
        consumer
            .sync_abi_snapshot(&producer.abi_snapshot().unwrap())
            .unwrap();
        assert!(
            consumer
                .abi_shader_module((MAX_SHADER_MODULES - 1) as u32)
                .is_ok()
        );
        let old = consumer.abi_shader_module(300).unwrap().id();
        producer.release_shader_module(ids[300]).unwrap();
        let replacement = producer.define_shader_module(descriptor).unwrap().id();
        assert_eq!(replacement.index, 300);
        let snapshot = producer.abi_snapshot().unwrap();
        let (retired, _, _) = consumer.abi_retired_programs(&snapshot).unwrap();
        assert!(retired.contains(&old));
        consumer.sync_abi_snapshot(&snapshot).unwrap();
        assert!(consumer.shader_module_ref(old).is_err());
        assert!(consumer.abi_shader_module(300).is_ok());
    }

    #[test]
    fn accepted_rows_use_cached_bytes_and_reject_same_generation_tampering() {
        let producer = ResourceTable::new();
        producer
            .define_shader_module(
                ShaderModuleDesc::wgsl("@compute @workgroup_size(1) fn main() {}".into()).unwrap(),
            )
            .unwrap();
        let snapshot = producer.abi_snapshot().unwrap().to_vec();
        let consumer = ResourceTable::new();

        consumer.sync_abi_snapshot(&snapshot).unwrap();
        assert_eq!(cache_counters(&consumer), (0, 0));
        consumer.sync_abi_snapshot(&snapshot).unwrap();
        assert_eq!(cache_counters(&consumer), (1, 0));

        let mut tampered = snapshot;
        // Four empty categories precede the shader count, row length and
        // generation. This bit is inside the immutable WGSL source payload.
        tampered[11] ^= 1;
        assert_eq!(
            consumer.sync_abi_snapshot(&tampered),
            Err(Error::InvalidDescriptor)
        );
        assert_eq!(cache_counters(&consumer).1, 0);
    }

    #[test]
    fn generation_reuse_replaces_cached_row_before_fast_path() {
        let producer = ResourceTable::new();
        let original = producer
            .define_shader_module(
                ShaderModuleDesc::wgsl("@compute @workgroup_size(1) fn main() {}".into()).unwrap(),
            )
            .unwrap()
            .id();
        let consumer = ResourceTable::new();
        consumer
            .sync_abi_snapshot(&producer.abi_snapshot().unwrap())
            .unwrap();
        let old = consumer.abi_shader_module(0).unwrap().id();

        producer.release_shader_module(original).unwrap();
        consumer
            .sync_abi_snapshot(&producer.abi_snapshot().unwrap())
            .unwrap();
        assert!(consumer.shader_module_ref(old).is_err());

        producer
            .define_shader_module(
                ShaderModuleDesc::wgsl("@compute @workgroup_size(2) fn main() {}".into()).unwrap(),
            )
            .unwrap();
        let replacement = producer.abi_snapshot().unwrap().to_vec();
        consumer.sync_abi_snapshot(&replacement).unwrap();
        let current = consumer.abi_shader_module(0).unwrap().id();
        assert_ne!(current, old);
        assert!(consumer.shader_module_ref(current).is_ok());

        consumer.sync_abi_snapshot(&replacement).unwrap();
        assert_eq!(cache_counters(&consumer), (1, 0));
    }
}
