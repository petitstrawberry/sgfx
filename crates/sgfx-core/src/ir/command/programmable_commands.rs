//! Programmable pass validation and shader-write dependencies.
use super::*;
use crate::ir::{
    programmable::{binding_writes, resource_alias},
    *,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum PendingWrite {
    Buffer(BufferId),
    Texture(TextureId),
}
#[derive(Clone, Copy)]
pub(super) enum AnnouncedAccess {
    Buffer(BufferId, BufferAccess),
    Texture(TextureId, TextureAccess),
}
impl AnnouncedAccess {
    fn resource(self) -> PendingWrite {
        match self {
            Self::Buffer(id, _) => PendingWrite::Buffer(id),
            Self::Texture(id, _) => PendingWrite::Texture(id),
        }
    }
}
struct GroupAccesses {
    writes: Vec<PendingWrite>,
    used: Vec<PendingWrite>,
}
impl<'r, 'data> CommandEncoder<'r, 'data> {
    /// Copy a non-zero, four-byte-aligned range outside passes.
    pub fn copy_buffer_to_buffer(
        &mut self,
        source: BufferRef<'r>,
        source_offset: u64,
        destination: BufferRef<'r>,
        destination_offset: u64,
        size: u64,
    ) -> Result<()> {
        self.ensure_outside_pass()?;
        let src = self.resources.buffer(source)?;
        let dst = self.resources.buffer(destination)?;
        Self::require_buffer_usage(src.usage(), BufferUsage::COPY_SRC)?;
        Self::require_buffer_usage(dst.usage(), BufferUsage::COPY_DST)?;
        if size == 0
            || !source_offset.is_multiple_of(4)
            || !destination_offset.is_multiple_of(4)
            || !size.is_multiple_of(4)
        {
            return Err(Error::InvalidValue);
        }
        Self::validate_byte_range(source_offset, size, src.size())?;
        Self::validate_byte_range(destination_offset, size, dst.size())?;
        // Whole-resource copy aliasing is not portable across WebGPU implementations.
        if source == destination {
            return Err(Error::ResourceAccessConflict);
        }
        self.check_buffer_access(source, BufferAccess::CopySource)?;
        self.check_buffer_access(destination, BufferAccess::CopyDestination)?;
        self.push(Command::CopyBufferToBuffer {
            source,
            source_offset,
            destination,
            destination_offset,
            size,
        })
    }
    /// Record a whole-resource execution and memory dependency outside passes.
    ///
    /// Storage writes must be followed by this command before another access.
    /// Access flags are checked against resource usage. Backends validate and
    /// implement the dependency; WGPU's ordered passes implement it implicitly.
    pub fn resource_barrier(&mut self, barrier: ResourceBarrier<'r>) -> Result<()> {
        self.ensure_outside_pass()?;
        let announced = match barrier {
            ResourceBarrier::Buffer {
                buffer,
                before,
                after,
            } => {
                let desc = self.resources.buffer(buffer)?;
                Self::require_buffer_usage(desc.usage(), before.usage())?;
                Self::require_buffer_usage(desc.usage(), after.usage())?;
                let key = PendingWrite::Buffer(buffer.id());
                if self.pending_writes.contains(&key) && before != BufferAccess::StorageReadWrite {
                    return Err(Error::InvalidResourceAccess);
                }
                if self.announced_accesses.iter().any(|entry| matches!(entry, AnnouncedAccess::Buffer(id, access) if *id == buffer.id() && *access != before)) { return Err(Error::InvalidResourceAccess); }
                AnnouncedAccess::Buffer(buffer.id(), after)
            }
            ResourceBarrier::Texture {
                texture,
                before,
                after,
            } => {
                let desc = self.resources.texture(texture)?;
                Self::require_texture_usage(desc.usage(), before.usage())?;
                Self::require_texture_usage(desc.usage(), after.usage())?;
                let key = PendingWrite::Texture(texture.id());
                if self.pending_writes.contains(&key) && before != TextureAccess::StorageWrite {
                    return Err(Error::InvalidResourceAccess);
                }
                if self.announced_accesses.iter().any(|entry| matches!(entry, AnnouncedAccess::Texture(id, access) if *id == texture.id() && *access != before)) { return Err(Error::InvalidResourceAccess); }
                AnnouncedAccess::Texture(texture.id(), after)
            }
        };
        self.announced_accesses
            .try_reserve(1)
            .map_err(|_| Error::OutOfMemory)?;
        self.push(Command::ResourceBarrier(barrier))?;
        let key = announced.resource();
        self.pending_writes.retain(|pending| *pending != key);
        self.consume_access(key);
        self.announced_accesses.push(announced);
        Ok(())
    }
    /// Begin a compute pass. A dropped pass remains open and prevents finish.
    pub fn begin_compute_pass<'encoder>(
        &'encoder mut self,
    ) -> Result<ComputePassEncoder<'encoder, 'r, 'data>> {
        self.ensure_outside_pass()?;
        self.reserve_pass_begin()?;
        self.push(Command::BeginComputePass)?;
        self.pass_open = true;
        Ok(ComputePassEncoder {
            encoder: self,
            pipeline: None,
            bind_groups: [None; MAX_BIND_GROUPS],
        })
    }
    pub(super) fn check_buffer_access(
        &self,
        buffer: BufferRef<'_>,
        access: BufferAccess,
    ) -> Result<()> {
        if self
            .pending_writes
            .contains(&PendingWrite::Buffer(buffer.id()))
        {
            return Err(Error::MissingBarrier);
        }
        if self.announced_accesses.iter().any(|entry| matches!(entry, AnnouncedAccess::Buffer(id, expected) if *id == buffer.id() && *expected != access)) { return Err(Error::InvalidResourceAccess); }
        Ok(())
    }
    pub(super) fn check_texture_access(
        &self,
        texture: TextureRef<'_>,
        access: TextureAccess,
    ) -> Result<()> {
        if self
            .pending_writes
            .contains(&PendingWrite::Texture(texture.id()))
        {
            return Err(Error::MissingBarrier);
        }
        if self.announced_accesses.iter().any(|entry| matches!(entry, AnnouncedAccess::Texture(id, expected) if *id == texture.id() && *expected != access)) { return Err(Error::InvalidResourceAccess); }
        Ok(())
    }
    pub(super) fn consume_access(&mut self, resource: PendingWrite) {
        self.announced_accesses
            .retain(|entry| entry.resource() != resource);
    }
    fn validate_groups(
        &self,
        layout: &PipelineLayoutDesc,
        groups: &[Option<BindGroupRef<'r>>; MAX_BIND_GROUPS],
        attachments: &[TextureRef<'r>],
        vertex: Option<BufferRef<'r>>,
        index: Option<BufferRef<'r>>,
    ) -> Result<GroupAccesses> {
        let mut uses = Vec::new();
        for (slot, expected) in layout.bind_groups().iter().enumerate() {
            if expected.entries().is_empty() && groups[slot].is_none() {
                continue;
            }
            let group = self
                .resources
                .bind_group(groups[slot].ok_or(Error::BindGroupNotSet)?)?;
            if group.layout() != expected {
                return Err(Error::BindingLayoutMismatch);
            }
            for (entry, entry_layout) in group.entries().iter().zip(expected.entries()) {
                let resource = entry.resource();
                let writes = binding_writes(entry_layout.ty());
                match resource {
                    BindingResource::Buffer { buffer, .. } => {
                        let reference = self.resources.buffer_ref(buffer)?;
                        let access = match entry_layout.ty() {
                            BindingType::UniformBuffer => BufferAccess::Uniform,
                            BindingType::StorageBuffer { read_only: true } => {
                                BufferAccess::StorageRead
                            }
                            _ => BufferAccess::StorageReadWrite,
                        };
                        self.check_buffer_access(reference, access)?;
                        if writes && (vertex == Some(reference) || index == Some(reference)) {
                            return Err(Error::ResourceAccessConflict);
                        }
                    }
                    BindingResource::Texture(texture) => {
                        let reference = self.resources.texture_ref(texture)?;
                        self.check_texture_access(
                            reference,
                            if writes {
                                TextureAccess::StorageWrite
                            } else {
                                TextureAccess::Sampled
                            },
                        )?;
                        if attachments.contains(&reference) {
                            return Err(Error::AttachmentFeedback);
                        }
                    }
                    BindingResource::Sampler(_) => {}
                }
                if uses.iter().any(|(other, other_writes)| {
                    resource_alias(resource, *other) && (writes || *other_writes)
                }) {
                    return Err(Error::ResourceAccessConflict);
                }
                uses.try_reserve(1).map_err(|_| Error::OutOfMemory)?;
                uses.push((resource, writes));
            }
        }
        let mut pending = Vec::new();
        let mut used = Vec::new();
        pending
            .try_reserve(uses.len())
            .map_err(|_| Error::OutOfMemory)?;
        used.try_reserve(uses.len() + 2)
            .map_err(|_| Error::OutOfMemory)?;
        for (resource, writes) in uses {
            let key = match resource {
                BindingResource::Buffer { buffer, .. } => PendingWrite::Buffer(buffer),
                BindingResource::Texture(texture) => PendingWrite::Texture(texture),
                BindingResource::Sampler(_) => continue,
            };
            used.push(key);
            if writes {
                pending.push(key);
            }
        }
        if let Some(vertex) = vertex {
            used.push(PendingWrite::Buffer(vertex.id()));
        }
        if let Some(index) = index {
            used.push(PendingWrite::Buffer(index.id()));
        }
        Ok(GroupAccesses {
            writes: pending,
            used,
        })
    }
    fn record_with_writes(
        &mut self,
        command: Command<'r, 'data>,
        accesses: &GroupAccesses,
    ) -> Result<()> {
        self.pending_writes
            .try_reserve(accesses.writes.len())
            .map_err(|_| Error::OutOfMemory)?;
        self.push(command)?;
        for resource in &accesses.used {
            self.consume_access(*resource);
        }
        self.pending_writes.extend_from_slice(&accesses.writes);
        Ok(())
    }
}

/// Encoder for programmable dispatch commands in one active compute pass.
pub struct ComputePassEncoder<'encoder, 'r, 'data> {
    encoder: &'encoder mut CommandEncoder<'r, 'data>,
    pipeline: Option<ComputePipelineRef<'r>>,
    bind_groups: [Option<BindGroupRef<'r>>; MAX_BIND_GROUPS],
}
impl<'encoder, 'r, 'data> ComputePassEncoder<'encoder, 'r, 'data> {
    /// Select a compute pipeline from this command buffer's resource table.
    pub fn set_pipeline(&mut self, pipeline: ComputePipelineRef<'r>) -> Result<()> {
        self.encoder.resources.compute_pipeline(pipeline)?;
        self.encoder.push(Command::SetComputePipeline(pipeline))?;
        self.pipeline = Some(pipeline);
        Ok(())
    }
    /// Bind one descriptor set. Its pipeline compatibility is checked at dispatch.
    pub fn set_bind_group(&mut self, index: u32, bind_group: BindGroupRef<'r>) -> Result<()> {
        if index as usize >= MAX_BIND_GROUPS {
            return Err(Error::OutOfBounds);
        }
        self.encoder.resources.bind_group(bind_group)?;
        self.encoder
            .push(Command::SetBindGroup { index, bind_group })?;
        self.bind_groups[index as usize] = Some(bind_group);
        Ok(())
    }
    /// Dispatch a positive grid, with at most 65,535 workgroups per dimension.
    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) -> Result<()> {
        if [x, y, z].iter().any(|count| *count == 0 || *count > 65_535) {
            return Err(Error::InvalidValue);
        }
        let pipeline = self
            .encoder
            .resources
            .compute_pipeline(self.pipeline.ok_or(Error::PipelineNotSet)?)?;
        let writes =
            self.encoder
                .validate_groups(pipeline.layout(), &self.bind_groups, &[], None, None)?;
        self.encoder
            .record_with_writes(Command::Dispatch { x, y, z }, &writes)
    }
    /// End this compute pass, releasing the encoder for barriers or other passes.
    pub fn end(self) -> Result<()> {
        if self.encoder.commands.len() >= MAX_COMMANDS {
            return Err(Error::CommandLimitExceeded);
        }
        self.encoder
            .commands
            .try_reserve(1)
            .map_err(|_| Error::OutOfMemory)?;
        self.encoder.commands.push(Command::EndComputePass);
        self.encoder.pass_open = false;
        Ok(())
    }
}
impl<'encoder, 'r, 'data> RenderPassEncoder<'encoder, 'r, 'data> {
    /// Select programmable graphics with compatible color and optional depth formats.
    pub fn set_programmable_pipeline(
        &mut self,
        pipeline: ProgrammableRenderPipelineRef<'r>,
    ) -> Result<()> {
        let desc = self
            .encoder
            .resources
            .programmable_render_pipeline(pipeline)?;
        if desc.target_format() != self.target_format
            || desc
                .depth_state()
                .is_some_and(|depth| Some(depth.format()) != self.depth_format)
        {
            return Err(Error::PipelineTargetMismatch);
        }
        self.encoder
            .push(Command::SetProgrammablePipeline(pipeline))?;
        self.programmable_pipeline = Some(pipeline);
        self.pipeline = None;
        Ok(())
    }
    /// Bind one programmable descriptor set; compatibility is checked at draw.
    pub fn set_bind_group(&mut self, index: u32, bind_group: BindGroupRef<'r>) -> Result<()> {
        if index as usize >= MAX_BIND_GROUPS {
            return Err(Error::OutOfBounds);
        }
        self.encoder.resources.bind_group(bind_group)?;
        self.encoder
            .push(Command::SetBindGroup { index, bind_group })?;
        self.bind_groups[index as usize] = Some(bind_group);
        Ok(())
    }
    fn programmable_writes(
        &self,
        desc: &ProgrammableRenderPipelineDesc,
        indexed: bool,
    ) -> Result<GroupAccesses> {
        let mut attachments = [self.target, self.target];
        let attachment_count = if let Some(depth) = self.depth_target {
            attachments[1] = depth;
            2
        } else {
            1
        };
        self.encoder.validate_groups(
            desc.layout(),
            &self.bind_groups,
            &attachments[..attachment_count],
            desc.vertex_buffer()
                .and(self.vertex_buffer.map(|(buffer, _)| buffer)),
            if indexed {
                self.index_buffer.map(|(buffer, _, _)| buffer)
            } else {
                None
            },
        )
    }
    fn validate_cumulative_accesses(&mut self, accesses: &GroupAccesses) -> Result<()> {
        if accesses
            .writes
            .iter()
            .any(|resource| self.used_resources.contains(resource))
        {
            return Err(Error::ResourceAccessConflict);
        }
        self.used_resources
            .try_reserve(accesses.used.len())
            .map_err(|_| Error::OutOfMemory)?;
        Ok(())
    }
    fn validate_programmable_count(&self, count: u32) -> Result<ProgrammableRenderPipelineDesc> {
        if count == 0 || !count.is_multiple_of(3) {
            return Err(Error::InvalidValue);
        }
        self.encoder
            .resources
            .programmable_render_pipeline(self.programmable_pipeline.ok_or(Error::PipelineNotSet)?)
    }
    pub(super) fn draw_programmable(&mut self, count: u32, first: u32) -> Result<()> {
        let desc = self.validate_programmable_count(count)?;
        first.checked_add(count).ok_or(Error::Overflow)?;
        if let Some(layout) = desc.vertex_buffer() {
            let (buffer, offset) = self.vertex_buffer.ok_or(Error::VertexBufferNotSet)?;
            let buffer_desc = self.encoder.resources.buffer(buffer)?;
            self.encoder
                .check_buffer_access(buffer, BufferAccess::Vertex)?;
            if !offset.is_multiple_of(4) {
                return Err(Error::InvalidValue);
            }
            let bytes = (u64::from(first) + u64::from(count))
                .checked_mul(u64::from(layout.stride()))
                .ok_or(Error::Overflow)?;
            CommandEncoder::validate_byte_range(offset, bytes, buffer_desc.size())?;
        }
        let writes = self.programmable_writes(&desc, false)?;
        self.validate_cumulative_accesses(&writes)?;
        self.encoder.record_with_writes(
            Command::Draw {
                vertex_count: count,
                first_vertex: first,
            },
            &writes,
        )?;
        self.used_resources.extend_from_slice(&writes.used);
        Ok(())
    }
    pub(super) fn draw_indexed_programmable(
        &mut self,
        count: u32,
        first: u32,
        base_vertex: i32,
    ) -> Result<()> {
        let desc = self.validate_programmable_count(count)?;
        first.checked_add(count).ok_or(Error::Overflow)?;
        if let Some(layout) = desc.vertex_buffer() {
            let (buffer, offset) = self.vertex_buffer.ok_or(Error::VertexBufferNotSet)?;
            let buffer_desc = self.encoder.resources.buffer(buffer)?;
            self.encoder
                .check_buffer_access(buffer, BufferAccess::Vertex)?;
            if !offset.is_multiple_of(4) {
                return Err(Error::InvalidValue);
            }
            CommandEncoder::validate_byte_range(
                offset,
                u64::from(layout.stride()),
                buffer_desc.size(),
            )?;
        }
        let (buffer, offset, format) = self.index_buffer.ok_or(Error::IndexBufferNotSet)?;
        let buffer_desc = self.encoder.resources.buffer(buffer)?;
        self.encoder
            .check_buffer_access(buffer, BufferAccess::Index)?;
        let bytes = (u64::from(first) + u64::from(count))
            .checked_mul(format.byte_size())
            .ok_or(Error::Overflow)?;
        CommandEncoder::validate_byte_range(offset, bytes, buffer_desc.size())?;
        let writes = self.programmable_writes(&desc, true)?;
        self.validate_cumulative_accesses(&writes)?;
        self.encoder.record_with_writes(
            Command::DrawIndexed {
                index_count: count,
                first_index: first,
                base_vertex,
            },
            &writes,
        )?;
        self.used_resources.extend_from_slice(&writes.used);
        Ok(())
    }
}
