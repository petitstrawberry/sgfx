//! Bounded compute-only streams. GPU writes are retired and read back before
//! publishing buffer shadows, so later graphics uploads observe new revisions.
//! Mixed render/compute streams are rejected before any logical work is sent.
use super::*;
use crate::compute::Binding;
use alloc::sync::Arc;
use sgfx_codegen_virgl::programmable::{CompiledShader, compile_shader};

type Result<T> = core::result::Result<T, IrSubmitError>;

pub(super) fn contains_compute(commands: &CommandBuffer<'_, '_>) -> Result<bool> {
    #[cfg(feature = "backend-abi")]
    {
        Ok(commands.has_compute_commands()?)
    }
    #[cfg(not(feature = "backend-abi"))]
    {
        Ok(commands.commands().iter().any(|c| {
            matches!(
                c,
                Command::BeginComputePass
                    | Command::EndComputePass
                    | Command::SetComputePipeline(_)
                    | Command::Dispatch { .. }
            )
        }))
    }
}

pub(super) fn compile_pipeline(
    resources: &IrResources,
    id: ir::ComputePipelineId,
) -> Result<CompiledShader> {
    let desc = resources
        .resources
        .compute_pipeline(resources.resources.compute_pipeline_ref(id)?)?;
    let entry = desc.shader();
    let module = resources
        .resources
        .shader_module(resources.resources.shader_module_ref(entry.module())?)?;
    let shader = compile_shader(&module, ir::ShaderStage::Compute, entry.entry_point())
        .map_err(IrSubmitError::ShaderCompile)?;
    if shader.tgsi.len() > 48 * 1024
        || !shader.textures.is_empty()
        || !shader.image_query_levels.is_empty()
    {
        return Err(IrSubmitError::Unsupported(
            UnsupportedIrFeature::ResourceBindings,
        ));
    }
    let layout = |group: u32, binding: u32| -> Result<ir::BindGroupLayoutEntry> {
        let value = desc
            .layout()
            .bind_groups()
            .get(group as usize)
            .and_then(|g| g.entries().iter().find(|v| v.binding() == binding))
            .copied()
            .ok_or(ir::Error::BindingLayoutMismatch)?;
        if !value.visibility().contains(ir::ShaderStages::COMPUTE) {
            return Err(ir::Error::BindingLayoutMismatch.into());
        }
        Ok(value)
    };
    for binding in &shader.storage_buffers {
        match layout(binding.group, binding.binding)?.ty() {
            ir::BindingType::StorageBuffer { read_only } if !binding.writable || !read_only => {}
            _ => return Err(ir::Error::BindingLayoutMismatch.into()),
        }
    }
    for binding in &shader.uniform_buffers {
        if layout(binding.group, binding.binding)?.ty() != ir::BindingType::UniformBuffer {
            return Err(ir::Error::BindingLayoutMismatch.into());
        }
    }
    if let Some(push) = &shader.push_constants {
        for range in &push.required_ranges {
            for byte in range.clone() {
                if !desc.layout().push_constant_ranges().iter().any(|r| {
                    r.stages().contains(ir::ShaderStages::COMPUTE)
                        && byte >= r.offset()
                        && byte < r.offset() + r.size()
                }) {
                    return Err(ir::Error::BindingLayoutMismatch.into());
                }
            }
        }
    }
    Ok(shader)
}

struct Uniform {
    buffer: usize,
    offset: usize,
    length: usize,
    word: usize,
}
enum Event<'a> {
    Write {
        buffer: usize,
        offset: usize,
        data: &'a [u8],
    },
    Copy {
        source: usize,
        source_offset: usize,
        destination: usize,
        destination_offset: usize,
        size: usize,
    },
    Dispatch {
        shader: Rc<CompiledShader>,
        constants: Vec<u32>,
        uniforms: Vec<Uniform>,
        bindings: Vec<Binding>,
        groups: [u32; 3],
    },
}
struct Plan<'a> {
    slots: Vec<(usize, u64)>,
    buffers: Vec<Vec<u8>>,
    events: Vec<Event<'a>>,
    bytes: usize,
    pipelines: Vec<(ir::ComputePipelineId, Rc<CompiledShader>)>,
}
impl<'a> Plan<'a> {
    fn retain(&mut self, bytes: usize) -> Result<()> {
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .filter(|&n| n <= 64 * 1024 * 1024)
            .ok_or(IrSubmitError::SubmissionTooLarge)?;
        Ok(())
    }
    fn buffer(&mut self, resources: &IrResources, buffer: BufferRef<'_>) -> Result<usize> {
        if let Some(i) = self
            .slots
            .iter()
            .position(|(slot, _)| *slot == buffer.slot())
        {
            return Ok(i);
        }
        let desc = resources.resources.buffer(buffer)?;
        if desc.size() == 0 || desc.size() > 1024 * 1024 || !desc.size().is_multiple_of(4) {
            return Err(IrSubmitError::Unsupported(
                UnsupportedIrFeature::ResourceBindings,
            ));
        }
        let size = desc.size() as usize;
        self.retain(size)?;
        let revision = resources
            .buffer_revision(buffer.slot())?
            .checked_add(1)
            .ok_or(ir::Error::Overflow)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .map_err(|_| IrSubmitError::OutOfMemory)?;
        bytes.resize(size, 0);
        if let Some(shadow) = resources.shadow(buffer)? {
            let count = size.min(shadow.len());
            bytes[..count].copy_from_slice(&shadow[..count]);
        }
        self.slots
            .try_reserve(1)
            .map_err(|_| IrSubmitError::OutOfMemory)?;
        self.buffers
            .try_reserve(1)
            .map_err(|_| IrSubmitError::OutOfMemory)?;
        self.slots.push((buffer.slot(), revision));
        self.buffers.push(bytes);
        Ok(self.buffers.len() - 1)
    }
    fn range(&self, buffer: usize, offset: u64, size: u64) -> Result<(usize, usize)> {
        if offset.checked_add(size).ok_or(ir::Error::Overflow)? > self.buffers[buffer].len() as u64
        {
            return Err(ir::Error::OutOfBounds.into());
        }
        Ok((offset as usize, size as usize))
    }
    fn from_commands(resources: &IrResources, commands: &'a CommandBuffer<'_, '_>) -> Result<Self> {
        let mut plan = Self {
            slots: Vec::new(),
            buffers: Vec::new(),
            events: Vec::new(),
            bytes: 0,
            pipelines: Vec::new(),
        };
        let mut active = false;
        let mut pipeline = None;
        let mut groups = [None; 4];
        let mut push = [0u8; 128];
        let mut push_set = [false; 128];
        #[cfg(feature = "backend-abi")]
        let mut sequence = commands.command_reader();
        #[cfg(not(feature = "backend-abi"))]
        let mut sequence = commands.commands().iter();
        loop {
            #[cfg(feature = "backend-abi")]
            let Some(command) = sequence.next_command()? else {
                break;
            };
            #[cfg(not(feature = "backend-abi"))]
            let Some(command) = sequence.next() else {
                break;
            };
            plan.events
                .try_reserve(1)
                .map_err(|_| IrSubmitError::OutOfMemory)?;
            plan.retain(core::mem::size_of::<Event<'_>>())?;
            match command {
                Command::BeginComputePass if !active => {
                    active = true;
                    pipeline = None;
                    groups = [None; 4];
                    push_set.fill(false);
                }
                Command::EndComputePass if active => {
                    active = false;
                }
                Command::SetComputePipeline(p) if active => {
                    let compiled = if let Some((_, compiled)) =
                        plan.pipelines.iter().find(|(id, _)| *id == p.id())
                    {
                        compiled.clone()
                    } else {
                        let compiled = Rc::new(compile_pipeline(resources, p.id())?);
                        plan.retain(compiled.tgsi.len())?;
                        plan.pipelines
                            .try_reserve(1)
                            .map_err(|_| IrSubmitError::OutOfMemory)?;
                        plan.pipelines.push((p.id(), compiled.clone()));
                        compiled
                    };
                    pipeline = Some((p.id(), compiled));
                }
                Command::SetBindGroup { index, bind_group } if active => {
                    *groups
                        .get_mut(*index as usize)
                        .ok_or(ir::Error::OutOfBounds)? = Some(*bind_group);
                }
                Command::SetPushConstants {
                    stages,
                    offset,
                    data,
                } if active => {
                    if !stages.contains(ir::ShaderStages::COMPUTE)
                        || !offset.is_multiple_of(4)
                        || !data.len().is_multiple_of(4)
                    {
                        return Err(ir::Error::InvalidUsage.into());
                    }
                    let start = *offset as usize;
                    let end = start
                        .checked_add(data.len())
                        .filter(|&n| n <= 128)
                        .ok_or(ir::Error::OutOfBounds)?;
                    push[start..end].copy_from_slice(data);
                    push_set[start..end].fill(true);
                }
                Command::Dispatch { x, y, z } if active => {
                    if [x, y, z].iter().any(|&&n| n == 0 || n > 65535) {
                        return Err(ir::Error::OutOfBounds.into());
                    }
                    let (id, shader) = pipeline.as_ref().ok_or(ir::Error::InvalidUsage)?;
                    let desc = resources
                        .resources
                        .compute_pipeline(resources.resources.compute_pipeline_ref(*id)?)?;
                    let binding = |group: u32, binding: u32| -> Result<ir::BindingResource> {
                        let group_ref = groups
                            .get(group as usize)
                            .copied()
                            .flatten()
                            .ok_or(ir::Error::BindingLayoutMismatch)?;
                        let set = resources.resources.bind_group_shared(group_ref)?;
                        if desc.layout().bind_groups().get(group as usize) != Some(set.layout()) {
                            return Err(ir::Error::BindingLayoutMismatch.into());
                        }
                        set.entries()
                            .iter()
                            .find(|e| e.binding() == binding)
                            .map(|e| e.resource())
                            .ok_or(ir::Error::BindingLayoutMismatch.into())
                    };
                    let mut constants = Vec::new();
                    let words = shader
                        .uniform_buffers
                        .iter()
                        .map(|b| (b.first_register + b.size.div_ceil(16)) * 4)
                        .chain(
                            shader
                                .storage_buffers
                                .iter()
                                .map(|b| (b.first_register + 1) * 4),
                        )
                        .chain(
                            shader
                                .push_constants
                                .iter()
                                .map(|b| (b.first_register + b.size.div_ceil(16)) * 4),
                        )
                        .max()
                        .unwrap_or(0) as usize;
                    plan.retain(
                        words * 4
                            + shader.storage_buffers.len() * core::mem::size_of::<Binding>()
                            + shader.uniform_buffers.len() * core::mem::size_of::<Uniform>(),
                    )?;
                    constants
                        .try_reserve_exact(words)
                        .map_err(|_| IrSubmitError::OutOfMemory)?;
                    constants.resize(words, 0);
                    let mut last = 0;
                    let mut bindings = Vec::new();
                    bindings
                        .try_reserve_exact(shader.storage_buffers.len())
                        .map_err(|_| IrSubmitError::OutOfMemory)?;
                    for storage in &shader.storage_buffers {
                        let ir::BindingResource::Buffer {
                            buffer,
                            offset,
                            size,
                        } = binding(storage.group, storage.binding)?
                        else {
                            return Err(ir::Error::BindingLayoutMismatch.into());
                        };
                        let reference = resources.resources.buffer_ref(buffer)?;
                        if !resources
                            .resources
                            .buffer(reference)?
                            .usage()
                            .contains(BufferUsage::STORAGE)
                            || size == 0
                            || size > 256 * 1024
                            || !offset.is_multiple_of(4)
                            || !size.is_multiple_of(4)
                        {
                            return Err(ir::Error::InvalidUsage.into());
                        }
                        let index = plan.buffer(resources, reference)?;
                        plan.range(index, offset, size)?;
                        let start = storage.first_register as usize * 4;
                        constants[start] = offset as u32;
                        constants[start + 1] = size as u32;
                        last = last.max(start + 4);
                        bindings.push(Binding {
                            slot: storage.slot,
                            buffer: index,
                            writable: storage.writable,
                        });
                    }
                    let mut uniforms = Vec::new();
                    uniforms
                        .try_reserve_exact(shader.uniform_buffers.len())
                        .map_err(|_| IrSubmitError::OutOfMemory)?;
                    for uniform in &shader.uniform_buffers {
                        let ir::BindingResource::Buffer {
                            buffer,
                            offset,
                            size,
                        } = binding(uniform.group, uniform.binding)?
                        else {
                            return Err(ir::Error::BindingLayoutMismatch.into());
                        };
                        let reference = resources.resources.buffer_ref(buffer)?;
                        if !resources
                            .resources
                            .buffer(reference)?
                            .usage()
                            .contains(BufferUsage::UNIFORM)
                            || size < u64::from(uniform.required_size)
                            || size > 16384
                            || !offset.is_multiple_of(4)
                        {
                            return Err(ir::Error::InvalidUsage.into());
                        }
                        let index = plan.buffer(resources, reference)?;
                        let (offset, _) = plan.range(index, offset, size)?;
                        let word = uniform.first_register as usize * 4;
                        last = last.max(word + uniform.size.div_ceil(16) as usize * 4);
                        uniforms.push(Uniform {
                            buffer: index,
                            offset,
                            length: uniform.required_size as usize,
                            word,
                        });
                    }
                    if let Some(block) = &shader.push_constants {
                        for required in &block.required_ranges {
                            if !push_set[required.start as usize..required.end as usize]
                                .iter()
                                .all(|v| *v)
                            {
                                return Err(ir::Error::InvalidUsage.into());
                            }
                        }
                        let start = block.first_register as usize * 4;
                        for (i, word) in push[..block.size as usize].chunks_exact(4).enumerate() {
                            constants[start + i] = u32::from_le_bytes(word.try_into().unwrap());
                        }
                        last = last.max(start + block.size.div_ceil(16) as usize * 4);
                    }
                    constants.truncate(last);
                    plan.events.push(Event::Dispatch {
                        shader: shader.clone(),
                        constants,
                        uniforms,
                        bindings,
                        groups: [*x, *y, *z],
                    });
                }
                Command::WriteBuffer {
                    buffer,
                    offset,
                    data,
                } if !active => {
                    if !resources
                        .resources
                        .buffer(*buffer)?
                        .usage()
                        .contains(BufferUsage::COPY_DST)
                    {
                        return Err(ir::Error::InvalidUsage.into());
                    }
                    let index = plan.buffer(resources, *buffer)?;
                    let (offset, _) = plan.range(index, *offset, data.len() as u64)?;
                    plan.events.push(Event::Write {
                        buffer: index,
                        offset,
                        data,
                    });
                }
                Command::CopyBufferToBuffer {
                    source,
                    source_offset,
                    destination,
                    destination_offset,
                    size,
                } if !active => {
                    if !resources
                        .resources
                        .buffer(*source)?
                        .usage()
                        .contains(BufferUsage::COPY_SRC)
                        || !resources
                            .resources
                            .buffer(*destination)?
                            .usage()
                            .contains(BufferUsage::COPY_DST)
                    {
                        return Err(ir::Error::InvalidUsage.into());
                    }
                    let source = plan.buffer(resources, *source)?;
                    let destination = plan.buffer(resources, *destination)?;
                    let (source_offset, size) = plan.range(source, *source_offset, *size)?;
                    let (destination_offset, _) =
                        plan.range(destination, *destination_offset, size as u64)?;
                    if source == destination
                        && source_offset < destination_offset + size
                        && destination_offset < source_offset + size
                    {
                        return Err(ir::Error::InvalidUsage.into());
                    }
                    plan.events.push(Event::Copy {
                        source,
                        source_offset,
                        destination,
                        destination_offset,
                        size,
                    });
                }
                Command::ResourceBarrier(ir::ResourceBarrier::Buffer { buffer, .. }) if !active => {
                    resources.resources.buffer(*buffer)?;
                }
                _ => {
                    return Err(IrSubmitError::Unsupported(
                        UnsupportedIrFeature::CommandSequence,
                    ));
                }
            }
        }
        if active {
            return Err(ir::Error::InvalidUsage.into());
        }
        Ok(plan)
    }
    fn execute(&mut self, context: &crate::virgl::Context, accepted: &mut bool) -> Result<()> {
        for event in &mut self.events {
            match event {
                Event::Write {
                    buffer,
                    offset,
                    data,
                } => self.buffers[*buffer][*offset..*offset + data.len()].copy_from_slice(data),
                Event::Copy {
                    source,
                    source_offset,
                    destination,
                    destination_offset,
                    size,
                } => {
                    if source == destination {
                        self.buffers[*source].copy_within(
                            *source_offset..*source_offset + *size,
                            *destination_offset,
                        );
                    } else {
                        let (source, destination) = if *source < *destination {
                            let (first, second) = self.buffers.split_at_mut(*destination);
                            (&first[*source], &mut second[0])
                        } else {
                            let (first, second) = self.buffers.split_at_mut(*source);
                            (&second[0], &mut first[*destination])
                        };
                        destination[*destination_offset..*destination_offset + *size]
                            .copy_from_slice(&source[*source_offset..*source_offset + *size]);
                    }
                }
                Event::Dispatch {
                    shader,
                    constants,
                    uniforms,
                    bindings,
                    groups,
                } => {
                    for uniform in uniforms {
                        for (i, word) in self.buffers[uniform.buffer]
                            [uniform.offset..uniform.offset + uniform.length]
                            .chunks_exact(4)
                            .enumerate()
                        {
                            constants[uniform.word + i] =
                                u32::from_le_bytes(word.try_into().unwrap());
                        }
                    }
                    context.dispatch_compute(
                        shader,
                        constants,
                        &mut self.buffers,
                        bindings,
                        *groups,
                        accepted,
                    )?;
                }
            }
        }
        Ok(())
    }
}

impl Queue {
    pub(super) fn submit_compute_async(
        &self,
        context: &Context,
        resources: &mut IrResources,
        commands: &CommandBuffer<'_, '_>,
    ) -> core::result::Result<Submission, SubmitError<IrSubmitError, Submission>> {
        let preflight = (|| -> Result<_> {
            resources.ensure_usable()?;
            if !core::ptr::eq(resources.resources(), commands.resources()) {
                return Err(IrSubmitError::ResourceTableMismatch);
            }
            if resources.context_id != context.backend.context_id()
                || self.backend.context_id() != resources.context_id
            {
                return Err(IrSubmitError::ContextMismatch);
            }
            let plan = Plan::from_commands(resources, commands)?;
            let driver::Context::Virgl(native) = context.backend.as_ref();
            let limit = native.compute_transport_limit()?;
            for event in &plan.events {
                if let Event::Dispatch {
                    shader,
                    constants,
                    bindings,
                    ..
                } = event
                {
                    // Conservative packet bound includes continuation headers,
                    // two transfers per unique SSBO and complete state cleanup.
                    let text = (shader.tgsi.len() + 1).div_ceil(4) * 4;
                    let bytes = text
                        + text.div_ceil(16 * 1024) * 24
                        + constants.len() * 4
                        + 88
                        + bindings.len() * 160;
                    if bytes > limit {
                        return Err(IrSubmitError::SubmissionTooLarge);
                    }
                }
            }
            if !native.compute_supported() {
                return Err(IrSubmitError::Unsupported(
                    UnsupportedIrFeature::ProgrammableExecution,
                ));
            }
            self.backend.wait_idle()?;
            let signal = Arc::new(crate::dispatch::Signal::new()?);
            Ok((plan, signal))
        })();
        let (mut plan, signal) = preflight.map_err(SubmitError::Rejected)?;
        let mut accepted = false;
        let driver::Context::Virgl(native) = context.backend.as_ref();
        let result = plan.execute(native, &mut accepted);
        if let Err(error) = result {
            if !accepted {
                return Err(SubmitError::Rejected(error));
            }
            resources.submission_failed = true;
            signal.complete(Err(crate::dispatch::Failure::Unavailable));
            return Err(SubmitError::Failed {
                error,
                completion: Submission { signal },
            });
        }
        for ((slot, revision), bytes) in plan.slots.into_iter().zip(plan.buffers) {
            resources.buffer_shadows[slot] = Some(bytes);
            resources.buffer_revisions[slot] = revision;
            resources.canonical_buffer_revisions[slot] = None;
        }
        signal.complete(Ok(()));
        Ok(Submission { signal })
    }
}
