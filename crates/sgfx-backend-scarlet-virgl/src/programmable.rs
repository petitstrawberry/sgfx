//! Validation and lowering for the native programmable graphics subset.

use super::*;
use crate::PixelRect;
#[cfg(feature = "programmable")]
use sgfx_codegen_virgl::programmable::{compile_shader, validate_shader_module};

#[cfg(feature = "programmable")]
pub use sgfx_codegen_virgl::programmable::ShaderCompileError;

#[cfg(not(feature = "programmable"))]
#[derive(Debug)]
pub enum ShaderCompileError {}

impl IrResources {
    /// Parse and semantically validate a shader before admitting its handle.
    #[cfg(feature = "programmable")]
    pub fn validate_shader_module(&self, id: ir::ShaderModuleId) -> Result<(), IrSubmitError> {
        let module = self
            .resources
            .shader_module(self.resources.shader_module_ref(id)?)?;
        validate_shader_module(&module).map_err(IrSubmitError::ShaderCompile)
    }

    /// Reject programmable shader modules when runtime translation is omitted.
    #[cfg(not(feature = "programmable"))]
    pub fn validate_shader_module(&self, id: ir::ShaderModuleId) -> Result<(), IrSubmitError> {
        self.resources
            .shader_module(self.resources.shader_module_ref(id)?)?;
        Err(IrSubmitError::Unsupported(
            UnsupportedIrFeature::ProgrammableExecution,
        ))
    }

    /// Compile both stages and validate the supported native graphics state.
    #[cfg(feature = "programmable")]
    pub fn validate_programmable_render_pipeline(
        &self,
        id: ir::ProgrammableRenderPipelineId,
    ) -> Result<(), IrSubmitError> {
        self.compiled_pipeline(id).map(|_| ())
    }

    /// Reject programmable pipelines when runtime translation is omitted.
    #[cfg(not(feature = "programmable"))]
    pub fn validate_programmable_render_pipeline(
        &self,
        id: ir::ProgrammableRenderPipelineId,
    ) -> Result<(), IrSubmitError> {
        self.resources.programmable_render_pipeline_shared(
            self.resources.programmable_render_pipeline_ref(id)?,
        )?;
        Err(IrSubmitError::Unsupported(
            UnsupportedIrFeature::ProgrammableExecution,
        ))
    }

    /// Compute execution is not implemented by this native backend.
    pub fn validate_compute_pipeline(
        &self,
        id: ir::ComputePipelineId,
    ) -> Result<(), IrSubmitError> {
        self.resources
            .compute_pipeline(self.resources.compute_pipeline_ref(id)?)?;
        Err(IrSubmitError::Unsupported(
            UnsupportedIrFeature::ProgrammableExecution,
        ))
    }

    /// Read ordered CPU-visible bytes. Native graphics never writes buffer storage.
    /// Callers must observe the submission receipt before reading GPU image results.
    pub fn read_buffer(
        &self,
        id: ir::BufferId,
        offset: u64,
        size: u64,
    ) -> Result<Vec<u8>, IrSubmitError> {
        if self.submission_failed {
            return Err(IrSubmitError::SubmissionFailed);
        }
        let buffer = self.resources.buffer_ref(id)?;
        let descriptor = self.resources.buffer(buffer)?;
        if offset.checked_add(size).ok_or(ir::Error::Overflow)? > descriptor.size() {
            return Err(ir::Error::OutOfBounds.into());
        }
        let mut output = Vec::new();
        output
            .try_reserve_exact(usize::try_from(size).map_err(|_| ir::Error::Overflow)?)
            .map_err(|_| IrSubmitError::OutOfMemory)?;
        output.resize(size as usize, 0);
        if let Some(shadow) = self.shadow(buffer)? {
            let start = usize::try_from(offset).map_err(|_| ir::Error::Overflow)?;
            if start < shadow.len() {
                let length = output.len().min(shadow.len() - start);
                output[..length].copy_from_slice(&shadow[start..start + length]);
            }
        }
        Ok(output)
    }

    #[cfg(feature = "programmable")]
    pub(super) fn compiled_pipeline(
        &self,
        id: ir::ProgrammableRenderPipelineId,
    ) -> Result<Rc<driver::IrProgrammablePipeline>, IrSubmitError> {
        if let Some((_, compiled)) = self
            .programmable_pipelines
            .borrow()
            .iter()
            .find(|(candidate, _)| *candidate == id)
        {
            return Ok(Rc::clone(compiled));
        }
        let compiled = compile_pipeline(&self.resources, id)?;
        let mut cache = self.programmable_pipelines.borrow_mut();
        cache
            .try_reserve(1)
            .map_err(|_| IrSubmitError::OutOfMemory)?;
        cache.push((id, Rc::clone(&compiled)));
        Ok(compiled)
    }

    #[cfg(not(feature = "programmable"))]
    pub(super) fn compiled_pipeline(
        &self,
        id: ir::ProgrammableRenderPipelineId,
    ) -> Result<Rc<driver::IrProgrammablePipeline>, IrSubmitError> {
        self.resources.programmable_render_pipeline_shared(
            self.resources.programmable_render_pipeline_ref(id)?,
        )?;
        Err(IrSubmitError::Unsupported(
            UnsupportedIrFeature::ProgrammableExecution,
        ))
    }
}

#[cfg(feature = "programmable")]
fn compile_pipeline(
    resources: &ResourceTable,
    id: ir::ProgrammableRenderPipelineId,
) -> Result<Rc<driver::IrProgrammablePipeline>, IrSubmitError> {
    let reference = resources.programmable_render_pipeline_ref(id)?;
    let pipeline = resources.programmable_render_pipeline_shared(reference)?;
    if pipeline.color_targets().any(|target| {
        !matches!(
            target.format(),
            TextureFormat::Bgra8Unorm | TextureFormat::Rgba8Unorm
        )
    }) {
        return Err(IrSubmitError::Unsupported(
            UnsupportedIrFeature::PipelineTargetFormat,
        ));
    }
    if !matches!(
        pipeline.topology(),
        PrimitiveTopology::TriangleList | PrimitiveTopology::TriangleStrip
    ) {
        return Err(IrSubmitError::Unsupported(
            UnsupportedIrFeature::PrimitiveTopology,
        ));
    }
    if pipeline.vertex_buffers().len() > 8
        || pipeline
            .vertex_buffers()
            .iter()
            .flat_map(|layout| layout.attributes())
            .any(|a| a.location() >= 16)
    {
        return Err(IrSubmitError::Unsupported(
            UnsupportedIrFeature::VertexLayout,
        ));
    }
    for group in pipeline.layout().bind_groups() {
        if group.entries().iter().any(|binding| {
            !matches!(
                binding.ty(),
                ir::BindingType::UniformBuffer
                    | ir::BindingType::StorageBuffer { read_only: true }
                    | ir::BindingType::SampledTexture
                    | ir::BindingType::SampledTextureView {
                        dimension: ir::TextureViewDimension::D2
                            | ir::TextureViewDimension::D2Array
                            | ir::TextureViewDimension::Cube,
                        depth: _
                    }
                    | ir::BindingType::Sampler
                    | ir::BindingType::ComparisonSampler
            )
        }) {
            return Err(IrSubmitError::Unsupported(
                UnsupportedIrFeature::ResourceBindings,
            ));
        }
    }
    let compile = |entry: &ir::ShaderEntryPoint| {
        let module = resources.shader_module(resources.shader_module_ref(entry.module())?)?;
        compile_shader(&module, entry.stage(), entry.entry_point())
            .map_err(IrSubmitError::ShaderCompile)
    };
    let vertex = compile(pipeline.vertex())?;
    let fragment = compile(pipeline.fragment())?;
    let attributes = || {
        pipeline
            .vertex_buffers()
            .iter()
            .flat_map(|layout| layout.attributes())
    };
    if vertex
        .input_locations
        .iter()
        .any(|location| !attributes().any(|attribute| attribute.location() == *location))
        || fragment
            .input_locations
            .iter()
            .any(|location| !vertex.output_locations.contains(location))
    {
        return Err(ir::Error::InvalidDescriptor.into());
    }
    for input in &vertex.inputs {
        use sgfx_codegen_virgl::programmable::IoScalar;
        let attribute = attributes()
            .find(|attribute| attribute.location() == input.location)
            .ok_or(ir::Error::InvalidDescriptor)?;
        let (scalar, components) = match attribute.format() {
            VertexFormat::Sint32 => (IoScalar::Sint, 1),
            VertexFormat::Uint32 => (IoScalar::Uint, 1),
            VertexFormat::Sint16x4 => (IoScalar::Sint, 4),
            VertexFormat::Float16x2 | VertexFormat::Float32x2 => (IoScalar::Float, 2),
            VertexFormat::Float32x3 => (IoScalar::Float, 3),
            _ => (IoScalar::Float, 4),
        };
        if scalar != input.scalar || components != input.components {
            return Err(ir::Error::InvalidDescriptor.into());
        }
    }
    for input in &fragment.inputs {
        let output = vertex
            .outputs
            .iter()
            .find(|output| output.location == input.location)
            .ok_or(ir::Error::InvalidDescriptor)?;
        if input.components != output.components
            || input.scalar != output.scalar
            || input.interpolation != output.interpolation
        {
            return Err(ir::Error::InvalidDescriptor.into());
        }
    }
    // Each stage is bounded independently and uses VirGL shader continuations.
    if [&vertex, &fragment]
        .iter()
        .any(|shader| shader.tgsi.len() > 256 * 1024)
    {
        return Err(IrSubmitError::SubmissionTooLarge);
    }
    for shader in [&vertex, &fragment] {
        let visibility = match shader.stage {
            ir::ShaderStage::Vertex => ir::ShaderStages::VERTEX,
            ir::ShaderStage::Fragment => ir::ShaderStages::FRAGMENT,
            ir::ShaderStage::Compute => {
                return Err(IrSubmitError::Unsupported(
                    UnsupportedIrFeature::ProgrammableExecution,
                ));
            }
        };
        if let Some(push) = &shader.push_constants {
            let ranges = pipeline.layout().push_constant_ranges();
            // Require the declared source block within the bounded stage layout.
            for byte in 0..push.size {
                if !ranges.iter().any(|range| {
                    range.stages().contains(visibility)
                        && byte >= range.offset()
                        && byte < range.offset() + range.size()
                }) {
                    return Err(ir::Error::BindingLayoutMismatch.into());
                }
            }
        }
        for binding in &shader.storage_buffers {
            let layout = pipeline
                .layout()
                .bind_groups()
                .get(binding.group as usize)
                .and_then(|group| {
                    group
                        .entries()
                        .iter()
                        .find(|entry| entry.binding() == binding.binding)
                })
                .ok_or(ir::Error::BindingLayoutMismatch)?;
            if layout.ty() != (ir::BindingType::StorageBuffer { read_only: true })
                || !layout.visibility().contains(visibility)
            {
                return Err(ir::Error::BindingLayoutMismatch.into());
            }
        }
        for binding in &shader.uniform_buffers {
            let layout = pipeline
                .layout()
                .bind_groups()
                .get(binding.group as usize)
                .and_then(|group| {
                    group
                        .entries()
                        .iter()
                        .find(|entry| entry.binding() == binding.binding)
                })
                .ok_or(ir::Error::BindingLayoutMismatch)?;
            if layout.ty() != ir::BindingType::UniformBuffer
                || !layout.visibility().contains(visibility)
                || binding.size > 16 * 1024
                || binding
                    .first_register
                    .checked_add(binding.size.div_ceil(16))
                    .is_none_or(|end| end > 1024)
            {
                return Err(ir::Error::BindingLayoutMismatch.into());
            }
        }
        for query in &shader.image_query_levels {
            let layout = pipeline
                .layout()
                .bind_groups()
                .get(query.group as usize)
                .and_then(|group| {
                    group
                        .entries()
                        .iter()
                        .find(|entry| entry.binding() == query.binding)
                })
                .ok_or(ir::Error::BindingLayoutMismatch)?;
            if !matches!(
                layout.ty(),
                ir::BindingType::SampledTexture | ir::BindingType::SampledTextureView { .. }
            ) || !layout.visibility().contains(visibility)
            {
                return Err(ir::Error::BindingLayoutMismatch.into());
            }
        }
        for pair in &shader.textures {
            for (group, binding, ty) in [
                (
                    pair.image_group,
                    pair.image_binding,
                    ir::BindingType::SampledTextureView {
                        dimension: pair.dimension,
                        depth: pair.depth,
                    },
                ),
                (
                    pair.sampler_group,
                    pair.sampler_binding,
                    if pair.comparison {
                        ir::BindingType::ComparisonSampler
                    } else {
                        ir::BindingType::Sampler
                    },
                ),
            ]
            .into_iter()
            .take(if pair.uses_sampler { 2 } else { 1 })
            {
                let entry = pipeline
                    .layout()
                    .bind_groups()
                    .get(group as usize)
                    .and_then(|group| {
                        group
                            .entries()
                            .iter()
                            .find(|entry| entry.binding() == binding)
                    })
                    .ok_or(ir::Error::BindingLayoutMismatch)?;
                if !(entry.ty() == ty
                    || (entry.ty() == ir::BindingType::SampledTexture
                        && ty
                            == ir::BindingType::SampledTextureView {
                                dimension: ir::TextureViewDimension::D2,
                                depth: false,
                            }))
                    || !entry.visibility().contains(visibility)
                {
                    return Err(ir::Error::BindingLayoutMismatch.into());
                }
            }
        }
    }
    let compiled = Rc::new(driver::IrProgrammablePipeline {
        topology: pipeline.topology(),
        slot: reference.slot(),
        vertex,
        fragment,
        vertex_buffers: pipeline.vertex_buffers().to_vec(),
    });
    Ok(compiled)
}

impl Context {
    /// Wait for accepted work and read tightly packed pixels in the logical format.
    pub fn read_texture(
        &self,
        resources: &mut IrResources,
        id: ir::TextureId,
    ) -> Result<Vec<u8>, IrSubmitError> {
        if resources.submission_failed {
            return Err(IrSubmitError::SubmissionFailed);
        }
        if resources.context_id != self.backend.context_id() {
            return Err(IrSubmitError::ContextMismatch);
        }
        let reference = resources.resources.texture_ref(id)?;
        let descriptor = resources.resources.texture(reference)?;
        if descriptor.mip_level_count() != 1 || descriptor.array_layer_count() != 1 {
            return Err(IrSubmitError::Unsupported(UnsupportedIrFeature::Mipmaps));
        }
        if !descriptor.usage().contains(TextureUsage::COPY_SRC) {
            return Err(ir::Error::InvalidUsage.into());
        }
        if !matches!(
            descriptor.format(),
            TextureFormat::Bgra8Unorm | TextureFormat::Rgba8Unorm
        ) {
            return Err(IrSubmitError::Unsupported(
                UnsupportedIrFeature::TargetFormat,
            ));
        }
        let spec = texture_spec(reference, descriptor)?;
        let mut bytes = match resources.mapped_image(reference) {
            Ok(image) => {
                let length = usize::try_from(descriptor.byte_size()?)
                    .map_err(|_| IrSubmitError::OutOfMemory)?;
                let stride = spec.width.checked_mul(4).ok_or(ir::Error::OutOfBounds)?;
                let mut pixels = Vec::new();
                pixels
                    .try_reserve_exact(length)
                    .map_err(|_| IrSubmitError::OutOfMemory)?;
                pixels.resize(length, 0);
                // A mapped render target is owned by the imported image, rather
                // than the backend's internal texture allocation. This path
                // waits for scheduled work before reading that same GPU image.
                self.backend.readback_image_bgra(
                    &image.as_ref().backend,
                    &mut pixels,
                    stride,
                    PixelRect::new(0, 0, spec.width, spec.height),
                )?;
                pixels
            }
            Err(IrSubmitError::ImageNotMapped) => self
                .backend
                .readback_ir_texture(&mut resources.backend, spec)?,
            Err(error) => return Err(error),
        };
        if spec.format == IrTextureFormat::Rgba8 {
            for pixel in bytes.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
        }
        Ok(bytes)
    }
}

pub(super) fn validate_barrier(barrier: ir::ResourceBarrier<'_>) -> Result<(), IrSubmitError> {
    // Transfer, vertex/index fetch and attachment accesses execute in one ordered
    // VirGL context. Shader writes are rejected instead of being treated as no-ops.
    match barrier {
        ir::ResourceBarrier::Buffer { before, after, .. }
            if [before, after]
                .iter()
                .any(|access| matches!(access, ir::BufferAccess::StorageReadWrite)) =>
        {
            Err(IrSubmitError::Unsupported(
                UnsupportedIrFeature::ExplicitBarrier,
            ))
        }
        ir::ResourceBarrier::Texture { before, after, .. }
        | ir::ResourceBarrier::TextureMip { before, after, .. }
            if [before, after].contains(&ir::TextureAccess::StorageWrite) =>
        {
            Err(IrSubmitError::Unsupported(
                UnsupportedIrFeature::ExplicitBarrier,
            ))
        }
        _ => Ok(()),
    }
}

#[cfg(feature = "programmable")]
fn buffer_spec(
    resources: &IrResources,
    pending: &PendingBuffers,
    buffer: BufferRef<'_>,
) -> Result<IrBufferSpec, IrSubmitError> {
    let descriptor = resources.resources.buffer(buffer)?;
    let bytes = pending.bytes(resources, buffer)?;
    if descriptor.size() > u64::from(u32::MAX)
        || bytes.as_slice().is_empty()
        || !bytes.as_slice().len().is_multiple_of(4)
    {
        return Err(IrSubmitError::Unsupported(
            UnsupportedIrFeature::VertexBuffer,
        ));
    }
    Ok(IrBufferSpec {
        slot: buffer.slot(),
        size: descriptor.size(),
        revision: pending.revision(resources, buffer)?,
    })
}

#[cfg(feature = "programmable")]
fn constant_words(
    bytes: &[u8],
    padded_size: usize,
) -> Result<driver::IrConstantWords, IrSubmitError> {
    if !bytes.len().is_multiple_of(4)
        || !padded_size.is_multiple_of(16)
        || padded_size < bytes.len()
    {
        return Err(ir::Error::InvalidValue.into());
    }
    let len = padded_size / 4;
    if len <= 32 {
        let mut words = [0; 32];
        for (destination, word) in words.iter_mut().zip(bytes.chunks_exact(4)) {
            *destination = u32::from_ne_bytes([word[0], word[1], word[2], word[3]]);
        }
        Ok(driver::IrConstantWords::Inline { words, len })
    } else {
        let mut words = Vec::new();
        words
            .try_reserve_exact(len)
            .map_err(|_| IrSubmitError::OutOfMemory)?;
        words.extend(
            bytes
                .chunks_exact(4)
                .map(|word| u32::from_ne_bytes([word[0], word[1], word[2], word[3]])),
        );
        words.resize(len, 0);
        Ok(driver::IrConstantWords::Shared(words.into()))
    }
}

#[cfg(feature = "programmable")]
pub(super) fn decode_draw(
    resources: &IrResources,
    pending: &PendingBuffers,
    pass: &mut ActivePass<'_>,
    first: u32,
    count: u32,
    base_vertex: Option<i32>,
    instance_count: u32,
    first_instance: u32,
) -> Result<IrDraw, IrSubmitError> {
    let reference = pass.programmable.ok_or(ir::Error::PipelineNotSet)?;
    let pipeline = resources
        .resources
        .programmable_render_pipeline_shared(reference)?;
    let valid_count = match pipeline.topology() {
        PrimitiveTopology::TriangleList => count > 0 && count.is_multiple_of(3),
        PrimitiveTopology::TriangleStrip => count >= 3,
    };
    if !valid_count {
        return Err(ir::Error::InvalidValue.into());
    }
    let mut attached =
        core::iter::once(pass.attachment).chain(pass.additional_attachments.iter().copied());
    for target in pipeline.color_targets() {
        let Some(attachment) = attached.next() else {
            return Err(ir::Error::InvalidDescriptor.into());
        };
        if resources.resources.texture(attachment)?.format() != target.format() {
            return Err(ir::Error::InvalidDescriptor.into());
        }
    }
    if attached.next().is_some()
        || pipeline.depth_state().is_some() && pass.depth_attachment.is_none()
    {
        return Err(ir::Error::InvalidDescriptor.into());
    }
    let compiled = resources.compiled_pipeline(reference.id())?;
    let key = ProgrammableDrawKey {
        pipeline: reference,
        bind_groups: pass.bind_groups,
        push_constants: pass.push_constants,
    };
    let cached = pass
        .programmable_cache
        .as_ref()
        .and_then(|cache| cache.get(&key));
    let cached_constants = if cached.is_none() {
        pass.programmable_cache
            .as_ref()
            .map(|cache| cache.constants(&resources.resources, &key, pipeline.layout()))
            .transpose()?
            .flatten()
    } else {
        None
    };
    let mut constants = core::mem::take(&mut pass.constants_scratch);
    let mut textures = core::mem::take(&mut pass.textures_scratch);
    constants.clear();
    textures.clear();
    let mut storage_buffers = Vec::new();
    if cached.is_none() {
        let resource = |group_index: u32,
                        binding: u32|
         -> Result<ir::BindingResource, IrSubmitError> {
            let group_ref = pass
                .bind_groups
                .get(group_index as usize)
                .copied()
                .flatten()
                .ok_or(ir::Error::BindingLayoutMismatch)?;
            let group = resources.resources.bind_group_shared(group_ref)?;
            if pipeline.layout().bind_groups().get(group_index as usize) != Some(group.layout()) {
                return Err(ir::Error::BindingLayoutMismatch.into());
            }
            group
                .entries()
                .iter()
                .find(|entry| entry.binding() == binding)
                .map(|entry| entry.resource())
                .ok_or_else(|| ir::Error::BindingLayoutMismatch.into())
        };
        for shader in [&compiled.vertex, &compiled.fragment] {
            for binding in &shader.storage_buffers {
                let ir::BindingResource::Buffer {
                    buffer,
                    offset,
                    size,
                } = resource(binding.group, binding.binding)?
                else {
                    return Err(ir::Error::BindingLayoutMismatch.into());
                };
                let buffer = resources.resources.buffer_ref(buffer)?;
                let desc = resources.resources.buffer(buffer)?;
                if !desc.usage().contains(ir::BufferUsage::STORAGE)
                    || desc.size() > crate::virgl::MAX_STORAGE_VIEW_BYTES
                    || size == 0
                    || size > 256 * 1024
                    || !offset.is_multiple_of(4)
                    || !size.is_multiple_of(4)
                    || offset.checked_add(size).is_none_or(|end| end > desc.size())
                {
                    return Err(ir::Error::InvalidDescriptor.into());
                }
                storage_buffers.push(driver::IrStorageBufferBinding {
                    stage: shader.stage,
                    slot: binding.slot,
                    buffer: buffer_spec(resources, pending, buffer)?,
                });
                if cached_constants.is_none() {
                    let mut words = [0; 32];
                    words[0] = offset as u32;
                    words[1] = size as u32;
                    constants.push(driver::IrConstantBuffer {
                        stage: shader.stage,
                        first_register: binding.first_register,
                        words: driver::IrConstantWords::Inline { words, len: 4 },
                    });
                }
            }
            for query in &shader.image_query_levels {
                let (texture, levels) = match resource(query.group, query.binding)? {
                    ir::BindingResource::Texture(texture) => (texture, None),
                    ir::BindingResource::TextureView { texture, view } => {
                        (texture, Some(view.mip_level_count()))
                    }
                    _ => return Err(ir::Error::BindingLayoutMismatch.into()),
                };
                let texture = resources.resources.texture_ref(texture)?;
                let descriptor = resources.resources.texture(texture)?;
                if !descriptor.usage().contains(TextureUsage::SAMPLED) {
                    return Err(ir::Error::InvalidUsage.into());
                }
                let levels = levels.unwrap_or(descriptor.mip_level_count());
                if levels == 0 || levels > descriptor.mip_level_count() {
                    return Err(ir::Error::InvalidDescriptor.into());
                }
                if cached_constants.is_none() {
                    let mut words = [0; 32];
                    words[0] = levels;
                    constants.push(driver::IrConstantBuffer {
                        stage: shader.stage,
                        first_register: query.first_register,
                        words: driver::IrConstantWords::Inline { words, len: 4 },
                    });
                }
            }
            for binding in &shader.textures {
                let texture = match resource(binding.image_group, binding.image_binding)? {
                    ir::BindingResource::Texture(texture) => texture,
                    ir::BindingResource::TextureView { texture, view } => {
                        let desc = resources
                            .resources
                            .texture(resources.resources.texture_ref(texture)?)?;
                        if view.dimension() != binding.dimension
                            || view.format() != desc.format()
                            || view.base_mip_level() != 0
                            || view.mip_level_count() != desc.mip_level_count()
                            || view.base_array_layer() != 0
                            || view.array_layer_count() != desc.array_layer_count()
                        {
                            return Err(IrSubmitError::Unsupported(
                                UnsupportedIrFeature::ResourceBindings,
                            ));
                        }
                        texture
                    }
                    _ => return Err(ir::Error::BindingLayoutMismatch.into()),
                };
                let texture = resources.resources.texture_ref(texture)?;
                let descriptor = resources.resources.texture(texture)?;
                if !descriptor.usage().contains(TextureUsage::SAMPLED)
                    || texture == pass.attachment
                    || pass.additional_attachments.contains(&texture)
                    || pass.depth_attachment == Some(texture) && !pass.submission.depth_read_only
                {
                    return Err(ir::Error::InvalidUsage.into());
                }
                let dimension = if descriptor.cube_compatible() {
                    ir::TextureViewDimension::Cube
                } else if descriptor.array_layer_count() > 1 {
                    ir::TextureViewDimension::D2Array
                } else {
                    ir::TextureViewDimension::D2
                };
                if binding.dimension != dimension
                    || binding.depth != (descriptor.format() == TextureFormat::Depth32Float)
                {
                    return Err(ir::Error::BindingLayoutMismatch.into());
                }
                let sampler = if binding.uses_sampler {
                    let ir::BindingResource::Sampler(sampler) =
                        resource(binding.sampler_group, binding.sampler_binding)?
                    else {
                        return Err(ir::Error::BindingLayoutMismatch.into());
                    };
                    let sampler = resources.resources.sampler_ref(sampler)?;
                    let desc = resources.resources.sampler(sampler)?;
                    if desc.compare().is_some() != binding.comparison {
                        return Err(ir::Error::BindingLayoutMismatch.into());
                    }
                    Some(sampler_state(desc, sampler.slot()))
                } else {
                    None
                };
                textures.push(driver::IrTextureBinding {
                    stage: shader.stage,
                    slot: binding.slot,
                    texture: texture_spec(texture, descriptor)?,
                    sampler,
                });
            }
            if let Some(push) = shader
                .push_constants
                .as_ref()
                .filter(|_| cached_constants.is_none())
            {
                let stage = if shader.stage == ir::ShaderStage::Vertex {
                    0
                } else {
                    1
                };
                let words = constant_words(
                    &pass.push_constants[stage][..push.size as usize],
                    push.size.div_ceil(16) as usize * 16,
                )?;
                constants.push(driver::IrConstantBuffer {
                    stage: shader.stage,
                    first_register: push.first_register,
                    words,
                });
            }
            for binding in &shader.uniform_buffers {
                if cached_constants.is_some() {
                    continue;
                }
                let group_ref = pass
                    .bind_groups
                    .get(binding.group as usize)
                    .copied()
                    .flatten()
                    .ok_or(ir::Error::BindingLayoutMismatch)?;
                let group = resources.resources.bind_group_shared(group_ref)?;
                if pipeline.layout().bind_groups().get(binding.group as usize)
                    != Some(group.layout())
                {
                    return Err(ir::Error::BindingLayoutMismatch.into());
                }
                let entry = group
                    .entries()
                    .iter()
                    .find(|entry| entry.binding() == binding.binding)
                    .ok_or(ir::Error::BindingLayoutMismatch)?;
                let ir::BindingResource::Buffer {
                    buffer,
                    offset,
                    size,
                } = entry.resource()
                else {
                    return Err(ir::Error::BindingLayoutMismatch.into());
                };
                let constant_size = binding.size.next_multiple_of(16);
                if size < u64::from(binding.size)
                    || size > 16 * 1024
                    || offset > u64::from(u32::MAX)
                {
                    return Err(ir::Error::OutOfBounds.into());
                }
                let buffer = resources.resources.buffer_ref(buffer)?;
                let bytes = pending.bytes(resources, buffer)?;
                let end = offset
                    .checked_add(u64::from(binding.size))
                    .ok_or(ir::Error::Overflow)?;
                if end > bytes.as_slice().len() as u64 {
                    return Err(ir::Error::OutOfBounds.into());
                }
                let start = usize::try_from(offset).map_err(|_| ir::Error::Overflow)?;
                let end = usize::try_from(end).map_err(|_| ir::Error::Overflow)?;
                let words = constant_words(&bytes.as_slice()[start..end], constant_size as usize)?;
                constants
                    .try_reserve(1)
                    .map_err(|_| IrSubmitError::OutOfMemory)?;
                constants.push(driver::IrConstantBuffer {
                    stage: shader.stage,
                    first_register: binding.first_register,
                    words,
                });
            }
        }
    }
    let mut maximum_vertex = first
        .checked_add(count)
        .and_then(|end| end.checked_sub(1))
        .ok_or(ir::Error::Overflow)?;
    let index_buffer = if let Some(base_vertex) = base_vertex {
        let (buffer, offset, format) = pass.index_buffer.ok_or(ir::Error::IndexBufferNotSet)?;
        let bytes = pending.bytes(resources, buffer)?;
        let size = format.byte_size();
        let start = offset
            .checked_add(u64::from(first) * size)
            .ok_or(ir::Error::Overflow)?;
        let end = start
            .checked_add(u64::from(count) * size)
            .ok_or(ir::Error::Overflow)?;
        let range = bytes
            .as_slice()
            .get(
                usize::try_from(start).map_err(|_| ir::Error::Overflow)?
                    ..usize::try_from(end).map_err(|_| ir::Error::Overflow)?,
            )
            .ok_or(ir::Error::OutOfBounds)?;
        maximum_vertex = 0;
        for index in range.chunks_exact(size as usize) {
            let index = match format {
                IndexFormat::Uint16 => u32::from(read_u16(index, 0)?),
                IndexFormat::Uint32 => read_u32(index, 0)?,
            };
            let resolved = i64::from(index) + i64::from(base_vertex);
            maximum_vertex =
                maximum_vertex.max(u32::try_from(resolved).map_err(|_| ir::Error::OutOfBounds)?);
        }
        Some(driver::IrIndexBufferBinding {
            buffer: buffer_spec(resources, pending, buffer)?,
            offset: u32::try_from(offset).map_err(|_| ir::Error::Overflow)?,
            format,
            base_vertex,
        })
    } else {
        None
    };
    if instance_count == 0 || first_instance.checked_add(instance_count).is_none() {
        return Err(ir::Error::InvalidValue.into());
    }
    let mut vertex_buffers = [None; 8];
    for (slot, layout) in pipeline.vertex_buffers().iter().enumerate() {
        let (buffer, offset) = pass.vertex_buffers[slot].ok_or(ir::Error::VertexBufferNotSet)?;
        let bytes = pending.bytes(resources, buffer)?;
        let attribute_end = layout
            .attributes()
            .iter()
            .map(|a| u64::from(a.offset()) + u64::from(a.format().byte_size()))
            .max()
            .unwrap_or(0);
        let maximum = maximum_vertex;
        let end = offset
            .checked_add(u64::from(maximum) * u64::from(layout.stride()))
            .and_then(|start| start.checked_add(attribute_end))
            .ok_or(ir::Error::Overflow)?;
        if end > bytes.as_slice().len() as u64 {
            return Err(ir::Error::OutOfBounds.into());
        }
        vertex_buffers[slot] = Some(IrVertexBufferBinding {
            buffer: buffer_spec(resources, pending, buffer)?,
            offset: u32::try_from(offset).map_err(|_| ir::Error::Overflow)?,
        });
    }
    let vertex_buffer = vertex_buffers[0];
    let raster = pipeline.raster();
    let mut additional_color_blends = [None; 7];
    for (slot, target) in pipeline.color_targets().skip(1).enumerate() {
        additional_color_blends[slot] =
            Some((blend_state(target.blend()), target.write_mask().bits()));
    }
    // Writes are forbidden inside a render pass, so unchanged binding state
    // refers to the same owned constants. Bounds are still checked per draw.
    let programmable = if let Some(cached) = cached {
        draw_with_index(cached, index_buffer)
    } else {
        Rc::new(driver::IrProgrammableDraw {
            pipeline: compiled,
            index_buffer,
            constants: cached_constants.unwrap_or_else(|| Rc::from(constants.as_slice())),
            textures: Rc::from(textures.as_slice()),
            storage_buffers: storage_buffers.into(),
        })
    };
    pass.constants_scratch = constants;
    pass.textures_scratch = textures;
    pass.programmable_cache = Some(ProgrammableDrawCache {
        key,
        draw: Rc::clone(&programmable),
    });
    Ok(IrDraw {
        instance_count,
        first_instance,
        vertex_buffers,
        programmable: Some(programmable),
        start_vertex: first as usize,
        vertex_count: count as usize,
        vertex_buffer,
        pipeline: IrPipelineState {
            slot: reference.slot() + 256,
            fragment: IrFragmentProgram::Solid,
            blend: blend_state(pipeline.blend()),
            color_write_mask: pipeline.color_targets().next().unwrap().write_mask().bits(),
            additional_color_blends,
            cull_mode: match raster.cull_mode() {
                ir::CullMode::None => IrCullMode::None,
                ir::CullMode::Front => IrCullMode::Front,
                ir::CullMode::Back => IrCullMode::Back,
            },
            front_face: match raster.front_face() {
                ir::FrontFace::Clockwise => IrFrontFace::Clockwise,
                ir::FrontFace::CounterClockwise => IrFrontFace::CounterClockwise,
            },
            depth: pipeline.depth_state().map(|depth| IrDepthState {
                write_enabled: depth.write_enabled(),
                compare: match depth.compare() {
                    CompareFunction::Never => IrCompareFunction::Never,
                    CompareFunction::Less => IrCompareFunction::Less,
                    CompareFunction::Equal => IrCompareFunction::Equal,
                    CompareFunction::LessEqual => IrCompareFunction::LessEqual,
                    CompareFunction::Greater => IrCompareFunction::Greater,
                    CompareFunction::NotEqual => IrCompareFunction::NotEqual,
                    CompareFunction::GreaterEqual => IrCompareFunction::GreaterEqual,
                    CompareFunction::Always => IrCompareFunction::Always,
                },
            }),
        },
        texture: None,
        sampler: None,
        uniforms: IrUniforms {
            transform: [0.0; 16],
            color: [0.0; 4],
        },
        scissor: pass_scissor(pass),
        viewport: pass.viewport.map(ir::Viewport::components),
    })
}

#[cfg(feature = "programmable")]
fn draw_with_index(
    state: Rc<driver::IrProgrammableDraw>,
    index_buffer: Option<driver::IrIndexBufferBinding>,
) -> Rc<driver::IrProgrammableDraw> {
    if state.index_buffer == index_buffer {
        return state;
    }
    Rc::new(driver::IrProgrammableDraw {
        pipeline: Rc::clone(&state.pipeline),
        index_buffer,
        constants: Rc::clone(&state.constants),
        textures: Rc::clone(&state.textures),
        storage_buffers: Rc::clone(&state.storage_buffers),
    })
}

#[cfg(not(feature = "programmable"))]
pub(super) fn decode_draw(
    _resources: &IrResources,
    _pending: &PendingBuffers,
    _pass: &mut ActivePass<'_>,
    _first: u32,
    _count: u32,
    _base_vertex: Option<i32>,
    _instance_count: u32,
    _first_instance: u32,
) -> Result<IrDraw, IrSubmitError> {
    Err(IrSubmitError::Unsupported(
        UnsupportedIrFeature::ProgrammableExecution,
    ))
}

#[cfg(all(test, feature = "programmable"))]
mod tests {
    use super::*;
    use alloc::vec;
    const SHADER: &str = "struct Transform { mvp: mat4x4<f32>, }; @group(0) @binding(0) var<uniform> transform: Transform;
        struct Output { @builtin(position) position: vec4<f32>, @location(0) color: vec3<f32>, };
        @vertex fn vs(@location(0) position: vec3<f32>, @location(1) color: vec3<f32>) -> Output {
            var output: Output; output.position = transform.mvp * vec4<f32>(position, 1.0); output.color = color; return output;
        }
        @fragment fn fs(input: Output) -> @location(0) vec4<f32> { return vec4<f32>(input.color, 1.0); }";

    fn pipeline(
        table: &ResourceTable,
        vertex_format: VertexFormat,
        uniform_type: ir::BindingType,
    ) -> ir::ProgrammableRenderPipelineId {
        pipeline_source(table, vertex_format, uniform_type, SHADER)
    }

    fn pipeline_source(
        table: &ResourceTable,
        vertex_format: VertexFormat,
        uniform_type: ir::BindingType,
        source: &str,
    ) -> ir::ProgrammableRenderPipelineId {
        let module = table
            .define_shader_module(ir::ShaderModuleDesc::wgsl(source.into()).unwrap())
            .unwrap();
        let layout = ir::PipelineLayoutDesc::new(vec![
            ir::BindGroupLayoutDesc::new(vec![ir::BindGroupLayoutEntry::new(
                0,
                ir::ShaderStages::VERTEX,
                uniform_type,
            )])
            .unwrap(),
        ])
        .unwrap();
        table
            .define_programmable_render_pipeline(
                ir::ProgrammableRenderPipelineDesc::new(
                    ir::ShaderEntryPoint::new(module, ir::ShaderStage::Vertex, "vs".into())
                        .unwrap(),
                    ir::ShaderEntryPoint::new(module, ir::ShaderStage::Fragment, "fs".into())
                        .unwrap(),
                    layout,
                    TextureFormat::Rgba8Unorm,
                    Some(
                        ir::VertexBufferLayout::new(
                            32,
                            vec![
                                VertexAttribute::new(0, vertex_format, 0),
                                VertexAttribute::new(1, VertexFormat::Float32x3, 16),
                            ],
                        )
                        .unwrap(),
                    ),
                    PrimitiveTopology::TriangleList,
                    BlendState::REPLACE,
                    ir::RasterState::new(ir::CullMode::Back, ir::FrontFace::CounterClockwise),
                )
                .unwrap()
                .with_depth_state(ir::DepthState::new(
                    TextureFormat::Depth32Float,
                    CompareFunction::Less,
                    true,
                ))
                .unwrap(),
            )
            .unwrap()
            .id()
    }

    #[test]
    fn compiles_real_uniform_transformed_vertex_and_fragment_stages() {
        let table = ResourceTable::new();
        let id = pipeline(
            &table,
            VertexFormat::Float32x3,
            ir::BindingType::UniformBuffer,
        );
        let compiled = compile_pipeline(&table, id).unwrap();
        assert_eq!(compiled.vertex.uniform_buffers[0].size, 64);
        assert_eq!(compiled.vertex.uniform_buffers[0].first_register, 0);
        assert_eq!(compiled.vertex.vertex_inputs.len(), 2);
        assert!(compiled.vertex.tgsi.contains("CONST[1]"));
        assert!(compiled.fragment.tgsi.contains("GENERIC[0]"));
    }

    #[test]
    fn permits_fragment_output_without_color_attachment() {
        let table = ResourceTable::new();
        let source = "struct Transform { mvp: mat4x4<f32>, }; @group(0) @binding(0) var<uniform> transform: Transform;
            struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) color: vec3<f32>, };
            struct FragmentOutput { @location(0) color: vec4<f32>, @location(1) unused: vec4<f32>, };
            @vertex fn vs(@location(0) position: vec3<f32>, @location(1) color: vec3<f32>) -> VertexOutput {
                var output: VertexOutput; output.position = transform.mvp * vec4<f32>(position, 1.0);
                output.color = color; return output;
            }
            @fragment fn fs(input: VertexOutput) -> FragmentOutput {
                return FragmentOutput(vec4<f32>(input.color, 1.0), vec4<f32>(1.0));
            }";
        let id = pipeline_source(
            &table,
            VertexFormat::Float32x3,
            ir::BindingType::UniformBuffer,
            source,
        );
        let compiled = compile_pipeline(&table, id).unwrap();
        assert_eq!(compiled.fragment.output_locations, vec![0, 1]);
    }

    #[test]
    fn depth_only_fragment_stage_can_omit_all_color_outputs() {
        let table = ResourceTable::new();
        let source = SHADER.replace("@fragment fn fs(input: Output) -> @location(0) vec4<f32> { return vec4<f32>(input.color, 1.0); }", "@fragment fn fs() {}");
        let id = pipeline_source(
            &table,
            VertexFormat::Float32x3,
            ir::BindingType::UniformBuffer,
            &source,
        );
        assert!(
            compile_pipeline(&table, id)
                .unwrap()
                .fragment
                .output_locations
                .is_empty()
        );
    }

    #[test]
    fn readonly_storage_binding_compiles_as_a_buffer_view_with_descriptor_metadata() {
        let table = ResourceTable::new();
        let source = SHADER.replace("var<uniform>", "var<storage, read>");
        let id = pipeline_source(
            &table,
            VertexFormat::Float32x3,
            ir::BindingType::StorageBuffer { read_only: true },
            &source,
        );
        let compiled = compile_pipeline(&table, id).unwrap();
        assert!(compiled.vertex.uniform_buffers.is_empty());
        assert_eq!(compiled.vertex.storage_buffers.len(), 1);
        assert_eq!(compiled.vertex.storage_buffers[0].first_register, 0);
        assert!(compiled.vertex.tgsi.contains("BUFFER, UINT"));
    }

    #[test]
    fn small_constant_snapshots_preserve_words_and_register_padding() {
        for size in [4_usize, 116, 124, 128] {
            let expected: Vec<u32> = (0..size / 4).map(|i| 0x8000_0000 | i as u32).collect();
            let mut bytes: Vec<u8> = expected
                .iter()
                .flat_map(|word| word.to_ne_bytes())
                .collect();
            let snapshot = constant_words(&bytes, size.next_multiple_of(16)).unwrap();
            assert!(matches!(snapshot, driver::IrConstantWords::Inline { .. }));
            bytes.fill(0);
            assert_eq!(&snapshot[..expected.len()], expected.as_slice());
            assert!(snapshot[expected.len()..].iter().all(|word| *word == 0));
            let alternate = driver::IrConstantWords::Shared(Rc::from(snapshot.as_slice()));
            assert_eq!(snapshot, alternate);
        }
        assert!(constant_words(&[0; 3], 16).is_err());
        assert!(constant_words(&[0; 20], 16).is_err());
        assert!(constant_words(&[0; 16], 20).is_err());
    }

    #[test]
    fn larger_constant_snapshots_share_immutable_storage_without_truncation() {
        for size in [132_usize, 16 * 1024] {
            let expected: Vec<u32> = (0..size / 4).map(|i| i as u32 + 1).collect();
            let bytes: Vec<u8> = expected
                .iter()
                .flat_map(|word| word.to_ne_bytes())
                .collect();
            let snapshot = constant_words(&bytes, size.next_multiple_of(16)).unwrap();
            let retained = snapshot.clone();
            let (driver::IrConstantWords::Shared(original), driver::IrConstantWords::Shared(copy)) =
                (&snapshot, &retained)
            else {
                panic!("large constant banks must use shared storage");
            };
            assert!(Rc::ptr_eq(original, copy));
            drop(snapshot);
            assert_eq!(&retained[..expected.len()], expected.as_slice());
            assert!(retained[expected.len()..].iter().all(|word| *word == 0));
        }
    }

    #[test]
    fn shared_draw_state_detects_binding_changes_and_retains_older_snapshots() {
        let table = ResourceTable::new();
        let id = pipeline(
            &table,
            VertexFormat::Float32x3,
            ir::BindingType::UniformBuffer,
        );
        let other_id = pipeline(
            &table,
            VertexFormat::Float32x3,
            ir::BindingType::UniformBuffer,
        );
        let buffer = table
            .define_buffer(
                ir::BufferDesc::new(64, BufferUsage::UNIFORM | BufferUsage::INDEX).unwrap(),
            )
            .unwrap();
        let pipeline_ref = table.programmable_render_pipeline_ref(id).unwrap();
        let layout = table
            .programmable_render_pipeline_shared(pipeline_ref)
            .unwrap()
            .layout()
            .bind_groups()[0]
            .clone();
        let make_group = || {
            table
                .define_bind_group(
                    ir::BindGroupDesc::new(
                        &table,
                        layout.clone(),
                        vec![ir::BindGroupEntry::new(
                            0,
                            ir::BindingResource::Buffer {
                                buffer: buffer.id(),
                                offset: 0,
                                size: 64,
                            },
                        )],
                    )
                    .unwrap(),
                )
                .unwrap()
        };
        let group = make_group();
        let other_group = make_group();
        let mut key = ProgrammableDrawKey {
            pipeline: pipeline_ref,
            bind_groups: [None; ir::MAX_BIND_GROUPS],
            push_constants: [[0; 128]; 2],
        };
        key.bind_groups[0] = Some(group);
        let snapshot = Rc::new(driver::IrProgrammableDraw {
            pipeline: compile_pipeline(&table, id).unwrap(),
            index_buffer: None,
            constants: vec![driver::IrConstantBuffer {
                stage: ir::ShaderStage::Vertex,
                first_register: 0,
                words: vec![42; 16].into(),
            }]
            .into(),
            textures: Vec::new().into(),
            storage_buffers: Vec::new().into(),
        });
        let cache = ProgrammableDrawCache {
            key,
            draw: Rc::clone(&snapshot),
        };
        let older = cache.get(&key).unwrap();
        assert!(Rc::ptr_eq(&older, &snapshot));
        let pipeline_desc = table
            .programmable_render_pipeline_shared(pipeline_ref)
            .unwrap();
        let mut texture_only_change = key;
        texture_only_change.bind_groups[0] = Some(other_group);
        let shared_constants = cache
            .constants(&table, &texture_only_change, pipeline_desc.layout())
            .unwrap()
            .unwrap();
        assert!(Rc::ptr_eq(&shared_constants, &snapshot.constants));
        let other_buffer = table
            .define_buffer(ir::BufferDesc::new(64, BufferUsage::UNIFORM).unwrap())
            .unwrap();
        let changed_uniform = table
            .define_bind_group(
                ir::BindGroupDesc::new(
                    &table,
                    layout.clone(),
                    vec![ir::BindGroupEntry::new(
                        0,
                        ir::BindingResource::Buffer {
                            buffer: other_buffer.id(),
                            offset: 0,
                            size: 64,
                        },
                    )],
                )
                .unwrap(),
            )
            .unwrap();
        let mut changed_key = key;
        changed_key.bind_groups[0] = Some(changed_uniform);
        assert!(
            cache
                .constants(&table, &changed_key, pipeline_desc.layout())
                .unwrap()
                .is_none()
        );
        let incompatible = table
            .define_bind_group(
                ir::BindGroupDesc::new(
                    &table,
                    ir::BindGroupLayoutDesc::new(vec![]).unwrap(),
                    vec![],
                )
                .unwrap(),
            )
            .unwrap();
        changed_key.bind_groups[0] = Some(incompatible);
        assert!(matches!(
            cache.constants(&table, &changed_key, pipeline_desc.layout()),
            Err(IrSubmitError::InvalidIr(ir::Error::BindingLayoutMismatch))
        ));
        let mut changes = [key; 4];
        changes[0].pipeline = table.programmable_render_pipeline_ref(other_id).unwrap();
        changes[1].bind_groups[0] = Some(other_group);
        changes[2].push_constants[0][0] = 1;
        changes[3].push_constants[1][0] = 1;
        for (index, changed) in changes.into_iter().enumerate() {
            assert!(cache.get(&changed).is_none());
            if index != 1 {
                assert!(
                    cache
                        .constants(&table, &changed, pipeline_desc.layout())
                        .unwrap()
                        .is_none()
                );
            }
        }
        drop(cache);
        drop(snapshot);
        assert_eq!(older.constants[0].words.as_slice(), &[42; 16]);
        let index = driver::IrIndexBufferBinding {
            buffer: IrBufferSpec {
                slot: buffer.slot(),
                size: 64,
                revision: 1,
            },
            offset: 2,
            format: IndexFormat::Uint16,
            base_vertex: 7,
        };
        let indexed = draw_with_index(Rc::clone(&older), Some(index));
        assert!(!Rc::ptr_eq(&indexed, &older));
        assert!(Rc::ptr_eq(&indexed.constants, &older.constants));
        assert!(Rc::ptr_eq(&indexed.textures, &older.textures));
        assert!(older.index_buffer.is_none());
        assert_eq!(indexed.index_buffer.unwrap().base_vertex, 7);
        let same_index = draw_with_index(Rc::clone(&indexed), Some(index));
        assert!(Rc::ptr_eq(&same_index, &indexed));
    }

    #[test]
    fn rejects_vertex_fetch_type_mismatch_before_materialization() {
        let table = ResourceTable::new();
        let id = pipeline(
            &table,
            VertexFormat::Float32x2,
            ir::BindingType::UniformBuffer,
        );
        assert!(matches!(
            compile_pipeline(&table, id),
            Err(IrSubmitError::InvalidIr(ir::Error::InvalidDescriptor))
        ));
    }

    #[test]
    fn rejects_storage_layout_for_uniform_shader_binding() {
        let table = ResourceTable::new();
        let id = pipeline(
            &table,
            VertexFormat::Float32x3,
            ir::BindingType::StorageBuffer { read_only: true },
        );
        assert!(matches!(
            compile_pipeline(&table, id),
            Err(IrSubmitError::InvalidIr(ir::Error::BindingLayoutMismatch))
        ));
    }

    #[test]
    fn admits_ordered_transfer_barriers_but_rejects_shader_write_barriers() {
        let table = ResourceTable::new();
        let buffer = table
            .define_buffer(
                ir::BufferDesc::new(
                    64,
                    BufferUsage::COPY_DST | BufferUsage::VERTEX | BufferUsage::STORAGE,
                )
                .unwrap(),
            )
            .unwrap();
        validate_barrier(ir::ResourceBarrier::Buffer {
            buffer,
            before: ir::BufferAccess::CopyDestination,
            after: ir::BufferAccess::Vertex,
        })
        .unwrap();
        assert!(matches!(
            validate_barrier(ir::ResourceBarrier::Buffer {
                buffer,
                before: ir::BufferAccess::StorageReadWrite,
                after: ir::BufferAccess::Vertex
            }),
            Err(IrSubmitError::Unsupported(
                UnsupportedIrFeature::ExplicitBarrier
            ))
        ));
    }
}
