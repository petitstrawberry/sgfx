//! Validation and lowering for the native programmable graphics subset.

use super::*;
use sgfx_codegen_virgl::programmable::{compile_shader, validate_shader_module};

impl IrResources {
    /// Parse and semantically validate a shader before admitting its handle.
    pub fn validate_shader_module(&self, id: ir::ShaderModuleId) -> Result<(), IrSubmitError> {
        let module = self
            .resources
            .shader_module(self.resources.shader_module_ref(id)?)?;
        validate_shader_module(&module).map_err(IrSubmitError::ShaderCompile)
    }

    /// Compile both stages and validate the supported native graphics state.
    pub fn validate_programmable_render_pipeline(
        &self,
        id: ir::ProgrammableRenderPipelineId,
    ) -> Result<(), IrSubmitError> {
        self.compiled_pipeline(id).map(|_| ())
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
}

fn compile_pipeline(
    resources: &ResourceTable,
    id: ir::ProgrammableRenderPipelineId,
) -> Result<Rc<driver::IrProgrammablePipeline>, IrSubmitError> {
    let reference = resources.programmable_render_pipeline_ref(id)?;
    let pipeline = resources.programmable_render_pipeline(reference)?;
    if !matches!(
        pipeline.target_format(),
        TextureFormat::Bgra8Unorm | TextureFormat::Rgba8Unorm
    ) {
        return Err(IrSubmitError::Unsupported(
            UnsupportedIrFeature::PipelineTargetFormat,
        ));
    }
    if pipeline.topology() != PrimitiveTopology::TriangleList {
        return Err(IrSubmitError::Unsupported(
            UnsupportedIrFeature::PrimitiveTopology,
        ));
    }
    if let Some(layout) = pipeline.vertex_buffer()
        && layout
            .attributes()
            .iter()
            .enumerate()
            .any(|(index, attribute)| {
                attribute.location() != index as u32
                    || !matches!(
                        attribute.format(),
                        VertexFormat::Float32x2
                            | VertexFormat::Float32x3
                            | VertexFormat::Float32x4
                            | VertexFormat::Unorm8x4
                    )
            })
    {
        return Err(IrSubmitError::Unsupported(
            UnsupportedIrFeature::VertexLayout,
        ));
    }
    for group in pipeline.layout().bind_groups() {
        if group
            .entries()
            .iter()
            .any(|binding| binding.ty() != ir::BindingType::UniformBuffer)
        {
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
    if vertex.input_locations.iter().any(|location| {
        pipeline.vertex_buffer().is_none_or(|layout| {
            !layout
                .attributes()
                .iter()
                .any(|attribute| attribute.location() == *location)
        })
    }) || fragment
        .input_locations
        .iter()
        .any(|location| !vertex.output_locations.contains(location))
        || fragment.output_locations != [0]
    {
        return Err(ir::Error::InvalidDescriptor.into());
    }
    for (location, expected) in &vertex.vertex_inputs {
        let attribute = pipeline
            .vertex_buffer()
            .and_then(|layout| {
                layout
                    .attributes()
                    .iter()
                    .find(|attribute| attribute.location() == *location)
            })
            .ok_or(ir::Error::InvalidDescriptor)?;
        if attribute.format() != *expected
            && !(attribute.format() == VertexFormat::Unorm8x4
                && *expected == VertexFormat::Float32x4)
        {
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
    // Keep setup, both shader texts, bindings and one draw within a native packet.
    if vertex.tgsi.len().saturating_add(fragment.tgsi.len()) > 32 * 1024 {
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
    }
    let compiled = Rc::new(driver::IrProgrammablePipeline {
        slot: reference.slot(),
        vertex,
        fragment,
        vertex_buffer: pipeline.vertex_buffer().cloned(),
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
        let spec = texture_spec(reference, descriptor);
        let mut bytes = self
            .backend
            .readback_ir_texture(&mut resources.backend, spec)?;
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
            if [before, after].iter().any(|access| {
                matches!(
                    access,
                    ir::BufferAccess::StorageRead | ir::BufferAccess::StorageReadWrite
                )
            }) =>
        {
            Err(IrSubmitError::Unsupported(
                UnsupportedIrFeature::ExplicitBarrier,
            ))
        }
        ir::ResourceBarrier::Texture { before, after, .. }
            if [before, after].contains(&ir::TextureAccess::StorageWrite) =>
        {
            Err(IrSubmitError::Unsupported(
                UnsupportedIrFeature::ExplicitBarrier,
            ))
        }
        _ => Ok(()),
    }
}

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

pub(super) fn decode_draw(
    resources: &IrResources,
    pending: &PendingBuffers,
    pass: &ActivePass<'_>,
    first: u32,
    count: u32,
    base_vertex: Option<i32>,
) -> Result<IrDraw, IrSubmitError> {
    if count == 0 || !count.is_multiple_of(3) {
        return Err(ir::Error::InvalidValue.into());
    }
    let reference = pass.programmable.ok_or(ir::Error::PipelineNotSet)?;
    let pipeline = resources
        .resources
        .programmable_render_pipeline(reference)?;
    let target = resources.resources.texture(pass.attachment)?;
    if target.format() != pipeline.target_format()
        || pipeline.depth_state().is_some() && pass.depth_attachment.is_none()
    {
        return Err(ir::Error::InvalidDescriptor.into());
    }
    let compiled = resources.compiled_pipeline(reference.id())?;
    let mut constants = Vec::new();
    for shader in [&compiled.vertex, &compiled.fragment] {
        for binding in &shader.uniform_buffers {
            let group_ref = pass
                .bind_groups
                .get(binding.group as usize)
                .copied()
                .flatten()
                .ok_or(ir::Error::BindingLayoutMismatch)?;
            let group = resources.resources.bind_group(group_ref)?;
            if pipeline.layout().bind_groups().get(binding.group as usize) != Some(group.layout()) {
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
            if size < u64::from(constant_size)
                || !size.is_multiple_of(16)
                || size > 16 * 1024
                || offset > u64::from(u32::MAX)
            {
                return Err(ir::Error::OutOfBounds.into());
            }
            let buffer = resources.resources.buffer_ref(buffer)?;
            let bytes = pending.bytes(resources, buffer)?;
            let end = offset
                .checked_add(u64::from(constant_size))
                .ok_or(ir::Error::Overflow)?;
            if end > bytes.as_slice().len() as u64 {
                return Err(ir::Error::OutOfBounds.into());
            }
            let start = usize::try_from(offset).map_err(|_| ir::Error::Overflow)?;
            let end = usize::try_from(end).map_err(|_| ir::Error::Overflow)?;
            let mut words = Vec::new();
            words
                .try_reserve_exact(constant_size as usize / 4)
                .map_err(|_| IrSubmitError::OutOfMemory)?;
            words.extend(
                bytes.as_slice()[start..end]
                    .chunks_exact(4)
                    .map(|word| u32::from_ne_bytes([word[0], word[1], word[2], word[3]])),
            );
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
    let vertex_buffer = if let Some(layout) = pipeline.vertex_buffer() {
        let (buffer, offset) = pass.vertex_buffer.ok_or(ir::Error::VertexBufferNotSet)?;
        let bytes = pending.bytes(resources, buffer)?;
        let attribute_end = layout
            .attributes()
            .iter()
            .map(|attribute| {
                u64::from(attribute.offset()) + u64::from(attribute.format().byte_size())
            })
            .max()
            .unwrap_or(0);
        let end = offset
            .checked_add(u64::from(maximum_vertex) * u64::from(layout.stride()))
            .and_then(|start| start.checked_add(attribute_end))
            .ok_or(ir::Error::Overflow)?;
        if end > bytes.as_slice().len() as u64 {
            return Err(ir::Error::OutOfBounds.into());
        }
        Some(IrVertexBufferBinding {
            buffer: buffer_spec(resources, pending, buffer)?,
            offset: u32::try_from(offset).map_err(|_| ir::Error::Overflow)?,
        })
    } else {
        None
    };
    let raster = pipeline.raster();
    Ok(IrDraw {
        programmable: Some(Rc::new(driver::IrProgrammableDraw {
            pipeline: compiled,
            index_buffer,
            constants,
        })),
        start_vertex: first as usize,
        vertex_count: count as usize,
        vertex_buffer,
        pipeline: IrPipelineState {
            slot: reference.slot() + 256,
            fragment: IrFragmentProgram::Solid,
            blend: blend_state(pipeline.blend()),
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
    })
}

#[cfg(test)]
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
        let module = table
            .define_shader_module(ir::ShaderModuleDesc::wgsl(SHADER.into()).unwrap())
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
    fn rejects_storage_layout_even_when_shader_does_not_write() {
        let table = ResourceTable::new();
        let id = pipeline(
            &table,
            VertexFormat::Float32x3,
            ir::BindingType::StorageBuffer { read_only: true },
        );
        assert!(matches!(
            compile_pipeline(&table, id),
            Err(IrSubmitError::Unsupported(
                UnsupportedIrFeature::ResourceBindings
            ))
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
