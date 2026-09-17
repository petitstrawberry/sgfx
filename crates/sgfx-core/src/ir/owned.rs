//! Owned command recordings for frontends that retain resources by identity.
//!
//! An owned recording is not a validated command buffer. Replaying it through
//! [`OwnedCommandBuffer::record`] resolves every identity against the supplied
//! resource table and uses the same encoder validation as borrowed recording.

use alloc::vec::Vec;

use super::{
    BindGroupId, BufferAccess, BufferId, CommandBuffer, CommandEncoder, ComputePipelineId,
    DepthLoadOp, DrawUniforms, Error, FilterMode, IndexFormat, LoadOp, PixelRect,
    ProgrammableRenderPipelineId, RenderPassDesc, RenderPipelineId, ResourceBarrier, ResourceTable,
    Result, SamplerId, ShaderStages, StoreOp, TextureAccess, TextureId, TextureWrite, Viewport,
};

/// An owned depth attachment, validated when its recording is replayed.
#[derive(Debug, Clone, Copy)]
pub struct OwnedDepthAttachment {
    /// Depth attachment identity in the replay resource table.
    pub target: TextureId,
    /// Initial depth attachment operation.
    pub load: DepthLoadOp,
    /// Final depth attachment operation.
    pub store: StoreOp,
}

/// An owned render-pass descriptor, validated when its recording is replayed.
#[derive(Debug, Clone, Copy)]
pub struct OwnedRenderPassDesc {
    /// Color attachment identity in the replay resource table.
    pub target: TextureId,
    /// Non-empty render area.
    pub area: PixelRect,
    /// Initial color attachment operation.
    pub load: LoadOp,
    /// Final color attachment operation.
    pub store: StoreOp,
    /// Optional depth attachment.
    pub depth: Option<OwnedDepthAttachment>,
}

/// One additional owned color attachment, resolved against the replay table.
#[derive(Debug, Clone, Copy)]
pub struct OwnedColorAttachment {
    /// Allocation identity.
    pub target: TextureId,
    /// Initial contents operation.
    pub load: LoadOp,
    /// Final contents operation.
    pub store: StoreOp,
}

impl OwnedRenderPassDesc {
    fn resolve(self, resources: &ResourceTable) -> Result<RenderPassDesc<'_>> {
        let mut desc = RenderPassDesc::new(
            resources,
            resources.texture_ref(self.target)?,
            self.area,
            self.load,
            self.store,
        )?;
        if let Some(depth) = self.depth {
            desc = desc.with_depth_attachment(
                resources,
                resources.texture_ref(depth.target)?,
                depth.load,
                depth.store,
            )?;
        }
        Ok(desc)
    }
}

/// An owned resource transition, validated when its recording is replayed.
#[derive(Debug, Clone, Copy)]
pub enum OwnedResourceBarrier {
    /// Change the declared access for one mip level.
    TextureMip {
        /// Texture identity.
        texture: TextureId,
        /// Mip level.
        mip_level: u32,
        /// Access preceding the transition.
        before: TextureAccess,
        /// Access following the transition.
        after: TextureAccess,
    },
    /// Change the declared access for a buffer.
    Buffer {
        /// Buffer identity in the replay resource table.
        buffer: BufferId,
        /// Access preceding the transition.
        before: BufferAccess,
        /// Access following the transition.
        after: BufferAccess,
    },
    /// Change the declared access for a texture.
    Texture {
        /// Texture identity in the replay resource table.
        texture: TextureId,
        /// Access preceding the transition.
        before: TextureAccess,
        /// Access following the transition.
        after: TextureAccess,
    },
}

impl OwnedResourceBarrier {
    fn resolve(self, resources: &ResourceTable) -> Result<ResourceBarrier<'_>> {
        Ok(match self {
            Self::TextureMip {
                texture,
                mip_level,
                before,
                after,
            } => ResourceBarrier::TextureMip {
                texture: resources.texture_ref(texture)?,
                mip_level,
                before,
                after,
            },
            Self::Buffer {
                buffer,
                before,
                after,
            } => ResourceBarrier::Buffer {
                buffer: resources.buffer_ref(buffer)?,
                before,
                after,
            },
            Self::Texture {
                texture,
                before,
                after,
            } => ResourceBarrier::Texture {
                texture: resources.texture_ref(texture)?,
                before,
                after,
            },
        })
    }
}

/// One unvalidated, owned logical command.
///
/// Resource identities carry their owning table's identity. Upload bytes are
/// retained in the recording, so replayed commands borrow their data safely.
#[derive(Debug, Clone)]
pub enum OwnedCommand {
    /// Upload pixels to one explicitly selected layer and mip level.
    WriteTextureLayer {
        /// Destination allocation.
        texture: TextureId,
        /// Destination mip level.
        mip_level: u32,
        /// Destination array layer.
        array_layer: u32,
        /// Destination rectangle.
        destination: PixelRect,
        /// Source row stride.
        bytes_per_row: u32,
        /// Owned source pixels.
        data: Vec<u8>,
    },
    /// Upload owned pixels to an explicitly selected mip level.
    WriteTextureMip {
        /// Destination texture.
        texture: TextureId,
        /// Destination mip level.
        mip_level: u32,
        /// Destination rectangle.
        destination: PixelRect,
        /// Source row stride.
        bytes_per_row: u32,
        /// Owned pixels.
        data: Vec<u8>,
    },
    /// Scale a complete source mip into a complete destination mip on the GPU.
    BlitTexture {
        /// Source texture.
        source: TextureId,
        /// Source mip level.
        source_mip: u32,
        /// Destination texture.
        destination: TextureId,
        /// Destination mip level.
        destination_mip: u32,
        /// Texel filter.
        filter: FilterMode,
    },
    /// Upload owned bytes into a logical buffer.
    WriteBuffer {
        /// Destination buffer.
        buffer: BufferId,
        /// Destination byte offset.
        offset: u64,
        /// Owned source bytes.
        data: Vec<u8>,
    },
    /// Upload owned pixels into a logical texture.
    WriteTexture {
        /// Destination texture.
        texture: TextureId,
        /// Destination rectangle.
        destination: PixelRect,
        /// Source row stride in bytes.
        bytes_per_row: u32,
        /// Owned source pixels.
        data: Vec<u8>,
    },
    /// Copy a byte range between buffers.
    CopyBufferToBuffer {
        /// Source buffer.
        source: BufferId,
        /// Source byte offset.
        source_offset: u64,
        /// Destination buffer.
        destination: BufferId,
        /// Destination byte offset.
        destination_offset: u64,
        /// Number of bytes to copy.
        size: u64,
    },
    /// Copy equal-sized rectangles between textures.
    CopyTextureToTexture {
        /// Source texture.
        source: TextureId,
        /// Source rectangle.
        source_rect: PixelRect,
        /// Destination texture.
        destination: TextureId,
        /// Destination rectangle.
        destination_rect: PixelRect,
    },
    /// Declare an access transition outside a pass.
    ResourceBarrier(OwnedResourceBarrier),
    /// Begin a render pass.
    BeginRenderPass(OwnedRenderPassDesc),
    /// Begin a pass with additional color outputs or read-only depth.
    BeginRenderPassWithAttachments {
        /// First color output, render area and optional depth.
        desc: OwnedRenderPassDesc,
        /// Additional color outputs, in location order.
        colors: Vec<OwnedColorAttachment>,
        /// Permit depth reads by shaders and prohibit depth writes.
        read_only_depth: bool,
    },
    /// End the active render pass.
    EndRenderPass,
    /// Bind a fixed render pipeline.
    SetPipeline(RenderPipelineId),
    /// Bind a programmable render pipeline.
    SetProgrammablePipeline(ProgrammableRenderPipelineId),
    /// Bind a group for the active render or compute pass.
    SetBindGroup {
        /// Pipeline layout group index.
        index: u32,
        /// Bind group identity.
        bind_group: BindGroupId,
    },
    /// Bind a vertex buffer for the active render pass.
    SetVertexBuffer {
        /// Vertex buffer identity.
        buffer: BufferId,
        /// Byte offset of the first vertex.
        offset: u64,
    },
    /// Bind a numbered programmable vertex buffer slot.
    SetVertexBufferSlot {
        /// Zero-based slot.
        slot: u32,
        /// Vertex buffer identity.
        buffer: BufferId,
        /// Byte offset of vertex zero.
        offset: u64,
    },
    /// Bind an index buffer for the active render pass.
    SetIndexBuffer {
        /// Index buffer identity.
        buffer: BufferId,
        /// Byte offset of the first index.
        offset: u64,
        /// Encoded index width.
        format: IndexFormat,
    },
    /// Bind a sampled texture for the active render pass.
    SetTexture(TextureId),
    /// Bind a sampler for the active render pass.
    SetSampler(SamplerId),
    /// Set fixed render pipeline uniforms.
    SetUniforms(DrawUniforms),
    /// Set or reset the render scissor.
    SetScissor(Option<PixelRect>),
    /// Set the viewport for subsequent draws.
    SetViewport(Viewport),
    /// Update owned push-constant bytes for the selected programmable pipeline.
    SetPushConstants {
        /// Updated shader stages.
        stages: ShaderStages,
        /// First byte in the stage's push-constant block.
        offset: u32,
        /// Owned bytes.
        data: Vec<u8>,
    },
    /// Issue a non-indexed draw.
    Draw {
        /// Number of vertices.
        vertex_count: u32,
        /// First vertex relative to the bound buffer.
        first_vertex: u32,
    },
    /// Issue an instanced programmable draw.
    DrawInstanced {
        /// Number of vertices per instance.
        vertex_count: u32,
        /// First vertex.
        first_vertex: u32,
        /// Number of instances.
        instance_count: u32,
        /// First shader instance index.
        first_instance: u32,
    },
    /// Issue an indexed instanced programmable draw.
    DrawIndexedInstanced {
        /// Number of indices per instance.
        index_count: u32,
        /// First index.
        first_index: u32,
        /// Signed vertex offset.
        base_vertex: i32,
        /// Number of instances.
        instance_count: u32,
        /// First shader instance index.
        first_instance: u32,
    },
    /// Issue an indexed draw.
    DrawIndexed {
        /// Number of indices.
        index_count: u32,
        /// First index relative to the bound buffer.
        first_index: u32,
        /// Signed vertex offset.
        base_vertex: i32,
    },
    /// Begin a compute pass.
    BeginComputePass,
    /// End the active compute pass.
    EndComputePass,
    /// Bind a compute pipeline.
    SetComputePipeline(ComputePipelineId),
    /// Dispatch workgroups in the active compute pass.
    Dispatch {
        /// Workgroup count along the x axis.
        x: u32,
        /// Workgroup count along the y axis.
        y: u32,
        /// Workgroup count along the z axis.
        z: u32,
    },
}

/// An immutable, owned command recording that can be validated and replayed.
///
/// Construction retains the commands without validating them. A backend must
/// consume the validated [`CommandBuffer`] returned by [`Self::record`], rather
/// than executing the raw commands returned by [`Self::commands`].
#[derive(Debug, Clone, Default)]
pub struct OwnedCommandBuffer {
    commands: Vec<OwnedCommand>,
}

impl OwnedCommandBuffer {
    /// Retain an owned command sequence for later validation and replay.
    pub const fn new(commands: Vec<OwnedCommand>) -> Self {
        Self { commands }
    }

    /// Return the immutable, unvalidated command sequence.
    pub fn commands(&self) -> &[OwnedCommand] {
        &self.commands
    }

    /// Return the number of commands retained by this recording.
    pub const fn command_count(&self) -> usize {
        self.commands.len()
    }

    /// Return whether the recording contains no commands.
    pub const fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Validate this recording using the canonical borrowed command encoder.
    ///
    /// No backend work is performed. Errors are the same as [`Self::record`].
    pub fn validate(&self, resources: &ResourceTable) -> Result<()> {
        self.record(resources).map(|_| ())
    }

    /// Resolve resource identities and record a validated borrowed buffer.
    ///
    /// Resource ownership, usage, bounds, pipeline state, bindings, and command
    /// capacity are checked by [`CommandEncoder`] and its pass encoders. An
    /// unmatched or incorrectly nested pass command is rejected with
    /// [`Error::InvalidDescriptor`]. Upload bytes borrow this recording, while
    /// resource references borrow `resources`; neither is copied or extended.
    pub fn record<'r, 'data>(
        &'data self,
        resources: &'r ResourceTable,
    ) -> Result<CommandBuffer<'r, 'data>> {
        let mut encoder = CommandEncoder::new(resources);
        let mut commands = self.commands.iter();
        while let Some(command) = commands.next() {
            match command {
                OwnedCommand::WriteTextureLayer {
                    texture,
                    mip_level,
                    array_layer,
                    destination,
                    bytes_per_row,
                    data,
                } => encoder.write_texture(
                    resources.texture_ref(*texture)?,
                    TextureWrite::new(*destination, *bytes_per_row, data)?
                        .with_mip_level(*mip_level)
                        .with_array_layer(*array_layer),
                )?,
                OwnedCommand::WriteTextureMip {
                    texture,
                    mip_level,
                    destination,
                    bytes_per_row,
                    data,
                } => encoder.write_texture(
                    resources.texture_ref(*texture)?,
                    TextureWrite::new(*destination, *bytes_per_row, data)?
                        .with_mip_level(*mip_level),
                )?,
                OwnedCommand::BlitTexture {
                    source,
                    source_mip,
                    destination,
                    destination_mip,
                    filter,
                } => encoder.blit_texture(
                    resources.texture_ref(*source)?,
                    *source_mip,
                    resources.texture_ref(*destination)?,
                    *destination_mip,
                    *filter,
                )?,
                OwnedCommand::WriteBuffer {
                    buffer,
                    offset,
                    data,
                } => encoder.write_buffer(resources.buffer_ref(*buffer)?, *offset, data)?,
                OwnedCommand::WriteTexture {
                    texture,
                    destination,
                    bytes_per_row,
                    data,
                } => encoder.write_texture(
                    resources.texture_ref(*texture)?,
                    TextureWrite::new(*destination, *bytes_per_row, data)?,
                )?,
                OwnedCommand::CopyBufferToBuffer {
                    source,
                    source_offset,
                    destination,
                    destination_offset,
                    size,
                } => encoder.copy_buffer_to_buffer(
                    resources.buffer_ref(*source)?,
                    *source_offset,
                    resources.buffer_ref(*destination)?,
                    *destination_offset,
                    *size,
                )?,
                OwnedCommand::CopyTextureToTexture {
                    source,
                    source_rect,
                    destination,
                    destination_rect,
                } => encoder.copy_texture_to_texture(
                    resources.texture_ref(*source)?,
                    *source_rect,
                    resources.texture_ref(*destination)?,
                    *destination_rect,
                )?,
                OwnedCommand::ResourceBarrier(barrier) => {
                    encoder.resource_barrier(barrier.resolve(resources)?)?;
                }
                OwnedCommand::BeginRenderPass(desc)
                | OwnedCommand::BeginRenderPassWithAttachments { desc, .. } => {
                    let mut resolved = desc.resolve(resources)?;
                    if let OwnedCommand::BeginRenderPassWithAttachments {
                        colors,
                        read_only_depth,
                        ..
                    } = command
                    {
                        for color in colors {
                            resolved = resolved.with_color_attachment(
                                resources,
                                resources.texture_ref(color.target)?,
                                color.load,
                                color.store,
                            )?;
                        }
                        if *read_only_depth {
                            resolved = resolved.with_read_only_depth()?;
                        }
                    }
                    let mut pass = encoder.begin_render_pass(resolved)?;
                    let mut ended = false;
                    for command in commands.by_ref() {
                        match command {
                            OwnedCommand::EndRenderPass => {
                                ended = true;
                                break;
                            }
                            OwnedCommand::SetPipeline(pipeline) => {
                                pass.set_pipeline(resources.render_pipeline_ref(*pipeline)?)?;
                            }
                            OwnedCommand::SetProgrammablePipeline(pipeline) => {
                                pass.set_programmable_pipeline(
                                    resources.programmable_render_pipeline_ref(*pipeline)?,
                                )?;
                            }
                            OwnedCommand::SetBindGroup { index, bind_group } => {
                                pass.set_bind_group(
                                    *index,
                                    resources.bind_group_ref(*bind_group)?,
                                )?;
                            }
                            OwnedCommand::SetVertexBuffer { buffer, offset } => {
                                pass.set_vertex_buffer(resources.buffer_ref(*buffer)?, *offset)?;
                            }
                            OwnedCommand::SetVertexBufferSlot {
                                slot,
                                buffer,
                                offset,
                            } => {
                                pass.set_vertex_buffer_slot(
                                    *slot,
                                    resources.buffer_ref(*buffer)?,
                                    *offset,
                                )?;
                            }
                            OwnedCommand::SetIndexBuffer {
                                buffer,
                                offset,
                                format,
                            } => pass.set_index_buffer(
                                resources.buffer_ref(*buffer)?,
                                *offset,
                                *format,
                            )?,
                            OwnedCommand::SetTexture(texture) => {
                                pass.set_texture(resources.texture_ref(*texture)?)?;
                            }
                            OwnedCommand::SetSampler(sampler) => {
                                pass.set_sampler(resources.sampler_ref(*sampler)?)?;
                            }
                            OwnedCommand::SetUniforms(uniforms) => pass.set_uniforms(*uniforms)?,
                            OwnedCommand::SetScissor(scissor) => pass.set_scissor(*scissor)?,
                            OwnedCommand::SetViewport(viewport) => pass.set_viewport(*viewport)?,
                            OwnedCommand::SetPushConstants {
                                stages,
                                offset,
                                data,
                            } => pass.set_push_constants(*stages, *offset, data)?,
                            OwnedCommand::Draw {
                                vertex_count,
                                first_vertex,
                            } => pass.draw(*vertex_count, *first_vertex)?,
                            OwnedCommand::DrawIndexed {
                                index_count,
                                first_index,
                                base_vertex,
                            } => pass.draw_indexed(*index_count, *first_index, *base_vertex)?,
                            OwnedCommand::DrawInstanced {
                                vertex_count,
                                first_vertex,
                                instance_count,
                                first_instance,
                            } => pass.draw_instanced(
                                *vertex_count,
                                *first_vertex,
                                *instance_count,
                                *first_instance,
                            )?,
                            OwnedCommand::DrawIndexedInstanced {
                                index_count,
                                first_index,
                                base_vertex,
                                instance_count,
                                first_instance,
                            } => pass.draw_indexed_instanced(
                                *index_count,
                                *first_index,
                                *base_vertex,
                                *instance_count,
                                *first_instance,
                            )?,
                            _ => return Err(Error::InvalidDescriptor),
                        }
                    }
                    if !ended {
                        return Err(Error::InvalidDescriptor);
                    }
                    pass.end()?;
                }
                OwnedCommand::BeginComputePass => {
                    let mut pass = encoder.begin_compute_pass()?;
                    let mut ended = false;
                    for command in commands.by_ref() {
                        match command {
                            OwnedCommand::EndComputePass => {
                                ended = true;
                                break;
                            }
                            OwnedCommand::SetComputePipeline(pipeline) => {
                                pass.set_pipeline(resources.compute_pipeline_ref(*pipeline)?)?;
                            }
                            OwnedCommand::SetBindGroup { index, bind_group } => {
                                pass.set_bind_group(
                                    *index,
                                    resources.bind_group_ref(*bind_group)?,
                                )?;
                            }
                            OwnedCommand::Dispatch { x, y, z } => pass.dispatch(*x, *y, *z)?,
                            OwnedCommand::SetPushConstants {
                                stages,
                                offset,
                                data,
                            } => pass.set_push_constants(*stages, *offset, data)?,
                            _ => return Err(Error::InvalidDescriptor),
                        }
                    }
                    if !ended {
                        return Err(Error::InvalidDescriptor);
                    }
                    pass.end()?;
                }
                _ => return Err(Error::InvalidDescriptor),
            }
        }
        encoder.finish()
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::String;
    use alloc::vec;

    use super::*;
    use crate::ir::{
        AddressMode, BindGroupDesc, BindGroupEntry, BindGroupLayoutDesc, BindGroupLayoutEntry,
        BindingResource, BindingType, BufferDesc, BufferUsage, Command, ComputePipelineDesc,
        Extent2D, MAX_COMMANDS, PipelineLayoutDesc, SamplerDesc, ShaderEntryPoint,
        ShaderModuleDesc, ShaderStage, ShaderStages, TextureDesc, TextureFormat, TextureUsage,
    };

    fn render_desc(resources: &ResourceTable) -> OwnedRenderPassDesc {
        let target = resources
            .define_texture(
                TextureDesc::new(
                    TextureFormat::Rgba8Unorm,
                    Extent2D::new(4, 4).unwrap(),
                    TextureUsage::RENDER_ATTACHMENT,
                )
                .unwrap(),
            )
            .unwrap()
            .id();
        OwnedRenderPassDesc {
            target,
            area: PixelRect::new(0, 0, 4, 4).unwrap(),
            load: LoadOp::DontCare,
            store: StoreOp::Store,
            depth: None,
        }
    }

    #[test]
    fn replay_borrows_owned_upload_bytes_and_can_be_repeated() {
        let resources = ResourceTable::new();
        let buffer = resources
            .define_buffer(BufferDesc::new(16, BufferUsage::COPY_DST).unwrap())
            .unwrap()
            .id();
        let data = vec![1, 2, 3, 4];
        let source_address = data.as_ptr();
        let recording = OwnedCommandBuffer::new(vec![OwnedCommand::WriteBuffer {
            buffer,
            offset: 4,
            data,
        }]);
        for _ in 0..2 {
            let commands = recording.record(&resources).unwrap();
            match &commands.commands()[0] {
                Command::WriteBuffer {
                    buffer: actual_buffer,
                    offset,
                    data,
                } => {
                    assert_eq!(actual_buffer.id(), buffer);
                    assert_eq!(*offset, 4);
                    assert_eq!(*data, [1, 2, 3, 4]);
                    assert_eq!(data.as_ptr(), source_address);
                }
                _ => panic!("expected the recorded upload"),
            }
        }
    }

    #[test]
    fn replay_rejects_foreign_resource_identities() {
        let owner = ResourceTable::new();
        let foreign = ResourceTable::new();
        let buffer = owner
            .define_buffer(BufferDesc::new(4, BufferUsage::COPY_DST).unwrap())
            .unwrap()
            .id();
        // A matching slot and descriptor do not make identities interchangeable.
        foreign
            .define_buffer(BufferDesc::new(4, BufferUsage::COPY_DST).unwrap())
            .unwrap();
        let recording = OwnedCommandBuffer::new(vec![OwnedCommand::WriteBuffer {
            buffer,
            offset: 0,
            data: vec![1, 2, 3, 4],
        }]);
        assert_eq!(
            recording.validate(&foreign),
            Err(Error::ResourceTableMismatch)
        );
        assert_eq!(recording.validate(&owner), Ok(()));
    }

    #[test]
    fn upload_errors_match_borrowed_encoder_validation() {
        let resources = ResourceTable::new();
        let buffer = resources
            .define_buffer(BufferDesc::new(4, BufferUsage::COPY_DST).unwrap())
            .unwrap();
        for (offset, data) in [(0, vec![]), (3, vec![1, 2]), (u64::MAX, vec![1])] {
            let mut borrowed = CommandEncoder::new(&resources);
            let expected = borrowed.write_buffer(buffer, offset, &data);
            let recording = OwnedCommandBuffer::new(vec![OwnedCommand::WriteBuffer {
                buffer: buffer.id(),
                offset,
                data: data.clone(),
            }]);
            assert!(expected.is_err());
            assert_eq!(recording.validate(&resources), expected);
        }
    }

    #[test]
    fn replay_preserves_sequential_render_and_compute_passes() {
        let resources = ResourceTable::new();
        let desc = render_desc(&resources);
        let recording = OwnedCommandBuffer::new(vec![
            OwnedCommand::BeginRenderPass(desc),
            OwnedCommand::EndRenderPass,
            OwnedCommand::BeginComputePass,
            OwnedCommand::EndComputePass,
            OwnedCommand::BeginRenderPass(desc),
            OwnedCommand::EndRenderPass,
        ]);
        let commands = recording.record(&resources).unwrap();
        assert_eq!(commands.command_count(), 6);
        assert!(matches!(
            commands.commands()[0],
            Command::BeginRenderPass(_)
        ));
        assert!(matches!(commands.commands()[1], Command::EndRenderPass));
        assert!(matches!(commands.commands()[2], Command::BeginComputePass));
        assert!(matches!(commands.commands()[3], Command::EndComputePass));
        assert!(matches!(
            commands.commands()[4],
            Command::BeginRenderPass(_)
        ));
        assert!(matches!(commands.commands()[5], Command::EndRenderPass));
    }

    #[test]
    fn replay_resolves_compute_bindings_barriers_and_copies() {
        let resources = ResourceTable::new();
        let source = resources
            .define_buffer(
                BufferDesc::new(
                    4,
                    BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
                )
                .unwrap(),
            )
            .unwrap()
            .id();
        let destination = resources
            .define_buffer(BufferDesc::new(4, BufferUsage::COPY_DST).unwrap())
            .unwrap()
            .id();
        let layout = BindGroupLayoutDesc::new(vec![BindGroupLayoutEntry::new(
            0,
            ShaderStages::COMPUTE,
            BindingType::StorageBuffer { read_only: false },
        )])
        .unwrap();
        let bind_group = resources
            .define_bind_group(
                BindGroupDesc::new(
                    &resources,
                    layout.clone(),
                    vec![BindGroupEntry::new(
                        0,
                        BindingResource::Buffer {
                            buffer: source,
                            offset: 0,
                            size: 4,
                        },
                    )],
                )
                .unwrap(),
            )
            .unwrap()
            .id();
        let shader = resources
            .define_shader_module(
                ShaderModuleDesc::wgsl(String::from(
                    "@group(0) @binding(0) var<storage, read_write> value: u32;\n\
                     @compute @workgroup_size(1) fn main() { value = value + 1u; }",
                ))
                .unwrap(),
            )
            .unwrap();
        let pipeline = resources
            .define_compute_pipeline(
                ComputePipelineDesc::new(
                    ShaderEntryPoint::new(shader, ShaderStage::Compute, String::from("main"))
                        .unwrap(),
                    PipelineLayoutDesc::new(vec![layout]).unwrap(),
                )
                .unwrap(),
            )
            .unwrap()
            .id();
        let recording = OwnedCommandBuffer::new(vec![
            OwnedCommand::WriteBuffer {
                buffer: source,
                offset: 0,
                data: vec![0; 4],
            },
            OwnedCommand::ResourceBarrier(OwnedResourceBarrier::Buffer {
                buffer: source,
                before: BufferAccess::CopyDestination,
                after: BufferAccess::StorageReadWrite,
            }),
            OwnedCommand::BeginComputePass,
            OwnedCommand::SetComputePipeline(pipeline),
            OwnedCommand::SetBindGroup {
                index: 0,
                bind_group,
            },
            OwnedCommand::Dispatch { x: 1, y: 2, z: 3 },
            OwnedCommand::EndComputePass,
            OwnedCommand::ResourceBarrier(OwnedResourceBarrier::Buffer {
                buffer: source,
                before: BufferAccess::StorageReadWrite,
                after: BufferAccess::CopySource,
            }),
            OwnedCommand::CopyBufferToBuffer {
                source,
                source_offset: 0,
                destination,
                destination_offset: 0,
                size: 4,
            },
        ]);
        let commands = recording.record(&resources).unwrap();
        assert_eq!(commands.command_count(), 9);
        assert!(matches!(
            commands.commands()[5],
            Command::Dispatch { x: 1, y: 2, z: 3 }
        ));
        assert!(matches!(
            commands.commands()[8],
            Command::CopyBufferToBuffer { size: 4, .. }
        ));
    }

    #[test]
    fn replay_rejects_unmatched_and_nested_pass_commands() {
        let resources = ResourceTable::new();
        let desc = render_desc(&resources);
        for commands in [
            vec![OwnedCommand::EndRenderPass],
            vec![OwnedCommand::EndComputePass],
            vec![OwnedCommand::BeginRenderPass(desc)],
            vec![OwnedCommand::BeginComputePass],
            vec![
                OwnedCommand::BeginRenderPass(desc),
                OwnedCommand::EndComputePass,
            ],
            vec![OwnedCommand::BeginComputePass, OwnedCommand::EndRenderPass],
            vec![
                OwnedCommand::BeginRenderPass(desc),
                OwnedCommand::BeginComputePass,
            ],
            vec![
                OwnedCommand::BeginComputePass,
                OwnedCommand::BeginRenderPass(desc),
            ],
            vec![OwnedCommand::Dispatch { x: 1, y: 1, z: 1 }],
        ] {
            assert_eq!(
                OwnedCommandBuffer::new(commands).validate(&resources),
                Err(Error::InvalidDescriptor)
            );
        }
    }

    #[test]
    fn single_channel_mips_can_be_filtered_on_the_gpu() {
        let resources = ResourceTable::new();
        let texture = resources
            .define_texture(
                TextureDesc::new(
                    TextureFormat::R8Unorm,
                    Extent2D::new(8, 8).unwrap(),
                    TextureUsage::COPY_SRC | TextureUsage::COPY_DST | TextureUsage::SAMPLED,
                )
                .unwrap()
                .with_mip_level_count(4)
                .unwrap(),
            )
            .unwrap()
            .id();
        let commands = OwnedCommandBuffer::new(vec![
            OwnedCommand::WriteTextureMip {
                texture,
                mip_level: 0,
                destination: PixelRect::new(0, 0, 8, 8).unwrap(),
                bytes_per_row: 8,
                data: vec![128; 64],
            },
            OwnedCommand::ResourceBarrier(OwnedResourceBarrier::TextureMip {
                texture,
                mip_level: 0,
                before: TextureAccess::CopyDestination,
                after: TextureAccess::CopySource,
            }),
            OwnedCommand::BlitTexture {
                source: texture,
                source_mip: 0,
                destination: texture,
                destination_mip: 1,
                filter: FilterMode::Linear,
            },
        ]);
        assert_eq!(commands.record(&resources).unwrap().command_count(), 3);
    }

    #[test]
    fn mip_dimensions_uploads_and_dependencies_are_checked_per_level() {
        let resources = ResourceTable::new();
        let desc = TextureDesc::new(
            TextureFormat::Rgba8Unorm,
            Extent2D::new(7, 3).unwrap(),
            TextureUsage::COPY_SRC | TextureUsage::COPY_DST | TextureUsage::SAMPLED,
        )
        .unwrap();
        assert!(desc.with_mip_level_count(0).is_err());
        assert!(desc.with_mip_level_count(4).is_err());
        let desc = desc.with_mip_level_count(3).unwrap();
        assert_eq!(desc.mip_extent(1).unwrap(), Extent2D::new(3, 1).unwrap());
        assert_eq!(desc.mip_extent(2).unwrap(), Extent2D::new(1, 1).unwrap());
        assert_eq!(desc.mip_extent(3), Err(Error::OutOfBounds));
        assert_eq!(desc.byte_size(), Ok(100));
        let texture = resources.define_texture(desc).unwrap().id();
        let data = vec![7; 12];
        let recording = OwnedCommandBuffer::new(vec![
            OwnedCommand::WriteTextureMip {
                texture,
                mip_level: 1,
                destination: PixelRect::new(0, 0, 3, 1).unwrap(),
                bytes_per_row: 12,
                data: data.clone(),
            },
            OwnedCommand::ResourceBarrier(OwnedResourceBarrier::TextureMip {
                texture,
                mip_level: 1,
                before: TextureAccess::CopyDestination,
                after: TextureAccess::CopySource,
            }),
            OwnedCommand::BlitTexture {
                source: texture,
                source_mip: 1,
                destination: texture,
                destination_mip: 2,
                filter: FilterMode::Linear,
            },
            OwnedCommand::ResourceBarrier(OwnedResourceBarrier::TextureMip {
                texture,
                mip_level: 2,
                before: TextureAccess::CopyDestination,
                after: TextureAccess::Sampled,
            }),
        ]);
        assert_eq!(recording.record(&resources).unwrap().command_count(), 4);
        let mut wrong = recording.commands().to_vec();
        wrong.push(OwnedCommand::ResourceBarrier(
            OwnedResourceBarrier::TextureMip {
                texture,
                mip_level: 1,
                before: TextureAccess::CopyDestination,
                after: TextureAccess::Sampled,
            },
        ));
        assert_eq!(
            OwnedCommandBuffer::new(wrong).validate(&resources),
            Err(Error::InvalidResourceAccess)
        );
        for command in [
            OwnedCommand::WriteTextureMip {
                texture,
                mip_level: 2,
                destination: PixelRect::new(0, 0, 3, 1).unwrap(),
                bytes_per_row: 12,
                data,
            },
            OwnedCommand::BlitTexture {
                source: texture,
                source_mip: 3,
                destination: texture,
                destination_mip: 2,
                filter: FilterMode::Nearest,
            },
        ] {
            assert_eq!(
                OwnedCommandBuffer::new(vec![command]).validate(&resources),
                Err(Error::OutOfBounds)
            );
        }
        assert_eq!(
            OwnedCommandBuffer::new(vec![OwnedCommand::BlitTexture {
                source: texture,
                source_mip: 1,
                destination: texture,
                destination_mip: 1,
                filter: FilterMode::Nearest
            }])
            .validate(&resources),
            Err(Error::ResourceAccessConflict)
        );
    }

    #[test]
    fn mip_descriptors_reject_overflow_and_nonportable_depth_present_or_lod_values() {
        let extent = Extent2D::new(8, 8).unwrap();
        for (format, usage) in [
            (TextureFormat::Depth32Float, TextureUsage::RENDER_ATTACHMENT),
            (TextureFormat::Bgra8Unorm, TextureUsage::PRESENT),
        ] {
            assert!(
                TextureDesc::new(format, extent, usage)
                    .unwrap()
                    .with_mip_level_count(2)
                    .is_err()
            );
        }
        let huge = TextureDesc::new(
            TextureFormat::Rgba8Unorm,
            Extent2D::new(u32::MAX, u32::MAX).unwrap(),
            TextureUsage::COPY_SRC,
        )
        .unwrap();
        assert_eq!(huge.byte_size(), Err(Error::Overflow));
        let sampler = SamplerDesc::new(
            FilterMode::Linear,
            FilterMode::Nearest,
            AddressMode::Repeat,
            AddressMode::ClampToEdge,
        );
        for (min, max) in [
            (-1.0, 1.0),
            (2.0, 1.0),
            (f32::NAN, 1.0),
            (0.0, f32::INFINITY),
        ] {
            assert!(
                sampler
                    .with_mip_filter(FilterMode::Linear, min, max)
                    .is_err()
            );
        }
        assert_eq!(
            sampler
                .with_mip_filter(FilterMode::Nearest, -0.0, 0.0)
                .unwrap(),
            sampler
        );
        let sampler = sampler
            .with_mip_filter(FilterMode::Linear, 1.25, 9.5)
            .unwrap();
        assert_eq!(
            (sampler.min_lod(), sampler.max_lod(), sampler.mip_filter()),
            (1.25, 9.5, FilterMode::Linear)
        );
    }

    #[test]
    fn owned_draw_counts_follow_the_programmable_pipeline_topology() {
        use crate::ir::{
            BlendState, PrimitiveTopology, ProgrammableRenderPipelineDesc, RasterState,
        };
        let table = ResourceTable::new();
        let shader = table
            .define_shader_module(
                ShaderModuleDesc::wgsl(
                    "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }
             @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }"
                        .into(),
                )
                .unwrap(),
            )
            .unwrap();
        let pass = render_desc(&table);
        for (topology, accepted) in [
            (PrimitiveTopology::TriangleList, false),
            (PrimitiveTopology::TriangleStrip, true),
        ] {
            let pipeline = table
                .define_programmable_render_pipeline(
                    ProgrammableRenderPipelineDesc::new(
                        ShaderEntryPoint::new(shader, ShaderStage::Vertex, "vs".into()).unwrap(),
                        ShaderEntryPoint::new(shader, ShaderStage::Fragment, "fs".into()).unwrap(),
                        PipelineLayoutDesc::new(vec![]).unwrap(),
                        TextureFormat::Rgba8Unorm,
                        None,
                        topology,
                        BlendState::REPLACE,
                        RasterState::new(
                            crate::ir::CullMode::None,
                            crate::ir::FrontFace::CounterClockwise,
                        ),
                    )
                    .unwrap(),
                )
                .unwrap()
                .id();
            let recording = |count| {
                OwnedCommandBuffer::new(vec![
                    OwnedCommand::BeginRenderPass(pass),
                    OwnedCommand::SetProgrammablePipeline(pipeline),
                    OwnedCommand::Draw {
                        vertex_count: count,
                        first_vertex: 0,
                    },
                    OwnedCommand::EndRenderPass,
                ])
            };
            assert_eq!(recording(4).validate(&table).is_ok(), accepted);
            assert_eq!(recording(2).validate(&table), Err(Error::InvalidValue));
            assert_eq!(recording(0).validate(&table), Err(Error::InvalidValue));
            assert_eq!(recording(3).validate(&table), Ok(()));
        }
    }

    #[test]
    fn bind_group_capacity_rejects_growth_without_invalidating_existing_ids() {
        let table = ResourceTable::new();
        let desc = super::super::BindGroupDesc::new(
            &table,
            super::super::BindGroupLayoutDesc::new(vec![]).unwrap(),
            vec![],
        )
        .unwrap();
        let first = table.define_bind_group(desc.clone()).unwrap().id();
        for _ in 1..super::super::MAX_BIND_GROUP_DEFINITIONS {
            table.define_bind_group(desc.clone()).unwrap();
        }
        assert!(matches!(
            table.define_bind_group(desc),
            Err(Error::ResourceLimitExceeded)
        ));
        assert!(table.bind_group_ref(first).is_ok());
    }

    #[test]
    fn replay_enforces_canonical_command_capacity() {
        let resources = ResourceTable::new();
        let buffer = resources
            .define_buffer(BufferDesc::new(4, BufferUsage::COPY_DST).unwrap())
            .unwrap()
            .id();
        let command = OwnedCommand::WriteBuffer {
            buffer,
            offset: 0,
            data: vec![1, 2, 3, 4],
        };
        let at_limit = OwnedCommandBuffer::new(vec![command.clone(); MAX_COMMANDS]);
        assert_eq!(at_limit.validate(&resources), Ok(()));
        let above_limit = OwnedCommandBuffer::new(vec![command; MAX_COMMANDS + 1]);
        assert_eq!(
            above_limit.validate(&resources),
            Err(Error::CommandLimitExceeded)
        );
    }
}
