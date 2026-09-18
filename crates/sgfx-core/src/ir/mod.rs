//! Validated, backend-neutral logical graphics intermediate representation.
//!
//! Resources are descriptors held by [`crate::ir::ResourceTable`]. Commands recorded by
//! [`crate::ir::CommandEncoder`] borrow both that table and upload data, so a
//! backend can lower the finished [`crate::ir::CommandBuffer`] without accepting forged
//! resource identities.

pub(crate) mod command;
mod pipeline;
pub(crate) mod resource;
mod texture_view;
mod types;
pub use texture_view::{TextureViewDesc, TextureViewDimension};

pub use command::{
    Command, CommandBuffer, CommandEncoder, ComputePassEncoder, DepthAttachment, DepthLoadOp,
    LoadOp, MAX_COMMANDS, RenderPassDesc, RenderPassEncoder, StoreOp,
};
pub use pipeline::{
    BlendComponent, BlendFactor, BlendOp, BlendState, CompareFunction, CullMode, DepthState,
    DrawUniforms, FragmentProgram, FrontFace, IndexFormat, MAX_VERTEX_ATTRIBUTES,
    MAX_VERTEX_BUFFERS, PrimitiveTopology, RasterState, RenderPipelineDesc, TextureSampleMode,
    VertexAttribute, VertexBufferLayout, VertexFormat,
};
pub use resource::{
    AddressMode, BufferDesc, BufferId, BufferRef, BufferUsage, FilterMode,
    MAX_BIND_GROUP_DEFINITIONS, MAX_BUFFERS, MAX_RENDER_PIPELINES, MAX_SAMPLERS, MAX_TEXTURES,
    RenderPipelineId, RenderPipelineRef, ResourceTable, SamplerDesc, SamplerId, SamplerRef,
    TextureDesc, TextureFormat, TextureId, TextureRef, TextureUsage, TextureWrite,
};
pub use types::{Color, Error, Extent2D, PixelRect, Result, Transform, Viewport};

mod programmable;
mod render_target;
pub use render_target::{ColorAttachment, ColorTargetState, ColorWriteMask, MAX_COLOR_ATTACHMENTS};
mod shader;
pub use programmable::*;
pub use resource::{
    BindGroupId, BindGroupRef, ComputePipelineId, ComputePipelineRef, ProgrammableRenderPipelineId,
    ProgrammableRenderPipelineRef, ShaderModuleId, ShaderModuleRef,
};
pub use shader::*;
mod owned;
pub use owned::{
    OwnedColorAttachment, OwnedCommand, OwnedCommandBuffer, OwnedDepthAttachment,
    OwnedRenderPassDesc, OwnedResourceBarrier,
};
