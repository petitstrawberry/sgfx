//! Classification of validated transitions covered by implicit queue ordering.
//!
//! Callers must validate resource identity, usage, mip bounds, pass scope and
//! consecutive accesses before skipping execution. The capability must come
//! from the active backend, never its name or the presence of another feature.
use crate::ir;

pub fn can_skip(implicit_transitions: bool, commands: &[ir::OwnedCommand]) -> bool {
    implicit_transitions && !commands.is_empty() && commands.iter().all(ordinary_barrier)
}

pub fn ordinary_barrier(command: &ir::OwnedCommand) -> bool {
    match command {
        ir::OwnedCommand::ResourceBarrier(ir::OwnedResourceBarrier::Buffer {
            before,
            after,
            ..
        }) => ordinary_buffer_access(*before) && ordinary_buffer_access(*after),
        ir::OwnedCommand::ResourceBarrier(
            ir::OwnedResourceBarrier::Texture { before, after, .. }
            | ir::OwnedResourceBarrier::TextureMip { before, after, .. },
        ) => ordinary_texture_access(*before) && ordinary_texture_access(*after),
        _ => false,
    }
}

fn ordinary_buffer_access(access: ir::BufferAccess) -> bool {
    matches!(
        access,
        ir::BufferAccess::CopyDestination
            | ir::BufferAccess::CopySource
            | ir::BufferAccess::Vertex
            | ir::BufferAccess::Index
            | ir::BufferAccess::Uniform
    )
}

fn ordinary_texture_access(access: ir::TextureAccess) -> bool {
    // Storage and future access scopes always retain the explicit backend path.
    matches!(
        access,
        ir::TextureAccess::CopyDestination
            | ir::TextureAccess::CopySource
            | ir::TextureAccess::Sampled
            | ir::TextureAccess::RenderAttachment
    )
}
