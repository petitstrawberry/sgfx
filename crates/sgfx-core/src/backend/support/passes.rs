//! Attachment preservation checks for empty-pass elision.
use crate::ir::{self, DepthLoadOp, LoadOp};

/// All color and depth attachments retain their contents. This alone does not
/// permit skipping a pass: the backend must also prove there are no draws,
/// clears, queries or other side effects, and preserve barriers and completion.
pub fn pass_preserves_contents(desc: ir::RenderPassDesc<'_>) -> bool {
    desc.color_attachments().all(|attachment| {
        attachment.load() == LoadOp::Load && attachment.store() == ir::StoreOp::Store
    }) && desc.depth_attachment().is_none_or(|attachment| {
        attachment.load() == DepthLoadOp::Load && attachment.store() == ir::StoreOp::Store
    })
}
