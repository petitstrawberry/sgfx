//! VirGL packet budgets; packet-boundary ownership and staging are shared.
pub(crate) use sgfx_core::backend::support::packets::Error;
pub(crate) const UPLOAD_ARENA_COUNT: usize = 4;
pub(crate) type Packets = sgfx_core::backend::support::packets::Packets<
    { crate::scheduler::MAX_STREAM_BYTES },
    UPLOAD_ARENA_COUNT,
>;
