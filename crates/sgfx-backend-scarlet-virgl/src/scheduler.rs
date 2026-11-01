//! VirGL transport budgets; scheduling policy is shared with other backends.
pub(crate) use sgfx_core::backend::support::scheduler::*;
pub(crate) const MAX_SUBMISSIONS: usize = 16;
pub(crate) const MAX_PENDING_BYTES: usize = 64 * 1024 * 1024;
pub(crate) const MAX_STREAM_BYTES: usize = 512 * 1024 * 1024;
pub(crate) const LIMITS: Limits = Limits {
    submissions: MAX_SUBMISSIONS,
    pending_bytes: MAX_PENDING_BYTES,
    stream_bytes: MAX_STREAM_BYTES,
    in_flight: 16,
};
