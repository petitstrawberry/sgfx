//! Backend-owned execution contracts for portable SGFX command buffers.
//!
//! Execution and completion are distinct: successful execution accepts the
//! command stream, but does not portably imply GPU completion or presentation.

use crate::ir::CommandBuffer;

/// Executes portable SGFX command buffers using backend-owned state.
///
/// An implementation binds the queue, resource cache, platform context, and
/// any transport-specific state needed to validate, lower, and submit the IR.
/// Portable renderer code only records a complete [`CommandBuffer`] and does
/// not account for backend command sizes or submission boundaries.
///
/// # Execution contract
///
/// * Commands have their recorded order. Sequential successful calls on the
///   same bound queue preserve that order for accesses to shared resources.
///   This does not order independent queues, contexts, or external producers.
/// * Borrowed upload bytes are consumed or copied before the call returns.
///   No caller-owned command or upload borrow is retained by pending GPU work.
///   Backend-owned allocations and imported resource owners must remain valid
///   for that work, even if their physical release has to be deferred.
/// * Logical validation does not imply backend support. Unsupported semantics
///   must produce an error rather than being silently omitted or approximated.
/// * Success means the complete stream has been accepted. A backend may wait
///   for GPU completion, but callers cannot infer that from this trait. CPU
///   readback, cross-queue sharing, and presentation use separate backend or
///   platform synchronization contracts.
/// * Failure is not transactional: earlier uploads or submissions may already
///   have taken effect. Retrying a failed stream is not guaranteed to be safe;
///   recovery and delayed device errors follow the backend's documented policy.
pub trait CommandExecutor {
    /// Error returned when validation, lowering, or submission fails.
    type Error;

    /// Validate, lower, and execute one complete command buffer.
    ///
    /// # Arguments
    ///
    /// * `commands` - Portable commands and borrowed upload data to execute.
    ///
    /// # Returns
    ///
    /// Success after the backend's submission contract is satisfied, or the
    /// backend-specific validation, allocation, transport, or device error.
    /// Success alone is neither a GPU-completion fence nor a presentation
    /// acknowledgement. On either return path, borrowed upload data may be
    /// released once the caller drops the command buffer that borrows it.
    fn execute<'r, 'data>(
        &mut self,
        commands: &CommandBuffer<'r, 'data>,
    ) -> Result<(), Self::Error>;
}
