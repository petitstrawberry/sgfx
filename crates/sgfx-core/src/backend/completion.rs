//! Owned submission receipts and backend-neutral completion observation.

use core::time::Duration;

use super::CommandExecutor;
use crate::ir::CommandBuffer;

/// Whether all GPU accesses covered by a submission receipt have retired.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CompletionStatus {
    /// Covered work is still outstanding, including when a wait timed out.
    Pending,
    /// Covered GPU accesses have retired; presentation and external leases
    /// remain separate, as do mapping/readback and cache-coherency operations.
    Complete,
}

/// An owned receipt for one queue-ordered logical submission or failed prefix.
///
/// A receipt must not borrow the command buffer, uploads, table, or executor.
/// Dropping it does not cancel work or wait for completion. Backends retain
/// in-flight resource ownership independently of receipt ownership.
///
/// Completion covers all chunks/uploads of the logical submission and earlier
/// ordered work on its queue, not later submissions or unrelated consumers.
/// Device failure is an error, never proof that externally shared storage may
/// be reused. A later device loss may invalidate observation of older receipts.
pub trait Completion {
    /// Backend error observed while checking or waiting for completion.
    type Error;

    /// Progress completion observation without waiting for the GPU.
    ///
    /// # Returns
    ///
    /// The current completion status, or an observation/device error. This
    /// drives any backend event pump needed to observe completion; callers
    /// need not invoke an additional backend-specific polling function.
    fn poll(&self) -> Result<CompletionStatus, Self::Error>;

    /// Wait for the covered work, subject to an optional caller deadline.
    ///
    /// # Arguments
    ///
    /// * `timeout` - Maximum requested waiting duration; `None` allows an
    ///   unbounded wait and zero performs a nonblocking poll. This is not a
    ///   hard real-time scheduling guarantee.
    ///
    /// # Returns
    ///
    /// `Complete` when covered accesses have retired, `Pending` when the wait
    /// expires with work outstanding, or a backend error. Timeout does not
    /// cancel the submission, poison its receipt, or permit resource reuse.
    fn wait(&self, timeout: Option<Duration>) -> Result<CompletionStatus, Self::Error>;
}

/// Failure to accept a complete logical submission.
///
/// Immediate errors and completion errors are independent: a backend may
/// accept a prefix before failing. The attached receipt then covers all work
/// that may have been accepted. Neither failure nor dropping that receipt
/// authorizes replay or external resource reuse.
#[derive(Debug)]
#[non_exhaustive]
pub enum SubmitError<E, S> {
    /// Tracking/queue capacity is full. No work from this call was accepted;
    /// retry after making capacity available rather than blocking in submit.
    Busy,
    /// No work from this call was accepted. Earlier queue/device work is not
    /// rolled back or certified healthy by this result.
    Rejected(E),
    /// Work may have been accepted; retain its receipt when observing failure.
    Failed {
        /// Immediate validation, lowering, or submission error.
        error: E,
        /// Receipt covering every possibly accepted chunk and upload.
        completion: S,
    },
}

impl<E, S> SubmitError<E, S> {
    /// Adapt a backend error and receipt without losing partial-acceptance state.
    ///
    /// # Arguments
    ///
    /// * `error` - Conversion applied to the immediate error, when present.
    /// * `completion` - Conversion applied to a failed-prefix receipt, when present.
    ///
    /// # Returns
    ///
    /// The same acceptance classification with converted payloads. Facades
    /// should use this instead of an exhaustive match on an extensible enum.
    pub fn map<F, T>(
        self,
        error: impl FnOnce(E) -> F,
        completion: impl FnOnce(S) -> T,
    ) -> SubmitError<F, T> {
        match self {
            Self::Busy => SubmitError::Busy,
            Self::Rejected(value) => SubmitError::Rejected(error(value)),
            Self::Failed {
                error: value,
                completion: receipt,
            } => SubmitError::Failed {
                error: error(value),
                completion: completion(receipt),
            },
        }
    }
}

/// Accepts logical command streams with owned completion tracking.
///
/// This extends rather than changes [`CommandExecutor`]. Official 1.0 backends
/// must expose this interface, but existing third-party executors remain valid
/// implementations of the older trait.
pub trait CommandSubmitter: CommandExecutor {
    /// Owned backend receipt, independent of command/executor borrow lifetimes.
    type Submission: Completion<Error = Self::Error> + 'static;

    /// Accept a complete logical command stream without waiting for its GPU work.
    ///
    /// # Arguments
    ///
    /// * `commands` - Valid borrowed commands and uploads to validate and lower.
    ///
    /// # Returns
    ///
    /// A receipt for all accepted work, or an error distinguishing bounded
    /// backpressure, rejection, and possible partial acceptance. On every
    /// return path, the backend has consumed/copied all borrowed data needed
    /// by pending work. An empty stream establishes a queue checkpoint.
    /// CPU validation/lowering is allowed. A backend may synchronously create
    /// physical resources on their first use, before accepting uploads/copies/
    /// draws from this call, and must document that cold-setup boundary. After
    /// that setup, submission must not wait for GPU completion or capacity.
    /// Reusing already-materialized resources does not repeat the setup wait.
    fn submit<'r, 'data>(
        &mut self,
        commands: &CommandBuffer<'r, 'data>,
    ) -> Result<Self::Submission, SubmitError<Self::Error, Self::Submission>>;
}
