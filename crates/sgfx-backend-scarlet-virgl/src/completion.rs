//! Owned observation of every native chunk in one logical submission.

use alloc::vec::Vec;
use core::time::Duration;

use gpu_raw::{
    GPU_ABI_VERSION, GPU_COMPLETION_COMPLETE, GPU_COMPLETION_FAILED, GPU_COMPLETION_FAILURE_NONE,
    GPU_COMPLETION_PENDING, GpuCompletion, GpuCompletionInfo, GpuQueue, GpuSubmitError,
};
#[cfg(feature = "std")]
use scarlet_os::poll::{POLLIN, PollHandle, poll};
use sgfx_core::backend::{Completion, CompletionStatus, SubmitError};
#[cfg(not(feature = "std"))]
use std::poll::{POLLIN, PollHandle, poll};

use crate::{HandleError, HandleResult, IrSubmitError};

/// Owned completion receipt for a logical stream, including all accepted chunks.
///
/// The receipt does not borrow its executor, resource cache, or command data.
/// Dropping it neither waits nor cancels work: the kernel independently retains
/// accepted commands and their resources. Completion is not a presentation or
/// SWS buffer-release acknowledgement.
#[derive(Debug)]
pub struct Submission {
    chunks: Vec<GpuCompletion>,
    unobservable: bool,
}

impl Submission {
    fn status(
        &self,
        mut pending: impl FnMut(&GpuCompletion),
    ) -> Result<CompletionStatus, IrSubmitError> {
        if self.unobservable || self.chunks.is_empty() {
            return Err(IrSubmitError::CompletionUnavailable);
        }
        let mut complete = true;
        for chunk in &self.chunks {
            if completion_status(chunk.query()?)? == CompletionStatus::Pending {
                pending(chunk);
                complete = false;
            }
        }
        Ok(if complete {
            CompletionStatus::Complete
        } else {
            CompletionStatus::Pending
        })
    }
}

impl Completion for Submission {
    type Error = IrSubmitError;

    /// Query every accepted chunk without waiting for GPU work.
    ///
    /// # Returns
    ///
    /// `Complete` only when every covered chunk retired successfully; otherwise
    /// `Pending` or an observation/execution error. Unknown acceptance can never
    /// be certified using just an earlier chunk's receipt.
    fn poll(&self) -> Result<CompletionStatus, IrSubmitError> {
        self.status(|_| {})
    }

    /// Wait on the kernel's read-only completion handles.
    ///
    /// # Arguments
    ///
    /// * `timeout` - Maximum requested duration. Zero polls; `None` has no
    ///   deadline. Waiting does not cancel work or grant external buffer reuse.
    ///
    /// # Returns
    ///
    /// `Complete`, `Pending` after timeout, or an error. Already completed chunks
    /// are excluded from the native wait so their level readiness cannot spin
    /// the caller while later chunks are pending.
    fn wait(&self, timeout: Option<Duration>) -> Result<CompletionStatus, IrSubmitError> {
        if timeout == Some(Duration::ZERO) {
            return self.poll();
        }
        let started = monotonic_time_ns();
        let mut handles = Vec::new();
        handles
            .try_reserve_exact(self.chunks.len())
            .map_err(|_| IrSubmitError::OutOfMemory)?;
        loop {
            handles.clear();
            if self.status(|chunk| {
                handles.push(PollHandle::new(chunk.as_handle().as_raw() as u32, POLLIN));
            })? == CompletionStatus::Complete
            {
                return Ok(CompletionStatus::Complete);
            }
            let remaining = timeout.map(|limit| {
                limit.saturating_sub(Duration::from_nanos(
                    monotonic_time_ns().saturating_sub(started),
                ))
            });
            if remaining == Some(Duration::ZERO) {
                return Ok(CompletionStatus::Pending);
            }
            let timeout_ns = remaining.map_or(-1, |remaining| {
                remaining.as_nanos().min(i64::MAX as u128) as i64
            });
            poll(&mut handles, timeout_ns)
                .map_err(|error| IrSubmitError::Backend(HandleError::SystemError(error)))?;
        }
    }
}

fn completion_status(info: GpuCompletionInfo) -> Result<CompletionStatus, IrSubmitError> {
    if info.abi_version != GPU_ABI_VERSION || info.reserved != 0 || info.reserved2 != 0 {
        return Err(IrSubmitError::CompletionUnavailable);
    }
    match (info.state, info.failure) {
        (GPU_COMPLETION_PENDING, GPU_COMPLETION_FAILURE_NONE) => Ok(CompletionStatus::Pending),
        (GPU_COMPLETION_COMPLETE, GPU_COMPLETION_FAILURE_NONE) => Ok(CompletionStatus::Complete),
        (GPU_COMPLETION_FAILED, failure) => Err(IrSubmitError::CompletionFailed(failure)),
        _ => Err(IrSubmitError::CompletionUnavailable),
    }
}

#[cfg(feature = "std")]
fn monotonic_time_ns() -> u64 {
    scarlet_os::time::monotonic_time_ns()
}

#[cfg(not(feature = "std"))]
fn monotonic_time_ns() -> u64 {
    use std::syscall::{Syscall, syscall0};
    syscall0(Syscall::MonotonicTime) as u64
}

/// Per-call state, never a mode stored on a shared queue or connection.
pub(crate) enum SubmitMode {
    Synchronous,
    Tracked { receipt: Submission, busy: bool },
}

impl SubmitMode {
    pub(crate) fn tracked() -> Self {
        Self::Tracked {
            receipt: Submission {
                chunks: Vec::new(),
                unobservable: false,
            },
            busy: false,
        }
    }

    pub(crate) fn is_tracked(&self) -> bool {
        matches!(self, Self::Tracked { .. })
    }

    pub(crate) fn submit(&mut self, queue: &GpuQueue, commands: &[u8]) -> HandleResult<()> {
        let Self::Tracked { receipt, busy } = self else {
            return queue.submit(commands).map(|_| ());
        };
        // Reserve observation storage before handing the kernel any new work.
        receipt
            .chunks
            .try_reserve(1)
            .map_err(|_| HandleError::OutOfResources)?;
        match queue.submit_async(commands) {
            Ok(chunk) => {
                receipt.chunks.push(chunk);
                Ok(())
            }
            Err(GpuSubmitError::Busy) => {
                *busy = true;
                Err(HandleError::OutOfResources)
            }
            Err(GpuSubmitError::Rejected(error)) => Err(error),
            Err(GpuSubmitError::Failed { error, completion }) => {
                if let Some(chunk) = completion {
                    receipt.chunks.push(chunk);
                } else {
                    receipt.unobservable = true;
                }
                Err(error)
            }
            Err(_) => {
                receipt.unobservable = true;
                Err(HandleError::SystemError(-1))
            }
        }
    }

    pub(crate) fn accepted(&self) -> bool {
        matches!(self, Self::Tracked { receipt, .. } if receipt.unobservable || !receipt.chunks.is_empty())
    }

    pub(crate) fn finish(
        self,
        result: Result<(), IrSubmitError>,
    ) -> Result<Submission, SubmitError<IrSubmitError, Submission>> {
        let Self::Tracked { receipt, busy } = self else {
            return Err(SubmitError::Rejected(IrSubmitError::CompletionUnavailable));
        };
        match result {
            Ok(()) if !receipt.chunks.is_empty() && !receipt.unobservable => Ok(receipt),
            Ok(()) => Err(SubmitError::Failed {
                error: IrSubmitError::CompletionUnavailable,
                completion: receipt,
            }),
            Err(error) if receipt.unobservable || !receipt.chunks.is_empty() => {
                Err(SubmitError::Failed {
                    error,
                    completion: receipt,
                })
            }
            Err(_) if busy => Err(SubmitError::Busy),
            Err(error) => Err(SubmitError::Rejected(error)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{SubmitMode, completion_status};
    use crate::{HandleError, IrSubmitError};
    use core::time::Duration;
    use gpu_raw::{
        GPU_ABI_VERSION, GPU_COMPLETION_COMPLETE, GPU_COMPLETION_FAILED,
        GPU_COMPLETION_FAILURE_DEVICE_LOST, GPU_RESULT_SUCCESS, GpuCompletionInfo,
    };
    use sgfx_core::backend::{Completion, CompletionStatus, SubmitError};

    #[test]
    fn completion_requires_a_valid_successful_terminal_observation() {
        let mut info = GpuCompletionInfo::new();
        info.result = GPU_RESULT_SUCCESS;
        assert_eq!(completion_status(info).unwrap(), CompletionStatus::Pending);
        info.state = GPU_COMPLETION_COMPLETE;
        assert_eq!(completion_status(info).unwrap(), CompletionStatus::Complete);
        info.failure = GPU_COMPLETION_FAILURE_DEVICE_LOST;
        assert!(matches!(
            completion_status(info),
            Err(IrSubmitError::CompletionUnavailable)
        ));
        info.state = GPU_COMPLETION_FAILED;
        assert!(matches!(
            completion_status(info),
            Err(IrSubmitError::CompletionFailed(
                GPU_COMPLETION_FAILURE_DEVICE_LOST
            ))
        ));
    }

    #[test]
    fn malformed_completion_is_never_complete() {
        let mut info = GpuCompletionInfo::new();
        info.state = GPU_COMPLETION_COMPLETE;
        info.abi_version += 1;
        assert!(matches!(
            completion_status(info),
            Err(IrSubmitError::CompletionUnavailable)
        ));
        info.abi_version = GPU_ABI_VERSION;
        info.reserved = 1;
        assert!(matches!(
            completion_status(info),
            Err(IrSubmitError::CompletionUnavailable)
        ));
        info.reserved = 0;
        info.reserved2 = 1;
        assert!(matches!(
            completion_status(info),
            Err(IrSubmitError::CompletionUnavailable)
        ));
        info.reserved2 = 0;
        info.state = u32::MAX;
        assert!(matches!(
            completion_status(info),
            Err(IrSubmitError::CompletionUnavailable)
        ));
    }

    #[test]
    fn rejection_and_backpressure_require_no_accepted_work() {
        assert!(matches!(
            SubmitMode::tracked().finish(Err(IrSubmitError::InvalidVertexData)),
            Err(SubmitError::Rejected(IrSubmitError::InvalidVertexData))
        ));
        let mut mode = SubmitMode::tracked();
        if let SubmitMode::Tracked { busy, .. } = &mut mode {
            *busy = true;
        }
        assert!(matches!(
            mode.finish(Err(IrSubmitError::Backend(HandleError::OutOfResources))),
            Err(SubmitError::Busy)
        ));
    }

    #[test]
    fn unknown_acceptance_cannot_be_reclassified_as_busy_or_complete() {
        let mut mode = SubmitMode::tracked();
        if let SubmitMode::Tracked { receipt, busy } = &mut mode {
            receipt.unobservable = true;
            *busy = true;
        }
        assert!(mode.accepted());
        let Err(SubmitError::Failed { completion, .. }) =
            mode.finish(Err(IrSubmitError::CompletionUnavailable))
        else {
            panic!("unknown acceptance must retain a failure receipt");
        };
        assert!(matches!(
            completion.poll(),
            Err(IrSubmitError::CompletionUnavailable)
        ));
        assert!(matches!(
            completion.wait(Some(Duration::ZERO)),
            Err(IrSubmitError::CompletionUnavailable)
        ));
    }

    #[test]
    fn empty_receipt_requires_a_real_queue_checkpoint() {
        assert!(matches!(
            SubmitMode::tracked().finish(Ok(())),
            Err(SubmitError::Failed {
                error: IrSubmitError::CompletionUnavailable,
                ..
            })
        ));
    }
}
