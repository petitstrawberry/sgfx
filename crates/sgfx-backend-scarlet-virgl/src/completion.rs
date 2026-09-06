//! Owned observation of every native chunk in one logical submission.

use alloc::{sync::Arc, vec::Vec};
use core::{fmt, time::Duration};
use gpu_raw::{
    GPU_ABI_VERSION, GPU_COMPLETION_COMPLETE, GPU_COMPLETION_FAILED, GPU_COMPLETION_FAILURE_NONE,
    GPU_COMPLETION_PENDING, GpuCompletionInfo, GpuQueue,
};
use sgfx_core::backend::{Completion, CompletionStatus, SubmitError};

use crate::dispatch::{Chunk, NativeScheduler, Signal};
use crate::packets::{self, Packets};
use crate::scheduler::AdmissionError;
use crate::virgl::UploadArena;
use crate::{HandleError, HandleResult, IrSubmitError};

/// Owned completion receipt for a logical stream, including all accepted chunks.
///
/// The receipt does not borrow its executor, resource cache, or command data.
/// Dropping it neither waits nor cancels work: the queue worker and kernel retain
/// accepted commands and their resources, including not-yet-dispatched chunks.
/// Completion is not a presentation or SWS buffer-release acknowledgement.
#[derive(Clone)]
pub struct Submission {
    signal: Arc<Signal>,
}

impl fmt::Debug for Submission {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Submission")
            .field("status", &self.signal.poll())
            .finish()
    }
}

impl Completion for Submission {
    type Error = IrSubmitError;

    /// Query every accepted chunk without waiting for GPU work.
    ///
    /// # Returns
    ///
    /// Complete only when every covered chunk retired successfully; otherwise
    /// Pending or an observation/execution error. Unknown acceptance can never
    /// be certified using just an earlier chunk's receipt. The worker progresses
    /// independently of this observation call.
    fn poll(&self) -> Result<CompletionStatus, IrSubmitError> {
        self.signal.poll()
    }

    /// Wait for the queue worker's whole-stream completion notification.
    ///
    /// # Arguments
    ///
    /// * `timeout` - Maximum requested duration. Zero polls; None has no
    ///   deadline. Waiting does not cancel work or grant external buffer reuse.
    ///
    /// # Returns
    ///
    /// Complete, Pending after timeout, or an error. An earlier completed
    /// native chunk cannot spin the caller while later chunks remain queued.
    fn wait(&self, timeout: Option<Duration>) -> Result<CompletionStatus, IrSubmitError> {
        self.signal.wait(timeout)
    }
}

pub(crate) fn completion_status(
    info: GpuCompletionInfo,
) -> Result<CompletionStatus, IrSubmitError> {
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
pub(crate) fn monotonic_time_ns() -> u64 {
    scarlet_os::time::monotonic_time_ns()
}

#[cfg(not(feature = "std"))]
pub(crate) fn monotonic_time_ns() -> u64 {
    use std::syscall::{Syscall, syscall0};
    syscall0(Syscall::MonotonicTime) as u64
}

/// Per-call state, never a mode stored on a shared queue or connection.
pub(crate) enum SubmitMode {
    Synchronous,
    Tracked {
        receipt: Option<Submission>,
        busy: bool,
        too_large: bool,
        packets: Packets,
        arenas: Vec<Arc<UploadArena>>,
    },
}

impl SubmitMode {
    pub(crate) fn tracked() -> Self {
        Self::Tracked {
            receipt: None,
            busy: false,
            too_large: false,
            packets: Packets::new(gpu_raw::GPU_MAX_OPAQUE_COMMAND_SIZE as usize),
            arenas: Vec::new(),
        }
    }

    pub(crate) fn is_tracked(&self) -> bool {
        matches!(self, Self::Tracked { .. })
    }

    pub(crate) fn needs_upload_arena(&self) -> bool {
        matches!(self, Self::Tracked { arenas, .. } if arenas.is_empty())
    }

    pub(crate) fn set_upload_arenas(&mut self, source: &[Arc<UploadArena>]) -> HandleResult<()> {
        if let Self::Tracked { arenas, .. } = self {
            arenas
                .try_reserve_exact(source.len())
                .map_err(|_| HandleError::OutOfResources)?;
            arenas.extend(source.iter().cloned());
        }
        Ok(())
    }

    pub(crate) fn prepare_packet(&mut self, maximum: usize) -> HandleResult<()> {
        if let Self::Tracked {
            packets, too_large, ..
        } = self
        {
            packets
                .prepare(maximum)
                .map_err(|error| packet_error(error, too_large))?;
        }
        Ok(())
    }

    pub(crate) fn upload_range(&mut self, length: u32) -> HandleResult<Option<(u32, u32)>> {
        let Self::Tracked {
            packets,
            arenas,
            too_large,
            ..
        } = self
        else {
            return Ok(None);
        };
        let (index, offset) = packets
            .upload_range(length)
            .map_err(|error| packet_error(error, too_large))?;
        let arena = arenas.get(index).ok_or(HandleError::InvalidParameter)?;
        Ok(Some((arena.resource_id, offset)))
    }

    pub(crate) fn submit(&mut self, queue: &GpuQueue, commands: &[u8]) -> HandleResult<()> {
        let Self::Tracked {
            packets, too_large, ..
        } = self
        else {
            return queue.submit(commands).map(|_| ());
        };
        packets
            .append(commands)
            .map_err(|error| packet_error(error, too_large))
    }

    pub(crate) fn set_limit(&mut self, max_bytes: usize) {
        if let Self::Tracked { packets, .. } = self {
            *packets = Packets::new(max_bytes.min(gpu_raw::GPU_MAX_OPAQUE_COMMAND_SIZE as usize));
        }
    }

    pub(crate) fn flush(
        &mut self,
        scheduler: &NativeScheduler,
        queue: Arc<GpuQueue>,
    ) -> HandleResult<()> {
        let Self::Tracked {
            receipt,
            busy,
            too_large,
            packets,
            arenas,
        } = self
        else {
            return Ok(());
        };
        let staged = core::mem::replace(packets, Packets::new(0))
            .finish()
            .map_err(|error| packet_error(error, too_large))?;
        let mut chunks = Vec::new();
        chunks
            .try_reserve_exact(staged.len())
            .map_err(|_| HandleError::OutOfResources)?;
        for chunk in staged {
            let arena = chunk
                .arena
                .map(|index| {
                    arenas
                        .get(index)
                        .cloned()
                        .ok_or(HandleError::InvalidParameter)
                })
                .transpose()?;
            chunks.push(Chunk {
                commands: chunk.bytes,
                arena,
            });
        }
        match scheduler.enqueue(queue, chunks) {
            Ok(signal) => {
                *receipt = Some(Submission { signal });
                Ok(())
            }
            Err(AdmissionError::Busy) => {
                *busy = true;
                Err(HandleError::OutOfResources)
            }
            Err(AdmissionError::TooLarge) => {
                *too_large = true;
                Err(HandleError::OutOfResources)
            }
            Err(AdmissionError::OutOfMemory) => Err(HandleError::OutOfResources),
            Err(AdmissionError::Failed(_)) => Err(HandleError::SystemError(-1)),
        }
    }

    pub(crate) fn accepted(&self) -> bool {
        matches!(
            self,
            Self::Tracked {
                receipt: Some(_),
                ..
            }
        )
    }

    pub(crate) fn finish(
        self,
        result: Result<(), IrSubmitError>,
    ) -> Result<Submission, SubmitError<IrSubmitError, Submission>> {
        let Self::Tracked {
            receipt,
            busy,
            too_large,
            ..
        } = self
        else {
            return Err(SubmitError::Rejected(IrSubmitError::CompletionUnavailable));
        };
        match (result, receipt) {
            (Ok(()), Some(receipt)) => Ok(receipt),
            (Err(error), Some(completion)) => Err(SubmitError::Failed { error, completion }),
            (_, None) if busy => Err(SubmitError::Busy),
            (_, None) if too_large => Err(SubmitError::Rejected(IrSubmitError::SubmissionTooLarge)),
            (Err(error), None) => Err(SubmitError::Rejected(error)),
            (Ok(()), None) => Err(SubmitError::Rejected(IrSubmitError::CompletionUnavailable)),
        }
    }
}

fn packet_error(error: packets::Error, too_large: &mut bool) -> HandleError {
    *too_large |= error == packets::Error::TooLarge;
    HandleError::OutOfResources
}

#[cfg(test)]
mod tests {
    use super::{SubmitMode, completion_status};
    use crate::{HandleError, IrSubmitError};
    use gpu_raw::{
        GPU_ABI_VERSION, GPU_COMPLETION_COMPLETE, GPU_COMPLETION_FAILED,
        GPU_COMPLETION_FAILURE_DEVICE_LOST, GPU_RESULT_SUCCESS, GpuCompletionInfo,
    };
    use sgfx_core::backend::{CompletionStatus, SubmitError};
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
    fn empty_receipt_requires_a_real_queue_checkpoint() {
        assert!(matches!(
            SubmitMode::tracked().finish(Ok(())),
            Err(SubmitError::Rejected(IrSubmitError::CompletionUnavailable))
        ));
    }
}
