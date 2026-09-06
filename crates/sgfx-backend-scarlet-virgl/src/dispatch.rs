//! Autonomous context-ordered dispatch and owned logical completion signals.

use alloc::{sync::Arc, vec::Vec};
use core::sync::atomic::{AtomicBool, Ordering};
use core::time::Duration;
use std::sync::{Mutex, MutexGuard};

use gpu_raw::{GpuCompletion, GpuQueue, GpuSubmitError};
#[cfg(feature = "std")]
use scarlet_os::{
    ipc::pipe,
    poll::{POLLIN, PollHandle, poll},
};
#[cfg(not(feature = "std"))]
use std::{
    ipc::pipe,
    poll::{POLLIN, PollHandle, poll},
};

use crate::completion::{completion_status, monotonic_time_ns};
use crate::scheduler::{AdmissionError, DispatchError, Scheduler, Transport};
use crate::virgl::UploadArena;
use crate::{Handle, HandleError, HandleResult, IrSubmitError};
use sgfx_core::backend::CompletionStatus;

pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    #[cfg(feature = "std")]
    {
        mutex
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
    #[cfg(not(feature = "std"))]
    {
        mutex.lock()
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Failure {
    Backend(HandleError),
    Completion(u32),
    Unavailable,
}

impl From<Failure> for IrSubmitError {
    fn from(value: Failure) -> Self {
        match value {
            Failure::Backend(error) => Self::Backend(error),
            Failure::Completion(reason) => Self::CompletionFailed(reason),
            Failure::Unavailable => Self::CompletionUnavailable,
        }
    }
}

fn native_status(completion: &GpuCompletion) -> Result<bool, Failure> {
    let info = completion.query().map_err(Failure::Backend)?;
    completion_status(info)
        .map(|status| status == CompletionStatus::Complete)
        .map_err(|error| match error {
            IrSubmitError::CompletionFailed(reason) => Failure::Completion(reason),
            _ => Failure::Unavailable,
        })
}

// One coalesced byte is sufficient and cannot fill the pipe. Queue wakeups have
// one reader; terminal signals are never consumed, so all waiters see readiness.
struct Event {
    reader: Handle,
    writer: Handle,
    signalled: AtomicBool,
}

impl Event {
    fn new() -> HandleResult<Self> {
        let (reader, writer) = pipe().map_err(|_| HandleError::OutOfResources)?;
        Ok(Self {
            reader,
            writer,
            signalled: AtomicBool::new(false),
        })
    }

    fn notify(&self) {
        if !self.signalled.swap(true, Ordering::AcqRel) {
            let result = self
                .writer
                .as_stream()
                .and_then(|stream| stream.write(&[1]).map_err(|_| HandleError::SystemError(-1)));
            if result != Ok(1) {
                self.signalled.store(false, Ordering::Release);
            }
        }
    }

    fn consume(&self) -> HandleResult<()> {
        // A notifier publishes the flag before writing the byte. Never block
        // if that thread has not finished the write yet. Check the pipe itself
        // so an uncertain write response cannot leave an unread wake spinning.
        if poll(&mut [self.poll_handle()], 0).map_err(HandleError::SystemError)? != 0 {
            let mut byte = [0];
            let length = self
                .reader
                .as_stream()?
                .read(&mut byte)
                .map_err(|_| HandleError::SystemError(-1))?;
            if length != 1 {
                return Err(HandleError::SystemError(-1));
            }
            self.signalled.store(false, Ordering::Release);
        }
        Ok(())
    }

    fn poll_handle(&self) -> PollHandle {
        PollHandle::new(self.reader.as_raw() as u32, POLLIN)
    }
}

pub(crate) struct Signal {
    result: Mutex<Option<Result<(), Failure>>>,
    event: Event,
}

impl Signal {
    fn new() -> HandleResult<Self> {
        Ok(Self {
            result: Mutex::new(None),
            event: Event::new()?,
        })
    }

    pub(crate) fn poll(&self) -> Result<CompletionStatus, IrSubmitError> {
        match *lock(&self.result) {
            None => Ok(CompletionStatus::Pending),
            Some(Ok(())) => Ok(CompletionStatus::Complete),
            Some(Err(error)) => Err(error.into()),
        }
    }

    pub(crate) fn wait(
        &self,
        timeout: Option<Duration>,
    ) -> Result<CompletionStatus, IrSubmitError> {
        let started = monotonic_time_ns();
        loop {
            let status = self.poll()?;
            if status == CompletionStatus::Complete {
                return Ok(status);
            }
            let remaining = timeout.map(|limit| {
                limit.saturating_sub(Duration::from_nanos(
                    monotonic_time_ns().saturating_sub(started),
                ))
            });
            if remaining == Some(Duration::ZERO) {
                return Ok(CompletionStatus::Pending);
            }
            // A lost notification cannot strand terminal observation. Ordinary
            // completion wakes immediately; this is only an error fallback.
            let duration = remaining
                .unwrap_or(Duration::from_millis(100))
                .min(Duration::from_millis(100));
            poll(&mut [self.event.poll_handle()], duration.as_nanos() as i64)
                .map_err(|error| IrSubmitError::Backend(HandleError::SystemError(error)))?;
        }
    }

    fn complete(&self, result: Result<(), Failure>) {
        *lock(&self.result) = Some(result);
        self.event.notify();
    }
}

pub(crate) struct Chunk {
    pub(crate) commands: Vec<u8>,
    pub(crate) arena: Option<Arc<UploadArena>>,
}

struct Native;

impl Transport for Native {
    type Chunk = Chunk;
    // Scarlet queues retain their context and every attachment independently of
    // user handles. Holding this queue protects even not-yet-dispatched chunks.
    // Explicit detach/transfer/readback paths drain this scheduler first.
    type Owner = Arc<GpuQueue>;
    type Receipt = Arc<GpuCompletion>;
    type Signal = Arc<Signal>;
    type Error = Failure;

    fn size(chunk: &Chunk) -> usize {
        chunk.commands.len()
    }

    fn ready(&self, chunk: &Chunk) -> Result<bool, Failure> {
        let Some(arena) = &chunk.arena else {
            return Ok(true);
        };
        let previous = lock(&arena.completion).clone();
        previous
            .as_ref()
            .map_or(Ok(true), |completion| native_status(completion))
    }

    fn submit(
        &self,
        queue: &Arc<GpuQueue>,
        chunk: &Chunk,
    ) -> Result<Arc<GpuCompletion>, DispatchError<Failure>> {
        match queue.submit_async(&chunk.commands) {
            Ok(completion) => {
                let completion = Arc::new(completion);
                if let Some(arena) = &chunk.arena {
                    *lock(&arena.completion) = Some(Arc::clone(&completion));
                }
                Ok(completion)
            }
            Err(GpuSubmitError::Busy) => Err(DispatchError::Busy),
            Err(GpuSubmitError::Rejected(error) | GpuSubmitError::Failed { error, .. }) => {
                // Acceptance happened at logical enqueue. Any later rejection
                // or uncertain native acceptance fails the receipt and poisons
                // the queue, never replays it. Kernel-owned references protect
                // possibly accepted work after these observation handles drop.
                Err(DispatchError::Failed(Failure::Backend(error)))
            }
            Err(_) => Err(DispatchError::Failed(Failure::Unavailable)),
        }
    }

    fn poll(&self, receipt: &Arc<GpuCompletion>) -> Result<bool, Failure> {
        native_status(receipt)
    }
    fn complete(&self, signal: &Arc<Signal>, result: Result<(), Failure>) {
        signal.complete(result);
    }
}

struct Shared {
    scheduler: Mutex<Scheduler<Native>>,
    wake: Event,
    closing: AtomicBool,
}

pub(crate) struct NativeScheduler {
    shared: Arc<Shared>,
}

impl NativeScheduler {
    pub(crate) fn new() -> HandleResult<Self> {
        let shared = Arc::new(Shared {
            scheduler: Mutex::new(Scheduler::new()),
            wake: Event::new()?,
            closing: AtomicBool::new(false),
        });
        let worker = Arc::clone(&shared);
        std::thread::Builder::new()
            .spawn(move || run(worker))
            .map_err(|_| HandleError::OutOfResources)?;
        Ok(Self { shared })
    }

    pub(crate) fn enqueue(
        &self,
        queue: Arc<GpuQueue>,
        chunks: Vec<Chunk>,
    ) -> Result<Arc<Signal>, AdmissionError<Failure>> {
        let signal = Arc::new(Signal::new().map_err(|_| AdmissionError::OutOfMemory)?);
        lock(&self.shared.scheduler).enqueue(chunks, queue, Arc::clone(&signal))?;
        self.shared.wake.notify();
        Ok(signal)
    }

    pub(crate) fn wait_idle(&self) -> HandleResult<()> {
        let signal = {
            let scheduler = lock(&self.shared.scheduler);
            if scheduler.failure().is_some() {
                return Err(HandleError::SystemError(-1));
            }
            scheduler.last_signal().cloned()
        };
        if let Some(signal) = signal
            && signal
                .wait(None)
                .map_err(|_| HandleError::SystemError(-1))?
                != CompletionStatus::Complete
        {
            return Err(HandleError::SystemError(-1));
        }
        Ok(())
    }

    pub(crate) fn check(&self) -> HandleResult<()> {
        if lock(&self.shared.scheduler).failure().is_some() {
            Err(HandleError::SystemError(-1))
        } else {
            Ok(())
        }
    }
}

impl Drop for NativeScheduler {
    fn drop(&mut self) {
        // The worker, not a receipt/session, owns the shared dispatch state.
        // Closing the last frontend owner requests exit only after retirement.
        self.shared.closing.store(true, Ordering::Release);
        self.shared.wake.notify();
    }
}

struct WorkerGuard(Arc<Shared>);
impl Drop for WorkerGuard {
    fn drop(&mut self) {
        let mut scheduler = lock(&self.0.scheduler);
        if !scheduler.is_empty() {
            scheduler.fail(&Native, Failure::Unavailable);
        }
    }
}

fn run(shared: Arc<Shared>) {
    let _guard = WorkerGuard(Arc::clone(&shared));
    // One wake handle plus at most sixteen native completion handles.
    let mut handles = Vec::new();
    if handles.try_reserve_exact(17).is_err() {
        lock(&shared.scheduler).fail(&Native, Failure::Backend(HandleError::OutOfResources));
        return;
    }
    loop {
        if let Err(error) = shared.wake.consume() {
            lock(&shared.scheduler).fail(&Native, Failure::Backend(error));
            return;
        }
        handles.clear();
        handles.push(shared.wake.poll_handle());
        let (progressed, pending) = {
            let mut scheduler = lock(&shared.scheduler);
            let progressed = scheduler.advance(&Native);
            if scheduler.failure().is_some()
                || scheduler.is_empty() && shared.closing.load(Ordering::Acquire)
            {
                return;
            }
            handles.extend(
                scheduler
                    .receipts()
                    .map(|receipt| PollHandle::new(receipt.as_handle().as_raw() as u32, POLLIN)),
            );
            (progressed, !scheduler.is_empty())
        };
        if progressed {
            continue;
        }
        // With no local fence, Busy may come from another context on the
        // shared device. Only the worker retries that transport-level pressure.
        let timeout = if pending && handles.len() == 1 {
            1_000_000
        } else {
            100_000_000
        };
        if let Err(error) = poll(&mut handles, timeout) {
            lock(&shared.scheduler)
                .fail(&Native, Failure::Backend(HandleError::SystemError(error)));
            return;
        }
    }
}
