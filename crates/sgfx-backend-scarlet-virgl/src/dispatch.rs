//! Autonomous context-ordered dispatch and owned logical completion signals.

use alloc::{collections::VecDeque, sync::Arc, vec::Vec};
use core::sync::atomic::{AtomicBool, Ordering};
use core::time::Duration;
#[cfg(feature = "std")]
use std::sync::Condvar;
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
use crate::scheduler::{
    AdmissionError, DispatchError, Job, MAX_PENDING_BYTES, MAX_STREAM_BYTES, MAX_SUBMISSIONS,
    Scheduler, Transport,
};
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
    let info = completion.query().map_err(|error| {
        trace_failure("completion query", error);
        Failure::Backend(error)
    })?;
    completion_status(info)
        .map(|status| status == CompletionStatus::Complete)
        .map_err(|error| {
            trace_failure("completion status", &error);
            match error {
                IrSubmitError::CompletionFailed(reason) => Failure::Completion(reason),
                _ => Failure::Unavailable,
            }
        })
}

fn trace_failure(operation: &str, error: impl core::fmt::Debug) {
    #[cfg(feature = "std")]
    std::eprintln!("[SGFX native] {operation} failed: {error:?}");
    #[cfg(not(feature = "std"))]
    let _ = (operation, error);
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
    bytes: usize,
    result: Mutex<Option<Result<(), Failure>>>,
    #[cfg(feature = "std")]
    changed: Condvar,
    #[cfg(not(feature = "std"))]
    event: Event,
}

impl Signal {
    pub(crate) fn new() -> HandleResult<Self> {
        Self::with_bytes(0)
    }

    fn with_bytes(bytes: usize) -> HandleResult<Self> {
        Ok(Self {
            bytes,
            result: Mutex::new(None),
            #[cfg(feature = "std")]
            changed: Condvar::new(),
            #[cfg(not(feature = "std"))]
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

    #[cfg(feature = "std")]
    pub(crate) fn wait(
        &self,
        timeout: Option<Duration>,
    ) -> Result<CompletionStatus, IrSubmitError> {
        let started = monotonic_time_ns();
        let mut result = lock(&self.result);
        loop {
            match *result {
                Some(Ok(())) => return Ok(CompletionStatus::Complete),
                Some(Err(error)) => return Err(error.into()),
                None => {}
            }
            // Check and sleep under the same mutex used by complete. Receipts
            // need no pollable handle, so each stream avoids creating a pipe.
            result = if let Some(limit) = timeout {
                let remaining = limit.saturating_sub(Duration::from_nanos(
                    monotonic_time_ns().saturating_sub(started),
                ));
                if remaining.is_zero() {
                    return Ok(CompletionStatus::Pending);
                }
                self.changed
                    .wait_timeout(result, remaining)
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .0
            } else {
                self.changed
                    .wait(result)
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
            };
        }
    }

    #[cfg(not(feature = "std"))]
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

    pub(crate) fn complete(&self, result: Result<(), Failure>) {
        *lock(&self.result) = Some(result);
        #[cfg(feature = "std")]
        self.changed.notify_all();
        #[cfg(not(feature = "std"))]
        self.event.notify();
    }
}

pub(crate) struct Chunk {
    pub(crate) commands: Vec<u8>,
    pub(crate) arena: Option<Arc<UploadArena>>,
}

struct Native {
    shared: Arc<Shared>,
}

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
            Err(error) => {
                trace_failure("asynchronous submit", &error);
                // Acceptance happened at logical enqueue. Any later rejection
                // or uncertain native acceptance fails the receipt and poisons
                // the queue, never replays it. Kernel-owned references protect
                // possibly accepted work after these observation handles drop.
                let failure = match error {
                    GpuSubmitError::Rejected(error) | GpuSubmitError::Failed { error, .. } => {
                        Failure::Backend(error)
                    }
                    _ => Failure::Unavailable,
                };
                Err(DispatchError::Failed(failure))
            }
        }
    }

    fn poll(&self, receipt: &Arc<GpuCompletion>) -> Result<bool, Failure> {
        native_status(receipt)
    }
    fn complete(&self, signal: &Arc<Signal>, result: Result<(), Failure>) {
        self.shared.finish_signal(signal, result);
    }
}

// Only admission accounting is shared with producers. GPU syscalls and receipt
// polling run on the worker's private Scheduler, outside this short mutex.
struct Admission {
    pending: VecDeque<Job<Native>>,
    count: usize,
    bytes: usize,
    last: Option<Arc<Signal>>,
    failure: Option<Failure>,
}

struct Shared {
    admission: Mutex<Admission>,
    wake: Event,
    closing: AtomicBool,
    failed: AtomicBool,
}

impl Shared {
    fn finish_signal(&self, signal: &Arc<Signal>, result: Result<(), Failure>) {
        {
            let mut state = lock(&self.admission);
            state.count -= 1;
            state.bytes -= signal.bytes;
            if let Err(error) = result {
                state.failure = Some(error);
                self.failed.store(true, Ordering::Release);
            }
            if state.count == 0 {
                state.last = None;
            }
        }
        signal.complete(result);
    }
}

pub(crate) struct NativeScheduler {
    shared: Arc<Shared>,
}

impl NativeScheduler {
    pub(crate) fn new() -> HandleResult<Self> {
        let mut pending = VecDeque::new();
        pending
            .try_reserve(MAX_SUBMISSIONS)
            .map_err(|_| HandleError::OutOfResources)?;
        let mut scheduler = Scheduler::new();
        scheduler
            .reserve_admission()
            .map_err(|_| HandleError::OutOfResources)?;
        let shared = Arc::new(Shared {
            admission: Mutex::new(Admission {
                pending,
                count: 0,
                bytes: 0,
                last: None,
                failure: None,
            }),
            wake: Event::new()?,
            closing: AtomicBool::new(false),
            failed: AtomicBool::new(false),
        });
        let worker = Worker {
            native: Native {
                shared: Arc::clone(&shared),
            },
            scheduler,
        };
        std::thread::Builder::new()
            .spawn(move || worker.run())
            .map_err(|_| HandleError::OutOfResources)?;
        Ok(Self { shared })
    }

    pub(crate) fn enqueue(
        &self,
        queue: Arc<GpuQueue>,
        chunks: Vec<Chunk>,
    ) -> Result<Arc<Signal>, AdmissionError<Failure>> {
        let bytes = chunks
            .iter()
            .try_fold(0usize, |n, chunk| n.checked_add(chunk.commands.len()))
            .filter(|n| *n <= MAX_STREAM_BYTES)
            .ok_or(AdmissionError::TooLarge)?;
        let signal = Arc::new(Signal::with_bytes(bytes).map_err(|_| AdmissionError::OutOfMemory)?);
        let job = Job::prepare(chunks, queue, Arc::clone(&signal))?;
        {
            let mut state = lock(&self.shared.admission);
            if let Some(error) = state.failure {
                return Err(AdmissionError::Failed(error));
            }
            if state.count == MAX_SUBMISSIONS
                || (state.count != 0
                    && (bytes > MAX_PENDING_BYTES
                        || state.bytes.saturating_add(bytes) > MAX_PENDING_BYTES))
            {
                return Err(AdmissionError::Busy);
            }
            // Capacity was reserved at construction. Global admission includes
            // both this inbox and the worker's active streams.
            state.pending.push_back(job);
            state.count += 1;
            state.bytes += bytes;
            state.last = Some(Arc::clone(&signal));
        }
        self.shared.wake.notify();
        Ok(signal)
    }

    pub(crate) fn wait_idle(&self) -> HandleResult<()> {
        let signal = {
            let state = lock(&self.shared.admission);
            if state.failure.is_some() {
                return Err(HandleError::SystemError(-1));
            }
            state.last.clone()
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

    #[cfg(feature = "backend-abi")]
    pub(crate) fn is_idle(&self) -> HandleResult<bool> {
        let state = lock(&self.shared.admission);
        if state.failure.is_some() {
            return Err(HandleError::SystemError(-1));
        }
        Ok(state.count == 0)
    }

    pub(crate) fn check(&self) -> HandleResult<()> {
        if self.shared.failed.load(Ordering::Acquire) {
            Err(HandleError::SystemError(-1))
        } else {
            Ok(())
        }
    }
}

impl Drop for NativeScheduler {
    fn drop(&mut self) {
        self.shared.closing.store(true, Ordering::Release);
        self.shared.wake.notify();
    }
}

struct Worker {
    native: Native,
    scheduler: Scheduler<Native>,
}

impl Worker {
    fn fail(&mut self, error: Failure) {
        let pending = {
            let mut state = lock(&self.native.shared.admission);
            state.failure = Some(error);
            self.native.shared.failed.store(true, Ordering::Release);
            core::mem::take(&mut state.pending)
        };
        self.scheduler.fail(&self.native, error);
        for job in pending {
            self.native.complete(&job.signal, Err(error));
        }
    }

    fn run(mut self) {
        let mut handles = Vec::new();
        if handles.try_reserve_exact(17).is_err() {
            self.fail(Failure::Backend(HandleError::OutOfResources));
            return;
        }
        loop {
            if let Err(error) = self.native.shared.wake.consume() {
                self.fail(Failure::Backend(error));
                return;
            }
            loop {
                let job = { lock(&self.native.shared.admission).pending.pop_front() };
                let Some(job) = job else {
                    break;
                };
                let signal = Arc::clone(&job.signal);
                // The shared admission already reserved capacity and owns all
                // allocations. Transfer preserves FIFO without a GPU syscall.
                if self.scheduler.enqueue_prepared(job).is_err() {
                    self.native.complete(&signal, Err(Failure::Unavailable));
                    self.fail(Failure::Unavailable);
                    return;
                }
            }
            let progressed = self.scheduler.advance(&self.native);
            if let Some(error) = self.scheduler.failure() {
                self.fail(error);
                return;
            }
            if self.scheduler.is_empty()
                && self.native.shared.closing.load(Ordering::Acquire)
                && lock(&self.native.shared.admission).count == 0
            {
                return;
            }
            if progressed {
                continue;
            }
            handles.clear();
            handles.push(self.native.shared.wake.poll_handle());
            handles.extend(
                self.scheduler
                    .receipts()
                    .map(|receipt| PollHandle::new(receipt.as_handle().as_raw() as u32, POLLIN)),
            );
            let timeout = if !self.scheduler.is_empty() && handles.len() == 1 {
                1_000_000
            } else {
                100_000_000
            };
            if let Err(error) = poll(&mut handles, timeout) {
                self.fail(Failure::Backend(HandleError::SystemError(error)));
                return;
            }
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let pending = { lock(&self.native.shared.admission).count };
        if !self.scheduler.is_empty() || pending != 0 {
            self.fail(Failure::Unavailable);
        }
    }
}
