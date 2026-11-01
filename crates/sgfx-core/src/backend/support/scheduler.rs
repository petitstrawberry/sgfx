//! Bounded logical admission and ordered native dispatch, independent of syscalls.

extern crate alloc;

use alloc::{collections::VecDeque, vec::Vec};

/// Admission and native in-flight budgets selected by each transport.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub submissions: usize,
    pub pending_bytes: usize,
    pub stream_bytes: usize,
    pub in_flight: usize,
}
impl Limits {
    fn validate(self) {
        assert!(
            self.submissions > 0
                && self.pending_bytes > 0
                && self.stream_bytes > 0
                && self.in_flight > 0
        );
    }
}

pub enum DispatchError<E> {
    Busy,
    Failed(E),
}

#[derive(Debug, PartialEq, Eq)]
pub enum AdmissionError<E> {
    Busy,
    TooLarge,
    OutOfMemory,
    Failed(E),
}

pub trait Transport {
    type Chunk;
    type Owner;
    type Receipt;
    type Signal;
    type Error: Copy;

    fn size(chunk: &Self::Chunk) -> usize;
    fn ready(&self, chunk: &Self::Chunk) -> Result<bool, Self::Error>;
    fn submit(
        &self,
        owner: &Self::Owner,
        chunk: &Self::Chunk,
    ) -> Result<Self::Receipt, DispatchError<Self::Error>>;
    fn poll(&self, receipt: &Self::Receipt) -> Result<bool, Self::Error>;
    fn complete(&self, signal: &Self::Signal, result: Result<(), Self::Error>);
}

pub struct Job<T: Transport> {
    chunks: Vec<T::Chunk>,
    owner: T::Owner,
    pub signal: T::Signal,
    next: usize,
    receipts: VecDeque<T::Receipt>,
    bytes: usize,
}

impl<T: Transport> Job<T> {
    pub fn prepare(
        chunks: Vec<T::Chunk>,
        owner: T::Owner,
        signal: T::Signal,
        limits: Limits,
    ) -> Result<Self, AdmissionError<T::Error>> {
        limits.validate();
        let bytes = chunks
            .iter()
            .try_fold(0usize, |size, chunk| size.checked_add(T::size(chunk)))
            .filter(|size| *size <= limits.stream_bytes)
            .ok_or(AdmissionError::TooLarge)?;
        let mut receipts = VecDeque::new();
        receipts
            .try_reserve(chunks.len().min(limits.in_flight))
            .map_err(|_| AdmissionError::OutOfMemory)?;
        Ok(Self {
            chunks,
            owner,
            signal,
            next: 0,
            receipts,
            bytes,
        })
    }
}

pub struct Scheduler<T: Transport> {
    jobs: VecDeque<Job<T>>,
    bytes: usize,
    failure: Option<T::Error>,
    limits: Limits,
}

impl<T: Transport> Scheduler<T> {
    pub fn new(limits: Limits) -> Self {
        limits.validate();
        Self {
            limits,
            jobs: VecDeque::new(),
            bytes: 0,
            failure: None,
        }
    }

    pub fn failure(&self) -> Option<T::Error> {
        self.failure
    }

    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    pub fn receipts(&self) -> impl Iterator<Item = &T::Receipt> {
        self.jobs.iter().flat_map(|job| job.receipts.iter())
    }

    pub fn last_signal(&self) -> Option<&T::Signal> {
        self.jobs.back().map(|job| &job.signal)
    }

    pub fn reserve_admission(&mut self) -> Result<(), AdmissionError<T::Error>> {
        self.jobs
            .try_reserve(self.limits.submissions)
            .map_err(|_| AdmissionError::OutOfMemory)
    }

    // All fallible allocation precedes publication. No transport operation is
    // performed here, even when hardware currently has no free descriptors.
    pub fn enqueue(
        &mut self,
        chunks: Vec<T::Chunk>,
        owner: T::Owner,
        signal: T::Signal,
    ) -> Result<(), AdmissionError<T::Error>> {
        self.enqueue_prepared(Job::prepare(chunks, owner, signal, self.limits)?)
    }

    pub fn enqueue_prepared(&mut self, job: Job<T>) -> Result<(), AdmissionError<T::Error>> {
        if let Some(error) = self.failure {
            return Err(AdmissionError::Failed(error));
        }
        let bytes = job.bytes;
        if bytes > self.limits.stream_bytes {
            return Err(AdmissionError::TooLarge);
        }
        if self.jobs.len() >= self.limits.submissions
            || (!self.jobs.is_empty()
                && (bytes > self.limits.pending_bytes
                    || self.bytes.saturating_add(bytes) > self.limits.pending_bytes))
        {
            return Err(AdmissionError::Busy);
        }
        // Preparation may have used another queue's limits. Reserve this
        // queue's maximum before publishing ownership, never during dispatch.
        let mut job = job;
        job.receipts
            .try_reserve(job.chunks.len().min(self.limits.in_flight))
            .map_err(|_| AdmissionError::OutOfMemory)?;
        self.jobs
            .try_reserve(1)
            .map_err(|_| AdmissionError::OutOfMemory)?;
        self.jobs.push_back(job);
        self.bytes += bytes;
        Ok(())
    }

    pub fn fail(&mut self, transport: &T, error: T::Error) {
        self.failure = Some(error);
        for job in self.jobs.drain(..) {
            transport.complete(&job.signal, Err(error));
        }
        self.bytes = 0;
    }

    // Called only by the queue worker. Busy leaves the exact next chunk in
    // place; accepted chunks are never replayed and later jobs cannot pass it.
    // Nothing here waits for GPU completion.
    pub fn advance(&mut self, transport: &T) -> bool {
        let mut progressed = false;
        for job in &mut self.jobs {
            let mut index = 0;
            while let Some(receipt) = job.receipts.get(index) {
                match transport.poll(receipt) {
                    Ok(true) => {
                        // Completion readiness can arrive out of order. Remove
                        // every retired handle from the worker's poll set so
                        // a later signalled event cannot make it spin behind
                        // an earlier pending chunk. Job retirement below still
                        // requires the entire ordered prefix to complete.
                        job.receipts.remove(index);
                        progressed = true;
                    }
                    Ok(false) => index += 1,
                    Err(error) => {
                        self.fail(transport, error);
                        return true;
                    }
                }
            }
        }
        // Even a later fence observed first cannot certify its queue prefix.
        while self
            .jobs
            .front()
            .is_some_and(|job| job.next == job.chunks.len() && job.receipts.is_empty())
        {
            if let Some(job) = self.jobs.pop_front() {
                self.bytes -= job.bytes;
                transport.complete(&job.signal, Ok(()));
                progressed = true;
            }
        }

        let mut in_flight = self.receipts().count();
        for job in &mut self.jobs {
            while let Some(chunk) = job.chunks.get(job.next) {
                if in_flight >= self.limits.in_flight {
                    return progressed;
                }
                match transport.ready(chunk) {
                    Ok(true) => {}
                    Ok(false) => return progressed,
                    Err(error) => {
                        self.fail(transport, error);
                        return true;
                    }
                }
                match transport.submit(&job.owner, chunk) {
                    Ok(receipt) => {
                        job.receipts.push_back(receipt);
                        job.next += 1;
                        in_flight += 1;
                        progressed = true;
                    }
                    Err(DispatchError::Busy) => return progressed,
                    Err(DispatchError::Failed(error)) => {
                        self.fail(transport, error);
                        return true;
                    }
                }
            }
        }
        progressed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const MAX_SUBMISSIONS: usize = 16;
    const MAX_PENDING_BYTES: usize = 64 * 1024 * 1024;
    const MAX_STREAM_BYTES: usize = 512 * 1024 * 1024;
    const LIMITS: Limits = Limits {
        submissions: MAX_SUBMISSIONS,
        pending_bytes: MAX_PENDING_BYTES,
        stream_bytes: MAX_STREAM_BYTES,
        in_flight: 16,
    };
    use alloc::rc::Rc;
    use core::cell::{Cell, RefCell};

    type Signal = Rc<Cell<Option<Result<(), u32>>>>;

    #[derive(Default)]
    struct Fake {
        accepted: RefCell<Vec<u32>>,
        fences: RefCell<Vec<Signal>>,
        busy: Cell<bool>,
        ready: Cell<bool>,
        fail_submit: Cell<bool>,
        busy_after: Cell<Option<usize>>,
    }

    impl Transport for Fake {
        type Chunk = (u32, usize);
        type Owner = Rc<()>;
        type Receipt = Signal;
        type Signal = Signal;
        type Error = u32;
        fn size(chunk: &Self::Chunk) -> usize {
            chunk.1
        }
        fn ready(&self, _: &Self::Chunk) -> Result<bool, u32> {
            Ok(self.ready.get())
        }
        fn submit(&self, _: &Rc<()>, chunk: &Self::Chunk) -> Result<Signal, DispatchError<u32>> {
            if self.busy.get() {
                return Err(DispatchError::Busy);
            }
            if self
                .busy_after
                .get()
                .is_some_and(|limit| self.accepted.borrow().len() >= limit)
            {
                return Err(DispatchError::Busy);
            }
            if self.fail_submit.get() {
                return Err(DispatchError::Failed(7));
            }
            self.accepted.borrow_mut().push(chunk.0);
            let fence = Rc::new(Cell::new(None));
            self.fences.borrow_mut().push(Rc::clone(&fence));
            Ok(fence)
        }
        fn poll(&self, receipt: &Signal) -> Result<bool, u32> {
            receipt.get().transpose().map(|value| value.is_some())
        }
        fn complete(&self, signal: &Signal, result: Result<(), u32>) {
            signal.set(Some(result));
        }
    }

    fn add(queue: &mut Scheduler<Fake>, chunks: Vec<(u32, usize)>) -> Signal {
        let signal = Rc::new(Cell::new(None));
        queue
            .enqueue(chunks, Rc::new(()), Rc::clone(&signal))
            .unwrap();
        signal
    }

    fn ready() -> Fake {
        Fake {
            ready: Cell::new(true),
            ..Fake::default()
        }
    }

    #[test]
    fn large_logical_stream_is_admitted_once_then_dispatched_in_order() {
        let native = ready();
        let mut queue = Scheduler::new(LIMITS);
        let a = add(&mut queue, (0..40).map(|id| (id, 64 * 1024)).collect());
        let b = add(&mut queue, alloc::vec![(40, 0)]);
        assert!(Rc::ptr_eq(queue.last_signal().unwrap(), &b));
        assert!(native.accepted.borrow().is_empty());
        queue.advance(&native);
        assert_eq!(*native.accepted.borrow(), (0..16).collect::<Vec<_>>());
        assert_eq!(a.get(), None);
        for _ in 0..4 {
            for fence in native.fences.borrow().iter() {
                fence.set(Some(Ok(())));
            }
            queue.advance(&native);
        }
        assert_eq!(*native.accepted.borrow(), (0..41).collect::<Vec<_>>());
        assert_eq!(a.get(), Some(Ok(())));
        assert_eq!(b.get(), Some(Ok(())));
        assert!(queue.is_empty());
    }

    #[test]
    fn native_busy_and_upload_storage_pressure_never_replay_or_overtake() {
        let native = ready();
        let mut queue = Scheduler::new(LIMITS);
        add(&mut queue, alloc::vec![(1, 1), (2, 1)]);
        add(&mut queue, alloc::vec![(3, 1)]);
        native.busy.set(true);
        assert!(!queue.advance(&native));
        assert!(native.accepted.borrow().is_empty());
        native.busy.set(false);
        native.busy_after.set(Some(1));
        queue.advance(&native);
        assert_eq!(*native.accepted.borrow(), [1]);
        assert!(!queue.advance(&native));
        native.busy_after.set(None);
        native.ready.set(false);
        assert!(!queue.advance(&native));
        native.ready.set(true);
        queue.advance(&native);
        queue.advance(&native);
        assert_eq!(*native.accepted.borrow(), [1, 2, 3]);
    }

    #[test]
    fn dropped_receipt_and_owner_do_not_abandon_queued_work() {
        let native = ready();
        let mut queue = Scheduler::new(LIMITS);
        let owner = Rc::new(());
        let retained = Rc::downgrade(&owner);
        let signal = Rc::new(Cell::new(None));
        queue.enqueue(alloc::vec![(1, 1)], owner, signal).unwrap();
        assert!(retained.upgrade().is_some());
        queue.advance(&native);
        assert!(retained.upgrade().is_some());
        native.fences.borrow()[0].set(Some(Ok(())));
        queue.advance(&native);
        assert!(retained.upgrade().is_none());
    }

    #[test]
    fn byte_and_slot_pressure_are_side_effect_free_admission() {
        let mut queue = Scheduler::<Fake>::new(LIMITS);
        let signal = Rc::new(Cell::new(None));
        assert_eq!(
            queue.enqueue(
                alloc::vec![(1, MAX_STREAM_BYTES + 1)],
                Rc::new(()),
                Rc::clone(&signal)
            ),
            Err(AdmissionError::TooLarge)
        );
        add(&mut queue, alloc::vec![(1, MAX_PENDING_BYTES)]);
        assert_eq!(
            queue.enqueue(alloc::vec![(2, 1)], Rc::new(()), Rc::clone(&signal)),
            Err(AdmissionError::Busy)
        );
        let mut queue = Scheduler::<Fake>::new(LIMITS);
        for _ in 0..MAX_SUBMISSIONS {
            add(&mut queue, alloc::vec![(1, 0)]);
        }
        assert_eq!(
            queue.enqueue(alloc::vec![], Rc::new(()), signal),
            Err(AdmissionError::Busy)
        );
    }

    #[test]
    fn large_stream_waits_for_exclusive_admission_and_retires_every_chunk() {
        let native = ready();
        let mut queue = Scheduler::new(LIMITS);
        let prior = add(&mut queue, alloc::vec![(0, 1)]);
        let signal = Rc::new(Cell::new(None));
        let chunks = || alloc::vec![(1, MAX_PENDING_BYTES), (2, 65536)];
        assert_eq!(
            queue.enqueue(chunks(), Rc::new(()), Rc::clone(&signal)),
            Err(AdmissionError::Busy)
        );
        assert!(native.accepted.borrow().is_empty());
        queue.advance(&native);
        native.fences.borrow()[0].set(Some(Ok(())));
        queue.advance(&native);
        assert_eq!(prior.get(), Some(Ok(())));
        queue
            .enqueue(chunks(), Rc::new(()), Rc::clone(&signal))
            .unwrap();
        assert_eq!(
            queue.enqueue(alloc::vec![], Rc::new(()), Rc::new(Cell::new(None))),
            Err(AdmissionError::Busy)
        );
        queue.advance(&native);
        assert_eq!(&*native.accepted.borrow(), &[0, 1, 2]);
        native.fences.borrow()[2].set(Some(Ok(())));
        queue.advance(&native);
        assert_eq!(signal.get(), None);
        native.fences.borrow()[1].set(Some(Ok(())));
        queue.advance(&native);
        assert_eq!(signal.get(), Some(Ok(())));
        assert!(queue.is_empty());
        add(&mut queue, alloc::vec![(3, MAX_PENDING_BYTES)]);
    }

    #[test]
    fn later_completion_does_not_certify_an_unfinished_prefix() {
        let native = ready();
        let mut queue = Scheduler::new(LIMITS);
        let a = add(&mut queue, alloc::vec![(1, 1)]);
        let b = add(&mut queue, alloc::vec![(2, 1)]);
        queue.advance(&native);
        native.fences.borrow()[1].set(Some(Ok(())));
        queue.advance(&native);
        assert_eq!((a.get(), b.get()), (None, None));
        native.fences.borrow()[0].set(Some(Ok(())));
        queue.advance(&native);
        assert_eq!((a.get(), b.get()), (Some(Ok(())), Some(Ok(()))));
    }

    #[test]
    fn late_dispatch_and_completion_failures_poison_all_following_work() {
        for dispatch in [false, true] {
            let native = ready();
            let mut queue = Scheduler::new(LIMITS);
            let a = add(&mut queue, (0..17).map(|id| (id, 1)).collect());
            let b = add(&mut queue, alloc::vec![(18, 1)]);
            queue.advance(&native);
            if dispatch {
                native.fences.borrow()[0].set(Some(Ok(())));
                native.fail_submit.set(true);
            } else {
                native.fences.borrow()[0].set(Some(Err(7)));
            }
            queue.advance(&native);
            assert_eq!((a.get(), b.get()), (Some(Err(7)), Some(Err(7))));
            assert_eq!(native.accepted.borrow().len(), 16);
            assert_eq!(queue.failure(), Some(7));
            assert!(queue.is_empty());
            assert_eq!(
                queue.enqueue(alloc::vec![], Rc::new(()), a),
                Err(AdmissionError::Failed(7))
            );
        }
    }

    #[test]
    fn completed_later_chunk_leaves_wait_set_without_certifying_prefix() {
        let native = ready();
        let mut queue = Scheduler::new(LIMITS);
        let signal = add(&mut queue, alloc::vec![(1, 1), (2, 1)]);
        queue.advance(&native);
        native.fences.borrow()[1].set(Some(Ok(())));
        assert!(queue.advance(&native));
        assert_eq!(queue.receipts().count(), 1);
        assert_eq!(signal.get(), None);
        // A completed event must not remain in the worker's readiness set:
        // it would wake immediately while the earlier chunk is still pending.
        assert!(!queue.advance(&native));
        native.fences.borrow()[0].set(Some(Ok(())));
        queue.advance(&native);
        assert_eq!(signal.get(), Some(Ok(())));
    }

    #[test]
    fn later_chunk_failure_is_observed_while_first_chunk_is_pending() {
        let native = ready();
        let mut queue = Scheduler::new(LIMITS);
        let signal = add(&mut queue, alloc::vec![(1, 1), (2, 1)]);
        let successor = add(&mut queue, alloc::vec![(3, 1)]);
        queue.advance(&native);
        native.fences.borrow()[1].set(Some(Err(7)));
        assert!(queue.advance(&native));
        assert_eq!(signal.get(), Some(Err(7)));
        assert_eq!(successor.get(), Some(Err(7)));
        assert_eq!(queue.failure(), Some(7));
    }
    #[test]
    fn another_transport_controls_admission_and_in_flight_limits() {
        let limits = Limits {
            submissions: 2,
            pending_bytes: 8,
            stream_bytes: 16,
            in_flight: 1,
        };
        let native = ready();
        let mut queue = Scheduler::new(limits);
        let first = add(&mut queue, alloc::vec![(0, 4), (1, 4)]);
        let signal = Rc::new(Cell::new(None));
        assert_eq!(
            queue.enqueue(alloc::vec![(2, 1)], Rc::new(()), Rc::clone(&signal)),
            Err(AdmissionError::Busy)
        );
        queue.advance(&native);
        assert_eq!(&*native.accepted.borrow(), &[0]);
        native.fences.borrow()[0].set(Some(Ok(())));
        queue.advance(&native);
        assert_eq!(&*native.accepted.borrow(), &[0, 1]);
        assert_eq!(first.get(), None);
        native.fences.borrow()[1].set(Some(Ok(())));
        queue.advance(&native);
        assert_eq!(first.get(), Some(Ok(())));
        assert_eq!(
            queue.enqueue(alloc::vec![(3, 17)], Rc::new(()), signal),
            Err(AdmissionError::TooLarge)
        );
    }
}
