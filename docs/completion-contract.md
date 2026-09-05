# Submission and completion for 1.0

Status: the user approved including portable completion tracking **and actual
asynchronous Scarlet GPU submission** in the 1.0 scope. The detailed interface
below is the implementation design for that agreed direction, not a claim
that all backends already support it. Other policies in the
[1.0 contract draft](1.0-contract.md) still require release review.

## Boundary

Keep the existing `CommandExecutor::execute` acceptance contract and the
existing synchronous Scarlet GPU ABI operations. Introduce `CommandSubmitter`
as an additional interface; its `submit` returns an owned `Submission` which
implements `Completion`. Official 1.0 backends and the facade must expose it.
A native implementation which simply wraps a synchronous submit in an
already-complete receipt does not meet the asynchronous Scarlet release gate.

`submit` performs CPU validation/lowering and takes ownership of data needed
by accepted work, but does not wait for that work's GPU completion. This is
not a hard real-time latency promise. Capacity is bounded: `Busy` rejects a
new submission without accepting any of its work, rather than silently waiting
for a previous GPU submission to make room.

A receipt covers the complete logical command stream, including all backend
chunks and uploads, and the preceding work ordered on its queue. It does not
cover later submissions, unrelated queues, display presentation, or SWS leases.
An empty tracked submission establishes a queue checkpoint; it is not an
immediate CPU wait. Existing `execute` does not gain a completion guarantee.

Receipts own their observation state. They do not borrow the executor, logical
table, or upload slices, and can outlive those objects. Dropping a receipt is
not cancellation, completion, or permission to recycle shared storage. The
backend/kernel independently retains the allocations and resource references
required by accepted work, even when every receipt or session owner is dropped.

## Observation

- `poll()` makes nonblocking observation progress and returns `Pending` or
  `Complete`, or an error. It must not wait for GPU completion. A caller does
  not need to know which backend event pump drives this operation.
- `wait(timeout)` observes the same receipt while allowing a CPU wait. `None`
  means no caller deadline; zero duration is a poll. Reaching a deadline while
  work is outstanding returns `Pending`, without cancelling or failing it.
- `Complete` certifies that covered GPU accesses have retired. It does not
  replace a readback/map operation, cache-coherency protocol, external acquire,
  or compositor acknowledgement. It is not a guarantee of pixel correctness.
- Device loss must be reported as an error, not mistaken for completion because
  a driver drained its callbacks. A later device loss may conservatively make
  an older receipt report an error; no error authorizes external-buffer reuse.
- There is no requirement for an async Rust runtime or a new common GPU-to-GPU
  semaphore API. Event-loop/Future adapters can be additive.

## Submission errors and partial acceptance

`SubmitError` distinguishes three cases:

| Result | New work from this call | What the caller retains |
| --- | --- | --- |
| `Busy` | Nothing accepted. | Original logical commands; retry after making capacity available. |
| `Rejected(error)` | Nothing accepted. | The error; previous queue/device work is not rolled back or certified healthy. |
| `Failed { error, completion }` | A prefix may have been accepted. | A receipt covering all possibly accepted work, plus the immediate error. Do not replay the stream as if nothing ran. |

Backends may conservatively use `Failed` when they cannot prove a side-effect-free
rejection. The attached receipt can itself fail on device loss. Such failure
does not prove hardware quiescence: retain affected resources until backend
reset/retirement establishes safety. Borrowed command/upload memory must no
longer be needed on **every** return path.

## Scarlet implementation obligations

The current `GpuQueue::submit_and_signal` still waits inside the driver and
signals its timeline before returning. A new, explicitly advertised async
operation must instead return after acceptance. Existing synchronous control
codes and their result/layout meanings remain unchanged.

1. Reserve bounded submission capacity and retain kernel-owned command bytes,
   completion state, and the referenced images/buffers before returning an
   accepted response. Handle response-copy failure without losing ownership
   of work already accepted by the kernel.
2. Provide authoritative, read-only completion observation. A user-signallable
   generic timeline alone is not proof that GPU work finished. Existing
   selectable-point machinery can inform event integration, but users must
   not be able to signal a queue's completion themselves.
3. Separate driver enqueue from completion handling. VirtIO response storage
   and A618 command staging/fence storage must survive the syscall and must
   not be reused while hardware can access them. Bounded rings/pools replace
   the assumption that the previous synchronous call has already finished.
4. Preserve upload/copy/draw ordering and snapshot the resource authority for
   each accepted submission. Concurrent detach, handle close, process exit,
   timeout, fault and reset must not free in-flight resources prematurely.
5. Move SGFX's native chunking onto the new path. Track the last accepted chunk
   even when a later chunk fails. Adapt SWS/UI buffer retirement so they do not
   immediately wait after every submission or signal frame release too early.

## Implementation checkpoints

- [x] Core receipt/observation/error traits and deterministic contract tests.
- [x] WGPU tracked submission, bounded tracking, failure receipts and host facade.
- [x] Additive Scarlet async ABI and authoritative completion objects.
- [x] VirtIO/VirGL asynchronous enqueue/completion with retained resources.
- [ ] A618 asynchronous enqueue/completion and safe staging reuse.
- [ ] Native SGFX/facade support and consumer lifetime integration.
- [ ] Both architectures' tests/builds, real host rendering, and A618 hardware
  evidence, including multiple outstanding submissions and failure/teardown.

These are checkpoints, not independent substitutes for the agreed end-to-end
goal. No version bump or candidate tag is warranted by these checkpoints alone.

The initial core/host implementation passes 45 portable tests with Scarlet Rust
and upstream Rust, including five real-device completion scenarios on Metal.
Strict host Clippy and Rustdoc pass, as do std and backend-free checks for both
normal Scarlet targets. Those cross-checks preserve the old native path; they
do not certify native tracked/asynchronous submission. Existing tests continue
to cover the retained untracked executor alongside the new path.

Scarlet's [native implementation boundary](https://github.com/petitstrawberry/Scarlet/blob/992d1b9dace1bd8ec7e0366741ac93ac930c9ab7/docs/graphics/gpu-async-submission.md)
now includes the generic ABI **and real VirtIO/VirGL async execution**. The
driver retains up to 16 submissions across the device, publishes payloads with
independent retirement checkpoints, and progresses via a kernel worker with
IRQ notification and timed fallback. Legacy detach/transfer/presentation remains
synchronous and ordered after preceding async work. Faulted, unretired resources
remain bounded and quarantined without a reset/quiescence proof.

Kernel tests pass on both architectures (1,187 RISC-V / 1,158 AArch64), as do both
full builds. The new normal-std `gpu-async-smoke` passes check and strict Clippy
on both Scarlet targets. Real AArch64 QEMU release-image runs pass with VirGL
PCI/two CPUs and MMIO/one CPU: exact async clear/readback, ordered checkpoints,
detach, dropped receipts, and completion after closing all owner handles.
These are driver-level checks, not yet native SGFX adapter tests.

**A618 still advertises zero async capacity**, and native SGFX/facade/SWS/UI
still use their old paths. A618's staging/fence ownership and mapping retention,
plus SGFX native receipts and chunking, remain to be implemented. A raw control or
handle-adoption failure can also lose the current receipt; the SGFX adapter
must report an unobservable failure, not certify unknown accepted work using
only a preceding chunk's receipt. Native driver/consumer integration and fault,
reset, teardown and hardware evidence remain open.
