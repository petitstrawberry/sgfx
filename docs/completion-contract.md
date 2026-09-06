# Submission and completion for 1.0

Status: the user approved including portable completion tracking **and actual
asynchronous Scarlet GPU submission** in the 1.0 scope. The detailed interface
below is the implementation design for that agreed direction, not a claim
that all backends already support it. The
[1.0 execution contract](1.0-contract.md) also records the approved coordinated
Rust update policy; remaining rendering and lifecycle clauses require review.

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

The agreed native boundary permits **synchronous first-use physical resource
creation**. Scarlet/VirGL materializes all new buffers/images needed by a stream
before sending any of that stream's uploads, copies, or draws. These existing
resource-creation calls can drain earlier GPU work. They are not repeated for
cached resources. After cold setup, uploads/copies/draws and checkpoints use the
async transport. Submit does not wait for GPU completion or admission capacity;
only the independent dispatcher may wait before sending another native packet.

VirGL separates **logical acceptance** from **native dispatch**. A context-owned
worker accepts a fully validated/lowered, owned logical stream into a FIFO
bounded to 16 streams and 64 MiB of retained command bytes. Insufficient logical
capacity returns `Busy` before accepting any work; a single stream exceeding
64 MiB is explicitly `Rejected(SubmissionTooLarge)`. CPU initialization and
buffer revisions are restored on rejection. The advertised 2 MiB native request
limit is a packetization detail, not a mesh, texture, or UI frame-size limit.

The worker splits only at complete native packet boundaries, keeps up to 16
native requests in flight, and resumes the exact next packet after native Busy
or staging-arena pressure. It never replays an accepted prefix. Later logical
streams cannot overtake an undispatched prefix. Queue wrappers of the same
VirGL context share this dispatcher; native synchronous execution, transfer,
readback, and explicit detach drain preceding logical work first.

An accepted stream's receipt stays Pending until **all** its packets and its
ordered prefix complete. Later transport/adoption/observation failure fails the
receipt and poisons the dispatcher, including queued successors. Such errors do
not certify quiescence and must never be blindly replayed. The worker progresses
without receipt polling and drains accepted work after all frontend owners drop.
Scarlet's owning queue capability retains the context and attached resources
for packets not yet dispatched; kernel-owned requests retain dispatched work.

ScarletUI/SWS own frame boundaries, presentation, and lease release. They do not
split native uploads, subdivide persistent meshes, or run timed GPU admission
retry loops. At logical capacity pressure a frame may retire its own oldest
accepted receipt; it does not schedule another context's native work.

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

The frontend's `Error::is_recoverable_rejection()` positively classifies input,
support, and allocation limits that permit continued executor use. It applies
only to `SubmitError::Rejected`, never to completion or failed-prefix errors.
Unknown transport/device errors are not assumed recoverable. Even a recoverable
rejection does not certify earlier work's retirement: frame integrations must
retire their accepted prefix, discard the incomplete image, and encode a new
frame before presentation. An unchanged oversized stream must not be retried as
if it were transient Busy.

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
5. Move SGFX's native packets onto the new path, with one bounded logical admission
   and backend-owned dispatch per stream. Retain possibly accepted work on transport/adoption
   failure. Adapt SWS/UI buffer retirement so they do not immediately wait after
   every submission or signal frame release too early.

## Implementation checkpoints

- [x] Core receipt/observation/error traits and deterministic contract tests.
- [x] WGPU tracked submission, bounded tracking, failure receipts and host facade.
- [x] Additive Scarlet async ABI and authoritative completion objects.
- [x] VirtIO/VirGL asynchronous enqueue/completion with retained resources.
- [ ] A618 asynchronous enqueue/completion and safe staging reuse.
- [x] Native VirGL SGFX/facade tracked submission and rejection rollback.
- [x] SWS/ScarletUI frame-level observation and shared-image handoff integration.
- [ ] Both architectures' tests/builds, real host rendering, and A618 hardware
  evidence, including multiple outstanding submissions and failure/teardown.

These are checkpoints, not independent substitutes for the agreed end-to-end
goal. No version bump or candidate tag is warranted by these checkpoints alone.

The initial core/host implementation passes 45 portable tests with Scarlet Rust
and upstream Rust, including five real-device completion scenarios on Metal.
Strict host Clippy and Rustdoc passed at that checkpoint, as did std and
backend-free checks for both normal Scarlet targets. Those initial cross-checks
did not certify native tracked/asynchronous submission. Existing tests continue
to cover the retained untracked executor alongside the new path.

Scarlet's [native implementation boundary](https://github.com/petitstrawberry/Scarlet/blob/dev/docs/graphics/gpu-async-submission.md)
now includes the generic ABI **and real VirtIO/VirGL async execution**. The
driver retains up to 16 submissions across the device, publishes payloads with
independent retirement checkpoints, and progresses via a kernel worker with
IRQ notification and timed fallback. Legacy detach/transfer/presentation remains
synchronous and ordered after preceding async work. Faulted, unretired resources
remain bounded and quarantined without a reset/quiescence proof.

Kernel tests pass on both architectures (1,189 RISC-V / 1,160 AArch64), including
atomic admission above the legacy 64 KiB staging bound. At the earlier driver
checkpoint, both full builds passed. The normal-std `gpu-async-smoke` passed check and strict Clippy
on both Scarlet targets. Real AArch64 QEMU release-image runs pass with VirGL
PCI/two CPUs and MMIO/one CPU: exact async clear/readback, ordered checkpoints,
detach, dropped receipts, and completion after closing all owner handles.
These are driver-level checks, not yet native SGFX adapter tests.

The native VirGL adapter and facade now expose real asynchronous receipts.
Normal std checks pass on both Scarlet targets; the AArch64 native test
harness compiles. The former `sgfx-native-completion-smoke` was installed by
Scarlet's experimental bundle; the binary and dedicated image fixture were
removed at the user's request on 2026-09-06. The user had run the six-scenario
diagnostic successfully 15 times, including oversized rejection and initialization
rollback. That
verified the admission repair, but did not cover intermediate draw pixels:
gears, mesh swarm, and normal UI subsequently showed rendering corruption.

The native adapter had reused vertex storage through `RESOURCE_INLINE_WRITE`
after removing per-packet waits. VirGL's inline decoder leaves transfers
unsynchronized (its usage word is not a synchronization flag), so subsequent
uploads could overwrite vertices still consumed by an earlier GPU draw. See
the [inline decoder](https://chromium.googlesource.com/chromiumos/third_party/virglrenderer/+/refs/heads/master/src/vrend_decode.c)
and [buffer upload/copy implementation](https://chromium.googlesource.com/chromiumos/third_party/virglrenderer/+/refs/heads/master/src/vrend_renderer.c).

The local repair writes disjoint ranges of a private upload arena and issues
ordered GPU buffer copies into vertex storage. Each arena retains its native
completion independently of caller receipts and is recycled only after
explicit successful completion; an unobservable/failed arena is never reused.
The dispatcher uses a fixed ring of four 2 MiB arenas per resource cache,
allocated lazily through the documented synchronous first-use resource setup.
Native chunks use disjoint source ranges; the worker reuses an arena only after
its previous native fence completes. Consecutive logical streams rotate their
starting arena. Arbitrarily many chunks can therefore traverse the same bounded
ring within the logical byte budget, without allocating per-mesh staging or
waiting in submit. Staging pressure is internal worker scheduling, not caller Busy.

A seventh diagnostic now checks every strip of 32 differently colored draws,
both scratch-buffer reuse within one admission and persistent-buffer updates
across queued admissions, including dropped receipts and eight reuse rounds.
Its release binary builds. After building the repaired AArch64 release image,
the user confirmed normal operation on 2026-09-06. This closes the reported
rendering regression; the seventh diagnostic's individual runtime results were
not separately reported. Runtime verification remains user-operated.

ScarletUI's tracked frame executor retains bounded receipts and observes the
whole frame at the SWS handoff. Native upload partitioning belongs to SGFX.
SWS observes composition before presentation/release; producer image reuse
still requires the exact, separate SWS release token. A failed frame is not
published or reused. This does not add cross-process fence transfer or make the
entire render loop nonblocking. Renderer tests pass (37); the platform checks
on both normal Scarlet targets and AArch64 legacy std. The user confirmed the
repaired consumer rendering; fault/reset and other hardware remain separate
release evidence requirements.

The later Boxcraft failure exposed the old native-byte limit leaking into logical
admission: its 60,000-vertex update is 2.4 MB before command overhead. The worker
repair is covered by deterministic host dispatch/packet tests and a ScarletUI
test preserving one 60,000-vertex mesh and persistent buffer. The native diagnostic
now uploads a >2 MiB texture without consumer splitting, uploads a 9.6 MB vertex
buffer through arena-ring reuse, and rejects >64 MiB before initialization changes
are accepted. These revised native runtime cases remain user-operated; the earlier
normal-rendering confirmation does not certify this new dispatcher revision.

**A618 still advertises zero async capacity** and retains its explicit legacy
synchronous consumer path. Its staging/fence ownership and mapping retention
remain to be implemented. A raw control or handle-adoption failure can lose
the current receipt; the VirGL adapter reports an unobservable failure rather
than certifying unknown accepted work using only an older successful receipt.
Fault/reset, repeated native teardown and A618 hardware evidence remain open.
