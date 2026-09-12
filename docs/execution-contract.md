# SGFX execution contract

This records the existing execution boundary while preparing 1.0. It does not
declare the whole IR stable or introduce a new graphics API. Frontends record
portable commands; the backend owns validation of its supported subset,
resource materialization, lowering, transport limits, and submission.
The [1.0 API inventory](1.0-api-scope.md) records the current exported surface
and feature configurations. The [1.0 execution contract](1.0-contract.md) records
the approved coordinated Rust policy and remaining semantic review targets;
the user-approved [completion scope](completion-contract.md) adds
tracked submission and actual asynchronous Scarlet GPU execution as 1.0 gates.

The common Rust boundary is
[`CommandExecutor`](../crates/sgfx-core/src/backend.rs). The `sgfx` facade
delegates to the selected backend without strengthening its guarantees.

## Successful execution is not portable GPU completion

`execute()` returning `Ok(())` means the complete logical command stream was
accepted with its recorded ordering. Sequential successful executions on the
same bound queue preserve resource-access ordering. It does not mean that a
frame is visible, that the CPU may inspect GPU-written memory, or that another
queue or external image consumer can immediately use the results.

An empty stream is a no-op, not a portable wait for earlier work. A backend may
complete work synchronously; portable callers must rely only on the common
acceptance guarantee. The additional `CommandSubmitter` and `Completion`
interfaces now expose owned receipts, nonblocking observation and timed waits
on host WGPU, native VirGL and the matching asynchronous Adreno source set.
A driver without Adreno asynchronous capacity is explicitly unsupported.
There is no common cross-queue GPU semaphore API.

| Backend | Current success boundary | Observation / presentation |
| --- | --- | --- |
| WGPU | Uploads are copied into WGPU-owned staging and the encoded commands are queued. GPU execution remains asynchronous. | Same-queue operations retain ordering. Host surface presentation is separate; CPU readback must wait for the appropriate copy/map completion. |
| Scarlet VirGL tracked `submit` | One logical stream is accepted into the context-owned dispatcher. Its worker schedules native chunks using the asynchronous Scarlet GPU ABI. First-use resource creation can still synchronize. | The owned receipt covers all chunks and their ordered prefix. Observation is not required to drive dispatch. Readback and explicit detach drain preceding logical work. SWS leases remain separate. |
| Scarlet VirGL legacy `execute` / direct submit | Ordered backend operations retain the synchronous Scarlet GPU ABI path after draining preceding tracked work. Opaque submissions return after fenced backend completion; buffer-only updates may remain in the persistent shadow cache until needed. | Session BGRA readback is synchronous. Sharing an image with SWS still follows the SWS frame lifecycle. |
| Scarlet Adreno tracked `submit` | A bounded logical dispatcher owns commands, staging and physical resource references. It schedules native chunks through the A618 asynchronous queue, resuming after native capacity pressure without replaying accepted work. | The receipt covers every chunk and its ordered prefix. Legacy synchronous execution, readback and explicit release drain preceding work. A618 hardware verification remains outstanding; SWS leases remain separate. |

The Scarlet ABI retains synchronous `GpuQueue::submit` and adds tracked
asynchronous submission with authoritative completion observation in `gpu-raw`.
The [architecture](architecture.md) describes the division between IR,
backend scheduling and platform ownership. Both normal Scarlet targets support
Rust std; that runtime choice does not select synchronous execution.

The Adreno path requires the corresponding backend and A618 driver changes.
The [native integration script](../scripts/check-native-integration.sh) checks
the two userspace architectures against a companion checkout without replacing
the release lockfile. A successful cross-build does not establish A618 hardware
retirement, fault recovery or presentation conformance.

WGPU records uploads as encoder-owned staging copies, so uploads interleaved
with render/compute passes and buffer copies preserve command order. Shader
write dependencies use the core's explicit resource barriers; ordinary ordered
accesses and physical transitions are handled by WGPU. VirGL and Adreno reject
the programmable commands they cannot lower before native submission.

VirGL admits up to 16 logical streams and 64 MiB of retained command bytes.
Native packet splitting, FIFO progress and upload-arena reuse belong to the
backend. A receipt remains pending until every chunk and its ordered prefix
retire. Dropping a frontend owner or receipt does not cancel accepted work or
stop the worker. See the [completion contract](completion-contract.md) for
the admission and failure rules.

WGPU `CommandSubmitter::submit` returns an owned receipt. A private marker
written at the end of the submission identifies retirement even if a later
raw-queue submit races callback registration. The backend bounds tracked
callback slots, returns `Busy` when full, and retains the marker independently
of receipt ownership. These slots do not bound external raw WGPU or legacy
untracked submissions. `Device::new` installs the loss callback used by
observation; replacing it or independently wrapping an alias of that raw device
is incompatible with tracked observation.

Failed lowering returns a conservative failed-prefix checkpoint, not a false
rollback guarantee. Completion reports device loss separately from retirement.
Finite host waits drive nonblocking WGPU progress with short bounded sleeps;
indefinite native-host waits use the submission index. Browser blocking waits
are explicitly unsupported; browser support is not certified by native tests.

## Ownership and lifetime

Logical references belong to a particular `ResourceTable`, not merely to a
matching numeric slot. Backend caches and physical-image mappings must use
the originating table and compatible device/context. Equal descriptors or
slot numbers from another table do not make resources interchangeable.

Command buffers borrow their logical table and upload slices. The caller
keeps these borrows valid until execution returns and the command buffer is
dropped. After that, upload storage can be overwritten or freed: pending GPU
work must use backend-owned data, not retained pointers into those slices.
Backends retain the physical objects required by accepted work until they can
be released safely. Dropping a Rust cache or command buffer is not a portable
GPU wait, and logical resource definitions are not physical allocations.

Imported images have an additional producer/consumer lifetime. A by-value
handle import transfers that owned reference; successful import must retain
backend ownership for the mapping and accepted work. Keeping the producer's
content lease is a separate obligation under the frame-release protocol.
Releasing a borrowed view, finishing command recording, submitting GPU work,
and receiving a compositor release are different events. In particular, a
successful `execute()` does not authorize immediate reuse of an image still
owned by SWS for a pending frame.

## Validation, supported subsets, and errors

Core recording rejects a depth format in the color-target slot and requires
a matching depth attachment when a bound pipeline enables depth testing.
Disabling depth testing does not require removing the pass's depth attachment
from the logical IR; a backend may impose additional restrictions.

Rejected recording operations do not append commands or replace previously
accepted bindings. The command limit reserves room to end an open pass, and
abandoning a pass without `end()` leaves `finish()` invalid. These invariants,
table-qualified identities, byte-range checks, and borrowed uploads are tested
without a GPU in [`sgfx-core`'s validation suite](../crates/sgfx-core/tests/validation.rs).

The core checks logical descriptors and recording rules. Each backend also
validates its representable subset and context mappings. Unsupported commands
must return an explicit error; they must not be silently skipped or replaced
with a rendering approximation. For example, the native backends reject the
programmable shader/compute subset. WGPU supports ordered uploads between
passes; Scarlet backends own their upload and transport-chunking rules.

WGPU buffer upload offsets and byte lengths must be multiples of four. The
backend reports `Unsupported(BufferWriteAlignment)` for other valid IR byte
ranges, and `Unsupported(ResourceSize)` for buffers or texture dimensions
above the selected device's limits. These checks precede raw WGPU allocation
or upload validation; they do not narrow the portable IR. Real-device tests
check both the returned error and the absence of a raw WGPU validation error,
then verify valid operations still work after rejection.

Execution is not a transaction. Native tracked submission preflights its
logical command plan, but later backend failures can follow accepted uploads
or earlier passes. WGPU can
report validation or device failures asynchronously after queue acceptance.
Neither an `Err` nor the absence of an immediate error proves that no work ran.
Do not blindly replay a failed command buffer. Backend-specific recovery must
decide whether state can be reused or the context and mappings must be rebuilt.

Tracked submission distinguishes `Busy` and `Rejected` (the current logical
stream was not accepted) from `Failed` (possible accepted work with a receipt).
Neither rejection rolls back earlier submissions in the same frame. The facade's
`Error::is_recoverable_rejection()` classifies only an error inside `Rejected`;
it must not downgrade `Failed` or completion errors. ScarletUI retires an accepted
prefix before discarding a recoverably rejected frame and preserving its prior
presentation. An oversized unchanged stream is not a transient Busy condition.

## Approved completion scope and remaining conformance

The [completion contract](completion-contract.md) adds `CommandSubmitter`
without adding required methods to existing `CommandExecutor` implementations.
Portable receipt observation and actual asynchronous Scarlet submission are
required for 1.0. Failure recovery remains explicit; absent a documented recovery
boundary, callers must not replay failed work or assume external storage is
safe to reuse. No common capability-query or device-recovery API is added here.

Successful explicit native imported-image release must detach the mapping and
finish that session's outstanding accesses to the image before returning.
Failure does not promise rollback or permission to rebind/recycle. `Drop` is
memory-safe cleanup, not an observable fence. SWS release additionally depends
on the exact frame identity and all consumer uses, independently of unmapping.

This is the target guarantee, not a claim that native failure/teardown paths
have passed review. Audit those paths and rendering/backend-subset behavior
against the agreed completion scope and reviewed lifecycle rules before RC1.
The experimental Vulkan frontend is documented separately; its presence does
not certify a conformant Vulkan implementation or expand the approved release
subset automatically.
