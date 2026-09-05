# SGFX execution contract

This records the existing execution boundary while preparing 1.0. It does not
declare the whole IR stable or introduce a new graphics API. Frontends record
portable commands; the backend owns validation of its supported subset,
resource materialization, lowering, transport limits, and submission.
The [1.0 API inventory](1.0-api-scope.md) records the current exported surface
and feature configurations. The [1.0 contract decisions](1.0-contract.md) are
the authoritative release target; differences from today's behavior require
conformance work before RC1.

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
acceptance guarantee. SGFX does not currently expose a portable completion
token, wait operation, or cross-queue synchronization API.

| Backend | Current success boundary | Observation / presentation |
| --- | --- | --- |
| WGPU | Uploads are copied into WGPU-owned staging and the encoded commands are queued. GPU execution remains asynchronous. | Same-queue operations retain ordering. Host surface presentation is separate; CPU readback must wait for the appropriate copy/map completion. |
| Scarlet VirGL | Ordered backend operations use the synchronous Scarlet GPU ABI. Opaque submissions return after fenced backend completion; buffer-only updates may remain in the persistent shadow cache until needed. | Session BGRA readback is synchronous. Sharing an image with SWS still follows the SWS frame lifecycle. |
| Scarlet Adreno | Ordered chunks use the synchronous Scarlet GPU ABI; earlier chunks are drained before direct CPU-visible uploads. | Session BGRA readback is synchronous. SWS presentation and buffer reuse remain separate from command execution. |

The Scarlet ABI boundary is `GpuQueue::submit` in `gpu-raw`; this is not a
reason to make WGPU block or to equate a Scarlet userspace target with no_std.
Both normal Scarlet targets support Rust std.

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
with a rendering approximation. For example, WGPU currently rejects uploads
after the first copy or render pass, while Scarlet backends own their own
ordered upload and transport-chunking rules.

WGPU buffer upload offsets and byte lengths must be multiples of four. The
backend reports `Unsupported(BufferWriteAlignment)` for other valid IR byte
ranges, and `Unsupported(ResourceSize)` for buffers or texture dimensions
above the selected device's limits. These checks precede raw WGPU allocation
or upload validation; they do not narrow the portable IR. Real-device tests
check both the returned error and the absence of a raw WGPU validation error,
then verify valid operations still work after rejection.

Execution is not a transaction. VirGL preflights its command plan, but later
backend failures can follow completed uploads or earlier passes. Adreno can
submit a prefix before encountering a later unsupported operation. WGPU can
report validation or device failures asynchronously after queue acceptance.
Neither an `Err` nor the absence of an immediate error proves that no work ran.
Do not blindly replay a failed command buffer. Backend-specific recovery must
decide whether state can be reused or the context and mappings must be rebuilt.

## Selected 1.0 decisions and conformance boundary

The [1.0 decisions](1.0-contract.md) retain `CommandExecutor` without new
required methods, portable completion tokens, capability queries or a common
device-loss recovery API. Backend-specific observation/recovery remains
explicit; absent a documented recovery boundary, callers must not replay
failed work or assume externally shared storage is safe to reuse.

Successful explicit native imported-image release must detach the mapping and
finish that session's outstanding accesses to the image before returning.
Failure does not promise rollback or permission to rebind/recycle. `Drop` is
memory-safe cleanup, not an observable fence. SWS release additionally depends
on the exact frame identity and all consumer uses, independently of unmapping.

This is the target guarantee, not a claim that native failure/teardown paths
have passed review. Audit those paths and rendering/backend-subset behavior
against the decided contract before RC1. No Vulkan frontend or mandatory
common synchronization API is added to the release scope.
