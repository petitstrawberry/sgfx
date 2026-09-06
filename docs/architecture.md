# SGFX's execution boundary

SGFX connects renderer/API frontends to complete GPU execution backends. This
document maps the direction in [issue #1](https://github.com/petitstrawberry/sgfx/issues/1)
to the current code. It does not approve the remaining [1.0 contract proposals](1.0-contract.md).

## Recording and execution

```text
ScarletUI PaintCommand             future Vulkan frontend
          |                         (Vulkan object/state rules)
          v                                  |
scarlet-ui-renderer-sgfx                      |
          +----------------+-----------------+
                           v
          sgfx-core::ir + backend contracts
                           |
                 selected execution backend
                           |
                GPU / host graphics API
```

ScarletUI is a renderer and can lower directly into SGFX. A future Vulkan
frontend would own Vulkan object semantics, loader/ICD integration, extension
negotiation and WSI. Those concerns do not belong in the common IR. No Vulkan
frontend is implemented by this workspace or required for its current release
subset.

The current IR supports logical resources, fixed fragment programs, render
passes, uploads, copies and draws. Recording state belongs to a command encoder
and its resource table. There is no process-global implicit graphics context.
The [inventory](1.0-api-scope.md#the-implemented-portable-ir) lists the precise
subset and missing extensions, including arbitrary shaders and compute.

## Where responsibilities live

| Layer | Existing implementation | Responsibility |
| --- | --- | --- |
| Renderer frontend | External `scarlet-ui-renderer-sgfx` | Translate paint/scene data into logical SGFX resources and commands; choose frame contents and damage. |
| Canonical IR | [`sgfx-core::ir`](../crates/sgfx-core/src/ir) | Resource identity, immutable definitions, command-local bindings and portable recording validation. |
| Execution boundary | [`sgfx-core::backend`](../crates/sgfx-core/src/backend.rs) | Acceptance, ordering, borrowed-upload lifetime and optional tracked submission/completion. |
| Backend selection facade | [`sgfx`](../crates/sgfx/src/lib.rs) | Choose a compiled backend at the composition root and delegate its operations/errors. It is not another command IR. |
| Complete execution backend | [`sgfx-backend-wgpu`](../crates/sgfx-backend-wgpu/src), [`sgfx-backend-scarlet-virgl`](../crates/sgfx-backend-scarlet-virgl/src), external Adreno backend | Validate supported semantics, allocate/cache physical resources, lower commands, budget native packets, synchronize, submit and report execution faults. |
| Reusable packet helper | [`sgfx-codegen-virgl`](../crates/sgfx-codegen-virgl/src/lib.rs) | Encode VirGL setup bytes. This helper does not own a queue, resource lifetime or completion. |
| Platform and display integration | Scarlet GPU ABI, SWS, host window/surface adapters | Enforce native resource authority and presentation/buffer leases. Display release and GPU completion have separate conditions. |

The facade's existing `Instance`, window, device and mapped-target helpers are
composition APIs. Their presence does not put Vulkan device-enumeration policy
or WSI into `sgfx-core`. Likewise, the public direct VirGL/composition API is an
existing backend-specific surface, not an additional portable frontend contract.
Both surfaces remain exported and included in the compatibility review.

## Native scheduling is backend work

VirGL's tracked path already follows this division. `submit_ir_async` validates
and lowers borrowed commands before logical admission. The context's dispatcher
owns accepted packets, drives transport progress, and retains the queue even
after frontend owners disappear. Its receipt covers every native chunk and the
ordered prefix. The scheduler bounds logical admissions and retained command
bytes; the packet builder handles native request sizes and upload-arena reuse.
First-use resource creation can still synchronize.

ScarletUI's `FrameExecutor` retains logical receipts and observes the frame at
handoff. It does not calculate VirGL packet sizes or run timed transport retries.
Recoverable rejection drains accepted work before discarding the frame; GPU
failure does not authorize image reuse. The [completion contract](completion-contract.md)
defines these ownership and failure rules, including finite admission limits.

The legacy VirGL `execute`/direct path remains synchronous. Adreno still uses
synchronous execution and the facade rejects tracked submission as unsupported.
A618 asynchronous enqueue, completion and staging retention remain release work;
the facade does not manufacture completed receipts for it. Current boundaries
are recorded in the [execution contract](execution-contract.md).

## Extending this boundary

A new operation should have useful common execution semantics and a defined
unsupported/error path. Backend support can vary without changing the recorded
meaning. Frontends should record directly into the common command stream where
possible; backend resource/pipeline caches and necessary owned upload storage
belong below that boundary. Copying a whole second frontend IR at submission is
not a required part of this design.

New GPU support should fit behind the same recorder/executor boundary. Extract
more `sgfx-codegen-*` helpers when packet generation has an actual independent
consumer; a codegen crate alone does not satisfy the execution contract.

## Architecture and Rust compatibility are separate decisions

Being a driver IR does not by itself choose a Rust API stability policy. A
future Vulkan ICD can expose the Vulkan C ABI while linking its Rust components
as implementation details. That does not automatically make this workspace's
currently exported Rust APIs private or permit breaking existing Rust consumers.

Before RC1, the remaining review must choose the supported Rust surface and
its extension rules: the IR/traits, facade, direct backend/codegen APIs, exposed
dependency types, and feature bundles. The inventory describes what exists;
the draft proposes 1.x preservation rules. Neither document records blanket
approval of those proposals. Owned completion and safe rejection are already
agreed and are not reopened by that review.
