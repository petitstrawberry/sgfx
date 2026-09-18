# SGFX's execution boundary

SGFX connects renderer/API frontends to complete GPU execution backends. This
document maps the direction in [issue #1](https://github.com/petitstrawberry/sgfx/issues/1)
to the current code. The user approved the architecture and coordinated Rust
implementation policy on 2026-09-06. Other rendering and lifecycle clauses in
the [1.0 execution contract](1.0-contract.md) still require review.

## Recording and execution

```text
     Application                         ScarletUI
          |                                  |
     Vulkan C ABI                       PaintCommand
  (stable app boundary)                      |
          |                                  |
          v                                  v
     vulkan-sgfx                 scarlet-ui-renderer-sgfx
          |                                  |
          +----------------+-----------------+
                           v
                        SGFX IR
                           |
          +----------------+--------------+--------------+
          v                v              v              v
     WGPU backend    VirGL backend   AGX backend   A6xx backend
          |            + codegen      + codegen     + codegen
          v                |              |              |
     Host GPU API          +--------------+--------------+
          |                               v
          |                     Scarlet GPU ABI / kernel
          v                               v
       Host GPU                       Target GPU
```

This is the intended logical architecture, including the experimental host
Vulkan frontend and future native Vulkan/AGX work. Each build links the
components it uses; the branches are not independent
SGFX plugin-loading interfaces. A Vulkan driver can package:

```text
one ICD/library = vulkan-sgfx + sgfx-core + selected backend + needed codegen
```

ScarletUI is a renderer and can lower directly into SGFX. The experimental
[`vulkan-sgfx`](vulkan-sgfx.md) frontend owns Vulkan object semantics and
loader/ICD integration. Extension negotiation and any future WSI also belong
to that frontend. The initial ICD implements a documented headless subset
through WGPU and does not establish Vulkan conformance or native Scarlet
programmable execution.

The IR supports logical resources, fixed fragment programs, programmable
render/compute pipelines, bind groups, uploads, copies, draws, dispatches and
explicit same-queue resource dependencies. Recording state belongs to a
command encoder and its resource table. There is no process-global implicit
graphics context. The [inventory](1.0-api-scope.md#the-implemented-portable-ir)
distinguishes the WGPU programmable subset from native fixed rendering.

## Where responsibilities live

| Layer | Existing implementation | Responsibility |
| --- | --- | --- |
| Renderer frontend | External `scarlet-ui-renderer-sgfx` | Translate paint/scene data into logical SGFX resources and commands; choose frame contents and damage. |
| API frontend | Experimental [`vulkan-sgfx`](../crates/vulkan-sgfx) | Own Vulkan objects, loader dispatch and API recording rules; record canonical owned SGFX commands for backend execution. |
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
Both surfaces remain exported as coordinated Rust integration interfaces.

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

The legacy VirGL and Adreno `execute`/direct paths remain synchronous. The
Adreno tracked path uses the companion backend's bounded logical dispatcher
and the A618 driver's asynchronous queue. It retains staging and physical
resource owners until accepted work retires, and rejects a driver without
asynchronous capacity explicitly. Build against the matching published source
set selected by `Cargo.lock`.
A618 hardware and fault/reset evidence remain separate verification work.
Current boundaries are recorded in the [execution contract](execution-contract.md).

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

## Approved compatibility boundary

The application-facing graphics boundary is the Vulkan C ABI. SGFX's IR/traits,
facade, direct backend/codegen APIs, exposed Rust dependency types and feature
recipes are internal integration interfaces whose consumers are updated and
rebuilt together. There is no planned independently loaded SGFX Rust ABI, and
no blanket 1.x source freeze of these interfaces.

An internal signature change requires migrating affected components in the
same compatible revision set. It does not require changing the application's
Vulkan ABI. Direct SGFX Rust users participate in that source-level integration;
ScarletUI can continue its direct renderer path under the same arrangement.
The [change policy](1.0-contract.md#2-coordinated-rust-implementation-policy)
supersedes the earlier all-exports freeze and mandatory enum migration proposal.

The shared execution semantics remain the contract: resource identity and
ownership, ordering, bounded acceptance, completion, and safe failure handling.
Backend portability depends on implementing those semantics consistently.
