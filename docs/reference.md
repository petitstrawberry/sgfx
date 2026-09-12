# SGFX reference

This reference describes the SGFX 1.0.0 Rust integration interfaces. Start with
the [IR syntax reference](ir-reference.md) for resource declarations, command
arguments and a complete upload/draw recording example.

SGFX connects renderer frontends to complete execution backends. It is not a
window toolkit, a Vulkan implementation or an independently loadable Rust
driver ABI. Components using these Rust interfaces are updated and rebuilt
together; see the [compatibility policy](1.0-contract.md#2-coordinated-rust-implementation-policy)
and [architecture](architecture.md).

- [Crates and entry points](#crates-and-entry-points)
- [Dependencies and locked revisions](#dependencies-and-locked-revisions)
- [Features and backend selection](#features-and-backend-selection)
- [Platform setup](#platform-setup)
- [Execution and completion](#execution-and-completion)
- [Images, presentation and reuse](#images-presentation-and-reuse)
- [Backend-specific interfaces](#backend-specific-interfaces)
- [Generated API documentation](#generated-api-documentation)

## Crates and entry points

| Crate | Use it for | Source |
| --- | --- | --- |
| `sgfx-core` | Backend-neutral `ir` types and `backend` traits; `no_std` plus `alloc`. | [Core exports](../crates/sgfx-core/src/lib.rs) |
| `sgfx` | `ir`/`backend` reexports, backend selection and target-specific device/window/session helpers. | [Facade exports](../crates/sgfx/src/lib.rs) |
| `sgfx-backend-wgpu` | Execution on a host WGPU device, including headless integration and window presentation. | [WGPU API](../crates/sgfx-backend-wgpu/src/lib.rs) |
| `sgfx-backend-scarlet-virgl` | Scarlet GPU device/context integration, VirGL lowering and submission. | [VirGL API](../crates/sgfx-backend-scarlet-virgl/src/lib.rs) |
| `sgfx-codegen-virgl` | VirGL setup-command encoding; not resource allocation or execution. | [Codegen API](../crates/sgfx-codegen-virgl/src/lib.rs) |

The Adreno backend/codegen belong to `scarlet-project-chromebook`, and
`scarlet-ui-renderer-sgfx` belongs to ScarletUI. Neither is a crate maintained
inside this workspace.

## Dependencies and locked revisions

For the facade, including the default backends for the build target:

```toml
[dependencies]
sgfx = { git = "https://github.com/petitstrawberry/sgfx" }
```

A renderer that only records IR can depend on the core without a device backend:

```toml
[dependencies]
sgfx-core = { git = "https://github.com/petitstrawberry/sgfx", default-features = false }
```

The package version alone does not select a compatible transitive source set.
These versions describe this release checkout: a fresh resolution of the bare
Git URL follows its default branch, not necessarily the 1.0.0 source set.
Commit the consuming application/workspace's `Cargo.lock` and use `--locked`
for reproducible builds. Cargo uses that root lockfile, not the lockfiles of its
Git dependencies; a Scarlet bundle's `scarlet.lock` does not replace it.
Generate dependency changes with Cargo, not by editing lockfile entries.
Updating one Git dependency can also re-resolve its transitive dependencies.

Keep one `sgfx-core` source identity in each binary. Use the same Git URL and
selector throughout its dependency graph: mixing a bare Git source with a
`branch`, `tag` or `rev` selector can create separate copies of Rust types even
when the selected commits are equal. In particular, the external Adreno backend
imports the core from the bare SGFX Git URL. Do not add a selector to only one
edge of that graph.

The source identities used by the current native integration are:

| Components | Manifest source | Package versions |
| --- | --- | --- |
| SGFX facade, core, WGPU, VirGL and VirGL codegen | `https://github.com/petitstrawberry/sgfx`, no selector | `1.0.0` |
| Scarlet native runtime/GPU crates | `https://github.com/petitstrawberry/Scarlet`, `branch = "dev"` | `1.0.0` in the selected lock |
| Adreno backend/codegen | `https://github.com/petitstrawberry/scarlet-project-chromebook.git`, no selector | `0.1.0` |

The SGFX workspace's [Cargo.lock](../Cargo.lock) selects Scarlet revision
`d8a199815249c784a04c9dec60b6efedbb79d84e` and Adreno revision
`debc765f80ac7a582150b397bc8dd9cb52b4be89` for this source set. These are lockfile
resolutions, not changes to the manifest selectors above. The Chromebook
project and its userspace drivers do not need to share SGFX's version number.

Release inputs must not require sibling checkouts or developer-local path
patches. The patch in this repository's [workspace manifest](../Cargo.toml)
is different: it points to its own tracked `crates/sgfx-core` so the external
Adreno dependency and local workspace crates share one core. Git consumers do
not inherit that patch and do not need to copy it. Repository-contained path
dependencies and tracked vendored sources remain valid.

## Features and backend selection

See the [facade manifest](../crates/sgfx/Cargo.toml) for the feature definitions.

| Build configuration | Available facade |
| --- | --- |
| Defaults on a host target | Rust std and WGPU window/session execution. |
| Defaults on `riscv64gc-unknown-scarlet` or `aarch64-unknown-scarlet` | Rust std and compiled native VirGL/Adreno backends; the opened GPU determines automatic selection. |
| `default-features = false`, no features | IR/trait reexports and selection types, but no executor or usable backend. |
| No defaults, `backend-wgpu` | Host WGPU integration without SGFX environment lookup. WGPU itself still requires std. |
| No defaults, `std` | The normal runtime bundle, including all three `backend-*` features, subject to target gating. This is not a backend-free std configuration. |
| No defaults, `legacy-scarlet-std` | Older Scarlet userspace runtime configuration; `scarlet-std` is its compatibility alias. Not the normal std targets. |

Use `Instance::new()` to read `SGFX_BACKEND`, or
`Instance::with_preference(BackendPreference::...)` to choose explicitly.

| `SGFX_BACKEND` | Preference |
| --- | --- |
| Unset or `auto` | Automatic selection among compiled backends. |
| `wgpu` | Host WGPU. WGPU using Metal is still `wgpu`. |
| `scarlet-virgl` or `virgl` | Scarlet VirGL. |
| `scarlet-adreno` or `adreno` | Scarlet Adreno. |
| `metal` | Reserved direct Metal backend; currently unavailable. |

Parsing is case-sensitive and does not trim whitespace. An invalid name returns
`InvalidBackendPreference`; an unavailable explicit backend returns
`BackendUnavailable`. There is no silent fallback for an explicit choice.
Without a runtime supporting environment lookup, the preference is `Auto`.
On Scarlet, `Instance::backend()` under `Auto` is only the compiled default;
`Device::backend()` reports the backend selected after opening the GPU.

## Platform setup

Define logical resources through `ResourceTable` first. A presentation target
needs `TextureUsage::RENDER_ATTACHMENT | TextureUsage::PRESENT`; use
`Bgra8Unorm` for the native VirGL mapped-target path. Keep its `TextureId` and
an `Rc<ResourceTable>`. Descriptor syntax is in the [IR reference](ir-reference.md).

Both platform paths return a `MappedTargetSession` owning the queue, physical
resource cache and mapped target images. Pass `Rc::clone(&resources)` and the
target IDs to `create_mapped_target_session`. Record commands against that same
table, and use `session.executor()` to bind execution to the session.

### Host window

The [host facade](../crates/sgfx/src/host.rs) exposes:

| Operation | API |
| --- | --- |
| Create the surface/context | `unsafe Instance::create_window_context(display_handle, window_handle, width, height)` |
| Require compositor transparency | `unsafe Instance::create_window_context_with_transparency(display_handle, window_handle, width, height, transparent)` |
| Materialize logical targets | `WindowContext::create_mapped_target_session(resources, &targets)` |
| Reconfigure the surface | `WindowContext::resize(width, height)` |
| Queue a mapped image for presentation | `WindowContext::present(&session, target)` |

Handles are `raw-window-handle` 0.6 raw display/window handles. The unsafe
constructors require both handles to remain valid until the context is dropped:
drop the SGFX context before destroying the native window/display. Width and
height are physical pixels. Resize reconfigures the surface, not the immutable
logical target definitions; create and map new targets when their size changes.

### Scarlet GPU

The [Scarlet facade](../crates/sgfx/src/scarlet.rs) exposes:

| Operation | API |
| --- | --- |
| Open a GPU using the instance preference | `Instance::open_device(path)` |
| Open a GPU using environment/default selection | `Device::open(path)` |
| Inspect selected backend and coarse support | `Device::backend()`, `Device::capabilities()` |
| Create an execution context | `Device::create_context()` |
| Materialize logical targets | `Context::create_mapped_target_session(resources, &targets)` |
| Borrow a mapped output image | `MappedTargetSession::image(target)` |

The device path is supplied by the platform/application. The capability
booleans cover rendering, presentation, upload, readback and depth; they do not
certify every format, operation or combination. Scarlet presentation is handled
by the platform/SWS integration, not by a portable IR command.

## Execution and completion

Import the traits to call their methods:

```rust
use sgfx::backend::{CommandExecutor, CommandSubmitter, Completion};
```

| Method | Return / meaning |
| --- | --- |
| `executor.execute(&commands)` | `Result<(), Error>`: the whole logical stream was accepted, not necessarily completed. Empty execution is a no-op. |
| `executor.submit(&commands)` | `Result<Submission, SubmitError<Error, Submission>>`: tracked acceptance with an owned receipt. An empty submission establishes a queue checkpoint. |
| `receipt.poll()` | `Result<CompletionStatus, Error>`: nonblocking observation; also drives backend observation progress. |
| `receipt.wait(Some(duration))` | `Complete`, or `Pending` on timeout, or an error. Zero duration polls. |
| `receipt.wait(None)` | Wait without a caller deadline, where supported. |

`CompletionStatus` has `Pending` and `Complete` and is non-exhaustive. Timeout
does not cancel work. A receipt covers its logical submission and earlier
ordered work on that queue, not later submissions or other queues. `Complete`
is neither presentation acknowledgement nor CPU mapping/cache synchronization.

The submit result distinguishes these cases:

| Result | What the caller must retain or do |
| --- | --- |
| `Ok(receipt)` | Observe the receipt when completion is needed. |
| `SubmitError::Busy` | No work from this call was accepted. Make capacity available before retrying. |
| `SubmitError::Rejected(error)` | No work from this call was accepted, but earlier work is not rolled back or certified healthy. Correct the cause or recover the backend as appropriate. |
| `SubmitError::Failed { error, completion }` | A prefix may have been accepted. Preserve the attached receipt; do not blindly replay the stream or recycle shared images. |

`SubmitError` is also non-exhaustive. `SubmitError::map` converts error/receipt
types without losing acceptance information. The facade's
`Error::is_recoverable_rejection()` applies only to `Rejected`, never to a
failed-prefix or completion error. Device faults are errors, not completion.

On **every** return path from execute/submit, pending GPU work no longer borrows
the caller's uploads or commands. Upload memory may be reused after its Rust
borrows end. Receipts do not borrow the executor/table/upload data; dropping
one does not wait or cancel work. Backend retention of in-flight resources is
independent of receipt ownership.

WGPU, Scarlet VirGL and the matching Adreno source set implement tracked
submission. Native backends may synchronize during first-use physical resource
creation; accepted uploads/copies/draws use asynchronous dispatch. Adreno
requires an A618 driver advertising asynchronous capacity and retains its
separate synchronous execution path. Use the
[native integration check](../scripts/check-native-integration.sh) with a
companion checkout. Hardware verification remains separate from these build
interfaces. See [execution details](execution-contract.md)
and the [completion interface](../crates/sgfx-core/src/backend/completion.rs).

## Images, presentation and reuse

On Scarlet, `ImageRef::width()`, `height()` and `shared_handle()` expose a
borrowed view of a mapped output. Platform integration owns how that image is
handed to SWS and when its producer may render into it again.

`MappedTargetSession::import_shared_bgra_texture(texture, handle)` takes an
owned `Handle` by value for a logical sampled texture. That ownership does not
replace the producer's content lease. Use
`release_imported_texture(texture)` for explicit session-local retirement;
successful release must finish that session's outstanding uses before allowing
the mapping to be reused. Failure and `Drop` do not grant reuse permission, and
release is not an acknowledgement from every external consumer.

`readback_bgra(target, destination, destination_stride, rect)` is a synchronous
native readback when supported. The stride is in bytes; the source rectangle
is written at the same coordinates in the destination, not packed at its
origin. Keep sufficient destination storage for those coordinates.

Host `present` queues a surface frame; it does not prove display on a monitor.
Neither successful submission nor GPU completion replaces the SWS buffer-release
protocol. For ownership and failure rules, see the
[execution contract](execution-contract.md#ownership-and-lifetime).

## Backend-specific interfaces

For headless WGPU or an existing WGPU device, use
`sgfx_backend_wgpu::Device::new(raw_device, raw_queue)`, then create a context,
resources and queue. Its `raw` reexport exposes the matched WGPU version (24).
Do not replace the device-loss callback installed by the wrapper or separately
wrap aliases of the same raw device when using tracked observation.
The lower-level `Queue::submit` is untracked; use `Queue::submit_tracked` or
the executor's `CommandSubmitter::submit` when a receipt is required.

The native VirGL crate also exports direct rendering/composition APIs and
`Queue::submit_ir` / `submit_ir_async`. These are backend-specific interfaces,
not additional portable IR instructions. Packet splitting, dispatch budgets
and upload retention belong to the backend, not to renderer frontends.

## Generated API documentation

For exact signatures and method-level documentation from a matched checkout:

```sh
cargo doc --locked --no-deps -p sgfx -p sgfx-core -p sgfx-backend-wgpu -p sgfx-codegen-virgl
```

For native target-specific exports, use the Scarlet Rust toolchain and target:

```sh
cargo doc --locked --no-deps --target aarch64-unknown-scarlet -p sgfx -p sgfx-core -p sgfx-backend-scarlet-virgl
```

The normal Cargo output is `target/doc/sgfx/index.html`, or
`target/aarch64-unknown-scarlet/doc/sgfx/index.html` for the native command
(under `CARGO_TARGET_DIR` when overridden). Host documentation does not contain
the target-gated Scarlet facade. No generated documentation is required in Git.
