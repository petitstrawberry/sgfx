# SGFX

SGFX is a portable graphics execution IR and driver abstraction. Graphics
frontends record backend-neutral resources and commands, while complete
backends materialize resources, lower commands, and submit work to a GPU or
host graphics API.

Start with the [SGFX reference](docs/reference.md) for dependencies, backend
selection, device/window setup, execution and completion. The
[IR syntax reference](docs/ir-reference.md) documents descriptors, every command
and its arguments, recording rules, and an upload-and-draw example.

SGFX was initially migrated from the Scarlet repository without changing its
public API or execution behavior. The [architecture](docs/architecture.md)
maps [issue #1](https://github.com/petitstrawberry/sgfx/issues/1) onto the current
IR, execution backends and platform adapters, including native tracked submission.

The [1.0 execution contract](docs/1.0-contract.md) records the approved
architecture and coordinated Rust update policy, together with the remaining
rendering and lifecycle review targets. The agreed [completion scope](docs/completion-contract.md)
includes portable submission tracking and actual asynchronous Scarlet GPU
submission; implementation conformance is still required.
The [execution contract](docs/execution-contract.md) distinguishes today's
queue acceptance, GPU completion, upload lifetimes, and presentation.
The [1.0 API inventory](docs/1.0-api-scope.md) lists the actual IR subset,
Rust backend/facade interfaces, feature composition, and unfinished review.

## Workspace

- `sgfx-core`: backend-neutral resource descriptions, command IR, and backend contracts
- `sgfx`: execution facade and platform backend selection
- `sgfx-backend-wgpu`: host WGPU execution backend
- `sgfx-backend-scarlet-virgl`: Scarlet VirGL execution backend
- `sgfx-codegen-virgl`: platform-neutral VirGL command encoding helpers
- `vulkan-sgfx`: experimental Vulkan ICD using the programmable IR, with
  offscreen execution and macOS Metal WSI

The IR includes shader modules, programmable render/compute pipelines, resource
bind groups, buffer copies and explicit same-queue resource dependencies. The
WGPU executes graphics and compute from this subset. Scarlet VirGL executes the
bounded programmable graphics subset and rejects compute; Adreno currently
rejects programmable execution. See the [IR reference](docs/ir-reference.md) and
[Vulkan frontend scope](docs/vulkan-sgfx.md) for the supported operations.

`scarlet-ui-renderer-sgfx` remains in the ScarletUI repository because it is a
frontend. Scarlet GPU ABI crates such as `gpu-raw` remain in Scarlet.

## Compatibility and linking

All SGFX crates above, including their Rust `pub` interfaces and the external
Adreno backend/codegen, are components of a coordinated driver/renderer build.
Their Rust APIs can evolve with the frontend and backend consumers; SGFX 1.x
does not promise unchanged Rust signatures across releases. Direct Rust users
must select a compatible revision/lockfile set and rebuild its components.

The intended application-facing graphics boundary is the Vulkan C ABI. The
experimental `vulkan-sgfx` frontend depends only on the SGFX execution facade;
the product build links that facade, core, and its selected complete backend
into one ICD/library. The host build uses WGPU and the Scarlet QEMU build uses
VirGL. Independently loading SGFX Rust components is not planned. Native
Scarlet dynamic-loader support and Vulkan conformance remain future work.
Resource ownership, ordering, completion and failure semantics still form the
common backend contract. See the [approved policy](docs/1.0-contract.md#2-coordinated-rust-implementation-policy).

## Dependency

```toml
[dependencies]
sgfx = { git = "https://github.com/petitstrawberry/sgfx" }
```

The workspace pins coordinated Git dependencies such as Scarlet, Adreno, and
the shared `sgfx-core` type source to exact commits. Downstream applications
should also retain their resolved revisions in `Cargo.lock`. No workspace
source patch is required.

See [dependencies and locked revisions](docs/reference.md#dependencies-and-locked-revisions)
for the native source set, shared Rust type identity and workspace patch policy.

## Development

All components that exchange SGFX IR use the same exact `sgfx-core` Git commit.
Use the Scarlet and Adreno revisions declared in the manifests and selected in
`Cargo.lock`. Update the manifest `rev` and lockfile together when selecting a
new compatible source set, then run the locked target checks below.

CI runs the portable suite on both `scarlet-rust-toolchain` (the integration
baseline) and upstream Rust (the portability check). The two compilers use
separate target directories. Scarlet target checks always use the Scarlet
toolchain; `rust-toolchain.toml` selects the additional upstream check.

With the Scarlet toolchain on `PATH`, run the integration baseline with:

```bash
export CARGO_TARGET_DIR=target/scarlet-host
cargo-fmt fmt --all --check
cargo test --locked -p sgfx-core
cargo test --locked -p sgfx-codegen-virgl
cargo test --locked -p sgfx-backend-wgpu
cargo test --locked -p sgfx
cargo test --locked -p vulkan-sgfx
```

For the additional upstream check, use a separate target directory:

```bash
CARGO_TARGET_DIR=target/upstream-host rustup run nightly-2025-12-31 cargo test --locked -p sgfx-core -p sgfx-codegen-virgl -p sgfx-backend-wgpu -p sgfx
```

Scarlet provides `std` for both of its userspace targets. With the Scarlet
Rust toolchain active, check the facade and both compiled backends with:

```bash
cargo check --locked -p sgfx --target riscv64gc-unknown-scarlet
cargo check --locked -p sgfx --target aarch64-unknown-scarlet
```

The `legacy-scarlet-std` feature is retained only for the older no_std
userspace target and is not the normal Scarlet configuration.
