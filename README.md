# SGFX

SGFX is a portable graphics execution IR and driver abstraction. Graphics
frontends record backend-neutral resources and commands, while complete
backends materialize resources, lower commands, and submit work to a GPU or
host graphics API.

The current implementation was migrated from the Scarlet repository without
changing its public API or execution behavior. Its longer-term architecture is
tracked in [issue #1](https://github.com/petitstrawberry/sgfx/issues/1).

The [1.0 contract decisions](docs/1.0-contract.md) define the target API
extension rules, rendering and execution semantics, imported-image retirement,
and runtime feature policy. Implementation conformance is still required.
The [execution contract](docs/execution-contract.md) distinguishes today's
queue acceptance, GPU completion, upload lifetimes, and presentation.
The [1.0 API inventory](docs/1.0-api-scope.md) lists the actual IR subset,
public backend/facade surfaces, feature composition, and unfinished review.

## Workspace

- `sgfx-core`: backend-neutral resource descriptions, command IR, and backend contracts
- `sgfx`: compatibility facade and platform backend selection
- `sgfx-backend-wgpu`: host WGPU execution backend
- `sgfx-backend-scarlet-virgl`: Scarlet VirGL execution backend
- `sgfx-codegen-virgl`: platform-neutral VirGL command encoding helpers

`scarlet-ui-renderer-sgfx` remains in the ScarletUI repository because it is a
frontend. Scarlet GPU ABI crates such as `gpu-raw` remain in Scarlet.

## Dependency

```toml
[dependencies]
sgfx = { git = "https://github.com/petitstrawberry/sgfx" }
```

The resolved revisions are recorded in `Cargo.lock`; the manifests do not pin
SGFX or Adreno to a `rev`. Consumers do not need a compatibility patch for
SGFX's former Scarlet source.

## Development

This workspace overrides its own Git `sgfx-core` source with `crates/sgfx-core`
so the external Adreno backend uses the same IR types as the local crates.
This is a development-only override: Git consumers resolve the core from the
same repository naturally and do not inherit or need the workspace patch.

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
