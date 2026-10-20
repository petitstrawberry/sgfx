# Shared IR state recording

Repeated fixed-function and programmable pass settings are validated on every
call, but only state changes are recorded. The reduction happens in sgfx-core,
before the command buffer reaches WGPU, VirGL, Adreno or Maxwell. It applies to
both Rust command storage and the backend ABI word stream; it does not add a
second recording, change the IR schema or depend on a GPU driver.

The cache is local to each pass. Pipeline changes between fixed-function and
programmable graphics remain explicit. Resource references are validated before
checking the cached binding, including generation changes after retirement.
Uniform and viewport comparisons preserve signed-zero bits. Draws, dispatches,
uploads, barriers, and pass boundaries are not removed.

## CPU workload

Run from the SGFX repository:

```sh
cargo run -p sgfx-core --release --features backend-abi --example ir_recording_bench
cargo run -p sgfx-core --release --features backend-abi --example ir_recording_bench -- --abi
```

The workload records 1,200 textured triangle draws, each setting the same
pipeline, vertex buffer, texture, sampler, uniforms and scissor. It warms up
100 frames and measures 4,000 frames. Walking uses the backend's borrowed
command reader. No GPU is opened and no presentation or frame-rate result is
reported.

A paired Apple M3 Pro host sample on 2026-10-20, using Scarlet Rust 71dd0425890d
and Release optimization, measured:

| Storage | Commands before / after | Record before / after (us/frame) | Walk before / after (us/frame) |
| --- | --- | --- | --- |
| Rust commands | 8,402 / 1,208 | 361.732 / 67.450 | 7.068 / 1.081 |
| Backend ABI | 8,402 / 1,208 | 103.632 / 44.988 | 46.381 / 2.836 |

This deliberately exercises repeated state. Workloads changing every binding
will retain those commands and should not be assigned these speedups. These
single paired samples do not establish sustained application FPS or resolve
all SGFX performance issues.

## Validation and integration

The portable core passes 94 tests without backend-abi and 114 with backend-abi.
Seven new tests cover duplicate state, draw preservation, fresh pass state,
changes and failed validation, retired resources, signed zero, compute bindings,
and switches between fixed and programmable pipelines. The WGPU suite passes
44 tests on Metal, including a new readback regression checking two colors,
scissor boundaries and repeated bindings across two passes in both Rust and
ABI command storage. ScarletUI's SGFX
renderer passes 90 tests with the updated local core when run serially. Its
parallel run aborted in the custom Rust panic runtime; that run is not counted
as passing.

The optimized recorder is published at core revision
`77e64a81115d78fab7c63671c2a6c16aac060c05`. The facade, WGPU, VirGL and
Adreno producer/consumer manifests select that shared core source identity.
The core's existing ABI dependency stays on its unchanged stable revision;
no ABI or schema change is required. Adreno's pinned backend revision is
`4ac9d7ec890a178d934324809b8c68c8445650a1`, based on the previous facade
backend pin rather than unrelated platform changes. Its host suites pass
16 code-generation, 41 backend, 19 preparation and 26 submit-validation tests.
These tests do not establish performance on physical Adreno hardware.

ScarletUI and Scarlet image consumers must select the updated frontend and
core revisions together. Their integration and installed-artifact validation
are recorded in Scarlet's `docs/graphics/gpu-async-submission.md`. The C ABI
and existing dynamic driver manifests remain compatible, including Maxwell.
