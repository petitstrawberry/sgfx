# Dynamic Scarlet backends

On native 64-bit Scarlet, VirGL and Maxwell load at runtime through `scarlet-ld`.
The existing `backend-scarlet-virgl` and `backend-scarlet-maxwell` features select
installed `libsgfx_scarlet_virgl.so` and `libsgfx_scarlet_maxwell.so` drivers,
including for default features and legacy `no_std` applications. Adreno and host
WGPU retain their existing implementations. ELF32 and Linux-ABI clients retain
their static compatibility transport until their loaders support this ABI.

## Application and installation

Existing applications can keep their `backend-scarlet-virgl` selection. For a
VirGL-only standard-Rust client use `default-features = false,
features = ["std", "backend-scarlet-virgl"]`. Legacy processes use
`legacy-scarlet-std`. The `backend-dynamic` feature also exposes manifest-based
selection of other installed drivers and needs an explicit runtime feature.

For Maxwell-only clients use `default-features = false,
features = ["std", "backend-scarlet-maxwell"]`. Native production clients omit
the static VirGL/Maxwell implementations and their shader compilers; default
features still include compiled Adreno. `backend-scarlet-virgl-static` and
`backend-scarlet-maxwell-static` explicitly select the corresponding static
comparison backend, including in mixed builds with other dynamic drivers.
A standalone static feature does not enable the loader. ELF32 Maxwell retains
its static compatibility backend through a separate wrapper pinned to the
legacy revision; this compatibility path does not include the new standard-Rust
programmable shader compiler. Legacy native64 clients can load the standard-Rust
Maxwell plugin through the C ABI without linking that compiler into their
`no_std` process.

Install the library and this adjacent `scarlet-virgl.sgfx-driver` manifest in
`/system/lib/sgfx`:

```ini
abi=2
name=scarlet-virgl
gpu_backend=virtio-gpu
library=libsgfx_scarlet_virgl.so
```

For Switch GM20B, install `libsgfx_scarlet_maxwell.so` alongside
`scarlet-maxwell.sgfx-driver`:

```ini
abi=2
name=scarlet-maxwell
gpu_backend=nvidia-gm20b
library=libsgfx_scarlet_maxwell.so
```

Automatic GM20B selection and `SGFX_BACKEND=scarlet-maxwell` (or `maxwell`)
request this same manifest. A missing or incompatible Maxwell driver fails
creation; it does not select a static implementation.

`SGFX_DRIVER_PATH` replaces the search directories with a colon-separated list.
`SGFX_BACKEND` selects an installed manifest name. Auto selection matches the
GPU service's backend ID; ambiguity is an explicit error. Names need not be
compiled into the facade. Invalid/incompatible manifests are skipped and
reported if no compatible driver matches. Missing libraries, missing entry
points, and ABI/identity mismatches fail device creation.

Create a mapped-target session before recording commands. It enables canonical
recording on its `ResourceTable`. Alternatively call `enable_abi_commands()`
before creating encoders. Pre-existing native Rust recordings return an
explicit recording-mode error: submission does not secretly convert them.
The usual device/context/session, execute, submit and completion APIs remain
available. Maxwell also exports the optional YCbCr import extension, preserving
shared NV12 sampling conversions. Drivers without that extension, including
VirGL, return `Unsupported` for YCbCr import. The low-level `sgfx::driver`
facade requires the separately negotiated `DriverApi`; an installed driver
without that optional API remains usable by the mapped-session facade.

The facade and in-tree backends share a coordinated `sgfx-core` source pin.
This workspace overrides core and ABI with its own tracked sources for
development. Downstream applications testing these unpublished changes need
an equivalent source override; release builds must advance coordinated source
pins after publication.

Scarlet's desktop bundle installs the plugin and manifest for AArch64 and
RISC-V64. The base bundle supplies `/bin/scarlet-ld`. See Scarlet's
`docs/graphics/sgfx-dynamic-backends.md` for the standard image build and
`tools/sgfx-native-build.py` for reproducible driver installation and executable
linking with the current DSO-safe native toolchain. The installed compiler and
sysroot remain untouched.

`sgfx-probe` prints `linkage: dynamic` and the loaded library path. A missing
plugin is an error, never a silent static fallback. `DT_NEEDED` remains
empty because driver discovery uses `dlopen`; `PT_INTERP`, loader imports and
the probe's runtime path provide independent evidence of the linkage.

## Zero-copy boundary and ownership

The recording itself is a canonical `Vec<u64>`, written once by the encoder.
`submit` passes its existing address, length, table identity and command count.
The library reads borrowed words through a lending reader with one reusable
stack slot for decoded scalar operands. Native Rust recordings are borrowed in
place, and neither path moves the largest command enum on every iteration;
there is no second command vector, payload serialization, or bulk copy at the
library boundary. Upload fields point to the original caller-owned bytes.
Compatibility calls to `CommandBuffer::commands()` can materialize a Rust view;
the dynamic submission path uses `command_reader()` and never calls that API.

Resource descriptions (including shader source) are a separate cold path.
They are serialized into a revision-cached snapshot only when definitions
change. The plugin owns a private mirror; unchanged definitions stay cached.
Generational buffer IDs prevent stale recordings from referring to replacement
buffers. Retirement observes the existing context queue without retaining an
additional per-submit receipt and returns `Busy` if work remains; it never
introduces an implicit GPU wait into submission.

The backend consumes borrowed command/upload storage before `submit` returns,
including rejection and partial failure. The application may then release it.
Existing VirGL resource shadows, GPU staging, native packet ownership and kernel
transport copies still apply. This is **zero-copy across the plugin boundary**,
not a claim that every CPU-to-GPU upload is zero-copy. Native pending work owns
its packets and resources exactly as it does with the static backend.

Objects are opaque handles with library-owned destruction. No Rust `Vec`, `Rc`,
trait object, allocator ownership, enum layout or unwinding crosses the ABI.
Receipts reuse the existing native completion allocation, survive all facade
objects and support observation on another thread. Dropping a receipt neither
waits nor cancels accepted GPU work. Device/context/session calls remain
serialized. Loaded libraries stay resident until process exit so workers and
TLS destructors cannot jump into unloaded code.

Discovery, `dlopen`, symbol lookup and function-table validation happen once.
Steady-state submission uses one cached indirect call for the complete batch;
there is no per-draw FFI call, loader lock or symbol lookup. AArch64 LSE support
is detected by the initialized application runtime and handed to the plugin at
negotiation, since a library's private Rust runtime has no executable auxv.

## ABI v2

The C declaration is [`sgfx_backend.h`](../crates/sgfx-backend-abi/include/sgfx_backend.h);
the matching `no_std` Rust crate has compile-time layout/offset assertions.
Only 64-bit little-endian processes are supported. The original mapped-session entry point is
`sgfx_backend_get_api_v2(version, size, host_info, out_table)` (224-byte table).
The independent `sgfx_backend_get_driver_api_v2(version, size, out_table)`
extension (144-byte table) supports the low-level resource/queue API used by
Vulkan. The initial v1 draft is rejected: v2 carries generations for textures and
bind groups and bounded/flipped blit operands. Low-level operations require
the extension, including explicit texture and bind-group retirement. These are the plugin's only exported symbols.

The low-level extension borrows the same recorded command stream. Readback
writes directly into caller-owned buffers, and cloning a completion retains the
native reference count without allocating a second receipt. No allocator-owned
Rust object crosses the boundary.

Commands use a header `(record_word_count << 32) | opcode`, followed by scalar
words. The count includes the header and is at most 64. The batch count must
agree with the stream; unknown opcodes and malformed lengths are rejected.
Integer/float/flag/enum values are explicit ABI encodings, independent of Rust
layout. Pointers occur only in borrowed upload spans; this is an in-process
protocol, not a disk or IPC format. Buffer references carry slot and generation;
texture and bind-group references also carry slot and generation. Other
immutable resources use slots in the mirrored table.

The v2 schemas are centralized in
[`commands.rs`](../crates/sgfx-core/src/ir/abi/commands.rs),
[`codec.rs`](../crates/sgfx-core/src/ir/abi/codec.rs), and
[`resources.rs`](../crates/sgfx-core/src/ir/abi/resources.rs).
Existing opcodes, enum numbers, field order and meanings are frozen. Changes
that are not compatible with v2 require a new ABI version/entry point, even if
the ordinary Rust APIs are updated together. Trusted plugins must initialize
every output on success and satisfy the memory/lifetime contract; loading a
plugin is not a sandbox boundary.

### Optional YCbCr extension

`sgfx_backend_get_ycbcr_api_v2` negotiates a separate 16-byte function table;
the existing 224-byte `BackendApi` and 144-byte `DriverApi` remain unchanged.
Its conversion record is 24 bytes: `size`, `reserved`, `matrix`, `range`,
`chroma_x`, `chroma_y`, all `u32`. Size must be 24 and reserved must be zero.
Matrix values 1/2 encode BT.601/BT.709, range values 1/2 encode limited/full,
and chroma values 1/2 encode cosited/midpoint. Other values are rejected.
The import call takes a session object, texture slot, transferred Scarlet
handle and this C record by value. The plugin consumes the handle on success
and failure; absent extension support leaves handle cleanup with the facade.
Discovery resolves the optional entry point once and rejects malformed table
negotiation. Rust enum representation never crosses the boundary.

## Reproducible native build and verification

Use Scarlet's Rust toolchain, matching Rust sources with the DSO-safe native
TLS namespace fix and executable-only CRT split, a sibling Scarlet checkout containing `scarlet-ld` and its
loader-smoke/ELF-audit tools, and QEMU with VirGL support. All Rust runtimes in
the interpreter, application and plugin must agree on the thread/TLS layout.
The build script refuses the old per-DSO colliding TLS-key implementation.
Add `--maxwell ../scarlet-project-switch` to also build, stage and ELF-audit the
Switch driver using its checked-in coordinated dependency pins. The smoke scene still runs
on VirtIO/VirGL; staging and ELF audits do not verify GM20B GPU execution.
The compiler also needs the GNU ELF OSABI fix (`petitstrawberry/rust`
commit `71dd0425890` or newer): retained AArch64 constructors can make LLD
emit OSABI 3, which the native compiler must accept before marking its output
as Scarlet OSABI 83. The build script checks that output without rewriting it.

```sh
python3 scripts/build-dynamic-backends.py --arch aarch64 \
  --rust-source /path/to/matching/patched/rust \
  --output target/dynamic-backends/aarch64
python3 scripts/run-dynamic-smoke.py --arch aarch64 \
  --kernel /path/to/scarlet-kernel \
  --staging target/dynamic-backends/aarch64/staging \
  --output target/dynamic-backends/aarch64/qemu
```

Omit `--rust-source` with an updated installed toolchain. `--offline` is
available after dependencies are cached. The script generates an isolated PIC
target/sysroot view, builds `std` and the interpreter consistently with ThinLTO
and one codegen unit (also applied to the static comparison), and never
modifies the installed compiler/sysroot. Rust implementation symbols use hidden
visibility before optimization, leaving the C entry exported; link-time hiding
alone cannot provide LLVM the same locality information. `--arch riscv64` selects the other
native architecture. macOS AArch64 can use `--accel hvf`; TCG is the default.
The test runner creates a disposable image, without changing the normal image.

On macOS the normal `cocoa,gl=on` display opens a window even for readback-only
tests. To retain VirGL without displaying windows or taking focus, build the
isolated Cocoa test variant (the renderer is unchanged):

```sh
nix build --impure --file scripts/qemu-cocoa-offscreen.nix \
  --argstr scarlet "$(cd ../Scarlet && pwd)" \
  --out-link target/dynamic-backends/qemu-offscreen
python3 scripts/run-dynamic-smoke.py --arch aarch64 --accel hvf \
  --qemu target/dynamic-backends/qemu-offscreen/bin/qemu-system-aarch64 \
  --kernel /path/to/scarlet-kernel \
  --staging target/dynamic-backends/aarch64/staging \
  --output target/dynamic-backends/aarch64/qemu
```

This uses Scarlet's pinned QEMU and changes only Cocoa window presentation and
application activation. It does not replace the installed QEMU. `-display none`
is not a substitute for an OpenGL display backend when testing VirGL.

`build.json` records compiler identity, target specification, commands, runtime source hash, artifact
sizes/hashes and ELF audits. VirGL exports its two C entry points; Maxwell also exports
`sgfx_backend_get_ycbcr_api_v2`. All other dynamic exports are rejected;
the application must import exactly `dlopen`, `dlsym`, `dlerror` from the
interpreter and have **no `DT_NEEDED` backend dependency**. Unsupported ELF TLS,
RELR, symbol versions and text relocations fail the audit. Each guest result
records QEMU and initramfs hashes and retains `input-build.json` when its
artifact hashes match the supplied staging tree.

The smoke fixture runs identical static and dynamic scenes in one guest and
checks triangle pixels, a 4 MiB upload, synchronous/asynchronous execution,
completion and object lifetimes, low-level queues, shared images, readback and
buffer retirement. When staged, the standard probe and legacy fixture also
verify dynamic loading and rejection of a missing driver. Both modes execute as fresh sibling processes
in four rounds, alternating their order. Each records medians over 80 samples
after 30 warmups for recording, submission, their combined elapsed time
(`total_ns`), and completion of 1-draw and 200-draw frames. `frame_ns` starts
before recording and ends after completion; older results without `total_ns`
started their frame timer at submission. AArch64 reads the architectural counter without a clock syscall per
sample; the clock source and measurement overhead are reported. These measure
elapsed time including preemption, not thread CPU time. `result.json` preserves
each round and computes paired dynamic/static ratios. Guest
correctness `PASS` is distinct from a performance guarantee: compare the
reported timings, retain `result.json`/`serial.log`, and account for host/QEMU
scheduling noise. Additional hardware and application workloads remain needed
to establish the broader performance impact of the native 64-bit default.

An additional AArch64 HVF diagnostic alternated static, static-with-canonical-IR,
and dynamic sessions **per frame in the same process**, with 30 warmups and
160 samples per mode. The decoder measured before the default migration produced these medians
(microseconds):

| Draws | Static record + submit | Dynamic record + submit | Static through completion | Dynamic through completion |
| --- | ---: | ---: | ---: | ---: |
| 1 | 87.083 | 103.041 | 2624.750 | 2651.833 |
| 200 | 355.083 | 353.833 | 2714.791 | 2699.208 |

For this run, the 200-draw batch was approximately equal in CPU-side elapsed
work; the small batch added 15.958 microseconds (18.3%). Through-completion
elapsed time differed by +1.03% and -0.57%, respectively. The small apparent
improvement is not evidence of a speedup. Earlier same-process diagnostics
showed small-batch increases of 10.8–15.5 microseconds (10.7–27.0%) and
200-draw increases of 0.8–3.0%. These are measurements of a small synthetic
workload under HVF, not a bound on application overhead or real hardware.
The pre-migration production fixture also ran four pairs of fresh sibling processes,
alternating their order. Its median paired dynamic/static ratios were:

| Draws | Record + submit ratio | Through-completion ratio | Median paired extra record + submit |
| --- | ---: | ---: | ---: |
| 1 | 1.509 | 1.031 | 26.855 us |
| 200 | 1.109 | 1.011 | 24.021 us |

Separate processes had substantially more variation than the per-frame
interleaved diagnostic (even static submission times varied by over 2x).
Neither result should be discarded: the same-process diagnostic isolates the
boundary more closely, while the separate production binaries include effects
of their different code/runtime layouts. These results support considering the
modularity tradeoff for batched rendering, but not a claim that overhead is
universally 1% or less. Many small submissions still warrant application-level
measurement.

The earlier comparison of 20 versus 72 microseconds used asymmetric process
lifetimes and a syscall-based timer. Repeated symmetric comparisons did not
reproduce a fixed 3.6x submission penalty, so that result must not be presented
as the cost of dynamic linking. Nor do the corrected comparisons prove exact
performance parity. Native 64-bit VirGL now defaults to the dynamic path;
`backend-scarlet-virgl-static` retains an explicit comparison build. ELF32 and
Linux-ABI clients keep the static compatibility transport.

The original implementation reported local evidence under `target/dynamic-backends/aarch64-runtime/`:
`qemu-scalar-interleaved/result.json` and `serial.log` contain the table above;
`qemu-interleaved/` and `qemu-hidden-interleaved/` contain the earlier diagnostic
results. The diagnostic source and exact build command are retained alongside
those directories as `interleaved_probe.final-source.rs` and
`interleaved-probe-command.json`; the disposable diagnostic binary deliberately
contains both backends, unlike the normal dynamic-only fixture. The final
pre-migration production-fixture check is in `qemu-final/`. Correctness checks include actual
VirGL triangle and upload readbacks, not only successful loading. RISC-V64
artifacts pass the same ELF audit; GPU execution was tested on AArch64 only.

The original implementation reported its default migration AArch64 smoke result in
`target/dynamic-backends/migration-aarch64-final/qemu/result.json`. It also
checks the standard probe, legacy client and low-level driver extension.
The standard-build SWS/ScarletUI/Vulkan integration result is in the sibling
Scarlet checkout at `target/sgfx-migration-desktop/result.json`: the GPU
compositor ran, the textured cube passed pixel readback, and 120 shared Vulkan
frames were presented. Standard-build executable audits and bundle assembly
evidence are retained in `target/sgfx-userspace/migration-elf-audit.json` and
`target/sgfx-bundle-policy/elf-check.json` there. These checks do not assert
application performance parity.

The dynamic-only AArch64 fixture is approximately 0.37 MB versus 2.45 MB for
static VirGL, with a separate approximately 2.47 MB driver. Exact sizes and
hashes are recorded in `build.json`. These are file sizes, not process RSS:
the current loader copies ELF segments into private anonymous mappings, so
this does not yet establish shared physical code pages across processes.

Portable CI checks both native and canonical recording semantics, malformed
records, metadata revisions/generations, discovery failures and ABI layout.
The boundary test counts allocations and checks original storage/payload
pointer identity over repeated imports and iteration. Native CI checks both
Scarlet architectures; actual `.so` linking/loading uses the script above.
