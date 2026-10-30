# Experimental Vulkan frontend

`vulkan-sgfx` is a loader-discoverable, **non-conformant development ICD**. It
implements a bounded executable subset to exercise Vulkan → SGFX portable IR →
backend execution, macOS presentation through Metal, and Linux ABI presentation
through Scarlet SWS with `VK_KHR_display` and `VK_KHR_wayland_surface`. The native
Scarlet VirGL adapter exposes core 1.1 aliases and, when native compute,
draw parameters, divisors, anisotropy and D32 transfers are available, the core
1.2 entry-point profile. Aggregate 1.1/1.2 feature/property queries agree with
their extension aliases, including the enforced 256 MiB allocation limit.
Other adapters continue to report Vulkan 1.0. These API versions are not
conformance claims: substantial mandatory core functionality is absent. Do not
select it as a general application driver. Cross-platform WSI, resource
reclamation with long-lived objects, broader API coverage and conformance work
remain incomplete.

## Implemented path

- Push descriptors and push descriptor templates retain immutable command-owned
  snapshots. Partial updates preserve compatible prior bindings, including
  replay after caller memory changes. Dynamic buffer descriptors are rejected
  for push layouts; one push set and up to 16 descriptors are supported.
- Capable native VirGL adapters support linear sampler anisotropy up to 16,
  nonzero per-instance attribute divisors, and shader base-instance,
  base-vertex and draw-index builtins. Indirect draw indices preserve skipped
  zero-count entries. Native vertex fetch also accepts signed normalized byte
  vectors, signed 16-bit pairs and 32-bit integer vectors. These features are
  negotiated separately and stay disabled
  on backends without the corresponding execution path.
- Point-list assembly, Min/Max color and alpha blending, explicit sampling
  gradients, fragment depth output, angle conversions and dynamic local-array
  stores execute through the normal IR/compiler paths. SPIR-V normalization
  retains external buffer offsets while repairing internal return-value struct
  layouts affected by Naga's point-size IO serialization.
- RGBA16F and RGBA32F sampled color targets, GPU clears, additive blending and
  raw packed transfers are enabled when the host format masks support both.
  Fragment output arrays become distinct attachment locations before validation.
  Depth-only dynamic rendering retains a private discarded color attachment and
  an empty fragment stage when the application supplies only a vertex stage.
  Fixed finite constant/slope depth bias reaches the native rasterizer.
- Scalar R32 signed/unsigned/float texel views use the original buffer and range
  as read-only storage. `texelFetch` preserves component bits and supplies the
  format's default zero/zero/one channels. Descriptor templates for texel views,
  other texel formats and texel-buffer size queries are not implemented.
- Native D32 images support checked staging uploads, image copies, readback and
  depth clears, with mip zero and one layer. The CGL host needs the depth-aware
  framebuffer copy path. GPU timestamps use the host's real GPU query result;
  the Metal bridge retires GL work before sampling a Metal GPU completion marker.
  A failed clock probe leaves timestamps disabled.

- An actual C ABI `cdylib`, ICD interface negotiation (versions 2–5), scoped
  procedure lookup, instance/physical-device/device/queue/command-buffer
  dispatchable handles with the loader's required first word, and checked
  object ownership. Only the three `vk_icd*` entrypoints are exported as unmangled
  symbols; ordinary Vulkan commands are returned through procedure lookup.
- SGFX discovers actual adapters and reports one `VkPhysicalDevice` for each.
  `vkCreateDevice` opens the selected adapter, rather than probing or choosing a
  backend again inside the Vulkan frontend. One graphics/compute/transfer queue
  is exposed according to that adapter's SGFX capabilities. Optional core
  physical-device features are not advertised.
- SPIR-V shader modules, compute pipelines, and vertex/fragment pipelines.
  Pipeline creation resolves specialization constants and uses Naga to validate
  and normalize Vulkan coordinate conventions before backend compilation.
  Scalar 32-bit specialization expressions are frozen before Naga parsing;
  unsupported expression operations return an error. Combined image samplers
  can pass through shader helper functions, including comparison sampling.
- Stage-specific push constants up to 128 bytes on capable Metal and native
  programmable VirGL adapters, with incremental updates and a value snapshot
  per draw/dispatch. VirGL stores its stage snapshot after UBO constants.
- Sampled RGBA8/BGRA8 mip chains on capable Metal adapters, checked per-mip
  uploads/barriers/readback, complete matching-format GPU mip blits with
  nearest/linear filtering, and sampler mip filtering/LOD clamps. Native VirGL
  uploads/samples color mip chains and emits full-mip color blits when the kernel
  reports real mip storage. Dynamic color attachments may select a nonzero mip:
  private GPU render targets and subresource blits preserve LOAD/STORE, partial
  render areas and other mip levels. The scratch targets are reused by format,
  size and attachment slot. Native readback selects mip zero; copy other levels
  to a single-level image before reading them. Older
  kernels and A618 reject unsupported mip storage and blits.
- Metal supports R8 uploads, layered textures, cube and array sampling, mutable
  UNORM/sRGB views, component swizzles, and sampled D32 depth with comparison
  samplers. Swizzles and packed vertex decoding execute in shaders and retain
  the original texture/buffer allocations. Write-only RGBA8 storage images can
  select one 2D layer and mip; image barriers order writes before later sampling.
- Descriptor sets for uniform/storage buffers (including dynamic offsets),
  separate images/samplers, combined image samplers, single-sample input
  attachments and write-only storage images;
  four sets with 16 logical bindings each, one descriptor per binding. Updates
  can span consecutive compatible bindings. Descriptor arrays are unsupported.
- Host-visible coherent memory with stable, aligned mappings; buffer memory
  binding, host upload/readback, and nonoverlapping allocations. Buffers have
  four-byte sizes; binding offsets use the reported alignment. Uniform ranges
  are limited to 16 KiB and storage ranges to 128 MiB.
- Primary command buffers and pools, pipeline/set binding, compute dispatch,
  render passes, non-indexed and indexed drawing, vertex buffers,
  buffer copies, checked buffer-to-image uploads, whole-image RGBA8/BGRA8
  mip readback, whole-resource/per-mip barriers, dynamic viewport/scissor,
  D32 depth testing, front-face/back-face culling, fences, and binary semaphores.
- Metal supports multiple color attachments with per-target write masks and
  blend state, read-only depth testing while sampling depth, and up to eight
  ordered subpasses with `vkCmdNextSubpass`. A render pass has at most eight
  attachments; its simultaneous color limit follows the selected backend.
  Each subpass becomes an SGFX GPU pass. Intermediate outputs are stored until
  their last use, and input attachments become same-pixel GPU texture loads.
  This path has no CPU image readback. Inputs currently require nonarray,
  single-sample float data and zero offsets. Feedback loops, self-dependencies,
  depth-only subpasses and resolves are unsupported. Native VirGL rejects MRT,
  input attachments and read-only depth until its backend implements them.
- Offscreen graphics: RGBA8_UNORM or BGRA8_UNORM color, optional D32_SFLOAT
  depth, optimal 2D images, one layer/sample, at most 2048×2048,
  triangle lists and nonindexed triangle strips on Metal, signed
  base vertices for indexed lists, up to eight vertex bindings, integer/half-float
  and packed signed-normal inputs on Metal, and instanced draws with a first
  instance index. Vertex attributes currently advance per vertex. Common alpha/additive blending and sampled 2D textures
  work. Attachments retain one mip; sampled images can contain a checked chain.
  Native VirGL also supports indexed strips without primitive restart. Indexed
  strips on WGPU, primitive restart, stencil, multisampling,
  partial/flipped/format-converting blits,
  and other dynamic state are absent.
  Native VirGL retains its existing single-stream, single-instance subset;
  unsupported typed views and new vertex formats are rejected explicitly.
- macOS WSI exposes `VK_KHR_surface`, `VK_EXT_metal_surface`, `VK_MVK_macos_surface`,
  `VK_KHR_portability_enumeration`, and `VK_KHR_swapchain`. It implements Metal
  surface creation, surface capability/format/mode queries, FIFO swapchains,
  acquire, and queue present. Swapchain images are SGFX `PRESENT` textures on
  the exact WGPU/Metal device selected by `vkCreateDevice`.
- Linux/musl builds with `scarlet-wsi` expose `VK_KHR_surface`, `VK_KHR_display`,
  and `VK_KHR_swapchain` through the unmodified Khronos loader. Standard display
  discovery/mode/plane procedures create a fullscreen SWS window, and registered
  GPU swapchain images are returned only on actual compositor release. The
  ordinary C `examples/display.c` checks 60 presentations and clean shutdown
  on Scarlet AArch64 QEMU. This does not yet establish vkQuake2 game rendering.
- Linux Wayland WSI targets the application's existing surface, including Wine
  GPU subsurfaces. A private `wp_scarlet_sgfx_v1` registration imports BGRA8
  native GPU images into SWS surface scenes. Private libwayland proxy wrappers
  and event queues preserve the application's listeners. FIFO waits for frame
  callbacks, reuse waits for buffer release, and queue work completes before
  compositor sampling. Replacement swapchains may use a different extent.
- `vkCmdClearColorImage` clears single-mip/layer RGBA8/BGRA8 UNORM transfer
  destinations using GPU render-pass clear operations, with complete range
  validation before emission. Other formats and subresource clears are rejected.
- Linux advertises `VK_KHR_get_physical_device_properties2` and
  `VK_KHR_external_memory_capabilities` for Wine's adapter queries. KHR queries
  wrap the supported core properties; device/driver UUIDs are stable and LUID
  validity is false. External buffer/image queries expose no supported handle
  type. This does not add external Vulkan memory export or import.

Each `vkCmd*` records into command-buffer-local owned storage on the calling
thread. Submit resolves the Vulkan handles and descriptor state into
`sgfx-core::ir::OwnedCommand`, then reconstructs validated borrowed IR for the
backend. This is a transitional representation: the local Vulkan command enum,
submit-time stream reconstruction, and append-only SGFX resource table still
need to be replaced by a reusable canonical owned program and generational
resource identities. Image-to-buffer copies currently split SGFX submissions
and use backend image readback followed by an ordered buffer write.

Resolved recordings are consumed when building the final submission stream.
Texture and buffer upload vectors move into that stream without a second
payload allocation/copy; application-owned staging memory is still snapshotted
during resolution. Descriptor buffer references are collected and sorted once
before deduplication, avoiding a growing linear search for each reference.
Validation, positioned descriptor/barrier/readback insertions, and completion
ownership are unchanged.

All `Rc`-backed core/backend state stays on one dedicated device worker. Vulkan
callers record commands without a worker round trip; resource creation,
submit-time resolution, and backend queue access use the worker. No `unsafe
Send` assertion or forged resource lifetime crosses threads.

## Completion and memory

Ordinary `vkQueueSubmit` returns after SGFX accepts the work. One completion
observer thread per Vulkan device waits on owned SGFX receipts, signals the
fence, and retires in-flight work. GPU acceptance and retirement errors do not
manufacture successful completion; uncertain backend failures mark the device
lost.

Texture-transfer-only recordings above 8 MiB are validated in full, then admitted
as ordered batches with at most 8 MiB of source pixels each. Large individual
uploads are divided at row boundaries, preserving subrectangles, mip levels,
array layers and overwrite order. Texture barriers, copies and mip blits retain
their position between uploads. Every batch receipt participates in fence
completion and readback waits. A failure after an accepted prefix drains its
receipts and marks the device lost; the prefix is never replayed. Mixed
render/transfer recordings retain their existing whole-recording path.

Commands that require the current CPU-shadow path remain synchronous. In
particular, image-to-buffer copy waits for image completion and performs backend
readback, and GPU-written mapped buffers are read back before submit returns.
This is a known transfer/memory-model limitation, rather than the normal path
for draw-only submissions.

Fence and binary-semaphore state use shared atomics outside the
device worker. They therefore remain observable while another thread is
blocked submitting work. A deterministic test covers an intentionally
unserviced worker channel. `vkQueueWaitIdle`/`vkDeviceWaitIdle` serialize behind
prior queue work. Acquire signals a semaphore or fence, queue submission consumes
wait semaphores and signals completion semaphores, and queue present consumes
its waits. `VK_KHR_timeline_semaphore` supports host signals, counter queries,
waits and queue values; queue signals become visible only after GPU retirement.

`VkPhysicalDeviceVulkan12Features` queries report the same timeline semaphore,
host query reset, indirect-count drawing and imageless framebuffer capabilities
as their individual feature records. Device creation accepts those aggregate
requests and routes timeline/query enablement to the existing execution paths;
unsupported aggregate fields are cleared on query and rejected on enablement.
This query/enablement support does not change the advertised Vulkan 1.1 API.

`VK_KHR_synchronization2` adds explicitly enabled `synchronization2` features,
64-bit stage/access scopes, `vkCmdPipelineBarrier2KHR`, `vkQueueSubmit2KHR` and
device event set/wait/reset commands. Split transfer/vertex-input/shader scopes,
`NONE`, attachment/read-only layout aliases, mixed binary/timeline submissions
and a single-device mask of zero or one are supported. Resource barriers use
the existing backend path; synchronization scopes conservatively retire prior
GPU work. This costs overlap and does not raise the advertised Vulkan API
version. Barriers and event commands remain outside render passes in this
bounded profile. Timestamp writes use the same ordered control points, including
pass splitting that preserves attachment contents.

`VK_KHR_dynamic_rendering` supports explicitly enabled single-layer, single-sample
rendering with 1–8 non-null color attachments (up to the backend limit) and
optional D32 depth. Pipelines retain `VkPipelineRenderingCreateInfo` formats
without application render-pass or framebuffer objects. Attachment loads,
stores and render areas use the existing render-pass execution path. Resolves,
stencil, multiview, null attachment slots, suspend/resume flags and depth-only
rendering remain unsupported. Read-only depth follows backend capability checks.

Indirect and indexed-indirect draws support multiple commands, GPU-written
argument buffers, signed vertex offsets and nonzero first-instance values.
`VK_KHR_draw_indirect_count` reads a GPU-written counter and clamps it to
`maxDrawCount`; the reported command limit is 4096. This portable implementation
retires preceding GPU work, reads the parameters, preserves/resumes attachments
and emits direct IR draws. It adds synchronization and readback overhead. Native
VirGL accepts transfer/update producers and retired compute-buffer producers in
separate submissions when the native compute capability probe succeeds.

Native VirGL's optional compute-buffer profile lowers WGSL/SPIR-V to TGSI `COMP`,
binds shader buffers and issues `LAUNCH_GRID` followed by a shader-buffer/update
memory barrier and fenced GPU readback. It supports fixed workgroups with at
most 128 invocations (`128,128,64` per-axis limits), `65535` groups per axis,
invocation/workgroup IDs, four storage bindings of up to 256 KiB, 16 KiB uniforms
and 128-byte push constants. Buffer allocations are bounded to 1 MiB; a stream
retains at most 64 MiB of buffer snapshots. Descriptor aliases share one physical
allocation, and range checks suppress invalid writes and return zero on invalid
reads. GPU writes update CPU shadows and invalidate cached buffer revisions, so
subsequent graphics submissions upload the new contents.

Compute-only streams can contain uploads, buffer copies, multiple compute passes
and buffer barriers. They execute synchronously even through the asynchronous
submission API; the returned receipt is already retired. Mixed graphics/compute
streams, shared workgroup memory, atomics, texture bindings and storage images
remain unsupported and are rejected before execution. If a dispatch may have
been accepted before a failure, submission reports a partial failure and makes
the resource cache unusable rather than allowing replay.

The `COMPUTE_BUFFERS` ABI flag and compute queue limits are advertised only after
an isolated dispatch modifies eight SSBO words and GPU-to-host readback verifies
every result. Failed probes leave the rendering context untouched. Plain macOS
CGL/OpenGL 4.1 has no GL compute support, so a renderer without a compute bridge
keeps this path disabled. The Scarlet virglrenderer fork's optional Metal compute
backend executes these same VirGL commands on Metal and synchronizes buffer bytes
with GL. Shared-memory limits remain zero for this SGFX profile.

Local validation of this profile passed five WGSL/SPIR-V compute semantic tests
(3D invocation IDs, read-modify-write, descriptor offsets, robust bounds and
arithmetic overflow, distinct UBO/push/storage constants, and explicit profile
rejection), plus parsing both generated compute shaders with the pinned
virglrenderer TGSI parser. On Apple M3 Pro, the native OpenJDK 25/HVF fixture now
passes two real SSBO read-modify-write dispatches, descriptor offset/range and
eight sentinel-word checks, followed by indirect-count rendering, 128 pixel
checks, visibility queries and synchronization2. Metal API validation is enabled
for the packaged host renderer. This validates actual guest-to-host GPU compute;
the same fixture also verifies disabled compute on a plain CGL renderer.

The host renderer is
[`petitstrawberry/virglrenderer` at cbe9e52f55a72e625da4b8ad5aba906b681fc46a](https://github.com/petitstrawberry/virglrenderer/commit/cbe9e52f55a72e625da4b8ad5aba906b681fc46a),
built with `-Dmetal-compute=enabled`. Its direct public-API test additionally
checks aliased SSBO bindings at slots 3/7, 3D workgroups/global IDs and 64 output
words, rejection of unbound storage and fence completion. The bridge uses
synchronous CPU copies between GL and Metal; it does not add images, shared
memory or mixed-stream execution to SGFX. The validated host library SHA-256 is
`f422834b493da917587cd5178d94c7371e08cc11ad6f2bdffbe9c113d88328da`.

The native ICD SHA-256 for that check is `d7a0f818219fd79b7a52eb98115e660b269a9c1d6ffb9a72346fcb00071416d4`;
the VirGL plugin SHA-256 is `52d98a21c98cce1bc26dd6e3bf9b8c18406b31d3edeeaafbcffc7467e70ef7c4`. These hashes identify
that validation build, independently of the earlier query/rendering builds below.


Timestamp query pools provide reset, 32/64-bit results, availability, WAIT,
strided buffer copies and optional `VK_EXT_host_query_reset`. Only real GPU
clocks enable `timestampValidBits=64` with a nanosecond period. An optional SGFX
v2 timing table preserves existing backend ABI layouts. Metal samples its GPU
clock after retiring prior commands; other WGPU backends use timestamp queries.
VirGL uses timestamp query objects and CPU-visible query-result buffers, requiring
Scarlet kernel `b229d6d007fbdf8e3b9a130c3feeaf94ee8019e9` or later. Older kernels,
missing timing tables and unusable host timers leave timestamp valid bits at zero.
Host query reset remains available when another query type is implemented.

Occlusion pools support `vkCmdBeginQuery`/`vkCmdEndQuery`, availability, reset,
32/64-bit reads and strided copies. WGPU and native VirGL currently expose only
non-precise results: zero means no sample passed, and nonzero means visible.
`occlusionQueryPrecise` remains false; `PRECISE` is rejected. Query scopes survive
pass splits used by indirect draws, timestamp writes and attachment preservation.
Nested queries of the same type, mismatched ends, resetting/copying active slots,
open queries at command-buffer end and invalid render-pass/subpass boundaries
are rejected during recording.

Pipeline-statistics pools are enabled only when the backend supplies statistics.
Results contain requested counters in increasing Vulkan bit order, followed by
availability when requested. Native VirGL uses one host query per selected
counter. Its optional protocols are probed on isolated contexts so rejecting a
statistics object cannot poison the application's render context; such hosts
can log the rejected probe. WGPU uses hardware VS/clipper/fragment/compute
counters and exact submitted IA vertex/primitive counts; GS/tessellation counters
are zero because those shader stages are not exposed. Queries aggregate all
contributing passes, excluding backend-generated clear draws. The optional
scoped-query ABI table leaves existing v2 table layouts unchanged. Metal does
not expose WGPU pipeline-statistics queries, and this macOS OpenGL host rejects
the native statistics protocol, so `pipelineStatisticsQuery` remains false here.
`PARTIAL` results and indexed/secondary-command-buffer query variants are outside
this bounded profile.

Vulkan 1.0 descriptor update rules apply: updating a set already bound in a
recording or executable command buffer invalidates that buffer. Reset and
re-record before submitting it. No update-after-bind feature is exposed.
One-time command buffers become invalid after successful submission.

Unsupported commands with a void Vulkan return type record the first failure;
`vkEndCommandBuffer` rejects the recording. Unsupported creation options and
features return errors. Unknown procedure names return null. Missing mandatory
core procedures are a known conformance gap, not successful placeholder stubs.

## Save a rendered image

From the repository root, run:

```sh
scripts/run-vulkan-demo.sh target/vulkan-demo.png
```

This builds a fresh release host ICD and `render_demo`, writes its absolute-path driver
manifest, finds an installed Vulkan loader, and renders a 1024×768 procedural
RGB triangle with a grid and orbital details. The fragment shader produces the
image; after real Vulkan submission, fence completion, image-to-buffer copy,
and memory mapping, the example saves the unchanged RGBA8 readback as a PNG.
The existing small `offscreen` smoke and the image demo share the same Vulkan
resource/command/readback implementation. The demo checks opaque output,
hundreds of distinct colors, and both bright and dark regions before saving.

The script requires a repository-compatible Rust toolchain (`cargo` and
`rustc`), Python 3, and an installed Vulkan loader. It searches ordinary system,
Homebrew, Vulkan SDK, and installed Nix loader paths without installing anything.
Use the platform dynamic-library search path when the loader is installed in a
nonstandard location.
The script selects the Rust host target explicitly so a Scarlet cross-target
configuration does not affect the demo. On Linux it selects Mesa llvmpipe with
surfaceless EGL to avoid GL-to-Vulkan recursion through Zink.

The console identifies the selected physical device, readback checks, and
absolute PNG path. The PNG encoder performs lossless encoding only; no image
generator, CPU renderer, or composited visual overlay is involved. `render_demo`
is an ordinary Vulkan application and contains no SGFX-specific loader or ICD
selection. The runner selects the freshly built ICD externally with the standard
`VK_DRIVER_FILES` loader setting.

## Present a Vulkan cube on macOS

From the repository root, run:

```sh
scripts/run-vulkan-windowed.sh
```

Use `--frames 120` for a finite automated run. The runner builds the release ICD and
the `windowed` example, writes an absolute ICD manifest, finds an installed
Khronos loader, sets only the standard loader and dynamic-library search
variables, and starts the application. The application itself calls
`ash::Entry::load`, enables the platform extensions returned by `ash-window`,
creates a `VK_EXT_metal_surface`, enables `VK_KHR_swapchain`, and uses a normal
acquire → submit → present loop with binary semaphores and a fence.

The presented scene is a rotating indexed cube using SPIR-V vertex and fragment
shaders, a uniform transform, vertex/index buffers, and D32 depth testing. The
application has no dependency on `vulkan_sgfx`, no alternate Vulkan-shaped API,
and no direct ICD entry-point lookup. At runtime the Khronos loader discovers
`libvulkan_sgfx.dylib` from `VK_DRIVER_FILES`; the ICD then executes through
`vulkan-sgfx → sgfx → sgfx-backend-wgpu → WGPU Metal`.

The runtime dependency chains are:

```text
render_demo / windowed (ordinary Vulkan applications using ash)
  → Khronos libvulkan loader
  → libvulkan_sgfx ICD, selected by its standard JSON manifest
  → vulkan-sgfx frontend and Vulkan object state
  → sgfx driver facade and canonical SGFX IR
  → sgfx-backend-wgpu → WGPU → Metal

vulkan-cube (current Scarlet executable test path)
  → vulkan-sgfx frontend linked into the test executable
  → sgfx driver facade and the same canonical SGFX IR
  → sgfx-backend-scarlet-virgl → scarlet-os /dev/gpu0
  → VirtIO-GPU / VirGL → host renderer
```

The native Scarlet toolchain supports `cdylib` output. The checked Vulkan path
still links the frontend into the application; native ICD discovery and
installation remain separate work. Its SGFX facade loads VirGL through the
native backend plugin described in [dynamic backends](dynamic-backends.md).

## Run the examples

From the repository root, with an installed Khronos loader:

```sh
cargo build --locked --release -p vulkan-sgfx --lib --examples
python3 crates/vulkan-sgfx/tools/write_icd_manifest.py \
  target/release/libvulkan_sgfx.dylib target/sgfx_icd.json
export VK_DRIVER_FILES="$PWD/target/sgfx_icd.json"
# Use the normal platform lookup path when the installed loader is elsewhere:
export DYLD_LIBRARY_PATH=/path/to/installed/vulkan-loader/lib
cargo run --locked --release -p vulkan-sgfx --example headless -- 16
cargo run --locked --release -p vulkan-sgfx --example headless -- 16 --push-constants
cargo test --locked -p vulkan-sgfx --lib
SGFX_ICD_LIBRARY="$PWD/target/release/libvulkan_sgfx.dylib" \
  cargo test --locked -p vulkan-sgfx --test contracts -- --ignored
```

Use `libvulkan_sgfx.so` and `LD_LIBRARY_PATH` on Linux. The `headless` example
uses `ash::Entry::load()` exclusively. It checks 256 compute output words and
reports CPU recording and submission durations. Its push-constant mode uses two
dispatches with distinct incremental values, including an update before pipeline
binding, and changes the values across submissions.

The separate `offscreen`, `loader_probe`, and defensive GPU contract diagnostics
can load the ICD directly. `SGFX_ICD_LIBRARY` selects the ICD library in those
test harnesses. `offscreen` checks a 64-square triangle. The two older examples
also accept `SGFX_VULKAN_LOADER` to open a specified real loader with `libloading`;
that setting is not read by the ICD or ordinary applications. `headless`,
`render_demo`, `windowed`, `shader_probe` and host `vulkan-cube` use the installed
Vulkan loader's standard entry point and external `VK_DRIVER_FILES` selection.
GPU contract tests are ignored by default for CPU-only CI; explicitly run them
against a built ICD on a machine with a supported GPU backend.

The host backend mask is fixed: Metal on macOS, GL on other host platforms.
The ICD never enables WGPU's Vulkan backend or reads WGPU backend-selection
configuration. Execution is verified on macOS Metal and Linux aarch64 with Mesa
26.1.3 llvmpipe (LLVM 21.1.8), surfaceless EGL 1.5/OpenGL ES 3.2, and the real
Khronos Vulkan Loader 1.4.341. The macOS check passed ordinary loader discovery,
a Metal surface and three-image FIFO swapchain, and 60 presented frames of the
SPIR-V indexed cube. The Linux check passed loader discovery and device
creation, 16 compute iterations checking all 256 output words, and a 64×64
offscreen triangle checking 112 center, corner, and asymmetric orientation
pixels. Linux GL must use a native GL driver or Mesa llvmpipe: Mesa Zink
implements GL through Vulkan and could recursively enter this ICD even with
WGPU's GL-only mask. For a Linux
software smoke environment, set `GALLIUM_DRIVER=llvmpipe` and
`LIBGL_ALWAYS_SOFTWARE=true` as described by the
[Mesa environment-variable documentation](https://docs.mesa3d.org/envvars.html#gallium-driver),
and verify the chosen Mesa driver. The Linux check also used
`EGL_PLATFORM=surfaceless`; GL availability and compute limits depend on the host.

## Limits and remaining work

- **Not Vulkan conformant.** The ICD implements a bounded executable subset.
  WSI currently covers macOS Metal and Scarlet Linux display/Wayland,
  FIFO mode and opaque composition. Windows presentation is through Wine's
  Win32-to-Wayland driver and Box64's native Vulkan wrapper. Direct3D/DXVK,
  general games, other platform surfaces, secondary command buffers, descriptor indexing,
  compiled pipeline-cache payloads, precise occlusion, legacy device-event commands, sparse resources, external memory, multisampling,
  indirect dispatch, partial/flipped/converting blits, or arbitrary
  raster state. Custom allocation callbacks are rejected at creation.
- **Resource lifetime capacity remains bounded.** SGFX tables use persistent
  append-only IDs. When no live Vulkan objects retain an epoch's IDs, the ICD
  invalidates stale recordings, drops backend caches, and creates a fresh table.
  A test creates/destroys 1250 buffers on an otherwise idle device, then performs
  real compute work. With long-lived pipelines/resources, unique transient
  resources and descriptor configurations still consume that epoch's table
  budget (including 8192 texture identities, 4096 buffers, 4096 memory allocations and 32768 bind-group configurations). Identical
  descriptor configurations reuse definitions and physical groups; GPU image
  remapping invalidates physical groups. Exhaustion returns an allocation
  error. General reclamation requires generational core/backend slots or a safe
  migration of live resources; this remains a blocker for a general driver.
- **Command programs are bounded to 65,536 commands.** Vulkan recording uses
  the canonical IR bound. Lowering may add IR operations, which must also fit
  this bound. Overflow rejects recording/submission without signaling success.
- **Scarlet VirGL is executable through the linked test path.** The
  `aarch64-unknown-scarlet` cube uses the same Vulkan frontend, SGFX facade,
  SPIR-V-to-TGSI lowering, `/dev/gpu0`, VirGL submit/completion, GPU readback,
  and `DisplaySurface`. The recorded cube result uses a linked frontend;
  native ICD discovery is not established by that result. The separate Linux/musl
  `scarlet-wsi` build is an actual `.so` ICD loaded by the ordinary Khronos
  loader on Scarlet's Linux ABI, with 60 KHR_display presentations verified.
  The glibc Linux build also passes GPU clear/readback through the standard
  loader and Wayland presentation through a SHM parent/GPU child. Identical
  Win64 probes under Wine/Box64 pass 4096-pixel readback and 40 presentations
  across two swapchain sizes. These checks establish a bounded path, not
  Vulkan conformance or general Windows application compatibility.
  A618 still
  rejects programmable execution and is not advertised as a Vulkan adapter.
- **Conservative memory and transfer costs.** Referenced host-visible buffers
  are uploaded from CPU shadow storage. GPU-written mapped buffers and
  image-to-buffer copies use blocking readback. This establishes functional
  ordering and observable output, not a performance result for a native Vulkan
  driver.

The ordinary-loader vkQuake2 release execution now displays the bundled map,
weapon and HUD with triangle billboard particles. See the
[application compatibility record](vulkan-game-compatibility.md) for the exact
build, game settings, GPU evidence and remaining limitations.

## ABI references

The implementation follows the loader's
[driver interface](https://github.com/KhronosGroup/Vulkan-Loader/blob/bde79ad2dd832db9180c4a6eca2e84ceb12b1bb0/docs/LoaderDriverInterface.md)
and [dispatchable-object layout](https://github.com/KhronosGroup/Vulkan-Headers/blob/ee2ec5fd83dafce291024683b50dc89219333076/include/vulkan/vk_icd.h).
Descriptor invalidation is specified by
[`vkUpdateDescriptorSets`](https://docs.vulkan.org/refpages/latest/refpages/source/vkUpdateDescriptorSets.html).
The Vulkan API version alone does not define an optional arbitrary subset.

## Skia Vulkan ABI work

The memory2 entry points use the same allocation, ownership, range and alignment
checks as the original API. Requirements queries preserve output chains and
write false dedicated-allocation requirements. Sparse queries report zero;
YCbCr creation returns `ERROR_FEATURE_NOT_PRESENT`. Disabled Vulkan 1.1 feature
records can accompany device creation, but nonzero unsupported features are
rejected. Layout-support queries share creation's pure validation and do not
allocate temporary layouts. Core properties2 queries require a 1.1 instance;
KHR names retain their earlier behavior.

The memory2 GPU integration contract binds real buffer/image allocations,
clears an RGBA image on the GPU, copies it to a mapped buffer and verifies all
16 red pixels. It also checks matching legacy/new requirements and supported
versus rejected descriptor layouts. Both ignored queue-family indices and
explicit family `0 -> 0` are tested for image and buffer barriers. The latter
must not be interpreted as an ownership transfer. Both GPU contracts passed
on macOS Metal; all 80 release unit tests passed as well.

On 2026-10-14 the normal Scarlet AArch64 release desktop image passed
`myrica-vulkan-probe` with HVF, four guest CPUs and `virtio-gpu-gl-pci`. WebKit
Skia Ganesh created an exported BGRA image, drew white/red/blue regions through
Vulkan and the native SGFX VirGL backend, checked GPU readback pixels, and
released the context, device, loader and image. All three independent lifecycle
cycles passed with ICD revision `7a62d766818b205fc21a69aae0d72792d4b374aa`
and WebKit revision `a666d28d0d4b46016bc9dd42edbfafd0ad81ad06`.
The preceding initialized C++ TLS/Rust TLS/errno isolation check also passed
across four native threads. The existing native cube's three GPU readback
cycles passed in the same guest.

These initial tests established Ganesh drawing through the native driver.
Subsequent native Myrica integration uses WebCore's Skia Vulkan renderer and
shared BGRA presentation, including native NV12/I420 video composition.

## Queue worker diagnostics and descriptor retirement

`SGFX_VULKAN_CALL_PROFILE=1` reports one-second API queue/service timings and
the six most expensive operations. Nested worker phases distinguish command
resolution, validation, buffer snapshots, IR recording, backend submission,
readback and descriptor retirement. Nested durations overlap the enclosing
API call; summing them double-counts work. The disabled path reads no clocks.

Transient descriptor groups are retired at the end of a queue packet. Backend
retirements now see one unchanged resource-table revision; logical identities
are invalidated after that batch. Interleaving those steps previously caused
the dynamic backend to resynchronize the entire table for every group.
Failed groups remain available for retry, while successful groups retire and
keep their generation checks. Native YouTube profiling identified 150-210ms
per second in the old retirement path. Unit tests cover a failure between two
successful releases, an unchanged revision throughout the backend batch,
retry and repeated slot reuse with immutable submitted descriptor snapshots.

Shader modules have a separate bounded table of 1,024 live entries; compute
and programmable graphics pipelines each retain their existing 256-entry
limits. A graphics pipeline can own two distinct shader stages. Several Skia
contexts share the device table, so a 256-entry shader table could exhaust
before the graphics-pipeline limit. The R3 metadata importer and retirement
scanner use the same declared capacities. Released generations remain invalid
and slots remain reusable. The simultaneous-pipeline contract test retains
160 graphics pipelines (320 distinct shader modules) while rendering and
validating their pixels, rather than testing only sequential create/destroy.


## Synchronization2 validation

On 2026-10-30, the host suite passed 123 unit tests and 29 opt-in GPU
contract tests against the built ICD on Metal. The synchronization2 GPU case
checks a future host timeline signal after queue acceptance, ignored binary
semaphore values, device events, partial buffer barriers and exact transfer
readback after fence completion. Run it with:

```sh
cargo build --locked -p vulkan-sgfx --lib
cargo test --locked -p vulkan-sgfx --lib
cargo test --locked -p vulkan-sgfx --test contracts -- --ignored --test-threads=1
```

The same queue/event/transfer path passed in native OpenJDK 25 on an AArch64
Scarlet HVF guest with the VirGL plugin. A second native test used only
`vkCmdPipelineBarrier2KHR` and `vkQueueSubmit2KHR` for 320×240 and 384×256
SDL/Vulkan swapchain presentation, including resize and keyboard/text input.
It also checked VMA allocation and reached C2 level 4 in render/swapchain
methods; these methods may subsequently deoptimize. Neither test establishes
Vulkan conformance or broader application compatibility.

| Validation artifact | SHA-256 |
| --- | --- |
| Native ICD | `2e7fa1542a8340fddf0e2f7c1af2f7d9b57d70ea394991dabf10de980b385777` |
| Native queue/event log | `544bf49335086c01f6af3908859136b8545c73f6e0128315b38e05e7990f1a0d` |
| Native presentation log | `f0195f8688119acb540ea3622cf295aa63aeed9df5e6274bf2f47b0b29d5a677` |


## Modern rendering validation

On 2026-10-30 all 33 opt-in Metal GPU contract tests passed. New pixel comparisons
cover dynamic rendering without Vulkan render-pass/framebuffer objects,
GPU-written indirect arguments/counters, first-instance values and signed
indexed vertex offsets. Timestamp tests check increasing hardware GPU time,
availability, 32/64-bit reads, strided copies, untouched padding and host reset.

A native AArch64 HVF/OpenJDK 25 test passed dynamic rendering and both indirect
count variants with GPU-updated argument/count buffers, baseVertex=-1, 128 exact
pixel comparisons and teardown. Its synchronization2 transfer/event/timeline
checks also passed. The guest rejects unsupported timestamp pools and reports
zero valid bits on this host. Direct CGL tests that bypass QEMU/virglrenderer on
Apple M3 Pro (OpenGL 4.1 Metal) report `GL_QUERY_COUNTER_BITS=0` and two available
queries with value zero, reproducing the timer limitation in macOS OpenGL.
This is separate from the working Metal GPU-clock path; zero values are not
replaced with CPU time. Reproduction sources, hashes and logs are retained in
`target/vulkan-next/summary.json` and its fixture directory.


| Modern native validation artifact | SHA-256 |
| --- | --- |
| AArch64 ICD | `625f58b4a6d01f09e3bcb94085d5cba33cdb67ee7cf1058c571d85e04d31b3db` |
| HVF guest log | `81c4dc7858b2c5e9ee1cd48aa6046f38e28baeaf9c6bc612935e0d6186335965` |
| Query-buffer kernel Image | `4c335ca8f0dbee755fcd14abdc490c7ab62195ca597f64dfc82f30fc46fd199c` |


## Scoped query validation

On 2026-10-30 all 35 opt-in Metal GPU contract tests passed, including visible
indirect drawing versus an empty occlusion scope, pass splitting, 32/64-bit
results, availability and buffer copies. The native AArch64 HVF/OpenJDK test
also passes visible/empty scopes around normal and indexed indirect-count draws,
result copies, host reads and host resets, while retaining 128 exact pixel
comparisons. Unsupported precise/statistics features are rejected explicitly.
This host cannot exercise positive pipeline-statistics results; their hardware
path remains unverified here. The conditional graphics contract checks IA, VS
and fragment counters on backends that expose the feature.


| Scoped-query native artifact | SHA-256 |
| --- | --- |
| AArch64 ICD | `d59c80831c5fa4606f474aaff39239a2484ce38b34c2e595581c36d0a3d78723` |
| HVF guest log | `87421c1fe93e98e54a70e909edd40573702d60a0dfd835c62cc3b72fcb42c859` |

Full receipts and logs are retained under `target/vulkan-next/scoped-summary.json`.
