# Experimental Vulkan frontend

`vulkan-sgfx` is a loader-discoverable, **non-conformant development ICD**. It
implements a bounded executable subset to exercise Vulkan → SGFX portable IR →
backend execution and, on macOS, presentation through Metal. Its manifest and
physical device use the Vulkan 1.0 ABI version. This is not a claim of Vulkan
1.0 conformance: substantial mandatory core functionality is absent. Do not
select it as a general application driver. Cross-platform WSI, resource
reclamation with long-lived objects, broader API coverage and conformance work
remain incomplete.

## Implemented path

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
  Naga validates and normalizes Vulkan coordinate conventions before SGFX
  shader definition; backend shader/pipeline validation occurs at creation.
- Stage-specific push constants up to 128 bytes on capable Metal adapters,
  with incremental updates and a value snapshot per draw/dispatch. Native
  VirGL reports no push-constant support and rejects nonempty ranges.
- Descriptor sets for uniform/storage buffers (including dynamic offsets),
  separate images/samplers and combined image samplers; four sets with 16
  logical bindings each, one descriptor per binding. Arrays are unsupported.
- Host-visible coherent memory with stable, aligned mappings; buffer memory
  binding, host upload/readback, and nonoverlapping allocations. Buffers have
  four-byte sizes; binding offsets use the reported alignment. Uniform ranges
  are limited to 16 KiB and storage ranges to 128 MiB.
- Primary command buffers and pools, pipeline/set binding, compute dispatch,
  single-color render passes, non-indexed and indexed drawing, vertex buffers,
  buffer copies, checked buffer-to-image uploads, whole-image RGBA8/BGRA8
  readback, supported whole-resource barriers, dynamic viewport/scissor,
  D32 depth testing, front-face/back-face culling, fences, and binary semaphores.
- Offscreen graphics: RGBA8_UNORM or BGRA8_UNORM color, optional D32_SFLOAT
  depth, optimal 2D images, one mip/layer/sample, at most 2048×2048, one color
  attachment, triangle lists, one vertex binding, four vertex formats, and one
  instance per draw. Common alpha/additive blending and sampled 2D textures
  work. Stencil, multisampling, mipmaps, image blits, indirect drawing,
  and other dynamic state are absent.
- macOS WSI exposes `VK_KHR_surface`, `VK_EXT_metal_surface`, `VK_MVK_macos_surface`,
  `VK_KHR_portability_enumeration`, and `VK_KHR_swapchain`. It implements Metal
  surface creation, surface capability/format/mode queries, FIFO swapchains,
  acquire, and queue present. Swapchain images are SGFX `PRESENT` textures on
  the exact WGPU/Metal device selected by `vkCreateDevice`.

Each `vkCmd*` records into command-buffer-local owned storage on the calling
thread. Submit resolves the Vulkan handles and descriptor state into
`sgfx-core::ir::OwnedCommand`, then reconstructs validated borrowed IR for the
backend. This is a transitional representation: the local Vulkan command enum,
submit-time stream reconstruction, and append-only SGFX resource table still
need to be replaced by a reusable canonical owned program and generational
resource identities. Image-to-buffer copies currently split SGFX submissions
and use backend image readback followed by an ordered buffer write.

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
its waits. Timeline semaphores are not implemented.

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

The Scarlet toolchain currently drops `cdylib` output, so its checked path is a
linked executable rather than loader-discovered `.so` integration. This
packaging difference is below the Vulkan calls exercised by the test program.

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

- **Not Vulkan conformant.** The ICD exposes 103 procedure names on macOS,
  including 16 `vkCmd*` operations, but this is only a bounded executable
  subset. WSI currently covers macOS Metal, FIFO mode, opaque composition, and
  fixed-size swapchains. Other platform surfaces, window-resize handling in the
  example, timeline semaphores, secondary command buffers, descriptor indexing,
  pipeline caches, queries, events, sparse resources, external memory, multisampling,
  multiple subpasses, indirect operations, mipmaps, image blits, or arbitrary
  raster state. Custom allocation callbacks are rejected at creation.
- **Resource lifetime capacity remains bounded.** SGFX tables use persistent
  append-only IDs. When no live Vulkan objects retain an epoch's IDs, the ICD
  invalidates stale recordings, drops backend caches, and creates a fresh table.
  A test creates/destroys 1250 buffers on an otherwise idle device, then performs
  real compute work. With long-lived pipelines/resources, unique transient
  resources and descriptor configurations still consume that epoch's table
  budget (including 1024 buffers and 256 bind-group configurations). Identical
  descriptor configurations reuse definitions. Exhaustion returns an allocation
  error. General reclamation requires generational core/backend slots or a safe
  migration of live resources; this remains a blocker for a general driver.
- **Scarlet VirGL is executable through the linked test path.** The
  `aarch64-unknown-scarlet` cube uses the same Vulkan frontend, SGFX facade,
  SPIR-V-to-TGSI lowering, `/dev/gpu0`, VirGL submit/completion, GPU readback,
  and `DisplaySurface`. The Scarlet toolchain currently drops the requested
  `cdylib`, so this result is not native `.so` loader integration. A618 still
  rejects programmable execution and is not advertised as a Vulkan adapter.
- **Conservative memory and transfer costs.** Referenced host-visible buffers
  are uploaded from CPU shadow storage. GPU-written mapped buffers and
  image-to-buffer copies use blocking readback. This establishes functional
  ordering and observable output, not a performance result for a native Vulkan
  driver.

## ABI references

The implementation follows the loader's
[driver interface](https://github.com/KhronosGroup/Vulkan-Loader/blob/bde79ad2dd832db9180c4a6eca2e84ceb12b1bb0/docs/LoaderDriverInterface.md)
and [dispatchable-object layout](https://github.com/KhronosGroup/Vulkan-Headers/blob/ee2ec5fd83dafce291024683b50dc89219333076/include/vulkan/vk_icd.h).
Descriptor invalidation is specified by
[`vkUpdateDescriptorSets`](https://docs.vulkan.org/refpages/latest/refpages/source/vkUpdateDescriptorSets.html).
The Vulkan API version alone does not define an optional arbitrary subset.
