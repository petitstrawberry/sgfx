# Experimental Vulkan frontend

`vulkan-sgfx` is a loader-discoverable, **non-conformant development ICD**. It
implements a bounded headless subset to exercise Vulkan → SGFX portable IR →
backend execution. Its manifest and physical device use the Vulkan 1.0 ABI
version. This is not a claim of Vulkan 1.0 conformance: substantial mandatory
core functionality is absent. Do not select it as a general application driver.
Native loader integration, resource reclamation with long-lived objects,
broader API coverage, and conformance work remain incomplete.

## Implemented path

- An actual C ABI `cdylib`, ICD interface negotiation (versions 2–5), scoped
  procedure lookup, instance/physical-device/device/queue/command-buffer
  dispatchable handles with the loader's required first word, and checked
  object ownership. Only the three `vk_icd*` entrypoints are exported as unmangled
  symbols; ordinary Vulkan commands are returned through procedure lookup.
- SGFX discovers actual adapters and reports one `VkPhysicalDevice` for each.
  `vkCreateDevice` opens the selected adapter, rather than probing or choosing a
  backend again inside the Vulkan frontend. One graphics/compute/transfer queue
  is exposed according to that adapter's SGFX capabilities. No extensions or
  optional physical-device features are advertised.
- SPIR-V shader modules, compute pipelines, and vertex/fragment pipelines.
  Naga validates and normalizes Vulkan coordinate conventions before SGFX
  shader definition; backend shader/pipeline validation occurs at creation.
- Descriptor sets for uniform and storage buffers; up to four sets with 16
  bindings each, one descriptor per binding, no dynamic offsets or arrays.
- Host-visible coherent memory with stable, aligned mappings; buffer memory
  binding, host upload/readback, and nonoverlapping allocations. Buffers have
  four-byte sizes; binding offsets use the reported alignment. Uniform ranges
  are limited to 16 KiB and storage ranges to 128 MiB.
- Primary command buffers and pools, pipeline/set binding, compute dispatch,
  single-color render passes, non-indexed and indexed drawing, vertex buffers,
  buffer copies, whole-image RGBA8 readback, supported whole-resource barriers,
  D32 depth testing, front-face/back-face culling, and fences.
- Offscreen graphics: RGBA8_UNORM color, optional D32_SFLOAT depth, optimal 2D
  images, one mip/layer/sample, at most 2048×2048, one color attachment,
  triangle lists, one vertex binding, four vertex formats, and one instance per
  draw. Blending, stencil, multisampling, indirect drawing, sampled images, and
  general dynamic state are absent.

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

Fence status and finite/zero-timeout waits use shared atomic state outside the
device worker. They therefore remain observable while another thread is
blocked submitting work. A deterministic test covers an intentionally
unserviced worker channel. `vkQueueWaitIdle`/`vkDeviceWaitIdle` serialize behind
prior queue work. No semaphores are implemented; submits containing semaphore
waits or signals are rejected.

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

This builds a fresh host ICD and `render_demo`, writes its absolute-path driver
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
Set `SGFX_VULKAN_LOADER=/absolute/path/to/libvulkan` to override discovery.
The script selects the Rust host target explicitly so a Scarlet cross-target
configuration does not affect the demo. On Linux it selects Mesa llvmpipe with
surfaceless EGL to avoid GL-to-Vulkan recursion through Zink.

The console identifies the loader, manifest, selected SGFX physical device,
readback checks, and absolute PNG path. The example refuses another ICD. The
PNG encoder performs lossless encoding only; no image generator, CPU renderer,
or composited visual overlay is involved. `render_demo` also accepts an explicit
PNG path when run directly with `SGFX_VULKAN_LOADER` and `VK_DRIVER_FILES` set.

## Run the examples

From the repository root:

```sh
cargo build -p vulkan-sgfx --lib --examples
cargo run -p vulkan-sgfx --example headless
cargo run -p vulkan-sgfx --example offscreen
cargo test -p vulkan-sgfx --lib
cargo test -p vulkan-sgfx --test contracts -- --ignored
```

The examples load the freshly built ICD directly by default. An optional first
argument or `SGFX_ICD_LIBRARY` selects another ICD library. `headless` compiles a
compute shader, verifies all 256 output words, and reports CPU recording and
submit durations. `offscreen` compiles vertex/fragment SPIR-V, renders a 64×64
triangle, and checks center, corner, and asymmetric pixels after readback.
GPU contract tests are ignored by default so CPU-only CI can run library tests;
run them explicitly on a machine with a supported host backend.

To exercise the real Khronos loader instead, first write a manifest with an
absolute library path (use `libvulkan_sgfx.so` on Linux):

```sh
python3 crates/vulkan-sgfx/tools/write_icd_manifest.py \
  target/debug/libvulkan_sgfx.dylib target/sgfx_icd.json
export VK_DRIVER_FILES="$PWD/target/sgfx_icd.json"
export SGFX_VULKAN_LOADER=/absolute/path/to/libvulkan.1.dylib
cargo run -p vulkan-sgfx --example loader_probe
cargo run -p vulkan-sgfx --example headless
cargo run -p vulkan-sgfx --example offscreen
```

`loader_probe` verifies loader discovery, reported device properties, logical
device and queue creation, idle, and destruction. The other two examples use
the real loader's normal `vkGetInstanceProcAddr` when `SGFX_VULKAN_LOADER` is set,
exercising loader dispatch for resource, command, and submission APIs as well.
The ICD accepts the loader's private device creation records while rejecting
unsupported application feature chains.

The host backend mask is fixed: Metal on macOS, GL on other host platforms.
The ICD never enables WGPU's Vulkan backend or reads WGPU backend-selection
configuration. Execution is verified on macOS Metal and Linux aarch64 with Mesa
26.1.3 llvmpipe (LLVM 21.1.8), surfaceless EGL 1.5/OpenGL ES 3.2, and the real
Khronos Vulkan Loader 1.4.341. The Linux check passed loader discovery and device
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

- **Not Vulkan conformant.** The ICD exposes 83 procedure names, including 12
  `vkCmd*` operations, but this is only a bounded executable subset. There are
  no surfaces/swapchains, presentation, semaphores,
  secondary command buffers, descriptor indexing, push constants, pipeline
  caches, queries, events, sparse resources, external memory, multisampling,
  complex render passes, indirect operations, sampled images, or arbitrary
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
