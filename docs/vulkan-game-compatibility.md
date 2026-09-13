# Vulkan application compatibility

The current development ICD exposes 103 procedures on macOS, including 16
command procedures. This remains a bounded, non-conformant Vulkan 1.0 path.

## Executable additions

- Scalar separate and combined image/sampler descriptors. Combined SPIR-V
  globals are structurally split into separate handles before Naga validation.
  Binding `b` becomes IR binding `2*b`, with its combined sampler at `2*b+1`.
- Uniform/storage dynamic descriptor offsets, ordered by set and binding.
  Effective ranges and 256-byte alignment are checked before submission.
- RGBA8/BGRA8 sampled images and checked staging-buffer uploads, including
  offset destinations, row padding, and partial final rows. Uploads currently
  copy coherent CPU-shadow bytes into owned `WriteTexture` commands; this is
  not a native GPU buffer-to-image transfer implementation. GPU-written sources
  in the same command stream are rejected.
- Static subrectangle and dynamic viewport/scissor state, clipped scissor
  coverage, and common source-alpha/additive blend operations.
- One subpass with one color reference at any attachment index and optional
  D32 depth. A color attachment may subsequently be sampled.
- `VK_MVK_macos_surface` alongside `VK_EXT_metal_surface`; both use the
  selected SGFX Metal adapter and the ordinary loader's swapchain dispatch.
- Native VirGL shader sampling: Naga image/sampler pairs reflect SGFX bindings
  into per-stage TGSI slots, with actual `TEX`, `TXL`, and `TXB` instructions.

Canonical IR adds `SetViewport` and `SetPushConstants`, bringing the owned
command enum to 24 variants. Pipeline layouts declare validated, stage-specific
push-constant ranges, bounded to 128 bytes. Incremental updates retain owned
bytes and compatible-layout information at every draw and dispatch. Metal
executes push constants; native VirGL currently reports a zero-byte limit and
rejects nonempty push-constant layouts.
The maximum IR bind-group entries increase from 16 to 32 to accommodate the
frontend's descriptor expansion. Sampling and upload already had canonical
IR resource/command types; the Vulkan lowering and VirGL programmable path now
use those types. Fixed pipelines and programmable pipelines remain independent.

## Ordinary-loader textured cube

Build with the host Rust toolchain and an installed Khronos Vulkan loader:

```sh
cargo build --locked --release -p vulkan-sgfx --features cube-demo --lib --bin vulkan-cube --example headless
python3 crates/vulkan-sgfx/tools/write_icd_manifest.py \
  target/release/libvulkan_sgfx.dylib target/sgfx-textured-icd.json
export VK_DRIVER_FILES="$PWD/target/sgfx-textured-icd.json"
# If the system loader is outside the normal macOS lookup paths:
export DYLD_LIBRARY_PATH=/path/to/installed/vulkan-loader/lib
target/release/vulkan-cube --textured --dynamic-viewport --dynamic-uniform \
  --push-constants --verify --output target/vulkan-textured-cube.png
target/release/examples/headless 16 --push-constants
```

Use `libvulkan_sgfx.so` and the system library lookup path on Linux. The host
application calls `ash::Entry::load()` and has no direct ICD-loading branch.
`SGFX_VULKAN_LOADER` and `SGFX_ICD_LIBRARY` are used only by separate repository
diagnostics, never by this host application.

On Apple M3 Pro, the 256-square GPU readbacks match exactly between static and
dynamic viewport/scissor and between ordinary and dynamic uniform descriptors.
Texture sampling changes 18,002 foreground pixels relative to the colored cube.
Incremental vertex push constants produce the same pixels as the uniform
transform. Two draws with different transforms match an exact split-image
control, and two compute dispatches preserve different incremental values across
16 submissions, checking all 256 output words. Depth draw-order invariance, a depth-disabled control, rotation, culling and
UINT16/UINT32 index equivalence also pass. A 512-square texture render is saved
from GPU readback. WGSL and SPIR-V sampling fixtures pass the Mesa TGSI parser
and VirGLRenderer 1.3.0 shader-object acceptance with no GL error.

## Upstream vkQuake2 probe

Upstream [vkQuake2](https://github.com/kondrak/vkQuake2/tree/6763f207229f97cffabb6fc2da72017a794b139b)
at commit `6763f207229f97cffabb6fc2da72017a794b139b` builds on macOS against the
installed Khronos loader and unmodified source. The official 1.5.10 release's
bundled demo data supplies `baseq2/pak0.pak`; no game code/data is incorporated
into SGFX.

With `VK_DRIVER_FILES` selecting this ICD, the game identifies
`SGFX Vulkan (Apple M3 Pro)` and creates its Metal surface, FIFO swapchain,
synchronization, three render passes, world/UI depth images, intermediate
color images, framebuffers, command pools and command buffers.

**No gameplay frame has been verified.** The next stop is the game's particle
texture mipmap generation, which calls the missing `vkCmdBlitImage`. Multi-mip
image creation is also unsupported. Release-mode game assertions do not stop
at those earlier failed creations, and the missing command produces a null
dispatch call; LLDB locates it in `generateMipmaps`.

The ordinary-loader `shader_probe` example accepts 22 of the game's 23
unmodified SPIR-V modules, including combined-sampler fragments and push-constant
transforms. The PointSize vertex module triggers a Naga 24 writer/parser
interface-structure layout regression. A post-normalization validation rejects
it before WGPU, preserving device usability for subsequent modules. An authored
minimal SPIR-V interface fixture covers this regression. Mip storage/blits,
sampler LOD behavior, more vertex formats/topologies and durable resource reuse
are still required for this game's renderer. Procedure counts
and successful initialization are not evidence of gameplay compatibility.
