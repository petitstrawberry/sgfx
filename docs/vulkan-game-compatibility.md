# Vulkan application compatibility

The current development ICD exposes 104 procedures on macOS, including 17
command procedures. This remains a bounded, non-conformant Vulkan 1.0 path.

## Executable additions

- Scalar separate and combined image/sampler descriptors. Combined SPIR-V
  globals are structurally split into separate handles before Naga validation.
  Binding `b` becomes IR binding `2*b`, with its combined sampler at `2*b+1`.
- Uniform/storage dynamic descriptor offsets, ordered by set and binding.
  Effective ranges and 256-byte alignment are checked before submission.
- RGBA8/BGRA8 sampled images and checked staging-buffer uploads, including
  offset destinations, row padding, and partial final rows. Uploads currently
  copy coherent CPU-shadow bytes into owned `WriteTextureMip` commands; this is
  not a native GPU buffer-to-image transfer implementation. GPU-written sources
  in the same command stream are rejected.
- Static subrectangle and dynamic viewport/scissor state, clipped scissor
  coverage, and common source-alpha/additive blend operations.
- One subpass with one color reference at any attachment index and optional
  D32 depth. A color attachment may subsequently be sampled.
- `VK_MVK_macos_surface` alongside `VK_EXT_metal_surface`; both use the
  selected SGFX Metal adapter and the ordinary loader's swapchain dispatch.
- Sampled RGBA8/BGRA8 mip chains, per-mip uploads/barriers/readback and GPU
  `vkCmdBlitImage` generation with nearest or linear filtering. Blits cover
  complete, positive-direction mip rectangles with matching formats and one
  layer. Partial/flipped blits and format conversion are rejected.
- Nearest/linear mip filters and nonzero sampler LOD clamps. Sampled views
  cover the declared chain; render/depth/storage/present attachments retain
  one mip. Native VirGL supports color mip storage and full-mip blits when the
  kernel advertises `GPU_EXECUTION_SUPPORT_IMAGE_MIPS`; older kernels and A618
  continue to reject them.
- Native VirGL shader sampling: Naga image/sampler pairs reflect SGFX bindings
  into per-stage TGSI slots, with actual `TEX`, `TXL`, and `TXB` instructions.
- Native VirGL bounded dynamic reads of arrays, vectors and matrix columns.
  Pointer reads take their value at the load, preserving preceding stores;
  dynamic stores, runtime-sized arrays and storage-buffer indexing remain
  unsupported. This covers ordinary vertex-indexed fullscreen triangles.
- Nonindexed triangle strips on Metal, plus signed base vertices for indexed
  triangle lists. Native VirGL also executes indexed and nonindexed strips,
  including seven-index strips and signed base vertices. WGPU rejects indexed
  strips before acceptance because its implicit primitive restart would change
  ordinary Vulkan indices.
- Swapchain transitions from `PRESENT_SRC_KHR` to transfer source and full-mip
  image readback with either implicit or explicitly tightly packed row sizes.
  Swapchain metadata retains the application's declared image usage.
- Immutable physical bind groups are cached across draws. Replacing or removing
  a mapped GPU image invalidates the cache; updates to the same image remain
  visible without rebuilding its group.

Canonical IR adds `SetViewport`, `SetPushConstants`, `WriteTextureMip` and
`BlitTexture`, bringing the owned command enum to 26 variants. Texture
descriptors carry checked mip counts/extents/total sizes, sampler descriptors
carry mip filtering/LOD clamps, and `TextureMip` barriers synchronize selected
levels independently. Pipeline layouts declare validated, stage-specific
push-constant ranges, bounded to 128 bytes. Incremental updates retain owned
bytes and compatible-layout information at every draw and dispatch. Metal
executes push constants. Native VirGL flattens each stage's push-constant block
after its UBO constants, with an owned snapshot per draw and a 128-byte limit.
Its mip, sampler and strip changes extend the backend's internal command data;
they reuse the already canonical IR commands and require no new public IR fork.
The maximum IR bind-group entries increase from 16 to 32 to accommodate the
frontend's descriptor expansion. Sampling and upload already had canonical
IR resource/command types; the Vulkan lowering and VirGL programmable path now
use those types. Fixed pipelines and programmable pipelines remain independent.
The canonical command bound is 65,536 and the bind-group definition bound is
4,096. Vulkan recording uses the canonical command bound; exhausting either
budget returns an error rather than silently dropping commands. These are
bounded application capacities, not general resource reclamation.

## Ordinary-loader textured cube

Build with the host Rust toolchain and an installed Khronos Vulkan loader:

```sh
cargo build --locked --release -p vulkan-sgfx --features cube-demo --lib --bin vulkan-cube --example headless --example mipmap
python3 crates/vulkan-sgfx/tools/write_icd_manifest.py \
  target/release/libvulkan_sgfx.dylib target/sgfx-textured-icd.json
export VK_DRIVER_FILES="$PWD/target/sgfx-textured-icd.json"
# If the system loader is outside the normal macOS lookup paths:
export DYLD_LIBRARY_PATH=/path/to/installed/vulkan-loader/lib
target/release/vulkan-cube --textured --dynamic-viewport --dynamic-uniform \
  --push-constants --verify --output target/vulkan-textured-cube.png
target/release/examples/headless 16 --push-constants
target/release/examples/mipmap
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

Six nonindexed four-vertex strips reproduce the indexed cube exactly. Both
positive and negative base vertices reproduce the original UINT16/UINT32
readbacks exactly. A separate GPU test confirms that rejecting an indexed strip
does not execute its preceding clear. Cached groups observe a red-to-green
update to one image and a subsequent remapping to a different blue image.

The ordinary-loader mipmap example checks eight actual GPU chains: RGBA/BGRA,
nearest/linear blits, and 8×8/7×3 dimensions. Every level matches a CPU reference
within one UNORM unit. Compute sampling verifies explicit LOD, nearest/linear
mip filters and fractional nonzero LOD clamps against GPU-read levels. Partial
uploads to mip 1 change only the selected pixel. A declared 4096-set descriptor
pool creates without materializing bind groups; its actual one-set descriptor
capacity correctly rejects a second allocation.

## Upstream vkQuake2 execution on macOS

Upstream [vkQuake2](https://github.com/kondrak/vkQuake2/tree/6763f207229f97cffabb6fc2da72017a794b139b)
at commit `6763f207229f97cffabb6fc2da72017a794b139b` builds on macOS against the
installed Khronos loader and unmodified source. The official 1.5.10 release's
bundled demo data supplies `baseq2/pak0.pak`; no game code/data is incorporated
into SGFX.

With `VK_DRIVER_FILES` selecting this ICD, the game identifies
`SGFX Vulkan (Apple M3 Pro)` and creates its Metal surface, FIFO swapchain,
synchronization, three render passes, world/UI depth images, intermediate
color images, framebuffers, command pools and command buffers.

The bundled `demo1` map now renders its textured world, first-person weapon,
crosshair and HUD at 640×480. The renderer's frame counter reached 1,193 while
the game continued submitting frames. The game's own `screenshot` command saved
two GPU-read TGA images with different views and no loading overlay. The actual
macOS window was also visually checked. This is a verified game rendering path.
Movement and combat have not been systematically tested, and sustained
performance has not been benchmarked.

The checked game executable uses upstream release optimization. Its Vulkan
renderer was built with `-O3 -g -D_DEBUG -DNDEBUG` to retain upstream Vulkan
result logging during diagnosis; source, shader modules and game data were
unmodified. The SGFX driver uses Cargo's release profile. Use the game's ordinary
`vk_point_particles=0` setting to select triangle billboard particles because
PointSize shaders and point pipelines remain unsupported. Optional line
pipelines are also rejected; the default filled rendering path does not use
them. Their creation errors remain visible in the diagnostic renderer's log.

After building this ICD and selecting its manifest as above, run from the
game's `macos/release` directory:

```sh
./quake2 +set vid_ref vk +set vid_fullscreen 0 +set vk_validation 0 \
  +set vk_mode 3 +set vk_point_particles 0 +map demo1
```

No game-specific loader or SGFX command interface is used by the game. It links
the installed Khronos Vulkan loader, which discovers the ICD from
`VK_DRIVER_FILES`. One of the game's TGA readbacks is serialized as the local
diagnostic artifact `target/vkquake2-sgfx-demo1-release.png`; game assets are not
bundled with SGFX.

The ordinary-loader `shader_probe` example accepts 22 of the game's 23
unmodified SPIR-V modules, including combined-sampler fragments and push-constant
transforms. The PointSize vertex module triggers a Naga 24 writer/parser
interface-structure layout regression. A post-normalization validation rejects
it before WGPU, preserving device usability for subsequent modules. An authored
minimal SPIR-V interface fixture covers this regression. Point/line rendering,
indexed strips on WGPU, general resource reclamation and wider Vulkan coverage
remain incomplete. This macOS result does not establish Scarlet game compatibility.

## Scarlet Linux ABI and ordinary-loader WSI

Build `vulkan-sgfx` on Linux/musl with
`--release --no-default-features --features scarlet-wsi`, linking the ordinary
Linux `libsws_client_c.so`. This enables the native Scarlet GPU namespace and
seven standard `VK_KHR_display` procedures alongside common surface/swapchain
queries. An unmodified Khronos loader reads the standard ICD JSON manifest.
The primary SWS output supplies a fullscreen display plane; three registered
SGFX GPU images rotate only after the compositor returns the exact buffer
identity and commit serial. Producer GPU completion precedes presentation.
Closing the window retires its retained image without waiting forever for the
last displayed buffer.

An ordinary C Vulkan application, `crates/vulkan-sgfx/examples/display.c`, links
only `libvulkan`. On Scarlet AArch64 QEMU it discovers
`SGFX Vulkan (Scarlet VirGL GPU 0)`, completes 60 acquire/submit/present iterations,
and shuts down cleanly. No preload library, private loader, or SGFX API is used
by this application.

Native VirGL backend release tests cover mip storage/uploads/blit packets,
strip draw ranges and push-constant lowering. Initial vkQuake2 textures and
Vulkan resources now initialize on the guest, and its game module and `demo1`
server load through Scarlet's existing Linux ABI. Bounded dynamic indexing
allows the original world-warp fullscreen shader to create its pipeline; the
game now presents its textured console background, the `demo1` 3D world,
weapon and HUD on Scarlet AArch64 QEMU. Actual QEMU captures show continuing
world frames and changed player views after input. A standard keyboard event
also toggles the game console and pauses the world. These checks use the normal
full-project release image. Subsequent release checks also execute the game's
own `screenshot` command: `quake00.tga` is extracted from the stopped guest disk
and decodes to the actual 1280x800 world, weapon and HUD. Both the original and
batched programmable backend save a 4,096,018-byte TGA. The batched capture has
12,898 distinct RGBA colors and opaque alpha throughout. The upstream `quit`
command shuts down Vulkan, returns exit status 0, and removes the SGFX worker
tasks. Combat and sustained FPS have not been verified. General game
compatibility has not been established. The SWS input/window adapter resides in
Scarlet's `user/lib/sws-client-c/examples/vkquake2`; upstream Vulkan renderer and
shader sources remain separate. A618 arbitrary SPIR-V execution and Chromebook
hardware testing remain incomplete. The tested release kernel and Linux ABI
module are Scarlet `7297aac3e91c09daecd4c09c4c9cb7d57d2e6af3`; the C window/input
adapter and shared SWS library source are
`838333bc392f5e345136aa84132c178de2b64c11`. These kernel changes add in-place
`mremap` shrinking and remove quadratic private-page reclamation. Real guest
checks preserve data across four 16 MiB shrinks and report 4 ms total for four
8 MiB partial unmaps, compared with 509 ms before batch physical reclamation.

## Native programmable submission cost

SGFX `cd37a426e9625efcbfe4f22681a342a47a8784f6` batches consecutive native
programmable draws using the same immutable pipeline. Previously every such
draw became a separate pass and transport packet, repeatedly allocating command
storage and re-emitting framebuffer/pass setup. Chunks now contain at most 64
draws and fit a conservative 64 KiB command budget. The estimate includes
zero-filled constant-register gaps, TGSI source, vertex elements and texture
bindings. The encoder still checks the actual packet size before submission;
the existing single-draw path remains available when a first-use estimate is
too large. Draw order, each draw's constants, pipeline boundaries and first-pass
clears are preserved. This changes only backend planning, with no public API or
canonical IR revision change.

All 48 native VirGL backend library tests pass in Linux release mode with
`--features std,programmable`. Three new regressions cover ordered constant
snapshots across chunks, fixed/programmable pipeline transitions with one clear,
and the byte limit for sparse constant-register banks.

A single release comparison on 2026-09-13 used AArch64 HVF, four guest CPUs,
8 GiB RAM, 1280x800 display WSI, the same upstream game and the kernel/C SDK
revisions above. Temporary clocks around `OwnedCommandBuffer::record`,
`queue.submit` and buffer destruction ran in separate verification images.
For loading-console submissions containing exactly 7,327 owned IR commands,
there were 36 samples in each run:

| Backend | Median `queue.submit` elapsed time |
| --- | --- |
| Original, SGFX `10eb666555e341032eae54cf01433b63ea88f00c` | 388 ms |
| Batched, SGFX `cd37a426e9625efcbfe4f22681a342a47a8784f6` | 200.5 ms |

This measures CPU submission through acceptance, not GPU completion or game
FPS. World views differed between the two runs, so their timings are not a
fixed-camera comparison. The batched run still spends hundreds of milliseconds
submitting world work, and the SGFX device worker remains busy on one CPU.
Buffer-shadow upload copying took only several milliseconds for the roughly
3 MiB world payload. Worker command construction, allocation/reclamation and
GPU/presentation waits were measured separately in the follow-up below. The
clean deployed release ICD is rebuilt from committed source without the clocks.

### Shared draw bindings and reusable buffer upload storage

A follow-up native backend change shares immutable constant/texture snapshots
while pipeline, bind groups and stage push constants remain unchanged inside a
render pass. Indexed draw offsets and signed base vertices keep their own draw
state and do not force another copy of the same bindings. Index/vertex bounds
are still validated for every draw. Buffer writes compare against the latest
ordered CPU shadow; identical data keeps its revision, while first writes and
changed/overlapping ranges still materialize their correct contents. Native
buffer uploads reuse one bounded command vector across all transport packets.
These are private backend changes; canonical IR and the native GPU/SWS SDK
revisions remain unchanged.

The 53 Linux release backend tests pass. Added regressions cover persistent
and pending shadow contents, initial zero writes, binding changes, independent
index state, and shared snapshot ownership across transport chunks.

Additional temporary worker/backend/WSI clocks used the same release kernel,
C SDK and upstream game as above. For exactly 915 loading-console draws in 18
native pass chunks (7,327 owned IR commands), 36 samples in each run give:

| Measured phase | Batched backend before sharing | Shared bindings/upload storage |
| --- | --- | --- |
| Backend plan through submission and cleanup | 201.5 ms | 60 ms |
| Destruction of the lowered drawing events | 139.5 ms | 16 ms |
| `queue.submit` through acceptance | 204 ms | 65 ms |
| Whole Vulkan queue worker job | 274.5 ms | 143.5 ms |

These are medians of CPU elapsed times, not FPS. World views and draw counts
differed between runs and are not a fixed-camera performance comparison.
During the measured world work, producer completion and SWS presentation
usually took 0-2 ms, while command recording/lowering and cleanup continued to
take tens or hundreds of milliseconds. The worker remains CPU-bound; this does
not establish playable performance. The changed release backend again renders
the `demo1` world, weapon and HUD. Its own 1280x800 TGA readback has 11,164
distinct RGBA colors. The upstream quit command returns zero, verified by an
`if`/`else` in the Linux shell, rather than interpolating `$?` in the native
shell. Temporary timing sources and diagnostic binaries are not deployed.

### Changed ranges in persistent native buffers

The native backend bounds changed bytes in each ordered buffer update and
merges subsequent writes to the same buffer. A partial GPU upload is used only
when its physical storage has the exact predecessor revision. New, stale or
partially failed storage is repaired from the complete initialized CPU shadow.
Upload boundaries preserve neighboring bytes in the transport's 32-bit words;
growth includes newly initialized zero-filled gaps. This does not change the
canonical buffer-update command or the GPU/SWS SDK interfaces.

All 57 Linux release backend tests pass, including changed-range merging,
comparison chunk boundaries, first writes, growth and stale-revision repair.
The release ICD renders the native game world, weapon and HUD; its 1280x800
TGA readback has 8,223 distinct RGBA colors and the upstream quit returns zero.
For the same 915 loading-console draws and 18 native pass chunks, 36 temporary
clock samples give a backend median of 57 ms versus 60 ms before changed-range
uploads. This is a modest CPU elapsed-time improvement and is not an FPS
measurement. World buffer preparation costs also depend on the view and are
not a controlled comparison. The deployed ICD contains no timing diagnostics.

### Ordered Vulkan insertions and a bounded game FPS check

Deferred descriptor bindings, barriers and image readbacks are appended in
command order during Vulkan recording resolution. Execution now consumes each
insertion once, preserving insertion order at the same command position and
processing trailing readbacks. It no longer scans all insertions for each SGFX
command. All 26 Linux release ICD library tests pass; a regression bounds the
position checks for 10,000 commands and verifies repeated trailing insertions.

Three ordinary upstream timedemo runs used 1280x800, four AArch64 HVF guest
CPUs, 8 GiB RAM, VirGL 1.3.0, identical game settings and the same Cocoa GL
display binary. The test demo contains the first 64 complete network messages
from the bundled `q2demo1.dm2`, followed by its ordinary EOF marker. The game
reports 57 timed frames in each run:

| Release ICD | Engine elapsed time | Engine FPS |
| --- | --- | --- |
| Shared native bindings (`f7c45be`) | 19.0 s | 3.0 |
| Changed native buffer ranges (`c57e68d`) | 17.1 s | 3.3 |
| Changed ranges and ordered Vulkan insertions | 15.7 s | 3.6 |

These are single, short demo runs without ICD profiling clocks. They show a
modest improvement, not sustained or playable performance. The copied demo
prefix is a private test asset and is not included in the repository or normal
image. The ordinary full bundled demo can be timed with
`+set timedemo 1 +demomap q2demo1.dm2`.

### Reusable heap for the Linux ABI ICD

On Linux with `scarlet-wsi`, the Rust ICD now uses
`dlmalloc::GlobalDlmalloc` for its command, draw and resource allocations.
Temporary allocation clocks identified substantial time inside the musl
allocation/free calls, including destruction of draw storage after submission.
Reusing a driver heap removes this repeated cost without changing the GPU
command stream, Vulkan procedures, canonical IR or native GPU/SWS SDK.
The ordinary Vulkan loader and the application's libc allocator are unchanged;
the ICD does not export replacement `malloc`/`free` entry points. Other build
configurations retain their existing allocator.

The same 57-frame demo prefix was checked on 2026-09-17 with release builds,
1280x800 output, four AArch64 HVF CPUs, 8 GiB RAM and the same game settings,
kernel and Cocoa VirGL display path:

| Release ICD | Engine elapsed time | Engine FPS |
| --- | --- | --- |
| Before the allocator change (`4cf4fc9`), first check | 15.2 s | 3.7 |
| Before the allocator change (`4cf4fc9`), repeated check | 13.5 s | 4.2 |
| Reusable driver heap | 1.1 s | 50.4 |

This is approximately a 12x improvement over the faster baseline in this short
test, not a sustained FPS measurement. The game reports rounded elapsed time
and FPS separately. Normal `demo1` gameplay was also checked after a fresh boot
with timedemo disabled: the world, weapon and HUD render, and player/menu input
updates the display. The final ICD contains no allocation profiling hooks.

All 31 ICD library tests pass on AArch64 Linux/musl, including threaded command
recording and host-allocation alignment/size checks. The locked release build
also succeeds. With the native SWS library in `/path/to/native-libs`, run on
Linux/musl:

```sh
LD_LIBRARY_PATH=/path/to/native-libs LIBRARY_PATH=/path/to/native-libs \
  cargo test --locked -p vulkan-sgfx --no-default-features \
  --features scarlet-wsi --lib
LIBRARY_PATH=/path/to/native-libs \
  cargo build --locked --release -p vulkan-sgfx --no-default-features \
  --features scarlet-wsi
```

Install `target/release/libvulkan_sgfx.so` as the ICD in the Scarlet Linux ABI
image. With the existing vkQuake2 launcher, start normal gameplay from the
Scarlet shell using:

```sh
abi-run linux-aarch64 /bin/sh /usr/games/vkquake2 \
  +set timedemo 0 +set viewsize 100 +map demo1
```

## SuperTuxKart 1.5 on macOS Metal

The release build at upstream commit
`1fb491f507216c5d181ccd85f29ff08eca003827` was checked on an Apple M3 Pro using
the [loader portability patch](../crates/vulkan-sgfx/compat/supertuxkart-1.5-vulkan-loader.patch).
SDL loads the installed Khronos Vulkan loader, and its standard
`VK_DRIVER_FILES` manifest selects the release SGFX ICD. Game shaders are
unchanged. The execution path is Vulkan → SGFX IR → WGPU → Metal.

Both forward rendering and the advanced lighting path render a race. The
lighting check used `--enable-dynamic-lights --race-now --track=lighthouse
--numkarts=1 --laps=3`, with screen-space reflections disabled and the render
target scale set to 1.0. The kart headlights illuminate the road and fence.
MRT outputs, sampled read-only depth and input attachments remain on the GPU.
An independent Vulkan contract checks every pixel after three subpasses,
including intermediate attachments with `DONT_CARE` final store operations.

This check does not establish all shadow/reflection settings, resize stability
or a performance baseline. Advanced lighting is not yet implemented by the
native Scarlet/VirGL backend.
