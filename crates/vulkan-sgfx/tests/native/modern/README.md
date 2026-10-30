# Native Vulkan rendering smoke test

Compile `ApplicationsSmoke.java` against native LWJGL 3.4.3 module JARs and run
with `VK_DRIVER_FILES` selecting the freshly built SGFX ICD. Required providers
are LWJGL core, SDL, Vulkan and Shaderc, a Vulkan loader and the SGFX backend DSO.
The native providers and VM runner are produced by `scarlet-bundle-lwjgl`.

The test uses no network or application assets. It checks synchronization2,
events, binary/timeline semaphores, dynamic rendering, indirect-count drawing,
indexed indirect drawing with a negative vertex offset and exact image readback.
The combined graphics path checks push descriptors, anisotropy 16, instance
divisor two, firstInstance two, signed BaseVertex and DrawIndex after a skipped
zero-count indirect command. It verifies 320 color pixels and 320 depth words
across triangle/indexed/point and Min/Max blending passes, including explicit
sampling gradients, packed signed/normalized inputs, angle conversion, dynamic
local-array stores and a signed texel view with byte offset 16.
Separate checks validate D32 upload/copy/clear, vertex-only depth rendering with
static depth bias and preserved texels, plus RGBA16F/RGBA32F additive blending.
The floating-point test compares 128 exact packed components, including negative
values and values above one. A mip-one attachment test compares 80 exact pixels
after a partial clear, LOAD and an in-pass clear/resume. Mip zero and pixels
outside the render area must retain their original values. A capacity case
holds 1100 real GPU images and completion observers, reads boundary pixels,
retires every image and repeats to verify slot reuse.
Occlusion checks compare visible/empty scopes, copied and host results, and host
reset. Pipeline-statistics-capable devices also check IA, VS and fragment counts;
other devices must reject statistics pools.
The aggregate Vulkan 1.2 feature query must agree with individual timeline/query
records and clear unsupported fields. Device creation enables the implemented
features through the aggregate record before the GPU operations run.
The indirect/count buffers are updated in the command stream. Timestamp-capable
devices additionally check timer results, query buffer copies and host reset.
Devices reporting zero timestamp bits must reject timestamp pools, and rendering
continues without timing. These checks do not establish Vulkan conformance.

Expected final markers are `MODERN_GRAPHICS_NATIVE PASS`, `SYNC2_NATIVE PASS`,
`SCOPED_QUERIES_NATIVE PASS`, `CORE11_PROPERTIES_NATIVE PASS`,
`CORE12_FEATURES_NATIVE PASS`, `VULKAN_REQUIREMENTS_NATIVE PASS`,
`DEPTH_TRANSFER_NATIVE PASS`, `DEPTH_ONLY_NATIVE PASS`, `FLOAT_COLOR_NATIVE PASS`,
`MIP_ATTACHMENT_NATIVE PASS`, `GPU_HANDLES_NATIVE PASS`, and either `QUERY_NATIVE PASS` or
`QUERY_NATIVE_UNAVAILABLE PASS`.

On macOS, `host-gl-clock.c` checks OpenGL timestamp counter bits and query values
without QEMU or VirGL. Build it with `clang -framework OpenGL host-gl-clock.c`.
A zero bit count means that OpenGL exposes no usable timestamp counter, even
when query objects complete successfully. This diagnostic does not use CPU time
as a replacement GPU counter.

For Cocoa/HVF runs include virtio keyboard and mouse devices. A Cocoa input
callback in the QEMU host can dereference absent keyboard state if an event
arrives in a GPU-only VM; that host failure must not be counted as a passing run.
