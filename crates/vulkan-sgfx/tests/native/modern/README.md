# Native Vulkan rendering smoke test

Compile `ApplicationsSmoke.java` against native LWJGL 3.4.3 module JARs and run
with `VK_DRIVER_FILES` selecting the freshly built SGFX ICD. Required providers
are LWJGL core, SDL, Vulkan and Shaderc, a Vulkan loader and the SGFX backend DSO.
The native providers and VM runner are produced by `scarlet-bundle-lwjgl`.

The test uses no network or application assets. It checks synchronization2,
events, binary/timeline semaphores, dynamic rendering, indirect-count drawing,
indexed indirect drawing with a negative vertex offset and exact image readback.
Occlusion checks compare visible/empty scopes, copied and host results, and host
reset. Pipeline-statistics-capable devices also check IA, VS and fragment counts;
other devices must reject statistics pools.
The indirect/count buffers are updated in the command stream. Timestamp-capable
devices additionally check timer results, query buffer copies and host reset.
Devices reporting zero timestamp bits must reject timestamp pools, and rendering
continues without timing. These checks do not establish Vulkan conformance.

Expected final markers are `MODERN_GRAPHICS_NATIVE PASS`, `SYNC2_NATIVE PASS`,
`SCOPED_QUERIES_NATIVE PASS`, and either `QUERY_NATIVE PASS` or
`QUERY_NATIVE_UNAVAILABLE PASS`.

On macOS, `host-gl-clock.c` checks OpenGL timestamp counter bits and query values
without QEMU or VirGL. Build it with `clang -framework OpenGL host-gl-clock.c`.
A zero bit count means that OpenGL exposes no usable timestamp counter, even
when query objects complete successfully. This diagnostic does not use CPU time
as a replacement GPU counter.
