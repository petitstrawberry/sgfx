# OpenTTD / Zink integration branch

Base: `a1115c4201d0cdd8aa43bd29a8a9c1551d162d0a` plus the existing Scarlet
Linux/Wayland presentation change (`725ee3c` on this branch).
This branch preserves the coordinated SGFX changes used by the integration;
it does not declare a new Vulkan conformance level.

Changes include Zink feature/API bootstrap, timeline and render-pass handling,
descriptor updates, combined sampler/component-packed SPIR-V lowering,
PointSize/FrontFacing translation, R8/RG8 and logical 1D textures, regional
transfer/readback, vertex/uniform fetch bounds, TriangleFan lowering, and
retirement of textures/bind groups/native VirGL objects.
All SGFX-owned consumers use the same workspace core IR. The root Git-source
patch is a workspace development override; external downstream users must pin
a coordinated version explicitly. Other native GPU backends were not exercised.

## Evidence and limits

The pre-publication integration ran on ARM64 QEMU/HVF on a Mac, using Scarlet's
Linux ABI and native Wayland bridge. OpenTTD 15.3 used SDL 2.30.0 Wayland/EGL,
Mesa 25.0.7 Zink/Kopper, and SGFX Vulkan -> native VirGL. The captured renderer
was `zink Vulkan 1.0(SGFX Vulkan (Scarlet VirGL GPU 0) (Driver Unknown))`, GL 2.1
and GLSL 1.20. Driver/device evidence and pixel readback distinguish this from
softpipe/llvmpipe fallback. ICD SHA256 for that actual guest run:
`5125b80eba72daa99299998da11cba9a6abb9995c3f6c71b6f2411425b61ab92`.

That run passed 242 core/native/Vulkan tests, GPU clear/triangle/texture and
palette/readback probes, and 18,302 OpenTTD frames before clean exit. Subsequent
bridge decoration validation passed launch, input, map scrolling, move, resize,
maximize/restore and close. This is emulated VirGL evidence; physical Scarlet
hardware and other Vulkan devices have not been tested.

The publishing copy was checked again on macOS: core/default codegen tests,
programmable codegen tests (38 passed, one external harness test ignored),
and Vulkan library tests (76 passed); formatting and diff checks passed.
Strict Clippy is not green: it reports `collapsible_if` in the existing codegen
return-guard code. Full GitHub CI/native backend matrix was not rerun.

OpenTTD fullscreen remains a known failure: a legal signed viewport extends
outside the attachment and is rejected by the current IR/backend attachment
bounds rule, leading to DEVICE_LOST. The bridge's independent SHM fullscreen
transition passes. Fix viewport validation/scissor behavior with GPU regression
coverage before claiming OpenTTD fullscreen support. Several unimplemented
Zink capability warnings also remain; this branch is an integration checkpoint.

SPIR-V test capture provenance and licenses are documented next to the fixtures.
Reproduction recipes and the bridge live tests are in the corresponding
`petitstrawberry/Scarlet` `codex/openttd-zink-wayland` branch.
