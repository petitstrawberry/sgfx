# VirGL control workload

This directory measures the same small offscreen drawing workload on two guests:

* `linux_gles.c` uses Linux DRM/GBM, Mesa EGL and OpenGL ES 3, and refuses any
  renderer whose `GL_RENDERER` does not contain `virgl`.
* `src/main.rs` submits SGFX fixed-function IR through
  `sgfx-backend-scarlet-virgl` to Scarlet's `/dev/gpu0`. Run the resulting
  AArch64 Linux binary with Scarlet's Linux ABI (`abi-run`).

Both use a 1280x800 target by default, clear it, and draw the same triangle
1200 times per frame. The solid color changes every 20 draws. The benchmark
warms up for 20 frames, measures 120 frames, waits for GPU completion each
frame, then reads back and checks center and left-edge pixels. A failed
readback makes the run fail. Width, height, draws, frames, warmup, and color
change interval can be set with `--width`, `--height`, `--draws`, `--frames`,
`--warmup`, and `--uniform-every` on both executables.

Build the Linux control in a Linux guest with matching Mesa development
headers and runtime libraries:

```sh
cc -O3 -std=c11 -Wall -Wextra linux_gles.c -lEGL -lGLESv2 -lgbm -o linux_gles
./linux_gles --draws 0 --frames 120 --warmup 50
./linux_gles --draws 1 --frames 120 --warmup 50
./linux_gles --draws 1200 --frames 120 --warmup 50
```

Build the Scarlet executable in an AArch64 Linux environment from this SGFX
checkout, using the standalone lockfile and release profile:

```sh
cargo build --release --locked --manifest-path benchmarks/virgl-compare/Cargo.toml
```

Place `benchmarks/virgl-compare/target/release/sgfx-virgl-compare` in the
Scarlet Linux ABI root filesystem, for example at `/usr/bin/sgfx-virgl-compare`.
Then run the matching cases in Scarlet's shell:

```sh
abi-run linux-aarch64 /usr/bin/sgfx-virgl-compare --draws 0 --frames 120 --warmup 50
abi-run linux-aarch64 /usr/bin/sgfx-virgl-compare --draws 1 --frames 120 --warmup 50
abi-run linux-aarch64 /usr/bin/sgfx-virgl-compare --draws 1200 --frames 120 --warmup 50
```

Compare `RESULT` lines at the same resolution and draw count. `fps` is measured
frames divided by the sum of their durations; `median_ms` and `p95_ms` are
per-frame latencies. Linux `calls_ms` is CPU time issuing GL calls, and
`finish_ms` is the `glFinish` wait. Scarlet `record_ms` builds SGFX IR,
`admission_ms` submits it, and `completion_ms` waits for completion. These
phases have different boundaries, so compare total frame time first. Use a
single VM at a time and a stable QEMU/HVF, host GPU, Mesa, and resolution
configuration. The pixel hash is diagnostic; image row order and channel order
may differ between the two paths. Run each case twice and report both results:
the first run can be materially slower even after the in-process warmup.

This is a control for draw submission and simple rasterization. It does not
include presentation, depth, textures, SPIR-V, Vulkan API translation, or the
vkQuake workload. A gap here localizes a basic SGFX/Scarlet/transport cost;
similar results here do not rule out costs in those omitted parts.
