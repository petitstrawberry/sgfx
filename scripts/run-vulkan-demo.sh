#!/bin/sh
# Build a fresh SGFX ICD and save a real-loader GPU render. No network installs.
set -eu

if [ "$#" -gt 1 ]; then
    echo "usage: $0 [OUTPUT.png]" >&2
    exit 2
fi
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_dir"
output=${1:-target/vulkan-demo.png}
for tool in cargo rustc python3; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "$tool is required on PATH; see docs/vulkan-sgfx.md" >&2
        exit 1
    fi
done

case $(uname -s) in
    Darwin)
        library_name=libvulkan_sgfx.dylib
        loader_name=libvulkan.1.dylib
        candidates="/opt/homebrew/lib/$loader_name /usr/local/lib/$loader_name"
        ;;
    Linux)
        library_name=libvulkan_sgfx.so
        loader_name=libvulkan.so.1
        candidates="/usr/lib/aarch64-linux-gnu/$loader_name /usr/lib/x86_64-linux-gnu/$loader_name /usr/lib64/$loader_name /usr/lib/$loader_name /lib/aarch64-linux-gnu/$loader_name /lib/x86_64-linux-gnu/$loader_name"
        # Force a software GL driver with no GL-to-Vulkan recursion through Zink.
        export GALLIUM_DRIVER=llvmpipe LIBGL_ALWAYS_SOFTWARE=true EGL_PLATFORM=surfaceless
        ;;
    *) echo "This demo runner supports macOS and Linux." >&2; exit 1 ;;
esac
if [ -z "${SGFX_VULKAN_LOADER:-}" ]; then
    if [ -n "${VULKAN_SDK:-}" ] && [ -f "$VULKAN_SDK/lib/$loader_name" ]; then
        SGFX_VULKAN_LOADER=$VULKAN_SDK/lib/$loader_name
    else
        # Ordinary system paths above contain no whitespace; quote every use.
        for candidate in $candidates /nix/store/*-vulkan-loader-*/lib/"$loader_name"; do
            if [ -f "$candidate" ]; then
                SGFX_VULKAN_LOADER=$candidate
                break
            fi
        done
    fi
fi
if [ -z "${SGFX_VULKAN_LOADER:-}" ] || [ ! -f "$SGFX_VULKAN_LOADER" ]; then
    echo "A Vulkan loader was not found. Set SGFX_VULKAN_LOADER to its library path." >&2
    exit 1
fi
export SGFX_VULKAN_LOADER
host_target=$(rustc -vV | sed -n 's/^host: //p')
if [ -z "$host_target" ]; then
    echo "Could not determine the Rust host target." >&2
    exit 1
fi
# Explicit host selection avoids inheriting a Scarlet cross-compilation target.
cargo build --locked -p vulkan-sgfx --lib --example render_demo --target "$host_target"
target_dir=$(cargo metadata --locked --format-version=1 --no-deps | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')
profile_dir=$target_dir/$host_target/debug
manifest=$profile_dir/sgfx-demo-icd.json
python3 crates/vulkan-sgfx/tools/write_icd_manifest.py "$profile_dir/$library_name" "$manifest"
export VK_DRIVER_FILES=$manifest
"$profile_dir/examples/render_demo" "$output"
