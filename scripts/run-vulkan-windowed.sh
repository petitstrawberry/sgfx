#!/bin/sh
# Build and run the ordinary Vulkan WSI cube through the system Khronos loader.
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_dir"

if [ "$(uname -s)" != Darwin ]; then
    echo "The SGFX VK_EXT_metal_surface implementation requires macOS." >&2
    exit 1
fi
for tool in cargo rustc python3; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "$tool is required on PATH." >&2
        exit 1
    fi
done

loader_path=
if [ -n "${VULKAN_SDK:-}" ] && [ -f "$VULKAN_SDK/lib/libvulkan.dylib" ]; then
    loader_path=$VULKAN_SDK/lib/libvulkan.dylib
else
    for candidate in /opt/homebrew/lib/libvulkan.dylib /usr/local/lib/libvulkan.dylib /nix/store/*-vulkan-loader-*/lib/libvulkan.dylib; do
        if [ -f "$candidate" ]; then
            loader_path=$candidate
            break
        fi
    done
fi
if [ -z "$loader_path" ]; then
    echo "A Khronos Vulkan loader was not found." >&2
    exit 1
fi

host_target=$(rustc -vV | sed -n 's/^host: //p')
cargo build --locked -p vulkan-sgfx --lib --example windowed --target "$host_target"
target_dir=$(cargo metadata --locked --format-version=1 --no-deps | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')
profile_dir=$target_dir/$host_target/debug
manifest=$profile_dir/sgfx-windowed-icd.json
python3 crates/vulkan-sgfx/tools/write_icd_manifest.py \
    "$profile_dir/libvulkan_sgfx.dylib" "$manifest" >/dev/null

loader_dir=$(dirname "$loader_path")
export DYLD_LIBRARY_PATH="$loader_dir${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export VK_DRIVER_FILES=$manifest
exec "$profile_dir/examples/windowed" "$@"
