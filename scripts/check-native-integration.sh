#!/bin/sh
# Check a coordinated SGFX/Adreno source set without changing release manifests
# or the workspace lockfile. Requires the Scarlet nightly Rust toolchain.
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: $0 /path/to/scarlet-project-chromebook" >&2
    exit 2
fi

sgfx_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
adreno_root=$(CDPATH= cd -- "$1" && pwd)
integration_dir="$sgfx_root/target/native-integration"
mkdir -p "$integration_dir"

python3 - "$adreno_root" "$integration_dir/config.toml" <<'PY'
import json
from pathlib import Path
import sys

source = Path(sys.argv[1])
packages = {
    "sgfx-backend-scarlet-adreno": source / "userspace/sgfx-backend-scarlet-adreno",
    "sgfx-codegen-adreno-a6xx": source / "userspace/sgfx-codegen-adreno-a6xx",
}
for directory in packages.values():
    if not (directory / "Cargo.toml").is_file():
        raise SystemExit(f"missing package manifest: {directory / 'Cargo.toml'}")
lines = ['[patch."https://github.com/petitstrawberry/scarlet-project-chromebook.git"]']
lines.extend(f"{name} = {{ path = {json.dumps(str(path))} }}" for name, path in packages.items())
Path(sys.argv[2]).write_text("\n".join(lines) + "\n")
PY

cp "$sgfx_root/Cargo.lock" "$integration_dir/Cargo.lock"
cd "$sgfx_root"
for native_target in riscv64gc-unknown-scarlet aarch64-unknown-scarlet; do
    cargo tree -Z unstable-options --config "$integration_dir/config.toml" \
        --lockfile-path "$integration_dir/Cargo.lock" \
        -p sgfx --target "$native_target" --invert sgfx-core
    cargo check -Z unstable-options --config "$integration_dir/config.toml" \
        --lockfile-path "$integration_dir/Cargo.lock" --locked \
        -p sgfx --target "$native_target"
done
