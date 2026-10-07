#!/usr/bin/env python3
"""Build Linux-ABI ICD and plugins together; stage a rootfs without deploying it.

Run in an AArch64 Linux build environment with Wayland development tools and
libsws_client_c on the linker path. Native ELFOSABI_SCARLET plugins are separate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess

TARGET = "aarch64-unknown-linux-gnu"
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maxwell-source", type=Path, help="scarlet-project-switch checkout")
    parser.add_argument("--sws-library", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "build.json"
    report_path.write_text(json.dumps({"result": "BUILDING"}) + "\n")
    env = os.environ.copy()
    for key in ("CARGO_ENCODED_RUSTFLAGS", "CARGO_UNSTABLE_BUILD_STD",
                "CARGO_UNSTABLE_BUILD_STD_FEATURES"):
        env.pop(key, None)
    env["CARGO_TARGET_DIR"] = str(output / "target-sgfx")
    env["RUSTFLAGS"] = env.get("RUSTFLAGS", "") + " -Zdefault-visibility=hidden"
    common = ["--release", "--target", TARGET]
    if args.offline:
        common.append("--offline")
    commands = []

    def run(command, cwd=ROOT):
        commands.append([str(item) for item in command])
        subprocess.run(commands[-1], cwd=cwd, env=env, check=True)

    run(["cargo", "build", "--locked", *common, "-p", "vulkan-sgfx", "--lib",
         "--no-default-features", "--features", "scarlet-wsi"])
    run(["cargo", "build", "--locked", *common, "-p", "sgfx-backend-scarlet-virgl-plugin", "--lib"])
    release = output / "target-sgfx" / TARGET / "release"
    rootfs = output / "rootfs"
    drivers = rootfs / "usr/lib/sgfx"
    libraries = rootfs / "usr/lib/aarch64-linux-gnu"
    manifests = rootfs / "usr/share/vulkan/icd.d"
    for directory in (drivers, libraries, manifests):
        directory.mkdir(parents=True, exist_ok=True)
    if not args.maxwell_source:
        for name in ("libsgfx_scarlet_maxwell.so", "scarlet-maxwell.sgfx-driver"):
            (drivers / name).unlink(missing_ok=True)
    artifacts = []

    def record(path, **details):
        artifacts.append({"path": str(path.relative_to(rootfs)),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **details})

    def stage(source, destination):
        data = source.read_bytes()
        if (data[:8] != b"\x7fELF\x02\x01\x01\x00"
                or struct.unpack_from("<HH", data, 16) != (3, 183)):
            raise RuntimeError(f"expected Linux ELF64 AArch64 shared library: {source}")
        shutil.copy2(source, destination)
        destination.chmod(0o755)
        symbols = subprocess.check_output(["readelf", "--dyn-syms", "-W", str(source)], text=True)
        if source.name.startswith("libsgfx_"):
            for entry in ("sgfx_backend_get_api_v2", "sgfx_backend_get_driver_api_v2"):
                if not any(line.split()[-1:] == [entry] and " UND " not in line for line in symbols.splitlines()):
                    raise RuntimeError(f"missing public C ABI v2 entry {entry}: {source}")
        if source.name == "libvulkan_sgfx.so" and b"sgfx_backend_get_api_v2" not in data:
            raise RuntimeError("ICD does not contain dynamic SGFX discovery")
        record(destination, dynamic=subprocess.check_output(["readelf", "-dW", str(source)], text=True))

    stage(release / "libvulkan_sgfx.so", libraries / "libvulkan_sgfx.so")
    stage(args.sws_library, libraries / "libsws_client_c.so")
    stage(release / "libsgfx_scarlet_virgl.so", drivers / "libsgfx_scarlet_virgl.so")
    (drivers / "scarlet-virgl.sgfx-driver").write_text(
        "abi=2\nname=scarlet-virgl\ngpu_backend=virtio-gpu\nlibrary=libsgfx_scarlet_virgl.so\n")
    record(drivers / "scarlet-virgl.sgfx-driver")
    (manifests / "sgfx.json").write_text(json.dumps({"file_format_version": "1.0.0", "ICD": {
        "library_path": "/usr/lib/aarch64-linux-gnu/libvulkan_sgfx.so", "api_version": "1.0.0"}}, indent=2) + "\n")
    record(manifests / "sgfx.json")
    backends = ["virtio-gpu"]
    if args.maxwell_source:
        # Patches make the plugin consume exactly the ABI/core used by this ICD.
        # Resolve its lockfile only in an isolated source copy, never in the caller's checkout.
        switch = output / "maxwell-source"
        for name in ("userspace", "shared"):
            shutil.copytree(args.maxwell_source.resolve() / name, switch / name,
                            ignore=shutil.ignore_patterns("target", ".git"), dirs_exist_ok=True)
        plugin = switch / "userspace/sgfx-backend-scarlet-maxwell-plugin"
        config = output / "maxwell-patches.toml"
        config.write_text('[patch."https://github.com/petitstrawberry/sgfx"]\n' + ''.join(
            f'{name} = {{ path = {json.dumps(str(ROOT / "crates" / name))} }}\n'
            for name in ("sgfx-core", "sgfx-backend-abi", "sgfx-codegen-virgl")))
        selected = ["--config", str(config), "--manifest-path", str(plugin / "Cargo.toml")]
        # A path override changes source identities only in this copied lockfile.
        run(["cargo", "build", *common, *selected, "--lib"], cwd=plugin)
        stage(release / "libsgfx_scarlet_maxwell.so", drivers / "libsgfx_scarlet_maxwell.so")
        shutil.copy2(plugin / "scarlet-maxwell.sgfx-driver", drivers / "scarlet-maxwell.sgfx-driver")
        record(drivers / "scarlet-maxwell.sgfx-driver")
        backends.append("nvidia-gm20b")
    if args.test:
        for package in ("sgfx-backend-abi", "sgfx-backend-loader", "vulkan-sgfx"):
            extra = ["--no-default-features", "--features", "scarlet-wsi"] if package == "vulkan-sgfx" else []
            run(["cargo", "test", "--locked", *common, "-p", package, "--lib", *extra])
        for backend in backends:
            run(["cargo", "run", "--locked", *common, "-p", "sgfx-backend-loader",
                 "--example", "probe_driver", "--", str(drivers), backend])
        run(["python3", ROOT / "crates/vulkan-sgfx/tools/probe_linux_icd.py",
             libraries / "libvulkan_sgfx.so"])
    report_path.write_text(json.dumps({"result": "PASS", "target": TARGET,
        "backends": backends, "artifacts": artifacts, "commands": commands,
        "gpu_execution_tested": False}, indent=2) + "\n")
    print(f"Linux ICD and ABI v2 plugins staged in {rootfs}")


if __name__ == "__main__":
    main()
