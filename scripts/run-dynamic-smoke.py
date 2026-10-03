#!/usr/bin/env python3
"""Run the SGFX dynamic/static comparison in a disposable Scarlet QEMU guest."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True


def benchmark_comparison(serial):
    """Preserve round medians and pair each dynamic run with its static run."""
    round_number = None
    samples = []
    for line in serial.splitlines():
        fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
        if line.startswith("SGFX_ROUND "):
            round_number = int(fields["round"])
        elif line.startswith("SGFX_BENCH "):
            samples.append({"round": round_number, **{
                key: value if key == "mode" else int(value)
                for key, value in fields.items()
            }})
    comparisons = []
    for draws in sorted({s["draws"] for s in samples}):
        ratios = {metric: [] for metric in ("record_ns", "submit_ns", "total_ns", "frame_ns")}
        for number in sorted({s["round"] for s in samples if s["round"] is not None}):
            pair = {s["mode"]: s for s in samples if s["draws"] == draws and s["round"] == number}
            if not {"static", "dynamic"}.issubset(pair):
                continue
            for metric in ratios:
                if pair["static"].get(metric, 0) > 0 and metric in pair["dynamic"]:
                    ratios[metric].append(pair["dynamic"][metric] / pair["static"][metric])
        comparisons.append({"draws": draws, "paired_rounds": len(ratios["submit_ns"]),
                            "median_dynamic_over_static": {
                                key: statistics.median(values) for key, values in ratios.items() if values
                            }})
    return {"samples": samples, "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("aarch64", "riscv64"), default="aarch64")
    parser.add_argument("--scarlet", type=Path, default=Path(__file__).resolve().parents[2] / "Scarlet")
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qemu", type=Path, help="explicit QEMU binary, e.g. an offscreen Cocoa build on macOS")
    parser.add_argument("--display", default="cocoa,gl=on" if sys.platform == "darwin" else "egl-headless")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--accel", choices=("tcg", "hvf"), default="tcg")
    args = parser.parse_args()
    source = args.scarlet.resolve() / "tools/loader-smoke/run-qemu.py"
    prepare = [sys.executable, str(source), "--arch", args.arch, "--kernel", str(args.kernel.resolve()),
               "--staging", str(args.staging.resolve()), "--output", str(args.output.resolve()), "--prepare-only"]
    subprocess.run(prepare, check=True)
    output = args.output.resolve()
    build = args.staging.resolve().parent / "build.json"
    if build.is_file():
        metadata = json.loads(build.read_text())
        # A diagnostic staging tree may replace a fixture beside a normal
        # build.json. Never attribute that fixture to the normal build audit.
        if metadata.get("result") == "PASS" and all(
            (args.staging / name).is_file() and hashlib.sha256(
                (args.staging / name).read_bytes()).hexdigest() == artifact["sha256"]
            for name, artifact in metadata["artifacts"].items()
        ):
            shutil.copyfile(build, output / "input-build.json")
    spec = importlib.util.spec_from_file_location("scarlet_loader_smoke", source)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.SUCCESS = re.compile(rb"\nSGFX_DYNAMIC_ALL_OK\r?\n")
    # Bootstrap console writes one byte per syscall. Wait for the complete
    # diagnostic instead of terminating at the first bytes of a panic message.
    runner.PANIC = re.compile(rb"SGFX_PANIC_END\r?\n|kernel panic[^\n]*\n", re.IGNORECASE)
    commands = json.loads((output / "commands.json").read_text())
    image_command = commands["image"]
    image_command[image_command.index("--cmdline") + 1] += " init.exec=/bin/sgfx-dynamic-smoke init.console=/dev/tty0"
    with (output / "image-build.log").open("ab") as log:
        subprocess.run(image_command, stdout=log, stderr=subprocess.STDOUT, check=True)
    command = commands["qemu"]
    if args.qemu:
        command[0] = str(args.qemu.resolve(strict=True))
    if args.accel == "hvf":
        if args.arch != "aarch64" or sys.platform != "darwin":
            parser.error("HVF requires AArch64 on macOS")
        command[command.index("-accel") + 1] = "hvf"
        command[command.index("-cpu") + 1] = "host"
        code = runner.firmware(("SCARLET_EFI_CODE_ARM64_HVF", "SCARLET_EFI_CODE_ARM64"))
        variables = runner.firmware(("SCARLET_EFI_VARS_ARM64_HVF", "SCARLET_EFI_VARS_ARM64"))
        shutil.copyfile(variables, output / "efi-vars.fd")
        for index, value in enumerate(command):
            if value.startswith("if=pflash,format=raw,unit=0,"):
                command[index] = f"if=pflash,format=raw,unit=0,file={runner.qemu_filename(code)},readonly=on"
    command[command.index("-display") + 1] = args.display
    for index, value in enumerate(command):
        if value.startswith("virtio-gpu-device,"):
            command[index] = value.replace("virtio-gpu-device,", "virtio-gpu-gl-device,", 1)
    commands["qemu"] = command
    (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    result = runner.run_guest(command, output, args.timeout)
    result["qemu_binary"] = {"path": shutil.which(command[0]), "sha256": hashlib.sha256(
        Path(shutil.which(command[0])).read_bytes()).hexdigest()}
    result["initramfs_sha256"] = hashlib.sha256((output / "initramfs.cpio").read_bytes()).hexdigest()
    serial = (output / "serial.log").read_text(errors="replace")
    result["checks"] = [line for line in serial.splitlines() if line.startswith("SGFX_")]
    result["benchmarks"] = benchmark_comparison(serial)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
