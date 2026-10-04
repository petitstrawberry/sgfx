#!/usr/bin/env python3
"""Build and stage the native SGFX plugin and identical static/dynamic GPU probes.

Only an isolated generated target is changed. No installed sysroot or toolchain
is edited. The resulting /init needs Scarlet's resident scarlet-ld interpreter.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

sys.dont_write_bytecode = True


def source_sysroot(output, rustc, source):
    """An isolated sysroot view; never replace files in the installed toolchain."""
    original = Path(subprocess.check_output([rustc, "--print", "sysroot"], text=True).strip())
    view = output / "sysroot"
    for relative in (Path(), Path("lib"), Path("lib/rustlib")):
        (view / relative).mkdir(parents=True, exist_ok=True)
        excluded = {Path(): "lib", Path("lib"): "rustlib", Path("lib/rustlib"): "src"}[relative]
        for entry in (original / relative).iterdir():
            link = view / relative / entry.name
            if entry.name != excluded and not link.exists():
                link.symlink_to(entry, target_is_directory=entry.is_dir())
    (view / "lib/rustlib/src").mkdir(exist_ok=True)
    link = view / "lib/rustlib/src/rust"
    if not link.is_symlink() or link.resolve() != source:
        if link.is_symlink():
            link.unlink()
        link.symlink_to(source, target_is_directory=True)
    wrapper = output / "rustc-with-sources"
    content = f"#!{sys.executable}\nimport os,sys\nos.execv({rustc!r}, [{rustc!r}, '--sysroot', {str(view)!r}, *sys.argv[1:]])\n"
    if not wrapper.exists() or wrapper.read_text() != content:
        wrapper.write_text(content)
    wrapper.chmod(0o755)
    return str(wrapper)


def audit(staging, scarlet, arch):
    source = scarlet / "tools/elf_audit.py"
    spec = importlib.util.spec_from_file_location("scarlet_elf_audit", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reports = []
    libraries = [(str(path.relative_to(staging)), set(), None)
                 for path in sorted((staging / "system/lib/sgfx").glob("*.so"))]
    for name, expected_imports, interpreter in [*libraries,
        ("bin/sgfx-dynamic-smoke", {"dlopen", "dlsym", "dlerror"}, "/bin/scarlet-ld"),
        ("bin/sgfx-static-smoke", set(), None),
        ("bin/scarlet-ld", set(), None),
        ("init", set(), None),
    ]:
        elf = module.Elf(staging / name)
        report = elf.report()
        imports = {s["name"] for s in report["undefined_relocated_symbols"]}
        if imports != expected_imports or report["needed"] or report["interpreter"] != interpreter:
            raise RuntimeError(f"unexpected ELF linkage: {name}: {report}")
        if report["machine"] != arch or report["osabi"] != 83:
            raise RuntimeError(f"unexpected ELF ABI: {name}")
        allowed = {"NONE", "RELATIVE", "JUMP_SLOT", "GLOB_DAT", "ABS64", "64"}
        if (report["tls"] or report["relr"] or report["textrel"] or report["symbol_versioning"]
                or set(report["relocations"]) - allowed):
            raise RuntimeError(f"unsupported loader requirements: {name}: {report}")
        if name.endswith(".so"):
            # Both hash styles are requested. DT_HASH's nchain gives a bounded
            # symbol count even for stripped artifacts.
            _, count = elf.unpack("II", elf.at_vaddr(elf.tag(4), 8))
            exports = []
            for index in range(1, count):
                symbol, info, _, section, _, _ = elf.unpack("IBBHQQ", elf.at_vaddr(elf.tag(6) + index * 24, 24))
                if section and info >> 4 in (1, 2):
                    exports.append(elf.dynstring(symbol))
            expected_exports = {"sgfx_backend_get_api_v2", "sgfx_backend_get_driver_api_v2"}
            if name.endswith("libsgfx_scarlet_maxwell.so"):
                expected_exports.add("sgfx_backend_get_ycbcr_api_v2")
            if set(exports) != expected_exports or report["entry"] != 0:
                raise RuntimeError(f"backend exports executable/Rust internals: {exports}")
            report["exports"] = exports
        report["path"] = name
        reports.append(report)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("aarch64", "riscv64"), default="aarch64")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scarlet", type=Path, default=Path(__file__).resolve().parents[2] / "Scarlet")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--maxwell", type=Path,
                        help="also build/stage the Maxwell plugin from this Switch checkout (VirGL smoke still runs on VirtIO)")
    parser.add_argument("--rust-source", type=Path, help="matching Rust source tree with Scarlet's DSO-safe std TLS runtime")
    args = parser.parse_args()
    if args.maxwell and args.arch != "aarch64":
        parser.error("the Switch Maxwell plugin requires --arch aarch64")
    repo = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "build.json").write_text(json.dumps({"result": "BUILDING"}) + "\n")
    original = "aarch64-unknown-scarlet" if args.arch == "aarch64" else "riscv64gc-unknown-scarlet"
    rustc = shutil.which("rustc")
    cargo = shutil.which("cargo")
    installed = Path(subprocess.check_output([rustc, "--print", "sysroot"], text=True).strip())
    source = args.rust_source.resolve() if args.rust_source else installed / "lib/rustlib/src/rust"
    tls_source = source / "library/std/src/sys/thread_local/key/scarlet.rs"
    tls_text = tls_source.read_text()
    if "static MAIN_TLS_BASE" in tls_text and "static NEXT_KEY" in tls_text:
        raise RuntimeError("this std has colliding per-DSO TLS keys; use --rust-source with the matching namespace fix (scarlet-rust-nix PR #25), or an updated toolchain")
    if args.rust_source:
        rustc = source_sysroot(output, rustc, source)
    spec = json.loads(subprocess.check_output([rustc, "--print", "target-spec-json", "-Zunstable-options", "--target", original], text=True))
    # Rust implementation symbols belong to this object. Tell LLVM before LTO,
    # not just the linker afterwards: only C-exported ABI entry points may be
    # interposed. The same setting is used for both comparison executables.
    spec.update({"dynamic-linking": True, "relocation-model": "pic", "dll-prefix": "lib", "dll-suffix": ".so",
                 "default-visibility": "hidden"})
    spec.setdefault("pre-link-args", {}).setdefault("gnu-lld", []).extend(["-z", "max-page-size=4096", "-z", "now", "--hash-style=both"])
    crt_source = source / f"library/rtstartup/scarlet-{args.arch}.S"
    if not crt_source.exists():
        raise RuntimeError("ThinLTO cdylibs require Scarlet's executable-only CRT split; use matching Rust sources with the native-host runtime fixes")
    if crt_source.exists():
        crt = output / "scarlet-crt0.o"
        temporary_crt = output / "scarlet-crt0.tmp.o"
        flags = [] if args.arch == "aarch64" else ["-march=rv64gc", "-mabi=lp64d"]
        subprocess.run([os.environ.get("TARGET_CC", "clang"), f"--target={args.arch}-unknown-none", *flags,
                        "-w", "-c", str(crt_source), "-o", str(temporary_crt)], check=True, capture_output=True)
        if not crt.exists() or crt.read_bytes() != temporary_crt.read_bytes():
            temporary_crt.replace(crt)
        else:
            temporary_crt.unlink()
        spec["pre-link-objects"] = {kind: [str(crt)] for kind in (
            "dynamic-nopic-exe", "dynamic-pic-exe", "static-nopic-exe", "static-pic-exe")}
    target = output / f"sgfx-dynamic-{args.arch}.json"
    target_text = json.dumps(spec, indent=2) + "\n"
    if not target.exists() or target.read_text() != target_text:
        target.write_text(target_text)
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(output / "cargo")
    env["RUSTFLAGS"] = ""
    env["RUSTC"] = rustc
    # Apply the same production optimization to the plugin and both comparison
    # programs. ThinLTO can inline the borrowed decoder across crate boundaries.
    env["CARGO_PROFILE_RELEASE_LTO"] = "thin"
    env["CARGO_PROFILE_RELEASE_CODEGEN_UNITS"] = "1"
    commands = []

    def run(arguments, cwd=repo):
        command = [str(v) for v in arguments]
        commands.append(command)
        subprocess.run(command, env=env, cwd=cwd, check=True)

    common = ["--release", "--locked", "--target", str(target), "-Zbuild-std=std,panic_abort", "-Zbuild-std-features=compiler-builtins-mem"]
    if args.offline:
        common.append("--offline")
    staging = output / "staging"
    drivers = staging / "system/lib/sgfx"
    drivers.mkdir(parents=True, exist_ok=True)
    (staging / "bin").mkdir(parents=True, exist_ok=True)
    (staging / "system/bin").mkdir(parents=True, exist_ok=True)
    for directory in ("dev/pts", "mnt/newroot", "root", "etc", "tmp"):
        (staging / directory).mkdir(parents=True, exist_ok=True)
    release = output / "cargo" / target.stem / "release"

    def stage(source, destination):
        data = source.read_bytes()
        machine = 183 if args.arch == "aarch64" else 243
        if data[:7] != b"\x7fELF\x02\x01\x01" or struct.unpack_from("<H", data, 18)[0] != machine:
            raise RuntimeError(f"not the expected ELF64 architecture: {source}")
        if data[7] != 83:
            raise RuntimeError(f"compiler did not mark ELFOSABI_SCARLET: {source}; use a toolchain with native GNU ELF tag support")
        destination.write_bytes(data)
        destination.chmod(0o755)

    run([cargo, "rustc", "-p", "sgfx-backend-scarlet-virgl-plugin", "--lib", *common,
         "--", "-C", "link-arg=-soname", "-C", "link-arg=libsgfx_scarlet_virgl.so",
         "-C", "link-arg=--exclude-libs=ALL", "-C", "link-arg=--entry=0",
         "-C", "link-arg=-z", "-C", "link-arg=defs"])
    stage(release / "libsgfx_scarlet_virgl.so", drivers / "libsgfx_scarlet_virgl.so")
    (drivers / "scarlet-virgl.sgfx-driver").write_text("abi=2\nname=scarlet-virgl\ngpu_backend=virtio-gpu\nlibrary=libsgfx_scarlet_virgl.so\n")
    if args.maxwell:
        manifest = args.maxwell.resolve() / "userspace/sgfx-backend-scarlet-maxwell-plugin/Cargo.toml"
        run([cargo, "rustc", "--manifest-path", manifest, "--lib", *common,
             "--", "-C", "link-arg=-soname", "-C", "link-arg=libsgfx_scarlet_maxwell.so",
             "-C", "link-arg=--exclude-libs=ALL", "-C", "link-arg=--entry=0",
             "-C", "link-arg=-z", "-C", "link-arg=defs"], args.maxwell.resolve())
        stage(release / "libsgfx_scarlet_maxwell.so", drivers / "libsgfx_scarlet_maxwell.so")
        (drivers / "scarlet-maxwell.sgfx-driver").write_text(
            "abi=2\nname=scarlet-maxwell\ngpu_backend=nvidia-gm20b\nlibrary=libsgfx_scarlet_maxwell.so\n")
    example = [cargo, "rustc", "-p", "sgfx", "--example", "dynamic_smoke", "--no-default-features"]
    run([*example, "--features", "std,backend-scarlet-virgl-static", *common])
    stage(release / "examples/dynamic_smoke", staging / "bin/sgfx-static-smoke")
    run([*example, "--features", "std,backend-dynamic,backend-scarlet-virgl", *common, "--", "-C", "link-arg=-pie",
         "-C", "link-arg=--dynamic-linker=/bin/scarlet-ld",
         # LLD needs a shared input to emit imports supplied by the interpreter.
         # --as-needed drops this unreferenced driver: the frontend has no
         # DT_NEEDED backend dependency. The ELF audit below enforces that.
         "-C", "link-arg=--as-needed", "-C", f"link-arg={release / 'libsgfx_scarlet_virgl.so'}",
         "-C", "link-arg=--unresolved-symbols=ignore-all"])
    stage(release / "examples/dynamic_smoke", staging / "bin/sgfx-dynamic-smoke")
    # Rebuild interpreter std from the same sources: all std instances must agree
    # on the native thread layout, including namespaces and cleanup records.
    run([cargo, "build", "--manifest-path", args.scarlet.resolve() / "user/scarlet-ld/Cargo.toml", *common], args.scarlet.resolve())
    stage(release / "scarlet-ld", staging / "bin/scarlet-ld")
    # Use the standard bootstrap policy, including devfs, stdio and Environment.
    # A bare /init cannot spawn ordinary children before this transition.
    bootstrap = output / "bootstrap"
    (bootstrap / "src").mkdir(parents=True, exist_ok=True)
    for filename, destination in (("init.rs", "main.rs"), ("bootstrap.rs", "bootstrap.rs")):
        shutil.copyfile(args.scarlet.resolve() / "user/bin/src" / filename, bootstrap / "src" / destination)
    (bootstrap / "Cargo.toml").write_text('[package]\nname="sgfx-smoke-init"\nversion="0.1.0"\nedition="2024"\n[workspace]\n[dependencies]\nscarlet-std={path=' + json.dumps(str(args.scarlet.resolve() / "user/lib/std")) + '}\n[profile.release]\npanic="abort"\n')
    run([cargo, "build", "--release", "--manifest-path", bootstrap / "Cargo.toml", "--target", original,
         *(["--offline"] if args.offline else [])])
    stage(output / "cargo" / original / "release/sgfx-smoke-init", staging / "init")
    artifacts = {str(p.relative_to(staging)): {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in staging.rglob("*") if p.is_file()}
    reports = audit(staging, args.scarlet.resolve(), args.arch)
    runtime_files = [tls_source, source / "library/std/src/sys/thread/scarlet.rs",
                     source / "library/std/src/sys/thread_local/mod.rs",
                     source / "library/std/src/sys/pal/scarlet/common.rs", crt_source]
    (output / "build.json").write_text(json.dumps({"result": "PASS", "compiler": subprocess.check_output([rustc, "-vV"], text=True),
                                                 "rust_source": str(source), "tls_source_sha256": hashlib.sha256(tls_source.read_bytes()).hexdigest(),
                                                 "runtime_sources": {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime_files},
                                                 "profile": {"lto": "thin", "codegen_units": 1}, "target": spec,
                                                 "commands": commands, "artifacts": artifacts, "elf": reports}, indent=2) + "\n")
    print(staging)


if __name__ == "__main__":
    main()
