"""Configure application linkage around the driver staged by the SDK."""
import json
from pathlib import Path
import sys
import tomllib

mode, project, destination = sys.argv[1:]
project, destination = Path(project), Path(destination)
manifest = tomllib.loads((project / "scarlet.toml").read_text())
config = tomllib.loads((project / manifest["bsp"]["path"] / ".cargo/config.toml").read_text())
arch = Path(config["build"]["target"]).name.split("-")[0]
target = {"aarch64": "aarch64-unknown-scarlet", "riscv64": "riscv64gc-unknown-scarlet", "riscv32": "riscv32gc-unknown-scarlet"}[arch]
flags = ["--cfg", 'getrandom_backend="custom"']
if arch != "riscv32":
    flags += ["-Zdefault-visibility=hidden"]
    if mode == "install":
        library = (destination / "libsgfx_scarlet_virgl.so").resolve(strict=True)
        flags += ["-C", "link-arg=--dynamic-linker=/bin/scarlet-ld", "-C", "link-arg=--as-needed", "-C", f"link-arg={library}", "-C", "link-arg=--unresolved-symbols=ignore-all"]
        (destination / "scarlet-virgl.sgfx-driver").write_text("abi=2\nname=scarlet-virgl\ngpu_backend=virtio-gpu\nlibrary=libsgfx_scarlet_virgl.so\n")
output = project / ".scarlet/sgfx/userspace.toml"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(f"[target.{target}]\nrustflags = {json.dumps(flags)}\n")
