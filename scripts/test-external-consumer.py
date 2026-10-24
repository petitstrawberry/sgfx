#!/usr/bin/env python3
"""Check public crate identity without the repository's workspace patches."""

import json
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    # Cargo ignores a dependency's workspace [patch] table. This fixture must
    # live outside the workspace to exercise the graph downstream users get.
    with tempfile.TemporaryDirectory(prefix="sgfx-consumer-") as directory:
        fixture = Path(directory)
        dependencies = []
        for name in ("sgfx", "sgfx-core", "sgfx-backend-abi", "sgfx-codegen-virgl"):
            features = ', features = ["backend-abi"]' if name == "sgfx-core" else ""
            dependencies.append(
                f'{name} = {{ path = {json.dumps(str(ROOT / "crates" / name))}, '
                f'default-features = false{features} }}'
            )
        (fixture / "Cargo.toml").write_text(
            '[package]\nname = "external-consumer"\nversion = "0.0.0"\n'
            'edition = "2024"\n\n[dependencies]\n' + "\n".join(dependencies) + "\n"
        )
        (fixture / "src").mkdir()
        (fixture / "src/main.rs").write_text('''
use sgfx_core::ir::*;

fn main() {
    let producer: ResourceTable = sgfx::ir::ResourceTable::new();
    let texture = producer.define_texture(TextureDesc::new(
        TextureFormat::Bgra8Unorm, Extent2D::new(1280, 720).unwrap(),
        TextureUsage::RENDER_ATTACHMENT | TextureUsage::PRESENT,
    ).unwrap()).unwrap();
    let mirror = ResourceTable::new();
    mirror.sync_abi_snapshot(&producer.abi_snapshot().unwrap()).unwrap();
    assert_eq!(mirror.texture(mirror.abi_texture(texture.slot() as u32).unwrap()).unwrap(),
               producer.texture(texture).unwrap());
    let _: sgfx_backend_abi::Span<u64> = sgfx_core::ir::abi::Span::from_slice(&[]);
    let commands = sgfx_codegen_virgl::build_vertex_color_setup(
        1, 2, 16, CullMode::None, FrontFace::CounterClockwise,
    );
    assert!(!commands.is_empty());
}
''')
        subprocess.run([
            "cargo", "run", "--manifest-path", str(fixture / "Cargo.toml"),
            "--target-dir", str(ROOT / "target/external-consumer"),
        ], check=True, cwd=fixture)


if __name__ == "__main__":
    main()
