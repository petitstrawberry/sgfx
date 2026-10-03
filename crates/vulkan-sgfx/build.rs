use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=protocol/scarlet-sgfx.xml");
    println!("cargo:rerun-if-changed=src/wayland_client.c");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux")
        || env::var_os("CARGO_FEATURE_SCARLET_WSI").is_none()
    {
        return;
    }
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    for (mode, name) in [
        ("client-header", "scarlet-sgfx-client.h"),
        ("private-code", "scarlet-sgfx-protocol.c"),
    ] {
        assert!(
            Command::new("wayland-scanner")
                .args([mode, "protocol/scarlet-sgfx.xml"])
                .arg(out.join(name))
                .status()
                .expect("install wayland-scanner for Scarlet Linux WSI")
                .success()
        );
    }
    cc::Build::new()
        .include(&out)
        .file("src/wayland_client.c")
        .file(out.join("scarlet-sgfx-protocol.c"))
        .warnings(true)
        .compile("sgfx_wayland_client");
    println!("cargo:rustc-link-lib=wayland-client");
}
