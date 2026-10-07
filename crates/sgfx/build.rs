mod build_policy;

fn main() {
    for name in ["sgfx_dynamic", "sgfx_dynamic_virgl"] {
        println!("cargo:rustc-check-cfg=cfg({name})");
    }
    let policy = build_policy::Policy::for_target(
        &std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default(),
        &std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default(),
        &std::env::var("CARGO_CFG_TARGET_POINTER_WIDTH").unwrap_or_default(),
        std::env::var_os("CARGO_FEATURE_BACKEND_DYNAMIC").is_some(),
        std::env::var_os("CARGO_FEATURE_SCARLET_NATIVE_API").is_some(),
    );
    for (enabled, name) in [
        (policy.dynamic, "sgfx_dynamic"),
        (policy.dynamic_virgl, "sgfx_dynamic_virgl"),
    ] {
        if enabled {
            println!("cargo:rustc-cfg={name}");
        }
    }
}
