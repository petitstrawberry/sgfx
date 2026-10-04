mod build_policy;

fn main() {
    for name in ["sgfx_dynamic", "sgfx_dynamic_virgl"] {
        println!("cargo:rustc-check-cfg=cfg({name})");
    }
    let policy = build_policy::Policy::new(
        std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("scarlet"),
        std::env::var("CARGO_CFG_TARGET_POINTER_WIDTH").as_deref() == Ok("64"),
        std::env::var_os("CARGO_FEATURE_BACKEND_DYNAMIC").is_some(),
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
