fn main() {
    println!("cargo:rustc-check-cfg=cfg(sgfx_dynamic_virgl)");
    let enabled = std::env::var_os("CARGO_FEATURE_BACKEND_DYNAMIC").is_some();
    let static_backend = std::env::var_os("CARGO_FEATURE_BACKEND_SCARLET_VIRGL_STATIC").is_some();
    if enabled
        && !static_backend
        && std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("scarlet")
        && std::env::var("CARGO_CFG_TARGET_POINTER_WIDTH").as_deref() == Ok("64")
    {
        println!("cargo:rustc-cfg=sgfx_dynamic_virgl");
    }
}
