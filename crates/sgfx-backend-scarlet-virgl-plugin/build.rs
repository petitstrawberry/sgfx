use std::env;

fn main() {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("scarlet") {
        for argument in ["--hash-style=both", "-z", "now", "--entry=0", "--exclude-libs=ALL", "-soname", "libsgfx_scarlet_virgl.so", "--unresolved-symbols=ignore-all"] {
            println!("cargo::rustc-link-arg-cdylib={argument}");
        }
    }
}
