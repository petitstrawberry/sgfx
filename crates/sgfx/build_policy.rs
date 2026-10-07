/// Compile-time routing shared by the build script and portable tests.
#[derive(Debug)]
pub struct Policy {
    pub dynamic: bool,
    pub dynamic_virgl: bool,
}

impl Policy {
    pub fn for_target(os: &str, arch: &str, width: &str, dynamic: bool, native_api: bool) -> Self {
        Self::new(
            os == "scarlet" || (os == "linux" && arch == "aarch64" && native_api),
            width == "64",
            dynamic,
        )
    }

    pub const fn new(scarlet: bool, native64: bool, dynamic: bool) -> Self {
        let dynamic = scarlet && native64 && dynamic;
        Self {
            dynamic,
            dynamic_virgl: dynamic,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Policy;

    #[test]
    fn native_dynamic_selection_is_backend_independent() {
        let policy = Policy::new(true, true, true);
        assert!(policy.dynamic && policy.dynamic_virgl);
    }

    #[test]
    fn disabled_feature_does_not_enable_dynamic_selection() {
        let policy = Policy::new(true, true, false);
        assert!(!policy.dynamic && !policy.dynamic_virgl);
    }

    #[test]
    fn dynamic_routing_requires_native_transport_and_elf64() {
        let elf32 = Policy::new(true, false, true);
        assert!(!elf32.dynamic && !elf32.dynamic_virgl);
        let host = Policy::new(false, true, true);
        assert!(!host.dynamic && !host.dynamic_virgl);
    }
    #[test]
    fn linux_native_api_uses_dynamic_drivers() {
        let linux = Policy::for_target("linux", "aarch64", "64", true, true);
        assert!(linux.dynamic && linux.dynamic_virgl);
        for arch in ["aarch64", "riscv64"] {
            let native = Policy::for_target("scarlet", arch, "64", true, false);
            assert!(native.dynamic && native.dynamic_virgl);
        }
    }

    #[test]
    fn linux_transport_requires_explicit_native_api_and_supported_target() {
        for (os, arch, width, native_api) in [
            ("linux", "aarch64", "64", false),
            ("linux", "x86_64", "64", true),
            ("linux", "aarch64", "32", true),
            ("macos", "aarch64", "64", true),
        ] {
            let policy = Policy::for_target(os, arch, width, true, native_api);
            assert!(!policy.dynamic && !policy.dynamic_virgl);
        }
    }
}
