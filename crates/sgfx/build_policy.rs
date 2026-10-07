/// Compile-time routing shared by the build script and portable tests.
#[derive(Debug)]
pub struct Policy {
    pub dynamic: bool,
    pub dynamic_virgl: bool,
}

impl Policy {
    pub fn for_target(
        os: &str,
        arch: &str,
        width: &str,
        dynamic: bool,
        native_api: bool,
        virgl_static: bool,
    ) -> Self {
        Self::new(
            os == "scarlet" || (os == "linux" && arch == "aarch64" && native_api),
            width == "64",
            dynamic,
            virgl_static,
        )
    }

    pub const fn new(
        native_transport: bool,
        native64: bool,
        dynamic: bool,
        virgl_static: bool,
    ) -> Self {
        let dynamic = native_transport && native64 && dynamic;
        Self {
            dynamic,
            dynamic_virgl: dynamic && !virgl_static,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Policy;

    #[test]
    fn native_dynamic_selection_is_backend_independent() {
        let policy = Policy::new(true, true, true, false);
        assert!(policy.dynamic && policy.dynamic_virgl);
    }

    #[test]
    fn virgl_static_comparison_keeps_other_dynamic_backends() {
        let policy = Policy::new(true, true, true, true);
        assert!(policy.dynamic);
        assert!(!policy.dynamic_virgl);
        let standalone = Policy::new(true, true, false, true);
        assert!(!standalone.dynamic && !standalone.dynamic_virgl);
    }

    #[test]
    fn disabled_feature_does_not_enable_dynamic_selection() {
        let policy = Policy::new(true, true, false, false);
        assert!(!policy.dynamic && !policy.dynamic_virgl);
    }

    #[test]
    fn dynamic_native_routing_requires_native_transport_and_elf64() {
        let elf32 = Policy::new(true, false, true, false);
        assert!(!elf32.dynamic && !elf32.dynamic_virgl);
        let host = Policy::new(false, true, true, false);
        assert!(!host.dynamic && !host.dynamic_virgl);
    }

    #[test]
    fn linux_native_api_uses_dynamic_drivers_unless_static_is_requested() {
        let linux = Policy::for_target("linux", "aarch64", "64", true, true, false);
        assert!(linux.dynamic && linux.dynamic_virgl);
        let comparison = Policy::for_target("linux", "aarch64", "64", true, true, true);
        assert!(comparison.dynamic && !comparison.dynamic_virgl);
        for (os, arch, width, native_api) in [
            ("linux", "aarch64", "64", false),
            ("linux", "x86_64", "64", true),
            ("linux", "aarch64", "32", true),
            ("macos", "aarch64", "64", true),
        ] {
            let policy = Policy::for_target(os, arch, width, true, native_api, false);
            assert!(!policy.dynamic && !policy.dynamic_virgl);
        }
    }
}
