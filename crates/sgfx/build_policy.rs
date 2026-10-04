/// Compile-time routing shared by the build script and portable tests.
#[derive(Debug)]
pub struct Policy {
    pub dynamic: bool,
    pub dynamic_virgl: bool,
    pub static_maxwell: bool,
}

impl Policy {
    pub const fn new(
        scarlet: bool,
        native64: bool,
        dynamic: bool,
        virgl_static: bool,
        maxwell: bool,
        maxwell_static: bool,
    ) -> Self {
        let dynamic = scarlet && native64 && dynamic;
        Self {
            dynamic,
            dynamic_virgl: dynamic && !virgl_static,
            static_maxwell: scarlet && maxwell && (!native64 || maxwell_static),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Policy;

    #[test]
    fn native_maxwell_defaults_to_dynamic_without_static_backend() {
        let policy = Policy::new(true, true, true, false, true, false);
        assert!(policy.dynamic && policy.dynamic_virgl);
        assert!(!policy.static_maxwell);
    }

    #[test]
    fn virgl_static_comparison_keeps_other_dynamic_backends() {
        let policy = Policy::new(true, true, true, true, true, false);
        assert!(policy.dynamic);
        assert!(!policy.dynamic_virgl && !policy.static_maxwell);
        let standalone = Policy::new(true, true, false, true, false, false);
        assert!(!standalone.dynamic && !standalone.dynamic_virgl);
    }

    #[test]
    fn maxwell_static_is_an_explicit_native_comparison() {
        let policy = Policy::new(true, true, true, false, true, true);
        assert!(policy.dynamic && policy.static_maxwell);
        let standalone = Policy::new(true, true, false, false, true, true);
        assert!(standalone.static_maxwell && !standalone.dynamic);
    }

    #[test]
    fn elf32_retains_static_maxwell_and_hosts_do_not_select_it() {
        let elf32 = Policy::new(true, false, true, false, true, false);
        assert!(!elf32.dynamic && elf32.static_maxwell);
        let host = Policy::new(false, true, true, false, true, true);
        assert!(!host.dynamic && !host.static_maxwell);
    }
}
