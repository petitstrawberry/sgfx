/// Compile-time routing shared by the build script and portable tests.
#[derive(Debug)]
pub struct Policy {
    pub dynamic: bool,
    pub dynamic_virgl: bool,
}

impl Policy {
    pub const fn new(scarlet: bool, native64: bool, dynamic: bool, virgl_static: bool) -> Self {
        let dynamic = scarlet && native64 && dynamic;
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
    fn dynamic_native_routing_requires_scarlet_elf64() {
        let elf32 = Policy::new(true, false, true, false);
        assert!(!elf32.dynamic && !elf32.dynamic_virgl);
        let host = Policy::new(false, true, true, false);
        assert!(!host.dynamic && !host.dynamic_virgl);
    }
}
