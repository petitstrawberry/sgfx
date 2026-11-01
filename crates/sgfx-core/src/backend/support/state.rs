//! Identity-first caching of immutable, reference-counted backend state.
use alloc::rc::Rc;

/// Remembers only the last successfully emitted immutable state. Update after
/// command encoding succeeds, and clear whenever native state is invalidated.
/// Comparing the same allocation avoids scanning constants or descriptor rows.
pub struct SharedState<T: ?Sized> {
    value: Option<Rc<T>>,
}
impl<T: ?Sized> Default for SharedState<T> {
    fn default() -> Self {
        Self { value: None }
    }
}
impl<T: ?Sized + PartialEq> SharedState<T> {
    pub fn matches(&self, value: &Rc<T>) -> bool {
        self.value
            .as_ref()
            .is_some_and(|old| Rc::ptr_eq(old, value) || old.as_ref() == value.as_ref())
    }
    /// Retain the allocation without cloning its contents.
    pub fn remember(&mut self, value: &Rc<T>) {
        self.value = Some(Rc::clone(value));
    }
    pub fn clear(&mut self) {
        self.value = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::Cell;
    #[test]
    fn identity_skips_content_comparison_and_reset_forgets_native_state() {
        struct Value(Rc<Cell<usize>>);
        impl PartialEq for Value {
            fn eq(&self, _: &Self) -> bool {
                self.0.set(self.0.get() + 1);
                true
            }
        }
        let comparisons = Rc::new(Cell::new(0));
        let value = Rc::new(Value(Rc::clone(&comparisons)));
        let equal = Rc::new(Value(Rc::clone(&comparisons)));
        let mut state = SharedState::default();
        assert!(!state.matches(&value));
        state.remember(&value);
        assert!(state.matches(&value));
        assert_eq!(comparisons.get(), 0);
        assert!(state.matches(&equal));
        assert_eq!(comparisons.get(), 1);
        state.clear();
        assert!(!state.matches(&value));
    }
}
