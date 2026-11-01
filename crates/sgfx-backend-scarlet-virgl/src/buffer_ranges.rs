//! Native adapter to the shared range index; diagnostics exist only in tests.
#[cfg(not(test))]
pub(crate) use sgfx_core::backend::support::ranges::*;
#[cfg(test)]
pub(crate) use sgfx_core::backend::support::ranges::{Segment, intersecting};

#[cfg(test)]
#[derive(Default)]
pub(crate) struct RangeIndex {
    inner: sgfx_core::backend::support::ranges::RangeIndex,
    pub visits: core::cell::Cell<usize>,
}
#[cfg(test)]
impl RangeIndex {
    pub fn all(&self) -> &[Segment] {
        self.inner.all()
    }
    pub fn query(&self, range: core::ops::Range<usize>) -> impl Iterator<Item = &Segment> {
        self.inner
            .query(range)
            .inspect(|_| self.visits.set(self.visits.get() + 1))
    }
    pub fn insert(
        &mut self,
        range: core::ops::Range<usize>,
        write: usize,
    ) -> Result<(), alloc::collections::TryReserveError> {
        self.inner.insert(range, write)
    }
}
