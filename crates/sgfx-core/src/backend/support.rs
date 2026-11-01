//! Reusable allocation-aware backend algorithms, independent of GPU protocols.
//!
//! - [`ranges`](crate::backend::support::ranges): disjoint last-write-wins references into caller-owned uploads.
//! - [`buffers`](crate::backend::support::buffers): sparse reads and bounded cumulative dirty ranges. Choose the
//!   interval capacity and transfer alignment for the actual GPU transport.
//! - [`buffer_diff`](crate::backend::support::buffer_diff): word comparisons and changed-range discovery without allocation.
//! - [`indices`](crate::backend::support::indices): revision-keyed exact/block extrema, without retaining bytes.
//!   Invalidate a slot on pending writes/copies, even before its revision changes.
//! - [`scheduler`](crate::backend::support::scheduler): bounded admission, ordered dispatch and prefix completion.
//!   Implement its transport hooks; OS workers, waits and GPU fences stay local.
//! - [`packets`](crate::backend::support::packets): packet-boundary splitting and move-based allocation adoption.
//!   Choose stream limits and upload arena counts; encode GPU commands locally.
//! - [`state`](crate::backend::support::state): identity-first immutable state reuse, cleared on native resets.
//! - [`passes`](crate::backend::support::passes): attachment checks for side-effect-free empty-pass elision.
//! - [`barriers`](crate::backend::support::barriers): classification only, requiring a separately negotiated backend
//!   guarantee and successful IR validation before execution may be skipped.
//!
//! These helpers never certify CPU mirrors, external image ownership, host
//! visibility, GPU completion or presentation merely from command acceptance.
//! They use `alloc` and static dispatch, with no OS/GPU dependencies or worker
//! threads. Reuse does not force another backend to adopt VirGL's limits.

pub mod barriers;
pub mod buffer_diff;
pub mod buffers;
pub mod indices;
pub mod packets;
pub mod passes;
pub mod ranges;
pub mod scheduler;
pub mod state;
