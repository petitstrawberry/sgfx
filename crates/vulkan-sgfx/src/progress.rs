//! Opt-in progress records taken after successful GPU retirement/presentation.
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Actual CPU snapshot work in one progress-report interval, not accepted GPU
/// bytes. Count the cloned spans before transport padding, once per buffer use.
#[derive(Default, Debug)]
pub(crate) struct HostUploads {
    pub(crate) bytes: u64,
    pub(crate) full_buffers: u64,
    pub(crate) partial_buffers: u64,
    pub(crate) max_buffer_bytes: u64,
    pub(crate) mapped_bytes: u64,
    pub(crate) unmapped_bytes: u64,
}

impl HostUploads {
    pub(crate) fn copied_span(
        &mut self,
        bytes: usize,
        buffer_bytes: &mut u64,
        full: bool,
        mapped: bool,
    ) {
        if bytes == 0 {
            return;
        }
        if *buffer_bytes == 0 {
            if full {
                self.full_buffers = self.full_buffers.saturating_add(1);
            } else {
                self.partial_buffers = self.partial_buffers.saturating_add(1);
            }
        }
        let bytes = bytes as u64;
        *buffer_bytes = buffer_bytes.saturating_add(bytes);
        self.bytes = self.bytes.saturating_add(bytes);
        self.max_buffer_bytes = self.max_buffer_bytes.max(*buffer_bytes);
        if mapped {
            self.mapped_bytes = self.mapped_bytes.saturating_add(bytes);
        } else {
            self.unmapped_bytes = self.unmapped_bytes.saturating_add(bytes);
        }
    }

    fn merge(&mut self, other: &Self) {
        self.bytes = self.bytes.saturating_add(other.bytes);
        self.full_buffers = self.full_buffers.saturating_add(other.full_buffers);
        self.partial_buffers = self.partial_buffers.saturating_add(other.partial_buffers);
        self.max_buffer_bytes = self.max_buffer_bytes.max(other.max_buffer_bytes);
        self.mapped_bytes = self.mapped_bytes.saturating_add(other.mapped_bytes);
        self.unmapped_bytes = self.unmapped_bytes.saturating_add(other.unmapped_bytes);
    }
}

pub(crate) struct HostUploadBatch {
    state: &'static Mutex<Progress>,
    pub(crate) stats: HostUploads,
}

impl Drop for HostUploadBatch {
    fn drop(&mut self) {
        if self.stats.bytes != 0 {
            self.state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .host_uploads
                .merge(&self.stats);
        }
    }
}

/// Reuse the existing cached native environment gate. No environment lookup,
/// allocation or progress lock per buffer/span; merge once even on early error.
pub(crate) fn host_uploads() -> Option<HostUploadBatch> {
    Some(HostUploadBatch {
        state: state()?,
        stats: HostUploads::default(),
    })
}

struct Progress {
    started: Instant,
    reported: Instant,
    streams: u64,
    draws: u64,
    frames: u64,
    last_present: Option<Instant>,
    longest_frame_gap: Duration,
    reported_frames: u64,
    host_uploads: HostUploads,
}

fn state() -> Option<&'static Mutex<Progress>> {
    static STATE: OnceLock<Option<Mutex<Progress>>> = OnceLock::new();
    STATE
        .get_or_init(|| {
            (crate::api::process_env("SGFX_VULKAN_PROGRESS").as_deref() == Some("1")).then(|| {
                let now = Instant::now();
                Mutex::new(Progress {
                    started: now,
                    reported: now,
                    streams: 0,
                    draws: 0,
                    frames: 0,
                    last_present: None,
                    longest_frame_gap: Duration::ZERO,
                    reported_frames: 0,
                    host_uploads: HostUploads::default(),
                })
            })
        })
        .as_ref()
}

pub(crate) fn submitted(draws: u64) {
    if let Some(state) = state() {
        let mut state = state.lock().unwrap_or_else(|e| e.into_inner());
        state.streams += 1;
        state.draws += draws;
    }
}

pub(crate) fn presented(swapchain: u64) {
    crate::frame_profile::presented();
    let Some(state) = state() else {
        return;
    };
    let mut state = state.lock().unwrap_or_else(|e| e.into_inner());
    state.frames += 1;
    let now = Instant::now();
    if let Some(previous) = state.last_present {
        state.longest_frame_gap = state.longest_frame_gap.max(now.duration_since(previous));
    }
    state.last_present = Some(now);
    if state.frames == 1 || now.duration_since(state.reported) >= Duration::from_secs(5) {
        // Scarlet's present path waits for successful queue retirement before
        // reaching this point. Other platforms report presentation only.
        let retired = cfg!(any(
            target_os = "scarlet",
            all(target_os = "linux", feature = "scarlet-wsi")
        ));
        eprintln!(
            "SGFX_VULKAN_PROGRESS elapsed_ms={} swapchain={swapchain} presented_frames={} accepted_streams={} accepted_draws={} gpu_idle_before_present={retired} interval_ms={} interval_frames={} max_frame_gap_ms={} host_upload_bytes={} host_upload_full_buffers={} host_upload_partial_buffers={} host_upload_max_buffer_bytes={} host_upload_mapped_bytes={} host_upload_unmapped_bytes={}",
            now.duration_since(state.started).as_millis(),
            state.frames,
            state.streams,
            state.draws,
            now.duration_since(state.reported).as_millis(),
            state.frames - state.reported_frames,
            state.longest_frame_gap.as_millis(),
            state.host_uploads.bytes,
            state.host_uploads.full_buffers,
            state.host_uploads.partial_buffers,
            state.host_uploads.max_buffer_bytes,
            state.host_uploads.mapped_bytes,
            state.host_uploads.unmapped_bytes,
        );
        state.reported = now;
        state.reported_frames = state.frames;
        state.longest_frame_gap = Duration::ZERO;
        state.host_uploads = HostUploads::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merging_intervals_sums_work_but_keeps_the_largest_single_buffer_upload() {
        let mut interval = HostUploads::default();
        for _ in 0..2 {
            let mut snapshot = HostUploads::default();
            snapshot.copied_span(8, &mut 0, true, false);
            let mut buffer_bytes = 0;
            snapshot.copied_span(4, &mut buffer_bytes, false, true);
            snapshot.copied_span(12, &mut buffer_bytes, false, true);
            interval.merge(&snapshot);
        }
        assert_eq!(interval.bytes, 48);
        assert_eq!((interval.full_buffers, interval.partial_buffers), (2, 2));
        assert_eq!(interval.max_buffer_bytes, 16);
        assert_eq!((interval.mapped_bytes, interval.unmapped_bytes), (32, 16));
        let reported = std::mem::take(&mut interval);
        assert_eq!(reported.bytes, 48);
        assert_eq!((interval.bytes, interval.max_buffer_bytes), (0, 0));
    }
}
