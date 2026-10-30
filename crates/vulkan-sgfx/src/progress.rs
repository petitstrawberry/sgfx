//! Opt-in progress records taken after successful GPU retirement/presentation.
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

struct Progress {
    started: Instant,
    reported: Instant,
    streams: u64,
    draws: u64,
    frames: u64,
    last_present: Option<Instant>,
    longest_frame_gap: Duration,
    reported_frames: u64,
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
            "SGFX_VULKAN_PROGRESS elapsed_ms={} swapchain={swapchain} presented_frames={} accepted_streams={} accepted_draws={} gpu_idle_before_present={retired} interval_ms={} interval_frames={} max_frame_gap_ms={}",
            now.duration_since(state.started).as_millis(),
            state.frames,
            state.streams,
            state.draws,
            now.duration_since(state.reported).as_millis(),
            state.frames - state.reported_frames,
            state.longest_frame_gap.as_millis()
        );
        state.reported = now;
        state.reported_frames = state.frames;
        state.longest_frame_gap = Duration::ZERO;
    }
}
