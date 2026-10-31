//! Optional timestamped CPU lanes. Intervals may overlap across threads; they
//! are not GPU execution times. Output is buffered and never changes GPU work.
use std::cell::{Cell, RefCell};
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

const CAPACITY: usize = 65_536;
std::thread_local! { static RECORDING: RefCell<[(u64, u64); 3]> = const { RefCell::new([(0, 0); 3]) }; }

// Frequent command-recording calls accumulate on the calling thread. They do
// not allocate, take the trace mutex, or emit an event for every command.
pub(crate) struct RecordingSpan(Option<(&'static Profile, u64, usize)>);
impl RecordingSpan {
    pub(crate) fn new() -> Self {
        Self::part(0)
    }
    pub(crate) fn descriptors() -> Self {
        Self::part(1)
    }
    pub(crate) fn invalidation() -> Self {
        Self::part(2)
    }
    fn part(index: usize) -> Self {
        Self(profile().map(|p| (p, p.now(), index)))
    }
}
impl Drop for RecordingSpan {
    fn drop(&mut self) {
        if let Some((profile, begin, index)) = self.0 {
            let elapsed = profile.now().saturating_sub(begin);
            RECORDING.with(|cell| {
                let mut values = cell.borrow_mut();
                values[index].0 += elapsed;
                values[index].1 += 1;
            });
        }
    }
}
struct Event {
    thread: u64,
    begin: u64,
    end: u64,
    kind: &'static str,
    name: &'static str,
    frame: u64,
}
struct Profile {
    epoch: Instant,
    capture_start_ns: u64,
    capture_end_ns: u64,
    events: Mutex<Vec<Event>>,
    spare: Mutex<Vec<Event>>,
    output: Mutex<BufWriter<std::fs::File>>,
    frames: AtomicU64,
    // Read by the platform WSI presentation hook.
    #[cfg_attr(
        not(any(
            target_os = "scarlet",
            all(target_os = "linux", feature = "scarlet-wsi")
        )),
        allow(dead_code)
    )]
    presented: AtomicU64,
    dropped: AtomicU64,
}
fn profile() -> Option<&'static Profile> {
    static STATE: OnceLock<Option<Profile>> = OnceLock::new();
    STATE
        .get_or_init(|| {
            if crate::api::process_env("SGFX_VULKAN_FRAME_PROFILE").as_deref() != Some("1") {
                return None;
            }
            let path = format!("/tmp/sgfx-frames-{}.tsv", std::process::id());
            let mut output = BufWriter::with_capacity(65536, std::fs::File::create(path).ok()?);
            writeln!(output, "kind\tthread\tbegin_ns\tend_ns\tframe\toperation").ok()?;
            let capture_start_ns = crate::api::process_env("SGFX_VULKAN_FRAME_PROFILE_START_MS")
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(0)
                .saturating_mul(1_000_000);
            let capture_end_ns = crate::api::process_env("SGFX_VULKAN_FRAME_PROFILE_DURATION_MS")
                .and_then(|value| value.parse::<u64>().ok())
                .map_or(u64::MAX, |ms| {
                    capture_start_ns.saturating_add(ms.saturating_mul(1_000_000))
                });
            Some(Profile {
                epoch: Instant::now(),
                capture_start_ns,
                capture_end_ns,
                events: Mutex::new(Vec::with_capacity(CAPACITY)),
                spare: Mutex::new(Vec::with_capacity(CAPACITY)),
                output: Mutex::new(output),
                frames: AtomicU64::new(0),
                presented: AtomicU64::new(0),
                dropped: AtomicU64::new(0),
            })
        })
        .as_ref()
}
fn thread_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    std::thread_local! { static ID: Cell<u64> = const { Cell::new(0) }; }
    ID.with(|id| {
        let mut value = id.get();
        if value == 0 {
            value = NEXT.fetch_add(1, Ordering::Relaxed);
            id.set(value);
        }
        value
    })
}
impl Profile {
    fn now(&self) -> u64 {
        self.epoch.elapsed().as_nanos() as u64
    }
    fn push(&self, event: Event) {
        if event.end < self.capture_start_ns || event.begin >= self.capture_end_ns {
            return;
        }
        let mut events = self.events.lock().unwrap_or_else(|e| e.into_inner());
        if events.len() < CAPACITY {
            events.push(event);
        } else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
    fn flush(&self) {
        let begin = self.now();
        let mut batch = self.spare.lock().unwrap_or_else(|e| e.into_inner());
        {
            let mut events = self.events.lock().unwrap_or_else(|e| e.into_inner());
            std::mem::swap(&mut *batch, &mut *events);
        }
        let mut output = self.output.lock().unwrap_or_else(|e| e.into_inner());
        for event in batch.drain(..) {
            let _ = writeln!(
                output,
                "{}\t{}\t{}\t{}\t{}\t{}",
                event.kind, event.thread, event.begin, event.end, event.frame, event.name
            );
        }
        let dropped = self.dropped.swap(0, Ordering::Relaxed);
        if dropped != 0 {
            let _ = writeln!(output, "dropped\t0\t0\t0\t{dropped}\tcapacity");
        }
        let _ = output.flush();
        drop(output);
        self.push(Event {
            thread: thread_id(),
            begin,
            end: self.now(),
            kind: "work",
            name: "profiler::flush",
            frame: 0,
        });
    }
}
pub(crate) struct Span(Option<(&'static Profile, Event)>);
#[cfg_attr(
    not(any(
        target_os = "scarlet",
        all(target_os = "linux", feature = "scarlet-wsi")
    )),
    allow(dead_code)
)]
pub(crate) fn presented() {
    if let Some(profile) = profile() {
        let now = profile.now();
        profile.push(Event {
            thread: thread_id(),
            begin: now,
            end: now,
            kind: "presented",
            name: "image_submitted_to_compositor",
            frame: profile.presented.fetch_add(1, Ordering::Relaxed) + 1,
        });
    }
}
impl Span {
    pub(crate) fn new(kind: &'static str, name: &'static str) -> Self {
        Self(profile().map(|profile| {
            (
                profile,
                Event {
                    thread: thread_id(),
                    begin: profile.now(),
                    end: 0,
                    kind,
                    name,
                    frame: 0,
                },
            )
        }))
    }
}
impl Drop for Span {
    fn drop(&mut self) {
        let Some((profile, mut event)) = self.0.take() else {
            return;
        };
        event.end = profile.now();
        let present = event.kind == "present";
        if present {
            event.frame = profile.frames.fetch_add(1, Ordering::Relaxed) + 1;
        }
        if present {
            let values = RECORDING.with(|cell| cell.replace([(0, 0); 3]));
            // Counter values occupy the frame/value column. Timestamp them
            // at this present boundary; these are not synthetic intervals.
            for (name, (ns, calls)) in [
                "command_record",
                "descriptor_update",
                "descriptor_invalidation",
            ]
            .into_iter()
            .zip(values)
            {
                for (kind, value) in [("record_ns", ns), ("record_calls", calls)] {
                    profile.push(Event {
                        thread: event.thread,
                        begin: event.end,
                        end: event.end,
                        kind,
                        name,
                        frame: value,
                    });
                }
            }
        }
        let flush = present
            && (event.frame.is_multiple_of(30)
                || profile
                    .events
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .len()
                    > CAPACITY / 2);
        profile.push(event);
        if flush {
            profile.flush();
        }
    }
}

#[cfg(target_os = "scarlet")]
pub(crate) fn enabled() -> bool {
    profile().is_some()
}

// SWS reports durations immediately after releasing its shared connection
// mutex. Only the presentation, size and GPU-release calls report here; these
// intervals join the calling presentation thread's existing frame timeline.
#[cfg(target_os = "scarlet")]
pub(crate) extern "C" fn sws_timing(operation: u32, wait_ns: u64, service_ns: u64) {
    let Some(profile) = profile() else {
        return;
    };
    let name = match operation {
        2 => "sws_client::gpu_commit",
        3 => "sws_client::window_size",
        4 => "sws_client::gpu_poll",
        _ => "sws_client::other",
    };
    let end = profile.now();
    let acquired = end.saturating_sub(service_ns);
    let thread = thread_id();
    profile.push(Event {
        thread,
        begin: acquired.saturating_sub(wait_ns),
        end: acquired,
        kind: "sws_lock",
        name,
        frame: 0,
    });
    profile.push(Event {
        thread,
        begin: acquired,
        end,
        kind: "sws_service",
        name,
        frame: 0,
    });
}

#[cfg(target_os = "scarlet")]
pub(crate) fn sws_buffer_event(kind: &'static str, serial: u64) {
    if let Some(profile) = profile() {
        let now = profile.now();
        profile.push(Event {
            thread: thread_id(),
            begin: now,
            end: now,
            kind,
            name: "sws_image_lease",
            frame: serial,
        });
    }
}
