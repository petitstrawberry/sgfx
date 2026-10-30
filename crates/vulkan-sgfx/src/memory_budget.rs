//! Bound logical device allocations using native shared-memory capacity.
use std::sync::OnceLock;

const MIB: u64 = 1024 * 1024;
const FALLBACK: usize = 512 * MIB as usize;

#[cfg(any(target_os = "scarlet", test))]
fn from_capacity(bytes: u64) -> usize {
    // Host mirrors, backend shadows and GPU resources may coexist. Leave
    // three quarters of physical RAM for them and for the application/kernel.
    ((bytes / 4).min(2048 * MIB) / MIB * MIB) as usize
}

pub(crate) fn heap_size() -> usize {
    static SIZE: OnceLock<usize> = OnceLock::new();
    *SIZE.get_or_init(|| {
        #[cfg(target_os = "scarlet")]
        if let Ok(info) = scarlet_os::system::memory_info() {
            let size = from_capacity(info.total_bytes);
            if size != 0 {
                eprintln!("[SGFX Vulkan] shared memory heap: bytes={size} physical_total={} physical_free={}", info.total_bytes, info.free_bytes);
                return size;
            }
        }
        FALLBACK
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn heap_tracks_capacity_with_room_for_mirrors_and_a_hard_cap() {
        assert_eq!(from_capacity(512 * MIB), 128 * MIB as usize);
        assert_eq!(from_capacity(2048 * MIB), 512 * MIB as usize);
        assert_eq!(from_capacity(8192 * MIB), 2048 * MIB as usize);
        assert_eq!(from_capacity(u64::MAX), 2048 * MIB as usize);
        assert_eq!(from_capacity(0), 0);
    }
}
