//! Maintenance1 on the native VirGL path. Pool trimming releases unused
//! registry capacity; allocated command buffers and their recordings survive.
use super::*;

fn trim_registry(registry: &mut CommandRegistry, pool: u64) {
    if !registry.pools.contains_key(&pool) {
        return;
    }
    // Freed command buffers already drop their Arc-owned storage. The only
    // unused pool bookkeeping retained here is hash-table capacity. Shrinking
    // it neither resets allocated recordings nor invalidates submit snapshots.
    registry.commands.shrink_to_fit();
    registry.pools.shrink_to_fit();
}

pub(super) unsafe extern "system" fn trim_command_pool(
    device: vk::Device,
    pool: vk::CommandPool,
    flags: vk::CommandPoolTrimFlags,
) {
    if !flags.is_empty() {
        return;
    }
    let Ok(driver) = driver(device.as_raw(), Kind::Device) else {
        return;
    };
    let mut registry = driver.recordings.lock().unwrap_or_else(|e| e.into_inner());
    trim_registry(&mut registry, pool.as_raw());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn trim_preserves_executable_recordings_and_submit_snapshots() {
        let mut registry = CommandRegistry::default();
        registry
            .pools
            .insert(7, vk::CommandPoolCreateFlags::empty());
        let mut recording = Recording::new(7);
        recording.state = RecordingState::Executable;
        Arc::make_mut(&mut recording.commands).push(RecordedCommand::SetViewport(vk::Viewport {
            width: 8.0,
            height: -8.0,
            y: 8.0,
            max_depth: 1.0,
            ..Default::default()
        }));
        let snapshot = recording.clone();
        let recording = Arc::new(Mutex::new(recording));
        registry.commands.insert(9, recording.clone());
        trim_registry(&mut registry, 7);
        let retained = registry.commands.get(&9).unwrap().lock().unwrap();
        assert!(matches!(retained.state, RecordingState::Executable));
        assert!(Arc::ptr_eq(&retained.commands, &snapshot.commands));
        assert_eq!(retained.commands.len(), 1);
        assert!(Arc::ptr_eq(registry.commands.get(&9).unwrap(), &recording));
    }
}
