//! Owned Vulkan allocation-chain metadata and dedicated binding restrictions.
use ash::vk;
const INVALID: vk::Result = vk::Result::ERROR_INITIALIZATION_FAILED;
const UNSUPPORTED: vk::Result = vk::Result::ERROR_FEATURE_NOT_PRESENT;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DedicatedTarget {
    Buffer(vk::Buffer),
    Image(vk::Image),
}

/// Copy the recognized chain before dispatching work to the resource worker.
pub(crate) unsafe fn parse(
    mut node: *const vk::BaseInStructure<'_>,
) -> Result<Option<DedicatedTarget>, vk::Result> {
    let mut target = None;
    let mut flags_seen = false;
    for _ in 0..4 {
        let Some(base) = node.as_ref() else {
            return Ok(target);
        };
        match base.s_type {
            vk::StructureType::MEMORY_DEDICATED_ALLOCATE_INFO if target.is_none() => {
                let info = &*node.cast::<vk::MemoryDedicatedAllocateInfo<'_>>();
                target = Some(
                    match (
                        info.buffer == vk::Buffer::null(),
                        info.image == vk::Image::null(),
                    ) {
                        (false, true) => DedicatedTarget::Buffer(info.buffer),
                        (true, false) => DedicatedTarget::Image(info.image),
                        _ => return Err(INVALID),
                    },
                );
            }
            vk::StructureType::MEMORY_ALLOCATE_FLAGS_INFO if !flags_seen => {
                flags_seen = true;
                let info = &*node.cast::<vk::MemoryAllocateFlagsInfo<'_>>();
                if !vk::MemoryAllocateFlags::DEVICE_MASK.contains(info.flags)
                    || (info.flags.contains(vk::MemoryAllocateFlags::DEVICE_MASK)
                        && info.device_mask != 1)
                {
                    return Err(UNSUPPORTED);
                }
            }
            _ => return Err(UNSUPPORTED),
        }
        node = base.p_next;
    }
    Err(UNSUPPORTED)
}

pub(crate) fn allocation_size(
    resources: &crate::resources::Resources,
    target: DedicatedTarget,
) -> Option<u64> {
    match target {
        DedicatedTarget::Buffer(handle) => resources
            .buffers
            .get(&handle)
            .map(|b| b.size.next_multiple_of(256)),
        DedicatedTarget::Image(handle) => resources
            .images
            .get(&handle)
            .filter(|i| i.swapchain.is_none())
            .map(|i| i.byte_size()),
    }
}

pub(crate) fn permits(
    resources: &crate::resources::Resources,
    memory: vk::DeviceMemory,
    target: DedicatedTarget,
    offset: u64,
) -> bool {
    resources
        .dedicated_allocations
        .get(&memory)
        .is_none_or(|dedicated| offset == 0 && *dedicated == target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;
    #[test]
    fn chains_are_owned_and_only_one_device_and_target_are_allowed() {
        unsafe {
            for (flags, device_mask, expected) in [
                (
                    vk::MemoryAllocateFlags::DEVICE_MASK,
                    1,
                    Ok(Some(DedicatedTarget::Buffer(vk::Buffer::from_raw(7)))),
                ),
                (vk::MemoryAllocateFlags::DEVICE_MASK, 2, Err(UNSUPPORTED)),
                (vk::MemoryAllocateFlags::DEVICE_ADDRESS, 1, Err(UNSUPPORTED)),
            ] {
                let flags = vk::MemoryAllocateFlagsInfo::default()
                    .flags(flags)
                    .device_mask(device_mask);
                let mut dedicated =
                    vk::MemoryDedicatedAllocateInfo::default().buffer(vk::Buffer::from_raw(7));
                dedicated.p_next = (&flags as *const vk::MemoryAllocateFlagsInfo<'_>).cast();
                assert_eq!(
                    parse((&dedicated as *const vk::MemoryDedicatedAllocateInfo<'_>).cast()),
                    expected
                );
            }
        }
    }
    #[test]
    fn ambiguous_targets_and_duplicate_records_are_rejected() {
        unsafe {
            let info = vk::MemoryDedicatedAllocateInfo::default();
            assert_eq!(
                parse((&info as *const vk::MemoryDedicatedAllocateInfo<'_>).cast()),
                Err(INVALID)
            );
            let first = vk::MemoryDedicatedAllocateInfo::default().buffer(vk::Buffer::from_raw(1));
            let mut second =
                vk::MemoryDedicatedAllocateInfo::default().image(vk::Image::from_raw(2));
            second.p_next = (&first as *const vk::MemoryDedicatedAllocateInfo<'_>).cast();
            assert_eq!(
                parse((&second as *const vk::MemoryDedicatedAllocateInfo<'_>).cast()),
                Err(UNSUPPORTED)
            );
        }
    }
}
