//! Small Skia-style allocations must not exhaust the old 1024-object limit.
use super::*;

#[test]
#[ignore = "requires a native GPU and freshly built SGFX ICD"]
fn small_buffer_allocations_reach_advertised_limit_and_recycle() {
    let context = Context::new();
    unsafe {
        let physical = context.instance.enumerate_physical_devices().unwrap()[0];
        let limit = context
            .instance
            .get_physical_device_properties(physical)
            .limits
            .max_memory_allocation_count as usize;
        assert!(limit > 1024);
        let mut objects = Vec::new();
        for _ in 0..limit {
            let buffer = context
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(256)
                        .usage(vk::BufferUsageFlags::UNIFORM_BUFFER),
                    None,
                )
                .unwrap();
            let memory = context
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(256)
                        .memory_type_index(0),
                    None,
                )
                .unwrap();
            context
                .device
                .bind_buffer_memory(buffer, memory, 0)
                .unwrap();
            objects.push((buffer, memory));
        }
        assert_eq!(
            context.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(256)
                    .memory_type_index(0),
                None,
            ),
            Err(vk::Result::ERROR_TOO_MANY_OBJECTS)
        );
        let (buffer, memory) = objects.pop().unwrap();
        context.device.destroy_buffer(buffer, None);
        context.device.free_memory(memory, None);
        let buffer = context
            .device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(256)
                    .usage(vk::BufferUsageFlags::UNIFORM_BUFFER),
                None,
            )
            .unwrap();
        let memory = context
            .device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(256)
                    .memory_type_index(0),
                None,
            )
            .unwrap();
        context
            .device
            .bind_buffer_memory(buffer, memory, 0)
            .unwrap();
        objects.push((buffer, memory));
        for (buffer, memory) in objects {
            context.device.destroy_buffer(buffer, None);
            context.device.free_memory(memory, None);
        }
    }
}
