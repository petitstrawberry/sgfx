//! Actual device execution, including a host signal after queue acceptance.
use super::*;

#[test]
#[ignore = "requires the built ICD and a native GPU"]
fn sync2_transfers_events_and_mixed_semaphores_retire_before_the_fence() {
    let context = Context::configured(true);
    let source = Storage::with_usage(
        &context,
        vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
    );
    let destination = Storage::with_usage(&context, vk::BufferUsageFlags::TRANSFER_DST);
    unsafe {
        let device = &context.device;
        let sync = ash::khr::synchronization2::Device::new(&context.instance, device);
        let timeline_api = ash::khr::timeline_semaphore::Device::new(&context.instance, device);
        let mut semaphore_type =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        let timeline = device
            .create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut semaphore_type),
                None,
            )
            .unwrap();
        let binary = device
            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            .unwrap();
        let event = device
            .create_event(
                &vk::EventCreateInfo::default().flags(vk::EventCreateFlags::DEVICE_ONLY),
                None,
            )
            .unwrap();
        let fence = device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .unwrap();
        device
            .begin_command_buffer(context.command, &vk::CommandBufferBeginInfo::default())
            .unwrap();
        device.cmd_fill_buffer(context.command, source.buffer, 0, BYTES, 0x12345678);
        let buffers = [vk::BufferMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::CLEAR)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .src_queue_family_index(0)
            .dst_queue_family_index(0)
            .buffer(source.buffer)
            .offset(4)
            .size(16)];
        let dependency = vk::DependencyInfo::default().buffer_memory_barriers(&buffers);
        sync.cmd_set_event2(context.command, event, &dependency);
        sync.cmd_wait_events2(context.command, &[event], &[dependency]);
        device.cmd_copy_buffer(
            context.command,
            source.buffer,
            destination.buffer,
            &[vk::BufferCopy::default().size(BYTES)],
        );
        let host = [vk::BufferMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(destination.buffer)
            .size(vk::WHOLE_SIZE)];
        sync.cmd_pipeline_barrier2(
            context.command,
            &vk::DependencyInfo::default().buffer_memory_barriers(&host),
        );
        sync.cmd_reset_event2(
            context.command,
            event,
            vk::PipelineStageFlags2::ALL_COMMANDS,
        );
        device.end_command_buffer(context.command).unwrap();
        let waits = [vk::SemaphoreSubmitInfo::default()
            .semaphore(timeline)
            .value(1)
            .stage_mask(vk::PipelineStageFlags2::COPY)];
        let commands = [vk::CommandBufferSubmitInfo::default()
            .command_buffer(context.command)
            .device_mask(0)];
        // Nonzero binary semaphore values are ignored, not timeline values.
        let first_signal = [vk::SemaphoreSubmitInfo::default()
            .semaphore(binary)
            .value(73)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let second_wait = [vk::SemaphoreSubmitInfo::default()
            .semaphore(binary)
            .value(999)
            .stage_mask(vk::PipelineStageFlags2::NONE)];
        let second_signal = [vk::SemaphoreSubmitInfo::default()
            .semaphore(timeline)
            .value(2)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        sync.queue_submit2(
            context.queue,
            &[
                vk::SubmitInfo2::default()
                    .wait_semaphore_infos(&waits)
                    .command_buffer_infos(&commands)
                    .signal_semaphore_infos(&first_signal),
                vk::SubmitInfo2::default()
                    .wait_semaphore_infos(&second_wait)
                    .signal_semaphore_infos(&second_signal),
            ],
            fence,
        )
        .unwrap();
        assert!(!device.get_fence_status(fence).unwrap());
        assert_eq!(
            timeline_api.get_semaphore_counter_value(timeline).unwrap(),
            0
        );
        timeline_api
            .signal_semaphore(
                &vk::SemaphoreSignalInfo::default()
                    .semaphore(timeline)
                    .value(1),
            )
            .unwrap();
        device
            .wait_for_fences(&[fence], true, 5_000_000_000)
            .unwrap();
        assert_eq!(
            timeline_api.get_semaphore_counter_value(timeline).unwrap(),
            2
        );
        assert_eq!(destination.read(), vec![0x12345678; WORDS]);
        device.destroy_fence(fence, None);
        device.destroy_event(event, None);
        device.destroy_semaphore(binary, None);
        device.destroy_semaphore(timeline, None);
    }
}
