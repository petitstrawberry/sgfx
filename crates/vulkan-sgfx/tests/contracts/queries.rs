use super::*;

#[test]
#[ignore = "requires the built ICD and a native GPU with timestamp queries"]
fn gpu_timestamps_preserve_availability_stride_copy_and_host_reset() {
    let c = Context::configured(true);
    let output = Storage::with_usage(&c, vk::BufferUsageFlags::TRANSFER_DST);
    unsafe {
        let d = &c.device;
        let reset = ash::ext::host_query_reset::Device::new(&c.instance, d);
        let sync = ash::khr::synchronization2::Device::new(&c.instance, d);
        let pool = d
            .create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2),
                None,
            )
            .unwrap();
        let sentinel = 0xfeedabcd12345678;
        let mut initial = [[sentinel; 3]; 2];
        let flags = vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY;
        assert_eq!(
            d.get_query_pool_results(pool, 0, &mut initial, flags),
            Err(vk::Result::NOT_READY)
        );
        assert_eq!(initial, [[sentinel, 0, sentinel]; 2]);
        d.begin_command_buffer(c.command, &Default::default())
            .unwrap();
        d.cmd_reset_query_pool(c.command, pool, 0, 2);
        d.cmd_write_timestamp(c.command, vk::PipelineStageFlags::TOP_OF_PIPE, pool, 0);
        d.cmd_fill_buffer(c.command, output.buffer, 0, BYTES, 0);
        sync.cmd_write_timestamp2(c.command, vk::PipelineStageFlags2::ALL_COMMANDS, pool, 1);
        d.cmd_copy_query_pool_results(
            c.command,
            pool,
            0,
            2,
            output.buffer,
            8,
            24,
            flags | vk::QueryResultFlags::WAIT,
        );
        d.end_command_buffer(c.command).unwrap();
        let fence = d.create_fence(&Default::default(), None).unwrap();
        c.submit(fence).unwrap();
        d.wait_for_fences(&[fence], true, 5_000_000_000).unwrap();
        let mut results = [[sentinel; 3]; 2];
        d.get_query_pool_results(pool, 0, &mut results, flags | vk::QueryResultFlags::WAIT)
            .unwrap();
        assert!(results[0][0] > 0);
        assert!(results[1][0] >= results[0][0]);
        assert_eq!(results[0][1..], [1, sentinel]);
        assert_eq!(results[1][1..], [1, sentinel]);
        let words = output.read();
        let read = |index| u64::from(words[index]) | (u64::from(words[index + 1]) << 32);
        assert_eq!(read(2), results[0][0]);
        assert_eq!(read(4), 1);
        assert_eq!(read(8), results[1][0]);
        assert_eq!(read(10), 1);
        (reset.fp().reset_query_pool_ext)(d.handle(), pool, 0, 1);
        let mut unavailable = [[sentinel; 3]; 1];
        assert_eq!(
            d.get_query_pool_results(pool, 0, &mut unavailable, flags),
            Err(vk::Result::NOT_READY)
        );
        assert_eq!(unavailable, [[sentinel, 0, sentinel]]);
        let mut narrow = [[0u32; 2]; 1];
        d.get_query_pool_results(
            pool,
            1,
            &mut narrow,
            vk::QueryResultFlags::WITH_AVAILABILITY,
        )
        .unwrap();
        assert_eq!(narrow[0], [results[1][0] as u32, 1]);
        d.destroy_fence(fence, None);
        d.destroy_query_pool(pool, None);
    }
}

#[test]
#[ignore = "requires the built ICD and a native GPU"]
fn unsupported_statistics_and_precise_queries_are_not_enabled_silently() {
    let c = Context::configured(true);
    unsafe {
        let physical = c.instance.enumerate_physical_devices().unwrap()[0];
        let features = c.instance.get_physical_device_features(physical);
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(0)
            .queue_priorities(&[1.0])];
        if features.pipeline_statistics_query == vk::FALSE {
            let requested = vk::PhysicalDeviceFeatures::default().pipeline_statistics_query(true);
            assert!(matches!(
                c.instance.create_device(
                    physical,
                    &vk::DeviceCreateInfo::default()
                        .queue_create_infos(&queues)
                        .enabled_features(&requested),
                    None
                ),
                Err(vk::Result::ERROR_FEATURE_NOT_PRESENT)
            ));
            assert_eq!(
                c.device.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::PIPELINE_STATISTICS)
                        .query_count(1)
                        .pipeline_statistics(
                            vk::QueryPipelineStatisticFlags::FRAGMENT_SHADER_INVOCATIONS
                        ),
                    None
                ),
                Err(vk::Result::ERROR_FEATURE_NOT_PRESENT)
            );
        }
        if features.occlusion_query_precise == vk::FALSE {
            let pool = c
                .device
                .create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::OCCLUSION)
                        .query_count(1),
                    None,
                )
                .unwrap();
            c.device
                .begin_command_buffer(c.command, &Default::default())
                .unwrap();
            c.device
                .cmd_begin_query(c.command, pool, 0, vk::QueryControlFlags::PRECISE);
            assert_eq!(
                c.device.end_command_buffer(c.command),
                Err(vk::Result::ERROR_FEATURE_NOT_PRESENT)
            );
            c.device.destroy_query_pool(pool, None);
        }
    }
}
