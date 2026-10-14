//! Exercise the memory2 ABI against real device objects and GPU transfers.
use super::*;
#[test]
#[ignore = "requires the built ICD and a native GPU"]
fn memory2_bindings_feed_real_gpu_clear_and_readback() {
    gpu_clear_and_readback(vk::QUEUE_FAMILY_IGNORED);
}

#[test]
#[ignore = "requires the built ICD and a native GPU"]
fn explicit_same_family_image_and_buffer_barriers_feed_gpu_readback() {
    gpu_clear_and_readback(0);
}

fn gpu_clear_and_readback(queue_family: u32) {
    let context = Context::new();
    unsafe {
        let device = &context.device;
        let buffer = device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(64)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
            .unwrap();
        let old = device.get_buffer_memory_requirements(buffer);
        let mut dedicated =
            vk::MemoryDedicatedRequirements::default().requires_dedicated_allocation(true);
        let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated);
        device.get_buffer_memory_requirements2(
            &vk::BufferMemoryRequirementsInfo2::default().buffer(buffer),
            &mut requirements,
        );
        assert_eq!(requirements.memory_requirements.size, old.size);
        assert_eq!(requirements.memory_requirements.alignment, old.alignment);
        assert_eq!(
            requirements.memory_requirements.memory_type_bits,
            old.memory_type_bits
        );
        assert_eq!(dedicated.requires_dedicated_allocation, vk::FALSE);
        let buffer_memory = device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(old.size)
                    .memory_type_index(0),
                None,
            )
            .unwrap();
        device
            .bind_buffer_memory2(&[vk::BindBufferMemoryInfo::default()
                .buffer(buffer)
                .memory(buffer_memory)])
            .unwrap();
        let image = device
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .extent(vk::Extent3D {
                        width: 4,
                        height: 4,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
            .unwrap();
        let old = device.get_image_memory_requirements(image);
        let mut requirements = vk::MemoryRequirements2::default();
        device.get_image_memory_requirements2(
            &vk::ImageMemoryRequirementsInfo2::default().image(image),
            &mut requirements,
        );
        assert_eq!(requirements.memory_requirements.size, old.size);
        let image_memory = device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(old.size)
                    .memory_type_index(0),
                None,
            )
            .unwrap();
        device
            .bind_image_memory2(&[vk::BindImageMemoryInfo::default()
                .image(image)
                .memory(image_memory)])
            .unwrap();
        let binding = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)];
        let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding);
        let mut support = vk::DescriptorSetLayoutSupport::default();
        device.get_descriptor_set_layout_support(&info, &mut support);
        assert_eq!(support.supported, vk::TRUE);
        let layout = device.create_descriptor_set_layout(&info, None).unwrap();
        device.destroy_descriptor_set_layout(layout, None);
        let invalid = [vk::DescriptorSetLayoutBinding {
            descriptor_count: 2,
            ..binding[0]
        }];
        let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&invalid);
        device.get_descriptor_set_layout_support(&info, &mut support);
        assert_eq!(support.supported, vk::FALSE);
        assert_eq!(
            device.create_descriptor_set_layout(&info, None),
            Err(vk::Result::ERROR_FEATURE_NOT_PRESENT)
        );
        device
            .begin_command_buffer(context.command, &vk::CommandBufferBeginInfo::default())
            .unwrap();
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        let barrier = vk::ImageMemoryBarrier::default()
            .src_queue_family_index(queue_family)
            .dst_queue_family_index(queue_family)
            .image(image)
            .subresource_range(range);
        device.cmd_pipeline_barrier(
            context.command,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)],
        );
        device.cmd_clear_color_image(
            context.command,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &vk::ClearColorValue {
                float32: [1.0, 0.0, 0.0, 1.0],
            },
            &[range],
        );
        device.cmd_pipeline_barrier(
            context.command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)],
        );
        device.cmd_copy_image_to_buffer(
            context.command,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            buffer,
            &[vk::BufferImageCopy::default()
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D {
                    width: 4,
                    height: 4,
                    depth: 1,
                })],
        );
        device.cmd_pipeline_barrier(
            context.command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[],
            &[vk::BufferMemoryBarrier::default()
                .src_queue_family_index(queue_family)
                .dst_queue_family_index(queue_family)
                .buffer(buffer)
                .size(vk::WHOLE_SIZE)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)],
            &[],
        );
        device.end_command_buffer(context.command).unwrap();
        context.submit(vk::Fence::null()).unwrap();
        device.queue_wait_idle(context.queue).unwrap();
        let mapped = device
            .map_memory(buffer_memory, 0, 64, vk::MemoryMapFlags::empty())
            .unwrap();
        let bytes = std::slice::from_raw_parts(mapped.cast::<u8>(), 64);
        assert!(bytes.chunks_exact(4).all(|pixel| pixel == [255, 0, 0, 255]));
        device.unmap_memory(buffer_memory);
        device.destroy_buffer(buffer, None);
        device.destroy_image(image, None);
        device.free_memory(buffer_memory, None);
        device.free_memory(image_memory, None);
    }
}
