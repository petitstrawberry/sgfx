use super::*;

#[test]
#[ignore = "requires a native GPU and freshly built SGFX ICD"]
fn storage_image_view_writes_and_samples_a_selected_layer_and_mip() {
    let context = Context::new();
    let output = Storage::new(&context);
    let module = naga::front::wgsl::parse_str(r#"
        @group(0) @binding(0) var destination:texture_storage_2d<rgba8unorm,write>;
        @group(0) @binding(1) var source:texture_2d<f32>;
        @group(0) @binding(2) var<storage,read_write> result:array<u32>;
        @compute @workgroup_size(1) fn write() { textureStore(destination,vec2<i32>(0),vec4(0.25,0.5,0.75,1.0)); }
        @compute @workgroup_size(1) fn read() {
            let pixel = vec4<u32>(round(textureLoad(source,vec2<i32>(0),0) * 255.0));
            result[0]=pixel.r; result[1]=pixel.g; result[2]=pixel.b; result[3]=pixel.a;
        }
    "#).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    let words = naga::back::spv::write_vec(&module, &info, &Default::default(), None).unwrap();
    unsafe {
        let device = &context.device;
        let image = device
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .extent(vk::Extent3D {
                        width: 8,
                        height: 8,
                        depth: 1,
                    })
                    .mip_levels(3)
                    .array_layers(6)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED),
                None,
            )
            .unwrap();
        let requirements = device.get_image_memory_requirements(image);
        let memory = device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(requirements.memory_type_bits.trailing_zeros()),
                None,
            )
            .unwrap();
        device.bind_image_memory(image, memory, 0).unwrap();
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(2)
            .level_count(1)
            .base_array_layer(4)
            .layer_count(1);
        let view = device
            .create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .subresource_range(range),
                None,
            )
            .unwrap();
        let bindings = [
            vk::DescriptorType::STORAGE_IMAGE,
            vk::DescriptorType::SAMPLED_IMAGE,
            vk::DescriptorType::STORAGE_BUFFER,
        ]
        .into_iter()
        .enumerate()
        .map(|(binding, ty)| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding as u32)
                .descriptor_type(ty)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect::<Vec<_>>();
        let set_layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
            .unwrap();
        let layouts = [set_layout];
        let layout = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                None,
            )
            .unwrap();
        let sizes = bindings
            .iter()
            .map(|binding| vk::DescriptorPoolSize {
                ty: binding.descriptor_type,
                descriptor_count: 1,
            })
            .collect::<Vec<_>>();
        let pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&sizes),
                None,
            )
            .unwrap();
        let set = device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )
            .unwrap()[0];
        let image_info = [vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(output.buffer)
            .range(BYTES)];
        device.update_descriptor_sets(
            &[
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&image_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&image_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffer_info),
            ],
            &[],
        );
        let shader = device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            .unwrap();
        let pipelines = [c"write", c"read"].map(|name| {
            device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .layout(layout)
                        .stage(
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::COMPUTE)
                                .module(shader)
                                .name(name),
                        )],
                    None,
                )
                .unwrap()[0]
        });
        context.reset();
        device
            .begin_command_buffer(context.command, &vk::CommandBufferBeginInfo::default())
            .unwrap();
        let transition = vk::ImageMemoryBarrier::default()
            .image(image)
            .subresource_range(range)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE);
        device.cmd_pipeline_barrier(
            context.command,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[transition],
        );
        device.cmd_bind_pipeline(
            context.command,
            vk::PipelineBindPoint::COMPUTE,
            pipelines[0],
        );
        device.cmd_bind_descriptor_sets(
            context.command,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[set],
            &[],
        );
        device.cmd_dispatch(context.command, 1, 1, 1);
        device.cmd_pipeline_barrier(
            context.command,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[transition
                .old_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)],
        );
        device.cmd_bind_pipeline(
            context.command,
            vk::PipelineBindPoint::COMPUTE,
            pipelines[1],
        );
        device.cmd_dispatch(context.command, 1, 1, 1);
        device.cmd_pipeline_barrier(
            context.command,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)],
            &[],
            &[],
        );
        device.end_command_buffer(context.command).unwrap();
        let fence = device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .unwrap();
        context.submit(fence).unwrap();
        device
            .wait_for_fences(&[fence], true, 5_000_000_000)
            .unwrap();
        assert_eq!(&output.read()[..4], &[64, 128, 191, 255]);
        device.destroy_fence(fence, None);
        for pipeline in pipelines {
            device.destroy_pipeline(pipeline, None);
        }
        device.destroy_shader_module(shader, None);
        device.destroy_descriptor_pool(pool, None);
        device.destroy_pipeline_layout(layout, None);
        device.destroy_descriptor_set_layout(set_layout, None);
        device.destroy_image_view(view, None);
        device.destroy_image(image, None);
        device.free_memory(memory, None);
    }
}

#[test]
#[ignore = "requires a native GPU and freshly built SGFX ICD"]
fn layered_upload_swizzled_views_and_specialization_execute_through_vulkan() {
    let context = Context::new();
    let upload = Storage::with_usage(&context, vk::BufferUsageFlags::TRANSFER_SRC);
    let output = Storage::new(&context);
    let module = naga::front::wgsl::parse_str(r#"
        @group(0) @binding(0) var image: texture_2d<f32>;
        @group(0) @binding(1) var image_sampler: sampler;
        @group(0) @binding(2) var<storage, read_write> output: array<u32>;
        @compute @workgroup_size(1) fn main() {
            let pixel = vec4<u32>(round(textureSampleLevel(image, image_sampler, vec2<f32>(0.25), 0.0) * 255.0));
            output[0] = pixel.r; output[1] = pixel.g; output[2] = pixel.b; output[3] = pixel.a;
            output[4] = 1009u;
        }
    "#).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    let mut words =
        naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
            .unwrap();
    // Turn a scalar constant into SpecId 7, preserving the shader's executable body.
    let mut offset = 5;
    let mut spec_id = None;
    let mut types_start = None;
    while offset < words.len() {
        let opcode = words[offset] & 0xffff;
        let count = (words[offset] >> 16) as usize;
        if types_start.is_none() && (19..=39).contains(&opcode) {
            types_start = Some(offset);
        }
        if opcode == 43 && count == 4 && words[offset + 3] == 1009 {
            words[offset] = (4 << 16) | 50;
            spec_id = Some(words[offset + 2]);
        }
        offset += count;
    }
    words.splice(
        types_start.unwrap()..types_start.unwrap(),
        [(4 << 16) | 71, spec_id.unwrap(), 1, 7],
    );
    unsafe {
        let device = &context.device;
        let mapping = device
            .map_memory(upload.memory, 0, BYTES, vk::MemoryMapFlags::empty())
            .unwrap()
            .cast::<u8>();
        for layer in 0..6usize {
            for row in 0..2usize {
                for col in 0..2usize {
                    *mapping.add(layer * 12 + row * 4 + col) =
                        (layer * 32 + row * 8 + col * 4) as u8;
                }
            }
        }
        device.unmap_memory(upload.memory);
        let image = device
            .create_image(
                &vk::ImageCreateInfo::default()
                    .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE)
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R8_UNORM)
                    .extent(vk::Extent3D {
                        width: 2,
                        height: 2,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(6)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED),
                None,
            )
            .unwrap();
        let requirements = device.get_image_memory_requirements(image);
        let memory = device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(requirements.memory_type_bits.trailing_zeros()),
                None,
            )
            .unwrap();
        device.bind_image_memory(image, memory, 0).unwrap();
        let maps = [
            vk::ComponentMapping::default(),
            vk::ComponentMapping {
                r: vk::ComponentSwizzle::ONE,
                g: vk::ComponentSwizzle::ONE,
                b: vk::ComponentSwizzle::ONE,
                a: vk::ComponentSwizzle::R,
            },
            vk::ComponentMapping {
                r: vk::ComponentSwizzle::B,
                g: vk::ComponentSwizzle::G,
                b: vk::ComponentSwizzle::R,
                a: vk::ComponentSwizzle::A,
            },
        ];
        let views = maps.map(|components| {
            device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .format(vk::Format::R8_UNORM)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .components(components)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .base_array_layer(3)
                                .layer_count(1),
                        ),
                    None,
                )
                .unwrap()
        });
        let sampler = device
            .create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )
            .unwrap();
        let bindings = [
            vk::DescriptorType::SAMPLED_IMAGE,
            vk::DescriptorType::SAMPLER,
            vk::DescriptorType::STORAGE_BUFFER,
        ]
        .into_iter()
        .enumerate()
        .map(|(binding, ty)| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding as u32)
                .descriptor_type(ty)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect::<Vec<_>>();
        let set_layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
            .unwrap();
        let layouts = [set_layout];
        let layout = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                None,
            )
            .unwrap();
        let sizes = bindings
            .iter()
            .map(|binding| vk::DescriptorPoolSize {
                ty: binding.descriptor_type,
                descriptor_count: 1,
            })
            .collect::<Vec<_>>();
        let pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&sizes),
                None,
            )
            .unwrap();
        let set = device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )
            .unwrap()[0];
        let shader = device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            .unwrap();
        let entries = [vk::SpecializationMapEntry {
            constant_id: 7,
            offset: 0,
            size: 4,
        }];
        let data = 5003u32.to_ne_bytes();
        let specialization = vk::SpecializationInfo::default()
            .map_entries(&entries)
            .data(&data);
        let pipeline = device
            .create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .layout(layout)
                    .stage(
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .module(shader)
                            .name(c"main")
                            .specialization_info(&specialization),
                    )],
                None,
            )
            .unwrap()[0];
        let fence = device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .unwrap();
        for (index, (view, expected)) in views
            .iter()
            .zip([
                [96, 0, 0, 255, 5003],
                [255, 255, 255, 96, 5003],
                [0, 0, 96, 255, 5003],
            ])
            .enumerate()
        {
            let image_info = [vk::DescriptorImageInfo::default()
                .image_view(*view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let sampler_info = [vk::DescriptorImageInfo::default().sampler(sampler)];
            let buffer_info = [vk::DescriptorBufferInfo::default()
                .buffer(output.buffer)
                .range(BYTES)];
            device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&image_info),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::SAMPLER)
                        .image_info(&sampler_info),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(2)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&buffer_info),
                ],
                &[],
            );
            context.reset();
            device
                .begin_command_buffer(context.command, &vk::CommandBufferBeginInfo::default())
                .unwrap();
            if index == 0 {
                let range = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(6);
                let transition = vk::ImageMemoryBarrier::default()
                    .image(image)
                    .subresource_range(range)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                device.cmd_pipeline_barrier(
                    context.command,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[transition],
                );
                device.cmd_copy_buffer_to_image(
                    context.command,
                    upload.buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[vk::BufferImageCopy::default()
                        .buffer_row_length(4)
                        .buffer_image_height(3)
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .layer_count(6),
                        )
                        .image_extent(vk::Extent3D {
                            width: 2,
                            height: 2,
                            depth: 1,
                        })],
                );
                let transition = transition
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ);
                device.cmd_pipeline_barrier(
                    context.command,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[transition],
                );
            }
            device.cmd_bind_pipeline(context.command, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                context.command,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &[set],
                &[],
            );
            device.cmd_dispatch(context.command, 1, 1, 1);
            device.end_command_buffer(context.command).unwrap();
            context.submit(fence).unwrap();
            device
                .wait_for_fences(&[fence], true, 5_000_000_000)
                .unwrap();
            assert_eq!(&output.read()[..5], expected, "component mapping {index}");
            device.reset_fences(&[fence]).unwrap();
        }
        device.destroy_fence(fence, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_shader_module(shader, None);
        device.destroy_descriptor_pool(pool, None);
        device.destroy_pipeline_layout(layout, None);
        device.destroy_descriptor_set_layout(set_layout, None);
        device.destroy_sampler(sampler, None);
        for view in views {
            device.destroy_image_view(view, None);
        }
        device.destroy_image(image, None);
        device.free_memory(memory, None);
    }
}
