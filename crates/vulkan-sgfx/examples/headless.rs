//! Headless Vulkan compute/readback smoke test through the installed Vulkan loader.
//!
//! Build the driver first, then select its manifest externally with VK_DRIVER_FILES.
//! Run: `headless [ITERATIONS] [--push-constants]`; iterations default to 16.
//! This checks actual shader output, then reports CPU wall time spent recording
//! commands and inside vkQueueSubmit. Submission time includes any synchronous
//! work performed by the driver; it is not a GPU execution timestamp.

use ash::{Entry, vk};
use std::{error::Error, time::Duration, time::Instant};

const WORD_COUNT: usize = 256;
const BUFFER_BYTES: vk::DeviceSize = (WORD_COUNT * size_of::<u32>()) as vk::DeviceSize;

fn shader_words(push_constants: bool) -> Result<Vec<u32>, Box<dyn Error>> {
    let source = if push_constants {
        include_str!("../tests/assets/fill_push_constants.wgsl")
    } else {
        include_str!("../tests/assets/fill.wgsl")
    };
    let module = naga::front::wgsl::parse_str(source)?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::PUSH_CONSTANT,
    )
    .validate(&module)?;
    let pipeline = naga::back::spv::PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: "main".into(),
    };
    Ok(naga::back::spv::write_vec(
        &module,
        &info,
        &naga::back::spv::Options::default(),
        Some(&pipeline),
    )?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut iterations = None;
    let mut push_constants = false;
    for arg in std::env::args().skip(1) {
        if arg == "--push-constants" && !push_constants {
            push_constants = true;
        } else if iterations.is_none() && !arg.starts_with('-') {
            iterations = Some(arg.parse::<u32>()?);
        } else {
            return Err("usage: headless [ITERATIONS > 0] [--push-constants]".into());
        }
    }
    let iterations = iterations.unwrap_or(16);
    if iterations == 0 {
        return Err("iterations must be greater than zero".into());
    }
    let words = shader_words(push_constants)?;
    let entry = unsafe { Entry::load()? };
    run(&entry, &words, iterations, push_constants)
}

// Vulkan handles below are used only with their creating instance/device.
// Host access is bounded by BUFFER_BYTES and waits for the submission fence.
fn run(
    entry: &Entry,
    words: &[u32],
    iterations: u32,
    push_constants: bool,
) -> Result<(), Box<dyn Error>> {
    unsafe {
        let application = vk::ApplicationInfo::default()
            .application_name(c"sgfx-headless-smoke")
            .api_version(vk::API_VERSION_1_0);
        let instance = entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(&application),
            None,
        )?;
        let (physical_device, queue_family) = instance
            .enumerate_physical_devices()?
            .into_iter()
            .find_map(|physical| {
                instance
                    .get_physical_device_queue_family_properties(physical)
                    .iter()
                    .enumerate()
                    .find(|(_, properties)| {
                        properties.queue_count > 0
                            && properties.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    })
                    .map(|(index, _)| (physical, index as u32))
            })
            .ok_or("no Vulkan compute queue is available")?;
        let properties = instance.get_physical_device_properties(physical_device);
        println!(
            "Device: {}",
            std::ffi::CStr::from_ptr(properties.device_name.as_ptr()).to_string_lossy()
        );
        let queue_priorities = [1.0];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&queue_priorities)];
        let device = instance.create_device(
            physical_device,
            &vk::DeviceCreateInfo::default().queue_create_infos(&queue_info),
            None,
        )?;
        let queue = device.get_device_queue(queue_family, 0);
        let buffer = device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(BUFFER_BYTES)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )?;
        let requirements = device.get_buffer_memory_requirements(buffer);
        let memory_properties = instance.get_physical_device_memory_properties(physical_device);
        let memory_type_index = memory_properties.memory_types
            [..memory_properties.memory_type_count as usize]
            .iter()
            .enumerate()
            .find(|(index, memory_type)| {
                requirements.memory_type_bits & (1 << index) != 0
                    && memory_type.property_flags.contains(
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )
            })
            .map(|(index, _)| index as u32)
            .ok_or("no host-visible coherent storage-buffer memory type is available")?;
        let memory = device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type_index),
            None,
        )?;
        device.bind_buffer_memory(buffer, memory, 0)?;
        let mapped = device.map_memory(memory, 0, BUFFER_BYTES, vk::MemoryMapFlags::empty())?;
        std::ptr::write_bytes(mapped.cast::<u8>(), 0, BUFFER_BYTES as usize);
        device.unmap_memory(memory);

        let shader = device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None)
            .map_err(|error| format!("shader-module creation failed: {error:?}"))?;
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let descriptor_layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )?;
        let descriptor_layouts = [descriptor_layout];
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(16)];
        let pipeline_layout = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&descriptor_layouts)
                    .push_constant_ranges(if push_constants { &ranges } else { &[] }),
                None,
            )
            .map_err(|error| format!("pipeline-layout creation failed: {error:?}"))?;
        let pipeline_info = [vk::ComputePipelineCreateInfo::default()
            .stage(
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(shader)
                    .name(c"main"),
            )
            .layout(pipeline_layout)];
        let pipeline = device
            .create_compute_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
            .map_err(|(_, error)| format!("compute-pipeline creation failed: {error:?}"))?[0];
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)];
        let descriptor_pool = device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes),
            None,
        )?;
        let descriptor_set = device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&descriptor_layouts),
        )?[0];
        let buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(0)
            .range(BUFFER_BYTES)];
        device.update_descriptor_sets(
            &[vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buffer_info)],
            &[],
        );

        let command_pool = device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(queue_family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )?;
        let command_buffer = device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )?[0];
        let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
        let mut recording = Duration::ZERO;
        let mut submission = Duration::ZERO;
        for iteration in 0..iterations {
            if iteration > 0 {
                device.reset_fences(&[fence])?;
                device
                    .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;
            }
            let recording_start = Instant::now();
            device.begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            if push_constants {
                let values = [0u32, 128, 3, 7 + iteration];
                device.cmd_push_constants(
                    command_buffer,
                    pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &values
                        .into_iter()
                        .flat_map(u32::to_ne_bytes)
                        .collect::<Vec<_>>(),
                );
            }
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            if push_constants {
                device.cmd_dispatch(command_buffer, 2, 1, 1);
                device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(
                            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                        )],
                    &[],
                    &[],
                );
                device.cmd_push_constants(
                    command_buffer,
                    pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &128u32.to_ne_bytes(),
                );
                device.cmd_push_constants(
                    command_buffer,
                    pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    8,
                    &[5u32, 19 + iteration]
                        .into_iter()
                        .flat_map(u32::to_ne_bytes)
                        .collect::<Vec<_>>(),
                );
                device.cmd_dispatch(command_buffer, 2, 1, 1);
            } else {
                device.cmd_dispatch(command_buffer, WORD_COUNT as u32 / 64, 1, 1);
            }
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)],
                &[],
                &[],
            );
            device
                .end_command_buffer(command_buffer)
                .map_err(|error| format!("command recording failed: {error:?}"))?;
            recording += recording_start.elapsed();
            let command_buffers = [command_buffer];
            let submits = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
            let submission_start = Instant::now();
            device
                .queue_submit(queue, &submits, fence)
                .map_err(|error| format!("queue submission failed: {error:?}"))?;
            submission += submission_start.elapsed();
            // Fence waits are excluded from both timing measurements above.
            device.wait_for_fences(&[fence], true, u64::MAX)?;
        }

        let mapped = device.map_memory(memory, 0, BUFFER_BYTES, vk::MemoryMapFlags::empty())?;
        // Read bytes instead of casting the mapping to u32 to avoid depending on
        // any host allocator's alignment in a software ICD.
        let output = std::slice::from_raw_parts(mapped.cast::<u8>(), BUFFER_BYTES as usize);
        let mismatch = output
            .chunks_exact(4)
            .enumerate()
            .find_map(|(index, bytes)| {
                let observed = u32::from_ne_bytes(bytes.try_into().unwrap());
                let expected = if push_constants {
                    let (multiplier, bias) = if index < 128 { (3, 7) } else { (5, 19) };
                    index as u32 * multiplier + bias + iterations - 1
                } else {
                    index as u32 * 3 + 7
                };
                (observed != expected).then_some((index, observed, expected))
            });
        device.unmap_memory(memory);

        device.destroy_fence(fence, None);
        device.destroy_command_pool(command_pool, None);
        device.destroy_descriptor_pool(descriptor_pool, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_descriptor_set_layout(descriptor_layout, None);
        device.destroy_shader_module(shader, None);
        device.destroy_buffer(buffer, None);
        device.free_memory(memory, None);
        device.destroy_device(None);
        instance.destroy_instance(None);

        if let Some((index, observed, expected)) = mismatch {
            return Err(format!(
                "compute readback mismatch at {index}: got {observed}, expected {expected}"
            )
            .into());
        }
        println!("PASS: SPIR-V compute wrote all {WORD_COUNT} expected u32 values");
        if push_constants {
            println!(
                "PASS: two dispatches retain distinct incremental push constants, including a value set before pipeline binding"
            );
        }
        println!("iterations: {iterations}; total work: 4 workgroups × 64 invocations");
        println!(
            "command_record_cpu_ns: total={} mean={} (begin/bind/dispatch/barrier/end; excludes reset)",
            recording.as_nanos(),
            recording.as_nanos() / u128::from(iterations),
        );
        println!(
            "queue_submit_cpu_ns: total={} mean={} (vkQueueSubmit call; excludes fence wait; includes synchronous driver work)",
            submission.as_nanos(),
            submission.as_nanos() / u128::from(iterations),
        );
        Ok(())
    }
}
