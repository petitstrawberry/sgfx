//! Headless Vulkan compute/readback smoke test, using the ICD or Vulkan loader.
//!
//! Build the driver first: `cargo build -p vulkan-sgfx`.
//! Run: `cargo run -p vulkan-sgfx --example headless -- [ICD_LIBRARY] [ITERATIONS]`.
//! `SGFX_ICD_LIBRARY` can also select the library; iterations default to 16.
//! To exercise loader discovery, set `SGFX_VULKAN_LOADER` to a Vulkan loader
//! library and `VK_DRIVER_FILES` to the SGFX ICD manifest instead.
//! This checks actual shader output, then reports CPU wall time spent recording
//! commands and inside vkQueueSubmit. Submission time includes any synchronous
//! work performed by the driver; it is not a GPU execution timestamp.

use ash::{Entry, vk};
use std::{error::Error, path::PathBuf, time::Duration, time::Instant};

const WORD_COUNT: usize = 256;
const BUFFER_BYTES: vk::DeviceSize = (WORD_COUNT * size_of::<u32>()) as vk::DeviceSize;

fn shader_words() -> Result<Vec<u32>, Box<dyn Error>> {
    let module = naga::front::wgsl::parse_str(include_str!("../tests/assets/fill.wgsl"))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
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

fn default_library() -> Result<PathBuf, Box<dyn Error>> {
    let executable = std::env::current_exe()?;
    let profile_dir = executable
        .parent()
        .and_then(|examples| examples.parent())
        .ok_or("cannot locate the Cargo profile directory")?;
    Ok(profile_dir.join(format!(
        "{}vulkan_sgfx{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX,
    )))
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let direct_library = args
        .next()
        .or_else(|| std::env::var_os("SGFX_ICD_LIBRARY"))
        .map(PathBuf::from);
    let loader_path = std::env::var_os("SGFX_VULKAN_LOADER").map(PathBuf::from);
    let use_loader = loader_path.is_some();
    let library_path = loader_path
        .or(direct_library)
        .map(Ok)
        .unwrap_or_else(default_library)?;
    let iterations = args
        .next()
        .map(|arg| arg.to_string_lossy().parse::<u32>())
        .transpose()?
        .unwrap_or(16);
    if iterations == 0 || args.next().is_some() {
        return Err("usage: headless [ICD_LIBRARY] [ITERATIONS > 0]".into());
    }
    let words = shader_words()?;

    // The library is kept alive through every call and resource destruction.
    let library = unsafe { libloading::Library::new(&library_path)? };
    let symbol: &[u8] = if use_loader {
        b"vkGetInstanceProcAddr\0"
    } else {
        b"vk_icdGetInstanceProcAddr\0"
    };
    let get_instance_proc_addr = unsafe { *library.get::<vk::PFN_vkGetInstanceProcAddr>(symbol)? };
    let entry = unsafe {
        Entry::from_static_fn(ash::StaticFn {
            get_instance_proc_addr,
        })
    };
    run(&entry, &words, iterations)?;
    let source = if use_loader { "Vulkan loader" } else { "ICD" };
    println!("{source}: {}", library_path.display());
    Ok(())
}

// Vulkan handles below are used only with their creating instance/device.
// Host access is bounded by BUFFER_BYTES and waits for the submission fence.
fn run(entry: &Entry, words: &[u32], iterations: u32) -> Result<(), Box<dyn Error>> {
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
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None)?;
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
        let pipeline_layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(&descriptor_layouts),
            None,
        )?;
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
            .map_err(|(_, error)| error)?[0];
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
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            device.cmd_dispatch(command_buffer, WORD_COUNT as u32 / 64, 1, 1);
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
            device.end_command_buffer(command_buffer)?;
            recording += recording_start.elapsed();
            let command_buffers = [command_buffer];
            let submits = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
            let submission_start = Instant::now();
            device.queue_submit(queue, &submits, fence)?;
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
                let expected = index as u32 * 3 + 7;
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
        println!("iterations: {iterations}; dispatch: 4 workgroups × 64 invocations");
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
