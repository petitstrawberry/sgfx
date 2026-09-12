//! Opt-in integration checks against the built experimental ICD.
//!
//! Run `cargo build -p vulkan-sgfx`, then
//! `cargo test -p vulkan-sgfx --test contracts -- --ignored`.
//! `SGFX_ICD_LIBRARY` overrides the library path. These require a native GPU.
//! Negative cases exercise this ICD's defensive errors for invalid usage;
//! portable Vulkan applications must not make those invalid calls.

use ash::{Entry, vk};
use std::path::PathBuf;

#[path = "contracts/errors.rs"]
mod errors;

const WORDS: usize = 256;
const BYTES: u64 = (WORDS * size_of::<u32>()) as u64;

struct Context {
    device: ash::Device,
    queue: vk::Queue,
    command: vk::CommandBuffer,
    pool: vk::CommandPool,
    memory: vk::PhysicalDeviceMemoryProperties,
    instance: ash::Instance,
    _entry: Entry,
    _library: libloading::Library,
}

impl Context {
    fn new() -> Self {
        unsafe {
            let path = std::env::var_os("SGFX_ICD_LIBRARY")
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    std::env::current_exe()
                        .unwrap()
                        .parent()
                        .unwrap()
                        .parent()
                        .unwrap()
                        .join(format!(
                            "{}vulkan_sgfx{}",
                            std::env::consts::DLL_PREFIX,
                            std::env::consts::DLL_SUFFIX
                        ))
                });
            let library = libloading::Library::new(&path).unwrap_or_else(|error| {
                panic!("build the ICD first ({}): {error}", path.display())
            });
            let get_instance_proc_addr = *library
                .get::<vk::PFN_vkGetInstanceProcAddr>(b"vk_icdGetInstanceProcAddr\0")
                .unwrap();
            let entry = Entry::from_static_fn(ash::StaticFn {
                get_instance_proc_addr,
            });
            let application = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_0);
            let instance = entry
                .create_instance(
                    &vk::InstanceCreateInfo::default().application_info(&application),
                    None,
                )
                .unwrap();
            let physical = instance.enumerate_physical_devices().unwrap()[0];
            let family = instance
                .get_physical_device_queue_family_properties(physical)
                .iter()
                .position(|q| q.queue_count > 0 && q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .unwrap() as u32;
            let priorities = [1.0];
            let queues = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&priorities)];
            let device = instance
                .create_device(
                    physical,
                    &vk::DeviceCreateInfo::default().queue_create_infos(&queues),
                    None,
                )
                .expect("a native GPU adapter is required for this opt-in test");
            let queue = device.get_device_queue(family, 0);
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .unwrap();
            let command = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .unwrap()[0];
            let memory = instance.get_physical_device_memory_properties(physical);
            Self {
                device,
                queue,
                command,
                pool,
                memory,
                instance,
                _entry: entry,
                _library: library,
            }
        }
    }

    fn reset(&self) {
        unsafe {
            self.device
                .reset_command_buffer(self.command, vk::CommandBufferResetFlags::empty())
                .unwrap();
        }
    }

    fn submit(&self, fence: vk::Fence) -> Result<(), vk::Result> {
        let commands = [self.command];
        unsafe {
            self.device.queue_submit(
                self.queue,
                &[vk::SubmitInfo::default().command_buffers(&commands)],
                fence,
            )
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

struct Storage<'a> {
    context: &'a Context,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

impl<'a> Storage<'a> {
    fn new(context: &'a Context) -> Self {
        Self::with_usage(context, vk::BufferUsageFlags::STORAGE_BUFFER)
    }

    fn with_usage(context: &'a Context, usage: vk::BufferUsageFlags) -> Self {
        unsafe {
            let buffer = context
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(BYTES)
                        .usage(usage)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .unwrap();
            let requirements = context.device.get_buffer_memory_requirements(buffer);
            let memory_type = context.memory.memory_types
                [..context.memory.memory_type_count as usize]
                .iter()
                .enumerate()
                .position(|(index, memory)| {
                    requirements.memory_type_bits & (1u32 << index) != 0
                        && memory.property_flags.contains(
                            vk::MemoryPropertyFlags::HOST_VISIBLE
                                | vk::MemoryPropertyFlags::HOST_COHERENT,
                        )
                })
                .unwrap() as u32;
            let memory = context
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(requirements.size)
                        .memory_type_index(memory_type),
                    None,
                )
                .unwrap();
            context
                .device
                .bind_buffer_memory(buffer, memory, 0)
                .unwrap();
            let mapping = context
                .device
                .map_memory(memory, 0, BYTES, vk::MemoryMapFlags::empty())
                .unwrap();
            std::ptr::write_bytes(mapping.cast::<u8>(), 0, BYTES as usize);
            context.device.unmap_memory(memory);
            Self {
                context,
                buffer,
                memory,
            }
        }
    }

    fn read(&self) -> Vec<u32> {
        unsafe {
            let mapping = self
                .context
                .device
                .map_memory(self.memory, 0, BYTES, vk::MemoryMapFlags::empty())
                .unwrap();
            let words = std::slice::from_raw_parts(mapping.cast::<u8>(), BYTES as usize)
                .chunks_exact(4)
                .map(|bytes| u32::from_ne_bytes(bytes.try_into().unwrap()))
                .collect();
            self.context.device.unmap_memory(self.memory);
            words
        }
    }
}

impl Drop for Storage<'_> {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_buffer(self.buffer, None);
            self.context.device.free_memory(self.memory, None);
        }
    }
}

struct Compute<'a> {
    context: &'a Context,
    shader: vk::ShaderModule,
    descriptor_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
}

impl<'a> Compute<'a> {
    fn new(context: &'a Context) -> Self {
        let module = naga::front::wgsl::parse_str(include_str!("assets/fill.wgsl")).unwrap();
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
        let words = naga::back::spv::write_vec(
            &module,
            &info,
            &naga::back::spv::Options::default(),
            Some(&naga::back::spv::PipelineOptions {
                shader_stage: naga::ShaderStage::Compute,
                entry_point: "main".into(),
            }),
        )
        .unwrap();
        unsafe {
            let device = &context.device;
            let shader = device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
                .unwrap();
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)];
            let descriptor_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .unwrap();
            let layouts = [descriptor_layout];
            let layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                    None,
                )
                .unwrap();
            let pipeline = device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .layout(layout)
                        .stage(
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::COMPUTE)
                                .module(shader)
                                .name(c"main"),
                        )],
                    None,
                )
                .unwrap()[0];
            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)];
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
            Self {
                context,
                shader,
                descriptor_layout,
                layout,
                pipeline,
                pool,
                set,
            }
        }
    }

    fn update(&self, storage: &Storage<'_>) {
        let buffers = [vk::DescriptorBufferInfo::default()
            .buffer(storage.buffer)
            .range(BYTES)];
        unsafe {
            self.context.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(self.set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffers)],
                &[],
            );
        }
    }

    fn record(&self, descriptors_first: bool) {
        self.record_commands(descriptors_first, None);
    }

    fn record_commands(&self, descriptors_first: bool, copy: Option<(&Storage<'_>, &Storage<'_>)>) {
        let device = &self.context.device;
        let command = self.context.command;
        unsafe {
            device
                .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
                .unwrap();
            let bind_set = || {
                device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    self.layout,
                    0,
                    &[self.set],
                    &[],
                );
            };
            if descriptors_first {
                bind_set();
            }
            device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            if !descriptors_first {
                bind_set();
            }
            device.cmd_dispatch(command, 4, 1, 1);
            let (source_stage, source_access) = if let Some((source, destination)) = copy {
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                        .dst_access_mask(vk::AccessFlags::MEMORY_READ)],
                    &[],
                    &[],
                );
                device.cmd_copy_buffer(
                    command,
                    source.buffer,
                    destination.buffer,
                    &[vk::BufferCopy::default().size(BYTES)],
                );
                (
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::TRANSFER_WRITE,
                )
            } else {
                (
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::AccessFlags::SHADER_WRITE,
                )
            };
            device.cmd_pipeline_barrier(
                command,
                source_stage,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(source_access)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)],
                &[],
                &[],
            );
            device.end_command_buffer(command).unwrap();
        }
    }
}

impl Drop for Compute<'_> {
    fn drop(&mut self) {
        unsafe {
            let device = &self.context.device;
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_layout, None);
            device.destroy_shader_module(self.shader, None);
        }
    }
}

#[test]
#[ignore = "requires a built SGFX ICD and a native GPU adapter"]
fn descriptor_updates_before_binding_and_both_binding_orders_execute() {
    let context = Context::new();
    let first = Storage::new(&context);
    let second = Storage::new(&context);
    let compute = Compute::new(&context);
    let expected: Vec<_> = (0..WORDS as u32).map(|index| index * 3 + 7).collect();
    unsafe {
        let fence = context
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .unwrap();
        // Updates happen before binding. Without UPDATE_AFTER_BIND, updates to
        // sets referenced by executable command buffers invalidate those buffers.
        compute.update(&first);
        compute.update(&second);
        compute.record(true);
        context.submit(fence).unwrap();
        context
            .device
            .wait_for_fences(&[fence], true, u64::MAX)
            .unwrap();
        assert_eq!(first.read(), vec![0; WORDS]);
        assert_eq!(second.read(), expected);

        context.reset();
        context.device.reset_fences(&[fence]).unwrap();
        compute.update(&first);
        compute.record(false);
        context.submit(fence).unwrap();
        context
            .device
            .wait_for_fences(&[fence], true, u64::MAX)
            .unwrap();
        assert_eq!(first.read(), expected);
        context.device.destroy_fence(fence, None);
    }
}

#[test]
#[ignore = "requires a built SGFX ICD and a native GPU adapter"]
fn destroying_a_recorded_storage_buffer_rejects_submit_without_signaling_fence() {
    let context = Context::new();
    let storage = Storage::new(&context);
    let compute = Compute::new(&context);
    compute.update(&storage);
    compute.record(false);
    drop(storage);
    unsafe {
        let fence = context
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .unwrap();
        assert_eq!(
            context.submit(fence),
            Err(vk::Result::ERROR_INITIALIZATION_FAILED)
        );
        assert_eq!(context.device.get_fence_status(fence), Ok(false));
        assert_eq!(
            context.device.wait_for_fences(&[fence], true, 0),
            Err(vk::Result::TIMEOUT)
        );
        context.device.destroy_fence(fence, None);
    }
}

#[test]
#[ignore = "requires a built SGFX ICD and a native GPU adapter"]
fn descriptor_update_invalidates_executable_commands_until_rerecorded() {
    let context = Context::new();
    let first = Storage::new(&context);
    let second = Storage::new(&context);
    let compute = Compute::new(&context);
    compute.update(&first);
    compute.record(false);
    // Vulkan 1.0 has no UPDATE_AFTER_BIND feature. This host update is allowed,
    // but invalidates the executable command buffer that referenced the set.
    // https://docs.vulkan.org/refpages/latest/refpages/source/vkUpdateDescriptorSets.html
    compute.update(&second);
    unsafe {
        let fence = context
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .unwrap();
        assert_eq!(
            context.submit(fence),
            Err(vk::Result::ERROR_INITIALIZATION_FAILED)
        );
        assert_eq!(context.device.get_fence_status(fence), Ok(false));

        context.reset();
        compute.record(true);
        context.submit(fence).unwrap();
        context
            .device
            .wait_for_fences(&[fence], true, u64::MAX)
            .unwrap();
        assert_eq!(first.read(), vec![0; WORDS]);
        assert_eq!(
            second.read(),
            (0..WORDS as u32).map(|i| i * 3 + 7).collect::<Vec<_>>()
        );
        context.device.destroy_fence(fence, None);
    }
}

#[test]
#[ignore = "requires a built SGFX ICD and a native GPU adapter"]
fn repeated_buffer_destruction_reclaims_idle_resource_capacity() {
    let context = Context::new();
    // Context owns only an empty command pool/buffer. No pipeline, descriptor,
    // or recorded IR object remains live while each unbound buffer is retired.
    // More than 1024 creations expose the old monotonically growing IR table.
    unsafe {
        for iteration in 0..1250 {
            let buffer = context
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(4)
                        .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .unwrap_or_else(|error| {
                    panic!("buffer creation {iteration} exhausted retired capacity: {error:?}")
                });
            context.device.destroy_buffer(buffer, None);
        }
    }

    // A real GPU workload must still work using objects created after reclaim.
    let storage = Storage::new(&context);
    let compute = Compute::new(&context);
    compute.update(&storage);
    compute.record(false);
    unsafe {
        let fence = context
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .unwrap();
        context.submit(fence).unwrap();
        context
            .device
            .wait_for_fences(&[fence], true, u64::MAX)
            .unwrap();
        assert_eq!(
            storage.read(),
            (0..WORDS as u32).map(|i| i * 3 + 7).collect::<Vec<_>>()
        );
        context.device.destroy_fence(fence, None);
    }
}

#[test]
#[ignore = "requires a built SGFX ICD and a native GPU adapter"]
fn global_memory_barrier_orders_compute_writes_before_buffer_copy() {
    let context = Context::new();
    let source = Storage::with_usage(
        &context,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
    );
    let destination = Storage::with_usage(&context, vk::BufferUsageFlags::TRANSFER_DST);
    let compute = Compute::new(&context);
    compute.update(&source);
    compute.record_commands(false, Some((&source, &destination)));
    unsafe {
        let fence = context
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .unwrap();
        context.submit(fence).unwrap();
        context
            .device
            .wait_for_fences(&[fence], true, u64::MAX)
            .unwrap();
        assert_eq!(
            destination.read(),
            (0..WORDS as u32).map(|i| i * 3 + 7).collect::<Vec<_>>()
        );
        context.device.destroy_fence(fence, None);
    }
}
