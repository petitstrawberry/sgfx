//! Ordinary-loader Vulkan GPU mip generation, readback and sampler LOD checks.
use ash::{Entry, vk};
use std::{error::Error, ffi::CStr};

#[path = "support/cube.rs"]
mod cube;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

struct Context {
    entry: Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    memory: vk::PhysicalDeviceMemoryProperties,
}
impl Context {
    fn new() -> Result<Self> {
        let entry = unsafe { Entry::load()? };
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_0);
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app),
                None,
            )?
        };
        let physical = unsafe { instance.enumerate_physical_devices()? }[0];
        let properties = unsafe { instance.get_physical_device_properties(physical) };
        println!(
            "Device: {}",
            unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }.to_string_lossy()
        );
        let priorities = [1.0];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(0)
            .queue_priorities(&priorities)];
        let device = unsafe {
            instance.create_device(
                physical,
                &vk::DeviceCreateInfo::default().queue_create_infos(&queues),
                None,
            )?
        };
        let queue = unsafe { device.get_device_queue(0, 0) };
        let pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(0),
                None,
            )?
        };
        let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
        Ok(Self {
            entry,
            instance,
            device,
            queue,
            pool,
            memory,
        })
    }
    fn memory(&self, requirements: vk::MemoryRequirements) -> Result<vk::DeviceMemory> {
        let index = cube::memory_type(
            &self.memory,
            requirements,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        Ok(unsafe {
            self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(index),
                None,
            )?
        })
    }
    fn submit(&self, record: impl FnOnce(vk::CommandBuffer)) -> Result<()> {
        unsafe {
            let command = self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            self.device.begin_command_buffer(
                command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            record(command);
            self.device.end_command_buffer(command)?;
            self.device.queue_submit(
                self.queue,
                &[vk::SubmitInfo::default().command_buffers(&[command])],
                vk::Fence::null(),
            )?;
            self.device.queue_wait_idle(self.queue)?;
            self.device.free_command_buffers(self.pool, &[command]);
        }
        Ok(())
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
        let _ = &self.entry;
    }
}
struct Buffer<'a> {
    context: &'a Context,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
}
impl<'a> Buffer<'a> {
    fn new(context: &'a Context, size: u64, usage: vk::BufferUsageFlags) -> Result<Self> {
        let buffer = unsafe {
            context.device.create_buffer(
                &vk::BufferCreateInfo::default().size(size).usage(usage),
                None,
            )?
        };
        let memory =
            context.memory(unsafe { context.device.get_buffer_memory_requirements(buffer) })?;
        unsafe {
            context.device.bind_buffer_memory(buffer, memory, 0)?;
        }
        Ok(Self {
            context,
            buffer,
            memory,
            size,
        })
    }
    fn write(&self, bytes: &[u8]) -> Result<()> {
        assert_eq!(bytes.len() as u64, self.size);
        unsafe {
            let mapped = self.context.device.map_memory(
                self.memory,
                0,
                self.size,
                vk::MemoryMapFlags::empty(),
            )?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.cast(), bytes.len());
            self.context.device.unmap_memory(self.memory);
        }
        Ok(())
    }
    fn read(&self) -> Result<Vec<u8>> {
        unsafe {
            let mapped = self.context.device.map_memory(
                self.memory,
                0,
                self.size,
                vk::MemoryMapFlags::empty(),
            )?;
            let bytes =
                std::slice::from_raw_parts(mapped.cast::<u8>(), self.size as usize).to_vec();
            self.context.device.unmap_memory(self.memory);
            Ok(bytes)
        }
    }
}
impl Drop for Buffer<'_> {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_buffer(self.buffer, None);
            self.context.device.free_memory(self.memory, None);
        }
    }
}

fn extent(size: [u32; 2], mip: u32) -> vk::Extent3D {
    vk::Extent3D {
        width: (size[0] >> mip).max(1),
        height: (size[1] >> mip).max(1),
        depth: 1,
    }
}
fn layers(mip: u32) -> vk::ImageSubresourceLayers {
    vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(mip)
        .layer_count(1)
}
fn transition(
    device: &ash::Device,
    command: vk::CommandBuffer,
    image: vk::Image,
    mip: u32,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
) {
    let access = |layout| match layout {
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => vk::AccessFlags::TRANSFER_WRITE,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => vk::AccessFlags::TRANSFER_READ,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => vk::AccessFlags::SHADER_READ,
        _ => vk::AccessFlags::empty(),
    };
    unsafe {
        device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[vk::ImageMemoryBarrier::default()
                .image(image)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .old_layout(old)
                .new_layout(new)
                .src_access_mask(access(old))
                .dst_access_mask(access(new))
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(mip)
                        .level_count(1)
                        .layer_count(1),
                )],
        );
    }
}

fn run(context: &Context, size: [u32; 2], format: vk::Format, filter: vk::Filter) -> Result<()> {
    let device = &context.device;
    let levels = size[0].max(size[1]).ilog2() + 1;
    let image = unsafe {
        device.create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(extent(size, 0))
                .mip_levels(levels)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(
                    vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::TRANSFER_SRC
                        | vk::ImageUsageFlags::TRANSFER_DST,
                ),
            None,
        )?
    };
    let memory = context.memory(unsafe { device.get_image_memory_requirements(image) })?;
    unsafe {
        device.bind_image_memory(image, memory, 0)?;
    }
    let mut base = Vec::new();
    for y in 0..size[1] {
        for x in 0..size[0] {
            let mut color = [x as u8 * 24, y as u8 * 24, ((x ^ y) & 1) as u8 * 128, 255];
            if format == vk::Format::B8G8R8A8_UNORM {
                color.swap(0, 2);
            }
            base.extend(color);
        }
    }
    let upload = Buffer::new(
        context,
        base.len() as u64,
        vk::BufferUsageFlags::TRANSFER_SRC,
    )?;
    upload.write(&base)?;
    let total: u64 = (0..levels)
        .map(|mip| {
            let e = extent(size, mip);
            u64::from(e.width) * u64::from(e.height) * 4
        })
        .sum();
    let readback = Buffer::new(context, total, vk::BufferUsageFlags::TRANSFER_DST)?;
    context.submit(|command| unsafe {
        for mip in 0..levels {
            transition(
                device,
                command,
                image,
                mip,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
        }
        device.cmd_copy_buffer_to_image(
            command,
            upload.buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::BufferImageCopy::default()
                .image_subresource(layers(0))
                .image_extent(extent(size, 0))],
        );
        for mip in 1..levels {
            transition(
                device,
                command,
                image,
                mip - 1,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            let src = extent(size, mip - 1);
            let dst = extent(size, mip);
            device.cmd_blit_image(
                command,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageBlit::default()
                    .src_subresource(layers(mip - 1))
                    .dst_subresource(layers(mip))
                    .src_offsets([
                        vk::Offset3D::default(),
                        vk::Offset3D {
                            x: src.width as i32,
                            y: src.height as i32,
                            z: 1,
                        },
                    ])
                    .dst_offsets([
                        vk::Offset3D::default(),
                        vk::Offset3D {
                            x: dst.width as i32,
                            y: dst.height as i32,
                            z: 1,
                        },
                    ])],
                filter,
            );
            transition(
                device,
                command,
                image,
                mip - 1,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
        }
        transition(
            device,
            command,
            image,
            levels - 1,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
        let mut offset = 0;
        for mip in 0..levels {
            let e = extent(size, mip);
            transition(
                device,
                command,
                image,
                mip,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            device.cmd_copy_image_to_buffer(
                command,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                readback.buffer,
                &[vk::BufferImageCopy::default()
                    .buffer_offset(offset)
                    .image_subresource(layers(mip))
                    .image_extent(e)],
            );
            transition(
                device,
                command,
                image,
                mip,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            offset += u64::from(e.width) * u64::from(e.height) * 4;
        }
    })?;
    let mut bytes = readback.read()?;
    if format == vk::Format::B8G8R8A8_UNORM {
        for pixel in bytes.chunks_exact_mut(4) {
            pixel.swap(0, 2);
        }
    }
    let mut mip_pixels = Vec::new();
    let mut offset = 0;
    for mip in 0..levels {
        let e = extent(size, mip);
        let length = e.width as usize * e.height as usize * 4;
        mip_pixels.push(bytes[offset..offset + length].to_vec());
        offset += length;
    }
    for mip in 1..levels {
        let src = extent(size, mip - 1);
        let dst = extent(size, mip);
        for y in 0..dst.height {
            for x in 0..dst.width {
                let coordinates = [
                    (x as f32 + 0.5) * src.width as f32 / dst.width as f32,
                    (y as f32 + 0.5) * src.height as f32 / dst.height as f32,
                ];
                let sample = |x: i32, y: i32, c: usize| -> f32 {
                    let x = x.clamp(0, src.width as i32 - 1) as usize;
                    let y = y.clamp(0, src.height as i32 - 1) as usize;
                    mip_pixels[mip as usize - 1][(y * src.width as usize + x) * 4 + c] as f32
                };
                for c in 0..4 {
                    let expected = if filter == vk::Filter::NEAREST {
                        sample(
                            coordinates[0].floor() as i32,
                            coordinates[1].floor() as i32,
                            c,
                        )
                    } else {
                        let px = coordinates[0] - 0.5;
                        let py = coordinates[1] - 0.5;
                        let ix = px.floor() as i32;
                        let iy = py.floor() as i32;
                        let fx = px.fract();
                        let fy = py.fract();
                        (sample(ix, iy, c) * (1.0 - fx) + sample(ix + 1, iy, c) * fx) * (1.0 - fy)
                            + (sample(ix, iy + 1, c) * (1.0 - fx) + sample(ix + 1, iy + 1, c) * fx)
                                * fy
                    };
                    let observed = mip_pixels[mip as usize]
                        [(y as usize * dst.width as usize + x as usize) * 4 + c]
                        as f32;
                    if (observed - expected).abs() > 1.0 {
                        return Err(format!(
                            "mip {mip}, ({x},{y}), channel {c}: {observed} != {expected}"
                        )
                        .into());
                    }
                }
            }
        }
    }
    let mut expected_base = base;
    if format == vk::Format::B8G8R8A8_UNORM {
        for pixel in expected_base.chunks_exact_mut(4) {
            pixel.swap(0, 2);
        }
    }
    if mip_pixels[0] != expected_base {
        return Err("base mip GPU upload/readback mismatch".into());
    }
    if size == [8, 8] {
        verify_sampling(context, image, size, format, &mip_pixels)?;
        let mut changed = [217, 53, 119, 255];
        if format == vk::Format::B8G8R8A8_UNORM {
            changed.swap(0, 2);
        }
        let patch = Buffer::new(context, 4, vk::BufferUsageFlags::TRANSFER_SRC)?;
        patch.write(&changed)?;
        let patched = Buffer::new(context, 64, vk::BufferUsageFlags::TRANSFER_DST)?;
        context.submit(|command| unsafe {
            transition(
                device,
                command,
                image,
                1,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            device.cmd_copy_buffer_to_image(
                command,
                patch.buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::default()
                    .image_subresource(layers(1))
                    .image_offset(vk::Offset3D { x: 1, y: 1, z: 0 })
                    .image_extent(vk::Extent3D {
                        width: 1,
                        height: 1,
                        depth: 1,
                    })],
            );
            transition(
                device,
                command,
                image,
                1,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            device.cmd_copy_image_to_buffer(
                command,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                patched.buffer,
                &[vk::BufferImageCopy::default()
                    .image_subresource(layers(1))
                    .image_extent(extent(size, 1))],
            );
        })?;
        let mut patched = patched.read()?;
        if format == vk::Format::B8G8R8A8_UNORM {
            for pixel in patched.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
        }
        let mut expected = mip_pixels[1].clone();
        expected[20..24].copy_from_slice(&[217, 53, 119, 255]);
        if patched != expected {
            return Err("partial nonzero-mip upload GPU readback mismatch".into());
        }
        println!("PASS: partial upload to mip 1 changes exactly the selected pixel");
    }
    unsafe {
        device.destroy_image(image, None);
        device.free_memory(memory, None);
    }
    println!("PASS: {format:?} {filter:?} {size:?} GPU blits and all {levels} mip readbacks");
    Ok(())
}

fn verify_sampling(
    context: &Context,
    image: vk::Image,
    size: [u32; 2],
    format: vk::Format,
    pixels: &[Vec<u8>],
) -> Result<()> {
    let device = &context.device;
    let words = cube::shader_words(
        include_str!("assets/mipmap.wgsl"),
        naga::ShaderStage::Compute,
        "main",
    )?;
    unsafe {
        let shader = device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?;
        let bindings = [
            (0, vk::DescriptorType::SAMPLED_IMAGE),
            (1, vk::DescriptorType::SAMPLER),
            (2, vk::DescriptorType::STORAGE_BUFFER),
        ]
        .map(|(binding, ty)| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(ty)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        });
        let set_layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )?;
        let layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]),
            None,
        )?;
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
            .map_err(|(_, error)| error)?[0];
        let sizes = bindings.map(|b| {
            vk::DescriptorPoolSize::default()
                .ty(b.descriptor_type)
                .descriptor_count(1)
        });
        let pool = device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                // A large declared set capacity must not preallocate IR groups.
                .max_sets(4096)
                .pool_sizes(&sizes),
            None,
        )?;
        let set = device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&[set_layout]),
        )?[0];
        assert_eq!(
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&[set_layout]),
            ),
            Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY)
        );
        let view = device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(4)
                        .layer_count(1),
                ),
            None,
        )?;
        let output = Buffer::new(context, 64, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        output.write(&[0; 64])?;
        for (mode, min, max) in [
            (vk::SamplerMipmapMode::NEAREST, 0.0, 3.0),
            (vk::SamplerMipmapMode::LINEAR, 0.0, 3.0),
            (vk::SamplerMipmapMode::LINEAR, 1.25, 2.25),
        ] {
            let sampler = device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .mipmap_mode(mode)
                    .min_lod(min)
                    .max_lod(max)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?;
            let image_info = [vk::DescriptorImageInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let sampler_info = [vk::DescriptorImageInfo::default().sampler(sampler)];
            let buffer_info = [vk::DescriptorBufferInfo::default()
                .buffer(output.buffer)
                .range(64)];
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
            context.submit(|command| {
                device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline);
                device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    layout,
                    0,
                    &[set],
                    &[],
                );
                device.cmd_dispatch(command, 4, 1, 1);
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::HOST_READ)],
                    &[],
                    &[],
                );
            })?;
            let observed = output.read()?;
            let sample = |level: u32, channel: usize| {
                let e = extent(size, level);
                let x = (0.37 * e.width as f32) as usize;
                let y = (0.63 * e.height as f32) as usize;
                pixels[level as usize][(y * e.width as usize + x) * 4 + channel] as f32
            };
            for index in 0..4 {
                for channel in 0..4 {
                    let lod = (index as f32 + 0.25).clamp(min, max);
                    let expected = if mode == vk::SamplerMipmapMode::NEAREST {
                        sample((lod + 0.5).floor() as u32, channel)
                    } else {
                        let lo = lod.floor() as u32;
                        let hi = (lo + 1).min(3);
                        sample(lo, channel) * (1.0 - lod.fract())
                            + sample(hi, channel) * lod.fract()
                    };
                    let offset = index * 16 + channel * 4;
                    let actual =
                        u32::from_ne_bytes(observed[offset..offset + 4].try_into().unwrap()) as f32;
                    if (actual - expected).abs() > 1.0 {
                        return Err(format!("sampler {mode:?} [{min},{max}] LOD {lod} channel {channel}: {actual} != {expected}").into());
                    }
                }
            }
            device.destroy_sampler(sampler, None);
        }
        device.destroy_image_view(view, None);
        device.destroy_descriptor_pool(pool, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(layout, None);
        device.destroy_descriptor_set_layout(set_layout, None);
        device.destroy_shader_module(shader, None);
    }
    println!("PASS: GPU explicit LOD sampling, nearest/linear mip filters and nonzero LOD clamps");
    Ok(())
}
fn main() -> Result<()> {
    let context = Context::new()?;
    for size in [[8, 8], [7, 3]] {
        for format in [vk::Format::R8G8B8A8_UNORM, vk::Format::B8G8R8A8_UNORM] {
            for filter in [vk::Filter::NEAREST, vk::Filter::LINEAR] {
                run(&context, size, format, filter)?;
            }
        }
    }
    Ok(())
}
