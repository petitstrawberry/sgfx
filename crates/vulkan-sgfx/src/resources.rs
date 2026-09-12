//! Host-visible Vulkan resources and their portable SGFX identities.
use ash::vk::{self, Handle};
use sgfx_core::ir;
use std::{
    collections::HashMap,
    ffi::{CStr, c_void},
    panic::{AssertUnwindSafe, catch_unwind},
    ptr, slice,
};

const LIMIT: usize = 1024;
const MAX_ALLOCATION: u64 = 256 * 1024 * 1024;
const MAX_ALLOCATED: usize = 512 * 1024 * 1024;
pub(crate) struct Buffer {
    pub id: ir::BufferId,
    pub size: u64,
    pub usage: vk::BufferUsageFlags,
    pub bound: Option<(vk::DeviceMemory, u64)>,
}
/// Host allocation with the eight-byte base alignment advertised by the ICD.
/// The backing words never grow after allocation, preserving every mapped pointer.
pub(crate) struct AlignedBytes {
    words: Vec<u64>,
    len: usize,
}
impl AlignedBytes {
    fn zeroed(len: usize) -> Result<Self, vk::Result> {
        let word_count = len
            .checked_add(7)
            .ok_or(vk::Result::ERROR_OUT_OF_HOST_MEMORY)?
            / 8;
        let mut words = Vec::new();
        words
            .try_reserve_exact(word_count)
            .map_err(|_| vk::Result::ERROR_OUT_OF_HOST_MEMORY)?;
        words.resize(word_count, 0);
        Ok(Self { words, len })
    }
}
impl std::ops::Deref for AlignedBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.words.as_ptr().cast(), self.len) }
    }
}
impl std::ops::DerefMut for AlignedBytes {
    fn deref_mut(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.words.as_mut_ptr().cast(), self.len) }
    }
}
pub(crate) struct Memory {
    pub bytes: AlignedBytes,
    pub mapped: bool,
}
#[derive(Clone, Copy)]
pub(crate) enum Pipeline {
    Compute(ir::ComputePipelineId),
    Graphics(ir::ProgrammableRenderPipelineId),
}
#[derive(Clone, Copy)]
pub(crate) enum DescriptorBinding {
    Buffer {
        buffer: vk::Buffer,
        offset: u64,
        range: u64,
    },
}
pub(crate) struct DescriptorPool {
    pub max_sets: usize,
    pub free_sets: bool,
    pub capacity: HashMap<vk::DescriptorType, u32>,
    pub remaining: HashMap<vk::DescriptorType, u32>,
}
pub(crate) struct DescriptorSet {
    pub pool: vk::DescriptorPool,
    pub layout: ir::BindGroupLayoutDesc,
    pub bindings: HashMap<u32, DescriptorBinding>,
    pub cached_group: Option<ir::BindGroupId>,
    pub invalid: bool,
}
#[derive(Default)]
pub(crate) struct Resources {
    pub buffers: HashMap<vk::Buffer, Buffer>,
    pub memories: HashMap<vk::DeviceMemory, Memory>,
    pub shaders: HashMap<vk::ShaderModule, ir::ShaderModuleId>,
    pub set_layouts: HashMap<vk::DescriptorSetLayout, ir::BindGroupLayoutDesc>,
    pub pipeline_layouts: HashMap<vk::PipelineLayout, ir::PipelineLayoutDesc>,
    pub descriptor_pools: HashMap<vk::DescriptorPool, DescriptorPool>,
    pub descriptor_sets: HashMap<vk::DescriptorSet, DescriptorSet>,
    bind_group_cache: Vec<(ir::BindGroupDesc, ir::BindGroupId)>,
    pub pipelines: HashMap<vk::Pipeline, Pipeline>,
    pub graphics_extents: HashMap<vk::Pipeline, vk::Extent2D>,
    pub images: HashMap<vk::Image, crate::images::Image>,
    pub views: HashMap<vk::ImageView, vk::Image>,
    pub render_passes: HashMap<vk::RenderPass, crate::images::RenderPass>,
    pub framebuffers: HashMap<vk::Framebuffer, crate::images::Framebuffer>,
}
impl Resources {
    pub fn new() -> Self {
        Self::default()
    }
    /// Whether any Vulkan object still depends on this epoch's IR table.
    /// Host memory and descriptor-pool quotas survive epoch reclamation because
    /// they contain no IR identities; layouts conservatively keep their epoch.
    pub(crate) fn has_live_ir_objects(&self) -> bool {
        !self.buffers.is_empty()
            || !self.images.is_empty()
            || !self.shaders.is_empty()
            || !self.pipelines.is_empty()
            || !self.set_layouts.is_empty()
            || !self.pipeline_layouts.is_empty()
            || !self.descriptor_sets.is_empty()
    }

    /// Forget every cached identity before swapping to a fresh IR table.
    pub(crate) fn clear_ir_cache(&mut self) {
        self.bind_group_cache.clear();
        for set in self.descriptor_sets.values_mut() {
            set.cached_group = None;
        }
    }

    pub fn descriptor_group(
        &mut self,
        table: &ir::ResourceTable,
        handle: vk::DescriptorSet,
    ) -> Result<ir::BindGroupId, vk::Result> {
        let set = self
            .descriptor_sets
            .get(&handle)
            .ok_or(vk::Result::ERROR_UNKNOWN)?;
        if set.invalid {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let mut entries = Vec::with_capacity(set.bindings.len());
        for (&binding, &resource) in &set.bindings {
            let resource = match resource {
                DescriptorBinding::Buffer {
                    buffer,
                    offset,
                    range,
                } => {
                    let buf = self.buffers.get(&buffer).ok_or(vk::Result::ERROR_UNKNOWN)?;
                    if buf.bound.is_none() {
                        return Err(vk::Result::ERROR_UNKNOWN);
                    }
                    let size = if range == vk::WHOLE_SIZE {
                        buf.size
                            .checked_sub(offset)
                            .ok_or(vk::Result::ERROR_UNKNOWN)?
                    } else {
                        range
                    };
                    ir::BindingResource::Buffer {
                        buffer: buf.id,
                        offset,
                        size,
                    }
                }
            };
            entries.push(ir::BindGroupEntry::new(binding, resource));
        }
        let desc = ir::BindGroupDesc::new(table, set.layout.clone(), entries).map_err(failure)?;
        if let Some(id) = set.cached_group {
            return Ok(id);
        }
        let id = if let Some((_, id)) = self
            .bind_group_cache
            .iter()
            .find(|(cached, _)| *cached == desc)
        {
            *id
        } else {
            let id = table.define_bind_group(desc.clone()).map_err(failure)?.id();
            self.bind_group_cache.push((desc, id));
            id
        };
        self.descriptor_sets
            .get_mut(&handle)
            .ok_or(vk::Result::ERROR_UNKNOWN)?
            .cached_group = Some(id);
        Ok(id)
    }
}
/// Check the subset's dedicated, nonaliasing memory binding contract.
pub(crate) fn memory_available(
    resources: &Resources,
    memory: vk::DeviceMemory,
    offset: u64,
    size: u64,
) -> bool {
    let Some(end) = offset.checked_add(size) else {
        return false;
    };
    if size == 0
        || resources
            .memories
            .get(&memory)
            .is_none_or(|m| end > m.bytes.len() as u64)
    {
        return false;
    }
    let overlaps = |bound: Option<(vk::DeviceMemory, u64)>, length: u64| {
        bound.is_some_and(|(m, start)| {
            m == memory
                && start
                    .checked_add(length)
                    .is_none_or(|bound_end| start < end && offset < bound_end)
        })
    };
    !resources
        .buffers
        .values()
        .any(|b| overlaps(b.bound, b.size))
        && !resources.images.values().any(|i| {
            overlaps(
                i.bound,
                u64::from(i.extent.width) * u64::from(i.extent.height) * 4,
            )
        })
}
pub(crate) fn backend_failure(error: crate::runtime::BackendError) -> vk::Result {
    crate::runtime::backend_failure(error)
}
pub(crate) fn failure(error: ir::Error) -> vk::Result {
    match error {
        ir::Error::OutOfMemory => vk::Result::ERROR_OUT_OF_HOST_MEMORY,
        ir::Error::ResourceLimitExceeded => vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
        _ => vk::Result::ERROR_FEATURE_NOT_PRESENT,
    }
}
fn ffi(f: impl FnOnce() -> Result<(), vk::Result>) -> vk::Result {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => vk::Result::SUCCESS,
        Ok(Err(e)) => e,
        Err(_) => vk::Result::ERROR_UNKNOWN,
    }
}
unsafe fn copied<T: Copy>(p: *const T, count: usize, max: usize) -> Result<Vec<T>, vk::Result> {
    if count > max || (count > 0 && p.is_null()) {
        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
    }
    if count == 0 {
        return Ok(Vec::new());
    }
    Ok(slice::from_raw_parts(p, count).to_vec())
}
fn descriptor_type(ty: ir::BindingType) -> vk::DescriptorType {
    match ty {
        ir::BindingType::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
        _ => vk::DescriptorType::STORAGE_BUFFER,
    }
}
fn stages(flags: vk::ShaderStageFlags) -> Result<ir::ShaderStages, vk::Result> {
    let allowed = vk::ShaderStageFlags::VERTEX
        | vk::ShaderStageFlags::FRAGMENT
        | vk::ShaderStageFlags::COMPUTE;
    if flags.is_empty() || !allowed.contains(flags) {
        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
    }
    let mut result = ir::ShaderStages::empty();
    if flags.contains(vk::ShaderStageFlags::VERTEX) {
        result |= ir::ShaderStages::VERTEX;
    }
    if flags.contains(vk::ShaderStageFlags::FRAGMENT) {
        result |= ir::ShaderStages::FRAGMENT;
    }
    if flags.contains(vk::ShaderStageFlags::COMPUTE) {
        result |= ir::ShaderStages::COMPUTE;
    }
    Ok(result)
}
unsafe extern "system" fn create_buffer(
    device: vk::Device,
    info: *const vk::BufferCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::Buffer,
) -> vk::Result {
    ffi(|| {
        if out.is_null() {
            return Err(vk::Result::ERROR_UNKNOWN);
        }
        out.write(vk::Buffer::null());
        if info.is_null() || !allocator.is_null() {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let i = &*info;
        if i.s_type != vk::StructureType::BUFFER_CREATE_INFO {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let allowed = vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::UNIFORM_BUFFER
            | vk::BufferUsageFlags::STORAGE_BUFFER;
        if !i.p_next.is_null()
            || !i.flags.is_empty()
            || i.size == 0
            || i.size > MAX_ALLOCATION
            || !i.size.is_multiple_of(4)
            || i.usage.is_empty()
            || !allowed.contains(i.usage)
            || i.sharing_mode != vk::SharingMode::EXCLUSIVE
        {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let size = i.size;
        let usage = i.usage;
        let mut ir_usage = ir::BufferUsage::COPY_SRC | ir::BufferUsage::COPY_DST;
        for (vk_flag, ir_flag) in [
            (vk::BufferUsageFlags::VERTEX_BUFFER, ir::BufferUsage::VERTEX),
            (vk::BufferUsageFlags::INDEX_BUFFER, ir::BufferUsage::INDEX),
            (
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                ir::BufferUsage::UNIFORM,
            ),
            (
                vk::BufferUsageFlags::STORAGE_BUFFER,
                ir::BufferUsage::STORAGE,
            ),
        ] {
            if usage.contains(vk_flag) {
                ir_usage |= ir_flag;
            }
        }
        let handle = crate::api::with_device(device, move |r| {
            if r.resources.buffers.len() >= LIMIT {
                return Err(vk::Result::ERROR_TOO_MANY_OBJECTS);
            }
            let id = r
                .table
                .define_buffer(ir::BufferDesc::new(size, ir_usage).map_err(failure)?)
                .map_err(failure)?
                .id();
            let handle = vk::Buffer::from_raw(crate::api::next_id());
            r.resources.buffers.insert(
                handle,
                Buffer {
                    id,
                    size,
                    usage,
                    bound: None,
                },
            );
            Ok(handle)
        })?;
        out.write(handle);
        Ok(())
    })
}
unsafe extern "system" fn destroy_buffer(
    device: vk::Device,
    buffer: vk::Buffer,
    _allocator: *const vk::AllocationCallbacks<'_>,
) {
    let _ = crate::api::with_device(device, move |r| {
        r.resources.buffers.remove(&buffer);
        Ok(())
    });
}
unsafe extern "system" fn buffer_requirements(
    device: vk::Device,
    buffer: vk::Buffer,
    out: *mut vk::MemoryRequirements,
) {
    if out.is_null() {
        return;
    }
    out.write(vk::MemoryRequirements::default());
    if let Ok(size) = crate::api::with_device(device, move |r| {
        r.resources
            .buffers
            .get(&buffer)
            .map(|b| b.size)
            .ok_or(vk::Result::ERROR_UNKNOWN)
    }) {
        out.write(vk::MemoryRequirements {
            size,
            alignment: 256,
            memory_type_bits: 1,
        });
    }
}
unsafe extern "system" fn allocate_memory(
    device: vk::Device,
    info: *const vk::MemoryAllocateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::DeviceMemory,
) -> vk::Result {
    ffi(|| {
        if out.is_null() {
            return Err(vk::Result::ERROR_UNKNOWN);
        }
        out.write(vk::DeviceMemory::null());
        if info.is_null() || !allocator.is_null() {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let i = &*info;
        if i.s_type != vk::StructureType::MEMORY_ALLOCATE_INFO {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        if !i.p_next.is_null()
            || i.memory_type_index != 0
            || i.allocation_size == 0
            || i.allocation_size > MAX_ALLOCATION
        {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let size =
            usize::try_from(i.allocation_size).map_err(|_| vk::Result::ERROR_OUT_OF_HOST_MEMORY)?;
        let handle = crate::api::with_device(device, move |r| {
            if r.resources.memories.len() >= LIMIT {
                return Err(vk::Result::ERROR_TOO_MANY_OBJECTS);
            }
            let allocated: usize = r.resources.memories.values().map(|m| m.bytes.len()).sum();
            if allocated
                .checked_add(size)
                .is_none_or(|v| v > MAX_ALLOCATED)
            {
                return Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
            }
            let bytes = AlignedBytes::zeroed(size)?;
            let handle = vk::DeviceMemory::from_raw(crate::api::next_id());
            r.resources.memories.insert(
                handle,
                Memory {
                    bytes,
                    mapped: false,
                },
            );
            Ok(handle)
        })?;
        out.write(handle);
        Ok(())
    })
}
unsafe extern "system" fn free_memory(
    device: vk::Device,
    memory: vk::DeviceMemory,
    _allocator: *const vk::AllocationCallbacks<'_>,
) {
    let _ = crate::api::with_device(device, move |r| {
        r.resources.memories.remove(&memory);
        for buffer in r.resources.buffers.values_mut() {
            if buffer.bound.is_some_and(|(m, _)| m == memory) {
                buffer.bound = None;
            }
        }
        for image in r.resources.images.values_mut() {
            if image.bound.is_some_and(|(m, _)| m == memory) {
                image.bound = None;
            }
        }
        Ok(())
    });
}
unsafe extern "system" fn bind_buffer_memory(
    device: vk::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    offset: vk::DeviceSize,
) -> vk::Result {
    ffi(|| {
        crate::api::with_device(device, move |r| {
            if !memory_available(
                &r.resources,
                memory,
                offset,
                r.resources
                    .buffers
                    .get(&buffer)
                    .ok_or(vk::Result::ERROR_UNKNOWN)?
                    .size,
            ) {
                return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
            }
            let mem = r
                .resources
                .memories
                .get(&memory)
                .ok_or(vk::Result::ERROR_UNKNOWN)?;
            let buf = r
                .resources
                .buffers
                .get_mut(&buffer)
                .ok_or(vk::Result::ERROR_UNKNOWN)?;
            if buf.bound.is_some()
                || !offset.is_multiple_of(256)
                || offset
                    .checked_add(buf.size)
                    .is_none_or(|end| end > mem.bytes.len() as u64)
            {
                return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
            }
            buf.bound = Some((memory, offset));
            Ok(())
        })
    })
}
unsafe extern "system" fn map_memory(
    device: vk::Device,
    memory: vk::DeviceMemory,
    offset: u64,
    size: u64,
    flags: vk::MemoryMapFlags,
    out: *mut *mut c_void,
) -> vk::Result {
    ffi(|| {
        if out.is_null() {
            return Err(vk::Result::ERROR_MEMORY_MAP_FAILED);
        }
        out.write(ptr::null_mut());
        if !flags.is_empty() {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let address = crate::api::with_device(device, move |r| {
            let mem = r
                .resources
                .memories
                .get_mut(&memory)
                .ok_or(vk::Result::ERROR_MEMORY_MAP_FAILED)?;
            let len = mem.bytes.len() as u64;
            let size = if size == vk::WHOLE_SIZE {
                len.checked_sub(offset)
                    .ok_or(vk::Result::ERROR_MEMORY_MAP_FAILED)?
            } else {
                size
            };
            if mem.mapped || size == 0 || offset.checked_add(size).is_none_or(|end| end > len) {
                return Err(vk::Result::ERROR_MEMORY_MAP_FAILED);
            }
            mem.mapped = true;
            Ok(mem.bytes.as_mut_ptr().add(offset as usize) as usize)
        })?;
        out.write(address as *mut c_void);
        Ok(())
    })
}
unsafe extern "system" fn unmap_memory(device: vk::Device, memory: vk::DeviceMemory) {
    let _ = crate::api::with_device(device, move |r| {
        if let Some(mem) = r.resources.memories.get_mut(&memory) {
            mem.mapped = false;
        }
        Ok(())
    });
}
unsafe extern "system" fn memory_ranges(
    device: vk::Device,
    count: u32,
    ranges: *const vk::MappedMemoryRange<'_>,
) -> vk::Result {
    ffi(|| {
        let ranges = copied(ranges, count as usize, LIMIT)?;
        let mut owned = Vec::with_capacity(ranges.len());
        for range in ranges {
            if range.s_type != vk::StructureType::MAPPED_MEMORY_RANGE || !range.p_next.is_null() {
                return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
            }
            owned.push((range.memory, range.offset, range.size));
        }
        crate::api::with_device(device, move |r| {
            for (memory, offset, size) in owned {
                let mem = r
                    .resources
                    .memories
                    .get(&memory)
                    .ok_or(vk::Result::ERROR_MEMORY_MAP_FAILED)?;
                let len = mem.bytes.len() as u64;
                let size = if size == vk::WHOLE_SIZE {
                    len.checked_sub(offset)
                        .ok_or(vk::Result::ERROR_MEMORY_MAP_FAILED)?
                } else {
                    size
                };
                if !mem.mapped || size == 0 || offset.checked_add(size).is_none_or(|end| end > len)
                {
                    return Err(vk::Result::ERROR_MEMORY_MAP_FAILED);
                }
            }
            Ok(())
        })
    })
}
unsafe extern "system" fn memory_commitment(
    device: vk::Device,
    memory: vk::DeviceMemory,
    out: *mut u64,
) {
    if !out.is_null() {
        out.write(
            crate::api::with_device(device, move |r| {
                r.resources
                    .memories
                    .get(&memory)
                    .map(|m| m.bytes.len() as u64)
                    .ok_or(vk::Result::ERROR_UNKNOWN)
            })
            .unwrap_or(0),
        );
    }
}
/// Normalize Vulkan clip-space Y to SGFX's shader convention once, retaining entry-point names.
fn normalize_spirv(words: Vec<u32>) -> Result<ir::ShaderModuleDesc, vk::Result> {
    ir::ShaderModuleDesc::spirv(words.clone()).map_err(failure)?;
    let options = naga::front::spv::Options {
        adjust_coordinate_space: true,
        strict_capabilities: true,
        block_ctx_dump_prefix: None,
    };
    let module = naga::front::spv::Frontend::new(words.into_iter(), &options)
        .parse()
        .map_err(|_| vk::Result::ERROR_FEATURE_NOT_PRESENT)?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .map_err(|_| vk::Result::ERROR_FEATURE_NOT_PRESENT)?;
    let mut output = naga::back::spv::Options {
        lang_version: (1, 0),
        ..Default::default()
    };
    output
        .flags
        .remove(naga::back::spv::WriterFlags::ADJUST_COORDINATE_SPACE);
    let normalized = naga::back::spv::write_vec(&module, &info, &output, None)
        .map_err(|_| vk::Result::ERROR_FEATURE_NOT_PRESENT)?;
    ir::ShaderModuleDesc::spirv(normalized).map_err(failure)
}
unsafe extern "system" fn create_shader_module(
    device: vk::Device,
    info: *const vk::ShaderModuleCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::ShaderModule,
) -> vk::Result {
    ffi(|| {
        if out.is_null() {
            return Err(vk::Result::ERROR_UNKNOWN);
        }
        out.write(vk::ShaderModule::null());
        if info.is_null() || !allocator.is_null() {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let i = &*info;
        if i.s_type != vk::StructureType::SHADER_MODULE_CREATE_INFO {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        if !i.p_next.is_null() || !i.flags.is_empty() || !i.code_size.is_multiple_of(4) {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let words = copied(i.p_code, i.code_size / 4, ir::MAX_SPIRV_WORDS)?;
        let desc = normalize_spirv(words)?;
        let handle = crate::api::with_device(device, move |r| {
            if r.resources.shaders.len() >= LIMIT {
                return Err(vk::Result::ERROR_TOO_MANY_OBJECTS);
            }
            let id = r.table.define_shader_module(desc).map_err(failure)?.id();
            r.cache
                .validate_shader_module(id)
                .map_err(backend_failure)?;
            let handle = vk::ShaderModule::from_raw(crate::api::next_id());
            r.resources.shaders.insert(handle, id);
            Ok(handle)
        })?;
        out.write(handle);
        Ok(())
    })
}
unsafe extern "system" fn destroy_shader_module(
    device: vk::Device,
    handle: vk::ShaderModule,
    _allocator: *const vk::AllocationCallbacks<'_>,
) {
    let _ = crate::api::with_device(device, move |r| {
        r.resources.shaders.remove(&handle);
        Ok(())
    });
}
unsafe extern "system" fn create_descriptor_set_layout(
    device: vk::Device,
    info: *const vk::DescriptorSetLayoutCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::DescriptorSetLayout,
) -> vk::Result {
    ffi(|| {
        if out.is_null() {
            return Err(vk::Result::ERROR_UNKNOWN);
        }
        out.write(vk::DescriptorSetLayout::null());
        if info.is_null() || !allocator.is_null() {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let i = &*info;
        if i.s_type != vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        if !i.p_next.is_null() || !i.flags.is_empty() {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let bindings = copied(
            i.p_bindings,
            i.binding_count as usize,
            ir::MAX_BINDINGS_PER_GROUP,
        )?;
        let mut entries = Vec::with_capacity(bindings.len());
        for binding in bindings {
            if binding.descriptor_count != 1 || !binding.p_immutable_samplers.is_null() {
                return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
            }
            let ty = match binding.descriptor_type {
                vk::DescriptorType::UNIFORM_BUFFER => ir::BindingType::UniformBuffer,
                vk::DescriptorType::STORAGE_BUFFER => {
                    ir::BindingType::StorageBuffer { read_only: false }
                }
                _ => return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT),
            };
            entries.push(ir::BindGroupLayoutEntry::new(
                binding.binding,
                stages(binding.stage_flags)?,
                ty,
            ));
        }
        let desc = ir::BindGroupLayoutDesc::new(entries).map_err(failure)?;
        let handle = crate::api::with_device(device, move |r| {
            if r.resources.set_layouts.len() >= LIMIT {
                return Err(vk::Result::ERROR_TOO_MANY_OBJECTS);
            }
            let handle = vk::DescriptorSetLayout::from_raw(crate::api::next_id());
            r.resources.set_layouts.insert(handle, desc);
            Ok(handle)
        })?;
        out.write(handle);
        Ok(())
    })
}
unsafe extern "system" fn destroy_descriptor_set_layout(
    device: vk::Device,
    handle: vk::DescriptorSetLayout,
    _allocator: *const vk::AllocationCallbacks<'_>,
) {
    let _ = crate::api::with_device(device, move |r| {
        r.resources.set_layouts.remove(&handle);
        Ok(())
    });
}
unsafe extern "system" fn create_pipeline_layout(
    device: vk::Device,
    info: *const vk::PipelineLayoutCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::PipelineLayout,
) -> vk::Result {
    ffi(|| {
        if out.is_null() {
            return Err(vk::Result::ERROR_UNKNOWN);
        }
        out.write(vk::PipelineLayout::null());
        if info.is_null() || !allocator.is_null() {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let i = &*info;
        if i.s_type != vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        if !i.p_next.is_null() || !i.flags.is_empty() || i.push_constant_range_count != 0 {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let handles = copied(
            i.p_set_layouts,
            i.set_layout_count as usize,
            ir::MAX_BIND_GROUPS,
        )?;
        let handle = crate::api::with_device(device, move |r| {
            if r.resources.pipeline_layouts.len() >= LIMIT {
                return Err(vk::Result::ERROR_TOO_MANY_OBJECTS);
            }
            let groups = handles
                .into_iter()
                .map(|h| {
                    r.resources
                        .set_layouts
                        .get(&h)
                        .cloned()
                        .ok_or(vk::Result::ERROR_UNKNOWN)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let desc = ir::PipelineLayoutDesc::new(groups).map_err(failure)?;
            let handle = vk::PipelineLayout::from_raw(crate::api::next_id());
            r.resources.pipeline_layouts.insert(handle, desc);
            Ok(handle)
        })?;
        out.write(handle);
        Ok(())
    })
}
unsafe extern "system" fn destroy_pipeline_layout(
    device: vk::Device,
    handle: vk::PipelineLayout,
    _allocator: *const vk::AllocationCallbacks<'_>,
) {
    let _ = crate::api::with_device(device, move |r| {
        r.resources.pipeline_layouts.remove(&handle);
        Ok(())
    });
}
unsafe extern "system" fn create_descriptor_pool(
    device: vk::Device,
    info: *const vk::DescriptorPoolCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::DescriptorPool,
) -> vk::Result {
    ffi(|| {
        if out.is_null() {
            return Err(vk::Result::ERROR_UNKNOWN);
        }
        out.write(vk::DescriptorPool::null());
        if info.is_null() || !allocator.is_null() {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let i = &*info;
        if i.s_type != vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        if !i.p_next.is_null()
            || !vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET.contains(i.flags)
            || i.max_sets == 0
            || i.max_sets as usize > LIMIT
        {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let mut capacity = HashMap::<vk::DescriptorType, u32>::new();
        for size in copied(i.p_pool_sizes, i.pool_size_count as usize, 32)? {
            if !matches!(
                size.ty,
                vk::DescriptorType::UNIFORM_BUFFER | vk::DescriptorType::STORAGE_BUFFER
            ) || size.descriptor_count == 0
            {
                return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
            }
            let count = capacity.entry(size.ty).or_default();
            *count = count
                .checked_add(size.descriptor_count)
                .filter(|v| *v <= 16384)
                .ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)?;
        }
        let max_sets = i.max_sets as usize;
        let free_sets = i
            .flags
            .contains(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
        let handle = crate::api::with_device(device, move |r| {
            if r.resources.descriptor_pools.len() >= LIMIT {
                return Err(vk::Result::ERROR_TOO_MANY_OBJECTS);
            }
            let handle = vk::DescriptorPool::from_raw(crate::api::next_id());
            r.resources.descriptor_pools.insert(
                handle,
                DescriptorPool {
                    max_sets,
                    free_sets,
                    remaining: capacity.clone(),
                    capacity,
                },
            );
            Ok(handle)
        })?;
        out.write(handle);
        Ok(())
    })
}
unsafe extern "system" fn destroy_descriptor_pool(
    device: vk::Device,
    handle: vk::DescriptorPool,
    _allocator: *const vk::AllocationCallbacks<'_>,
) {
    let _ = crate::api::with_device(device, move |r| {
        r.resources.descriptor_pools.remove(&handle);
        r.resources.descriptor_sets.retain(|_, s| s.pool != handle);
        Ok(())
    });
}
unsafe extern "system" fn reset_descriptor_pool(
    device: vk::Device,
    handle: vk::DescriptorPool,
    flags: vk::DescriptorPoolResetFlags,
) -> vk::Result {
    ffi(|| {
        if !flags.is_empty() {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        crate::api::with_device(device, move |r| {
            let pool = r
                .resources
                .descriptor_pools
                .get_mut(&handle)
                .ok_or(vk::Result::ERROR_UNKNOWN)?;
            pool.remaining = pool.capacity.clone();
            r.resources.descriptor_sets.retain(|_, s| s.pool != handle);
            Ok(())
        })
    })
}
unsafe extern "system" fn allocate_descriptor_sets(
    device: vk::Device,
    info: *const vk::DescriptorSetAllocateInfo<'_>,
    out: *mut vk::DescriptorSet,
) -> vk::Result {
    ffi(|| {
        if info.is_null() {
            return Err(vk::Result::ERROR_UNKNOWN);
        }
        let i = &*info;
        if i.s_type != vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        if !i.p_next.is_null() || i.descriptor_set_count == 0 || out.is_null() {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let layouts = copied(i.p_set_layouts, i.descriptor_set_count as usize, LIMIT)?;
        for index in 0..layouts.len() {
            out.add(index).write(vk::DescriptorSet::null());
        }
        let pool_handle = i.descriptor_pool;
        let handles = crate::api::with_device(device, move |r| {
            if r.resources
                .descriptor_sets
                .len()
                .checked_add(layouts.len())
                .is_none_or(|n| n > LIMIT)
            {
                return Err(vk::Result::ERROR_TOO_MANY_OBJECTS);
            }
            let layouts = layouts
                .into_iter()
                .map(|h| {
                    r.resources
                        .set_layouts
                        .get(&h)
                        .cloned()
                        .ok_or(vk::Result::ERROR_UNKNOWN)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let existing = r
                .resources
                .descriptor_sets
                .values()
                .filter(|s| s.pool == pool_handle)
                .count();
            let pool = r
                .resources
                .descriptor_pools
                .get_mut(&pool_handle)
                .ok_or(vk::Result::ERROR_UNKNOWN)?;
            if existing + layouts.len() > pool.max_sets {
                return Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY);
            }
            let mut remaining = pool.remaining.clone();
            for layout in &layouts {
                for entry in layout.entries() {
                    let slot = remaining.entry(descriptor_type(entry.ty())).or_default();
                    *slot = slot
                        .checked_sub(1)
                        .ok_or(vk::Result::ERROR_OUT_OF_POOL_MEMORY)?;
                }
            }
            pool.remaining = remaining;
            let mut result = Vec::with_capacity(layouts.len());
            for layout in layouts {
                let handle = vk::DescriptorSet::from_raw(crate::api::next_id());
                r.resources.descriptor_sets.insert(
                    handle,
                    DescriptorSet {
                        pool: pool_handle,
                        layout,
                        bindings: HashMap::new(),
                        cached_group: None,
                        invalid: false,
                    },
                );
                result.push(handle);
            }
            Ok(result)
        })?;
        ptr::copy_nonoverlapping(handles.as_ptr(), out, handles.len());
        Ok(())
    })
}
unsafe extern "system" fn free_descriptor_sets(
    device: vk::Device,
    pool_handle: vk::DescriptorPool,
    count: u32,
    sets: *const vk::DescriptorSet,
) -> vk::Result {
    ffi(|| {
        let handles = copied(sets, count as usize, LIMIT)?;
        crate::api::with_device(device, move |r| {
            let pool = r
                .resources
                .descriptor_pools
                .get(&pool_handle)
                .ok_or(vk::Result::ERROR_UNKNOWN)?;
            if !pool.free_sets {
                return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
            }
            let mut seen = std::collections::HashSet::new();
            for handle in &handles {
                if *handle == vk::DescriptorSet::null() {
                    continue;
                }
                if !seen.insert(*handle)
                    || r.resources
                        .descriptor_sets
                        .get(handle)
                        .is_none_or(|s| s.pool != pool_handle)
                {
                    return Err(vk::Result::ERROR_UNKNOWN);
                }
            }
            for handle in handles {
                if let Some(set) = r.resources.descriptor_sets.remove(&handle) {
                    let pool = r
                        .resources
                        .descriptor_pools
                        .get_mut(&pool_handle)
                        .ok_or(vk::Result::ERROR_UNKNOWN)?;
                    for entry in set.layout.entries() {
                        *pool
                            .remaining
                            .entry(descriptor_type(entry.ty()))
                            .or_default() += 1;
                    }
                }
            }
            Ok(())
        })
    })
}
#[derive(Clone)]
struct Write {
    set: vk::DescriptorSet,
    binding: u32,
    ty: vk::DescriptorType,
    value: Option<DescriptorBinding>,
}
unsafe extern "system" fn update_descriptor_sets(
    device: vk::Device,
    write_count: u32,
    writes: *const vk::WriteDescriptorSet<'_>,
    copy_count: u32,
    copies: *const vk::CopyDescriptorSet<'_>,
) {
    let result = ffi(|| {
        let writes = copied(writes, write_count as usize, LIMIT)?;
        let copies = copied(copies, copy_count as usize, LIMIT)?;
        let mut owned_writes = Vec::with_capacity(writes.len());
        for write in writes {
            let valid = write.s_type == vk::StructureType::WRITE_DESCRIPTOR_SET
                && write.p_next.is_null()
                && write.descriptor_count == 1
                && write.dst_array_element == 0
                && matches!(
                    write.descriptor_type,
                    vk::DescriptorType::UNIFORM_BUFFER | vk::DescriptorType::STORAGE_BUFFER
                )
                && !write.p_buffer_info.is_null();
            let value = if valid {
                let b = *write.p_buffer_info;
                Some(DescriptorBinding::Buffer {
                    buffer: b.buffer,
                    offset: b.offset,
                    range: b.range,
                })
            } else {
                None
            };
            owned_writes.push(Write {
                set: write.dst_set,
                binding: write.dst_binding,
                ty: write.descriptor_type,
                value,
            });
        }
        let owned_copies = copies
            .into_iter()
            .map(|c| {
                (
                    c.src_set,
                    c.src_binding,
                    c.dst_set,
                    c.dst_binding,
                    c.s_type == vk::StructureType::COPY_DESCRIPTOR_SET
                        && c.p_next.is_null()
                        && c.descriptor_count == 1
                        && c.src_array_element == 0
                        && c.dst_array_element == 0,
                )
            })
            .collect::<Vec<_>>();
        crate::api::with_device(device, move |r| {
            let modified_sets = owned_writes
                .iter()
                .map(|w| w.set)
                .chain(owned_copies.iter().map(|c| c.2))
                .collect::<Vec<_>>();
            crate::api::invalidate_descriptor_sets(r, &modified_sets);
            for write in owned_writes {
                let valid = write.value.is_some_and(|v| match v {
                    DescriptorBinding::Buffer {
                        buffer,
                        offset,
                        range,
                    } => r.resources.buffers.get(&buffer).is_some_and(|b| {
                        let size = if range == vk::WHOLE_SIZE {
                            b.size.saturating_sub(offset)
                        } else {
                            range
                        };
                        size > 0
                            && size
                                <= if write.ty == vk::DescriptorType::UNIFORM_BUFFER {
                                    16 * 1024
                                } else {
                                    128 * 1024 * 1024
                                }
                            && size.is_multiple_of(4)
                            && offset.is_multiple_of(256)
                            && offset.checked_add(size).is_some_and(|end| end <= b.size)
                            && b.usage
                                .contains(if write.ty == vk::DescriptorType::UNIFORM_BUFFER {
                                    vk::BufferUsageFlags::UNIFORM_BUFFER
                                } else {
                                    vk::BufferUsageFlags::STORAGE_BUFFER
                                })
                    }),
                });
                if let Some(set) = r.resources.descriptor_sets.get_mut(&write.set) {
                    let matches = set.layout.entries().iter().any(|e| {
                        e.binding() == write.binding && descriptor_type(e.ty()) == write.ty
                    });
                    set.cached_group = None;
                    if valid && matches {
                        if let Some(value) = write.value {
                            set.bindings.insert(write.binding, value);
                        }
                    } else {
                        set.invalid = true;
                    }
                }
            }
            for (src, src_binding, dst, dst_binding, valid) in owned_copies {
                let source = r
                    .resources
                    .descriptor_sets
                    .get(&src)
                    .filter(|s| !s.invalid)
                    .and_then(|s| {
                        s.layout
                            .entries()
                            .iter()
                            .find(|e| e.binding() == src_binding)
                            .and_then(|e| {
                                s.bindings
                                    .get(&src_binding)
                                    .copied()
                                    .map(|value| (e.ty(), value))
                            })
                    });
                if let Some(set) = r.resources.descriptor_sets.get_mut(&dst) {
                    set.cached_group = None;
                    if valid
                        && source.is_some_and(|(ty, _)| {
                            set.layout
                                .entries()
                                .iter()
                                .any(|e| e.binding() == dst_binding && e.ty() == ty)
                        })
                    {
                        if let Some((_, value)) = source {
                            set.bindings.insert(dst_binding, value);
                        }
                    } else {
                        set.invalid = true;
                    }
                }
            }
            Ok(())
        })
    });
    if result != vk::Result::SUCCESS {
        // This Vulkan entry point cannot return an error. Never silently preserve
        // stale bindings after a batch that exceeds the documented subset limits.
        let _ = crate::api::with_device(device, |r| {
            r.lost = true;
            Ok(())
        });
    }
}
unsafe extern "system" fn create_compute_pipelines(
    device: vk::Device,
    cache: vk::PipelineCache,
    count: u32,
    infos: *const vk::ComputePipelineCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::Pipeline,
) -> vk::Result {
    ffi(|| {
        if out.is_null() || count == 0 || count as usize > LIMIT {
            return Err(vk::Result::ERROR_UNKNOWN);
        }
        for index in 0..count as usize {
            out.add(index).write(vk::Pipeline::null());
        }
        if !allocator.is_null() || cache != vk::PipelineCache::null() {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let infos = copied(infos, count as usize, LIMIT)?;
        let mut requests = Vec::with_capacity(infos.len());
        for info in infos {
            let stage = info.stage;
            if info.s_type != vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO
                || stage.s_type != vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO
                || !info.p_next.is_null()
                || !info.flags.is_empty()
                || !stage.p_next.is_null()
                || !stage.flags.is_empty()
                || stage.stage != vk::ShaderStageFlags::COMPUTE
                || !stage.p_specialization_info.is_null()
                || stage.p_name.is_null()
                || info.base_pipeline_handle != vk::Pipeline::null()
            {
                return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
            }
            let name = CStr::from_ptr(stage.p_name)
                .to_str()
                .map_err(|_| vk::Result::ERROR_FEATURE_NOT_PRESENT)?;
            if name.len() > 1024 {
                return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
            }
            requests.push((stage.module, name.to_owned(), info.layout));
        }
        let handles = crate::api::with_device(device, move |r| {
            if r.resources
                .pipelines
                .len()
                .checked_add(requests.len())
                .is_none_or(|n| n > 256)
            {
                return Err(vk::Result::ERROR_TOO_MANY_OBJECTS);
            }
            let mut descriptors = Vec::with_capacity(requests.len());
            for (module, name, layout) in requests {
                let module = *r
                    .resources
                    .shaders
                    .get(&module)
                    .ok_or(vk::Result::ERROR_UNKNOWN)?;
                let shader = ir::ShaderEntryPoint::new(
                    r.table.shader_module_ref(module).map_err(failure)?,
                    ir::ShaderStage::Compute,
                    name,
                )
                .map_err(failure)?;
                let layout = r
                    .resources
                    .pipeline_layouts
                    .get(&layout)
                    .cloned()
                    .ok_or(vk::Result::ERROR_UNKNOWN)?;
                descriptors.push(ir::ComputePipelineDesc::new(shader, layout).map_err(failure)?);
            }
            let mut ids = Vec::with_capacity(descriptors.len());
            for desc in descriptors {
                let id = r.table.define_compute_pipeline(desc).map_err(failure)?.id();
                r.cache
                    .validate_compute_pipeline(id)
                    .map_err(backend_failure)?;
                ids.push(id);
            }
            let mut result = Vec::with_capacity(ids.len());
            for id in ids {
                let handle = vk::Pipeline::from_raw(crate::api::next_id());
                r.resources.pipelines.insert(handle, Pipeline::Compute(id));
                result.push(handle);
            }
            Ok(result)
        })?;
        ptr::copy_nonoverlapping(handles.as_ptr(), out, handles.len());
        Ok(())
    })
}
unsafe extern "system" fn destroy_pipeline(
    device: vk::Device,
    handle: vk::Pipeline,
    _allocator: *const vk::AllocationCallbacks<'_>,
) {
    let _ = crate::api::with_device(device, move |r| {
        r.resources.pipelines.remove(&handle);
        r.resources.graphics_extents.remove(&handle);
        Ok(())
    });
}

pub(crate) fn lookup(name: &CStr) -> vk::PFN_vkVoidFunction {
    macro_rules! entry {
        ($f:ident,$ty:ty) => {{
            let f: $ty = $f;
            Some(unsafe { std::mem::transmute::<$ty, unsafe extern "system" fn()>(f) })
        }};
    }
    match name.to_bytes() {
        b"vkCreateBuffer" => entry!(create_buffer, vk::PFN_vkCreateBuffer),
        b"vkDestroyBuffer" => entry!(destroy_buffer, vk::PFN_vkDestroyBuffer),
        b"vkGetBufferMemoryRequirements" => {
            entry!(buffer_requirements, vk::PFN_vkGetBufferMemoryRequirements)
        }
        b"vkAllocateMemory" => entry!(allocate_memory, vk::PFN_vkAllocateMemory),
        b"vkFreeMemory" => entry!(free_memory, vk::PFN_vkFreeMemory),
        b"vkBindBufferMemory" => entry!(bind_buffer_memory, vk::PFN_vkBindBufferMemory),
        b"vkMapMemory" => entry!(map_memory, vk::PFN_vkMapMemory),
        b"vkUnmapMemory" => entry!(unmap_memory, vk::PFN_vkUnmapMemory),
        b"vkFlushMappedMemoryRanges" => entry!(memory_ranges, vk::PFN_vkFlushMappedMemoryRanges),
        b"vkInvalidateMappedMemoryRanges" => {
            entry!(memory_ranges, vk::PFN_vkInvalidateMappedMemoryRanges)
        }
        b"vkGetDeviceMemoryCommitment" => {
            entry!(memory_commitment, vk::PFN_vkGetDeviceMemoryCommitment)
        }
        b"vkCreateShaderModule" => entry!(create_shader_module, vk::PFN_vkCreateShaderModule),
        b"vkDestroyShaderModule" => entry!(destroy_shader_module, vk::PFN_vkDestroyShaderModule),
        b"vkCreateDescriptorSetLayout" => entry!(
            create_descriptor_set_layout,
            vk::PFN_vkCreateDescriptorSetLayout
        ),
        b"vkDestroyDescriptorSetLayout" => entry!(
            destroy_descriptor_set_layout,
            vk::PFN_vkDestroyDescriptorSetLayout
        ),
        b"vkCreatePipelineLayout" => entry!(create_pipeline_layout, vk::PFN_vkCreatePipelineLayout),
        b"vkDestroyPipelineLayout" => {
            entry!(destroy_pipeline_layout, vk::PFN_vkDestroyPipelineLayout)
        }
        b"vkCreateDescriptorPool" => entry!(create_descriptor_pool, vk::PFN_vkCreateDescriptorPool),
        b"vkDestroyDescriptorPool" => {
            entry!(destroy_descriptor_pool, vk::PFN_vkDestroyDescriptorPool)
        }
        b"vkResetDescriptorPool" => entry!(reset_descriptor_pool, vk::PFN_vkResetDescriptorPool),
        b"vkAllocateDescriptorSets" => {
            entry!(allocate_descriptor_sets, vk::PFN_vkAllocateDescriptorSets)
        }
        b"vkFreeDescriptorSets" => entry!(free_descriptor_sets, vk::PFN_vkFreeDescriptorSets),
        b"vkUpdateDescriptorSets" => entry!(update_descriptor_sets, vk::PFN_vkUpdateDescriptorSets),
        b"vkCreateComputePipelines" => {
            entry!(create_compute_pipelines, vk::PFN_vkCreateComputePipelines)
        }
        b"vkDestroyPipeline" => entry!(destroy_pipeline, vk::PFN_vkDestroyPipeline),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_cache_clear_cannot_reuse_an_old_table_identity() {
        let old_table = ir::ResourceTable::new();
        let new_table = ir::ResourceTable::new();
        let mut resources = Resources::new();
        let handle = vk::DescriptorSet::from_raw(1);
        resources.descriptor_sets.insert(
            handle,
            DescriptorSet {
                pool: vk::DescriptorPool::null(),
                layout: ir::BindGroupLayoutDesc::new(vec![]).unwrap(),
                bindings: HashMap::new(),
                cached_group: None,
                invalid: false,
            },
        );
        let old_group = resources.descriptor_group(&old_table, handle).unwrap();
        resources.clear_ir_cache();
        let new_group = resources.descriptor_group(&new_table, handle).unwrap();
        assert_ne!(old_group, new_group);
        assert!(new_table.bind_group_ref(new_group).is_ok());
        assert!(new_table.bind_group_ref(old_group).is_err());
    }

    #[test]
    fn host_allocations_and_empty_pools_do_not_pin_an_ir_epoch() {
        let mut resources = Resources::new();
        resources.memories.insert(
            vk::DeviceMemory::from_raw(1),
            Memory {
                bytes: AlignedBytes::zeroed(256).unwrap(),
                mapped: true,
            },
        );
        resources.descriptor_pools.insert(
            vk::DescriptorPool::from_raw(2),
            DescriptorPool {
                max_sets: 1,
                free_sets: true,
                capacity: HashMap::new(),
                remaining: HashMap::new(),
            },
        );
        assert!(!resources.has_live_ir_objects());
        let layout = vk::DescriptorSetLayout::from_raw(3);
        resources
            .set_layouts
            .insert(layout, ir::BindGroupLayoutDesc::new(vec![]).unwrap());
        assert!(resources.has_live_ir_objects());
        resources.set_layouts.remove(&layout);
        assert!(!resources.has_live_ir_objects());
        resources.pipeline_layouts.insert(
            vk::PipelineLayout::from_raw(4),
            ir::PipelineLayoutDesc::new(vec![]).unwrap(),
        );
        assert!(resources.has_live_ir_objects());
        resources.pipeline_layouts.clear();
        assert!(!resources.has_live_ir_objects());
        assert_eq!(
            failure(ir::Error::ResourceLimitExceeded),
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
        );
    }

    #[test]
    fn host_allocations_are_aligned_and_keep_exact_requested_size() {
        for size in [1, 7, 8, 9, 1024] {
            let mut bytes = AlignedBytes::zeroed(size).unwrap();
            assert_eq!(bytes.len(), size);
            assert_eq!(bytes.as_ptr() as usize % 8, 0);
            assert!(bytes.iter().all(|b| *b == 0));
            let address = bytes.as_mut_ptr();
            bytes[size - 1] = 17;
            assert_eq!(address, bytes.as_mut_ptr());
            assert_eq!(bytes[size - 1], 17);
        }
    }

    #[test]
    fn spirv_normalization_preserves_vulkan_entry_names() {
        let module =
            naga::front::wgsl::parse_str("@compute @workgroup_size(1) fn main() {}").unwrap();
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
        let mut words =
            naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
                .unwrap();
        let mut cursor = 5;
        while cursor < words.len() {
            let count = (words[cursor] >> 16) as usize;
            if words[cursor] & 0xffff == 15 {
                // OpEntryPoint name begins after stage and ID.
                words[cursor + 3] = u32::from_le_bytes(*b"m-!?");
                break;
            }
            cursor += count;
        }
        assert!(cursor < words.len());
        let table = ir::ResourceTable::new();
        let shader = table
            .define_shader_module(normalize_spirv(words).unwrap())
            .unwrap();
        assert!(ir::ShaderEntryPoint::new(shader, ir::ShaderStage::Compute, "m-!?".into()).is_ok());
        assert!(normalize_spirv(vec![0; 5]).is_err());
    }

    #[test]
    fn memory_binding_rejects_aliases_and_overflow() {
        let table = ir::ResourceTable::new();
        let mut resources = Resources::new();
        let memory = vk::DeviceMemory::from_raw(1);
        resources.memories.insert(
            memory,
            Memory {
                bytes: AlignedBytes::zeroed(1024).unwrap(),
                mapped: false,
            },
        );
        let buffer = table
            .define_buffer(ir::BufferDesc::new(256, ir::BufferUsage::STORAGE).unwrap())
            .unwrap()
            .id();
        resources.buffers.insert(
            vk::Buffer::from_raw(2),
            Buffer {
                id: buffer,
                size: 256,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                bound: Some((memory, 256)),
            },
        );
        assert!(memory_available(&resources, memory, 0, 256));
        assert!(memory_available(&resources, memory, 512, 512));
        assert!(!memory_available(&resources, memory, 255, 2));
        assert!(!memory_available(&resources, memory, 511, 2));
        assert!(!memory_available(&resources, memory, 1024, 1));
        assert!(!memory_available(&resources, memory, u64::MAX, 2));
        assert!(!memory_available(
            &resources,
            vk::DeviceMemory::from_raw(99),
            0,
            1
        ));
    }

    #[test]
    fn descriptor_group_cache_still_checks_resource_lifetime() {
        let table = ir::ResourceTable::new();
        let mut resources = Resources::new();
        let buffer_handle = vk::Buffer::from_raw(1);
        let set_handle = vk::DescriptorSet::from_raw(2);
        let buffer = table
            .define_buffer(ir::BufferDesc::new(512, ir::BufferUsage::STORAGE).unwrap())
            .unwrap()
            .id();
        resources.buffers.insert(
            buffer_handle,
            Buffer {
                id: buffer,
                size: 512,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                bound: Some((vk::DeviceMemory::from_raw(3), 0)),
            },
        );
        let layout = ir::BindGroupLayoutDesc::new(vec![ir::BindGroupLayoutEntry::new(
            0,
            ir::ShaderStages::COMPUTE,
            ir::BindingType::StorageBuffer { read_only: false },
        )])
        .unwrap();
        resources.descriptor_sets.insert(
            set_handle,
            DescriptorSet {
                pool: vk::DescriptorPool::null(),
                layout,
                bindings: HashMap::from([(
                    0,
                    DescriptorBinding::Buffer {
                        buffer: buffer_handle,
                        offset: 256,
                        range: vk::WHOLE_SIZE,
                    },
                )]),
                cached_group: None,
                invalid: false,
            },
        );
        let first = resources.descriptor_group(&table, set_handle).unwrap();
        for _ in 0..1100 {
            resources
                .descriptor_sets
                .get_mut(&set_handle)
                .unwrap()
                .cached_group = None;
            assert_eq!(
                resources.descriptor_group(&table, set_handle).unwrap(),
                first
            );
        }
        resources.buffers.get_mut(&buffer_handle).unwrap().bound = None;
        assert!(resources.descriptor_group(&table, set_handle).is_err());
        resources.buffers.remove(&buffer_handle);
        assert!(resources.descriptor_group(&table, set_handle).is_err());
    }

    #[test]
    fn descriptor_group_rejects_unaligned_ranges_and_missing_bindings() {
        let table = ir::ResourceTable::new();
        let mut resources = Resources::new();
        let handle = vk::DescriptorSet::from_raw(1);
        let layout = ir::BindGroupLayoutDesc::new(vec![ir::BindGroupLayoutEntry::new(
            0,
            ir::ShaderStages::COMPUTE,
            ir::BindingType::UniformBuffer,
        )])
        .unwrap();
        resources.descriptor_sets.insert(
            handle,
            DescriptorSet {
                pool: vk::DescriptorPool::null(),
                layout,
                bindings: HashMap::new(),
                cached_group: None,
                invalid: false,
            },
        );
        assert!(resources.descriptor_group(&table, handle).is_err());
        let buffer_handle = vk::Buffer::from_raw(2);
        let buffer = table
            .define_buffer(ir::BufferDesc::new(512, ir::BufferUsage::UNIFORM).unwrap())
            .unwrap()
            .id();
        resources.buffers.insert(
            buffer_handle,
            Buffer {
                id: buffer,
                size: 512,
                usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                bound: Some((vk::DeviceMemory::from_raw(3), 0)),
            },
        );
        resources
            .descriptor_sets
            .get_mut(&handle)
            .unwrap()
            .bindings
            .insert(
                0,
                DescriptorBinding::Buffer {
                    buffer: buffer_handle,
                    offset: 4,
                    range: 16,
                },
            );
        assert!(resources.descriptor_group(&table, handle).is_err());
    }
}
