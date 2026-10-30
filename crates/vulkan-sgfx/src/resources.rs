//! Host-visible Vulkan resources and their portable SGFX identities.
use ash::vk::{self, Handle};
use sgfx::ir;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    ffi::{CStr, c_void},
    panic::{AssertUnwindSafe, catch_unwind},
    ptr, slice,
};

const LIMIT: usize = 1024;
// Skia keeps many small uniform/staging allocations alive alongside images.
// Keep their count separate from metadata limits and from the byte budget.
pub(crate) const MAX_MEMORY_ALLOCATIONS: usize = 4_096;
const MAX_DESCRIPTOR_SETS: usize = 16_384;
pub(crate) const MAX_ALLOCATION: u64 = 256 * 1024 * 1024;
pub(crate) struct Buffer {
    pub id: ir::BufferId,
    pub size: u64,
    pub usage: vk::BufferUsageFlags,
    pub bound: Option<(vk::DeviceMemory, u64)>,
}
#[derive(Clone, Copy)]
pub(crate) struct BufferView {
    pub format: vk::Format,
    pub buffer: vk::Buffer,
    pub offset: u64,
    pub range: u64,
}
/// Host allocation with the eight-byte base alignment advertised by the ICD.
/// The backing words never grow after allocation, preserving every mapped pointer.
pub(crate) struct AlignedBytes {
    words: Vec<u64>,
    len: usize,
}
impl AlignedBytes {
    /// GPU image storage lives in the backend. Reserve its logical byte range
    /// without allocating and clearing a redundant CPU mirror for every image.
    /// Buffers and explicit mappings materialize the mirror before access.
    fn unbacked(len: usize) -> Result<Self, vk::Result> {
        len.checked_add(7)
            .ok_or(vk::Result::ERROR_OUT_OF_HOST_MEMORY)?;
        Ok(Self {
            words: Vec::new(),
            len,
        })
    }
    #[cfg(test)]
    pub(crate) fn zeroed(len: usize) -> Result<Self, vk::Result> {
        let mut bytes = Self::unbacked(len)?;
        bytes.materialize()?;
        Ok(bytes)
    }
    fn len(&self) -> usize {
        self.len
    }
    fn materialize(&mut self) -> Result<(), vk::Result> {
        let word_count = self.len.div_ceil(8);
        if self.words.len() == word_count {
            return Ok(());
        }
        self.words
            .try_reserve_exact(word_count)
            .map_err(|_| vk::Result::ERROR_OUT_OF_HOST_MEMORY)?;
        self.words.resize(word_count, 0);
        Ok(())
    }
}
impl std::ops::Deref for AlignedBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        assert_eq!(
            self.words.len(),
            self.len.div_ceil(8),
            "CPU mirror must be materialized before access"
        );
        unsafe { slice::from_raw_parts(self.words.as_ptr().cast(), self.len) }
    }
}
impl std::ops::DerefMut for AlignedBytes {
    fn deref_mut(&mut self) -> &mut [u8] {
        assert_eq!(
            self.words.len(),
            self.len.div_ceil(8),
            "CPU mirror must be materialized before access"
        );
        unsafe { slice::from_raw_parts_mut(self.words.as_mut_ptr().cast(), self.len) }
    }
}
pub(crate) struct Memory {
    pub bytes: AlignedBytes,
    pub mapped: bool,
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Pipeline {
    Compute(ir::ComputePipelineId),
    Graphics(ir::ProgrammableRenderPipelineId),
}
#[derive(Debug, Clone, Copy)]
pub(crate) enum DescriptorBinding {
    TexelBuffer(vk::BufferView),
    Buffer {
        buffer: vk::Buffer,
        offset: u64,
        range: u64,
    },
    Image {
        view: vk::ImageView,
        sampler: vk::Sampler,
        layout: vk::ImageLayout,
    },
    Sampler(vk::Sampler),
}
pub(crate) struct DescriptorPool {
    pub max_sets: usize,
    pub free_sets: bool,
    pub capacity: HashMap<vk::DescriptorType, u32>,
    pub remaining: HashMap<vk::DescriptorType, u32>,
}
#[derive(Clone)]
pub(crate) struct DescriptorSet {
    pub texel_formats: HashMap<u32, vk::Format>,
    pub pool: vk::DescriptorPool,
    pub layout: ir::BindGroupLayoutDesc,
    pub types: BTreeMap<u32, vk::DescriptorType>,
    pub stages: BTreeMap<u32, vk::ShaderStageFlags>,
    pub bindings: HashMap<u32, DescriptorBinding>,
    pub cached_group: Option<ir::BindGroupId>,
    pub invalid: bool,
}
/// Vulkan binding numbers are labels, not indices into SGFX's bounded slots.
/// Each logical descriptor reserves an even resource slot and an adjacent
/// sampler slot. Preserve the existing low-number ABI and pack sparse labels
/// into unused pairs, with the same map used by layouts, shaders and updates.
pub(crate) type DescriptorBindingMap = BTreeMap<u32, u32>;
fn descriptor_binding_map(
    types: &BTreeMap<u32, vk::DescriptorType>,
) -> Result<DescriptorBindingMap, vk::Result> {
    let pairs = ir::MAX_BINDINGS_PER_GROUP as u32 / 2;
    if types.len() > pairs as usize {
        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
    }
    let mut map = types
        .keys()
        .filter(|&&b| b < pairs)
        .map(|&b| (b, b * 2))
        .collect::<BTreeMap<_, _>>();
    for &binding in types.keys().filter(|&&b| b >= pairs) {
        let slot = (0..pairs)
            .map(|slot| slot * 2)
            .find(|slot| !map.values().any(|used| used == slot))
            .ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)?;
        map.insert(binding, slot);
    }
    Ok(map)
}
impl DescriptorSet {
    pub(crate) fn binding_map(&self) -> DescriptorBindingMap {
        descriptor_binding_map(&self.types).expect("validated descriptor layout")
    }
    pub(crate) fn source_to_ir(&self, binding: u32) -> Option<u32> {
        self.binding_map().get(&binding).copied()
    }
    fn ir_to_source(&self, slot: u32) -> Option<u32> {
        self.binding_map()
            .into_iter()
            .find_map(|(binding, base)| (base == slot / 2 * 2).then_some(binding))
    }
}
pub(crate) type Specialization = Vec<(u32, Vec<u8>)>;
type ShaderVariant = (
    Specialization,
    Vec<Option<bool>>,
    Vec<DescriptorBindingMap>,
    ir::ShaderModuleId,
);
pub(crate) struct Shader {
    words: Vec<u32>,
    variants: Vec<ShaderVariant>,
}
impl Shader {
    pub(crate) fn input_bindings(
        &self,
    ) -> Result<Vec<crate::input_attachments::InputBinding>, vk::Result> {
        crate::input_attachments::bindings(&self.words)
    }
    pub(crate) fn variant_for_layout(
        &mut self,
        table: &ir::ResourceTable,
        values: &Specialization,
        inputs: &[Option<bool>],
        bindings: &[DescriptorBindingMap],
    ) -> Result<ir::ShaderModuleId, vk::Result> {
        if let Some((_, _, _, id)) =
            self.variants
                .iter()
                .find(|(key, input_key, binding_key, _)| {
                    key == values && input_key == inputs && binding_key == bindings
                })
        {
            return Ok(*id);
        }
        let desc = normalize_spirv_specialized(self.words.clone(), values, inputs, bindings)?;
        let id = table.define_shader_module(desc).map_err(failure)?.id();
        self.variants
            .push((values.clone(), inputs.to_vec(), bindings.to_vec(), id));
        Ok(id)
    }
}

pub(crate) unsafe fn specialization(
    info: *const vk::SpecializationInfo<'_>,
) -> Result<Specialization, vk::Result> {
    let Some(info) = info.as_ref() else {
        return Ok(Vec::new());
    };
    let invalid = vk::Result::ERROR_INITIALIZATION_FAILED;
    let entries = copied(info.p_map_entries, info.map_entry_count as usize, 256)?;
    let data = copied(info.p_data.cast::<u8>(), info.data_size, 4096)?;
    let mut result = Vec::with_capacity(entries.len());
    for entry in entries {
        let start = entry.offset as usize;
        let end = start.checked_add(entry.size).ok_or(invalid)?;
        let bytes = data.get(start..end).ok_or(invalid)?;
        if bytes.is_empty() || result.iter().any(|(id, _)| *id == entry.constant_id) {
            return Err(invalid);
        }
        result.push((entry.constant_id, bytes.to_vec()));
    }
    result.sort_by_key(|(id, _)| *id);
    Ok(result)
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
struct ShaderViews {
    textures: crate::spirv::TextureSwizzles,
    narrow_texels: Vec<(u32, u32)>,
}
impl ShaderViews {
    fn is_empty(&self) -> bool {
        self.textures.is_empty() && self.narrow_texels.is_empty()
    }
}
#[derive(Default)]
pub(crate) struct Resources {
    pub dedicated_allocations: HashMap<vk::DeviceMemory, crate::memory_allocation::DedicatedTarget>,
    pub mip_render_targets: HashMap<(vk::Format, u32, u32, usize), ir::TextureId>,
    pub depth_only_fragment: Option<ir::ShaderModuleId>,
    pub depth_clear_colors: HashMap<(u32, u32), ir::TextureId>,
    pub image_view_formats: HashMap<vk::Image, Vec<vk::Format>>,
    pub descriptor_templates:
        HashMap<vk::DescriptorUpdateTemplate, crate::api::descriptor_template::Template>,
    pub buffers: HashMap<vk::Buffer, Buffer>,
    pub buffer_views: HashMap<vk::BufferView, BufferView>,
    pub memories: HashMap<vk::DeviceMemory, Memory>,
    /// Buffers whose unmapped host allocation has already been staged to SGFX.
    pub uploaded_unmapped_buffers: HashSet<vk::Buffer>,
    pub shaders: HashMap<vk::ShaderModule, Shader>,
    // Destroying VkShaderModule must not retire modules still referenced by a
    // live VkPipeline. View-swizzle modules have no VkShaderModule owner.
    orphaned_shader_modules: HashSet<ir::ShaderModuleId>,
    pub push_set_layouts: HashSet<vk::DescriptorSetLayout>,
    pub pipeline_push_sets: HashMap<vk::PipelineLayout, (u32, DescriptorSet)>,
    pub pushed_set_owners: HashMap<vk::DescriptorSet, std::sync::Weak<()>>,
    pub set_layouts: HashMap<vk::DescriptorSetLayout, ir::BindGroupLayoutDesc>,
    pub set_layout_types: HashMap<vk::DescriptorSetLayout, BTreeMap<u32, vk::DescriptorType>>,
    pub set_layout_stages: HashMap<vk::DescriptorSetLayout, BTreeMap<u32, vk::ShaderStageFlags>>,
    pub samplers: HashMap<vk::Sampler, ir::SamplerId>,
    pub pipeline_layouts: HashMap<vk::PipelineLayout, ir::PipelineLayoutDesc>,
    pub pipeline_binding_maps: HashMap<vk::PipelineLayout, Vec<DescriptorBindingMap>>,
    pub pipeline_binding_stages:
        HashMap<vk::PipelineLayout, Vec<BTreeMap<u32, vk::ShaderStageFlags>>>,
    pub descriptor_pools: HashMap<vk::DescriptorPool, DescriptorPool>,
    pub descriptor_sets: HashMap<vk::DescriptorSet, DescriptorSet>,
    bind_group_cache: Vec<(ir::BindGroupDesc, ir::BindGroupId)>,
    pub pipeline_caches: HashMap<vk::PipelineCache, Vec<u8>>,
    pub pipelines: HashMap<vk::Pipeline, Pipeline>,
    graphics_swizzles: HashMap<(vk::Pipeline, ShaderViews), ir::ProgrammableRenderPipelineId>,
    compute_swizzles: HashMap<(vk::Pipeline, ShaderViews), ir::ComputePipelineId>,
    pub graphics_extents: HashMap<vk::Pipeline, vk::Extent2D>,
    pub graphics_state: HashMap<vk::Pipeline, crate::images::GraphicsDynamicState>,
    pub graphics_inputs: HashMap<vk::Pipeline, Vec<crate::input_attachments::InputBinding>>,
    pub graphics_subpasses: HashMap<vk::Pipeline, (crate::render_pass::RenderPass, usize)>,
    pub images: HashMap<vk::Image, crate::images::Image>,
    pub views: HashMap<vk::ImageView, crate::images::ImageView>,
    pub render_passes: HashMap<vk::RenderPass, crate::images::RenderPass>,
    pub framebuffers: HashMap<vk::Framebuffer, crate::images::Framebuffer>,
    #[cfg(any(target_os = "scarlet", target_os = "macos", feature = "scarlet-wsi"))]
    pub swapchains: HashMap<vk::SwapchainKHR, crate::wsi::Swapchain>,
}
impl Resources {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn forget_uploads_for_memory(&mut self, memory: vk::DeviceMemory) {
        let buffers = &self.buffers;
        self.uploaded_unmapped_buffers.retain(|handle| {
            buffers
                .get(handle)
                .is_some_and(|buffer| buffer.bound.is_some_and(|(bound, _)| bound != memory))
        });
    }
    pub(crate) fn forget_uploads_for_memory_range(
        &mut self,
        memory: vk::DeviceMemory,
        range: std::ops::Range<u64>,
        synchronized: vk::Buffer,
    ) {
        let buffers = &self.buffers;
        self.uploaded_unmapped_buffers.retain(|handle| {
            *handle == synchronized
                || buffers.get(handle).is_some_and(|buffer| {
                    buffer.bound.is_some_and(|(bound, offset)| {
                        bound != memory
                            || offset >= range.end
                            || offset + buffer.size <= range.start
                    })
                })
        });
    }
    /// Cache shader variants for the component maps of statically used views.
    /// Identity bindings use the original pipeline without allocating a key.
    pub(crate) fn graphics_pipeline_for_views(
        &mut self,
        table: &ir::ResourceTable,
        handle: vk::Pipeline,
        sets: &BTreeMap<u32, vk::DescriptorSet>,
    ) -> Result<ir::ProgrammableRenderPipelineId, vk::Result> {
        let invalid = vk::Result::ERROR_INITIALIZATION_FAILED;
        let Some(Pipeline::Graphics(id)) = self.pipelines.get(&handle).copied() else {
            return Err(invalid);
        };
        let desc = table
            .programmable_render_pipeline_shared(
                table
                    .programmable_render_pipeline_ref(id)
                    .map_err(failure)?,
            )
            .map_err(failure)?;
        let maps = self.texture_swizzles(desc.layout(), sets)?;
        if maps.is_empty() {
            return Ok(id);
        }
        if let Some((_, id)) = self
            .graphics_swizzles
            .iter()
            .find(|((pipeline, key), _)| *pipeline == handle && *key == maps)
        {
            return Ok(*id);
        }
        let previous_modules = self.orphaned_shader_modules.clone();
        let result = (|| {
            let mut shader =
                |entry| swizzle_entry(table, entry, &maps, &mut self.orphaned_shader_modules);
            let mut variant = ir::ProgrammableRenderPipelineDesc::new(
                shader(desc.vertex())?,
                shader(desc.fragment())?,
                desc.layout().clone(),
                desc.target_format(),
                None,
                desc.topology(),
                desc.blend(),
                desc.raster(),
            )
            .map_err(failure)?
            .with_vertex_buffers(desc.vertex_buffers().to_vec())
            .map_err(failure)?
            .with_color_targets(desc.color_targets().collect())
            .map_err(failure)?;
            if let Some(depth) = desc.depth_state() {
                variant = variant.with_depth_state(depth).map_err(failure)?;
            }
            let id = table
                .define_programmable_render_pipeline(variant)
                .map_err(failure)?
                .id();
            self.graphics_swizzles.insert((handle, maps), id);
            Ok(id)
        })();
        if result.is_err() {
            self.rollback_swizzle_modules(table, &previous_modules)?;
        }
        result
    }

    pub(crate) fn compute_pipeline_for_views(
        &mut self,
        table: &ir::ResourceTable,
        handle: vk::Pipeline,
        sets: &BTreeMap<u32, vk::DescriptorSet>,
    ) -> Result<ir::ComputePipelineId, vk::Result> {
        let invalid = vk::Result::ERROR_INITIALIZATION_FAILED;
        let Some(Pipeline::Compute(id)) = self.pipelines.get(&handle).copied() else {
            return Err(invalid);
        };
        let desc = table
            .compute_pipeline(table.compute_pipeline_ref(id).map_err(failure)?)
            .map_err(failure)?;
        let maps = self.texture_swizzles(desc.layout(), sets)?;
        if maps.is_empty() {
            return Ok(id);
        }
        if let Some((_, id)) = self
            .compute_swizzles
            .iter()
            .find(|((pipeline, key), _)| *pipeline == handle && *key == maps)
        {
            return Ok(*id);
        }
        let previous_modules = self.orphaned_shader_modules.clone();
        let result = (|| {
            let variant = ir::ComputePipelineDesc::new(
                swizzle_entry(
                    table,
                    desc.shader(),
                    &maps,
                    &mut self.orphaned_shader_modules,
                )?,
                desc.layout().clone(),
            )
            .map_err(failure)?;
            let id = table
                .define_compute_pipeline(variant)
                .map_err(failure)?
                .id();
            self.compute_swizzles.insert((handle, maps), id);
            Ok(id)
        })();
        if result.is_err() {
            self.rollback_swizzle_modules(table, &previous_modules)?;
        }
        result
    }

    fn rollback_swizzle_modules(
        &mut self,
        table: &ir::ResourceTable,
        previous: &HashSet<ir::ShaderModuleId>,
    ) -> Result<(), vk::Result> {
        let created = self
            .orphaned_shader_modules
            .difference(previous)
            .copied()
            .collect::<Vec<_>>();
        // These descriptors were never published to an executor/materialization
        // cache: a failed view variant cannot retain just its first shader.
        for id in created {
            table.release_shader_module(id).map_err(failure)?;
            self.orphaned_shader_modules.remove(&id);
        }
        Ok(())
    }

    fn texture_swizzles(
        &self,
        layout: &ir::PipelineLayoutDesc,
        sets: &BTreeMap<u32, vk::DescriptorSet>,
    ) -> Result<ShaderViews, vk::Result> {
        let invalid = vk::Result::ERROR_INITIALIZATION_FAILED;
        let mut maps = ShaderViews::default();
        for (group, layout) in layout.bind_groups().iter().enumerate() {
            for entry in layout.entries() {
                if !matches!(
                    entry.ty(),
                    ir::BindingType::SampledTextureView { .. }
                        | ir::BindingType::StorageBuffer { read_only: true }
                ) {
                    continue;
                }
                let set = sets
                    .get(&(group as u32))
                    .and_then(|set| self.descriptor_sets.get(set))
                    .ok_or(invalid)?;
                if matches!(entry.ty(), ir::BindingType::StorageBuffer { .. }) {
                    let binding = set.ir_to_source(entry.binding()).ok_or(invalid)?;
                    if set.texel_formats.get(&binding) == Some(&vk::Format::R8_SINT) {
                        maps.narrow_texels.push((group as u32, entry.binding()));
                    }
                    continue;
                }
                let Some(DescriptorBinding::Image { view, .. }) = set
                    .bindings
                    .get(&set.ir_to_source(entry.binding()).ok_or(invalid)?)
                else {
                    return Err(invalid);
                };
                let view = self.views.get(view).ok_or(invalid)?;
                if view.components != [0, 1, 2, 3] {
                    maps.textures
                        .push((group as u32, entry.binding(), view.components));
                }
            }
        }
        Ok(maps)
    }

    /// Whether any Vulkan object still depends on this epoch's IR table.
    /// Host memory and descriptor-pool quotas survive epoch reclamation because
    /// they contain no IR identities; layouts conservatively keep their epoch.
    pub(crate) fn has_live_ir_objects(&self) -> bool {
        !self.buffers.is_empty()
            || !self.samplers.is_empty()
            || !self.images.is_empty()
            || !self.shaders.is_empty()
            || !self.pipelines.is_empty()
            || !self.set_layouts.is_empty()
            || !self.pipeline_layouts.is_empty()
            || !self.descriptor_sets.is_empty()
            || {
                #[cfg(any(target_os = "scarlet", target_os = "macos", feature = "scarlet-wsi"))]
                {
                    !self.swapchains.is_empty()
                }
                #[cfg(not(any(
                    target_os = "scarlet",
                    target_os = "macos",
                    feature = "scarlet-wsi"
                )))]
                {
                    false
                }
            }
    }

    /// Forget every cached identity before swapping to a fresh IR table.
    pub(crate) fn clear_ir_cache(&mut self) {
        self.mip_render_targets.clear();
        self.depth_clear_colors.clear();
        self.depth_only_fragment = None;
        self.uploaded_unmapped_buffers.clear();
        self.bind_group_cache.clear();
        self.graphics_swizzles.clear();
        self.compute_swizzles.clear();
        self.orphaned_shader_modules.clear();
        for set in self.descriptor_sets.values_mut() {
            set.cached_group = None;
        }
    }

    /// Retire descriptor snapshots after every recording in a queue packet has
    /// been encoded. Vulkan command recordings retain set handles, not these
    /// transient IR identities; submitted backend commands own their bindings.
    /// Keep the single shared empty group used to clear inactive layout slots.
    pub(crate) fn release_descriptor_groups(
        &mut self,
        table: &ir::ResourceTable,
        mut release_backend: impl FnMut(ir::BindGroupId) -> Result<(), vk::Result>,
    ) -> Result<(), vk::Result> {
        for set in self.descriptor_sets.values_mut() {
            set.cached_group = None;
        }
        let mut result = Ok(());
        let mut retired = Vec::new();
        for (desc, id) in std::mem::take(&mut self.bind_group_cache) {
            if desc.layout().entries().is_empty() {
                self.bind_group_cache.push((desc, id));
            } else if let Err(error) = release_backend(id) {
                // Preserve failed retirements for diagnosis or a retry, while
                // still releasing the rest of this packet's snapshots.
                self.bind_group_cache.push((desc, id));
                result = Err(error);
            } else {
                retired.push((desc, id));
            }
        }
        // Keep one table revision for the whole backend retirement batch.
        // Each logical release changes the ABI revision; interleaving it with
        // backend calls would serialize the entire resource table per group.
        for (desc, id) in retired {
            if let Err(error) = table.release_bind_group(id).map_err(failure) {
                self.bind_group_cache.push((desc, id));
                result = Err(error);
            }
        }
        result
    }

    pub(crate) fn empty_bind_group(
        &mut self,
        table: &ir::ResourceTable,
    ) -> Result<ir::BindGroupId, vk::Result> {
        if let Some((_, id)) = self
            .bind_group_cache
            .iter()
            .find(|(desc, _)| desc.layout().entries().is_empty())
        {
            return Ok(*id);
        }
        let layout = ir::BindGroupLayoutDesc::new(Vec::new()).map_err(failure)?;
        let desc = ir::BindGroupDesc::new(table, layout, Vec::new()).map_err(failure)?;
        let id = table.define_bind_group(desc.clone()).map_err(failure)?.id();
        self.bind_group_cache.push((desc, id));
        Ok(id)
    }

    #[cfg(test)]
    pub fn descriptor_group(
        &mut self,
        table: &ir::ResourceTable,
        handle: vk::DescriptorSet,
        dynamic_offsets: &[u32],
    ) -> Result<ir::BindGroupId, vk::Result> {
        let layout = self
            .descriptor_sets
            .get(&handle)
            .ok_or(vk::Result::ERROR_UNKNOWN)?
            .layout
            .clone();
        self.descriptor_group_with_layout(table, handle, dynamic_offsets, &layout, 256, 256)
    }

    pub fn descriptor_group_with_layout(
        &mut self,
        table: &ir::ResourceTable,
        handle: vk::DescriptorSet,
        dynamic_offsets: &[u32],
        layout: &ir::BindGroupLayoutDesc,
        uniform_alignment: u32,
        storage_alignment: u32,
    ) -> Result<ir::BindGroupId, vk::Result> {
        let invalid = vk::Result::ERROR_INITIALIZATION_FAILED;
        let set = self.descriptor_sets.get(&handle).ok_or(invalid)?;
        if set.invalid || !specialized_layout_matches(&set.layout, layout) {
            return Err(invalid);
        }
        let dynamic_bindings = set
            .types
            .iter()
            .filter(|(_, ty)| {
                matches!(
                    **ty,
                    vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                        | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
                )
            })
            .map(|(&binding, ty)| (binding, *ty))
            .collect::<Vec<_>>();
        if dynamic_bindings.len() != dynamic_offsets.len()
            || dynamic_offsets
                .iter()
                .zip(&dynamic_bindings)
                .any(|(offset, (_, ty))| {
                    let alignment = if *ty == vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC {
                        uniform_alignment
                    } else {
                        storage_alignment
                    };
                    !offset.is_multiple_of(alignment)
                })
        {
            return Err(invalid);
        }
        let mut entries = Vec::with_capacity(layout.entries().len());
        // Only statically used descriptors need resources. Vulkan's layout
        // itself does not contain image dimensionality or sampler comparison.
        for entry in layout.entries() {
            let binding = set.ir_to_source(entry.binding()).ok_or(invalid)?;
            let resource = *set.bindings.get(&binding).ok_or(invalid)?;
            let resource = match resource {
                DescriptorBinding::TexelBuffer(_) => {
                    return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                }
                DescriptorBinding::Buffer {
                    buffer,
                    offset,
                    range,
                } => {
                    let buf = self.buffers.get(&buffer).ok_or(invalid)?;
                    if buf.bound.is_none() {
                        return Err(invalid);
                    }
                    let size = if range == vk::WHOLE_SIZE {
                        buf.size.checked_sub(offset).ok_or(invalid)?
                    } else {
                        range
                    };
                    let dynamic = dynamic_bindings
                        .iter()
                        .position(|(b, _)| *b == binding)
                        .map(|index| u64::from(dynamic_offsets[index]))
                        .unwrap_or(0);
                    if range == vk::WHOLE_SIZE && dynamic != 0 {
                        return Err(invalid);
                    }
                    let offset = offset.checked_add(dynamic).ok_or(invalid)?;
                    if offset.checked_add(size).is_none_or(|end| end > buf.size) {
                        return Err(invalid);
                    }
                    let size = if set.texel_formats.get(&binding) == Some(&vk::Format::R8_SINT) {
                        size.next_multiple_of(4)
                    } else {
                        size
                    };
                    ir::BindingResource::Buffer {
                        buffer: buf.id,
                        offset,
                        size,
                    }
                }
                DescriptorBinding::Image {
                    view,
                    sampler,
                    layout: image_layout,
                } => {
                    if entry.binding() % 2 == 1 {
                        if set.types.get(&binding)
                            != Some(&vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        {
                            return Err(invalid);
                        }
                        ir::BindingResource::Sampler(*self.samplers.get(&sampler).ok_or(invalid)?)
                    } else {
                        if !matches!(
                            image_layout,
                            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                                | vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
                                | vk::ImageLayout::GENERAL
                        ) {
                            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                        }
                        let view = self.views.get(&view).ok_or(invalid)?;
                        let image = self.images.get(&view.image).ok_or(invalid)?;
                        let storage = matches!(entry.ty(), ir::BindingType::StorageTexture { .. });
                        if !image.usable()
                            || !image.usage.contains(if storage {
                                vk::ImageUsageFlags::STORAGE
                            } else if set.types.get(&binding)
                                == Some(&vk::DescriptorType::INPUT_ATTACHMENT)
                            {
                                vk::ImageUsageFlags::INPUT_ATTACHMENT
                            } else {
                                vk::ImageUsageFlags::SAMPLED
                            })
                            || (storage
                                && (image_layout != vk::ImageLayout::GENERAL
                                    || view.components != [0, 1, 2, 3]))
                        {
                            return Err(invalid);
                        }
                        if matches!(
                            entry.ty(),
                            ir::BindingType::SampledTextureView { .. }
                                | ir::BindingType::StorageTexture { .. }
                        ) {
                            ir::BindingResource::TextureView {
                                texture: image.id,
                                view: view.desc,
                            }
                        } else {
                            ir::BindingResource::Texture(image.id)
                        }
                    }
                }
                DescriptorBinding::Sampler(sampler) => {
                    ir::BindingResource::Sampler(*self.samplers.get(&sampler).ok_or(invalid)?)
                }
            };
            entries.push(ir::BindGroupEntry::new(entry.binding(), resource));
        }
        let desc = ir::BindGroupDesc::new(table, layout.clone(), entries).map_err(failure)?;
        // Validate live resources above even on a cache hit. A set may be used
        // by pipelines with distinct reflected views of its Vulkan layout.
        if dynamic_offsets.is_empty()
            && let Some(id) = set.cached_group
        {
            let cached = table
                .bind_group(table.bind_group_ref(id).map_err(failure)?)
                .map_err(failure)?;
            if cached == desc {
                return Ok(id);
            }
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
        if dynamic_offsets.is_empty() {
            self.descriptor_sets
                .get_mut(&handle)
                .ok_or(invalid)?
                .cached_group = Some(id);
        }
        Ok(id)
    }
}
fn swizzle_entry(
    table: &ir::ResourceTable,
    entry: &ir::ShaderEntryPoint,
    maps: &ShaderViews,
    orphaned: &mut HashSet<ir::ShaderModuleId>,
) -> Result<ir::ShaderEntryPoint, vk::Result> {
    let original = table
        .shader_module(table.shader_module_ref(entry.module()).map_err(failure)?)
        .map_err(failure)?;
    let ir::ShaderSource::SpirV(words) = original.source() else {
        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
    };
    let changed = crate::spirv::sampled_image_swizzles(words, &maps.textures)?;
    let changed = crate::texel_buffers::narrow_signed_bytes(changed, &maps.narrow_texels)?;
    if changed == *words {
        return Ok(entry.clone());
    }
    let module = table
        .define_shader_module(ir::ShaderModuleDesc::spirv(changed).map_err(failure)?)
        .map_err(failure)?;
    orphaned.insert(module.id());
    ir::ShaderEntryPoint::new(module, entry.stage(), entry.entry_point().into()).map_err(failure)
}

/// A shader may specialize and omit entries from a Vulkan descriptor layout.
pub(crate) fn specialized_layout_matches(
    declared: &ir::BindGroupLayoutDesc,
    used: &ir::BindGroupLayoutDesc,
) -> bool {
    used.entries().iter().all(|entry| {
        declared.entries().iter().any(|original| {
            original.binding() == entry.binding()
                && original.visibility() == entry.visibility()
                && (original.ty() == entry.ty()
                    || matches!(
                        (original.ty(), entry.ty()),
                        (
                            ir::BindingType::SampledTexture,
                            ir::BindingType::SampledTextureView { .. }
                        ) | (ir::BindingType::Sampler, ir::BindingType::ComparisonSampler)
                            | (
                                ir::BindingType::StorageBuffer { read_only: false },
                                ir::BindingType::StorageBuffer { read_only: true }
                            )
                    ))
        })
    })
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
        && !resources
            .images
            .values()
            .any(|i| overlaps(i.bound, i.byte_size()))
}
pub(crate) fn backend_failure(error: crate::runtime::BackendError) -> vk::Result {
    crate::runtime::backend_failure(error)
}
pub(crate) fn failure(error: ir::Error) -> vk::Result {
    eprintln!("[SGFX Vulkan] IR failure: {error:?}");
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
fn supported_descriptor_type(ty: vk::DescriptorType) -> bool {
    matches!(
        ty,
        vk::DescriptorType::UNIFORM_BUFFER
            | vk::DescriptorType::STORAGE_BUFFER
            | vk::DescriptorType::SAMPLED_IMAGE
            | vk::DescriptorType::INPUT_ATTACHMENT
            | vk::DescriptorType::STORAGE_IMAGE
            | vk::DescriptorType::SAMPLER
            | vk::DescriptorType::COMBINED_IMAGE_SAMPLER
            | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
            | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
    )
}
pub(crate) fn stages(flags: vk::ShaderStageFlags) -> Result<ir::ShaderStages, vk::Result> {
    let flags = if flags == vk::ShaderStageFlags::ALL {
        vk::ShaderStageFlags::VERTEX
            | vk::ShaderStageFlags::FRAGMENT
            | vk::ShaderStageFlags::COMPUTE
    } else if flags == vk::ShaderStageFlags::ALL_GRAPHICS {
        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT
    } else {
        flags
    };
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
            | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER
            | vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER
            | vk::BufferUsageFlags::INDIRECT_BUFFER;
        if !i.p_next.is_null()
            || !i.flags.is_empty()
            || i.size == 0
            || i.size > MAX_ALLOCATION
            || i.usage.is_empty()
            || !allowed.contains(i.usage)
            || i.sharing_mode != vk::SharingMode::EXCLUSIVE
        {
            eprintln!(
                "SGFX Vulkan: unsupported buffer size={} usage={:?} flags={:?} sharing={:?} pnext={:?}",
                i.size, i.usage, i.flags, i.sharing_mode, i.p_next
            );
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let size = i.size;
        // Vulkan buffer sizes are byte counts. Pad only the SGFX backing
        // buffer for backends that upload in four-byte units.
        let backing_size = size.next_multiple_of(4);
        let usage = i.usage;
        // Usage flags authorize future operations; reserving the byte storage
        // does not execute them. Texel views and indirect draws remain rejected
        // at their typed entry points until the backend supports those commands.
        let mut ir_usage = ir::BufferUsage::COPY_SRC | ir::BufferUsage::COPY_DST;
        for (vk_flag, ir_flag) in [
            (vk::BufferUsageFlags::VERTEX_BUFFER, ir::BufferUsage::VERTEX),
            (vk::BufferUsageFlags::INDEX_BUFFER, ir::BufferUsage::INDEX),
            (
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                ir::BufferUsage::UNIFORM,
            ),
            (
                vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER,
                ir::BufferUsage::STORAGE,
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
            if r.resources.buffers.len() >= ir::MAX_BUFFERS {
                return Err(vk::Result::ERROR_TOO_MANY_OBJECTS);
            }
            let id = r
                .table
                .define_buffer(ir::BufferDesc::new(backing_size, ir_usage).map_err(failure)?)
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
        if let Some(id) = r.resources.buffers.get(&buffer).map(|buffer| buffer.id) {
            let shared = r
                .resources
                .buffers
                .iter()
                .any(|(handle, other)| *handle != buffer && other.id == id);
            // The backend cache owns the physical allocation for this slot.
            // Complete submitted work before dropping it or recycling the ID.
            if !shared {
                r.in_flight.wait();
                r.cache
                    .release_buffer(id)
                    .map_err(crate::runtime::backend_failure)?;
                r.table.release_buffer(id).map_err(failure)?;
            }
            r.resources.buffers.remove(&buffer);
            r.resources.uploaded_unmapped_buffers.remove(&buffer);
        }
        Ok(())
    });
}
pub(crate) unsafe extern "system" fn buffer_requirements(
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
            size: size.next_multiple_of(256),
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
        let dedicated = crate::memory_allocation::parse(i.p_next.cast())?;
        if i.memory_type_index != 0 || i.allocation_size == 0 || i.allocation_size > MAX_ALLOCATION
        {
            let tag = i
                .p_next
                .cast::<vk::BaseInStructure<'_>>()
                .as_ref()
                .map(|node| node.s_type);
            eprintln!(
                "[SGFX Vulkan] memory allocation rejected: size={} type={} pnext={tag:?} max={MAX_ALLOCATION}",
                i.allocation_size, i.memory_type_index
            );
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let size =
            usize::try_from(i.allocation_size).map_err(|_| vk::Result::ERROR_OUT_OF_HOST_MEMORY)?;
        let handle = crate::api::with_device(device, move |r| {
            if let Some(target) = dedicated
                && crate::memory_allocation::allocation_size(&r.resources, target)
                    != Some(size as u64)
            {
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            }
            if r.resources.memories.len() >= MAX_MEMORY_ALLOCATIONS {
                return Err(vk::Result::ERROR_TOO_MANY_OBJECTS);
            }
            let budget = crate::memory_budget::heap_size();
            let allocated: usize = r.resources.memories.values().map(|m| m.bytes.len()).sum();
            if allocated.checked_add(size).is_none_or(|v| v > budget) {
                // Allocation failures must identify the exhausted budget even
                // in ordinary builds. Do not dump mapped application payloads.
                let mut largest: Vec<_> = r
                    .resources
                    .memories
                    .values()
                    .map(|m| (m.bytes.len(), m.mapped))
                    .collect();
                largest.sort_unstable_by(|a, b| b.0.cmp(&a.0));
                largest.truncate(12);
                let mut image_groups = BTreeMap::<_, (usize, u64)>::new();
                for image in r
                    .resources
                    .images
                    .values()
                    .filter(|image| image.bound.is_some())
                {
                    let key = (
                        image.extent.width,
                        image.extent.height,
                        image.format.as_raw(),
                        image.usage.as_raw(),
                    );
                    let group = image_groups.entry(key).or_default();
                    group.0 += 1;
                    group.1 = group.1.saturating_add(image.byte_size());
                }
                let mut image_groups: Vec<_> = image_groups.into_iter().collect();
                image_groups.sort_unstable_by(|a, b| b.1.1.cmp(&a.1.1));
                image_groups.truncate(12);
                let mut buffer_groups = BTreeMap::<_, (usize, u64)>::new();
                for buffer in r
                    .resources
                    .buffers
                    .values()
                    .filter(|buffer| buffer.bound.is_some())
                {
                    let group = buffer_groups.entry(buffer.usage.as_raw()).or_default();
                    group.0 += 1;
                    group.1 = group.1.saturating_add(buffer.size);
                }
                let cpu_bytes = r
                    .resources
                    .memories
                    .values()
                    .map(|memory| memory.bytes.words.len() * 8)
                    .sum::<usize>();
                eprintln!(
                    "[SGFX Vulkan] allocation budget exhausted: requested={size} allocated={allocated} limit={budget} allocations={} buffers={} images={} largest_bytes_mapped={largest:?}",
                    r.resources.memories.len(),
                    r.resources.buffers.len(),
                    r.resources.images.len(),
                );
                eprintln!(
                    "[SGFX Vulkan] allocation owners: cpu_mirror_bytes={cpu_bytes} image_groups=((width,height,format,usage),(count,bytes)) {image_groups:?} buffer_groups=(usage,(count,bytes)) {buffer_groups:?}"
                );
                return Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
            }
            let bytes = AlignedBytes::unbacked(size)?;
            let handle = vk::DeviceMemory::from_raw(crate::api::next_id());
            if let Some(target) = dedicated {
                r.resources.dedicated_allocations.insert(handle, target);
            }
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
        r.resources.forget_uploads_for_memory(memory);
        r.resources.memories.remove(&memory);
        r.resources.dedicated_allocations.remove(&memory);
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
pub(crate) unsafe extern "system" fn bind_buffer_memory(
    device: vk::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    offset: vk::DeviceSize,
) -> vk::Result {
    ffi(|| {
        crate::api::with_device(device, move |r| {
            if !crate::memory_allocation::permits(
                &r.resources,
                memory,
                crate::memory_allocation::DedicatedTarget::Buffer(buffer),
                offset,
            ) {
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            }
            let current = r
                .resources
                .buffers
                .get(&buffer)
                .ok_or(vk::Result::ERROR_UNKNOWN)?;
            // Equal byte ranges share one physical GPU identity. Partial buffer
            // aliases and image aliases still require backend memory views.
            let alias = r.resources.buffers.iter().find_map(|(handle, other)| {
                (*handle != buffer
                    && other.bound == Some((memory, offset))
                    && other.size == current.size)
                    .then_some(other.id)
            });
            if let Some(alias_id) = alias {
                let current_desc = r
                    .table
                    .buffer(r.table.buffer_ref(current.id).map_err(failure)?)
                    .map_err(failure)?;
                let alias_desc = r
                    .table
                    .buffer(r.table.buffer_ref(alias_id).map_err(failure)?)
                    .map_err(failure)?;
                if current.bound.is_some()
                    || !offset.is_multiple_of(256)
                    || !alias_desc.usage().contains(current_desc.usage())
                {
                    return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                }
                let unused_id = current.id;
                r.table.release_buffer(unused_id).map_err(failure)?;
                let current = r.resources.buffers.get_mut(&buffer).unwrap();
                current.id = alias_id;
                current.bound = Some((memory, offset));
                return Ok(());
            }
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
                if std::env::var_os("SGFX_VULKAN_TRACE").is_some() {
                    eprintln!(
                        "[SGFX Vulkan] buffer memory overlap/range: buffer={buffer:?} memory={memory:?} offset={offset} size={:?} allocation={:?}",
                        r.resources.buffers.get(&buffer).map(|b| b.size),
                        r.resources.memories.get(&memory).map(|m| m.bytes.len())
                    );
                }
                return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
            }
            let mem = r
                .resources
                .memories
                .get_mut(&memory)
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
                if std::env::var_os("SGFX_VULKAN_TRACE").is_some() {
                    eprintln!(
                        "[SGFX Vulkan] buffer bind: buffer={buffer:?} offset={offset} size={} bound={:?} allocation={}",
                        buf.size,
                        buf.bound,
                        mem.bytes.len()
                    );
                }
                return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
            }
            // Bound buffers can be snapshotted without an application mapping.
            // Allocate their CPU storage before publishing a successful bind.
            mem.bytes.materialize()?;
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
            mem.bytes
                .materialize()
                .map_err(|_| vk::Result::ERROR_MEMORY_MAP_FAILED)?;
            mem.mapped = true;
            let address = mem.bytes.as_mut_ptr().add(offset as usize) as usize;
            r.resources.forget_uploads_for_memory(memory);
            Ok(address)
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
#[cfg(test)]
pub(crate) fn normalize_spirv(words: Vec<u32>) -> Result<ir::ShaderModuleDesc, vk::Result> {
    normalize_spirv_specialized(words, &Vec::new(), &[], &[])
}
#[cfg(test)]
pub(crate) fn normalize_spirv_for_layout(
    words: Vec<u32>,
    bindings: &[DescriptorBindingMap],
) -> Result<ir::ShaderModuleDesc, vk::Result> {
    normalize_spirv_specialized(words, &Vec::new(), &[], bindings)
}
fn normalize_spirv_specialized(
    words: Vec<u32>,
    values: &Specialization,
    inputs: &[Option<bool>],
    bindings: &[DescriptorBindingMap],
) -> Result<ir::ShaderModuleDesc, vk::Result> {
    ir::ShaderModuleDesc::spirv(words.clone())
        .map_err(failure)
        .inspect_err(|e| eprintln!("[SGFX Vulkan] shader resource header: {e:?}"))?;
    let words = crate::specialization::freeze(words, values).inspect_err(|error| {
        eprintln!("[SGFX Vulkan] shader specialization: {error:?}");
    })?;
    let words = crate::input_attachments::lower(words, inputs).inspect_err(|error| {
        eprintln!("[SGFX Vulkan] shader input attachments: {error:?}");
    })?;
    let words = crate::texel_buffers::lower(words)
        .inspect_err(|e| eprintln!("[SGFX Vulkan] texel buffer lowering: {e:?}"))?;
    let words = crate::spirv::separate_combined_samplers(words).inspect_err(|error| {
        eprintln!("[SGFX Vulkan] shader combined samplers: {error:?}");
    })?;
    let words = crate::uniform_arrays::lower(words).inspect_err(|error| {
        eprintln!("[SGFX Vulkan] shader uniform arrays: {error:?}");
    })?;
    let words = crate::stage_inputs::lower(words).inspect_err(|error| {
        eprintln!("[SGFX Vulkan] shader stage inputs: {error:?}");
    })?;
    let options = naga::front::spv::Options {
        adjust_coordinate_space: true,
        strict_capabilities: true,
        block_ctx_dump_prefix: None,
    };
    let draw_parameters = crate::spirv::instructions(&words)?
        .iter()
        .any(|i| i[0] & 65535 == 71 && i.len() == 4 && i[2] == 11 && (4424..=4426).contains(&i[3]));
    let words = ir::spirv_compat::naga24_spirv(&words)
        .map_err(|_| vk::Result::ERROR_FEATURE_NOT_PRESENT)?;
    let mut module = naga::front::spv::Frontend::new(words.into_iter(), &options)
        .parse()
        .map_err(|e| {
            {
                eprintln!("[SGFX Vulkan] SPIR-V parse: {e:?}");
            }
            vk::Result::ERROR_FEATURE_NOT_PRESENT
        })?;
    crate::shader_functions::specialize(&mut module)
        .inspect_err(|e| eprintln!("[SGFX Vulkan] shader function specialization: {e:?}"))?;
    crate::shader_functions::resolve_sampler_types(&mut module)
        .inspect_err(|e| eprintln!("[SGFX Vulkan] shader sampler types: {e:?}"))?;
    crate::shader_functions::flatten_fragment_output_arrays(&mut module)?;
    crate::shader_functions::widen_fragment_outputs(&mut module)
        .inspect_err(|e| eprintln!("[SGFX Vulkan] shader fragment outputs: {e:?}"))?;
    // SPIR-V combined image samplers become two Naga globals with the same
    // Vulkan binding. SGFX has separate texture/sampler resources. Reserve an
    // adjacent pair per logical binding, including ordinary buffer bindings.
    let images = module
        .global_variables
        .iter()
        .filter_map(|(_, global)| {
            matches!(module.types[global.ty].inner, naga::TypeInner::Image { .. })
                .then(|| global.binding.clone())
                .flatten()
        })
        .collect::<Vec<_>>();
    for (_, global) in module.global_variables.iter_mut() {
        if let Some(binding) = global.binding.as_mut() {
            let combined_sampler = matches!(
                module.types[global.ty].inner,
                naga::TypeInner::Sampler { .. }
            ) && images.iter().any(|image| image == binding);
            let base = if bindings.is_empty() {
                binding
                    .binding
                    .checked_mul(2)
                    .filter(|slot| *slot < ir::MAX_BINDINGS_PER_GROUP as u32)
            } else {
                bindings
                    .get(binding.group as usize)
                    .and_then(|map| map.get(&binding.binding))
                    .copied()
            }
            .ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)?;
            binding.binding = base + u32::from(combined_sampler);
        }
    }
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::PUSH_CONSTANT,
    )
    .validate(&module)
    .map_err(|e| {
        {
            eprintln!("[SGFX Vulkan] shader validation: {e:?}");
        }
        vk::Result::ERROR_FEATURE_NOT_PRESENT
    })?;
    let mut output = naga::back::spv::Options {
        lang_version: if draw_parameters { (1, 3) } else { (1, 0) },
        ..Default::default()
    };
    output
        .flags
        .remove(naga::back::spv::WriterFlags::ADJUST_COORDINATE_SPACE);
    let mut normalized =
        naga::back::spv::write_vec(&module, &info, &output, None).map_err(|error| {
            {
                eprintln!("[SGFX Vulkan] normalized SPIR-V: {error:?}");
            }
            vk::Result::ERROR_FEATURE_NOT_PRESENT
        })?;
    if draw_parameters
        && !crate::spirv::instructions(&normalized)?
            .iter()
            .any(|i| i[0] & 65535 == 17 && i.get(1) == Some(&4427))
    {
        normalized.splice(5..5, [(2u32 << 16) | 17, 4427]);
    }
    // Validate the representation consumed by backends, including the SPIR-V
    // writer/parser round trip. WGPU 24 cannot format SPIR-V validation spans
    // against its empty text source and would otherwise panic on this path.
    let options = naga::front::spv::Options {
        adjust_coordinate_space: false,
        ..options
    };
    let normalized = crate::spirv::function_struct_layouts(normalized)?;
    let normalized_module = naga::front::spv::Frontend::new(
        ir::spirv_compat::naga24_spirv(&normalized)
            .map_err(|_| vk::Result::ERROR_FEATURE_NOT_PRESENT)?
            .into_iter(),
        &options,
    )
    .parse()
    .map_err(|error| {
        {
            eprintln!("[SGFX Vulkan] normalized SPIR-V: {error:?}");
        }
        vk::Result::ERROR_FEATURE_NOT_PRESENT
    })?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::PUSH_CONSTANT,
    )
    .validate(&normalized_module)
    .map_err(|error| {
        {
            eprintln!("[SGFX Vulkan] normalized SPIR-V: {error:?}");
        }
        vk::Result::ERROR_FEATURE_NOT_PRESENT
    })?;
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
        ir::ShaderModuleDesc::spirv(words.clone()).map_err(failure)?;
        let handle = crate::api::with_device(device, move |r| {
            if r.resources.shaders.len() >= LIMIT {
                return Err(vk::Result::ERROR_TOO_MANY_OBJECTS);
            }
            let handle = vk::ShaderModule::from_raw(crate::api::next_id());
            r.resources.shaders.insert(
                handle,
                Shader {
                    words,
                    variants: Vec::new(),
                },
            );
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
    let _ = crate::api::with_device(device, move |r| retire_vk_shader_module(r, handle));
}
fn retire_vk_shader_module(
    r: &mut crate::runtime::Runtime,
    handle: vk::ShaderModule,
) -> Result<(), vk::Result> {
    if let Some(shader) = r.resources.shaders.remove(&handle) {
        r.resources
            .orphaned_shader_modules
            .extend(shader.variants.into_iter().map(|variant| variant.3));
        retire_unused_shader_modules(r)?;
    }
    Ok(())
}
type DescriptorLayoutParts = (
    ir::BindGroupLayoutDesc,
    BTreeMap<u32, vk::DescriptorType>,
    BTreeMap<u32, vk::ShaderStageFlags>,
);

/// Pure validation shared by creation and maintenance3 support queries.
pub(crate) unsafe fn descriptor_layout(
    i: &vk::DescriptorSetLayoutCreateInfo<'_>,
) -> Result<DescriptorLayoutParts, vk::Result> {
    if i.s_type != vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO {
        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
    }
    if !vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR.contains(i.flags) {
        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
    }
    if !i.p_next.is_null() {
        let flags = &*i
            .p_next
            .cast::<vk::DescriptorSetLayoutBindingFlagsCreateInfo<'_>>();
        if flags.s_type != vk::StructureType::DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO
            || !flags.p_next.is_null()
            || (flags.binding_count != 0 && flags.binding_count != i.binding_count)
            || copied(
                flags.p_binding_flags,
                flags.binding_count as usize,
                ir::MAX_BINDINGS_PER_GROUP,
            )?
            .iter()
            .any(|flag| !flag.is_empty())
        {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
    }
    let bindings = copied(
        i.p_bindings,
        i.binding_count as usize,
        ir::MAX_BINDINGS_PER_GROUP,
    )?;
    let mut entries = Vec::with_capacity(bindings.len());
    let mut types = BTreeMap::new();
    let mut original_stages = BTreeMap::new();
    for binding in &bindings {
        if binding.descriptor_count != 1
            || !binding.p_immutable_samplers.is_null()
            || types
                .insert(binding.binding, binding.descriptor_type)
                .is_some()
        {
            if std::env::var_os("SGFX_VULKAN_TRACE").is_some() {
                eprintln!(
                    "[SGFX Vulkan] descriptor layout binding={} count={} type={:?} stages={:?}",
                    binding.binding,
                    binding.descriptor_count,
                    binding.descriptor_type,
                    binding.stage_flags
                );
            }
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        if i.flags
            .contains(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
            && matches!(
                binding.descriptor_type,
                vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                    | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
            )
        {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        original_stages.insert(binding.binding, binding.stage_flags);
    }
    let binding_map = descriptor_binding_map(&types)?;
    for binding in bindings {
        let ty = match binding.descriptor_type {
            vk::DescriptorType::UNIFORM_BUFFER | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC => {
                ir::BindingType::UniformBuffer
            }
            vk::DescriptorType::UNIFORM_TEXEL_BUFFER => {
                ir::BindingType::StorageBuffer { read_only: true }
            }
            vk::DescriptorType::STORAGE_BUFFER | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC => {
                ir::BindingType::StorageBuffer { read_only: false }
            }
            vk::DescriptorType::SAMPLED_IMAGE
            | vk::DescriptorType::INPUT_ATTACHMENT
            | vk::DescriptorType::COMBINED_IMAGE_SAMPLER => ir::BindingType::SampledTexture,
            vk::DescriptorType::SAMPLER => ir::BindingType::Sampler,
            vk::DescriptorType::STORAGE_IMAGE => ir::BindingType::StorageTexture {
                format: ir::TextureFormat::Rgba8Unorm,
                access: ir::StorageTextureAccess::WriteOnly,
            },
            _ => return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT),
        };
        let supported = binding.stage_flags
            & (vk::ShaderStageFlags::VERTEX
                | vk::ShaderStageFlags::FRAGMENT
                | vk::ShaderStageFlags::COMPUTE);
        let visibility = if binding.stage_flags == vk::ShaderStageFlags::ALL
            || binding.stage_flags == vk::ShaderStageFlags::ALL_GRAPHICS
        {
            stages(binding.stage_flags)?
        } else if supported.is_empty() {
            continue;
        } else {
            stages(supported)?
        };
        entries.push(ir::BindGroupLayoutEntry::new(
            binding_map[&binding.binding],
            visibility,
            ty,
        ));
        if binding.descriptor_type == vk::DescriptorType::COMBINED_IMAGE_SAMPLER {
            entries.push(ir::BindGroupLayoutEntry::new(
                binding_map[&binding.binding] + 1,
                visibility,
                ir::BindingType::Sampler,
            ));
        }
    }
    let desc = ir::BindGroupLayoutDesc::new(entries).map_err(failure)?;
    Ok((desc, types, original_stages))
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
        let pushed = (*info)
            .flags
            .contains(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR);
        let (desc, types, original_stages) = descriptor_layout(&*info).inspect_err(|error| {
            let i = &*info;
            eprintln!("[SGFX Vulkan] descriptor layout rejected: {error:?} flags={:?} pnext={:?} bindings={:?}", i.flags, i.p_next, copied(i.p_bindings, i.binding_count as usize, ir::MAX_BINDINGS_PER_GROUP));
        })?;
        let handle = crate::api::with_device(device, move |r| {
            if r.resources.set_layouts.len() >= LIMIT {
                return Err(vk::Result::ERROR_TOO_MANY_OBJECTS);
            }
            let handle = vk::DescriptorSetLayout::from_raw(crate::api::next_id());
            if pushed {
                r.resources.push_set_layouts.insert(handle);
            }
            r.resources.set_layouts.insert(handle, desc);
            r.resources.set_layout_types.insert(handle, types);
            r.resources
                .set_layout_stages
                .insert(handle, original_stages);
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
        r.resources.push_set_layouts.remove(&handle);
        r.resources.set_layouts.remove(&handle);
        r.resources.set_layout_types.remove(&handle);
        r.resources.set_layout_stages.remove(&handle);
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
        if !i.p_next.is_null() || !i.flags.is_empty() {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let handles = copied(
            i.p_set_layouts,
            i.set_layout_count as usize,
            ir::MAX_BIND_GROUPS,
        )?;
        let push_ranges = copied(
            i.p_push_constant_ranges,
            i.push_constant_range_count as usize,
            3,
        )?
        .into_iter()
        .map(|range| {
            ir::PushConstantRange::new(stages(range.stage_flags)?, range.offset, range.size)
                .map_err(failure)
        })
        .collect::<Result<Vec<_>, vk::Result>>()?;
        let (handle, metadata) = crate::api::with_device(device, move |r| {
            if r.resources.pipeline_layouts.len() >= LIMIT {
                return Err(vk::Result::ERROR_TOO_MANY_OBJECTS);
            }
            let binding_maps = handles
                .iter()
                .map(|h| {
                    descriptor_binding_map(
                        r.resources
                            .set_layout_types
                            .get(h)
                            .ok_or(vk::Result::ERROR_UNKNOWN)?,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let binding_stages = handles
                .iter()
                .map(|h| {
                    r.resources
                        .set_layout_stages
                        .get(h)
                        .cloned()
                        .ok_or(vk::Result::ERROR_UNKNOWN)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut pushed = None;
            for (index, h) in handles.iter().enumerate() {
                if r.resources.push_set_layouts.contains(h) {
                    if pushed.is_some() {
                        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                    }
                    pushed = Some((
                        index as u32,
                        DescriptorSet {
                            texel_formats: HashMap::new(),
                            pool: vk::DescriptorPool::null(),
                            layout: r.resources.set_layouts[h].clone(),
                            types: r.resources.set_layout_types[h].clone(),
                            stages: r.resources.set_layout_stages[h].clone(),
                            bindings: HashMap::new(),
                            cached_group: None,
                            invalid: false,
                        },
                    ));
                }
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
            if push_ranges.iter().any(|range| {
                range.offset() + range.size() > r.capabilities.limits().max_push_constants_size
            }) {
                return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
            }
            let desc = ir::PipelineLayoutDesc::new(groups)
                .map_err(failure)?
                .with_push_constant_ranges(push_ranges)
                .map_err(failure)?;
            let handle = vk::PipelineLayout::from_raw(crate::api::next_id());
            if let Some(pushed) = pushed {
                r.resources.pipeline_push_sets.insert(handle, pushed);
            }
            r.resources.pipeline_layouts.insert(handle, desc.clone());
            r.resources
                .pipeline_binding_maps
                .insert(handle, binding_maps);
            r.resources
                .pipeline_binding_stages
                .insert(handle, binding_stages);
            Ok((handle, desc))
        })?;
        crate::api::set_pipeline_layout_metadata(device, handle, Some(metadata))?;
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
        r.resources.pipeline_push_sets.remove(&handle);
        r.resources.pipeline_layouts.remove(&handle);
        r.resources.pipeline_binding_maps.remove(&handle);
        r.resources.pipeline_binding_stages.remove(&handle);
        Ok(())
    });
    let _ = crate::api::set_pipeline_layout_metadata(device, handle, None);
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
            // Declared pool capacity does not materialize any SGFX bind groups.
            // The separate live-object and canonical-table budgets apply when
            // sets are allocated and descriptor configurations are submitted.
            || i.max_sets > 16384
        {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let mut capacity = HashMap::<vk::DescriptorType, u32>::new();
        for size in copied(i.p_pool_sizes, i.pool_size_count as usize, 32)? {
            if !supported_descriptor_type(size.ty) || size.descriptor_count == 0 {
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
        let layouts = copied(
            i.p_set_layouts,
            i.descriptor_set_count as usize,
            MAX_DESCRIPTOR_SETS,
        )?;
        for index in 0..layouts.len() {
            out.add(index).write(vk::DescriptorSet::null());
        }
        let pool_handle = i.descriptor_pool;
        let handles = crate::api::with_device(device, move |r| {
            if r.resources
                .descriptor_sets
                .len()
                .checked_add(layouts.len())
                .is_none_or(|n| n > MAX_DESCRIPTOR_SETS)
            {
                return Err(vk::Result::ERROR_TOO_MANY_OBJECTS);
            }
            if layouts
                .iter()
                .any(|h| r.resources.push_set_layouts.contains(h))
            {
                return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
            }
            let layouts = layouts
                .into_iter()
                .map(|h| {
                    Ok((
                        r.resources
                            .set_layouts
                            .get(&h)
                            .cloned()
                            .ok_or(vk::Result::ERROR_UNKNOWN)?,
                        r.resources
                            .set_layout_types
                            .get(&h)
                            .cloned()
                            .ok_or(vk::Result::ERROR_UNKNOWN)?,
                        r.resources
                            .set_layout_stages
                            .get(&h)
                            .cloned()
                            .ok_or(vk::Result::ERROR_UNKNOWN)?,
                    ))
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
            for (_, types, _) in &layouts {
                for ty in types.values() {
                    let slot = remaining.entry(*ty).or_default();
                    *slot = slot
                        .checked_sub(1)
                        .ok_or(vk::Result::ERROR_OUT_OF_POOL_MEMORY)?;
                }
            }
            pool.remaining = remaining;
            let mut result = Vec::with_capacity(layouts.len());
            for (layout, types, stages) in layouts {
                let handle = vk::DescriptorSet::from_raw(crate::api::next_id());
                r.resources.descriptor_sets.insert(
                    handle,
                    DescriptorSet {
                        texel_formats: HashMap::new(),
                        pool: pool_handle,
                        layout,
                        types,
                        stages,
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
                    for ty in set.types.values() {
                        *pool.remaining.entry(*ty).or_default() += 1;
                    }
                }
            }
            Ok(())
        })
    })
}
#[derive(Clone)]
pub(crate) struct Write {
    pub(crate) set: vk::DescriptorSet,
    pub(crate) binding: u32,
    pub(crate) element: u32,
    pub(crate) ty: vk::DescriptorType,
    pub(crate) value: Option<DescriptorBinding>,
}
// This frontend's descriptors each have count one. Vulkan updates may still
// span consecutive compatible bindings, skipping absent (zero-count) slots.
fn descriptor_write_binding(
    set: &DescriptorSet,
    first: u32,
    element: u32,
    ty: vk::DescriptorType,
) -> Option<u32> {
    if set.types.get(&first) != Some(&ty) {
        return None;
    }
    // Vulkan visibility remains relevant to consecutive-binding updates even
    // when an unsupported (unused) stage has no lowered SGFX entry.
    let visibility = set.stages.get(&first)?;
    let mut target = None;
    for (&binding, actual) in set.types.range(first..).take(element as usize + 1) {
        if *actual != ty || set.stages.get(&binding)? != visibility {
            return None;
        }
        target = Some(binding);
    }
    if set.types.range(first..).count() <= element as usize {
        None
    } else {
        target
    }
}

pub(crate) unsafe fn snapshot_descriptor_writes(
    writes: &[vk::WriteDescriptorSet<'_>],
) -> Vec<Write> {
    let mut owned_writes = Vec::with_capacity(writes.len());
    for write in writes {
        let valid = write.s_type == vk::StructureType::WRITE_DESCRIPTOR_SET
            && write.p_next.is_null()
            && (1..=16).contains(&write.descriptor_count)
            && write.dst_array_element == 0;
        for element in 0..if valid { write.descriptor_count } else { 1 } {
            let value = if !valid {
                None
            } else {
                match write.descriptor_type {
                    vk::DescriptorType::UNIFORM_TEXEL_BUFFER
                        if !write.p_texel_buffer_view.is_null() =>
                    {
                        Some(DescriptorBinding::TexelBuffer(
                            *write.p_texel_buffer_view.add(element as usize),
                        ))
                    }

                    vk::DescriptorType::UNIFORM_BUFFER
                    | vk::DescriptorType::STORAGE_BUFFER
                    | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                    | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
                        if !write.p_buffer_info.is_null() =>
                    {
                        let b = *write.p_buffer_info.add(element as usize);
                        Some(DescriptorBinding::Buffer {
                            buffer: b.buffer,
                            offset: b.offset,
                            range: b.range,
                        })
                    }
                    vk::DescriptorType::SAMPLED_IMAGE
                    | vk::DescriptorType::INPUT_ATTACHMENT
                    | vk::DescriptorType::STORAGE_IMAGE
                    | vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                        if !write.p_image_info.is_null() =>
                    {
                        let i = *write.p_image_info.add(element as usize);
                        Some(DescriptorBinding::Image {
                            view: i.image_view,
                            sampler: i.sampler,
                            layout: i.image_layout,
                        })
                    }
                    vk::DescriptorType::SAMPLER if !write.p_image_info.is_null() => {
                        Some(DescriptorBinding::Sampler(
                            (*write.p_image_info.add(element as usize)).sampler,
                        ))
                    }
                    _ => None,
                }
            };
            owned_writes.push(Write {
                set: write.dst_set,
                binding: write.dst_binding,
                element,
                ty: write.descriptor_type,
                value,
            });
        }
    }
    owned_writes
}

pub(crate) fn apply_descriptor_writes(r: &mut crate::runtime::Runtime, owned_writes: Vec<Write>) {
    for mut write in owned_writes {
        let texel_format = if let Some(DescriptorBinding::TexelBuffer(view)) = write.value {
            r.resources.buffer_views.get(&view).map(|v| v.format)
        } else {
            None
        };
        if let Some(DescriptorBinding::TexelBuffer(view)) = write.value {
            write.value = r
                .resources
                .buffer_views
                .get(&view)
                .filter(|v| {
                    matches!(
                        v.format,
                        vk::Format::R8_SINT
                            | vk::Format::R32_SINT
                            | vk::Format::R32_UINT
                            | vk::Format::R32_SFLOAT
                    )
                })
                .map(|v| DescriptorBinding::Buffer {
                    buffer: v.buffer,
                    offset: v.offset,
                    range: v.range,
                });
        }
        let valid = write.value.is_some_and(|v| match v {
            DescriptorBinding::TexelBuffer(_) => false,
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
                        <= if matches!(
                            write.ty,
                            vk::DescriptorType::UNIFORM_BUFFER
                                | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                        ) {
                            16 * 1024
                        } else {
                            128 * 1024 * 1024
                        }
                    && (texel_format == Some(vk::Format::R8_SINT) || size.is_multiple_of(4))
                    && offset.is_multiple_of(u64::from(
                        if matches!(
                            write.ty,
                            vk::DescriptorType::UNIFORM_BUFFER
                                | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                        ) {
                            r.capabilities
                                .limits()
                                .min_uniform_buffer_offset_alignment
                                .max(16)
                        } else {
                            r.capabilities
                                .limits()
                                .min_storage_buffer_offset_alignment
                                .max(4)
                        },
                    ))
                    && offset.checked_add(size).is_some_and(|end| end <= b.size)
                    && b.usage.contains(
                        if matches!(
                            write.ty,
                            vk::DescriptorType::UNIFORM_BUFFER
                                | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                        ) {
                            vk::BufferUsageFlags::UNIFORM_BUFFER
                        } else if write.ty == vk::DescriptorType::UNIFORM_TEXEL_BUFFER {
                            vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER
                        } else {
                            vk::BufferUsageFlags::STORAGE_BUFFER
                        },
                    )
            }),
            DescriptorBinding::Sampler(sampler) => r.resources.samplers.contains_key(&sampler),
            DescriptorBinding::Image {
                view,
                sampler,
                layout,
            } => {
                matches!(
                    layout,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                        | vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
                        | vk::ImageLayout::GENERAL
                ) && r
                    .resources
                    .views
                    .get(&view)
                    .and_then(|view| r.resources.images.get(&view.image))
                    .is_some_and(|image| {
                        image
                            .usage
                            .contains(if write.ty == vk::DescriptorType::STORAGE_IMAGE {
                                vk::ImageUsageFlags::STORAGE
                            } else if write.ty == vk::DescriptorType::INPUT_ATTACHMENT {
                                vk::ImageUsageFlags::INPUT_ATTACHMENT
                            } else {
                                vk::ImageUsageFlags::SAMPLED
                            })
                    })
                    && (write.ty != vk::DescriptorType::STORAGE_IMAGE
                        || (layout == vk::ImageLayout::GENERAL
                            && r.resources
                                .views
                                .get(&view)
                                .is_some_and(|view| view.components == [0, 1, 2, 3])))
                    && (write.ty != vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                        || r.resources.samplers.contains_key(&sampler))
            }
        });
        if let Some(set) = r.resources.descriptor_sets.get_mut(&write.set) {
            let binding = descriptor_write_binding(set, write.binding, write.element, write.ty);
            set.cached_group = None;
            if let (true, Some(binding), Some(value)) = (valid, binding, write.value) {
                set.bindings.insert(binding, value);
                if let Some(format) = texel_format {
                    set.texel_formats.insert(binding, format);
                } else {
                    set.texel_formats.remove(&binding);
                }
            } else {
                if std::env::var_os("SGFX_VULKAN_TRACE").is_some() {
                    eprintln!(
                        "[SGFX Vulkan] invalid descriptor write: set={:?} binding={} element={} type={:?} value={:?} valid_resource={valid} resolved_binding={binding:?}",
                        write.set, write.binding, write.element, write.ty, write.value
                    );
                }
                set.invalid = true;
            }
        }
    }
}

pub(crate) unsafe extern "system" fn update_descriptor_sets(
    device: vk::Device,
    write_count: u32,
    writes: *const vk::WriteDescriptorSet<'_>,
    copy_count: u32,
    copies: *const vk::CopyDescriptorSet<'_>,
) {
    let result = ffi(|| {
        let writes = copied(writes, write_count as usize, LIMIT)?;
        let copies = copied(copies, copy_count as usize, LIMIT)?;
        let owned_writes = snapshot_descriptor_writes(&writes);
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
        let modified_sets = owned_writes
            .iter()
            .map(|w| w.set)
            .chain(owned_copies.iter().map(|c| c.2))
            .collect::<Vec<_>>();
        crate::api::update_descriptors_owned(device, &modified_sets, move |r| {
            apply_descriptor_writes(r, owned_writes);
            for (src, src_binding, dst, dst_binding, valid) in owned_copies {
                let source = r
                    .resources
                    .descriptor_sets
                    .get(&src)
                    .filter(|s| !s.invalid)
                    .and_then(|s| {
                        s.types
                            .get(&src_binding)
                            .copied()
                            .zip(s.bindings.get(&src_binding).copied())
                    });
                let texel_format = r
                    .resources
                    .descriptor_sets
                    .get(&src)
                    .and_then(|s| s.texel_formats.get(&src_binding))
                    .copied();
                if let Some(set) = r.resources.descriptor_sets.get_mut(&dst) {
                    set.cached_group = None;
                    if valid
                        && source.is_some_and(|(ty, _)| set.types.get(&dst_binding) == Some(&ty))
                    {
                        if let Some((_, value)) = source {
                            set.bindings.insert(dst_binding, value);
                            if let Some(format) = texel_format {
                                set.texel_formats.insert(dst_binding, format);
                            } else {
                                set.texel_formats.remove(&dst_binding);
                            }
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
        if std::env::var_os("SGFX_VULKAN_TRACE").is_some() {
            eprintln!(
                "[SGFX Vulkan] descriptor update batch failed; marking device lost: writes={write_count} copies={copy_count} result={result:?}"
            );
        }
        crate::api::mark_device_lost(device);
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
        if !allocator.is_null() {
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
            requests.push((
                stage.module,
                name.to_owned(),
                info.layout,
                specialization(stage.p_specialization_info)?,
            ));
        }
        let handles = crate::api::with_device(device, move |r| {
            crate::api::bootstrap::validate_cache(&r.resources, cache)?;
            let checkpoints =
                shader_variant_checkpoints(&r.resources, requests.iter().map(|request| request.0));
            let result = (|| {
                if r.resources
                    .pipelines
                    .len()
                    .checked_add(requests.len())
                    .is_none_or(|n| n > 256)
                {
                    return Err(vk::Result::ERROR_TOO_MANY_OBJECTS);
                }
                let mut descriptors = Vec::with_capacity(requests.len());
                for (module, name, layout, values) in requests {
                    let bindings = r
                        .resources
                        .pipeline_binding_maps
                        .get(&layout)
                        .cloned()
                        .ok_or(vk::Result::ERROR_UNKNOWN)?;
                    let module = r
                        .resources
                        .shaders
                        .get_mut(&module)
                        .ok_or(vk::Result::ERROR_UNKNOWN)?
                        .variant_for_layout(&r.table, &values, &[], &bindings)?;
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
                    let layout = crate::spirv::specialize_layout(&r.table, &layout, &[&shader])?;
                    descriptors
                        .push(ir::ComputePipelineDesc::new(shader, layout).map_err(failure)?);
                }
                let mut ids = Vec::with_capacity(descriptors.len());
                for desc in descriptors {
                    let created = (|| {
                        let id = r.table.define_compute_pipeline(desc).map_err(failure)?.id();
                        if let Err(error) = r.cache.validate_compute_pipeline(id) {
                            let error = backend_failure(error);
                            retire_ir_pipeline(r, Pipeline::Compute(id))?;
                            return Err(error);
                        }
                        Ok(id)
                    })();
                    match created {
                        Ok(id) => ids.push(id),
                        Err(error) => {
                            // This frontend publishes compute handles only once the
                            // whole batch succeeds. Retire every unpublished prefix.
                            for id in ids {
                                retire_ir_pipeline(r, Pipeline::Compute(id))?;
                            }
                            return Err(error);
                        }
                    }
                }
                let mut result = Vec::with_capacity(ids.len());
                for id in ids {
                    let handle = vk::Pipeline::from_raw(crate::api::next_id());
                    r.resources.pipelines.insert(handle, Pipeline::Compute(id));
                    result.push(handle);
                }
                Ok(result)
            })();
            if result.is_err() {
                rollback_shader_variants(r, &checkpoints)?;
            }
            result
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
    let _ = crate::api::with_device(device, move |r| retire_vk_pipeline(r, handle));
}

fn retire_vk_pipeline(
    r: &mut crate::runtime::Runtime,
    handle: vk::Pipeline,
) -> Result<(), vk::Result> {
    let mut pipelines = r
        .resources
        .pipelines
        .get(&handle)
        .copied()
        .into_iter()
        .collect::<Vec<_>>();
    pipelines.extend(
        r.resources
            .graphics_swizzles
            .iter()
            .filter_map(|((owner, _), id)| (*owner == handle).then_some(Pipeline::Graphics(*id))),
    );
    pipelines.extend(
        r.resources
            .compute_swizzles
            .iter()
            .filter_map(|((owner, _), id)| (*owner == handle).then_some(Pipeline::Compute(*id))),
    );
    if !pipelines.is_empty() {
        for pipeline in pipelines {
            retire_ir_pipeline(r, pipeline)?;
        }
    }
    r.resources.pipelines.remove(&handle);
    r.resources
        .graphics_swizzles
        .retain(|(pipeline, _), _| *pipeline != handle);
    r.resources
        .compute_swizzles
        .retain(|(pipeline, _), _| *pipeline != handle);
    r.resources.graphics_extents.remove(&handle);
    r.resources.graphics_state.remove(&handle);
    r.resources.graphics_subpasses.remove(&handle);
    r.resources.graphics_inputs.remove(&handle);
    retire_unused_shader_modules(r)
}

/// Retire one logical pipeline only after its backend materialization. Used for
/// normal destruction and for unpublished pipelines whose validation failed.
pub(crate) fn retire_ir_pipeline(
    r: &mut crate::runtime::Runtime,
    pipeline: Pipeline,
) -> Result<(), vk::Result> {
    // Metadata synchronization also imports previous retirement tombstones.
    // Its nonblocking ABI refuses them while accepted GPU work is pending.
    // Do not turn an ordinary failed pipeline creation into device loss merely
    // because unrelated accepted work is still running. Never wait on pending
    // device requests from this worker itself.
    r.in_flight.wait();
    let result = (|| match pipeline {
        Pipeline::Graphics(id) => {
            r.cache
                .release_programmable_render_pipeline(id)
                .map_err(backend_failure)?;
            r.table
                .release_programmable_render_pipeline(id)
                .map_err(failure)
        }
        Pipeline::Compute(id) => {
            r.cache
                .release_compute_pipeline(id)
                .map_err(backend_failure)?;
            r.table.release_compute_pipeline(id).map_err(failure)
        }
    })();
    if result.is_err() {
        r.lost = true;
    }
    result
}

fn shader_variant_checkpoints(
    resources: &Resources,
    modules: impl IntoIterator<Item = vk::ShaderModule>,
) -> HashMap<vk::ShaderModule, usize> {
    modules
        .into_iter()
        .filter_map(|module| {
            resources
                .shaders
                .get(&module)
                .map(|shader| (module, shader.variants.len()))
        })
        .collect()
}

pub(crate) fn rollback_shader_variants(
    r: &mut crate::runtime::Runtime,
    checkpoints: &HashMap<vk::ShaderModule, usize>,
) -> Result<(), vk::Result> {
    // An unsuccessful attempt must not pin new IR modules in its live Vulkan
    // shader's specialization cache. Existing/shared variants precede the mark.
    for (module, mark) in checkpoints {
        if let Some(shader) = r.resources.shaders.get_mut(module) {
            r.resources
                .orphaned_shader_modules
                .extend(shader.variants.drain(*mark..).map(|variant| variant.3));
        }
    }
    if r.lost {
        return Err(vk::Result::ERROR_DEVICE_LOST);
    }
    retire_unused_shader_modules(r)
}

pub(crate) fn shader_checkpoints(
    resources: &Resources,
    modules: &[vk::ShaderModule],
) -> HashMap<vk::ShaderModule, usize> {
    shader_variant_checkpoints(resources, modules.iter().copied())
}

fn retire_unused_shader_modules(r: &mut crate::runtime::Runtime) -> Result<(), vk::Result> {
    let result = (|| {
        let mut retained = HashSet::new();
        for pipeline in r
            .resources
            .pipelines
            .values()
            .copied()
            .chain(
                r.resources
                    .graphics_swizzles
                    .values()
                    .copied()
                    .map(Pipeline::Graphics),
            )
            .chain(
                r.resources
                    .compute_swizzles
                    .values()
                    .copied()
                    .map(Pipeline::Compute),
            )
        {
            match pipeline {
                Pipeline::Graphics(id) => {
                    let desc = r
                        .table
                        .programmable_render_pipeline_shared(
                            r.table
                                .programmable_render_pipeline_ref(id)
                                .map_err(failure)?,
                        )
                        .map_err(failure)?;
                    retained.insert(desc.vertex().module());
                    retained.insert(desc.fragment().module());
                }
                Pipeline::Compute(id) => {
                    let desc = r
                        .table
                        .compute_pipeline(r.table.compute_pipeline_ref(id).map_err(failure)?)
                        .map_err(failure)?;
                    retained.insert(desc.shader().module());
                }
            }
        }
        // A Vulkan shader module owns source, not every specialization ever
        // requested. Drop cache entries once no live pipeline uses them; a later
        // pipeline can regenerate them from the original words.
        for shader in r.resources.shaders.values_mut() {
            shader.variants.retain(|variant| {
                if retained.contains(&variant.3) {
                    true
                } else {
                    r.resources.orphaned_shader_modules.insert(variant.3);
                    false
                }
            });
        }
        let retired = r
            .resources
            .orphaned_shader_modules
            .iter()
            .copied()
            .filter(|id| !retained.contains(id))
            .collect::<Vec<_>>();
        if !retired.is_empty() {
            r.in_flight.wait();
        }
        for id in retired {
            r.cache.release_shader_module(id).map_err(backend_failure)?;
            r.table.release_shader_module(id).map_err(failure)?;
            r.resources.orphaned_shader_modules.remove(&id);
        }
        Ok(())
    })();
    if result.is_err() {
        r.lost = true;
    }
    result
}

fn sampler_descriptor(
    i: &vk::SamplerCreateInfo<'_>,
    custom_allocator: bool,
) -> Result<ir::SamplerDesc, vk::Result> {
    if custom_allocator
        || i.s_type != vk::StructureType::SAMPLER_CREATE_INFO
        || !i.p_next.is_null()
        || !i.flags.is_empty()
        || !matches!(i.anisotropy_enable, vk::FALSE | vk::TRUE)
        || (i.anisotropy_enable == vk::TRUE
            && (!i.max_anisotropy.is_finite() || !(1.0..=16.0).contains(&i.max_anisotropy)))
        || i.unnormalized_coordinates != vk::FALSE
        || !matches!(i.compare_enable, vk::FALSE | vk::TRUE)
        || i.mip_lod_bias != 0.0
        || !i.max_lod.is_finite()
        || !i.min_lod.is_finite()
        || i.max_lod < i.min_lod
        || !matches!(
            i.mipmap_mode,
            vk::SamplerMipmapMode::NEAREST | vk::SamplerMipmapMode::LINEAR
        )
    {
        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
    }
    let border = [i.address_mode_u, i.address_mode_v, i.address_mode_w]
        .contains(&vk::SamplerAddressMode::CLAMP_TO_BORDER);
    if border
        && !matches!(
            i.border_color,
            vk::BorderColor::FLOAT_TRANSPARENT_BLACK | vk::BorderColor::INT_TRANSPARENT_BLACK
        )
    {
        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
    }
    let filter = |v| match v {
        vk::Filter::NEAREST => Ok(ir::FilterMode::Nearest),
        vk::Filter::LINEAR => Ok(ir::FilterMode::Linear),
        _ => Err(vk::Result::ERROR_FEATURE_NOT_PRESENT),
    };
    let address = |v| match v {
        vk::SamplerAddressMode::REPEAT => Ok(ir::AddressMode::Repeat),
        vk::SamplerAddressMode::CLAMP_TO_EDGE => Ok(ir::AddressMode::ClampToEdge),
        vk::SamplerAddressMode::MIRRORED_REPEAT => Ok(ir::AddressMode::MirrorRepeat),
        vk::SamplerAddressMode::CLAMP_TO_BORDER => Ok(ir::AddressMode::ClampToBorder),
        _ => Err(vk::Result::ERROR_FEATURE_NOT_PRESENT),
    };
    // SGFX supports 1D/2D, array and cube textures; no 3D textures. W is
    // unused for those images, but reject modes outside the enabled contract.
    address(i.address_mode_w)?;
    Ok(ir::SamplerDesc::new(
        filter(i.min_filter)?,
        filter(i.mag_filter)?,
        address(i.address_mode_u)?,
        address(i.address_mode_v)?,
    )
    .with_mip_filter(
        match i.mipmap_mode {
            vk::SamplerMipmapMode::LINEAR => ir::FilterMode::Linear,
            _ => ir::FilterMode::Nearest,
        },
        i.min_lod,
        i.max_lod,
    )
    .map_err(failure)?
    .with_anisotropy(if i.anisotropy_enable == vk::TRUE {
        i.max_anisotropy.ceil() as u8
    } else {
        1
    })
    .map_err(failure)?
    .with_compare(if i.compare_enable != vk::FALSE {
        Some(crate::images::compare(i.compare_op)?)
    } else {
        None
    }))
}

unsafe extern "system" fn create_sampler(
    device: vk::Device,
    info: *const vk::SamplerCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::Sampler,
) -> vk::Result {
    let result = ffi(|| {
        if out.is_null() {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        out.write(vk::Sampler::null());
        let i = info
            .as_ref()
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let desc = sampler_descriptor(i, !allocator.is_null())?;
        let handle = crate::api::with_device(device, move |r| {
            if desc.anisotropy() > 1 && !r.capabilities.supports_anisotropy_16() {
                return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
            }
            let id = r.table.define_sampler(desc).map_err(failure)?.id();
            let handle = vk::Sampler::from_raw(crate::api::next_id());
            r.resources.samplers.insert(handle, id);
            Ok(handle)
        })?;
        out.write(handle);
        Ok(())
    });
    if result != vk::Result::SUCCESS {
        eprintln!(
            "[SGFX Vulkan] CreateSampler failure={result:?} custom_allocator={} info={:?}",
            !allocator.is_null(),
            info.as_ref(),
        );
    }
    result
}
unsafe extern "system" fn destroy_sampler(
    device: vk::Device,
    sampler: vk::Sampler,
    _allocator: *const vk::AllocationCallbacks<'_>,
) {
    let _ = crate::api::with_device(device, move |r| {
        r.resources.samplers.remove(&sampler);
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
        b"vkCreateSampler" => entry!(create_sampler, vk::PFN_vkCreateSampler),
        b"vkDestroySampler" => entry!(destroy_sampler, vk::PFN_vkDestroySampler),
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

    #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
    fn lifecycle_runtime() -> crate::runtime::Runtime {
        let instance = sgfx::driver::Instance::new().unwrap();
        crate::runtime::Runtime::new(
            instance.adapters()[0].clone(),
            Default::default(),
            Default::default(),
            Default::default(),
        )
        .unwrap()
    }

    #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
    fn compute_module(table: &ir::ResourceTable) -> ir::ShaderModuleId {
        table
            .define_shader_module(
                ir::ShaderModuleDesc::wgsl("@compute @workgroup_size(1) fn main() {}".to_owned())
                    .unwrap(),
            )
            .unwrap()
            .id()
    }

    #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
    #[test]
    #[ignore = "requires a native GPU to construct the actual Runtime"]
    fn long_lived_shader_evicts_unused_variants_across_more_than_256_pipelines() {
        let mut runtime = lifecycle_runtime();
        let module = vk::ShaderModule::from_raw(1);
        runtime.resources.shaders.insert(
            module,
            Shader {
                words: Vec::new(),
                variants: Vec::new(),
            },
        );
        for iteration in 0u32..1024 {
            let id = compute_module(&runtime.table);
            runtime
                .resources
                .shaders
                .get_mut(&module)
                .unwrap()
                .variants
                .push((
                    vec![(0, iteration.to_le_bytes().to_vec())],
                    Vec::new(),
                    Vec::new(),
                    id,
                ));
            let shader = ir::ShaderEntryPoint::new(
                runtime.table.shader_module_ref(id).unwrap(),
                ir::ShaderStage::Compute,
                "main".to_owned(),
            )
            .unwrap();
            let pipeline = runtime
                .table
                .define_compute_pipeline(
                    ir::ComputePipelineDesc::new(
                        shader,
                        ir::PipelineLayoutDesc::new(Vec::new()).unwrap(),
                    )
                    .unwrap(),
                )
                .unwrap()
                .id();
            let handle = vk::Pipeline::from_raw(2);
            runtime
                .resources
                .pipelines
                .insert(handle, Pipeline::Compute(pipeline));
            retire_vk_pipeline(&mut runtime, handle).unwrap();
            assert!(
                runtime.table.shader_module_ref(id).is_err(),
                "iteration {iteration}"
            );
            assert!(runtime.table.compute_pipeline_ref(pipeline).is_err());
            assert!(runtime.resources.shaders[&module].variants.is_empty());
            assert!(runtime.resources.orphaned_shader_modules.is_empty());
            assert!(!runtime.lost);
        }
    }

    #[cfg(all(not(target_os = "scarlet"), feature = "backend-wgpu"))]
    #[test]
    #[ignore = "requires a native GPU to construct the actual Runtime"]
    fn failed_shader_attempt_evicts_new_variant_but_preserves_shared_live_pipeline() {
        let mut runtime = lifecycle_runtime();
        let module = vk::ShaderModule::from_raw(1);
        let existing = compute_module(&runtime.table);
        runtime.resources.shaders.insert(
            module,
            Shader {
                words: Vec::new(),
                variants: vec![(Vec::new(), Vec::new(), Vec::new(), existing)],
            },
        );
        let shader = ir::ShaderEntryPoint::new(
            runtime.table.shader_module_ref(existing).unwrap(),
            ir::ShaderStage::Compute,
            "main".to_owned(),
        )
        .unwrap();
        let pipeline = runtime
            .table
            .define_compute_pipeline(
                ir::ComputePipelineDesc::new(
                    shader,
                    ir::PipelineLayoutDesc::new(Vec::new()).unwrap(),
                )
                .unwrap(),
            )
            .unwrap()
            .id();
        let handle = vk::Pipeline::from_raw(2);
        runtime
            .resources
            .pipelines
            .insert(handle, Pipeline::Compute(pipeline));
        let checkpoints = shader_checkpoints(&runtime.resources, &[module]);
        let failed = compute_module(&runtime.table);
        runtime
            .resources
            .shaders
            .get_mut(&module)
            .unwrap()
            .variants
            .push((vec![(0, vec![1, 0, 0, 0])], Vec::new(), Vec::new(), failed));
        rollback_shader_variants(&mut runtime, &checkpoints).unwrap();
        assert!(runtime.table.shader_module_ref(failed).is_err());
        assert_eq!(runtime.resources.shaders[&module].variants.len(), 1);
        assert!(runtime.table.shader_module_ref(existing).is_ok());
        retire_vk_shader_module(&mut runtime, module).unwrap();
        assert!(runtime.table.shader_module_ref(existing).is_ok());
        assert!(
            runtime
                .resources
                .orphaned_shader_modules
                .contains(&existing)
        );
        retire_vk_pipeline(&mut runtime, handle).unwrap();
        assert!(runtime.table.shader_module_ref(existing).is_err());
        assert!(runtime.resources.orphaned_shader_modules.is_empty());
        assert!(!runtime.lost);
    }

    fn decal_sampler() -> vk::SamplerCreateInfo<'static> {
        vk::SamplerCreateInfo::default()
            .min_filter(vk::Filter::LINEAR)
            .mag_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
            .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK)
            .min_lod(0.0)
            .max_lod(1000.0)
    }

    #[test]
    fn skia_decal_sampler_retains_transparent_border_and_lod() {
        let desc = sampler_descriptor(&decal_sampler(), false).unwrap();
        assert_eq!(desc.address_u(), ir::AddressMode::ClampToBorder);
        assert_eq!(desc.address_v(), ir::AddressMode::ClampToBorder);
        assert_eq!(desc.min_filter(), ir::FilterMode::Linear);
        assert_eq!(desc.mag_filter(), ir::FilterMode::Linear);
        assert_eq!(desc.max_lod(), 1000.0);
        assert!(
            sampler_descriptor(
                &decal_sampler().border_color(vk::BorderColor::INT_TRANSPARENT_BLACK),
                false
            )
            .is_ok()
        );
    }

    #[test]
    fn r32_texel_fetch_preserves_signed_bits_and_default_channels() {
        let words = include_bytes!("../tests/assets/texel-buffer.vert.spv")
            .chunks_exact(4)
            .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        let module = normalize_spirv(words).unwrap();
        let shader = sgfx_codegen_virgl::programmable::compile_shader(
            &module,
            ir::ShaderStage::Vertex,
            "main",
        )
        .unwrap();
        assert_eq!(shader.storage_buffers.len(), 1);
        assert!(shader.tgsi.contains("BUFFER"));
    }
    #[test]
    fn fragment_output_arrays_expand_to_distinct_attachment_locations() {
        let words = include_bytes!("../tests/assets/output-array.frag.spv")
            .chunks_exact(4)
            .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        let module = normalize_spirv(words).unwrap();
        let shader = sgfx_codegen_virgl::programmable::compile_shader(
            &module,
            ir::ShaderStage::Fragment,
            "main",
        )
        .unwrap();
        assert_eq!(shader.output_locations, [0, 1]);
    }
    #[test]
    fn point_size_and_vec4_varying_roundtrip() {
        let words = include_bytes!("../tests/assets/point-varying.vert.spv")
            .chunks_exact(4)
            .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        normalize_spirv(words).unwrap();
    }
    #[test]
    fn scalar_color_and_fragment_depth_roundtrip() {
        let words = include_bytes!("../tests/assets/scalar-depth.frag.spv")
            .chunks_exact(4)
            .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        normalize_spirv(words).unwrap();
    }
    #[test]
    fn shader_draw_parameters_and_flat_instance_varyings_normalize() {
        let words = include_bytes!("../tests/assets/requirements.vert.spv")
            .chunks_exact(4)
            .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        let module = normalize_spirv(words).unwrap();
        let shader = sgfx_codegen_virgl::programmable::compile_shader(
            &module,
            ir::ShaderStage::Vertex,
            "main",
        )
        .unwrap();
        assert!(shader.first_instance_register.is_some());
    }
    #[test]
    fn nonzero_border_and_unimplemented_sampler_controls_are_rejected() {
        for color in [
            vk::BorderColor::FLOAT_OPAQUE_BLACK,
            vk::BorderColor::INT_OPAQUE_BLACK,
            vk::BorderColor::FLOAT_OPAQUE_WHITE,
            vk::BorderColor::INT_OPAQUE_WHITE,
        ] {
            assert_eq!(
                sampler_descriptor(&decal_sampler().border_color(color), false),
                Err(vk::Result::ERROR_FEATURE_NOT_PRESENT)
            );
        }
        for info in [
            decal_sampler()
                .anisotropy_enable(true)
                .max_anisotropy(f32::NAN),
            decal_sampler().anisotropy_enable(true).max_anisotropy(0.0),
            decal_sampler().anisotropy_enable(true).max_anisotropy(17.0),
            decal_sampler().mip_lod_bias(0.5),
            decal_sampler().unnormalized_coordinates(true),
            decal_sampler().address_mode_w(vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE),
        ] {
            assert_eq!(
                sampler_descriptor(&info, false),
                Err(vk::Result::ERROR_FEATURE_NOT_PRESENT)
            );
        }
        // Border color is ignored by Vulkan when no axis clamps to the border.
        assert!(
            sampler_descriptor(
                &decal_sampler()
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE),
                false
            )
            .is_ok()
        );
    }

    #[test]
    fn sampler_failure_reports_info_and_clears_the_output_handle() {
        let mut out = vk::Sampler::from_raw(123);
        let result = unsafe {
            create_sampler(
                vk::Device::null(),
                &decal_sampler().border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE),
                ptr::null(),
                &mut out,
            )
        };
        assert_eq!(result, vk::Result::ERROR_FEATURE_NOT_PRESENT);
        assert_eq!(out, vk::Sampler::null());
    }

    #[test]
    fn sparse_descriptor_labels_use_distinct_bounded_slots() {
        let types = BTreeMap::from([
            (0, vk::DescriptorType::UNIFORM_BUFFER),
            (4, vk::DescriptorType::UNIFORM_BUFFER),
            (128, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
            (u32::MAX, vk::DescriptorType::STORAGE_BUFFER),
        ]);
        assert_eq!(
            descriptor_binding_map(&types).unwrap(),
            BTreeMap::from([(0, 0), (4, 8), (128, 2), (u32::MAX, 4),])
        );
        let full = (128..144)
            .map(|b| (b, vk::DescriptorType::UNIFORM_BUFFER))
            .collect();
        let map = descriptor_binding_map(&full).unwrap();
        assert_eq!(
            map.values().copied().collect::<Vec<_>>(),
            (0..32).step_by(2).collect::<Vec<_>>()
        );
        let excess = (128..145)
            .map(|b| (b, vk::DescriptorType::UNIFORM_BUFFER))
            .collect();
        assert_eq!(
            descriptor_binding_map(&excess),
            Err(vk::Result::ERROR_FEATURE_NOT_PRESENT)
        );
    }

    #[test]
    fn descriptor_writes_preserve_declared_stage_visibility() {
        let set = DescriptorSet {
            texel_formats: HashMap::new(),
            pool: vk::DescriptorPool::null(),
            layout: ir::BindGroupLayoutDesc::new(vec![]).unwrap(),
            types: [1, 3, 128]
                .into_iter()
                .map(|b| (b, vk::DescriptorType::UNIFORM_BUFFER))
                .collect(),
            stages: BTreeMap::from([
                (1, vk::ShaderStageFlags::GEOMETRY),
                (3, vk::ShaderStageFlags::GEOMETRY),
                (128, vk::ShaderStageFlags::FRAGMENT),
            ]),
            bindings: HashMap::new(),
            cached_group: None,
            invalid: false,
        };
        // Declared but unused stages still accept updates. Consecutive writes
        // may cross holes only while the ORIGINAL Vulkan visibility matches.
        assert_eq!(
            descriptor_write_binding(&set, 1, 0, vk::DescriptorType::UNIFORM_BUFFER),
            Some(1)
        );
        assert_eq!(
            descriptor_write_binding(&set, 1, 1, vk::DescriptorType::UNIFORM_BUFFER),
            Some(3)
        );
        assert_eq!(
            descriptor_write_binding(&set, 1, 2, vk::DescriptorType::UNIFORM_BUFFER),
            None
        );
        assert_eq!(
            descriptor_write_binding(&set, 2, 0, vk::DescriptorType::UNIFORM_BUFFER),
            None
        );
        assert_eq!(
            descriptor_write_binding(&set, 1, 0, vk::DescriptorType::STORAGE_BUFFER),
            None
        );
    }

    #[test]
    fn sparse_combined_sampler_shader_uses_layout_mapping_and_cache_key() {
        let module = naga::front::wgsl::parse_str(
            "@group(0) @binding(128) var image: texture_2d<f32>;\n\
             @group(0) @binding(129) var smp: sampler;\n\
             @fragment fn main() -> @location(0) vec4<f32> { return textureSample(image, smp, vec2<f32>(0.5)); }"
        ).unwrap();
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
        let mut words =
            naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
                .unwrap();
        // Represent the pair emitted by combined-sampler separation: both
        // globals retain one Vulkan binding before SGFX normalization.
        let mut cursor = 5;
        while cursor < words.len() {
            let count = (words[cursor] >> 16) as usize;
            if words[cursor] & 0xffff == 71
                && count == 4
                && words[cursor + 2] == 33
                && words[cursor + 3] == 129
            {
                words[cursor + 3] = 128;
            }
            cursor += count;
        }
        let table = ir::ResourceTable::new();
        let mut shader = Shader {
            words,
            variants: Vec::new(),
        };
        let first_map = [BTreeMap::from([(128, 0)])];
        let second_map = [BTreeMap::from([(128, 2)])];
        let first = shader
            .variant_for_layout(&table, &vec![], &[], &first_map)
            .unwrap();
        assert_eq!(
            first,
            shader
                .variant_for_layout(&table, &vec![], &[], &first_map)
                .unwrap()
        );
        let second = shader
            .variant_for_layout(&table, &vec![], &[], &second_map)
            .unwrap();
        assert_ne!(first, second);
        for (id, expected) in [(first, vec![0, 1]), (second, vec![2, 3])] {
            let desc = table
                .shader_module(table.shader_module_ref(id).unwrap())
                .unwrap();
            let ir::ShaderSource::SpirV(words) = desc.source() else {
                panic!()
            };
            let normalized = naga::front::spv::Frontend::new(
                words.iter().copied(),
                &naga::front::spv::Options::default(),
            )
            .parse()
            .unwrap();
            let mut slots = normalized
                .global_variables
                .iter()
                .filter_map(|(_, g)| g.binding.as_ref().map(|b| b.binding))
                .collect::<Vec<_>>();
            slots.sort_unstable();
            assert_eq!(slots, expected);
            sgfx_codegen_virgl::programmable::compile_shader(
                &desc,
                ir::ShaderStage::Fragment,
                "main",
            )
            .unwrap();
        }
    }

    #[test]
    fn dynamic_descriptor_offsets_select_checked_distinct_buffer_ranges() {
        let table = ir::ResourceTable::new();
        let mut resources = Resources::new();
        let buffer = table
            .define_buffer(ir::BufferDesc::new(1024, ir::BufferUsage::UNIFORM).unwrap())
            .unwrap()
            .id();
        let buffer_handle = vk::Buffer::from_raw(1);
        let set = vk::DescriptorSet::from_raw(2);
        resources.buffers.insert(
            buffer_handle,
            Buffer {
                id: buffer,
                size: 1024,
                usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                bound: Some((vk::DeviceMemory::from_raw(3), 0)),
            },
        );
        let layout = ir::BindGroupLayoutDesc::new(vec![ir::BindGroupLayoutEntry::new(
            0,
            ir::ShaderStages::VERTEX,
            ir::BindingType::UniformBuffer,
        )])
        .unwrap();
        resources.descriptor_sets.insert(
            set,
            DescriptorSet {
                texel_formats: HashMap::new(),
                pool: vk::DescriptorPool::null(),
                layout,
                types: BTreeMap::from([(0, vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)]),
                stages: BTreeMap::from([(0, vk::ShaderStageFlags::VERTEX)]),
                bindings: HashMap::from([(
                    0,
                    DescriptorBinding::Buffer {
                        buffer: buffer_handle,
                        offset: 256,
                        range: 64,
                    },
                )]),
                cached_group: None,
                invalid: false,
            },
        );
        let first = resources.descriptor_group(&table, set, &[0]).unwrap();
        let second = resources.descriptor_group(&table, set, &[256]).unwrap();
        assert_ne!(first, second);
        let descriptor = table
            .bind_group(table.bind_group_ref(second).unwrap())
            .unwrap();
        assert!(matches!(
            descriptor.entries()[0].resource(),
            ir::BindingResource::Buffer {
                offset: 512,
                size: 64,
                ..
            }
        ));
        assert_eq!(
            resources.descriptor_group(&table, set, &[0]).unwrap(),
            first
        );
        for offsets in [&[][..], &[1][..], &[768][..], &[u32::MAX][..], &[0, 0][..]] {
            assert!(resources.descriptor_group(&table, set, offsets).is_err());
        }
        resources
            .descriptor_sets
            .get_mut(&set)
            .unwrap()
            .bindings
            .insert(
                0,
                DescriptorBinding::Buffer {
                    buffer: buffer_handle,
                    offset: 256,
                    range: vk::WHOLE_SIZE,
                },
            );
        assert!(resources.descriptor_group(&table, set, &[256]).is_err());
    }

    #[test]
    fn staging_image_upload_preserves_padding_offsets_and_rejects_invalid_regions() {
        let table = ir::ResourceTable::new();
        let mut resources = Resources::new();
        let source = vk::Buffer::from_raw(1);
        let memory = vk::DeviceMemory::from_raw(2);
        let image = vk::Image::from_raw(3);
        let buffer = table
            .define_buffer(ir::BufferDesc::new(64, ir::BufferUsage::COPY_SRC).unwrap())
            .unwrap()
            .id();
        let texture = table
            .define_texture(
                ir::TextureDesc::new(
                    ir::TextureFormat::Rgba8Unorm,
                    ir::Extent2D::new(4, 4).unwrap(),
                    ir::TextureUsage::COPY_DST,
                )
                .unwrap(),
            )
            .unwrap()
            .id();
        resources.buffers.insert(
            source,
            Buffer {
                id: buffer,
                size: 64,
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                bound: Some((memory, 8)),
            },
        );
        let mut bytes = AlignedBytes::zeroed(72).unwrap();
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = index as u8;
        }
        resources.memories.insert(
            memory,
            Memory {
                bytes,
                mapped: false,
            },
        );
        resources.images.insert(
            image,
            crate::images::Image {
                mip_levels: 1,
                array_layers: 1,
                flags: vk::ImageCreateFlags::empty(),
                id: texture,
                format: vk::Format::R8G8B8A8_UNORM,
                extent: vk::Extent3D {
                    width: 4,
                    height: 4,
                    depth: 1,
                },
                usage: vk::ImageUsageFlags::TRANSFER_DST,
                bound: Some((vk::DeviceMemory::from_raw(4), 0)),
                swapchain: None,
                #[cfg(target_os = "scarlet")]
                shared: None,
            },
        );
        let region = vk::BufferImageCopy::default()
            .buffer_offset(4)
            .buffer_row_length(3)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .image_offset(vk::Offset3D { x: 1, y: 1, z: 0 })
            .image_extent(vk::Extent3D {
                width: 2,
                height: 2,
                depth: 1,
            });
        let ops = crate::transfer::upload(
            &resources,
            source,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        )
        .unwrap();
        assert!(
            matches!(&ops[0], ir::OwnedCommand::WriteTextureMip { mip_level: 0, bytes_per_row: 12, data, .. } if data == &(12u8..32).collect::<Vec<_>>())
        );
        let mut short = region;
        short.buffer_row_length = 1;
        let mut overflow = region;
        overflow.buffer_offset = u64::MAX - 3;
        let mut mip = region;
        mip.image_subresource.mip_level = 1;
        let mut outside = region;
        outside.image_offset.x = 3;
        for invalid in [short, overflow, mip, outside] {
            assert!(
                crate::transfer::upload(
                    &resources,
                    source,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[invalid]
                )
                .is_err()
            );
        }
        resources.memories.remove(&memory);
        assert!(
            crate::transfer::upload(
                &resources,
                source,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region]
            )
            .is_err()
        );
    }

    #[test]
    fn descriptor_retirement_error_preserves_failed_groups_and_releases_others() {
        let table = ir::ResourceTable::new();
        let mut resources = Resources::new();
        let buffer = table
            .define_buffer(ir::BufferDesc::new(64, ir::BufferUsage::UNIFORM).unwrap())
            .unwrap()
            .id();
        let layout = ir::BindGroupLayoutDesc::new(vec![ir::BindGroupLayoutEntry::new(
            0,
            ir::ShaderStages::VERTEX,
            ir::BindingType::UniformBuffer,
        )])
        .unwrap();
        let groups = [0, 16, 32].map(|offset| {
            let desc = ir::BindGroupDesc::new(
                &table,
                layout.clone(),
                vec![ir::BindGroupEntry::new(
                    0,
                    ir::BindingResource::Buffer {
                        buffer,
                        offset,
                        size: 16,
                    },
                )],
            )
            .unwrap();
            let id = table.define_bind_group(desc.clone()).unwrap().id();
            resources.bind_group_cache.push((desc, id));
            id
        });
        let empty = resources.empty_bind_group(&table).unwrap();
        let revision = table.abi_revision();
        let mut attempted = Vec::new();
        assert_eq!(
            resources.release_descriptor_groups(&table, |id| {
                assert_eq!(table.abi_revision(), revision);
                assert!(groups.iter().all(|id| table.bind_group_ref(*id).is_ok()));
                attempted.push(id);
                if id == groups[1] {
                    Err(vk::Result::ERROR_DEVICE_LOST)
                } else {
                    Ok(())
                }
            }),
            Err(vk::Result::ERROR_DEVICE_LOST)
        );
        assert_eq!(attempted, groups);
        assert!(table.bind_group_ref(groups[0]).is_err());
        assert!(table.bind_group_ref(groups[1]).is_ok());
        assert!(table.bind_group_ref(groups[2]).is_err());
        assert_eq!(resources.bind_group_cache.len(), 2);
        assert_eq!(resources.empty_bind_group(&table).unwrap(), empty);
        resources
            .release_descriptor_groups(&table, |_| Ok(()))
            .unwrap();
        assert!(table.bind_group_ref(groups[1]).is_err());
        assert_eq!(resources.bind_group_cache.len(), 1);
    }

    #[test]
    fn submitted_descriptor_snapshots_recycle_slots_without_aliasing_uniform_ranges() {
        let table = ir::ResourceTable::new();
        let mut resources = Resources::new();
        let packet_count = ir::MAX_BIND_GROUP_DEFINITIONS * 2 + 4;
        let buffer_size = packet_count as u64 * 128;
        let buffer_handle = vk::Buffer::from_raw(1);
        let set_handle = vk::DescriptorSet::from_raw(2);
        let buffer = table
            .define_buffer(ir::BufferDesc::new(buffer_size, ir::BufferUsage::UNIFORM).unwrap())
            .unwrap()
            .id();
        resources.buffers.insert(
            buffer_handle,
            Buffer {
                id: buffer,
                size: buffer_size,
                usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                bound: Some((vk::DeviceMemory::from_raw(3), 0)),
            },
        );
        let layout = ir::BindGroupLayoutDesc::new(vec![
            ir::BindGroupLayoutEntry::new(
                0,
                ir::ShaderStages::VERTEX,
                ir::BindingType::UniformBuffer,
            ),
            ir::BindGroupLayoutEntry::new(
                8,
                ir::ShaderStages::FRAGMENT,
                ir::BindingType::UniformBuffer,
            ),
        ])
        .unwrap();
        resources.descriptor_sets.insert(
            set_handle,
            DescriptorSet {
                texel_formats: HashMap::new(),
                pool: vk::DescriptorPool::null(),
                layout,
                types: BTreeMap::from([
                    (0, vk::DescriptorType::UNIFORM_BUFFER),
                    (4, vk::DescriptorType::UNIFORM_BUFFER),
                ]),
                stages: BTreeMap::from([
                    (0, vk::ShaderStageFlags::VERTEX),
                    (4, vk::ShaderStageFlags::FRAGMENT),
                ]),
                bindings: HashMap::new(),
                cached_group: None,
                invalid: false,
            },
        );
        let empty = resources.empty_bind_group(&table).unwrap();
        let mut previous = None;
        for packet in 0..packet_count {
            let offset = packet as u64 * 128;
            let set = resources.descriptor_sets.get_mut(&set_handle).unwrap();
            set.cached_group = None;
            set.bindings = HashMap::from([
                (
                    0,
                    DescriptorBinding::Buffer {
                        buffer: buffer_handle,
                        offset,
                        range: 24,
                    },
                ),
                (
                    4,
                    DescriptorBinding::Buffer {
                        buffer: buffer_handle,
                        offset: offset + 64,
                        range: 12,
                    },
                ),
            ]);
            let group = resources.descriptor_group(&table, set_handle, &[]).unwrap();
            assert_ne!(Some(group), previous);
            assert_eq!(
                resources.descriptor_group(&table, set_handle, &[]).unwrap(),
                group
            );
            let snapshot = table
                .bind_group_shared(table.bind_group_ref(group).unwrap())
                .unwrap();
            assert!(matches!(snapshot.entries()[0].resource(),
                ir::BindingResource::Buffer { offset: actual, size: 24, .. } if actual == offset));
            assert!(matches!(snapshot.entries()[1].resource(),
                ir::BindingResource::Buffer { offset: actual, size: 12, .. } if actual == offset + 64));
            let mut released = Vec::new();
            resources
                .release_descriptor_groups(&table, |id| {
                    released.push(id);
                    Ok(())
                })
                .unwrap();
            assert_eq!(released, vec![group]);
            assert!(table.bind_group_ref(group).is_err());
            assert!(
                resources.descriptor_sets[&set_handle]
                    .cached_group
                    .is_none()
            );
            assert_eq!(resources.bind_group_cache.len(), 1);
            assert_eq!(resources.empty_bind_group(&table).unwrap(), empty);
            // Already materialized native metadata remains immutable after its
            // logical slot is retired; later packets cannot overwrite it.
            assert!(matches!(snapshot.entries()[0].resource(),
                ir::BindingResource::Buffer { offset: actual, .. } if actual == offset));
            previous = Some(group);
        }
    }

    #[test]
    fn epoch_cache_clear_cannot_reuse_an_old_table_identity() {
        let old_table = ir::ResourceTable::new();
        let new_table = ir::ResourceTable::new();
        let mut resources = Resources::new();
        let handle = vk::DescriptorSet::from_raw(1);
        resources.descriptor_sets.insert(
            handle,
            DescriptorSet {
                texel_formats: HashMap::new(),
                pool: vk::DescriptorPool::null(),
                types: BTreeMap::new(),
                stages: BTreeMap::new(),
                layout: ir::BindGroupLayoutDesc::new(vec![]).unwrap(),
                bindings: HashMap::new(),
                cached_group: None,
                invalid: false,
            },
        );
        let old_group = resources.descriptor_group(&old_table, handle, &[]).unwrap();
        resources.clear_ir_cache();
        let new_group = resources.descriptor_group(&new_table, handle, &[]).unwrap();
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
    fn image_cpu_mirrors_materialize_only_on_host_access_and_keep_mapping_stable() {
        let mut bytes = AlignedBytes::unbacked(64 * 1024 * 1024 + 1).unwrap();
        assert_eq!(bytes.len(), 64 * 1024 * 1024 + 1);
        assert_eq!(bytes.words.capacity(), 0);
        bytes.materialize().unwrap();
        assert!(bytes.iter().all(|byte| *byte == 0));
        let address = bytes.as_mut_ptr();
        let last = bytes.len() - 1;
        bytes[last] = 17;
        bytes.materialize().unwrap();
        assert_eq!(bytes.as_mut_ptr(), address);
        assert_eq!(bytes[bytes.len() - 1], 17);
        assert!(matches!(
            AlignedBytes::unbacked(usize::MAX),
            Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY)
        ));
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
    fn spirv_round_trip_repairs_the_point_size_interface_layout_regression() {
        // An authored gl_PerVertex block with Position and PointSize, plus a
        // separate color output. Naga 24 validates the input but its writer/
        // parser round trip produces an undersized private output structure.
        let mut words = vec![0x07230203, 0x00010000, 0, 21, 0];
        for (opcode, operands) in [
            (17, vec![1]),
            (14, vec![0, 1]),
            (15, vec![0, 17, u32::from_le_bytes(*b"main"), 0, 15, 16]),
            (71, vec![6, 2]),
            (72, vec![6, 0, 11, 0]),
            (72, vec![6, 1, 11, 1]),
            (71, vec![16, 30, 0]),
            (19, vec![1]),
            (33, vec![2, 1]),
            (22, vec![3, 32]),
            (23, vec![4, 3, 4]),
            (30, vec![6, 4, 3]),
            (32, vec![7, 3, 6]),
            (32, vec![8, 3, 3]),
            (32, vec![9, 3, 4]),
            (21, vec![10, 32, 0]),
            (43, vec![10, 11, 0]),
            (43, vec![10, 12, 1]),
            (43, vec![3, 13, 1f32.to_bits()]),
            (44, vec![4, 14, 13, 13, 13, 13]),
            (59, vec![7, 15, 3]),
            (59, vec![9, 16, 3]),
            (54, vec![1, 17, 0, 2]),
            (248, vec![18]),
            (65, vec![8, 19, 15, 12]),
            (62, vec![19, 13]),
            (65, vec![9, 20, 15, 11]),
            (62, vec![20, 14]),
            (62, vec![16, 14]),
            (253, vec![]),
            (56, vec![]),
        ] {
            words.push(((operands.len() as u32 + 1) << 16) | opcode);
            words.extend(operands);
        }
        let input = naga::front::spv::Frontend::new(
            words.iter().copied(),
            &naga::front::spv::Options::default(),
        )
        .parse()
        .unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&input)
        .unwrap();
        let module = normalize_spirv(words).unwrap();
        let shader = sgfx_codegen_virgl::programmable::compile_shader(
            &module,
            ir::ShaderStage::Vertex,
            "main",
        )
        .unwrap();
        assert!(shader.tgsi.contains("PSIZE"));
        assert!(shader.tgsi.contains("GENERIC[0]"));
    }

    #[test]
    fn mip_blits_validate_subresources_and_memory_binding_covers_the_chain() {
        let table = ir::ResourceTable::new();
        let mut resources = Resources::new();
        let memory = vk::DeviceMemory::from_raw(1);
        resources.memories.insert(
            memory,
            Memory {
                bytes: AlignedBytes::zeroed(128).unwrap(),
                mapped: false,
            },
        );
        let texture = table
            .define_texture(
                ir::TextureDesc::new(
                    ir::TextureFormat::Rgba8Unorm,
                    ir::Extent2D::new(4, 4).unwrap(),
                    ir::TextureUsage::COPY_SRC | ir::TextureUsage::COPY_DST,
                )
                .unwrap()
                .with_mip_level_count(3)
                .unwrap(),
            )
            .unwrap()
            .id();
        let image = vk::Image::from_raw(2);
        resources.images.insert(
            image,
            crate::images::Image {
                id: texture,
                format: vk::Format::R8G8B8A8_UNORM,
                extent: vk::Extent3D {
                    width: 4,
                    height: 4,
                    depth: 1,
                },
                mip_levels: 3,
                array_layers: 1,
                flags: vk::ImageCreateFlags::empty(),
                usage: vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
                bound: Some((memory, 0)),
                swapchain: None,
                #[cfg(target_os = "scarlet")]
                shared: None,
            },
        );
        assert_eq!(resources.images[&image].byte_size(), 84);
        assert!(!memory_available(&resources, memory, 64, 4));
        assert!(!memory_available(&resources, memory, 80, 4));
        assert!(memory_available(&resources, memory, 84, 4));
        let layers = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1);
        let region = vk::ImageBlit::default()
            .src_subresource(layers)
            .src_offsets([vk::Offset3D::default(), vk::Offset3D { x: 4, y: 4, z: 1 }])
            .dst_subresource(layers.mip_level(1))
            .dst_offsets([vk::Offset3D::default(), vk::Offset3D { x: 2, y: 2, z: 1 }]);
        let lower = |region, source_layout, filter| {
            crate::transfer::blit(
                &resources,
                image,
                source_layout,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
                filter,
            )
        };
        let commands = lower(
            region,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::Filter::LINEAR,
        )
        .unwrap();
        assert!(matches!(
            commands[0],
            ir::OwnedCommand::BlitTextureRegion {
                source_mip: 0,
                destination_mip: 1,
                filter: ir::FilterMode::Linear,
                ..
            }
        ));
        let mut partial = region;
        partial.dst_offsets[1].x = 1;
        let mut flip = region;
        flip.src_offsets.swap(0, 1);
        let mut layer = region;
        layer.src_subresource.layer_count = 2;
        let mut nonexistent = region;
        nonexistent.dst_subresource.mip_level = 3;
        let mut same_level = region;
        same_level.dst_subresource = region.src_subresource;
        same_level.dst_offsets = region.src_offsets;
        let partial_commands = lower(
            partial,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::Filter::LINEAR,
        )
        .unwrap();
        assert!(
            matches!(&partial_commands[0], ir::OwnedCommand::BlitTextureRegion { destination_rect, .. } if destination_rect.width() == 1)
        );
        let mut outside = region;
        outside.src_offsets[1].x = 5;
        for invalid in [outside, flip, layer, nonexistent, same_level] {
            assert!(
                lower(
                    invalid,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::Filter::LINEAR
                )
                .is_err()
            );
        }
        assert!(
            lower(
                region,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::Filter::LINEAR
            )
            .is_err()
        );
        assert!(
            lower(
                region,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::Filter::CUBIC_EXT
            )
            .is_err()
        );
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
                texel_formats: HashMap::new(),
                pool: vk::DescriptorPool::null(),
                types: BTreeMap::from([(0, vk::DescriptorType::STORAGE_BUFFER)]),
                stages: BTreeMap::from([(0, vk::ShaderStageFlags::COMPUTE)]),
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
        let first = resources.descriptor_group(&table, set_handle, &[]).unwrap();
        for _ in 0..1100 {
            resources
                .descriptor_sets
                .get_mut(&set_handle)
                .unwrap()
                .cached_group = None;
            assert_eq!(
                resources.descriptor_group(&table, set_handle, &[]).unwrap(),
                first
            );
        }
        resources.buffers.get_mut(&buffer_handle).unwrap().bound = None;
        assert!(resources.descriptor_group(&table, set_handle, &[]).is_err());
        resources.buffers.remove(&buffer_handle);
        assert!(resources.descriptor_group(&table, set_handle, &[]).is_err());
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
                texel_formats: HashMap::new(),
                pool: vk::DescriptorPool::null(),
                types: BTreeMap::from([(0, vk::DescriptorType::UNIFORM_BUFFER)]),
                stages: BTreeMap::from([(0, vk::ShaderStageFlags::COMPUTE)]),
                layout,
                bindings: HashMap::new(),
                cached_group: None,
                invalid: false,
            },
        );
        assert!(resources.descriptor_group(&table, handle, &[]).is_err());
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
        assert!(resources.descriptor_group(&table, handle, &[]).is_err());
    }
}
