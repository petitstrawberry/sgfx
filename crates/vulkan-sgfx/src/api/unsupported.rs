//! Typed entry points for core operations not yet executable by this subset.
//! They reject recording/creation explicitly; they never manufacture GPU results.
//! WineD3D loads the complete core table before issuing its first command.
use super::record_error;
use ash::vk::*;
use std::ffi::*;
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_cmd_begin_query(
    command_buffer: CommandBuffer,
    query_pool: QueryPool,
    query: u32,
    flags: QueryControlFlags,
) {
    eprintln!("SGFX Vulkan: vkCmdBeginQuery is not supported by this backend subset");
    record_error(command_buffer, ash::vk::Result::ERROR_FEATURE_NOT_PRESENT);
}
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_cmd_clear_attachments(
    command_buffer: CommandBuffer,
    attachment_count: u32,
    p_attachments: *const ClearAttachment,
    rect_count: u32,
    p_rects: *const ClearRect,
) {
    eprintln!("SGFX Vulkan: vkCmdClearAttachments is not supported by this backend subset");
    record_error(command_buffer, ash::vk::Result::ERROR_FEATURE_NOT_PRESENT);
}
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_cmd_clear_depth_stencil_image(
    command_buffer: CommandBuffer,
    image: Image,
    image_layout: ImageLayout,
    p_depth_stencil: *const ClearDepthStencilValue,
    range_count: u32,
    p_ranges: *const ImageSubresourceRange,
) {
    eprintln!("SGFX Vulkan: vkCmdClearDepthStencilImage is not supported by this backend subset");
    record_error(command_buffer, ash::vk::Result::ERROR_FEATURE_NOT_PRESENT);
}
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_cmd_copy_image(
    command_buffer: CommandBuffer,
    src_image: Image,
    src_image_layout: ImageLayout,
    dst_image: Image,
    dst_image_layout: ImageLayout,
    region_count: u32,
    p_regions: *const ImageCopy,
) {
    eprintln!("SGFX Vulkan: vkCmdCopyImage is not supported by this backend subset");
    record_error(command_buffer, ash::vk::Result::ERROR_FEATURE_NOT_PRESENT);
}
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_cmd_copy_query_pool_results(
    command_buffer: CommandBuffer,
    query_pool: QueryPool,
    first_query: u32,
    query_count: u32,
    dst_buffer: Buffer,
    dst_offset: DeviceSize,
    stride: DeviceSize,
    flags: QueryResultFlags,
) {
    eprintln!("SGFX Vulkan: vkCmdCopyQueryPoolResults is not supported by this backend subset");
    record_error(command_buffer, ash::vk::Result::ERROR_FEATURE_NOT_PRESENT);
}
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_cmd_dispatch_indirect(
    command_buffer: CommandBuffer,
    buffer: Buffer,
    offset: DeviceSize,
) {
    eprintln!("SGFX Vulkan: vkCmdDispatchIndirect is not supported by this backend subset");
    record_error(command_buffer, ash::vk::Result::ERROR_FEATURE_NOT_PRESENT);
}
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_cmd_draw_indexed_indirect(
    command_buffer: CommandBuffer,
    buffer: Buffer,
    offset: DeviceSize,
    draw_count: u32,
    stride: u32,
) {
    eprintln!("SGFX Vulkan: vkCmdDrawIndexedIndirect is not supported by this backend subset");
    record_error(command_buffer, ash::vk::Result::ERROR_FEATURE_NOT_PRESENT);
}
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_cmd_draw_indirect(
    command_buffer: CommandBuffer,
    buffer: Buffer,
    offset: DeviceSize,
    draw_count: u32,
    stride: u32,
) {
    eprintln!("SGFX Vulkan: vkCmdDrawIndirect is not supported by this backend subset");
    record_error(command_buffer, ash::vk::Result::ERROR_FEATURE_NOT_PRESENT);
}
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_cmd_end_query(
    command_buffer: CommandBuffer,
    query_pool: QueryPool,
    query: u32,
) {
    eprintln!("SGFX Vulkan: vkCmdEndQuery is not supported by this backend subset");
    record_error(command_buffer, ash::vk::Result::ERROR_FEATURE_NOT_PRESENT);
}
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_cmd_execute_commands(
    command_buffer: CommandBuffer,
    command_buffer_count: u32,
    p_command_buffers: *const CommandBuffer,
) {
    eprintln!("SGFX Vulkan: vkCmdExecuteCommands is not supported by this backend subset");
    record_error(command_buffer, ash::vk::Result::ERROR_FEATURE_NOT_PRESENT);
}
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_cmd_fill_buffer(
    command_buffer: CommandBuffer,
    dst_buffer: Buffer,
    dst_offset: DeviceSize,
    size: DeviceSize,
    data: u32,
) {
    eprintln!("SGFX Vulkan: vkCmdFillBuffer is not supported by this backend subset");
    record_error(command_buffer, ash::vk::Result::ERROR_FEATURE_NOT_PRESENT);
}
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_cmd_reset_event(
    command_buffer: CommandBuffer,
    event: Event,
    stage_mask: PipelineStageFlags,
) {
    eprintln!("SGFX Vulkan: vkCmdResetEvent is not supported by this backend subset");
    record_error(command_buffer, ash::vk::Result::ERROR_FEATURE_NOT_PRESENT);
}
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_cmd_reset_query_pool(
    command_buffer: CommandBuffer,
    query_pool: QueryPool,
    first_query: u32,
    query_count: u32,
) {
    eprintln!("SGFX Vulkan: vkCmdResetQueryPool is not supported by this backend subset");
    record_error(command_buffer, ash::vk::Result::ERROR_FEATURE_NOT_PRESENT);
}
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_cmd_resolve_image(
    command_buffer: CommandBuffer,
    src_image: Image,
    src_image_layout: ImageLayout,
    dst_image: Image,
    dst_image_layout: ImageLayout,
    region_count: u32,
    p_regions: *const ImageResolve,
) {
    eprintln!("SGFX Vulkan: vkCmdResolveImage is not supported by this backend subset");
    record_error(command_buffer, ash::vk::Result::ERROR_FEATURE_NOT_PRESENT);
}
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_cmd_set_blend_constants(
    command_buffer: CommandBuffer,
    blend_constants: *const [f32; 4usize],
) {
    eprintln!("SGFX Vulkan: vkCmdSetBlendConstants is not supported by this backend subset");
    record_error(command_buffer, ash::vk::Result::ERROR_FEATURE_NOT_PRESENT);
}
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_cmd_set_depth_bias(
    command_buffer: CommandBuffer,
    depth_bias_constant_factor: f32,
    depth_bias_clamp: f32,
    depth_bias_slope_factor: f32,
) {
    eprintln!("SGFX Vulkan: vkCmdSetDepthBias is not supported by this backend subset");
    record_error(command_buffer, ash::vk::Result::ERROR_FEATURE_NOT_PRESENT);
}
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_cmd_set_depth_bounds(
    command_buffer: CommandBuffer,
    min_depth_bounds: f32,
    max_depth_bounds: f32,
) {
    eprintln!("SGFX Vulkan: vkCmdSetDepthBounds is not supported by this backend subset");
    record_error(command_buffer, ash::vk::Result::ERROR_FEATURE_NOT_PRESENT);
}
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_cmd_set_event(
    command_buffer: CommandBuffer,
    event: Event,
    stage_mask: PipelineStageFlags,
) {
    eprintln!("SGFX Vulkan: vkCmdSetEvent is not supported by this backend subset");
    record_error(command_buffer, ash::vk::Result::ERROR_FEATURE_NOT_PRESENT);
}
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_cmd_set_line_width(
    command_buffer: CommandBuffer,
    line_width: f32,
) {
    eprintln!("SGFX Vulkan: vkCmdSetLineWidth is not supported by this backend subset");
    record_error(command_buffer, ash::vk::Result::ERROR_FEATURE_NOT_PRESENT);
}
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_cmd_set_stencil_compare_mask(
    command_buffer: CommandBuffer,
    face_mask: StencilFaceFlags,
    compare_mask: u32,
) {
    eprintln!("SGFX Vulkan: vkCmdSetStencilCompareMask is not supported by this backend subset");
    record_error(command_buffer, ash::vk::Result::ERROR_FEATURE_NOT_PRESENT);
}
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_cmd_set_stencil_reference(
    command_buffer: CommandBuffer,
    face_mask: StencilFaceFlags,
    reference: u32,
) {
    eprintln!("SGFX Vulkan: vkCmdSetStencilReference is not supported by this backend subset");
    record_error(command_buffer, ash::vk::Result::ERROR_FEATURE_NOT_PRESENT);
}
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_cmd_set_stencil_write_mask(
    command_buffer: CommandBuffer,
    face_mask: StencilFaceFlags,
    write_mask: u32,
) {
    eprintln!("SGFX Vulkan: vkCmdSetStencilWriteMask is not supported by this backend subset");
    record_error(command_buffer, ash::vk::Result::ERROR_FEATURE_NOT_PRESENT);
}
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_cmd_update_buffer(
    command_buffer: CommandBuffer,
    dst_buffer: Buffer,
    dst_offset: DeviceSize,
    data_size: DeviceSize,
    p_data: *const c_void,
) {
    eprintln!("SGFX Vulkan: vkCmdUpdateBuffer is not supported by this backend subset");
    record_error(command_buffer, ash::vk::Result::ERROR_FEATURE_NOT_PRESENT);
}
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_cmd_wait_events(
    command_buffer: CommandBuffer,
    event_count: u32,
    p_events: *const Event,
    src_stage_mask: PipelineStageFlags,
    dst_stage_mask: PipelineStageFlags,
    memory_barrier_count: u32,
    p_memory_barriers: *const MemoryBarrier<'_>,
    buffer_memory_barrier_count: u32,
    p_buffer_memory_barriers: *const BufferMemoryBarrier<'_>,
    image_memory_barrier_count: u32,
    p_image_memory_barriers: *const ImageMemoryBarrier<'_>,
) {
    eprintln!("SGFX Vulkan: vkCmdWaitEvents is not supported by this backend subset");
    record_error(command_buffer, ash::vk::Result::ERROR_FEATURE_NOT_PRESENT);
}
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_cmd_write_timestamp(
    command_buffer: CommandBuffer,
    pipeline_stage: PipelineStageFlags,
    query_pool: QueryPool,
    query: u32,
) {
    eprintln!("SGFX Vulkan: vkCmdWriteTimestamp is not supported by this backend subset");
    record_error(command_buffer, ash::vk::Result::ERROR_FEATURE_NOT_PRESENT);
}
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_create_buffer_view(
    device: ash::vk::Device,
    p_create_info: *const BufferViewCreateInfo<'_>,
    p_allocator: *const AllocationCallbacks<'_>,
    p_view: *mut BufferView,
) -> Result {
    eprintln!("SGFX Vulkan: vkCreateBufferView is not supported by this backend subset");
    if !p_view.is_null() {
        *p_view = Default::default();
    }
    ash::vk::Result::ERROR_FEATURE_NOT_PRESENT
}
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_create_query_pool(
    device: ash::vk::Device,
    p_create_info: *const QueryPoolCreateInfo<'_>,
    p_allocator: *const AllocationCallbacks<'_>,
    p_query_pool: *mut QueryPool,
) -> Result {
    eprintln!("SGFX Vulkan: vkCreateQueryPool is not supported by this backend subset");
    if !p_query_pool.is_null() {
        *p_query_pool = Default::default();
    }
    ash::vk::Result::ERROR_FEATURE_NOT_PRESENT
}
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_destroy_buffer_view(
    device: ash::vk::Device,
    buffer_view: BufferView,
    p_allocator: *const AllocationCallbacks<'_>,
) {
    eprintln!("SGFX Vulkan: vkDestroyBufferView is not supported by this backend subset");
}
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_destroy_query_pool(
    device: ash::vk::Device,
    query_pool: QueryPool,
    p_allocator: *const AllocationCallbacks<'_>,
) {
    eprintln!("SGFX Vulkan: vkDestroyQueryPool is not supported by this backend subset");
}
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_get_image_subresource_layout(
    device: ash::vk::Device,
    image: Image,
    p_subresource: *const ImageSubresource,
    p_layout: *mut SubresourceLayout,
) {
    eprintln!("SGFX Vulkan: vkGetImageSubresourceLayout is not supported by this backend subset");
    if !p_layout.is_null() {
        *p_layout = Default::default();
    }
}
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_get_query_pool_results(
    device: ash::vk::Device,
    query_pool: QueryPool,
    first_query: u32,
    query_count: u32,
    data_size: usize,
    p_data: *mut c_void,
    stride: DeviceSize,
    flags: QueryResultFlags,
) -> Result {
    eprintln!("SGFX Vulkan: vkGetQueryPoolResults is not supported by this backend subset");
    ash::vk::Result::ERROR_FEATURE_NOT_PRESENT
}
#[allow(unused_variables)]
unsafe extern "system" fn unsupported_queue_bind_sparse(
    queue: Queue,
    bind_info_count: u32,
    p_bind_info: *const BindSparseInfo<'_>,
    fence: Fence,
) -> Result {
    eprintln!("SGFX Vulkan: vkQueueBindSparse is not supported by this backend subset");
    ash::vk::Result::ERROR_FEATURE_NOT_PRESENT
}
pub(super) fn lookup(name: &CStr) -> PFN_vkVoidFunction {
    macro_rules! entry {
        ($f:path,$ty:ty) => {{
            let function: $ty = $f;
            Some(unsafe { std::mem::transmute::<$ty, unsafe extern "system" fn()>(function) })
        }};
    }
    match name.to_bytes() {
        b"vkCmdBeginQuery" => entry!(unsupported_cmd_begin_query, PFN_vkCmdBeginQuery),
        b"vkCmdClearAttachments" => {
            entry!(unsupported_cmd_clear_attachments, PFN_vkCmdClearAttachments)
        }
        b"vkCmdClearDepthStencilImage" => entry!(
            unsupported_cmd_clear_depth_stencil_image,
            PFN_vkCmdClearDepthStencilImage
        ),
        b"vkCmdCopyImage" => entry!(unsupported_cmd_copy_image, PFN_vkCmdCopyImage),
        b"vkCmdCopyQueryPoolResults" => entry!(
            unsupported_cmd_copy_query_pool_results,
            PFN_vkCmdCopyQueryPoolResults
        ),
        b"vkCmdDispatchIndirect" => {
            entry!(unsupported_cmd_dispatch_indirect, PFN_vkCmdDispatchIndirect)
        }
        b"vkCmdDrawIndexedIndirect" => entry!(
            unsupported_cmd_draw_indexed_indirect,
            PFN_vkCmdDrawIndexedIndirect
        ),
        b"vkCmdDrawIndirect" => entry!(unsupported_cmd_draw_indirect, PFN_vkCmdDrawIndirect),
        b"vkCmdEndQuery" => entry!(unsupported_cmd_end_query, PFN_vkCmdEndQuery),
        b"vkCmdExecuteCommands" => {
            entry!(unsupported_cmd_execute_commands, PFN_vkCmdExecuteCommands)
        }
        b"vkCmdFillBuffer" => entry!(unsupported_cmd_fill_buffer, PFN_vkCmdFillBuffer),
        b"vkCmdResetEvent" => entry!(unsupported_cmd_reset_event, PFN_vkCmdResetEvent),
        b"vkCmdResetQueryPool" => entry!(unsupported_cmd_reset_query_pool, PFN_vkCmdResetQueryPool),
        b"vkCmdResolveImage" => entry!(unsupported_cmd_resolve_image, PFN_vkCmdResolveImage),
        b"vkCmdSetBlendConstants" => entry!(
            unsupported_cmd_set_blend_constants,
            PFN_vkCmdSetBlendConstants
        ),
        b"vkCmdSetDepthBias" => entry!(unsupported_cmd_set_depth_bias, PFN_vkCmdSetDepthBias),
        b"vkCmdSetDepthBounds" => entry!(unsupported_cmd_set_depth_bounds, PFN_vkCmdSetDepthBounds),
        b"vkCmdSetEvent" => entry!(unsupported_cmd_set_event, PFN_vkCmdSetEvent),
        b"vkCmdSetLineWidth" => entry!(unsupported_cmd_set_line_width, PFN_vkCmdSetLineWidth),
        b"vkCmdSetStencilCompareMask" => entry!(
            unsupported_cmd_set_stencil_compare_mask,
            PFN_vkCmdSetStencilCompareMask
        ),
        b"vkCmdSetStencilReference" => entry!(
            unsupported_cmd_set_stencil_reference,
            PFN_vkCmdSetStencilReference
        ),
        b"vkCmdSetStencilWriteMask" => entry!(
            unsupported_cmd_set_stencil_write_mask,
            PFN_vkCmdSetStencilWriteMask
        ),
        b"vkCmdUpdateBuffer" => entry!(unsupported_cmd_update_buffer, PFN_vkCmdUpdateBuffer),
        b"vkCmdWaitEvents" => entry!(unsupported_cmd_wait_events, PFN_vkCmdWaitEvents),
        b"vkCmdWriteTimestamp" => entry!(unsupported_cmd_write_timestamp, PFN_vkCmdWriteTimestamp),
        b"vkCreateBufferView" => entry!(unsupported_create_buffer_view, PFN_vkCreateBufferView),
        b"vkCreateQueryPool" => entry!(unsupported_create_query_pool, PFN_vkCreateQueryPool),
        b"vkDestroyBufferView" => entry!(unsupported_destroy_buffer_view, PFN_vkDestroyBufferView),
        b"vkDestroyQueryPool" => entry!(unsupported_destroy_query_pool, PFN_vkDestroyQueryPool),
        b"vkGetImageSubresourceLayout" => entry!(
            unsupported_get_image_subresource_layout,
            PFN_vkGetImageSubresourceLayout
        ),
        b"vkGetQueryPoolResults" => entry!(
            unsupported_get_query_pool_results,
            PFN_vkGetQueryPoolResults
        ),
        b"vkQueueBindSparse" => entry!(unsupported_queue_bind_sparse, PFN_vkQueueBindSparse),
        _ => None,
    }
}
