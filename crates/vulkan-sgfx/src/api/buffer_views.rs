//! Owned texel-view metadata. Shader use is validated separately from creation.
use super::*;
unsafe extern "system" fn create(
    device: vk::Device,
    info: *const vk::BufferViewCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::BufferView,
) -> vk::Result {
    if out.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    *out = vk::BufferView::null();
    status((|| {
        let i = info
            .as_ref()
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        if !allocator.is_null()
            || i.s_type != vk::StructureType::BUFFER_VIEW_CREATE_INFO
            || !i.p_next.is_null()
            || !i.flags.is_empty()
            || i.format != vk::Format::R8G8B8A8_UNORM
            || !i.offset.is_multiple_of(4)
        {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let view = crate::resources::BufferView {
            buffer: i.buffer,
            offset: i.offset,
            range: i.range,
        };
        *out = with_device(device, move |rt| {
            let buffer = rt
                .resources
                .buffers
                .get(&view.buffer)
                .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
            let range = if view.range == vk::WHOLE_SIZE {
                buffer.size.checked_sub(view.offset)
            } else {
                Some(view.range)
            }
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
            if buffer.bound.is_none()
                || !buffer.usage.intersects(
                    vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER
                        | vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER,
                )
                || range == 0
                || !range.is_multiple_of(4)
                || view
                    .offset
                    .checked_add(range)
                    .is_none_or(|end| end > buffer.size)
            {
                return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
            }
            if rt.resources.buffer_views.len() >= 4096 {
                return Err(vk::Result::ERROR_TOO_MANY_OBJECTS);
            }
            let handle = vk::BufferView::from_raw(next_id());
            rt.resources
                .buffer_views
                .insert(handle, crate::resources::BufferView { range, ..view });
            Ok(handle)
        })?;
        Ok(())
    })())
}
unsafe extern "system" fn destroy(
    device: vk::Device,
    view: vk::BufferView,
    _: *const vk::AllocationCallbacks<'_>,
) {
    let _ = with_device(device, move |rt| {
        rt.resources.buffer_views.remove(&view);
        Ok(())
    });
}
pub(super) fn lookup(name: &CStr) -> vk::PFN_vkVoidFunction {
    match name.to_bytes() {
        b"vkCreateBufferView" => {
            let f: vk::PFN_vkCreateBufferView = create;
            Some(unsafe {
                std::mem::transmute::<vk::PFN_vkCreateBufferView, unsafe extern "system" fn()>(f)
            })
        }
        b"vkDestroyBufferView" => {
            let f: vk::PFN_vkDestroyBufferView = destroy;
            Some(unsafe {
                std::mem::transmute::<vk::PFN_vkDestroyBufferView, unsafe extern "system" fn()>(f)
            })
        }
        _ => None,
    }
}
