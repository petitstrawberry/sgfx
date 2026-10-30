//! Descriptor-set templates snapshot caller data and use the regular update path.
use super::*;
#[derive(Clone)]
pub(crate) struct Template {
    entries: Vec<vk::DescriptorUpdateTemplateEntry>,
    types: BTreeMap<u32, vk::DescriptorType>,
    target: Option<(vk::PipelineBindPoint, vk::PipelineLayout, u32)>,
}
fn element_size(ty: vk::DescriptorType) -> Option<usize> {
    match ty {
        vk::DescriptorType::UNIFORM_BUFFER
        | vk::DescriptorType::STORAGE_BUFFER
        | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
        | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC => Some(size_of::<vk::DescriptorBufferInfo>()),
        vk::DescriptorType::SAMPLED_IMAGE
        | vk::DescriptorType::STORAGE_IMAGE
        | vk::DescriptorType::SAMPLER
        | vk::DescriptorType::COMBINED_IMAGE_SAMPLER
        | vk::DescriptorType::INPUT_ATTACHMENT => Some(size_of::<vk::DescriptorImageInfo>()),
        _ => None,
    }
}
unsafe extern "system" fn create(
    device: vk::Device,
    info: *const vk::DescriptorUpdateTemplateCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::DescriptorUpdateTemplate,
) -> vk::Result {
    if out.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    *out = vk::DescriptorUpdateTemplate::null();
    status((|| {
        let i = info
            .as_ref()
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        if !allocator.is_null()
            || i.s_type != vk::StructureType::DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO
            || !i.p_next.is_null()
            || !i.flags.is_empty()
            || !matches!(
                i.template_type,
                vk::DescriptorUpdateTemplateType::DESCRIPTOR_SET
                    | vk::DescriptorUpdateTemplateType::PUSH_DESCRIPTORS_KHR
            )
        {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        let entries = slice(
            i.p_descriptor_update_entries,
            i.descriptor_update_entry_count,
        )?
        .to_vec();
        let layout = i.descriptor_set_layout;
        let target = (i.template_type == vk::DescriptorUpdateTemplateType::PUSH_DESCRIPTORS_KHR)
            .then_some((i.pipeline_bind_point, i.pipeline_layout, i.set));
        let handle = with_device(device, move |rt| {
            let types = if let Some((point, pipeline, set)) = target {
                if !matches!(
                    point,
                    vk::PipelineBindPoint::GRAPHICS | vk::PipelineBindPoint::COMPUTE
                ) {
                    return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                }
                let (index, desc) = rt
                    .resources
                    .pipeline_push_sets
                    .get(&pipeline)
                    .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
                if *index != set {
                    return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                }
                desc.types.clone()
            } else {
                rt.resources
                    .set_layout_types
                    .get(&layout)
                    .cloned()
                    .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?
            };
            for entry in &entries {
                let size = element_size(entry.descriptor_type)
                    .ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)?;
                if entry.dst_array_element != 0
                    || entry.descriptor_count == 0
                    || entry.descriptor_count > 16
                    || entry
                        .stride
                        .checked_mul(entry.descriptor_count.saturating_sub(1) as usize)
                        .and_then(|n| n.checked_add(entry.offset))
                        .and_then(|n| n.checked_add(size))
                        .is_none_or(|n| n > 1024 * 1024)
                {
                    return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                }
                for index in 0..entry.descriptor_count {
                    if types.get(&(entry.dst_binding + index)) != Some(&entry.descriptor_type) {
                        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
                    }
                }
            }
            if rt.resources.descriptor_templates.len() >= 4096 {
                return Err(vk::Result::ERROR_TOO_MANY_OBJECTS);
            }
            let handle = vk::DescriptorUpdateTemplate::from_raw(next_id());
            rt.resources.descriptor_templates.insert(
                handle,
                Template {
                    entries,
                    types,
                    target,
                },
            );
            Ok(handle)
        })?;
        *out = handle;
        Ok(())
    })())
}
unsafe extern "system" fn destroy(
    device: vk::Device,
    template: vk::DescriptorUpdateTemplate,
    _: *const vk::AllocationCallbacks<'_>,
) {
    let _ = with_device(device, move |rt| {
        rt.resources.descriptor_templates.remove(&template);
        Ok(())
    });
}
unsafe extern "system" fn update(
    device: vk::Device,
    set: vk::DescriptorSet,
    template: vk::DescriptorUpdateTemplate,
    data: *const std::ffi::c_void,
) {
    let result = (|| {
        let t = with_device(device, move |rt| {
            let t = rt
                .resources
                .descriptor_templates
                .get(&template)
                .cloned()
                .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
            if t.target.is_some()
                || rt
                    .resources
                    .descriptor_sets
                    .get(&set)
                    .is_none_or(|s| s.types != t.types)
            {
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            }
            Ok(t)
        })?;
        if data.is_null() && !t.entries.is_empty() {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        for entry in t.entries {
            for index in 0..entry.descriptor_count {
                let ptr = data
                    .cast::<u8>()
                    .add(entry.offset + entry.stride * index as usize);
                let mut write = vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(entry.dst_binding + index)
                    .descriptor_type(entry.descriptor_type);
                if matches!(
                    entry.descriptor_type,
                    vk::DescriptorType::UNIFORM_BUFFER
                        | vk::DescriptorType::STORAGE_BUFFER
                        | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                        | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
                ) {
                    let info = [ptr.cast::<vk::DescriptorBufferInfo>().read_unaligned()];
                    write = write.buffer_info(&info);
                    crate::resources::update_descriptor_sets(
                        device,
                        1,
                        &write,
                        0,
                        std::ptr::null(),
                    );
                } else {
                    let info = [ptr.cast::<vk::DescriptorImageInfo>().read_unaligned()];
                    write = write.image_info(&info);
                    crate::resources::update_descriptor_sets(
                        device,
                        1,
                        &write,
                        0,
                        std::ptr::null(),
                    );
                }
            }
        }
        Ok(())
    })();
    if let Err(error) = result {
        if std::env::var_os("SGFX_VULKAN_TRACE").is_some() {
            eprintln!("[SGFX Vulkan] descriptor template: {error:?}");
        }
        let _ = with_device(device, move |rt| {
            if let Some(set) = rt.resources.descriptor_sets.get_mut(&set) {
                set.invalid = true;
            }
            Ok(())
        });
    }
}
pub(super) unsafe extern "system" fn push_template(
    command: vk::CommandBuffer,
    template: vk::DescriptorUpdateTemplate,
    layout: vk::PipelineLayout,
    set: u32,
    data: *const std::ffi::c_void,
) {
    let result = (|| {
        let d = driver(command.as_raw(), Kind::Command)?;
        let t = call(&d, move |rt| {
            let t = rt
                .resources
                .descriptor_templates
                .get(&template)
                .cloned()
                .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
            let (_, original, index) = t.target.ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)?;
            if index != set
                || !super::push_descriptors::compatible(rt, original, layout, set as usize)
            {
                return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
            }
            Ok(t)
        })?;
        if data.is_null() && !t.entries.is_empty() {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        let mut writes = Vec::new();
        for entry in &t.entries {
            for index in 0..entry.descriptor_count {
                let ptr = data
                    .cast::<u8>()
                    .add(entry.offset + entry.stride * index as usize);
                let value = if matches!(
                    entry.descriptor_type,
                    vk::DescriptorType::UNIFORM_BUFFER | vk::DescriptorType::STORAGE_BUFFER
                ) {
                    let b = ptr.cast::<vk::DescriptorBufferInfo>().read_unaligned();
                    crate::resources::DescriptorBinding::Buffer {
                        buffer: b.buffer,
                        offset: b.offset,
                        range: b.range,
                    }
                } else {
                    let i = ptr.cast::<vk::DescriptorImageInfo>().read_unaligned();
                    if entry.descriptor_type == vk::DescriptorType::SAMPLER {
                        crate::resources::DescriptorBinding::Sampler(i.sampler)
                    } else {
                        crate::resources::DescriptorBinding::Image {
                            view: i.image_view,
                            sampler: i.sampler,
                            layout: i.image_layout,
                        }
                    }
                };
                writes.push(crate::resources::Write {
                    set: vk::DescriptorSet::null(),
                    binding: entry.dst_binding + index,
                    element: 0,
                    ty: entry.descriptor_type,
                    value: Some(value),
                });
            }
        }
        super::push_descriptors::push_owned(command, t.target.unwrap().0, layout, set, writes)
    })();
    if let Err(error) = result {
        record_error(command, error);
    }
}

pub(super) fn lookup(name: &CStr) -> vk::PFN_vkVoidFunction {
    macro_rules! entry {
        ($f:ident,$ty:ty) => {{
            let f: $ty = $f;
            Some(unsafe { std::mem::transmute::<$ty, unsafe extern "system" fn()>(f) })
        }};
    }
    match name.to_bytes() {
        b"vkCreateDescriptorUpdateTemplate" | b"vkCreateDescriptorUpdateTemplateKHR" => {
            entry!(create, vk::PFN_vkCreateDescriptorUpdateTemplate)
        }
        b"vkDestroyDescriptorUpdateTemplate" | b"vkDestroyDescriptorUpdateTemplateKHR" => {
            entry!(destroy, vk::PFN_vkDestroyDescriptorUpdateTemplate)
        }
        b"vkUpdateDescriptorSetWithTemplate" | b"vkUpdateDescriptorSetWithTemplateKHR" => {
            entry!(update, vk::PFN_vkUpdateDescriptorSetWithTemplate)
        }
        _ => None,
    }
}
