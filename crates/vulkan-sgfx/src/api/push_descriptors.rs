//! Immutable, command-owned snapshots for push descriptors. Partial pushes copy
//! prior compatible state; neither caller storage nor a later push can change it.
use super::*;

pub(super) fn compatible(
    rt: &Runtime,
    a: vk::PipelineLayout,
    b: vk::PipelineLayout,
    set: usize,
) -> bool {
    let (Some(a_desc), Some(b_desc)) = (
        rt.resources.pipeline_layouts.get(&a),
        rt.resources.pipeline_layouts.get(&b),
    ) else {
        return false;
    };
    a_desc.push_constant_ranges() == b_desc.push_constant_ranges()
        && a_desc.bind_groups().get(..=set) == b_desc.bind_groups().get(..=set)
        && rt
            .resources
            .pipeline_binding_maps
            .get(&a)
            .and_then(|v| v.get(..=set))
            == rt
                .resources
                .pipeline_binding_maps
                .get(&b)
                .and_then(|v| v.get(..=set))
        && rt
            .resources
            .pipeline_binding_stages
            .get(&a)
            .and_then(|v| v.get(..=set))
            == rt
                .resources
                .pipeline_binding_stages
                .get(&b)
                .and_then(|v| v.get(..=set))
}

pub(super) fn push_owned(
    command: vk::CommandBuffer,
    point: vk::PipelineBindPoint,
    layout: vk::PipelineLayout,
    set: u32,
    mut writes: Vec<crate::resources::Write>,
) -> VkResult<()> {
    if !matches!(
        point,
        vk::PipelineBindPoint::GRAPHICS | vk::PipelineBindPoint::COMPUTE
    ) || set >= 4
    {
        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
    }
    let driver = driver(command.as_raw(), Kind::Command)?;
    let source = {
        let registry = driver.recordings.lock().unwrap_or_else(|e| e.into_inner());
        let cell = registry
            .commands
            .get(&command.as_raw())
            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
        let recording = cell.lock().unwrap_or_else(|e| e.into_inner());
        if recording.state != RecordingState::Recording {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        Arc::clone(&recording.commands)
    };
    let owner = Arc::new(());
    let weak = Arc::downgrade(&owner);
    let handle = call(&driver, move |rt| {
        let (index, mut descriptor) = rt
            .resources
            .pipeline_push_sets
            .get(&layout)
            .cloned()
            .ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)?;
        if index != set {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        for prior in source.iter().rev() {
            if let RecordedCommand::BindDescriptorSets {
                point: old_point,
                layout: old_layout,
                first,
                sets,
                pushed,
                ..
            } = prior
            {
                if *old_point != point {
                    continue;
                }
                if !compatible(rt, *old_layout, layout, set as usize) {
                    break;
                }
                if let Some(offset) = set
                    .checked_sub(*first)
                    .filter(|&n| (n as usize) < sets.len())
                {
                    if pushed.is_some() {
                        descriptor.bindings = rt
                            .resources
                            .descriptor_sets
                            .get(&sets[offset as usize])
                            .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?
                            .bindings
                            .clone();
                    }
                    break;
                }
            }
        }
        if rt.resources.pushed_set_owners.len() >= ir::MAX_COMMANDS {
            return Err(vk::Result::ERROR_TOO_MANY_OBJECTS);
        }
        let handle = vk::DescriptorSet::from_raw(next_id());
        for write in &mut writes {
            write.set = handle;
        }
        rt.resources.descriptor_sets.insert(handle, descriptor);
        rt.resources.pushed_set_owners.insert(handle, weak);
        crate::resources::apply_descriptor_writes(rt, writes);
        if rt.resources.descriptor_sets[&handle].invalid {
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        Ok(handle)
    })?;
    record(
        command,
        RecordedCommand::BindDescriptorSets {
            point,
            layout,
            first: set,
            sets: vec![handle],
            dynamic_offsets: Vec::new(),
            pushed: Some(owner),
        },
    );
    Ok(())
}

unsafe extern "system" fn push(
    command: vk::CommandBuffer,
    point: vk::PipelineBindPoint,
    layout: vk::PipelineLayout,
    set: u32,
    count: u32,
    writes: *const vk::WriteDescriptorSet<'_>,
) {
    let result = (|| {
        let writes = slice(writes, count)?;
        if writes.iter().any(|w| {
            matches!(
                w.descriptor_type,
                vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                    | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
            )
        }) {
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        }
        push_owned(
            command,
            point,
            layout,
            set,
            crate::resources::snapshot_descriptor_writes(writes),
        )
    })();
    if let Err(error) = result {
        record_error(command, error);
    }
}

pub(super) fn lookup(name: &CStr) -> vk::PFN_vkVoidFunction {
    if name.to_bytes() == b"vkCmdPushDescriptorSetKHR" {
        let f: vk::PFN_vkCmdPushDescriptorSetKHR = push;
        Some(unsafe {
            std::mem::transmute::<vk::PFN_vkCmdPushDescriptorSetKHR, unsafe extern "system" fn()>(f)
        })
    } else if name.to_bytes() == b"vkCmdPushDescriptorSetWithTemplateKHR" {
        let f: vk::PFN_vkCmdPushDescriptorSetWithTemplateKHR =
            super::descriptor_template::push_template;
        Some(unsafe {
            std::mem::transmute::<
                vk::PFN_vkCmdPushDescriptorSetWithTemplateKHR,
                unsafe extern "system" fn(),
            >(f)
        })
    } else {
        None
    }
}
