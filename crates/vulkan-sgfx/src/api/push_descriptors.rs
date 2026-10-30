//! Immutable, command-owned snapshots for push descriptors. Partial pushes copy
//! prior compatible state; neither caller storage nor a later push can change it.
use super::*;

pub(super) fn compatible(
    rt: &Runtime,
    a: vk::PipelineLayout,
    b: vk::PipelineLayout,
    set: usize,
) -> bool {
    compatible_resources(&rt.resources, a, b, set)
}

fn compatible_resources(
    resources: &crate::resources::Resources,
    a: vk::PipelineLayout,
    b: vk::PipelineLayout,
    set: usize,
) -> bool {
    let (Some(a_desc), Some(b_desc)) = (
        resources.pipeline_layouts.get(&a),
        resources.pipeline_layouts.get(&b),
    ) else {
        return false;
    };
    a_desc.push_constant_ranges() == b_desc.push_constant_ranges()
        && a_desc.bind_groups().get(..=set) == b_desc.bind_groups().get(..=set)
        && resources
            .pipeline_binding_maps
            .get(&a)
            .and_then(|v| v.get(..=set))
            == resources
                .pipeline_binding_maps
                .get(&b)
                .and_then(|v| v.get(..=set))
        && resources
            .pipeline_binding_stages
            .get(&a)
            .and_then(|v| v.get(..=set))
            == resources
                .pipeline_binding_stages
                .get(&b)
                .and_then(|v| v.get(..=set))
}

pub(super) type History = Vec<(vk::PipelineLayout, Option<(vk::DescriptorSet, bool)>)>;

pub(super) fn push_owned(
    command: vk::CommandBuffer,
    point: vk::PipelineBindPoint,
    layout: vk::PipelineLayout,
    set: u32,
    writes: Vec<crate::resources::Write>,
) -> VkResult<()> {
    let driver = driver(command.as_raw(), Kind::Command)?;
    let recording = driver
        .recordings
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .commands
        .get(&command.as_raw())
        .cloned()
        .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?;
    let mut recording = recording.lock().unwrap_or_else(|e| e.into_inner());
    record_owned(&mut recording, point, layout, set, writes)
}

fn record_owned(
    recording: &mut Recording,
    point: vk::PipelineBindPoint,
    layout: vk::PipelineLayout,
    set: u32,
    writes: Vec<crate::resources::Write>,
) -> VkResult<()> {
    if !matches!(
        point,
        vk::PipelineBindPoint::GRAPHICS | vk::PipelineBindPoint::COMPUTE
    ) || set >= 4
        || writes.iter().any(|write| {
            matches!(
                write.ty,
                vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                    | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
            )
        })
    {
        return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
    }
    if recording.state != RecordingState::Recording {
        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
    }
    // Append the materialization and its bind atomically with respect to both
    // recording snapshots and the command limit. Never retain the command Arc:
    // doing so makes every push copy the entire recording on append.
    if recording.commands.len() > ir::MAX_COMMANDS - 2 {
        recording.fail(vk::Result::ERROR_OUT_OF_HOST_MEMORY);
        return Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY);
    }
    let mut history = Vec::new();
    for prior in recording.commands.iter().rev() {
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
            let bound = set
                .checked_sub(*first)
                .and_then(|offset| sets.get(offset as usize))
                .map(|handle| (*handle, pushed.is_some()));
            // Even a bind to another set can disturb layout compatibility.
            history.push((*old_layout, bound));
            if bound.is_some() {
                break;
            }
        }
    }
    let owner = Arc::new(());
    let handle = vk::DescriptorSet::from_raw(next_id());
    recording.push(RecordedCommand::PushDescriptorWrites {
        point,
        layout,
        set,
        writes,
        handle,
        owner: Arc::downgrade(&owner),
        history,
    })?;
    recording.push(RecordedCommand::BindDescriptorSets {
        point,
        layout,
        first: set,
        sets: vec![handle],
        dynamic_offsets: Vec::new(),
        pushed: Some(owner),
    })
}

fn descriptor(
    resources: &crate::resources::Resources,
    layout: vk::PipelineLayout,
    set: u32,
    history: &History,
) -> VkResult<crate::resources::DescriptorSet> {
    let (index, mut descriptor) = resources
        .pipeline_push_sets
        .get(&layout)
        .cloned()
        .ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)?;
    if index != set {
        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
    }
    for (old_layout, bound) in history {
        if !compatible_resources(resources, *old_layout, layout, set as usize) {
            break;
        }
        if let Some((handle, pushed)) = bound {
            if *pushed {
                descriptor.bindings = resources
                    .descriptor_sets
                    .get(handle)
                    .ok_or(vk::Result::ERROR_INITIALIZATION_FAILED)?
                    .bindings
                    .clone();
            }
            break;
        }
    }
    Ok(descriptor)
}

pub(super) fn resolve(
    rt: &mut Runtime,
    layout: vk::PipelineLayout,
    set: u32,
    writes: &[crate::resources::Write],
    handle: vk::DescriptorSet,
    owner: &std::sync::Weak<()>,
    history: &History,
) -> VkResult<()> {
    // Replays and shared submission snapshots refer to the same immutable set.
    if let Some(existing) = rt.resources.pushed_set_owners.get(&handle) {
        return if existing.ptr_eq(owner)
            && rt
                .resources
                .descriptor_sets
                .get(&handle)
                .is_some_and(|set| !set.invalid)
        {
            Ok(())
        } else {
            Err(vk::Result::ERROR_INITIALIZATION_FAILED)
        };
    }
    let descriptor = descriptor(&rt.resources, layout, set, history)?;
    if rt.resources.pushed_set_owners.len() >= ir::MAX_COMMANDS {
        return Err(vk::Result::ERROR_TOO_MANY_OBJECTS);
    }
    let writes = writes
        .iter()
        .cloned()
        .map(|mut write| {
            write.set = handle;
            write
        })
        .collect();
    rt.resources.descriptor_sets.insert(handle, descriptor);
    rt.resources.pushed_set_owners.insert(handle, owner.clone());
    crate::resources::apply_descriptor_writes(rt, writes);
    if rt.resources.descriptor_sets[&handle].invalid {
        return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::{DescriptorBinding, DescriptorSet, Resources, Write};

    fn recording() -> Recording {
        let mut recording = Recording::new(0);
        recording.state = RecordingState::Recording;
        recording
    }

    fn bind(
        point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        first: u32,
        sets: Vec<vk::DescriptorSet>,
    ) -> RecordedCommand {
        RecordedCommand::BindDescriptorSets {
            point,
            layout,
            first,
            sets,
            dynamic_offsets: Vec::new(),
            pushed: None,
        }
    }

    fn buffer(buffer: u64) -> DescriptorBinding {
        DescriptorBinding::Buffer {
            buffer: vk::Buffer::from_raw(buffer),
            offset: 0,
            range: 16,
        }
    }

    fn fixture() -> (Resources, [vk::PipelineLayout; 3], vk::DescriptorSet) {
        let mut resources = Resources::new();
        let group = ir::BindGroupLayoutDesc::new(vec![
            ir::BindGroupLayoutEntry::new(
                0,
                ir::ShaderStages::COMPUTE,
                ir::BindingType::UniformBuffer,
            ),
            ir::BindGroupLayoutEntry::new(
                2,
                ir::ShaderStages::COMPUTE,
                ir::BindingType::UniformBuffer,
            ),
        ])
        .unwrap();
        let template = DescriptorSet::new(
            HashMap::new(),
            vk::DescriptorPool::null(),
            group.clone(),
            BTreeMap::from([
                (0, vk::DescriptorType::UNIFORM_BUFFER),
                (1, vk::DescriptorType::UNIFORM_BUFFER),
            ]),
            BTreeMap::from([
                (0, vk::ShaderStageFlags::COMPUTE),
                (1, vk::ShaderStageFlags::COMPUTE),
            ]),
            HashMap::new(),
        )
        .unwrap();
        let layouts = [1, 2, 3].map(vk::PipelineLayout::from_raw);
        for (index, layout) in layouts.iter().enumerate() {
            let desc = ir::PipelineLayoutDesc::new(vec![
                ir::BindGroupLayoutDesc::new(vec![]).unwrap(),
                group.clone(),
            ])
            .unwrap();
            let desc = if index == 2 {
                desc.with_push_constant_ranges(vec![
                    ir::PushConstantRange::new(ir::ShaderStages::COMPUTE, 0, 4).unwrap(),
                ])
                .unwrap()
            } else {
                desc
            };
            resources.pipeline_layouts.insert(*layout, desc);
            resources.pipeline_binding_maps.insert(
                *layout,
                vec![BTreeMap::new(), template.binding_map().clone()],
            );
            resources
                .pipeline_binding_stages
                .insert(*layout, vec![BTreeMap::new(), template.stages.clone()]);
            resources
                .pipeline_push_sets
                .insert(*layout, (1, template.clone()));
        }
        let prior = vk::DescriptorSet::from_raw(10);
        let mut descriptor = template;
        descriptor.bindings = HashMap::from([(0, buffer(20)), (1, buffer(21))]);
        resources.descriptor_sets.insert(prior, descriptor);
        (resources, layouts, prior)
    }

    #[test]
    fn history_keeps_intervening_layouts_and_stops_at_first_covering_bind() {
        let mut recording = recording();
        let point = vk::PipelineBindPoint::COMPUTE;
        let [a, b] = [1, 2].map(vk::PipelineLayout::from_raw);
        let unrelated = vk::DescriptorSet::from_raw(9);
        record_owned(&mut recording, point, a, 1, vec![]).unwrap();
        let RecordedCommand::PushDescriptorWrites {
            handle: prior,
            owner,
            ..
        } = &recording.commands[0]
        else {
            panic!()
        };
        let prior = *prior;
        let owner = owner.clone();
        recording
            .push(bind(vk::PipelineBindPoint::GRAPHICS, b, 1, vec![unrelated]))
            .unwrap();
        recording.push(bind(point, b, 0, vec![unrelated])).unwrap();
        let commands = Arc::as_ptr(&recording.commands);
        record_owned(&mut recording, point, b, 1, vec![]).unwrap();
        assert_eq!(Arc::as_ptr(&recording.commands), commands);
        assert_eq!(Arc::strong_count(&recording.commands), 1);
        assert_eq!(
            owner.strong_count(),
            1,
            "history must not retain prior command owners"
        );
        let RecordedCommand::PushDescriptorWrites { history, .. } = &recording.commands[4] else {
            panic!()
        };
        assert_eq!(history, &vec![(b, None), (a, Some((prior, true)))]);

        recording
            .push(bind(point, a, 0, vec![unrelated, unrelated]))
            .unwrap();
        record_owned(&mut recording, point, b, 1, vec![]).unwrap();
        let RecordedCommand::PushDescriptorWrites { history, .. } = &recording.commands[7] else {
            panic!()
        };
        assert_eq!(history, &vec![(a, Some((unrelated, false)))]);
    }

    #[test]
    fn partial_push_inherits_only_compatible_pushed_state_as_an_independent_snapshot() {
        let (mut resources, [a, b, incompatible], prior) = fixture();
        let history = vec![(b, None), (a, Some((prior, true)))];
        let mut inherited = descriptor(&resources, b, 1, &history).unwrap();
        assert_eq!(inherited.bindings.len(), 2);
        assert!(
            matches!(inherited.bindings[&0], DescriptorBinding::Buffer { buffer, .. } if buffer.as_raw() == 20)
        );
        inherited.bindings.insert(0, buffer(99));
        assert!(
            matches!(resources.descriptor_sets[&prior].bindings[&0], DescriptorBinding::Buffer { buffer, .. } if buffer.as_raw() == 20)
        );
        assert!(
            matches!(inherited.bindings[&1], DescriptorBinding::Buffer { buffer, .. } if buffer.as_raw() == 21)
        );
        for history in [
            vec![(incompatible, None), (a, Some((prior, true)))],
            vec![(a, Some((prior, false)))],
        ] {
            assert!(
                descriptor(&resources, b, 1, &history)
                    .unwrap()
                    .bindings
                    .is_empty()
            );
        }
        // Binding-map and visibility incompatibility also disturb partial state.
        resources.pipeline_binding_maps.get_mut(&a).unwrap()[1].insert(1, 6);
        assert!(
            descriptor(&resources, b, 1, &history)
                .unwrap()
                .bindings
                .is_empty()
        );
        resources
            .pipeline_binding_maps
            .insert(a, resources.pipeline_binding_maps[&b].clone());
        resources.pipeline_binding_stages.get_mut(&a).unwrap()[1]
            .insert(1, vk::ShaderStageFlags::VERTEX);
        assert!(
            descriptor(&resources, b, 1, &history)
                .unwrap()
                .bindings
                .is_empty()
        );
    }

    #[test]
    fn resolution_rejects_missing_generation_and_wrong_push_set() {
        let (mut resources, [a, b, _], prior) = fixture();
        let history = vec![(a, Some((prior, true)))];
        assert!(descriptor(&resources, b, 0, &history).is_err());
        resources.descriptor_sets.remove(&prior);
        assert!(descriptor(&resources, b, 1, &history).is_err());
        resources.pipeline_push_sets.remove(&b);
        assert!(descriptor(&resources, b, 1, &history).is_err());
    }

    #[test]
    fn recording_reuse_keeps_queued_push_owners_and_allocates_fresh_virtual_handles() {
        let mut recording = recording();
        let point = vk::PipelineBindPoint::COMPUTE;
        let layout = vk::PipelineLayout::from_raw(1);
        record_owned(&mut recording, point, layout, 0, vec![]).unwrap();
        let RecordedCommand::PushDescriptorWrites { handle, owner, .. } = &recording.commands[0]
        else {
            panic!()
        };
        let (handle, owner) = (*handle, owner.clone());
        recording.state = RecordingState::Executable;
        let queued = recording.clone();
        recording = Recording::new(0);
        recording.state = RecordingState::Recording;
        record_owned(&mut recording, point, layout, 0, vec![]).unwrap();
        let RecordedCommand::PushDescriptorWrites {
            handle: next,
            history,
            ..
        } = &recording.commands[0]
        else {
            panic!()
        };
        assert_ne!(*next, handle);
        assert!(history.is_empty());
        assert!(
            owner.upgrade().is_some(),
            "queued snapshot still owns its immutable descriptors"
        );
        drop(queued);
        assert!(owner.upgrade().is_none());
    }

    #[test]
    fn invalid_pushes_and_pair_capacity_are_checked_without_recording_half_a_push() {
        let mut recording = recording();
        let layout = vk::PipelineLayout::from_raw(1);
        for (point, set, writes) in [
            (vk::PipelineBindPoint::from_raw(-1), 0, vec![]),
            (vk::PipelineBindPoint::COMPUTE, 4, vec![]),
            (
                vk::PipelineBindPoint::COMPUTE,
                0,
                vec![Write {
                    set: vk::DescriptorSet::null(),
                    binding: 0,
                    element: 0,
                    ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                    value: Some(buffer(1)),
                }],
            ),
        ] {
            assert_eq!(
                record_owned(&mut recording, point, layout, set, writes),
                Err(vk::Result::ERROR_FEATURE_NOT_PRESENT)
            );
            assert!(recording.commands.is_empty());
        }
        Arc::make_mut(&mut recording.commands)
            .resize(ir::MAX_COMMANDS - 1, RecordedCommand::InertDynamicState);
        assert_eq!(
            record_owned(
                &mut recording,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                vec![]
            ),
            Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY)
        );
        assert_eq!(recording.commands.len(), ir::MAX_COMMANDS - 1);
        assert_eq!(recording.error, Some(vk::Result::ERROR_OUT_OF_HOST_MEMORY));
    }

    #[test]
    fn push_copies_caller_data_and_never_queues_a_worker_rpc() {
        let (sender, requests) = mpsc::channel();
        let driver = Arc::new(Driver {
            sender,
            thread: Mutex::new(None),
            completion_sender: mpsc::channel().0,
            completion_thread: Mutex::new(None),
            queue: AtomicU64::new(0),
            queue_sender: mpsc::channel().0,
            queue_thread: Mutex::new(None),
            timeline_enabled: false,
            synchronization2_enabled: false,
            dynamic_rendering_enabled: false,
            host_query_reset_enabled: false,
            timestamps: false,
            query_support: 0,
            occlusion_precise_enabled: false,
            pipeline_statistics_enabled: false,
            multi_draw_indirect_enabled: false,
            indirect_first_instance_enabled: false,
            indirect_count_enabled: false,
            queries: Default::default(),
            timelines: Default::default(),
            events: Default::default(),
            fences: Default::default(),
            semaphores: Default::default(),
            recordings: Default::default(),
            in_flight: Default::default(),
            pending: Default::default(),
            lost: Arc::new(AtomicBool::new(false)),
        });
        let command = vk::CommandBuffer::from_raw(add_handle(Kind::Command, &driver));
        let recording = Arc::new(Mutex::new(recording()));
        driver
            .recordings
            .lock()
            .unwrap()
            .commands
            .insert(command.as_raw(), recording.clone());
        let (done, returned) = mpsc::channel();
        let caller = std::thread::spawn(move || {
            let mut info = [vk::DescriptorBufferInfo::default()
                .buffer(vk::Buffer::from_raw(7))
                .range(16)];
            let writes = [vk::WriteDescriptorSet::default()
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&info)];
            unsafe {
                push(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    vk::PipelineLayout::from_raw(1),
                    0,
                    1,
                    writes.as_ptr(),
                );
            }
            info[0].buffer = vk::Buffer::from_raw(9);
            std::hint::black_box(&info);
            done.send(()).unwrap();
        });
        let returned = returned.recv_timeout(std::time::Duration::from_secs(5));
        let queued_rpc = requests.try_recv().is_ok();
        drop(requests);
        caller.join().unwrap();
        remove_handle(command.as_raw());
        assert!(
            returned.is_ok(),
            "recording must finish with an unserviced worker"
        );
        assert!(!queued_rpc);
        let recording = recording.lock().unwrap();
        assert!(recording.error.is_none());
        assert_eq!(recording.commands.len(), 2);
        let RecordedCommand::PushDescriptorWrites {
            writes,
            owner,
            handle,
            ..
        } = &recording.commands[0]
        else {
            panic!()
        };
        assert!(
            matches!(writes[0].value, Some(DescriptorBinding::Buffer { buffer, range: 16, .. }) if buffer.as_raw() == 7)
        );
        let RecordedCommand::BindDescriptorSets {
            sets,
            pushed: Some(strong),
            ..
        } = &recording.commands[1]
        else {
            panic!()
        };
        assert_eq!(sets, &vec![*handle]);
        assert!(owner.ptr_eq(&Arc::downgrade(strong)));
    }
}
