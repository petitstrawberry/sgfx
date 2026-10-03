use sgfx_core::ir::*;
use std::collections::BTreeSet;

fn empty_group(table: &ResourceTable) -> BindGroupDesc {
    BindGroupDesc::new(table, BindGroupLayoutDesc::new(vec![]).unwrap(), vec![]).unwrap()
}

#[test]
fn release_invalidates_ids_and_copied_references_before_and_after_slot_reuse() {
    let table = ResourceTable::new();
    let desc = empty_group(&table);
    let original = table.define_bind_group(desc.clone()).unwrap();
    let copied = original;
    let id = original.id();
    let retained_metadata = table.bind_group_shared(original).unwrap();

    table.release_bind_group(id).unwrap();
    assert_eq!(table.bind_group_ref(id), Err(Error::InvalidDescriptor));
    assert_eq!(table.bind_group(copied), Err(Error::InvalidDescriptor));
    assert_eq!(
        table.bind_group_shared(copied),
        Err(Error::InvalidDescriptor)
    );
    assert_eq!(*retained_metadata, desc);

    let sampler = table
        .define_sampler(SamplerDesc::new(
            FilterMode::Nearest,
            FilterMode::Nearest,
            AddressMode::ClampToEdge,
            AddressMode::ClampToEdge,
        ))
        .unwrap();
    let replacement_desc = BindGroupDesc::new(
        &table,
        BindGroupLayoutDesc::new(vec![BindGroupLayoutEntry::new(
            0,
            ShaderStages::COMPUTE,
            BindingType::Sampler,
        )])
        .unwrap(),
        vec![BindGroupEntry::new(
            0,
            BindingResource::Sampler(sampler.id()),
        )],
    )
    .unwrap();
    let replacement = table.define_bind_group(replacement_desc.clone()).unwrap();
    assert_eq!(replacement.slot(), original.slot());
    assert_ne!(replacement.id(), id);
    assert_ne!(replacement, copied);
    assert_eq!(table.bind_group_ref(id), Err(Error::InvalidDescriptor));
    assert_eq!(table.bind_group(copied), Err(Error::InvalidDescriptor));
    assert_eq!(
        table.bind_group_shared(copied),
        Err(Error::InvalidDescriptor)
    );
    assert_eq!(table.bind_group(replacement), Ok(replacement_desc));
    assert_eq!(*retained_metadata, desc);
}

#[test]
fn foreign_and_repeated_release_leave_live_groups_unchanged() {
    let table = ResourceTable::new();
    let foreign = ResourceTable::new();
    let desc = empty_group(&table);
    let original = table.define_bind_group(desc.clone()).unwrap();
    let other = foreign.define_bind_group(empty_group(&foreign)).unwrap();
    assert_eq!(original.slot(), other.slot());

    assert_eq!(
        table.release_bind_group(other.id()),
        Err(Error::ResourceTableMismatch)
    );
    assert_eq!(table.bind_group(original), Ok(desc.clone()));
    assert!(foreign.bind_group_ref(other.id()).is_ok());

    table.release_bind_group(original.id()).unwrap();
    assert_eq!(
        table.release_bind_group(original.id()),
        Err(Error::InvalidDescriptor)
    );
    let replacement = table.define_bind_group(desc.clone()).unwrap();
    assert_eq!(replacement.slot(), original.slot());
    assert_eq!(
        table.release_bind_group(original.id()),
        Err(Error::InvalidDescriptor)
    );
    assert_eq!(table.bind_group(replacement), Ok(desc));
    assert!(foreign.bind_group_ref(other.id()).is_ok());
}

#[test]
fn repeated_create_and_release_does_not_exhaust_definition_capacity() {
    let table = ResourceTable::new();
    let desc = empty_group(&table);
    let first = table.define_bind_group(desc.clone()).unwrap();
    let mut current = first;
    for _ in 0..MAX_BIND_GROUP_DEFINITIONS * 2 + 1 {
        table.release_bind_group(current.id()).unwrap();
        let next = table.define_bind_group(desc.clone()).unwrap();
        assert_eq!(next.slot(), first.slot());
        assert_ne!(next.id(), current.id());
        assert_ne!(next, current);
        assert_eq!(
            table.bind_group_ref(current.id()),
            Err(Error::InvalidDescriptor)
        );
        current = next;
    }
    assert_eq!(
        table.bind_group_ref(first.id()),
        Err(Error::InvalidDescriptor)
    );
    assert_eq!(table.bind_group(current), Ok(desc));
}

#[test]
fn definition_capacity_counts_live_groups_and_reuses_all_released_slots() {
    let table = ResourceTable::new();
    let desc = empty_group(&table);
    let original: Vec<_> = (0..MAX_BIND_GROUP_DEFINITIONS)
        .map(|_| table.define_bind_group(desc.clone()).unwrap())
        .collect();
    assert_eq!(
        table.define_bind_group(desc.clone()),
        Err(Error::ResourceLimitExceeded)
    );

    let mut free_slots = BTreeSet::new();
    for group in original.iter().step_by(2) {
        table.release_bind_group(group.id()).unwrap();
        free_slots.insert(group.slot());
    }
    let replacement_count = free_slots.len();
    for _ in 0..replacement_count {
        let replacement = table.define_bind_group(desc.clone()).unwrap();
        assert!(free_slots.remove(&replacement.slot()));
        assert_eq!(table.bind_group(replacement), Ok(desc.clone()));
    }
    assert!(free_slots.is_empty());
    assert_eq!(
        table.define_bind_group(desc.clone()),
        Err(Error::ResourceLimitExceeded)
    );
    for (index, group) in original.iter().enumerate() {
        if index % 2 == 0 {
            assert_eq!(
                table.bind_group_ref(group.id()),
                Err(Error::InvalidDescriptor)
            );
        } else {
            assert_eq!(table.bind_group(*group), Ok(desc.clone()));
        }
    }
}

#[test]
fn recorded_commands_reject_released_bind_groups_after_slot_reuse() {
    let table = ResourceTable::new();
    let desc = empty_group(&table);
    let group = table.define_bind_group(desc.clone()).unwrap();
    let shader = table
        .define_shader_module(
            ShaderModuleDesc::wgsl("@compute @workgroup_size(1) fn main() {}".into()).unwrap(),
        )
        .unwrap();
    let pipeline = table
        .define_compute_pipeline(
            ComputePipelineDesc::new(
                ShaderEntryPoint::new(shader, ShaderStage::Compute, "main".into()).unwrap(),
                PipelineLayoutDesc::new(vec![desc.layout().clone()]).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    let recording = |id| {
        OwnedCommandBuffer::new(vec![
            OwnedCommand::BeginComputePass,
            OwnedCommand::SetComputePipeline(pipeline.id()),
            OwnedCommand::SetBindGroup {
                index: 0,
                bind_group: id,
            },
            OwnedCommand::Dispatch { x: 1, y: 1, z: 1 },
            OwnedCommand::EndComputePass,
        ])
    };
    let stale_recording = recording(group.id());
    assert_eq!(stale_recording.validate(&table), Ok(()));

    table.release_bind_group(group.id()).unwrap();
    let replacement = table.define_bind_group(desc).unwrap();
    assert_eq!(replacement.slot(), group.slot());
    assert_eq!(
        stale_recording.validate(&table),
        Err(Error::InvalidDescriptor)
    );
    assert_eq!(recording(replacement.id()).validate(&table), Ok(()));

    let mut encoder = CommandEncoder::new(&table);
    let mut pass = encoder.begin_compute_pass().unwrap();
    pass.set_pipeline(pipeline).unwrap();
    assert_eq!(pass.set_bind_group(0, group), Err(Error::InvalidDescriptor));
    pass.set_bind_group(0, replacement).unwrap();
    pass.dispatch(1, 1, 1).unwrap();
    pass.end().unwrap();
    encoder.finish().unwrap();
}
