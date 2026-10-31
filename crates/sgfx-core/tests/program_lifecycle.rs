use sgfx_core::ir::*;

fn shader(table: &ResourceTable, serial: usize) -> ShaderModuleRef<'_> {
    table.define_shader_module(ShaderModuleDesc::wgsl(format!(
        "// {serial}\n@vertex fn vs()->@builtin(position) vec4<f32>{{return vec4<f32>(0.0);}} @fragment fn fs()->@location(0) vec4<f32>{{return vec4<f32>(1.0);}} @compute @workgroup_size(1) fn cs(){{}}"
    )).unwrap()).unwrap()
}
fn compute(module: ShaderModuleRef<'_>) -> ComputePipelineDesc {
    ComputePipelineDesc::new(
        ShaderEntryPoint::new(module, ShaderStage::Compute, "cs".into()).unwrap(),
        PipelineLayoutDesc::new(vec![]).unwrap(),
    )
    .unwrap()
}
fn graphics(module: ShaderModuleRef<'_>) -> ProgrammableRenderPipelineDesc {
    ProgrammableRenderPipelineDesc::new(
        ShaderEntryPoint::new(module, ShaderStage::Vertex, "vs".into()).unwrap(),
        ShaderEntryPoint::new(module, ShaderStage::Fragment, "fs".into()).unwrap(),
        PipelineLayoutDesc::new(vec![]).unwrap(),
        TextureFormat::Rgba8Unorm,
        None,
        PrimitiveTopology::TriangleList,
        BlendState::REPLACE,
        RasterState::new(CullMode::None, FrontFace::CounterClockwise),
    )
    .unwrap()
}

#[test]
fn program_slots_survive_1024_generations_and_reject_every_stale_identity_and_reference() {
    let table = ResourceTable::new();
    let mut previous = None;
    for serial in 0..1024 {
        let module = shader(&table, serial);
        let cp = table.define_compute_pipeline(compute(module)).unwrap();
        let gp = table
            .define_programmable_render_pipeline(graphics(module))
            .unwrap();
        assert_eq!((module.slot(), cp.slot(), gp.slot()), (0, 0, 0));
        assert_eq!(
            (module.generation(), cp.generation(), gp.generation()),
            (serial as u64, serial as u64, serial as u64)
        );
        if let Some((old_m, old_c, old_g)) = previous {
            assert_ne!(module.id(), old_m);
            assert!(table.shader_module_ref(old_m).is_err());
            assert!(table.compute_pipeline_ref(old_c).is_err());
            assert!(table.programmable_render_pipeline_ref(old_g).is_err());
        }
        let snapshot = table.programmable_render_pipeline_shared(gp).unwrap();
        let owned = OwnedCommandBuffer::new(vec![
            OwnedCommand::BeginComputePass,
            OwnedCommand::SetComputePipeline(cp.id()),
            OwnedCommand::EndComputePass,
        ]);
        owned.validate(&table).unwrap();
        #[cfg(feature = "backend-abi")]
        let revision = table.abi_revision();
        assert_eq!(
            table.release_shader_module(module.id()),
            Err(Error::InvalidDescriptor)
        );
        #[cfg(feature = "backend-abi")]
        assert_eq!(table.abi_revision(), revision);
        table.release_compute_pipeline(cp.id()).unwrap();
        assert_eq!(
            table.release_shader_module(module.id()),
            Err(Error::InvalidDescriptor)
        );
        table.release_programmable_render_pipeline(gp.id()).unwrap();
        table.release_shader_module(module.id()).unwrap();
        assert!(table.shader_module(module).is_err());
        assert!(table.compute_pipeline(cp).is_err());
        assert!(table.programmable_render_pipeline_shared(gp).is_err());
        assert_eq!(owned.validate(&table), Err(Error::InvalidDescriptor));
        assert_eq!(snapshot.vertex().module(), module.id());
        assert!(
            table
                .define_programmable_render_pipeline((*snapshot).clone())
                .is_err()
        );
        previous = Some((module.id(), cp.id(), gp.id()));
    }
}

#[test]
fn foreign_and_repeated_release_do_not_change_live_program_slots() {
    let table = ResourceTable::new();
    let foreign = ResourceTable::new();
    let module = shader(&table, 1);
    let cp = table.define_compute_pipeline(compute(module)).unwrap();
    let gp = table
        .define_programmable_render_pipeline(graphics(module))
        .unwrap();
    assert_eq!(
        foreign.release_shader_module(module.id()),
        Err(Error::ResourceTableMismatch)
    );
    assert_eq!(
        foreign.release_compute_pipeline(cp.id()),
        Err(Error::ResourceTableMismatch)
    );
    assert_eq!(
        foreign.release_programmable_render_pipeline(gp.id()),
        Err(Error::ResourceTableMismatch)
    );
    table.release_compute_pipeline(cp.id()).unwrap();
    assert_eq!(
        table.release_compute_pipeline(cp.id()),
        Err(Error::InvalidDescriptor)
    );
    table.release_programmable_render_pipeline(gp.id()).unwrap();
    table.release_shader_module(module.id()).unwrap();
    assert_eq!(
        table.release_shader_module(module.id()),
        Err(Error::InvalidDescriptor)
    );
    let replacement = shader(&table, 2);
    assert_eq!(replacement.slot(), module.slot());
    assert_ne!(replacement, module);
    assert!(table.shader_module(replacement).is_ok());
}

#[test]
fn graphics_reload_overlaps_two_240_pipeline_sets_and_reuses_only_retired_slots() {
    let table = ResourceTable::new();
    let module_a = shader(&table, 0);
    let module_b = shader(&table, 1);
    let allocate = |module| {
        (0..240)
            .map(|_| {
                table
                    .define_programmable_render_pipeline(graphics(module))
                    .unwrap()
            })
            .collect::<Vec<_>>()
    };
    let old = allocate(module_a);
    let live = allocate(module_b);
    assert_eq!(live.last().unwrap().slot(), 479);
    #[cfg(feature = "backend-abi")]
    let mirror = ResourceTable::new();
    #[cfg(feature = "backend-abi")]
    mirror
        .sync_abi_snapshot(&table.abi_snapshot().unwrap())
        .unwrap();
    for pipeline in &old {
        table
            .release_programmable_render_pipeline(pipeline.id())
            .unwrap();
    }
    table.release_shader_module(module_a.id()).unwrap();
    #[cfg(feature = "backend-abi")]
    {
        let snapshot = table.abi_snapshot().unwrap();
        let (_, _, retired) = mirror.abi_retired_programs(&snapshot).unwrap();
        assert_eq!(retired.len(), old.len());
        mirror.sync_abi_snapshot(&snapshot).unwrap();
    }
    let replacement_module = shader(&table, 2);
    let replacements = allocate(replacement_module);
    for pipeline in &replacements {
        let retired = old[pipeline.slot()];
        assert_eq!(pipeline.generation(), retired.generation() + 1);
        assert!(
            table
                .programmable_render_pipeline_ref(retired.id())
                .is_err()
        );
        assert!(table.programmable_render_pipeline_shared(retired).is_err());
        assert_eq!(
            table
                .programmable_render_pipeline_shared(*pipeline)
                .unwrap()
                .vertex()
                .module(),
            replacement_module.id()
        );
    }
    #[cfg(feature = "backend-abi")]
    mirror
        .sync_abi_snapshot(&table.abi_snapshot().unwrap())
        .unwrap();
    for pipeline in &live {
        assert_eq!(
            table
                .programmable_render_pipeline_ref(pipeline.id())
                .unwrap(),
            *pipeline
        );
        assert_eq!(pipeline.generation(), 0);
        assert_eq!(
            table
                .programmable_render_pipeline_shared(*pipeline)
                .unwrap()
                .vertex()
                .module(),
            module_b.id()
        );
        #[cfg(feature = "backend-abi")]
        assert_eq!(
            mirror
                .abi_programmable_render_pipeline(pipeline.slot() as u32)
                .unwrap()
                .generation(),
            0
        );
    }
    #[cfg(feature = "backend-abi")]
    for pipeline in &replacements {
        assert_eq!(
            mirror
                .abi_programmable_render_pipeline(pipeline.slot() as u32)
                .unwrap()
                .generation(),
            pipeline.generation()
        );
    }
    for pipeline in live.iter().chain(&replacements) {
        table
            .release_programmable_render_pipeline(pipeline.id())
            .unwrap();
    }
    table.release_shader_module(module_b.id()).unwrap();
    table
        .release_shader_module(replacement_module.id())
        .unwrap();
}

#[test]
fn unreleased_program_categories_enforce_their_declared_live_slot_limits() {
    for kind in 0..3 {
        let table = ResourceTable::new();
        let module = shader(&table, 0);
        let limit = [
            MAX_SHADER_MODULES,
            MAX_COMPUTE_PIPELINES,
            MAX_PROGRAMMABLE_RENDER_PIPELINES,
        ][kind];
        for index in 0..limit {
            match kind {
                0 if index > 0 => {
                    shader(&table, index);
                }
                1 => {
                    table.define_compute_pipeline(compute(module)).unwrap();
                }
                2 => {
                    table
                        .define_programmable_render_pipeline(graphics(module))
                        .unwrap();
                }
                _ => (),
            }
        }
        let result = match kind {
            0 => table
                .define_shader_module(ShaderModuleDesc::wgsl("overflow".into()).unwrap())
                .map(|_| ()),
            1 => table.define_compute_pipeline(compute(module)).map(|_| ()),
            _ => table
                .define_programmable_render_pipeline(graphics(module))
                .map(|_| ()),
        };
        assert_eq!(result, Err(Error::ResourceLimitExceeded));
    }
}

#[cfg(feature = "backend-abi")]
mod abi {
    use super::*;
    use sgfx_core::ir::abi::{Batch, Span};

    #[test]
    fn r3_retirement_scan_and_reused_program_slots_reject_old_recordings() {
        let host = ResourceTable::new();
        host.enable_abi_commands();
        let mirror = ResourceTable::new();
        for serial in 0..1024 {
            let module = shader(&host, serial);
            let cp = host.define_compute_pipeline(compute(module)).unwrap();
            let gp = host
                .define_programmable_render_pipeline(graphics(module))
                .unwrap();
            let mut encoder = CommandEncoder::new(&host);
            let mut pass = encoder.begin_compute_pass().unwrap();
            pass.set_pipeline(cp).unwrap();
            pass.end().unwrap();
            let commands = encoder.finish().unwrap();
            let (source, _) = mirror
                .sync_abi_snapshot(&host.abi_snapshot().unwrap())
                .unwrap();
            let old_module = mirror.abi_shader_module(module.slot() as u32).unwrap().id();
            let old_cp = mirror.abi_compute_pipeline(cp.slot() as u32).unwrap().id();
            let old_gp = mirror
                .abi_programmable_render_pipeline(gp.slot() as u32)
                .unwrap()
                .id();
            let batch = commands.abi_batch().unwrap();
            let imported = unsafe { CommandBuffer::from_abi(&mirror, source, batch) }.unwrap();
            assert!(imported.iter_commands().collect::<Result<Vec<_>>>().is_ok());
            host.release_compute_pipeline(cp.id()).unwrap();
            host.release_programmable_render_pipeline(gp.id()).unwrap();
            host.release_shader_module(module.id()).unwrap();
            let snapshot = host.abi_snapshot().unwrap();
            assert_eq!(
                mirror.abi_retired_programs(&snapshot).unwrap(),
                (vec![old_module], vec![old_cp], vec![old_gp])
            );
            mirror.sync_abi_snapshot(&snapshot).unwrap();
            assert_eq!(
                imported.iter_commands().collect::<Result<Vec<_>>>().err(),
                Some(Error::InvalidDescriptor)
            );
            assert!(mirror.shader_module_ref(old_module).is_err());
            assert!(mirror.compute_pipeline_ref(old_cp).is_err());
            assert!(mirror.programmable_render_pipeline_ref(old_gp).is_err());
        }
    }

    // Frozen R2 wire fixture: no generation/tombstone prefix on shader/compute
    // rows, and only the module/pipeline index in entry points/commands.
    fn legacy_compute_snapshot() -> Vec<u64> {
        let source = b"@compute @workgroup_size(1) fn main() {}";
        let mut shader = vec![1, source.len() as u64];
        for bytes in source.chunks(8) {
            let mut word = [0; 8];
            word[..bytes.len()].copy_from_slice(bytes);
            shader.push(u64::from_le_bytes(word));
        }
        let mut snapshot = vec![0x3252_5846_4753, 77, 1, 0, 0, 0, 0, 1, shader.len() as u64];
        snapshot.extend(shader);
        snapshot.extend([0, 1, 6, 0, 2, 4, 0x6e69_616d, 0, 0, 0]);
        snapshot
    }

    #[test]
    fn r2_snapshot_and_index_only_compute_commands_remain_compatible() {
        let mirror = ResourceTable::new();
        let snapshot = legacy_compute_snapshot();
        let (source, _) = mirror.sync_abi_snapshot(&snapshot).unwrap();
        assert_eq!(mirror.abi_shader_module(0).unwrap().generation(), 0);
        assert_eq!(mirror.abi_compute_pipeline(0).unwrap().generation(), 0);
        assert_eq!(
            mirror.abi_retired_programs(&snapshot).unwrap(),
            (vec![], vec![], vec![])
        );
        let words = [
            (1 << 32) | 19,
            (2 << 32) | 21,
            0,
            (4 << 32) | 22,
            1,
            1,
            1,
            (1 << 32) | 20,
        ];
        let imported = unsafe {
            CommandBuffer::from_abi(
                &mirror,
                source,
                Batch {
                    table: source,
                    words: Span::from_slice(&words),
                    count: 4,
                },
            )
        }
        .unwrap();
        let commands = imported
            .iter_commands()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert!(matches!(commands[1], Command::SetComputePipeline(p) if p.generation() == 0));
        mirror.sync_abi_snapshot(&snapshot).unwrap();
        let mut changed = snapshot;
        changed[11] ^= 1; // Mutate immutable WGSL payload in the existing R2 row.
        assert!(mirror.sync_abi_snapshot(&changed).is_err());
    }

    #[test]
    fn generation_overflow_does_not_retire_a_live_shader_or_bump_revision() {
        let host = ResourceTable::new();
        let module = shader(&host, 0);
        let mut snapshot = host.abi_snapshot().unwrap().to_vec();
        // Four empty categories precede the first shader row.
        snapshot[9] = u64::MAX;
        let mirror = ResourceTable::new();
        mirror.sync_abi_snapshot(&snapshot).unwrap();
        let module = mirror.abi_shader_module(module.slot() as u32).unwrap();
        let revision = mirror.abi_revision();
        assert_eq!(
            mirror.release_shader_module(module.id()),
            Err(Error::ResourceLimitExceeded)
        );
        assert_eq!(mirror.abi_revision(), revision);
        assert!(mirror.shader_module(module).is_ok());
    }

    #[test]
    fn snapshot_cannot_retire_a_shader_behind_unchanged_cached_pipeline_rows() {
        for graphics_pipeline in [false, true] {
            let host = ResourceTable::new();
            let module = shader(&host, 0);
            if graphics_pipeline {
                host.define_programmable_render_pipeline(graphics(module))
                    .unwrap();
            } else {
                host.define_compute_pipeline(compute(module)).unwrap();
            }
            let mut snapshot = host.abi_snapshot().unwrap().to_vec();
            let mirror = ResourceTable::new();
            mirror.sync_abi_snapshot(&snapshot).unwrap();
            // The host API forbids this operation. Inject a next-generation
            // shader tombstone while keeping an otherwise identical pipeline
            // row, exercising import's cached-row reference-integrity check.
            let old_row_length = snapshot[8] as usize;
            snapshot[8] = 2;
            snapshot.splice(9..9 + old_row_length, [1, 0]);
            assert!(mirror.sync_abi_snapshot(&snapshot).is_err());
        }
    }
}
