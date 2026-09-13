//! Incremental Vulkan push-constant state, snapshotted at each draw/dispatch.
use ash::vk;
use sgfx::ir;

const STAGES: [ir::ShaderStages; 3] = [
    ir::ShaderStages::VERTEX,
    ir::ShaderStages::FRAGMENT,
    ir::ShaderStages::COMPUTE,
];
const LENGTH: usize = ir::MAX_PUSH_CONSTANT_BYTES as usize;

#[derive(Clone)]
pub(crate) struct PushConstants {
    bytes: [[u8; LENGTH]; 3],
    owners: [[Option<usize>; LENGTH]; 3],
    layouts: Vec<Vec<ir::PushConstantRange>>,
}
impl Default for PushConstants {
    fn default() -> Self {
        Self {
            bytes: [[0; LENGTH]; 3],
            owners: [[None; LENGTH]; 3],
            layouts: Vec::new(),
        }
    }
}
impl PushConstants {
    pub fn update(
        &mut self,
        layout: &ir::PipelineLayoutDesc,
        stages: ir::ShaderStages,
        offset: u32,
        data: &[u8],
    ) -> Result<(), vk::Result> {
        layout
            .validate_push_constants(stages, offset, data)
            .map_err(crate::resources::failure)?;
        let ranges = layout.push_constant_ranges();
        let owner = match self.layouts.iter().position(|stored| stored == ranges) {
            Some(index) => index,
            None => {
                self.layouts.push(ranges.to_vec());
                self.layouts.len() - 1
            }
        };
        let range = offset as usize..offset as usize + data.len();
        for (index, stage) in STAGES.into_iter().enumerate() {
            if stages.contains(stage) {
                self.bytes[index][range.clone()].copy_from_slice(data);
                self.owners[index][range.clone()].fill(Some(owner));
            }
        }
        Ok(())
    }

    /// Split overlapping ranges at their boundaries so each update provides
    /// every stage declaring those bytes, as required by Vulkan and WGPU.
    pub fn snapshot(
        &self,
        layout: &ir::PipelineLayoutDesc,
    ) -> Result<Vec<ir::OwnedCommand>, vk::Result> {
        let ranges = layout.push_constant_ranges();
        let mut boundaries = ranges
            .iter()
            .flat_map(|range| [range.offset(), range.offset() + range.size()])
            .collect::<Vec<_>>();
        boundaries.sort_unstable();
        boundaries.dedup();
        let mut commands = Vec::new();
        for window in boundaries.windows(2) {
            let start = window[0];
            let end = window[1];
            let mut stages = ir::ShaderStages::empty();
            for range in ranges {
                if range.offset() <= start && end <= range.offset() + range.size() {
                    stages |= range.stages();
                }
            }
            if stages.is_empty() {
                continue;
            }
            let mut bytes = vec![0; (end - start) as usize];
            let mut known = vec![false; bytes.len()];
            for (index, stage) in STAGES.into_iter().enumerate() {
                if !stages.contains(stage) {
                    continue;
                }
                for byte in start as usize..end as usize {
                    if let Some(owner) = self.owners[index][byte] {
                        if self.layouts[owner] != ranges {
                            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                        }
                        let destination = byte - start as usize;
                        if known[destination] && bytes[destination] != self.bytes[index][byte] {
                            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
                        }
                        bytes[destination] = self.bytes[index][byte];
                        known[destination] = true;
                    }
                }
            }
            commands.push(ir::OwnedCommand::SetPushConstants {
                stages,
                offset: start,
                data: bytes,
            });
        }
        Ok(commands)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn layout(ranges: &[(ir::ShaderStages, u32, u32)]) -> ir::PipelineLayoutDesc {
        ir::PipelineLayoutDesc::new(vec![])
            .unwrap()
            .with_push_constant_ranges(
                ranges
                    .iter()
                    .map(|&(stage, offset, size)| {
                        ir::PushConstantRange::new(stage, offset, size).unwrap()
                    })
                    .collect(),
            )
            .unwrap()
    }
    fn values(commands: &[ir::OwnedCommand]) -> Vec<(ir::ShaderStages, u32, Vec<u8>)> {
        commands
            .iter()
            .map(|command| match command {
                ir::OwnedCommand::SetPushConstants {
                    stages,
                    offset,
                    data,
                } => (*stages, *offset, data.clone()),
                _ => panic!("push-constant snapshot contains another command"),
            })
            .collect()
    }
    #[test]
    fn snapshots_own_incremental_bytes_and_split_overlapping_stage_ranges() {
        let layout = layout(&[(STAGES[0], 0, 32), (STAGES[1], 0, 16)]);
        let mut state = PushConstants::default();
        state
            .update(&layout, STAGES[0] | STAGES[1], 0, &[1; 16])
            .unwrap();
        state.update(&layout, STAGES[0], 16, &[2; 16]).unwrap();
        let first = state.snapshot(&layout).unwrap();
        state
            .update(&layout, STAGES[0] | STAGES[1], 4, &[3; 4])
            .unwrap();
        assert_eq!(
            values(&first),
            vec![
                (STAGES[0] | STAGES[1], 0, vec![1; 16]),
                (STAGES[0], 16, vec![2; 16])
            ]
        );
        let mut updated = vec![1; 16];
        updated[4..8].fill(3);
        assert_eq!(
            values(&state.snapshot(&layout).unwrap()),
            vec![
                (STAGES[0] | STAGES[1], 0, updated),
                (STAGES[0], 16, vec![2; 16])
            ]
        );
        assert!(state.update(&layout, STAGES[0], 0, &[9; 4]).is_err());
    }
    #[test]
    fn incompatible_layouts_do_not_disturb_old_bytes_but_cannot_supply_active_ranges() {
        let first = layout(&[(STAGES[1], 0, 16)]);
        let second = layout(&[(STAGES[1], 16, 16)]);
        let conflicting = layout(&[(STAGES[1], 0, 32)]);
        let mut state = PushConstants::default();
        state.update(&first, STAGES[1], 0, &[4; 16]).unwrap();
        state.update(&second, STAGES[1], 16, &[5; 16]).unwrap();
        assert_eq!(
            values(&state.snapshot(&first).unwrap()),
            vec![(STAGES[1], 0, vec![4; 16])]
        );
        assert_eq!(
            values(&state.snapshot(&second).unwrap()),
            vec![(STAGES[1], 16, vec![5; 16])]
        );
        assert!(state.snapshot(&conflicting).is_err());
        assert!(state.snapshot(&layout(&[])).unwrap().is_empty());
    }
}
