//! Compute system values and robust byte-addressed SSBO operations.
use super::*;

impl Compiler<'_> {
    pub(super) fn compute_input(
        &mut self,
        shape: Shape,
        binding: &Binding,
        input: bool,
    ) -> Result<Value> {
        if !input {
            return Err(unsupported("compute output"));
        }
        let Binding::BuiltIn(builtin) = binding else {
            return Err(unsupported("compute location input"));
        };
        let system = |compiler: &mut Self, index, semantic: &str| {
            let declaration = format!("DCL SV[{index}], {semantic}");
            if !compiler.declarations.contains(&declaration) {
                compiler.declarations.push(declaration);
            }
            Value {
                shape: Shape::Vector(K::Uint, 3),
                lanes: (0..3)
                    .map(|component| Lane {
                        register: format!("SV[{index}]"),
                        component,
                        constant_byte: None,
                    })
                    .collect(),
                writable: false,
                indirect: None,
            }
        };
        match builtin {
            BuiltIn::WorkGroupId => Ok(system(self, 0, "BLOCK_ID")),
            BuiltIn::NumWorkGroups => Ok(system(self, 1, "GRID_SIZE")),
            BuiltIn::LocalInvocationId => Ok(system(self, 2, "THREAD_ID")),
            BuiltIn::WorkGroupSize => {
                let mut lanes = Vec::new();
                for size in self.workgroup_size {
                    lanes.push(self.immediate(Literal::U32(size))?.lanes[0].clone());
                }
                Ok(Value {
                    shape,
                    lanes,
                    writable: false,
                    indirect: None,
                })
            }
            BuiltIn::GlobalInvocationId => {
                let block = system(self, 0, "BLOCK_ID");
                let thread = system(self, 2, "THREAD_ID");
                let result = self.allocate(shape, false)?;
                for i in 0..3 {
                    let size = self.immediate(Literal::U32(self.workgroup_size[i]))?;
                    self.instruction("UMUL", &result.lanes[i], &[&block.lanes[i], &size.lanes[0]])?;
                    self.instruction(
                        "UADD",
                        &result.lanes[i],
                        &[&result.lanes[i], &thread.lanes[i]],
                    )?;
                }
                Ok(result)
            }
            BuiltIn::LocalInvocationIndex => {
                let thread = system(self, 2, "THREAD_ID");
                let result = self.allocate(shape, false)?;
                let width = self.immediate(Literal::U32(self.workgroup_size[0]))?;
                let height = self.immediate(Literal::U32(self.workgroup_size[1]))?;
                let dst = &result.lanes[0];
                self.instruction("UMUL", dst, &[&thread.lanes[2], &height.lanes[0]])?;
                self.instruction("UADD", dst, &[dst, &thread.lanes[1]])?;
                self.instruction("UMUL", dst, &[dst, &width.lanes[0]])?;
                self.instruction("UADD", dst, &[dst, &thread.lanes[0]])?;
                Ok(result)
            }
            _ => Err(unsupported("compute builtin")),
        }
    }

    /// Saturate invalid indexing rather than allowing 32-bit arithmetic to wrap
    /// an out-of-bounds access back into a valid descriptor or adjacent member.
    pub(super) fn check_compute_index(
        &mut self,
        inner: &T,
        constant: Option<u32>,
        index: Option<&Value>,
        stride: u32,
        base: &Lane,
        address: &Lane,
    ) -> Result<()> {
        let valid = self.allocate(Shape::Scalar(K::Bool), false)?;
        let invalid = self.immediate(Literal::U32(u32::MAX))?;
        self.instruction("USGE", &valid.lanes[0], &[address, base])?;
        self.instruction(
            "UCMP",
            address,
            &[&valid.lanes[0], address, &invalid.lanes[0]],
        )?;
        if stride == 0 {
            return Ok(());
        }
        let limit = match inner {
            T::Array {
                size: naga::ArraySize::Constant(n),
                ..
            } => n.get(),
            T::Vector { size, .. } => *size as u32,
            T::Matrix { columns, .. } => *columns as u32,
            _ => u32::MAX / stride,
        }
        .min(u32::MAX / stride);
        let index = match constant {
            Some(i) => self.immediate(Literal::U32(i))?,
            None => index.ok_or_else(|| unsupported("compute index"))?.clone(),
        };
        let limit = self.immediate(Literal::U32(limit))?;
        self.instruction("USLT", &valid.lanes[0], &[&index.lanes[0], &limit.lanes[0]])?;
        self.instruction(
            "UCMP",
            address,
            &[&valid.lanes[0], address, &invalid.lanes[0]],
        )?;
        Ok(())
    }

    fn compute_address(
        &mut self,
        metadata: u32,
        offset: &Lane,
        extra: u32,
    ) -> Result<(Lane, Lane)> {
        let relative = self.allocate(Shape::Scalar(K::Uint), false)?.lanes[0].clone();
        let address = self.allocate(Shape::Scalar(K::Uint), false)?.lanes[0].clone();
        let valid = self.allocate(Shape::Scalar(K::Bool), false)?.lanes[0].clone();
        let nowrap = self.allocate(Shape::Scalar(K::Bool), false)?.lanes[0].clone();
        let extra = self.immediate(Literal::U32(extra))?;
        let base = Lane {
            register: format!("CONST[{metadata}]"),
            component: 0,
            constant_byte: None,
        };
        let length = Lane {
            component: 1,
            ..base.clone()
        };
        self.instruction("UADD", &relative, &[offset, &extra.lanes[0]])?;
        self.instruction("USGE", &nowrap, &[&relative, offset])?;
        self.instruction("USLT", &valid, &[&relative, &length])?;
        self.instruction("AND", &valid, &[&valid, &nowrap])?;
        self.instruction("UADD", &address, &[&base, &relative])?;
        Ok((address, valid))
    }

    pub(super) fn compute_load(
        &mut self,
        inner: &T,
        slot: u32,
        metadata: u32,
        offset: &Lane,
    ) -> Result<Value> {
        let result = self.allocate(self.shape(inner)?, false)?;
        let mut offsets = Vec::new();
        self.storage_offsets(inner, 0, &mut offsets)?;
        let zero = self.immediate(Literal::U32(0))?;
        // TGSI LOAD's mask selects consecutive words in the source buffer, so
        // always load .x then move it into the compiler's packed scalar lane.
        self.temp_lanes = self.temp_lanes.div_ceil(4) * 4;
        let scratch = self.allocate(Shape::Vector(K::Uint, 4), false)?;
        for (destination, extra) in result.lanes.iter().zip(offsets) {
            let (address, valid) = self.compute_address(metadata, offset, extra)?;
            self.instruction("MOV", destination, &[&zero.lanes[0]])?;
            self.instructions.push(format!("UIF {}", valid.src()));
            self.instructions.push(format!(
                "LOAD {}.x, BUFFER[{slot}], {}",
                scratch.lanes[0].register,
                address.src()
            ));
            self.instruction("MOV", destination, &[&scratch.lanes[0]])?;
            self.instructions.push("ENDIF".into());
        }
        Ok(result)
    }

    pub(super) fn compute_store(
        &mut self,
        inner: &T,
        slot: u32,
        metadata: u32,
        offset: &Lane,
        value: &Value,
    ) -> Result<()> {
        let mut offsets = Vec::new();
        self.storage_offsets(inner, 0, &mut offsets)?;
        if offsets.len() != value.lanes.len() {
            return Err(unsupported("compute storage store layout"));
        }
        for (source, extra) in value.lanes.iter().zip(offsets) {
            let (address, valid) = self.compute_address(metadata, offset, extra)?;
            self.instructions.push(format!("UIF {}", valid.src()));
            self.instructions.push(format!(
                "STORE BUFFER[{slot}].x, {}, {}",
                address.src(),
                source.src()
            ));
            self.instructions.push("ENDIF".into());
        }
        Ok(())
    }
}
