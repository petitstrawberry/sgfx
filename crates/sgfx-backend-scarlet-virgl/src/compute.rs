//! Native VirGL shader buffers, fenced dispatch and coherent readback.
use crate::virgl::*;
use alloc::{rc::Rc, vec::Vec};
use gpu_raw::{GpuBuffer, GpuContext};
use scarlet_os::handle::capability::memory_mapping::{MemoryMappingOps, flags, prot};
use scarlet_os::handle::{HandleError, HandleResult};
use sgfx_codegen_virgl::programmable::{CompiledShader, compile_shader};
use sgfx_core::ir::{ShaderModuleDesc, ShaderStage};

pub(crate) struct Binding {
    pub slot: u32,
    pub buffer: usize,
    pub writable: bool,
}

struct Backing {
    context: Rc<GpuContext>,
    buffer: GpuBuffer,
    resource: u32,
    address: usize,
    mapped_size: usize,
    retired: bool,
}
impl Drop for Backing {
    fn drop(&mut self) {
        unsafe {
            let _ = MemoryMappingOps::munmap(self.address, self.mapped_size);
        }
        // An uncertain fence keeps its attachment until context destruction.
        if self.retired {
            let _ = self.context.detach_buffer(&self.buffer);
        }
    }
}

fn transfer(words: &mut Vec<u8>, resource: u32, size: u32, direction: u32) {
    push_dword(words, command_header(43, 0, 13));
    for word in [resource, 0, 0, 0, 0, 0, 0, 0, size, 1, 1, 0, direction] {
        push_dword(words, word);
    }
}

impl Device {
    pub(crate) fn supports_compute(self: &Rc<Self>) -> bool {
        if let Some(supported) = self.compute_supported.get() {
            return supported;
        }
        let supported = (|| -> HandleResult<bool> {
            let context = self.create_context()?;
            let source = ShaderModuleDesc::wgsl(alloc::string::String::from(
                "@group(0) @binding(0) var<storage,read_write> data:array<u32>;\n\
                 @compute @workgroup_size(4) fn main(@builtin(global_invocation_id) id:vec3<u32>) {\n\
                 data[id.x]=data[id.x]*3u+id.x*7u+6699u; }"
            )).map_err(|_| HandleError::InvalidParameter)?;
            let shader = compile_shader(&source, ShaderStage::Compute, "main")
                .map_err(|_| HandleError::Unsupported)?;
            let mut bytes = alloc::vec![alloc::vec![0;32]];
            for word in bytes[0].chunks_exact_mut(4) { word.copy_from_slice(&13u32.to_le_bytes()); }
            let mut accepted = false;
            context.dispatch_compute(&shader, &[0, 32, 0, 0], &mut bytes,
                &[Binding { slot:0, buffer:0, writable:true }], [2,1,1], &mut accepted)?;
            Ok(bytes[0].chunks_exact(4).enumerate().all(|(i, word)| {
                u32::from_le_bytes(word.try_into().unwrap()) == 13*3+i as u32*7+6699
            }))
        })().unwrap_or(false);
        self.compute_supported.set(Some(supported));
        supported
    }
}

impl Context {
    pub(crate) fn compute_transport_limit(&self) -> HandleResult<usize> {
        Ok(self.raw.create_queue()?.max_opaque_command_size() as usize)
    }
    pub(crate) fn compute_supported(&self) -> bool {
        self.device.supports_compute()
    }

    /// Inputs and output allocations belong to the caller. Temporary GPU
    /// buffers preserve aliases between descriptor ranges in one allocation.
    /// `accepted` is set before dispatch: an uncertain error cannot be retried
    /// as a rejected logical submission.
    pub(crate) fn dispatch_compute(
        &self,
        shader: &CompiledShader,
        constants: &[u32],
        buffers: &mut [Vec<u8>],
        bindings: &[Binding],
        groups: [u32; 3],
        accepted: &mut bool,
    ) -> HandleResult<()> {
        wait_scheduled(&self.scheduler)?;
        let size = shader.workgroup_size.ok_or(HandleError::InvalidParameter)?;
        if groups.iter().any(|&n| n == 0 || n > 65535) || bindings.len() > 4 {
            return Err(HandleError::InvalidParameter);
        }
        let queue = self.raw.create_queue()?;
        let mut backing = Vec::<(usize, Backing)>::new();
        backing
            .try_reserve_exact(bindings.len())
            .map_err(|_| HandleError::OutOfResources)?;
        for binding in bindings {
            if backing.iter().any(|(index, _)| *index == binding.buffer) {
                continue;
            }
            let bytes = buffers
                .get(binding.buffer)
                .ok_or(HandleError::InvalidParameter)?;
            if bytes.is_empty() || bytes.len() > 1024 * 1024 || !bytes.len().is_multiple_of(4) {
                return Err(HandleError::InvalidParameter);
            }
            let mapped_size = bytes.len().div_ceil(4096) * 4096;
            let buffer = self
                .device
                .raw
                .create_buffer(mapped_size as u64, gpu_raw::GPU_BUFFER_FLAG_CPU_VISIBLE)?;
            let resource = resource_id_from_token(self.raw.attach_buffer(&buffer)?)?;
            let address = match buffer.as_handle().as_memory_mapping().and_then(|mapping| {
                unsafe { mapping.mmap(0, mapped_size, prot::READ | prot::WRITE, flags::SHARED, 0) }
                    .map_err(|_| HandleError::OutOfResources)
            }) {
                Ok(address) => address,
                Err(error) => {
                    let _ = self.raw.detach_buffer(&buffer);
                    return Err(error);
                }
            };
            backing.push((
                binding.buffer,
                Backing {
                    context: self.raw.clone(),
                    buffer,
                    resource,
                    address,
                    mapped_size,
                    retired: true,
                },
            ));
            unsafe {
                core::ptr::copy_nonoverlapping(bytes.as_ptr(), address as *mut u8, bytes.len());
            }
        }
        let handle = self.allocate_object_handle()?;
        let mut words = Vec::new();
        words
            .try_reserve_exact(shader.tgsi.len() + constants.len() * 4 + 4096)
            .map_err(|_| HandleError::OutOfResources)?;
        push_programmable_shader(&mut words, handle, 5, &shader.tgsi)?;
        push_bind_shader(&mut words, 5, handle);
        push_constant_words(&mut words, 5, constants)?;
        for (index, buffer) in &backing {
            transfer(&mut words, buffer.resource, buffers[*index].len() as u32, 1);
        }
        for binding in bindings {
            let buffer = &backing
                .iter()
                .find(|(index, _)| *index == binding.buffer)
                .ok_or(HandleError::InvalidParameter)?
                .1;
            push_dword(&mut words, command_header(34, 0, 5));
            for value in [
                5,
                binding.slot,
                0,
                buffers[binding.buffer].len() as u32,
                buffer.resource,
            ] {
                push_dword(&mut words, value);
            }
        }
        push_dword(&mut words, command_header(37, 0, 8));
        for value in size.into_iter().chain(groups).chain([0, 0]) {
            push_dword(&mut words, value);
        }
        // Protocol Gallium barrier bits, not modern Gallium enum values.
        push_dword(&mut words, command_header(36, 0, 1));
        push_dword(&mut words, (1 << 1) | (1 << 12));
        for (index, buffer) in &backing {
            if bindings.iter().any(|b| b.buffer == *index && b.writable) {
                transfer(&mut words, buffer.resource, buffers[*index].len() as u32, 2);
            }
        }
        for binding in bindings {
            push_dword(&mut words, command_header(34, 0, 5));
            for value in [5, binding.slot, 0, 0, 0] {
                push_dword(&mut words, value);
            }
        }
        push_bind_shader(&mut words, 5, 0);
        push_destroy_object(&mut words, 4, handle);
        // Reject the complete packet before dispatch when it exceeds transport.
        if words.len() > queue.max_opaque_command_size() as usize {
            return Err(HandleError::Unsupported);
        }
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::Release);
        for (_, buffer) in &mut backing {
            buffer.retired = false;
        }
        *accepted = true;
        queue.submit(&words)?;
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::Acquire);
        for (index, buffer) in &mut backing {
            buffer.retired = true;
            if bindings.iter().any(|b| b.buffer == *index && b.writable) {
                let bytes = &mut buffers[*index];
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        buffer.address as *const u8,
                        bytes.as_mut_ptr(),
                        bytes.len(),
                    );
                }
            }
        }
        Ok(())
    }
}
