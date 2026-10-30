//! VirGL query objects own result backing until GPU retirement and teardown.
use super::virgl::*;
use alloc::{rc::Rc, vec::Vec};
use core::cell::RefCell;
use gpu_raw::{GpuBuffer, GpuContext};
use scarlet_os::handle::capability::memory_mapping::{MemoryMappingOps, flags, prot};
use scarlet_os::handle::{HandleError, HandleResult};
pub(crate) struct Counter {
    raw: Rc<GpuContext>,
    buffer: GpuBuffer,
    handle: u32,
    resource: u32,
    address: usize,
    active: bool,
}
impl Drop for Counter {
    fn drop(&mut self) {
        let mut words = Vec::new();
        if self.active {
            push_dword(&mut words, command_header(20, 0, 1));
            push_dword(&mut words, self.handle);
        }
        push_destroy_object(&mut words, 9, self.handle);
        let destroyed = self
            .raw
            .create_queue()
            .and_then(|q| q.submit(&words))
            .is_ok();
        unsafe {
            let _ = MemoryMappingOps::munmap(self.address, 4096);
        }
        // A failed destruction can leave a host query referencing this backing.
        // Keep the kernel context attachment until that context is destroyed.
        if destroyed {
            let _ = self.raw.detach_buffer(&self.buffer);
        }
    }
}
pub(crate) type Scopes = RefCell<[Option<Vec<Counter>>; 2]>;
impl Context {
    pub(crate) fn begin_query(&self, kind: u32, mask: u32) -> HandleResult<()> {
        if kind > 1 || (kind == 1 && (mask == 0 || mask & !0x7ff != 0)) {
            return Err(HandleError::InvalidParameter);
        }
        wait_scheduled(&self.scheduler)?;
        let mut scopes = self.queries.borrow_mut();
        if scopes[kind as usize].is_some() {
            return Err(HandleError::InvalidParameter);
        }
        let indices: Vec<_> = if kind == 0 {
            alloc::vec![0]
        } else {
            (0..11).filter(|i| mask & (1 << i) != 0).collect()
        };
        let mut counters = Vec::new();
        for index in indices {
            let handle = self.allocate_object_handle()?;
            let buffer = self.device.raw.create_buffer(
                4096,
                gpu_raw::GPU_BUFFER_FLAG_CPU_VISIBLE | gpu_raw::GPU_BUFFER_FLAG_QUERY_RESULT,
            )?;
            let resource = resource_id_from_token(self.raw.attach_buffer(&buffer)?)?;
            let address = match buffer.as_handle().as_memory_mapping().and_then(|m| {
                unsafe { m.mmap(0, 4096, prot::READ | prot::WRITE, flags::SHARED, 0) }
                    .map_err(|_| HandleError::OutOfResources)
            }) {
                Ok(p) => p,
                Err(e) => {
                    let _ = self.raw.detach_buffer(&buffer);
                    return Err(e);
                }
            };
            let mut counter = Counter {
                raw: self.raw.clone(),
                buffer,
                handle,
                resource,
                address,
                active: false,
            };
            let mut words = Vec::new();
            push_dword(&mut words, command_header(1, 9, 4));
            // Protocol query type 1 is any-samples-passed; 10 is a selected
            // pipeline-statistics counter. Vulkan and VirGL share index order.
            for value in [
                handle,
                if kind == 0 { 1 } else { 10 | (index << 16) },
                0,
                resource,
            ] {
                push_dword(&mut words, value);
            }
            push_dword(&mut words, command_header(19, 0, 1));
            push_dword(&mut words, handle);
            self.raw.create_queue()?.submit(&words)?;
            counter.active = true;
            counters.push(counter);
        }
        scopes[kind as usize] = Some(counters);
        Ok(())
    }
    pub(crate) fn end_query(&self, kind: u32, count: usize) -> HandleResult<Vec<u64>> {
        if kind > 1 {
            return Err(HandleError::InvalidParameter);
        }
        wait_scheduled(&self.scheduler)?;
        let mut counters = self.queries.borrow_mut()[kind as usize]
            .take()
            .ok_or(HandleError::InvalidParameter)?;
        if counters.len() != count {
            return Err(HandleError::InvalidParameter);
        }
        let mut result = Vec::new();
        for counter in &mut counters {
            let mut words = Vec::new();
            push_dword(&mut words, command_header(20, 0, 1));
            push_dword(&mut words, counter.handle);
            push_dword(&mut words, command_header(21, 0, 2));
            push_dword(&mut words, counter.handle);
            push_dword(&mut words, 1);
            self.raw.create_queue()?.submit(&words)?;
            counter.active = false;
            let mut done = false;
            for _ in 0..100 {
                let mut transfer = Vec::new();
                push_dword(&mut transfer, command_header(43, 0, 13));
                for word in [counter.resource, 0, 0, 0, 0, 0, 0, 0, 16, 1, 1, 0, 2] {
                    push_dword(&mut transfer, word);
                }
                self.raw.create_queue()?.submit(&transfer)?;
                if unsafe { core::ptr::read_volatile(counter.address as *const u32) } == 1 {
                    done = true;
                    break;
                }
                std::thread::sleep(core::time::Duration::from_millis(1));
            }
            if !done {
                return Err(HandleError::Unsupported);
            }
            result.push(unsafe { core::ptr::read_volatile((counter.address + 8) as *const u64) });
        }
        Ok(result)
    }
}
