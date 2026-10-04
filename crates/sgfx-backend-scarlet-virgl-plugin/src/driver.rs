//! Low-level resource/queue extension. All Rust ownership stays in this DSO.
use super::*;

struct Resources {
    inner: virgl::IrResources,
    context: virgl::Context,
    table: Rc<ir::ResourceTable>,
    source: u64,
    revision: u64,
    poisoned: bool,
}
struct Queue {
    inner: virgl::Queue,
    context: virgl::Context,
}

unsafe extern "C" fn create_resources(p: Object, metadata: Span<u64>, out: *mut Object) -> i32 {
    if out.is_null() {
        return abi::INVALID;
    }
    unsafe {
        out.write(core::ptr::null_mut());
    }
    status((|| {
        let context = unsafe { object::<virgl::Context>(p) }?;
        let table = Rc::new(ir::ResourceTable::new());
        let (source, revision) = table
            .sync_abi_snapshot(unsafe { span(metadata) }?)
            .map_err(|_| abi::INVALID)?;
        let inner = context.create_ir_resources(table.clone()).map_err(error)?;
        let resources = Resources {
            inner,
            context: context.clone(),
            table,
            source,
            revision,
            poisoned: false,
        };
        unsafe {
            out.write(Box::into_raw(Box::new(resources)).cast());
        }
        Ok(())
    })())
}
unsafe extern "C" fn drop_resources(p: Object) {
    if !p.is_null() {
        unsafe {
            drop(Box::from_raw(p.cast::<Resources>()));
        }
    }
}
unsafe extern "C" fn sync_resources(p: Object, metadata: Span<u64>) -> i32 {
    status((|| {
        let r = unsafe { object::<Resources>(p) }?;
        if r.poisoned {
            return Err(abi::DEVICE_LOST);
        }
        let words = unsafe { span(metadata) }?;
        if words.len() < 3 || words[1] != r.source || words[2] < r.revision {
            return Err(abi::INVALID);
        }
        if words[2] == r.revision {
            return Ok(());
        }
        let (textures, buffers) = r
            .table
            .abi_retired_resources(words)
            .map_err(|_| abi::INVALID)?;
        if (!textures.is_empty() || !buffers.is_empty())
            && !r.context.is_idle().map_err(handle_error)?
        {
            return Err(abi::BUSY);
        }
        for id in textures {
            r.inner.release_texture(id).map_err(error)?;
        }
        for id in buffers {
            r.inner.release_buffer(id).map_err(error)?;
        }
        match r.table.sync_abi_snapshot(words) {
            Ok((_, revision)) => r.revision = revision,
            Err(_) => {
                r.poisoned = true;
                return Err(abi::INVALID);
            }
        }
        Ok(())
    })())
}
unsafe extern "C" fn release_buffer(p: Object, slot: u32) -> i32 {
    status((|| {
        let r = unsafe { object::<Resources>(p) }?;
        if r.poisoned {
            return Err(abi::DEVICE_LOST);
        }
        let id = r.table.abi_buffer(slot).map_err(|_| abi::INVALID)?.id();
        // A client may release directly as well as via a metadata retirement.
        if !r.context.is_idle().map_err(handle_error)? {
            return Err(abi::BUSY);
        }
        r.inner.release_buffer(id).map_err(error)
    })())
}
unsafe extern "C" fn release_texture(p: Object, slot: u32) -> i32 {
    status((|| {
        let r = unsafe { object::<Resources>(p) }?;
        if r.poisoned {
            return Err(abi::DEVICE_LOST);
        }
        let id = r.table.abi_texture(slot).map_err(|_| abi::INVALID)?.id();
        if !r.context.is_idle().map_err(handle_error)? {
            return Err(abi::BUSY);
        }
        r.inner.release_texture(id).map_err(error)
    })())
}
unsafe extern "C" fn release_bind_group(p: Object, slot: u32) -> i32 {
    status((|| {
        let r = unsafe { object::<Resources>(p) }?;
        if r.poisoned {
            return Err(abi::DEVICE_LOST);
        }
        let id = r.table.abi_bind_group(slot).map_err(|_| abi::INVALID)?.id();
        r.inner.release_bind_group(id).map_err(error)
    })())
}
unsafe extern "C" fn validate(p: Object, kind: u32, slot: u32) -> i32 {
    status((|| {
        let r = unsafe { object::<Resources>(p) }?;
        if r.poisoned {
            return Err(abi::DEVICE_LOST);
        }
        match kind {
            abi::VALIDATE_SHADER => r.inner.validate_shader_module(
                r.table
                    .abi_shader_module(slot)
                    .map_err(|_| abi::INVALID)?
                    .id(),
            ),
            abi::VALIDATE_RENDER_PIPELINE => r.inner.validate_programmable_render_pipeline(
                r.table
                    .abi_programmable_render_pipeline(slot)
                    .map_err(|_| abi::INVALID)?
                    .id(),
            ),
            abi::VALIDATE_COMPUTE_PIPELINE => r.inner.validate_compute_pipeline(
                r.table
                    .abi_compute_pipeline(slot)
                    .map_err(|_| abi::INVALID)?
                    .id(),
            ),
            _ => return Err(abi::INVALID),
        }
        .map_err(error)
    })())
}
unsafe fn output<'a>(ptr: *mut u8, len: usize) -> Result<&'a mut [u8], i32> {
    if len == 0 {
        return Ok(&mut []);
    }
    if ptr.is_null() || len > isize::MAX as usize {
        return Err(abi::INVALID);
    }
    // SAFETY: the ABI caller promises exclusive initialized output storage.
    Ok(unsafe { core::slice::from_raw_parts_mut(ptr, len) })
}
unsafe extern "C" fn read_buffer(
    p: Object,
    slot: u32,
    offset: u64,
    out: *mut u8,
    len: usize,
) -> i32 {
    status((|| {
        let r = unsafe { object::<Resources>(p) }?;
        if r.poisoned {
            return Err(abi::DEVICE_LOST);
        }
        let id = r.table.abi_buffer(slot).map_err(|_| abi::INVALID)?.id();
        r.inner
            .read_buffer_into(id, offset, unsafe { output(out, len) }?)
            .map_err(error)
    })())
}
unsafe extern "C" fn create_image(
    p: Object,
    width: u32,
    height: u32,
    out: *mut Object,
    info: *mut abi::ImageInfo,
) -> i32 {
    if out.is_null() || info.is_null() {
        return abi::INVALID;
    }
    unsafe {
        out.write(core::ptr::null_mut());
        info.write(abi::ImageInfo {
            handle: -1,
            ..Default::default()
        });
    }
    status((|| {
        let context = unsafe { object::<virgl::Context>(p) }?;
        let image = context
            .create_shared_image(width, height)
            .map_err(handle_error)?;
        let handle = image.shared_handle().duplicate().map_err(handle_error)?;
        unsafe {
            info.write(abi::ImageInfo {
                width: image.width(),
                height: image.height(),
                handle: handle.as_raw(),
                reserved: 0,
            });
            out.write(Box::into_raw(Box::new(Rc::new(image))).cast());
        }
        std::mem::forget(handle);
        Ok(())
    })())
}
unsafe extern "C" fn drop_image(p: Object) {
    if !p.is_null() {
        unsafe {
            drop(Box::from_raw(p.cast::<Rc<virgl::Image>>()));
        }
    }
}
unsafe extern "C" fn map_image(p: Object, slot: u32, image: Object) -> i32 {
    status((|| {
        let r = unsafe { object::<Resources>(p) }?;
        if r.poisoned {
            return Err(abi::DEVICE_LOST);
        }
        let image = unsafe { object::<Rc<virgl::Image>>(image) }?;
        let id = r.table.abi_texture(slot).map_err(|_| abi::INVALID)?.id();
        r.inner.map_image(id, image.clone()).map_err(error)
    })())
}
unsafe extern "C" fn unmap_image(p: Object, slot: u32) -> i32 {
    status((|| {
        let r = unsafe { object::<Resources>(p) }?;
        if r.poisoned {
            return Err(abi::DEVICE_LOST);
        }
        let id = r.table.abi_texture(slot).map_err(|_| abi::INVALID)?.id();
        r.inner.unmap_image(id).map_err(error)
    })())
}
unsafe extern "C" fn create_queue(p: Object, out: *mut Object) -> i32 {
    if out.is_null() {
        return abi::INVALID;
    }
    unsafe {
        out.write(core::ptr::null_mut());
    }
    status((|| {
        let context = unsafe { object::<virgl::Context>(p) }?;
        let inner = context.create_queue().map_err(handle_error)?;
        unsafe {
            out.write(
                Box::into_raw(Box::new(Queue {
                    inner,
                    context: context.clone(),
                }))
                .cast(),
            );
        }
        Ok(())
    })())
}
unsafe extern "C" fn drop_queue(p: Object) {
    if !p.is_null() {
        unsafe {
            drop(Box::from_raw(p.cast::<Queue>()));
        }
    }
}
unsafe extern "C" fn submit(
    p: Object,
    resources: Object,
    batch: *const abi::Batch,
    out: *mut abi::SubmitResult,
) {
    if out.is_null() {
        return;
    }
    unsafe {
        out.write(abi::SubmitResult::default());
    }
    let result = (|| {
        let q = unsafe { object::<Queue>(p) }?;
        let r = unsafe { object::<Resources>(resources) }?;
        if r.poisoned {
            return Err(abi::DEVICE_LOST);
        }
        let batch = unsafe { batch.as_ref() }.ok_or(abi::INVALID)?;
        let commands =
            unsafe { ir::CommandBuffer::from_abi(&r.table, r.source, core_batch(*batch)) }
                .map_err(|_| abi::INVALID)?;
        let (disposition, error, receipt) =
            match q.inner.submit_ir_async(&q.context, &mut r.inner, &commands) {
                Ok(receipt) => (abi::ACCEPTED, abi::OK, receipt),
                Err(SubmitError::Busy) => return Err(abi::BUSY),
                Err(SubmitError::Rejected(e)) => return Err(error(e)),
                Err(SubmitError::Failed {
                    error: e,
                    completion,
                }) => (abi::PARTIAL, error(e), completion),
                Err(_) => return Err(abi::DEVICE_LOST),
            };
        Ok(abi::SubmitResult {
            disposition,
            error,
            receipt: receipt.into_abi_object(),
        })
    })();
    unsafe {
        out.write(match result {
            Ok(value) => value,
            Err(error) => abi::SubmitResult {
                error,
                ..Default::default()
            },
        });
    }
}
unsafe extern "C" fn read_texture(
    p: Object,
    resources: Object,
    slot: u32,
    out: *mut u8,
    len: usize,
) -> i32 {
    status((|| {
        let context = unsafe { object::<virgl::Context>(p) }?;
        let r = unsafe { object::<Resources>(resources) }?;
        if r.poisoned {
            return Err(abi::DEVICE_LOST);
        }
        let id = r.table.abi_texture(slot).map_err(|_| abi::INVALID)?.id();
        context
            .read_texture_into(&mut r.inner, id, unsafe { output(out, len) }?)
            .map_err(error)
    })())
}

unsafe extern "C" fn clone_receipt(p: Object) {
    if !p.is_null() {
        unsafe {
            virgl::Submission::clone_abi_object(p);
        }
    }
}

/// Negotiate the optional low-level v2 extension after the main entry point.
/// # Safety
/// `out` is writable for `size` bytes; all objects subsequently used come from
/// this DSO's main/extension tables and obey their ownership contracts.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sgfx_backend_get_driver_api_v2(
    version: u32,
    size: usize,
    out: *mut abi::DriverApi,
) -> i32 {
    if version != abi::VERSION || size < core::mem::size_of::<abi::DriverApi>() {
        return abi::ABI_MISMATCH;
    }
    if out.is_null() {
        return abi::INVALID;
    }
    unsafe {
        out.write(abi::DriverApi {
            version: abi::VERSION,
            size: core::mem::size_of::<abi::DriverApi>() as u32,
            create_resources,
            drop_resources,
            sync_resources,
            release_buffer,
            validate,
            read_buffer,
            create_image,
            drop_image,
            map_image,
            unmap_image,
            create_queue,
            drop_queue,
            submit,
            read_texture,
            clone_receipt,
            release_texture,
            release_bind_group,
        });
    }
    abi::OK
}
