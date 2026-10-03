//! Transfer writes retain their source bytes at recording time and execute on
//! the GPU in command order, rather than modifying mapped memory immediately.
use super::*;
const INVALID: vk::Result = vk::Result::ERROR_INITIALIZATION_FAILED;
pub(super) fn apply_write(
    resources: &crate::resources::Resources,
    rec: &mut ResolvedRecording,
    buffer: vk::Buffer,
    offset: u64,
    data: Vec<u8>,
) -> VkResult<()> {
    let dst = resources.buffers.get(&buffer).ok_or(INVALID)?;
    if rec.render.is_some()
        || dst.bound.is_none()
        || !dst.usage.contains(vk::BufferUsageFlags::TRANSFER_DST)
        || data.is_empty()
        || !offset.is_multiple_of(4)
        || !data.len().is_multiple_of(4)
        || offset
            .checked_add(data.len() as u64)
            .is_none_or(|end| end > dst.size)
    {
        return Err(INVALID);
    }
    rec.ops.push(ir::OwnedCommand::WriteBuffer {
        buffer: dst.id,
        offset,
        data,
    });
    rec.used_buffers.push(buffer);
    rec.readback_buffers.push(buffer);
    rec.written_buffers.push(buffer);
    Ok(())
}
pub(super) fn fill_data(buffer_size: u64, offset: u64, size: u64, word: u32) -> VkResult<Vec<u8>> {
    if offset >= buffer_size || !offset.is_multiple_of(4) {
        return Err(INVALID);
    }
    let size = if size == vk::WHOLE_SIZE {
        (buffer_size - offset) & !3
    } else {
        size
    };
    if size == 0
        || !size.is_multiple_of(4)
        || offset.checked_add(size).is_none_or(|end| end > buffer_size)
    {
        return Err(INVALID);
    }
    let size = usize::try_from(size).map_err(|_| vk::Result::ERROR_OUT_OF_HOST_MEMORY)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(size)
        .map_err(|_| vk::Result::ERROR_OUT_OF_HOST_MEMORY)?;
    bytes.resize(size, 0);
    for chunk in bytes.chunks_exact_mut(4) {
        chunk.copy_from_slice(&word.to_ne_bytes());
    }
    Ok(bytes)
}
unsafe extern "system" fn update_buffer(
    command: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: u64,
    size: u64,
    data: *const std::ffi::c_void,
) {
    if data.is_null()
        || size == 0
        || size > 65536
        || !size.is_multiple_of(4)
        || !offset.is_multiple_of(4)
    {
        record_error(command, INVALID);
        return;
    }
    let bytes = std::slice::from_raw_parts(data.cast::<u8>(), size as usize).to_vec();
    record(
        command,
        RecordedCommand::UpdateBuffer {
            buffer,
            offset,
            data: bytes,
        },
    );
}
unsafe extern "system" fn fill_buffer(
    command: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: u64,
    size: u64,
    word: u32,
) {
    if !offset.is_multiple_of(4) || size == 0 || (size != vk::WHOLE_SIZE && !size.is_multiple_of(4))
    {
        record_error(command, INVALID);
        return;
    }
    record(
        command,
        RecordedCommand::FillBuffer {
            buffer,
            offset,
            size,
            word,
        },
    );
}
pub(super) fn lookup(name: &CStr) -> vk::PFN_vkVoidFunction {
    macro_rules! entry {
        ($f:path,$ty:ty) => {{
            let function: $ty = $f;
            Some(unsafe { std::mem::transmute::<$ty, unsafe extern "system" fn()>(function) })
        }};
    }
    match name.to_bytes() {
        b"vkCmdUpdateBuffer" => entry!(update_buffer, vk::PFN_vkCmdUpdateBuffer),
        b"vkCmdFillBuffer" => entry!(fill_buffer, vk::PFN_vkCmdFillBuffer),
        _ => None,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fill_whole_size_honors_offset_and_word_order() {
        assert_eq!(
            fill_data(15, 4, vk::WHOLE_SIZE, 0x12345678).unwrap(),
            [0x12345678u32.to_ne_bytes(), 0x12345678u32.to_ne_bytes()].concat()
        );
        assert!(fill_data(16, 2, 4, 0).is_err());
        assert!(fill_data(16, 12, 8, 0).is_err());
        assert!(fill_data(16, 16, vk::WHOLE_SIZE, 0).is_err());
        assert!(fill_data(16, 0, 3, 0).is_err());
    }
}
