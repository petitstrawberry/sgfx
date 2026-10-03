//! States which have no effect on this pipeline subset.
//! Constant blend factors, stencil tests and nonzero depth bias are rejected;
//! supported triangles use unit-width rasterization and zero depth bias.
use super::*;
fn inert(command: vk::CommandBuffer) {
    record(command, RecordedCommand::InertDynamicState);
}
unsafe extern "system" fn line(command: vk::CommandBuffer, width: f32) {
    if width == 1.0 {
        inert(command);
    } else {
        record_error(command, vk::Result::ERROR_FEATURE_NOT_PRESENT);
    }
}
unsafe extern "system" fn bias(command: vk::CommandBuffer, constant: f32, clamp: f32, slope: f32) {
    if constant == 0.0 && clamp == 0.0 && slope == 0.0 {
        inert(command);
    } else {
        record_error(command, vk::Result::ERROR_FEATURE_NOT_PRESENT);
    }
}
unsafe extern "system" fn blend(command: vk::CommandBuffer, values: *const [f32; 4]) {
    if values.is_null() {
        record_error(command, vk::Result::ERROR_INITIALIZATION_FAILED);
    } else {
        inert(command);
    }
}
unsafe extern "system" fn stencil(command: vk::CommandBuffer, faces: vk::StencilFaceFlags, _: u32) {
    if faces.is_empty() || !vk::StencilFaceFlags::FRONT_AND_BACK.contains(faces) {
        record_error(command, vk::Result::ERROR_INITIALIZATION_FAILED);
    } else {
        inert(command);
    }
}
pub(super) fn lookup(name: &CStr) -> vk::PFN_vkVoidFunction {
    macro_rules! entry {
        ($f:ident,$ty:ty) => {{
            let f: $ty = $f;
            Some(unsafe { std::mem::transmute::<$ty, unsafe extern "system" fn()>(f) })
        }};
    }
    match name.to_bytes() {
        b"vkCmdSetLineWidth" => entry!(line, vk::PFN_vkCmdSetLineWidth),
        b"vkCmdSetDepthBias" => entry!(bias, vk::PFN_vkCmdSetDepthBias),
        b"vkCmdSetBlendConstants" => entry!(blend, vk::PFN_vkCmdSetBlendConstants),
        b"vkCmdSetStencilReference" => entry!(stencil, vk::PFN_vkCmdSetStencilReference),
        b"vkCmdSetStencilCompareMask" => entry!(stencil, vk::PFN_vkCmdSetStencilCompareMask),
        b"vkCmdSetStencilWriteMask" => entry!(stencil, vk::PFN_vkCmdSetStencilWriteMask),
        _ => None,
    }
}
