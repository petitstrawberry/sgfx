//! Regression checks for the experimental ICD's defensive error contract.
//!
//! Several calls deliberately violate Vulkan valid usage. They test this ICD's
//! explicit rejection and recovery behavior, and are not portable Vulkan usage
//! or a substitute for Vulkan validation layers or conformance testing.

use super::Context;
use ash::vk;

#[test]
#[ignore = "requires a built SGFX ICD and a native GPU adapter"]
fn invalid_recording_order_is_rejected_and_reset_recovers() {
    let context = Context::new();
    let device = &context.device;
    let command = context.command;
    let begin = vk::CommandBufferBeginInfo::default();

    // All handles belong to this context. The intentionally invalid recording
    // order below is supported only by the SGFX ICD's defensive error contract.
    unsafe {
        assert_eq!(
            device.end_command_buffer(command),
            Err(vk::Result::ERROR_INITIALIZATION_FAILED),
            "ending an initial command buffer must fail",
        );
        device
            .reset_command_buffer(command, vk::CommandBufferResetFlags::empty())
            .expect("reset must recover from ending before beginning");
        device
            .begin_command_buffer(command, &begin)
            .expect("begin after reset");
        assert_eq!(
            device.begin_command_buffer(command, &begin),
            Err(vk::Result::ERROR_INITIALIZATION_FAILED),
            "nested begin must fail",
        );
        device
            .end_command_buffer(command)
            .expect("a rejected nested begin must preserve the active recording");
        device
            .reset_command_buffer(command, vk::CommandBufferResetFlags::empty())
            .expect("reset after nested begin rejection");
        device
            .begin_command_buffer(command, &begin)
            .expect("begin before invalid render-pass command");
        device.cmd_end_render_pass(command);
        assert_eq!(
            device.end_command_buffer(command),
            Err(vk::Result::ERROR_INITIALIZATION_FAILED),
            "ending a nonexistent render pass must be reported by end",
        );

        device
            .reset_command_buffer(command, vk::CommandBufferResetFlags::empty())
            .expect("reset must clear a sticky recording error");
        device
            .begin_command_buffer(command, &begin)
            .expect("recording must restart after reset");
        device
            .end_command_buffer(command)
            .expect("a valid empty recording must succeed after reset");
    }
}

#[test]
#[ignore = "requires a built SGFX ICD and a native GPU adapter"]
fn unsupported_recording_features_return_exact_errors() {
    let context = Context::new();
    let device = &context.device;
    let command = context.command;
    let begin = vk::CommandBufferBeginInfo::default();

    // These unsupported calls deliberately test rejection through the direct
    // SGFX ICD entrypoints rather than through a validating Vulkan loader.
    unsafe {
        assert_eq!(
            device.begin_command_buffer(
                command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE),
            ),
            Err(vk::Result::ERROR_FEATURE_NOT_PRESENT),
            "render-pass continuation is outside this ICD's supported subset",
        );
        device
            .begin_command_buffer(command, &begin)
            .expect("rejected begin flags must leave the buffer reusable");
        device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[],
        );
        // A later, different error must not overwrite the first recording
        // error, because void Vulkan commands report failures only at end.
        device.cmd_end_render_pass(command);
        assert_eq!(
            device.end_command_buffer(command),
            Err(vk::Result::ERROR_FEATURE_NOT_PRESENT),
            "unsupported stages must remain the first sticky recording error",
        );

        device
            .reset_command_buffer(command, vk::CommandBufferResetFlags::empty())
            .expect("reset after unsupported barrier");
        device
            .begin_command_buffer(command, &begin)
            .expect("begin after unsupported barrier");
        // The ICD checks its dispatch-count limit before pipeline state.
        device.cmd_dispatch(command, 65_536, 1, 1);
        assert_eq!(
            device.end_command_buffer(command),
            Err(vk::Result::ERROR_FEATURE_NOT_PRESENT),
            "dispatch dimensions above 65535 must be rejected explicitly",
        );
        device
            .reset_command_buffer(command, vk::CommandBufferResetFlags::empty())
            .expect("reset after oversized dispatch");
        device
            .begin_command_buffer(command, &begin)
            .expect("begin after oversized dispatch");
        device
            .end_command_buffer(command)
            .expect("recording must recover after unsupported features");
    }
}

#[test]
#[ignore = "requires a built SGFX ICD and a native GPU adapter"]
fn fence_creation_timeout_reset_and_empty_submission() {
    let context = Context::new();
    let device = &context.device;

    // Fence handles stay alive until after every queue operation and wait.
    unsafe {
        let unsignaled = device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .expect("create unsignaled fence");
        assert_eq!(device.get_fence_status(unsignaled), Ok(false));
        assert_eq!(
            device.wait_for_fences(&[unsignaled], true, 0),
            Err(vk::Result::TIMEOUT),
            "a zero-timeout wait must not signal an unsignaled fence",
        );
        assert_eq!(device.get_fence_status(unsignaled), Ok(false));

        let signaled = device
            .create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )
            .expect("create signaled fence");
        assert_eq!(device.get_fence_status(signaled), Ok(true));
        device
            .wait_for_fences(&[signaled], true, 0)
            .expect("a signaled fence must satisfy a zero-timeout wait");
        device
            .reset_fences(&[signaled])
            .expect("reset an initially signaled fence");
        assert_eq!(device.get_fence_status(signaled), Ok(false));

        device
            .queue_submit(context.queue, &[], unsignaled)
            .expect("an empty queue submission must signal its fence");
        device
            .wait_for_fences(&[unsignaled], true, 0)
            .expect("empty submission completion must satisfy its fence");
        assert_eq!(device.get_fence_status(unsignaled), Ok(true));
        device
            .reset_fences(&[unsignaled])
            .expect("reset a fence after empty submission");
        assert_eq!(device.get_fence_status(unsignaled), Ok(false));

        device.destroy_fence(signaled, None);
        device.destroy_fence(unsignaled, None);
    }
}

#[test]
#[ignore = "requires a built SGFX ICD and a native GPU adapter"]
fn one_time_submission_requires_reset_and_rerecording() {
    let context = Context::new();
    let device = &context.device;
    let begin =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    // Resubmitting a retired ONE_TIME_SUBMIT recording deliberately exercises
    // the experimental ICD's defensive error contract for invalid usage.
    unsafe {
        let fence = device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .expect("create fence for one-time submission");
        device
            .begin_command_buffer(context.command, &begin)
            .expect("begin one-time recording");
        device
            .end_command_buffer(context.command)
            .expect("end empty one-time recording");
        context.submit(fence).expect("first one-time submission");
        device
            .wait_for_fences(&[fence], true, u64::MAX)
            .expect("wait for one-time submission to retire");
        assert_eq!(device.get_fence_status(fence), Ok(true));

        device
            .reset_fences(&[fence])
            .expect("reset fence after first submission");
        assert_eq!(
            context.submit(fence),
            Err(vk::Result::ERROR_INITIALIZATION_FAILED),
            "a completed one-time recording must not be submitted again",
        );
        assert_eq!(
            device.get_fence_status(fence),
            Ok(false),
            "a rejected resubmission must leave its fence unsignaled",
        );
        assert_eq!(
            device.wait_for_fences(&[fence], true, 0),
            Err(vk::Result::TIMEOUT),
        );

        context.reset();
        device
            .begin_command_buffer(context.command, &begin)
            .expect("begin a fresh one-time recording after reset");
        device
            .end_command_buffer(context.command)
            .expect("end fresh one-time recording");
        context
            .submit(fence)
            .expect("reset and rerecording must permit a new submission");
        device
            .wait_for_fences(&[fence], true, u64::MAX)
            .expect("wait for the fresh one-time recording");
        assert_eq!(device.get_fence_status(fence), Ok(true));
        device.destroy_fence(fence, None);
    }
}

#[test]
#[ignore = "requires a built SGFX ICD and a native GPU adapter"]
fn vertex_and_index_bindings_reject_wrong_usage_alignment_and_dead_handles() {
    use super::Storage;
    let context = Context::new();
    let vertex = Storage::with_usage(&context, vk::BufferUsageFlags::VERTEX_BUFFER);
    let index = Storage::with_usage(&context, vk::BufferUsageFlags::INDEX_BUFFER);
    let device = &context.device;
    let command = context.command;
    let begin = vk::CommandBufferBeginInfo::default();
    unsafe {
        for (buffer, offset) in [(index.buffer, 0), (vertex.buffer, super::BYTES)] {
            context.reset();
            device.begin_command_buffer(command, &begin).unwrap();
            device.cmd_bind_vertex_buffers(command, 0, &[buffer], &[offset]);
            device.end_command_buffer(command).unwrap();
            assert_eq!(
                context.submit(vk::Fence::null()),
                Err(vk::Result::ERROR_INITIALIZATION_FAILED)
            );
        }
        context.reset();
        device.begin_command_buffer(command, &begin).unwrap();
        device.cmd_bind_vertex_buffers(command, 0, &[vertex.buffer], &[2]);
        assert_eq!(
            device.end_command_buffer(command),
            Err(vk::Result::ERROR_INITIALIZATION_FAILED)
        );
        context.reset();
        device.begin_command_buffer(command, &begin).unwrap();
        device.cmd_bind_vertex_buffers(command, 1, &[vertex.buffer], &[0]);
        assert_eq!(
            device.end_command_buffer(command),
            Err(vk::Result::ERROR_FEATURE_NOT_PRESENT)
        );
        context.reset();
        device.begin_command_buffer(command, &begin).unwrap();
        device.cmd_bind_index_buffer(command, vertex.buffer, 0, vk::IndexType::UINT16);
        device.end_command_buffer(command).unwrap();
        assert_eq!(
            context.submit(vk::Fence::null()),
            Err(vk::Result::ERROR_INITIALIZATION_FAILED)
        );
        for (offset, format) in [(1, vk::IndexType::UINT16), (2, vk::IndexType::UINT32)] {
            context.reset();
            device.begin_command_buffer(command, &begin).unwrap();
            device.cmd_bind_index_buffer(command, index.buffer, offset, format);
            assert_eq!(
                device.end_command_buffer(command),
                Err(vk::Result::ERROR_INITIALIZATION_FAILED)
            );
        }
        context.reset();
        device.begin_command_buffer(command, &begin).unwrap();
        device.cmd_bind_vertex_buffers(command, 0, &[vertex.buffer], &[4]);
        device.cmd_bind_index_buffer(command, index.buffer, 2, vk::IndexType::UINT16);
        device.end_command_buffer(command).unwrap();
        context.submit(vk::Fence::null()).unwrap();
        drop(vertex);
        assert_eq!(
            context.submit(vk::Fence::null()),
            Err(vk::Result::ERROR_INITIALIZATION_FAILED)
        );
    }
}
