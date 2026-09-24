#![cfg(feature = "backend-abi")]

use sgfx_core::ir::*;

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};
thread_local! {
    static ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) };
}
struct CountingAllocator;
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|count| count.set(count.get().map(|n| n + 1)));
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCATIONS.with(|count| count.set(count.get().map(|n| n + 1)));
        unsafe { System.realloc(pointer, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn plugin_boundary_borrows_the_original_command_storage_and_upload_bytes() {
    let host = ResourceTable::new();
    host.enable_abi_commands();
    let buffer = host
        .define_buffer(BufferDesc::new(16, BufferUsage::COPY_DST).unwrap())
        .unwrap();
    let payload = [0x5a; 16];
    let mut encoder = CommandEncoder::new(&host);
    encoder.write_buffer(buffer, 0, &payload).unwrap();
    let recording = encoder.finish().unwrap();
    let batch = recording.abi_batch().unwrap();
    assert_eq!(batch.count, 1);
    // v1 WriteBuffer is exactly header, slot, generation, offset, address, length.
    let words = unsafe { batch.words.as_slice() };
    assert_eq!(words.len(), 6);
    assert_eq!(words[0], (6u64 << 32) | 1);
    assert_eq!(words[4], payload.as_ptr() as u64);
    let mirror = ResourceTable::new();
    let (identity, _) = mirror
        .sync_abi_snapshot(&host.abi_snapshot().unwrap())
        .unwrap();
    let imported = unsafe { CommandBuffer::from_abi(&mirror, identity, batch) }.unwrap();
    assert!(!imported.has_compute_commands().unwrap());
    assert_eq!(imported.abi_batch().unwrap().words.data, batch.words.data);
    ALLOCATIONS.with(|count| count.set(Some(0)));
    for _ in 0..100 {
        let batch = std::hint::black_box(recording.abi_batch().unwrap());
        let imported = unsafe { CommandBuffer::from_abi(&mirror, identity, batch) }.unwrap();
        let Command::WriteBuffer { data, .. } = imported.iter_commands().next().unwrap().unwrap()
        else {
            panic!("upload")
        };
        assert_eq!(data.as_ptr(), payload.as_ptr());
        assert_eq!(data, &payload);
        let mut reader = imported.command_reader();
        let Some(Command::WriteBuffer { data, .. }) = reader.next_command().unwrap() else {
            panic!("borrowed upload")
        };
        assert_eq!(data.as_ptr(), payload.as_ptr());
        assert_eq!(*data, &payload);
        assert!(reader.next_command().unwrap().is_none());
    }
    let allocations = ALLOCATIONS.with(|count| count.replace(None)).unwrap();
    assert_eq!(allocations, 0, "boundary or iteration allocated");
    assert_eq!(recording.abi_batch().unwrap().words.data, batch.words.data);
}

#[test]
fn command_reader_borrows_native_recordings_in_place() {
    let table = ResourceTable::new();
    let buffer = table
        .define_buffer(BufferDesc::new(4, BufferUsage::COPY_DST).unwrap())
        .unwrap();
    let payload = [7; 4];
    let mut encoder = CommandEncoder::new(&table);
    encoder.write_buffer(buffer, 0, &payload).unwrap();
    let commands = encoder.finish().unwrap();
    let mut reader = commands.command_reader();
    assert!(core::ptr::eq(
        reader.next_command().unwrap().unwrap(),
        &commands.commands()[0]
    ));
    assert!(reader.next_command().unwrap().is_none());
}

#[test]
fn resource_mirror_preserves_programmable_bindings_and_unused_retired_groups() {
    let host = ResourceTable::new();
    host.enable_abi_commands();
    let storage = host
        .define_buffer(BufferDesc::new(16, BufferUsage::STORAGE).unwrap())
        .unwrap();
    let layout = BindGroupLayoutDesc::new(vec![BindGroupLayoutEntry::new(
        0,
        ShaderStages::COMPUTE,
        BindingType::StorageBuffer { read_only: true },
    )])
    .unwrap();
    let group = host
        .define_bind_group(
            BindGroupDesc::new(
                &host,
                layout.clone(),
                vec![BindGroupEntry::new(
                    0,
                    BindingResource::Buffer {
                        buffer: storage.id(),
                        offset: 0,
                        size: 16,
                    },
                )],
            )
            .unwrap(),
        )
        .unwrap();
    let module = host
        .define_shader_module(
            ShaderModuleDesc::wgsl("@compute @workgroup_size(1) fn main() {}".into()).unwrap(),
        )
        .unwrap();
    let pipeline = host
        .define_compute_pipeline(
            ComputePipelineDesc::new(
                ShaderEntryPoint::new(module, ShaderStage::Compute, "main".into()).unwrap(),
                PipelineLayoutDesc::new(vec![layout]).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    let mut encoder = CommandEncoder::new(&host);
    let mut pass = encoder.begin_compute_pass().unwrap();
    pass.set_pipeline(pipeline).unwrap();
    pass.set_bind_group(0, group).unwrap();
    pass.dispatch(2, 3, 4).unwrap();
    pass.end().unwrap();
    let commands = encoder.finish().unwrap();
    let mirror = ResourceTable::new();
    let (source, _) = mirror
        .sync_abi_snapshot(&host.abi_snapshot().unwrap())
        .unwrap();
    let imported =
        unsafe { CommandBuffer::from_abi(&mirror, source, commands.abi_batch().unwrap()) }.unwrap();
    assert!(imported.has_compute_commands().unwrap());
    let values = imported
        .iter_commands()
        .collect::<Result<Vec<_>>>()
        .unwrap();
    let Command::SetComputePipeline(pipeline) = values[1] else {
        panic!("compute pipeline")
    };
    assert_eq!(
        mirror
            .compute_pipeline(pipeline)
            .unwrap()
            .shader()
            .entry_point(),
        "main"
    );
    let Command::SetBindGroup { bind_group, .. } = values[2] else {
        panic!("bindings")
    };
    let desc = mirror.bind_group(bind_group).unwrap();
    let BindingResource::Buffer {
        buffer,
        offset,
        size,
    } = desc.entries()[0].resource()
    else {
        panic!("buffer")
    };
    assert_eq!((offset, size), (0, 16));
    assert!(mirror.buffer_ref(buffer).is_ok());
    assert!(matches!(values[3], Command::Dispatch { x: 2, y: 3, z: 4 }));
    host.release_buffer(storage.id()).unwrap();
    // A second session can import a table even after old bindings became stale.
    let newer = ResourceTable::new();
    let (source, _) = newer
        .sync_abi_snapshot(&host.abi_snapshot().unwrap())
        .unwrap();
    let imported =
        unsafe { CommandBuffer::from_abi(&newer, source, commands.abi_batch().unwrap()) }.unwrap();
    let values = imported
        .iter_commands()
        .collect::<Result<Vec<_>>>()
        .unwrap();
    let Command::SetBindGroup { bind_group, .. } = values[2] else {
        panic!("bindings")
    };
    let desc = newer.bind_group(bind_group).unwrap();
    let BindingResource::Buffer { buffer, .. } = desc.entries()[0].resource() else {
        panic!("buffer")
    };
    assert!(
        newer.buffer_ref(buffer).is_err(),
        "retired buffer revived by snapshot"
    );
}

#[test]
fn metadata_cache_preserves_storage_and_reused_buffers_reject_stale_recordings() {
    let host = ResourceTable::new();
    host.enable_abi_commands();
    let desc = BufferDesc::new(16, BufferUsage::COPY_DST).unwrap();
    let old = host.define_buffer(desc).unwrap().id();
    let data = [1; 16];
    let mut encoder = CommandEncoder::new(&host);
    encoder
        .write_buffer(host.buffer_ref(old).unwrap(), 0, &data)
        .unwrap();
    let recording = encoder.finish().unwrap();
    let batch = recording.abi_batch().unwrap();
    let mirror = ResourceTable::new();
    let (identity, revision) = mirror
        .sync_abi_snapshot(&host.abi_snapshot().unwrap())
        .unwrap();
    let pointer = host.abi_snapshot().unwrap().as_ptr();
    assert_eq!(host.abi_snapshot().unwrap().as_ptr(), pointer);
    assert_eq!(host.abi_revision(), revision);
    host.release_buffer(old).unwrap();
    mirror
        .sync_abi_snapshot(&host.abi_snapshot().unwrap())
        .unwrap();
    let new = host.define_buffer(desc).unwrap().id();
    assert_ne!(old, new);
    mirror
        .sync_abi_snapshot(&host.abi_snapshot().unwrap())
        .unwrap();
    let imported = unsafe { CommandBuffer::from_abi(&mirror, identity, batch) }.unwrap();
    assert!(imported.iter_commands().next().unwrap().is_err());
}

#[test]
fn malformed_framing_and_foreign_tables_are_rejected() {
    let table = ResourceTable::new();
    for words in [
        &[0u64][..],
        &[(2u64 << 32) | 5][..],
        &[(1u64 << 32) | 999][..],
        &[(1u64 << 32) | 5, 99][..],
    ] {
        let batch = abi::Batch {
            table: 1,
            words: abi::Span::from_slice(words),
            count: 1,
        };
        let imported = unsafe { CommandBuffer::from_abi(&table, 1, batch) }.unwrap();
        assert!(imported.has_compute_commands().is_err());
        assert!(imported.iter_commands().any(|v| v.is_err()));
        let mut reader = imported.command_reader();
        loop {
            match reader.next_command() {
                Ok(Some(_)) => continue,
                Err(_) => break,
                Ok(None) => panic!("malformed stream accepted"),
            }
        }
        assert!(reader.next_command().unwrap().is_none());
        assert!(unsafe { CommandBuffer::from_abi(&table, 2, batch) }.is_err());
    }
}

#[test]
fn imported_render_pass_preserves_eight_attachments_and_read_only_depth() {
    let host = ResourceTable::new();
    host.enable_abi_commands();
    let extent = Extent2D::new(4, 4).unwrap();
    let color = TextureDesc::new(
        TextureFormat::Bgra8Unorm,
        extent,
        TextureUsage::RENDER_ATTACHMENT,
    )
    .unwrap();
    let targets: Vec<_> = (0..8)
        .map(|_| host.define_texture(color).unwrap())
        .collect();
    let depth = host
        .define_texture(
            TextureDesc::new(
                TextureFormat::Depth32Float,
                extent,
                TextureUsage::RENDER_ATTACHMENT | TextureUsage::SAMPLED,
            )
            .unwrap(),
        )
        .unwrap();
    let mut desc = RenderPassDesc::new(
        &host,
        targets[0],
        PixelRect::new(0, 0, 4, 4).unwrap(),
        LoadOp::Clear(Color::rgba(0.1, 0.2, 0.3, 1.0).unwrap()),
        StoreOp::Store,
    )
    .unwrap();
    for target in &targets[1..] {
        desc = desc
            .with_color_attachment(
                &host,
                *target,
                LoadOp::Clear(Color::rgba(1.0, 0.0, 0.0, 1.0).unwrap()),
                StoreOp::Store,
            )
            .unwrap();
    }
    desc = desc
        .with_depth_attachment(&host, depth, DepthLoadOp::Load, StoreOp::Store)
        .unwrap()
        .with_read_only_depth()
        .unwrap();
    let mut encoder = CommandEncoder::new(&host);
    encoder.begin_render_pass(desc).unwrap().end().unwrap();
    let recording = encoder.finish().unwrap();
    let mirror = ResourceTable::new();
    let (source, _) = mirror
        .sync_abi_snapshot(&host.abi_snapshot().unwrap())
        .unwrap();
    let imported =
        unsafe { CommandBuffer::from_abi(&mirror, source, recording.abi_batch().unwrap()) }
            .unwrap();
    let mut reader = imported.command_reader();
    let Some(Command::BeginRenderPass(pass)) = reader.next_command().unwrap() else {
        panic!("pass")
    };
    assert_eq!(pass.color_attachments().count(), 8);
    assert!(pass.depth_attachment().unwrap().read_only());
}
