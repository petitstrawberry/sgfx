//! Real-device checks for the portable command execution contract.

use sgfx_core::backend::CommandExecutor;

use super::*;

fn headless_device() -> Option<Device> {
    let instance = raw::Instance::new(&raw::InstanceDescriptor::default());
    let Some(adapter) = request_headless_adapter(&instance) else {
        #[cfg(target_os = "macos")]
        panic!("Metal adapter must be available for SGFX execution contract tests");
        #[cfg(not(target_os = "macos"))]
        {
            eprintln!("skipping SGFX execution contract test: no adapter available");
            return None;
        }
    };
    eprintln!("SGFX execution contract adapter: {:?}", adapter.get_info());
    let (device, queue) =
        pollster::block_on(adapter.request_device(&raw::DeviceDescriptor::default(), None))
            .expect("headless WGPU device");
    Some(Device::new(device, queue))
}

fn readback_pixels(device: &Device, image: &Image) -> Vec<[u8; 4]> {
    let stride = (image.width() * 4).div_ceil(256) * 256;
    let readback = device.raw_device().create_buffer(&raw::BufferDescriptor {
        label: Some("SGFX execution contract readback"),
        size: u64::from(stride) * u64::from(image.height()),
        usage: raw::BufferUsages::COPY_DST | raw::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device
        .raw_device()
        .create_command_encoder(&raw::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        raw::TexelCopyTextureInfo {
            texture: image.raw_texture(),
            mip_level: 0,
            origin: raw::Origin3d::ZERO,
            aspect: raw::TextureAspect::All,
        },
        raw::TexelCopyBufferInfo {
            buffer: &readback,
            layout: raw::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride),
                rows_per_image: Some(image.height()),
            },
        },
        raw::Extent3d {
            width: image.width(),
            height: image.height(),
            depth_or_array_layers: 1,
        },
    );
    device.raw_queue().submit([encoder.finish()]);
    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(raw::MapMode::Read, move |result| {
        sender.send(result).expect("send readback result");
    });
    let _ = device.raw_device().poll(raw::Maintain::Wait);
    receiver
        .recv()
        .expect("receive readback result")
        .expect("map readback");
    let bytes = slice.get_mapped_range();
    bytes
        .chunks_exact(stride as usize)
        .flat_map(|row| row[..image.width() as usize * 4].chunks_exact(4))
        .map(|pixel| pixel.try_into().expect("BGRA pixel"))
        .collect()
}

#[test]
fn uploads_outlive_caller_storage_and_keep_submission_order() {
    let _guard = HEADLESS_WGPU_TEST_LOCK.lock().expect("lock WGPU tests");
    let Some(device) = headless_device() else {
        return;
    };
    let context = device.create_context();
    let queue = context.create_queue();
    let table = Rc::new(ResourceTable::new());
    let extent = Extent2D::new(4, 4).expect("extent");
    let area = PixelRect::new(0, 0, 4, 4).expect("area");
    let source = table
        .define_texture(
            TextureDesc::new(
                TextureFormat::Bgra8Unorm,
                extent,
                TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            )
            .expect("source descriptor"),
        )
        .expect("source");
    let mut cache = context.create_resources(Rc::clone(&table));
    let colors = [[17, 31, 211, 255], [197, 83, 29, 255]];
    let mut images = Vec::new();
    for color in colors {
        let target = table
            .define_texture(
                TextureDesc::new(
                    TextureFormat::Bgra8Unorm,
                    extent,
                    TextureUsage::RENDER_ATTACHMENT
                        | TextureUsage::PRESENT
                        | TextureUsage::COPY_DST,
                )
                .expect("target descriptor"),
            )
            .expect("target");
        let image = context
            .create_image(4, 4, TextureFormat::Bgra8Unorm)
            .expect("image");
        cache
            .map_image(target.id(), Arc::clone(&image))
            .expect("map image");
        let mut pixels = color.repeat(16);
        let mut encoder = CommandEncoder::new(table.as_ref());
        encoder
            .write_texture(
                source,
                TextureWrite::new(area, 16, &pixels).expect("upload"),
            )
            .expect("record upload");
        encoder
            .copy_texture_to_texture(source, area, target, area)
            .expect("record copy");
        let commands = encoder.finish().expect("finish commands");
        queue
            .executor(&mut cache)
            .execute(&commands)
            .expect("execute");
        drop(commands);
        // No wait: the backend must not retain a borrow of this allocation.
        pixels.fill(0);
        drop(pixels);
        images.push(image);
    }
    // The source texture is owned only by the cache and pending WGPU work.
    // Drop the cache before waiting; each queued copy must still see its own
    // upload, not the later upload or the caller's zeroed allocation.
    drop(cache);
    drop(queue);
    drop(context);
    drop(table);
    for (image, color) in images.iter().zip(colors) {
        assert_eq!(readback_pixels(&device, image), vec![color; 16]);
    }
}

#[test]
fn executor_rejects_a_different_resource_table_even_when_empty() {
    let _guard = HEADLESS_WGPU_TEST_LOCK.lock().expect("lock WGPU tests");
    let Some(device) = headless_device() else {
        return;
    };
    let context = device.create_context();
    let queue = context.create_queue();
    let table = Rc::new(ResourceTable::new());
    let other_table = ResourceTable::new();
    let mut cache = context.create_resources(Rc::clone(&table));
    let empty = CommandEncoder::new(&table).finish().expect("empty stream");
    queue
        .executor(&mut cache)
        .execute(&empty)
        .expect("empty no-op");
    let foreign = CommandEncoder::new(&other_table)
        .finish()
        .expect("foreign stream");
    assert!(matches!(
        queue.executor(&mut cache).execute(&foreign),
        Err(Error::ResourceTableMismatch)
    ));
}

#[test]
fn executor_rejects_late_upload_instead_of_silently_reordering_it() {
    let _guard = HEADLESS_WGPU_TEST_LOCK.lock().expect("lock WGPU tests");
    let Some(device) = headless_device() else {
        return;
    };
    let context = device.create_context();
    let table = Rc::new(ResourceTable::new());
    let descriptor = TextureDesc::new(
        TextureFormat::Bgra8Unorm,
        Extent2D::new(1, 1).expect("extent"),
        TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
    )
    .expect("descriptor");
    let source = table.define_texture(descriptor).expect("source");
    let target = table.define_texture(descriptor).expect("target");
    let area = PixelRect::new(0, 0, 1, 1).expect("area");
    let mut encoder = CommandEncoder::new(table.as_ref());
    encoder
        .copy_texture_to_texture(source, area, target, area)
        .expect("copy");
    encoder
        .write_texture(
            source,
            TextureWrite::new(area, 4, &[0, 0, 255, 255]).expect("upload"),
        )
        .expect("core accepts an ordered late upload");
    let commands = encoder.finish().expect("finish stream");
    let mut cache = context.create_resources(Rc::clone(&table));
    assert!(matches!(
        context
            .create_queue()
            .executor(&mut cache)
            .execute(&commands),
        Err(Error::Unsupported(UnsupportedFeature::LateUpload))
    ));
}
