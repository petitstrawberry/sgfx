use sgfx_core::ir::*;

fn cube_allocation() -> TextureDesc {
    TextureDesc::new(
        TextureFormat::Rgba8Unorm,
        Extent2D::new(8, 8).unwrap(),
        TextureUsage::SAMPLED | TextureUsage::COPY_DST,
    )
    .unwrap()
    .with_mip_level_count(4)
    .unwrap()
    .with_array_layer_count(6)
    .unwrap()
}

#[test]
fn cube_view_preserves_allocation_and_subresource_bounds() {
    let texture = cube_allocation();
    assert_eq!(texture.byte_size().unwrap(), (64 + 16 + 4 + 1) * 4 * 6);
    let view = TextureViewDesc::new(
        texture,
        TextureFormat::Rgba8UnormSrgb,
        TextureViewDimension::Cube,
        1,
        3,
        0,
        6,
    )
    .unwrap();
    assert_eq!(view.base_mip_level(), 1);
    assert_eq!(view.mip_level_count(), 3);
    for (base_mip, mips, base_layer, layers) in [
        (0, 0, 0, 6),
        (1, 4, 0, 6),
        (u32::MAX, 2, 0, 6),
        (0, 4, 1, 6),
        (0, 4, 0, 5),
        (0, 4, u32::MAX, 6),
    ] {
        assert!(
            TextureViewDesc::new(
                texture,
                TextureFormat::Rgba8Unorm,
                TextureViewDimension::Cube,
                base_mip,
                mips,
                base_layer,
                layers
            )
            .is_err()
        );
    }
    assert!(
        TextureViewDesc::new(
            texture,
            TextureFormat::R8Unorm,
            TextureViewDimension::Cube,
            0,
            4,
            0,
            6
        )
        .is_err()
    );
    let nonsquare = TextureDesc::new(
        TextureFormat::Rgba8Unorm,
        Extent2D::new(8, 4).unwrap(),
        TextureUsage::SAMPLED,
    )
    .unwrap()
    .with_array_layer_count(6)
    .unwrap();
    assert!(view.validate(nonsquare).is_err());
}

#[test]
fn sampled_views_require_matching_dimension_and_depth_type() {
    let table = ResourceTable::new();
    let desc = cube_allocation();
    let texture = table.define_texture(desc).unwrap();
    let view =
        TextureViewDesc::new(desc, desc.format(), TextureViewDimension::Cube, 0, 4, 0, 6).unwrap();
    for (dimension, depth, valid) in [
        (TextureViewDimension::Cube, false, true),
        (TextureViewDimension::D2Array, false, false),
        (TextureViewDimension::Cube, true, false),
    ] {
        let layout = BindGroupLayoutDesc::new(vec![BindGroupLayoutEntry::new(
            0,
            ShaderStages::FRAGMENT,
            BindingType::SampledTextureView { dimension, depth },
        )])
        .unwrap();
        assert_eq!(
            BindGroupDesc::new(
                &table,
                layout.clone(),
                vec![BindGroupEntry::new(
                    0,
                    BindingResource::TextureView {
                        texture: texture.id(),
                        view
                    }
                )]
            )
            .is_ok(),
            valid
        );
        let other = ResourceTable::new();
        assert!(
            BindGroupDesc::new(
                &other,
                layout,
                vec![BindGroupEntry::new(
                    0,
                    BindingResource::TextureView {
                        texture: texture.id(),
                        view
                    }
                )]
            )
            .is_err()
        );
    }
}

#[test]
fn uploads_select_one_layer_and_reject_out_of_range_layers() {
    let table = ResourceTable::new();
    let texture = table.define_texture(cube_allocation()).unwrap();
    let bytes = [0x55; 4];
    let write = TextureWrite::new(PixelRect::new(0, 0, 1, 1).unwrap(), 4, &bytes)
        .unwrap()
        .with_mip_level(3)
        .with_array_layer(5);
    let mut encoder = CommandEncoder::new(&table);
    assert!(
        encoder
            .write_texture(texture, write.with_array_layer(6))
            .is_err()
    );
    encoder.write_texture(texture, write).unwrap();
    let commands = encoder.finish().unwrap();
    assert_eq!(commands.command_count(), 1);
    assert_eq!(write.array_layer(), 5);
    let recording = OwnedCommandBuffer::new(vec![OwnedCommand::WriteTextureLayer {
        texture: texture.id(),
        mip_level: 3,
        array_layer: 5,
        destination: write.destination(),
        bytes_per_row: 4,
        data: bytes.to_vec(),
    }]);
    let replayed = recording.record(&table).unwrap();
    match replayed.commands()[0] {
        Command::WriteTexture { write, .. } => {
            assert_eq!(write.mip_level(), 3);
            assert_eq!(write.array_layer(), 5);
            assert_eq!(write.data(), bytes);
        }
        _ => panic!("expected layered texture upload"),
    }
}

#[test]
fn sampled_depth_is_valid_but_storage_depth_is_not() {
    let size = Extent2D::new(8, 8).unwrap();
    assert!(
        TextureDesc::new(
            TextureFormat::Depth32Float,
            size,
            TextureUsage::SAMPLED | TextureUsage::RENDER_ATTACHMENT
        )
        .is_ok()
    );
    assert!(TextureDesc::new(TextureFormat::Depth32Float, size, TextureUsage::STORAGE).is_err());
}
