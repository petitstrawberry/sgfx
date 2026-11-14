//! Public float blits must agree with TextureFormat::blit_compatible.
use sgfx_core::ir::*;

fn texture(table: &ResourceTable, format: TextureFormat) -> TextureRef<'_> {
    table
        .define_texture(
            TextureDesc::new(
                format,
                Extent2D::new(16, 16).unwrap(),
                TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            )
            .unwrap()
            .with_mip_level_count(2)
            .unwrap(),
        )
        .unwrap()
}

#[test]
fn float_blits_record_full_mips_regions_filters_and_flips() {
    for format in [TextureFormat::Rgba16Float, TextureFormat::Rgba32Float] {
        for filter in [FilterMode::Nearest, FilterMode::Linear] {
            let table = ResourceTable::new();
            #[cfg(feature = "backend-abi")]
            table.enable_abi_commands();
            let source = texture(&table, format);
            let destination = texture(&table, format);
            let src = PixelRect::new(1, 2, 4, 4).unwrap();
            let dst = PixelRect::new(2, 3, 8, 8).unwrap();
            let mut encoder = CommandEncoder::new(&table);
            encoder
                .blit_texture(source, 0, destination, 1, filter)
                .unwrap();
            encoder
                .blit_texture_region(source, 1, src, destination, 0, dst, filter)
                .unwrap();
            for flips in [[true, false], [false, true], [true, true]] {
                encoder
                    .blit_texture_region_flipped(source, 1, src, destination, 0, dst, filter, flips)
                    .unwrap();
            }
            // Different levels of the same allocation are also valid.
            encoder.blit_texture(source, 0, source, 1, filter).unwrap();
            let recording = encoder.finish().unwrap();
            assert_eq!(recording.command_count(), 6);
            for (index, command) in recording.commands().iter().enumerate() {
                let Command::BlitTexture {
                    source: s,
                    source_mip,
                    destination: d,
                    destination_mip,
                    source_rect,
                    destination_rect,
                    filter: actual_filter,
                    flips,
                } = command
                else {
                    panic!("expected blit");
                };
                assert_eq!(s.id(), source.id());
                assert_eq!(
                    d.id(),
                    if index == 5 {
                        source.id()
                    } else {
                        destination.id()
                    }
                );
                assert_eq!(*actual_filter, filter);
                if index == 0 || index == 5 {
                    assert_eq!((*source_mip, *destination_mip), (0, 1));
                    assert_eq!(*source_rect, PixelRect::new(0, 0, 16, 16).unwrap());
                    assert_eq!(*destination_rect, PixelRect::new(0, 0, 8, 8).unwrap());
                } else {
                    assert_eq!((*source_mip, *destination_mip), (1, 0));
                    assert_eq!((*source_rect, *destination_rect), (src, dst));
                }
                assert_eq!(
                    *flips,
                    match index {
                        2 => [true, false],
                        3 => [false, true],
                        4 => [true, true],
                        _ => [false; 2],
                    }
                );
            }
        }
    }
}

#[test]
fn float_blits_preserve_format_bounds_usage_and_alias_rejections() {
    let table = ResourceTable::new();
    #[cfg(feature = "backend-abi")]
    table.enable_abi_commands();
    let source = texture(&table, TextureFormat::Rgba16Float);
    let other_format = texture(&table, TextureFormat::Rgba32Float);
    let destination = texture(&table, TextureFormat::Rgba16Float);
    let depth = table
        .define_texture(
            TextureDesc::new(
                TextureFormat::Depth32Float,
                Extent2D::new(16, 16).unwrap(),
                TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            )
            .unwrap(),
        )
        .unwrap();
    let readonly = table
        .define_texture(
            TextureDesc::new(
                TextureFormat::Rgba16Float,
                Extent2D::new(16, 16).unwrap(),
                TextureUsage::COPY_SRC,
            )
            .unwrap(),
        )
        .unwrap();
    let mut encoder = CommandEncoder::new(&table);
    assert_eq!(
        encoder.blit_texture(source, 0, other_format, 0, FilterMode::Linear),
        Err(Error::InvalidDescriptor)
    );
    assert_eq!(
        encoder.blit_texture(depth, 0, destination, 0, FilterMode::Nearest),
        Err(Error::InvalidDescriptor)
    );
    assert_eq!(
        encoder.blit_texture(depth, 0, depth, 0, FilterMode::Nearest),
        Err(Error::InvalidDescriptor)
    );
    assert_eq!(
        encoder.blit_texture(source, 0, source, 0, FilterMode::Nearest),
        Err(Error::ResourceAccessConflict)
    );
    assert_eq!(
        encoder.blit_texture(source, 0, readonly, 0, FilterMode::Nearest),
        Err(Error::InvalidUsage)
    );
    assert_eq!(
        encoder.blit_texture(source, 2, destination, 0, FilterMode::Nearest),
        Err(Error::OutOfBounds)
    );
    assert_eq!(
        encoder.blit_texture_region(
            source,
            1,
            PixelRect::new(0, 0, 9, 8).unwrap(),
            destination,
            0,
            PixelRect::new(0, 0, 16, 16).unwrap(),
            FilterMode::Linear
        ),
        Err(Error::OutOfBounds)
    );
    assert_eq!(encoder.finish().unwrap().command_count(), 0);
}
