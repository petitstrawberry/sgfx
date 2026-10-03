use sgfx_core::ir::*;

fn descriptor() -> TextureDesc {
    TextureDesc::new(
        TextureFormat::Rgba8Unorm,
        Extent2D::new(1, 1).unwrap(),
        TextureUsage::COPY_DST | TextureUsage::SAMPLED,
    )
    .unwrap()
}

#[test]
fn retired_ids_references_and_recordings_never_alias_reused_slots() {
    let table = ResourceTable::new();
    let original = table.define_texture(descriptor()).unwrap();
    let recording = |id| {
        OwnedCommandBuffer::new(vec![OwnedCommand::WriteTexture {
            texture: id,
            destination: PixelRect::new(0, 0, 1, 1).unwrap(),
            bytes_per_row: 4,
            data: vec![1, 2, 3, 255],
        }])
    };
    let stale = recording(original.id());
    assert_eq!(stale.validate(&table), Ok(()));
    table.release_texture(original.id()).unwrap();
    let replacement = table.define_texture(descriptor()).unwrap();
    assert_eq!(replacement.slot(), original.slot());
    assert_ne!(replacement.id(), original.id());
    assert_ne!(replacement, original);
    assert_eq!(
        table.texture_ref(original.id()),
        Err(Error::InvalidDescriptor)
    );
    assert_eq!(table.texture(original), Err(Error::InvalidDescriptor));
    assert_eq!(stale.validate(&table), Err(Error::InvalidDescriptor));
    assert_eq!(recording(replacement.id()).validate(&table), Ok(()));
    assert_eq!(
        table.release_texture(original.id()),
        Err(Error::InvalidDescriptor)
    );
    assert_eq!(table.texture(replacement), Ok(descriptor()));
}

#[test]
fn image_churn_exceeds_old_append_only_limit_without_growing_slots() {
    let table = ResourceTable::new();
    let mut previous = None;
    for _ in 0..MAX_TEXTURES * 8 + 1 {
        let image = table.define_texture(descriptor()).unwrap();
        assert_eq!(image.slot(), 0);
        if let Some(old) = previous {
            assert_ne!(old, image.id());
            assert_eq!(table.texture_ref(old), Err(Error::InvalidDescriptor));
        }
        table.release_texture(image.id()).unwrap();
        previous = Some(image.id());
    }
}

#[test]
fn live_capacity_and_foreign_identity_checks_are_preserved() {
    let table = ResourceTable::new();
    let other = ResourceTable::new();
    let foreign = other.define_texture(descriptor()).unwrap();
    let live: Vec<_> = (0..MAX_TEXTURES)
        .map(|_| table.define_texture(descriptor()).unwrap())
        .collect();
    assert_eq!(
        table.define_texture(descriptor()),
        Err(Error::ResourceLimitExceeded)
    );
    assert_eq!(
        table.release_texture(foreign.id()),
        Err(Error::ResourceTableMismatch)
    );
    let retired = live[MAX_TEXTURES / 2];
    table.release_texture(retired.id()).unwrap();
    let replacement = table.define_texture(descriptor()).unwrap();
    assert_eq!(replacement.slot(), retired.slot());
    assert_eq!(
        table.define_texture(descriptor()),
        Err(Error::ResourceLimitExceeded)
    );
    for image in live {
        assert_eq!(
            table.texture(image),
            if image == retired {
                Err(Error::InvalidDescriptor)
            } else {
                Ok(descriptor())
            }
        );
    }
}
