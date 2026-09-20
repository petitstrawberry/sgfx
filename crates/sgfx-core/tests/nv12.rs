use sgfx_core::ir::{Extent2D, TextureDesc, TextureFormat, TextureUsage};
#[test]
fn nv12_storage_is_plane_aware_and_sampled_only() {
    let extent = Extent2D::new(3, 3).unwrap();
    let d = TextureDesc::new(TextureFormat::Nv12, extent, TextureUsage::SAMPLED).unwrap();
    assert_eq!(TextureFormat::Nv12.bytes_per_pixel(), None);
    assert_eq!(d.byte_size().unwrap(), 9 + 8);
    assert!(
        TextureDesc::new(
            TextureFormat::Nv12,
            extent,
            TextureUsage::SAMPLED | TextureUsage::COPY_DST
        )
        .is_err()
    );
    assert!(d.with_mip_level_count(2).is_err());
    assert!(d.with_array_layer_count(2).is_err());
}
