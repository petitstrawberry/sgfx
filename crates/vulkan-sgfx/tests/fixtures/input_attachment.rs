// A fragment shader containing genuine Vulkan SubpassData image reads.
// Three inputs compose (color.r, normal.g, depth.r, 1); one input copies RGBA.
pub fn input_fragment(count: usize) -> Vec<u32> {
    assert!(count == 1 || count == 3);
    let mut words = vec![0x07230203, 0x00010000, 0, 29, 0];
    let mut emit = |op: u32, args: &[u32]| {
        words.push(((args.len() as u32 + 1) << 16) | op);
        words.extend_from_slice(args);
    };
    emit(17, &[1]);
    emit(17, &[40]);
    emit(14, &[0, 1]);
    emit(15, &[4, 16, u32::from_le_bytes(*b"main"), 0, 15]);
    emit(16, &[16, 7]);
    for i in 0..count as u32 {
        emit(71, &[12 + i, 33, i]);
        emit(71, &[12 + i, 34, 0]);
        emit(71, &[12 + i, 43, i]);
    }
    emit(71, &[15, 30, 0]);
    emit(19, &[1]);
    emit(33, &[2, 1]);
    emit(22, &[3, 32]);
    emit(23, &[4, 3, 4]);
    emit(21, &[5, 32, 1]);
    emit(23, &[6, 5, 2]);
    emit(25, &[7, 3, 6, 0, 0, 0, 2, 0]);
    emit(32, &[8, 0, 7]);
    emit(32, &[9, 3, 4]);
    emit(43, &[5, 10, 0]);
    emit(44, &[6, 11, 10, 10]);
    emit(43, &[3, 27, 1.0f32.to_bits()]);
    for i in 0..count as u32 {
        emit(59, &[8, 12 + i, 0]);
    }
    emit(59, &[9, 15, 3]);
    emit(54, &[1, 16, 0, 2]);
    emit(248, &[17]);
    for i in 0..count as u32 {
        emit(61, &[7, 18 + i, 12 + i]);
        emit(98, &[4, 21 + i, 18 + i, 11]);
    }
    if count == 3 {
        emit(81, &[3, 24, 21, 0]);
        emit(81, &[3, 25, 22, 1]);
        emit(81, &[3, 26, 23, 0]);
        emit(80, &[4, 28, 24, 25, 26, 27]);
        emit(62, &[15, 28]);
    } else {
        emit(62, &[15, 21]);
    }
    emit(253, &[]);
    emit(56, &[]);
    words
}
