// Pixel coordinates intentionally match the demo's 1024 x 768 render target.
// One full-screen triangle; every visible detail is evaluated by the GPU.

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(positions[vertex_index], 0.0, 1.0);
}

fn line(distance: f32, width: f32, pixel: f32) -> f32 {
    return 1.0 - smoothstep(width, width + pixel, abs(distance));
}

fn disc(p: vec2<f32>, center: vec2<f32>, radius: f32, pixel: f32) -> f32 {
    return 1.0 - smoothstep(radius, radius + pixel, length(p - center));
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let resolution = vec2<f32>(1024.0, 768.0);
    let pixel = 1.0 / resolution.y;
    let uv = position.xy / resolution;
    let p = (position.xy - resolution * vec2<f32>(0.5, 0.51)) / resolution.y;
    let radius = length(p);

    // A restrained blue-black field with a softly lit central atmosphere.
    var color = vec3<f32>(0.014, 0.024, 0.051);
    color += vec3<f32>(0.022, 0.043, 0.077) * exp(-radius * radius * 3.4);
    color += vec3<f32>(0.006, 0.035, 0.037)
        * exp(-length(p - vec2<f32>(-0.40, 0.20)) * 5.0);
    color += vec3<f32>(0.035, 0.006, 0.024)
        * exp(-length(p - vec2<f32>(0.31, -0.22)) * 5.0);

    // One-pixel grid lines, gently fading toward the outer edges.
    let grid_distance = abs(fract((p + vec2<f32>(0.0, 0.005)) / 0.0625 + 0.5) - 0.5) * 0.0625;
    let grid = 1.0 - smoothstep(0.0, pixel, min(grid_distance.x, grid_distance.y));
    color += vec3<f32>(0.035, 0.065, 0.095) * grid * exp(-radius * radius * 2.6) * 0.37;

    // Two precise orbital rings and segmented outer calibration marks.
    let ring = line(radius - 0.414, 0.0003, pixel);
    let outer_ring = line(radius - 0.438, 0.00015, pixel);
    let angle = atan2(p.y, p.x);
    let ticks = pow(max(cos(angle * 96.0), 0.0), 28.0)
        * smoothstep(0.446, 0.448, radius)
        * (1.0 - smoothstep(0.454, 0.456, radius));
    color += vec3<f32>(0.11, 0.21, 0.30) * (ring * 0.46 + outer_ring * 0.18 + ticks * 0.56);
    let cyan_arc = ring * smoothstep(0.2, 0.8, -sin(angle - 0.7));
    color += vec3<f32>(0.02, 0.20, 0.28) * cyan_arc * 0.62;

    // Signed edge distances from the triangle's analytic barycentric weights.
    // The RGB corners sit at (512, 138), (251, 599), and (773, 599).
    let top_weight = (0.27 - p.y) / 0.60;
    let right_weight = (1.0 - top_weight + p.x / 0.34) * 0.5;
    let left_weight = 1.0 - top_weight - right_weight;
    let edge_distance = min(top_weight * 0.60, min(left_weight, right_weight) * 0.59162);
    let face = smoothstep(-pixel * 0.7, pixel * 0.7, edge_distance);
    let weights = max(vec3<f32>(top_weight, left_weight, right_weight), vec3<f32>(0.0));
    let normalized_weights = weights / (weights.x + weights.y + weights.z);
    let rose = vec3<f32>(1.0, 0.19, 0.43);
    let cyan = vec3<f32>(0.03, 0.88, 1.0);
    let amber = vec3<f32>(1.0, 0.69, 0.12);
    let spectrum = rose * normalized_weights.x
        + cyan * normalized_weights.y + amber * normalized_weights.z;

    // Colored light scatters beyond the contour without requiring blending.
    let outer_distance = max(-edge_distance, 0.0);
    let halo = exp(-outer_distance * 35.0) * (1.0 - face);
    color += spectrum * halo * 0.105;
    color += spectrum * exp(-outer_distance * 130.0) * (1.0 - face) * 0.16;

    // A smooth spectral face with a beveled rim and a narrow diagonal sheen.
    let body_light = 0.70 + 0.20 * exp(-length(p - vec2<f32>(-0.09, -0.04)) * 3.0);
    let bevel = exp(-max(edge_distance, 0.0) * 155.0);
    let sheen = exp(-pow((p.x + p.y * 0.56 + 0.11) * 15.0, 2.0));
    var face_color = spectrum * body_light;
    face_color += vec3<f32>(0.09, 0.11, 0.14) * sheen;
    face_color += spectrum * bevel * 0.12;
    color = mix(color, face_color, face);
    let rim = exp(-abs(edge_distance) * 720.0);
    color += mix(spectrum, vec3<f32>(1.0), 0.56) * rim * 0.48;

    // Three small white-hot corner nodes anchor the scene to the orbit.
    let top = vec2<f32>(0.0, -0.33);
    let left = vec2<f32>(-0.34, 0.27);
    let right = vec2<f32>(0.34, 0.27);
    color += rose * exp(-length(p - top) * 105.0) * 0.32;
    color += cyan * exp(-length(p - left) * 105.0) * 0.32;
    color += amber * exp(-length(p - right) * 105.0) * 0.32;
    let nodes = disc(p, top, 0.0022, pixel)
        + disc(p, left, 0.0022, pixel) + disc(p, right, 0.0022, pixel);
    color = mix(color, vec3<f32>(0.96, 0.98, 1.0), min(nodes, 1.0));

    // Small registration marks and an RGB key finish the instrument-like frame.
    let frame_x = abs(p.x) - 0.586;
    let frame_y = abs(p.y + 0.01) - 0.414;
    let horizontal = line(frame_y, 0.0003, pixel)
        * (1.0 - smoothstep(0.0, pixel, frame_x))
        * smoothstep(-0.025, -0.023, frame_x);
    let vertical = line(frame_x, 0.0003, pixel)
        * (1.0 - smoothstep(0.0, pixel, frame_y))
        * smoothstep(-0.025, -0.023, frame_y);
    color += vec3<f32>(0.16, 0.24, 0.31) * (horizontal + vertical) * 0.75;
    let key_y = line(p.y - 0.411, 0.0015, pixel);
    color += key_y * (
        rose * (1.0 - smoothstep(0.010, 0.012, abs(p.x + 0.031)))
        + cyan * (1.0 - smoothstep(0.010, 0.012, abs(p.x)))
        + amber * (1.0 - smoothstep(0.010, 0.012, abs(p.x - 0.031)))
    ) * 0.70;

    // Sub-byte deterministic dithering keeps the dark gradients smooth in RGBA8.
    let noise = fract(sin(dot(position.xy, vec2<f32>(12.9898, 78.233))) * 43758.5453) - 0.5;
    let vignette = 1.0 - smoothstep(0.30, 0.84, length(uv - vec2<f32>(0.5))) * 0.35;
    color = color * vignette + vec3<f32>(noise / 255.0);
    return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
