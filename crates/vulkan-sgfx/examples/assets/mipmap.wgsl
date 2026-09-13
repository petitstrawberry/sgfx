@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var image_sampler: sampler;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<u32>>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < 4u {
        let color = textureSampleLevel(image, image_sampler, vec2<f32>(0.37, 0.63), f32(id.x) + 0.25);
        output[id.x] = vec4<u32>(round(color * 255.0));
    }
}
