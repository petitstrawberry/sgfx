@group(0) @binding(0)
var<storage, read_write> output: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < 256u {
        output[id.x] = id.x * 3u + 7u;
    }
}
