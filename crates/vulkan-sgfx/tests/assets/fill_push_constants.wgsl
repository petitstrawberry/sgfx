struct Parameters {
    base: u32,
    count: u32,
    multiplier: u32,
    bias: u32,
};
var<push_constant> parameters: Parameters;
@group(0) @binding(0)
var<storage, read_write> output: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < parameters.count {
        let index = parameters.base + id.x;
        output[index] = index * parameters.multiplier + parameters.bias;
    }
}
