# SGFX IR syntax reference

The SGFX 1.0.0 IR is a typed Rust representation, not a textual DSL, shader
language or serialized wire format. Its public names live in `sgfx_core::ir`
and are reexported unchanged as `sgfx::ir`. Descriptors and references are
constructed through validated APIs; commands are recorded with
`CommandEncoder` and inspected through `CommandBuffer::commands()`.

This page gives the construction syntax and every command payload. Signature
tables omit implementation bodies and use `Result<T>` for `ir::Result<T>`
(`Result<T, ir::Error>`). The final example is Rust code that records a complete
upload-and-draw stream. Device setup and submission are in the
[SGFX reference](reference.md).

- [Values and coordinates](#values-and-coordinates)
- [Resource declarations](#resource-declarations)
- [Pipeline syntax](#pipeline-syntax)
- [Render-pass syntax](#render-pass-syntax)
- [Command syntax](#command-syntax)
- [Recording rules and limits](#recording-rules-and-limits)
- [Example: upload and draw a triangle](#example-upload-and-draw-a-triangle)

## Values and coordinates

Source: [value types](../crates/sgfx-core/src/ir/types.rs).

| Constructor | Result | Meaning |
| --- | --- | --- |
| `Extent2D::new(width: u32, height: u32)` | `Result<Extent2D>` | Nonzero pixel dimensions. |
| `PixelRect::new(x: u32, y: u32, width: u32, height: u32)` | `Result<PixelRect>` | Nonempty, overflow-safe rectangle. |
| `Color::rgba(red: f32, green: f32, blue: f32, alpha: f32)` | `Result<Color>` | Finite components; no implicit clamp to `[0, 1]`. |
| `Transform::from_columns(columns: [f32; 16])` | `Result<Transform>` | Finite column-major 4×4 matrix. |
| `Transform::identity()` | `Transform` | Identity transform. |
| `DrawUniforms::new(transform: Transform, color: Color)` | `DrawUniforms` | Per-draw transform and color. |

Pixel rectangles use a top-left origin, x right and y down, with half-open
bounds `[x, x + width) × [y, y + height)`. Transform matrices multiply column
vectors. Position attributes expand from two/three components to `(x, y, 0, 1)`
or `(x, y, z, 1)`; four components retain their `w`. The portable rendering
convention uses clip-space depth `0 <= z <= w` and NDC x/y in `[-1, 1]`, y up.
The viewport spans the whole attachment; a pass area clips rendering and
clearing rather than rescaling that viewport. See the
[rendering conventions](1.0-contract.md#4-rendering-meaning-not-backend-dependent-approximations).

## Resource declarations

Source: [resource descriptors and identities](../crates/sgfx-core/src/ir/resource.rs).

| Constructor | Result |
| --- | --- |
| `ResourceTable::new()` | `ResourceTable` |
| `TextureDesc::new(format: TextureFormat, extent: Extent2D, usage: TextureUsage)` | `Result<TextureDesc>` |
| `BufferDesc::new(size: u64, usage: BufferUsage)` | `Result<BufferDesc>` |
| `SamplerDesc::new(min_filter: FilterMode, mag_filter: FilterMode, address_u: AddressMode, address_v: AddressMode)` | `SamplerDesc` |
| `TextureWrite::new(destination: PixelRect, bytes_per_row: u32, data: &'data [u8])` | `Result<TextureWrite<'data>>` |

| Resource | Define in a table | Persistent ID | Resolve an ID in its owning table |
| --- | --- | --- | --- |
| Texture | `table.define_texture(desc)` → `Result<TextureRef<'_>>` | `TextureId` | `table.texture_ref(id)` |
| Buffer | `table.define_buffer(desc)` → `Result<BufferRef<'_>>` | `BufferId` | `table.buffer_ref(id)` |
| Sampler | `table.define_sampler(desc)` → `Result<SamplerRef<'_>>` | `SamplerId` | `table.sampler_ref(id)` |
| Render pipeline | `table.define_render_pipeline(desc)` → `Result<RenderPipelineRef<'_>>` | `RenderPipelineId` | `table.render_pipeline_ref(id)` |

Every reference exposes `.id()` to retain its identity without retaining a Rust
borrow. IDs are qualified by their originating table; `.slot()` is not a
process-global resource identifier. Resolve the ID to a reference before
recording. `table.texture(reference)`, `buffer`, `sampler` and `render_pipeline`
return the corresponding descriptor through a fallible lookup.

Definitions are immutable and append-only. Defining a resource does not
allocate a GPU object, and dropping its ID does not remove its table entry.
Keep definitions and backend caches across frames instead of redefining the
same resources each frame. Tables are not concurrent recorders.

### Formats, usage and sampling

| Type | Values |
| --- | --- |
| `TextureFormat` | `Bgra8Unorm`, `Rgba8Unorm` (4 bytes/pixel), `R8Unorm` (1 byte/pixel), `Depth32Float` (4 bytes/pixel). |
| `TextureUsage` | `SAMPLED`, `RENDER_ATTACHMENT`, `COPY_SRC`, `COPY_DST`, `PRESENT`; combine with `\|` or `.union()`. |
| `BufferUsage` | `VERTEX`, `INDEX`, `COPY_SRC`, `COPY_DST`; combine with `\|` or `.union()`. |
| `FilterMode` | `Nearest`, `Linear`. |
| `AddressMode` | `ClampToEdge`, `Repeat`, `MirrorRepeat`. |

Texture usage must be nonempty. `Depth32Float` permits only
`RENDER_ATTACHMENT`, so it cannot be uploaded, copied or sampled through this
IR. Buffer size and usage must both be nonzero. `COPY_SRC` on a buffer is a
usage flag; there is no buffer-copy instruction in the current command enum.

For a texture upload, the source stride is in bytes and must be at least
`destination.width() * format.bytes_per_pixel()`. The slice must contain at
least `(height - 1) * bytes_per_row + width * bytes_per_pixel` bytes; the final
row needs no trailing padding. `TextureWrite::new` checks the nonzero stride;
the encoder checks the format, bounds and slice length when recording it.

## Pipeline syntax

Source: [pipeline descriptors](../crates/sgfx-core/src/ir/pipeline.rs).

| Constructor / builder | Result |
| --- | --- |
| `VertexAttribute::new(location: u32, format: VertexFormat, offset: u32)` | `VertexAttribute` |
| `VertexBufferLayout::new(stride: u32, attributes: Vec<VertexAttribute>)` | `Result<VertexBufferLayout>` |
| `RenderPipelineDesc::new(target_format: TextureFormat, topology: PrimitiveTopology, vertex_buffer: VertexBufferLayout, fragment: FragmentProgram, blend: BlendState, raster: RasterState)` | `Result<RenderPipelineDesc>` |
| `pipeline.with_depth_state(depth: DepthState)` | `Result<RenderPipelineDesc>`; consumes and returns the descriptor. |
| `DepthState::new(format: TextureFormat, compare: CompareFunction, write_enabled: bool)` | `DepthState`; the pipeline builder validates its depth format. |
| `RasterState::new(cull_mode: CullMode, front_face: FrontFace)` | `RasterState` |
| `BlendComponent::new(source_factor: BlendFactor, destination_factor: BlendFactor, operation: BlendOp)` | `BlendComponent` |
| `BlendState::new(color: BlendComponent, alpha: BlendComponent)` | `BlendState` |

There is one interleaved vertex buffer. Stride and attribute offsets are bytes;
attribute locations must be unique and each attribute must fit the stride.
Location 0 is always position (`Float32x2`, `Float32x3` or `Float32x4`).
Fragment selection determines the other required attributes:

| `FragmentProgram` syntax | Location 1 | Location 2 |
| --- | --- | --- |
| `FragmentProgram::Solid` | Not required | Not required |
| `FragmentProgram::VertexColor` | Color: `Float32x3`, `Float32x4` or `Unorm8x4` | Not required |
| `FragmentProgram::Texture(mode)` | UV: `Float32x2` | Not required |
| `FragmentProgram::TextureVertexColor(mode)` | Color: `Float32x3`, `Float32x4` or `Unorm8x4` | UV: `Float32x2` |

`mode` is `TextureSampleMode::Rgba`, `RgbIgnoreAlpha` or `AlphaMask`.
`RgbIgnoreAlpha` uses sampled RGB with alpha one. `AlphaMask` uses coverage to
modulate the uniform color: red for `R8Unorm`, alpha for RGBA/BGRA textures.
`Solid` uses the uniform color; vertex-color modes also multiply by it.

| State type | Values |
| --- | --- |
| `PrimitiveTopology` | `TriangleList` only. |
| `IndexFormat` | `Uint16`, `Uint32`. |
| `CullMode` | `None`, `Front`, `Back`. |
| `FrontFace` | `Clockwise`, `CounterClockwise` in NDC. |
| `CompareFunction` | `Never`, `Less`, `Equal`, `LessEqual`, `Greater`, `NotEqual`, `GreaterEqual`, `Always`. |
| `BlendFactor` | `Zero`, `One`, `SourceAlpha`, `OneMinusSourceAlpha`, `DestinationAlpha`, `OneMinusDestinationAlpha`. |
| `BlendOp` | `Add`, `Subtract`, `ReverseSubtract`. |

Built-in blend states are `BlendState::REPLACE`,
`SOURCE_OVER_STRAIGHT_ALPHA` and `DESTINATION_IN`. Color and alpha blend
components are independent. The unorm formats do not introduce implicit gamma
conversion or premultiplication. A pipeline's target format must be a color
format; optional depth state uses `Depth32Float`. `with_depth_stencil` is a
compatibility alias for `with_depth_state`.

## Render-pass syntax

Source: [pass descriptors](../crates/sgfx-core/src/ir/command.rs).

| Constructor / builder | Result |
| --- | --- |
| `RenderPassDesc::new(resources: &'r ResourceTable, target: TextureRef<'r>, area: PixelRect, load: LoadOp, store: StoreOp)` | `Result<RenderPassDesc<'r>>` |
| `pass.with_depth_attachment(resources: &'r ResourceTable, target: TextureRef<'r>, load: DepthLoadOp, store: StoreOp)` | `Result<RenderPassDesc<'r>>`; consumes and returns the descriptor. |

The color target needs `RENDER_ATTACHMENT` usage and the area must fit inside
it. The optional depth texture needs the same extent as the color texture and
`Depth32Float` format. `DepthAttachment` is obtained through the builder, not
through public struct fields.

```rust
pub enum LoadOp { Load, Clear(Color), DontCare }
pub enum DepthLoadOp { Load, Clear(f32), DontCare }
pub enum StoreOp { Store, DontCare }
```

`Load` preserves existing contents in the pass area, `Clear` initializes that
area, and `DontCare` permits discarding contents. Depth clears must be finite
and in `[0, 1]`. `Store` retains results; do not depend on discarded or
uninitialized pixels. A clear-only pass needs no pipeline or draw bindings.

## Command syntax

These are all variants of
[`Command<'r, 'data>`](../crates/sgfx-core/src/ir/command.rs), with documentation
comments omitted. `'r` is the resource-table borrow and `'data` is the upload
data borrow. The representation contains Rust references, not serialized IDs
or an on-disk command language.

```rust
pub enum Command<'r, 'data> {
    WriteBuffer {
        buffer: BufferRef<'r>,
        offset: u64,
        data: &'data [u8],
    },
    WriteTexture {
        texture: TextureRef<'r>,
        write: TextureWrite<'data>,
    },
    CopyTextureToTexture {
        source: TextureRef<'r>,
        source_rect: PixelRect,
        destination: TextureRef<'r>,
        destination_rect: PixelRect,
    },
    BeginRenderPass(RenderPassDesc<'r>),
    EndRenderPass,
    SetPipeline(RenderPipelineRef<'r>),
    SetVertexBuffer {
        buffer: BufferRef<'r>,
        offset: u64,
    },
    SetIndexBuffer {
        buffer: BufferRef<'r>,
        offset: u64,
        format: IndexFormat,
    },
    SetTexture(TextureRef<'r>),
    SetSampler(SamplerRef<'r>),
    SetUniforms(DrawUniforms),
    SetScissor(Option<PixelRect>),
    Draw {
        vertex_count: u32,
        first_vertex: u32,
    },
    DrawIndexed {
        index_count: u32,
        first_index: u32,
        base_vertex: i32,
    },
}
```

Use the encoders to construct a validated `CommandBuffer`; there is no public
constructor accepting an arbitrary `Vec<Command>`.

### Outside a render pass

Create an encoder with `CommandEncoder::new(&resources)`.

| Encoder method arguments | Result | Recorded command |
| --- | --- | --- |
| `write_buffer(buffer: BufferRef<'r>, offset: u64, data: &'data [u8])` | `Result<()>` | `WriteBuffer` |
| `write_texture(texture: TextureRef<'r>, write: TextureWrite<'data>)` | `Result<()>` | `WriteTexture` |
| `copy_texture_to_texture(source: TextureRef<'r>, source_rect: PixelRect, destination: TextureRef<'r>, destination_rect: PixelRect)` | `Result<()>` | `CopyTextureToTexture` |
| `begin_render_pass(desc: RenderPassDesc<'r>)` | `Result<RenderPassEncoder<'encoder, 'r, 'data>>` borrowing the encoder | `BeginRenderPass` |
| `finish(self)` | `Result<CommandBuffer<'r, 'data>>` | None; consumes the encoder. |

Uploads require `COPY_DST`. Texture copies require `COPY_SRC`/`COPY_DST`, equal
formats, equal rectangle extents and in-bounds rectangles; overlapping
self-copies are rejected. There is no upload, copy or nested pass inside a pass.

### Inside a render pass

All methods below act on the `RenderPassEncoder` and return `Result<()>`.
Only `end(self)` consumes the pass; the others borrow it mutably.

| Method arguments | Recorded command | Required state / validation |
| --- | --- | --- |
| `set_pipeline(pipeline: RenderPipelineRef<'r>)` | `SetPipeline` | Color format and any required depth format match the pass. |
| `set_vertex_buffer(buffer: BufferRef<'r>, offset: u64)` | `SetVertexBuffer` | `VERTEX` usage and in-bounds byte offset; draw validates stride alignment. |
| `set_index_buffer(buffer: BufferRef<'r>, offset: u64, format: IndexFormat)` | `SetIndexBuffer` | `INDEX` usage, in-bounds byte offset aligned to the index width. |
| `set_texture(texture: TextureRef<'r>)` | `SetTexture` | `SAMPLED` usage; cannot sample the active color target. |
| `set_sampler(sampler: SamplerRef<'r>)` | `SetSampler` | Sampler belongs to this table. |
| `set_uniforms(uniforms: DrawUniforms)` | `SetUniforms` | Explicit per-draw transform and color. |
| `set_scissor(scissor: Option<PixelRect>)` | `SetScissor` | `Some(rect)` lies inside the pass area; `None` resets to that area. |
| `draw(vertex_count: u32, first_vertex: u32)` | `Draw` | Pipeline, vertex buffer and uniforms; valid vertex range. |
| `draw_indexed(index_count: u32, first_index: u32, base_vertex: i32)` | `DrawIndexed` | Draw bindings plus index buffer; valid index byte range. |
| `end(self)` | `EndRenderPass` | Explicitly closes the pass. |

Draw counts must be positive multiples of three. `first_vertex` and
`first_index` are element indices, not bytes; binding offsets are bytes.
Textured pipelines additionally require both texture and sampler bindings.
The core validates index storage ranges but does not scan uploaded index values
to prove which vertices they address; device-specific validation is backend work.

## Recording rules and limits

Fixed drawing streams contain uploads/copies and non-nested render passes in
recorded order. The programmable extension below also adds compute passes.
Each new pass starts without a pipeline, buffer or uniform binding; state does
not carry from a previous pass. Before each draw, set the bindings required by
its pipeline. Within a pass, accepted bindings persist until changed.
An abandoned pass is not implicitly ended: `finish()` returns
`RenderPassStillActive` unless `end()` succeeded.

A rejected recording operation does not append a command or replace accepted
bindings. References from another table return `ResourceTableMismatch`.
Missing bindings, invalid usage, out-of-bounds ranges and exhausted capacity
are explicit `ir::Error` values. This local recording rule is distinct from
non-transactional [backend execution failures](reference.md#execution-and-completion).

| Bound | Value | Scope |
| --- | --- | --- |
| `MAX_TEXTURES` | 1,024 | Per resource table. |
| `MAX_BUFFERS` | 1,024 | Per resource table. |
| `MAX_SAMPLERS` | 256 | Per resource table. |
| `MAX_RENDER_PIPELINES` | 256 | Per resource table. |
| `MAX_VERTEX_ATTRIBUTES` | 16 | Per vertex layout. |
| `MAX_COMMANDS` | 4,096 | Per command buffer, including pass begin/end and bindings. |

Beginning a pass reserves room for its end command. These are IR capacity
bounds, not promises about GPU memory or device limits. Command buffers borrow
the table and upload slices; backend execution must consume/copy any upload
data needed by pending work before returning.

Backend support can be narrower than valid IR. For example, WGPU requires
buffer upload offsets and lengths to be multiples of four. Encoder-owned staging
copies preserve upload order between copies and render/compute passes. Fixed
render pipelines require pass/pipeline depth-state presence to agree, and WGPU
rejects partial rectangular depth clears. Such restrictions must return errors,
not silently change command meanings.

Shader modules, compute dispatch and explicit same-queue resource dependencies
are part of the [programmable extension](#programmable-graphics-and-compute).
Instancing, multiple color attachments, mip/array/3D textures and cross-queue
semaphore commands remain outside the implemented subset. Presentation and
completion observation are outside the command enum.

## Example: upload and draw a triangle

This std Rust example needs only `sgfx-core`. It records a BGRA target clear and
a solid-color triangle, including a vertex upload; it does not open a GPU or
display a window. The vertex bytes remain alive while the command buffer
borrows them. No external cast library or unsafe slice conversion is needed.

```rust
use sgfx_core::ir::*;

fn main() -> Result<()> {
    let resources = ResourceTable::new();
    let extent = Extent2D::new(640, 480)?;
    let area = PixelRect::new(0, 0, 640, 480)?;
    let target = resources.define_texture(TextureDesc::new(
        TextureFormat::Bgra8Unorm,
        extent,
        TextureUsage::RENDER_ATTACHMENT | TextureUsage::PRESENT,
    )?)?;

    // Three Float32x2 positions, 8 bytes per vertex.
    let positions: [f32; 6] = [-0.5, -0.5, 0.5, -0.5, 0.0, 0.5];
    let vertex_bytes: Vec<u8> = positions
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let vertices = resources.define_buffer(BufferDesc::new(
        vertex_bytes.len() as u64,
        BufferUsage::VERTEX | BufferUsage::COPY_DST,
    )?)?;
    let layout = VertexBufferLayout::new(
        8,
        vec![VertexAttribute::new(0, VertexFormat::Float32x2, 0)],
    )?;
    let pipeline = resources.define_render_pipeline(RenderPipelineDesc::new(
        TextureFormat::Bgra8Unorm,
        PrimitiveTopology::TriangleList,
        layout,
        FragmentProgram::Solid,
        BlendState::REPLACE,
        RasterState::new(CullMode::None, FrontFace::CounterClockwise),
    )?)?;

    let mut encoder = CommandEncoder::new(&resources);
    encoder.write_buffer(vertices, 0, &vertex_bytes)?;
    let desc = RenderPassDesc::new(
        &resources,
        target,
        area,
        LoadOp::Clear(Color::rgba(0.0, 0.0, 0.0, 1.0)?),
        StoreOp::Store,
    )?;
    let mut pass = encoder.begin_render_pass(desc)?;
    pass.set_pipeline(pipeline)?;
    pass.set_vertex_buffer(vertices, 0)?;
    pass.set_uniforms(DrawUniforms::new(
        Transform::identity(),
        Color::rgba(0.8, 0.1, 0.2, 1.0)?,
    ))?;
    pass.draw(3, 0)?;
    pass.end()?;
    let commands = encoder.finish()?;

    // Hand &commands to an executor mapped to this table and target.id().
    assert_eq!(commands.command_count(), 7);
    Ok(())
}
```

The recorded order is `WriteBuffer`, `BeginRenderPass`, `SetPipeline`,
`SetVertexBuffer`, `SetUniforms`, `Draw`, `EndRenderPass`. For repeated frames,
keep the table, resource IDs and mapped session; create a new encoder without
adding duplicate definitions. The [platform setup](reference.md#platform-setup)
and [execution reference](reference.md#execution-and-completion) describe how
to materialize the target, submit this stream and observe its completion.

## Programmable graphics and compute

The fixed `FragmentProgram` API remains available. Programmable pipelines use
separate descriptor and reference types, so backends supporting only the fixed
subset can reject programmable command streams before executing any command.
The WGPU backend lowers the programmable subset. A native backend must advertise
and implement a feature before accepting it; shader acceptance is not a promise
that every SPIR-V capability or every device format is supported.

`ShaderModuleDesc::spirv(Vec<u32>)` owns SPIR-V 1.0–1.6 words and validates the
header, bounded instruction framing, and vertex/fragment/compute entry-point
metadata. The cap is `MAX_SPIRV_WORDS` (1,048,576 words). This is structural
validation: the execution backend must parse and semantically validate all shader
instructions and capabilities before submission. `ShaderModuleDesc::wgsl(String)`
owns nonempty, NUL-free WGSL up to `MAX_WGSL_BYTES` (4 MiB); full WGSL syntax,
entry-point existence, and stage validation belong to the backend. Source getters
return borrowed data, while resource-table descriptor lookups return owned clones.

| Constructor or method | Meaning |
| --- | --- |
| `table.define_shader_module(desc)` | Define a module and obtain `ShaderModuleRef`. |
| `ShaderEntryPoint::new(module, stage, entry_point: String)` | Select `ShaderStage::Vertex`, `Fragment`, or `Compute`. SPIR-V entry-point stage/name is checked here. |
| `BindGroupLayoutEntry::new(binding, visibility, ty)` | Describe one binding; visibility is `ShaderStages::VERTEX`, `FRAGMENT`, `COMPUTE`, or their union. |
| `BindGroupLayoutDesc::new(entries)` | Validate and sort unique binding numbers in `0..16`. Empty layouts are valid. |
| `PipelineLayoutDesc::new(bind_groups)` | Define up to four descriptor sets, in set-number order. |
| `BindGroupDesc::new(&table, layout, entries)` | Validate exact layout coverage, resource ownership, types, usage, alignment, and byte ranges. |
| `table.define_bind_group(desc)` | Retain immutable bindings, rechecking ownership. |
| `ComputePipelineDesc::new(shader, layout)` | Describe one compute entry point and its resource interface. |
| `table.define_compute_pipeline(desc)` | Retain a compute pipeline with a table-owned shader module. |
| `ProgrammableRenderPipelineDesc::new(vertex, fragment, layout, target_format, vertex_buffer, topology, blend, raster)` | Describe vertex/fragment shaders, one color target, and fixed raster state. `vertex_buffer: None` allows geometry generated from vertex index. |
| `desc.with_depth_state(depth)` | Add the existing `Depth32Float` comparison/write state. |
| `table.define_programmable_render_pipeline(desc)` | Retain a programmable render pipeline. |

All new reference types expose `.id()` and `.slot()`. Persistent identities resolve
with `table.shader_module_ref`, `bind_group_ref`, `compute_pipeline_ref`, or
`programmable_render_pipeline_ref`. Descriptors retain table-qualified IDs;
defining a pipeline or group in another table fails with `ResourceTableMismatch`.

`BindGroupEntry::new(binding, resource)` binds one of:

| `BindingType` | `BindingResource` | Required usage |
| --- | --- | --- |
| `UniformBuffer` | `Buffer { buffer: BufferId, offset, size }` | `BufferUsage::UNIFORM` |
| `StorageBuffer { read_only }` | `Buffer { buffer: BufferId, offset, size }` | `BufferUsage::STORAGE` |
| `SampledTexture` | `Texture(TextureId)` | `TextureUsage::SAMPLED`; filterable 2D color format |
| `Sampler` | `Sampler(SamplerId)` | Existing portable filtering sampler |
| `StorageTexture { format, access: WriteOnly }` | `Texture(TextureId)` | `TextureUsage::STORAGE`; `Rgba8Unorm` only |

Uniform/storage offsets are multiples of 256 bytes. Sizes are nonzero multiples
of four and must fit the buffer. Writable aliases of the same whole buffer or
texture are rejected within or across descriptor sets used by one command, even
if buffer byte ranges do not overlap. Sampling or storing an active color/depth
attachment is rejected as attachment feedback. Read-only aliases are valid.
Device-specific size limits and shader/layout compatibility are checked by the
backend. Empty descriptor sets may remain unbound; a backend must synthesize them
if its native API requires an actual empty binding object.

New commands and recording methods:

| Command payload | Recording method |
| --- | --- |
| `SetProgrammablePipeline(ProgrammableRenderPipelineRef)` | `render_pass.set_programmable_pipeline(pipeline)` |
| `SetBindGroup { index, bind_group }` | `render_pass.set_bind_group(index, group)` or `compute_pass.set_bind_group(index, group)` |
| `BeginComputePass` | `encoder.begin_compute_pass()` |
| `SetComputePipeline(ComputePipelineRef)` | `compute_pass.set_pipeline(pipeline)` |
| `Dispatch { x, y, z }` | `compute_pass.dispatch(x, y, z)`; each workgroup count is `1..=65535` |
| `EndComputePass` | `compute_pass.end()` |
| `CopyBufferToBuffer { source, source_offset, destination, destination_offset, size }` | `encoder.copy_buffer_to_buffer(source, source_offset, destination, destination_offset, size)` |
| `ResourceBarrier(ResourceBarrier)` | `encoder.resource_barrier(barrier)` |

Programmable draws use the existing `draw` and `draw_indexed` commands. They require
matching descriptor sets but no fixed `DrawUniforms`, texture slot, or sampler
slot. Vertexless nonindexed draws require no vertex/index buffer. Indexed draws
always require an index buffer; a backend validates indirect vertex addresses
against opaque index data. Both draw forms retain positive triangle-list counts.
Copies require distinct buffers, copy usage, nonzero sizes, and four-byte-aligned
offsets/sizes. All new passes retain the bounded command encoder's reserved end
slot and explicit `end()` requirement.

### Access and barrier contract

The command stream is a single ordered queue. Existing upload/copy/render accesses
retain their automatic ordered dependency semantics. Shader storage writes require
an explicit whole-resource dependency before the next access to that resource in
the same command buffer. A missing dependency returns `MissingBarrier`; a barrier
with the wrong preceding shader-write access or a next command inconsistent with
its destination access returns `InvalidResourceAccess`. Failed commands do not
consume a dependency or alter the recorded stream.

```rust
encoder.resource_barrier(ResourceBarrier::Buffer {
    buffer: output,
    before: BufferAccess::StorageReadWrite,
    after: BufferAccess::CopySource,
})?;
encoder.copy_buffer_to_buffer(output, 0, staging, 0, byte_count)?;
```

`BufferAccess` is `CopyDestination`, `CopySource`, `Vertex`, `Index`, `Uniform`,
`StorageRead`, or `StorageReadWrite`. `TextureAccess` is `CopyDestination`,
`CopySource`, `Sampled`, `RenderAttachment`, or `StorageWrite`. Both sides must be
permitted by the resource's declared usage. A barrier's destination describes the
first actual following access; intervening state binding does not consume it.
Shader access scopes cover every shader stage. A sequence of barriers without an
intervening access must connect matching destination/source scopes.

Barriers are outside render/compute passes. End the pass before synchronizing and
starting the consumer pass. Repeated accesses after a storage write inside one
pass are rejected. Render passes also reject a writable storage alias of any
resource read by an earlier draw in that pass, matching whole-pass GPU resource
tracking. A merely bound index buffer does not count as an access in a nonindexed
draw. Validation tracks the current command buffer; execution backends retain
responsibility for ordering and memory visibility across submitted command
buffers, including imported resources. The first-consumer check does not relieve a backend
of automatic visibility transitions for later accesses: for example, a storage
read followed by a copy read still requires the earlier producer to be visible
to the copy operation. WGPU implements accepted dependencies with
its ordered command stream and automatic resource transitions.

### Owned recording and the implemented subset

`OwnedCommandBuffer::new(Vec<OwnedCommand>)` retains persistent IDs and owns upload
bytes without borrowing a resource table. Its source commands are inspectable and
may be malformed until validated. `owned.record(&table)` resolves IDs and replays
through `CommandEncoder`, returning a borrowed, fully validated `CommandBuffer`;
`owned.validate(&table)` performs the same checks without retaining the result.
Uploads borrow the owned source bytes during replay. Replaying does not introduce
another lowering IR or extend lifetimes with unsafe code. Owned render-pass,
depth-attachment, and barrier forms replace references with their persistent IDs.

This extension serves offscreen vertex/fragment rendering, descriptor-backed
uniforms and storage, compute-buffer workloads, RGBA8 storage images, and compute
to copy/draw dependencies. It retains one color attachment, optional depth, one
interleaved vertex buffer, single-sample 2D textures with exactly one mip and one
layer, direct draws/dispatches, and one queue. MRT, mip chains, texture arrays/3D
images, multisampling, descriptor arrays/indexing, dynamic descriptor offsets,
push constants, indirect commands, tessellation/geometry stages, presentation,
and queue-family ownership transfer are deferred. No descriptor encodes those
features, so they cannot be silently flattened into the supported subset.
