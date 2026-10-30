//! Private Scarlet/VirGL resource and queue dispatch.

use alloc::{rc::Rc, vec::Vec};
use gpu_raw::{Gpu, GpuQueryInfo};

use crate::completion::SubmitMode;
#[cfg(feature = "std")]
use scarlet_os::handle::{Handle, HandleError, HandleResult};
#[cfg(not(feature = "std"))]
use std::handle::{Handle, HandleError, HandleResult};

use crate::{
    Capabilities, Color, PipelineDesc, PixelRect, SourceAlpha, VertexClip4Color3, Viewport, virgl,
};
#[cfg(feature = "std")]
use scarlet_os::ipc::SharedMemory;
#[cfg(not(feature = "std"))]
use std::ipc::SharedMemory;

pub(crate) enum Device {
    Virgl(Rc<virgl::Device>),
}

impl Device {
    pub(crate) fn open(path: &str) -> HandleResult<Self> {
        let backend = Rc::new(virgl::Device::open(path)?);
        Ok(Self::Virgl(backend))
    }

    pub(crate) fn supports(info: &GpuQueryInfo) -> bool {
        virgl::Device::supports(info)
    }

    pub(crate) fn from_gpu(gpu: Gpu, info: GpuQueryInfo) -> HandleResult<Self> {
        let backend = Rc::new(virgl::Device::from_gpu(gpu, info)?);
        Ok(Self::Virgl(backend))
    }

    pub(crate) fn capabilities(&self) -> Capabilities {
        match self {
            Self::Virgl(device) => device.capabilities(),
        }
    }

    pub(crate) fn create_context(&self) -> HandleResult<Context> {
        match self {
            Self::Virgl(device) => Ok(Context::Virgl(Rc::new(device.create_context()?))),
        }
    }
}

#[derive(Clone)]
pub(crate) enum Context {
    Virgl(Rc<virgl::Context>),
}

impl Context {
    #[cfg(feature = "std")]
    pub(crate) fn begin_query(&self, kind: u32, mask: u32) -> HandleResult<()> {
        match self {
            Self::Virgl(c) => c.begin_query(kind, mask),
        }
    }
    #[cfg(feature = "std")]
    pub(crate) fn end_query(&self, kind: u32, count: usize) -> HandleResult<Vec<u64>> {
        match self {
            Self::Virgl(c) => c.end_query(kind, count),
        }
    }
    #[cfg(feature = "std")]
    pub(crate) fn timestamp_ns(&self) -> HandleResult<u64> {
        match self {
            Self::Virgl(c) => c.timestamp_ns(),
        }
    }

    #[cfg(feature = "backend-abi")]
    pub(crate) fn is_idle(&self) -> HandleResult<bool> {
        match self {
            Self::Virgl(context) => context.is_idle(),
        }
    }

    pub(crate) fn cleanup_failed(&self) -> bool {
        match self {
            Self::Virgl(context) => context.cleanup_failed(),
        }
    }
    pub(crate) fn abandon_cleanup(&self) {
        match self {
            Self::Virgl(context) => context.abandon_cleanup(),
        }
    }
    pub(crate) fn retire_ir_resources(&self, resources: &mut IrResources) -> HandleResult<()> {
        match (self, resources) {
            (Self::Virgl(context), IrResources::Virgl(resources)) => {
                context.retire_ir_resources(resources)
            }
        }
    }
    pub(crate) fn capabilities(&self) -> Capabilities {
        match self {
            Self::Virgl(context) => context.capabilities(),
        }
    }

    pub(crate) fn context_id(&self) -> i32 {
        match self {
            Self::Virgl(context) => context.context_id(),
        }
    }

    pub(crate) fn create_ir_resources(&self) -> HandleResult<IrResources> {
        match self {
            Self::Virgl(context) => Ok(IrResources::Virgl(context.create_ir_resources()?)),
        }
    }

    pub(crate) fn map_ir_image(
        &self,
        resources: &mut IrResources,
        texture: IrTextureSpec,
        image: &Image,
    ) -> HandleResult<()> {
        match (self, resources, image) {
            (Self::Virgl(context), IrResources::Virgl(resources), Image::Virgl(image)) => {
                context.map_ir_image(resources, texture, image)
            }
        }
    }

    pub(crate) fn map_ir_texture(
        &self,
        resources: &mut IrResources,
        texture: IrTextureSpec,
        image: &Texture,
    ) -> HandleResult<()> {
        match (self, resources, image) {
            (Self::Virgl(context), IrResources::Virgl(resources), Texture::Virgl(image)) => {
                context.map_ir_texture(resources, texture, image)
            }
        }
    }

    pub(crate) fn unmap_ir_texture(
        &self,
        resources: &mut IrResources,
        texture: IrTextureSpec,
        image: &Texture,
    ) -> HandleResult<()> {
        match (self, resources, image) {
            (Self::Virgl(context), IrResources::Virgl(resources), Texture::Virgl(image)) => {
                context.unmap_ir_texture(resources, texture, image)
            }
        }
    }

    pub(crate) fn readback_ir_texture(
        &self,
        resources: &mut IrResources,
        texture: IrTextureSpec,
        pixels: &mut [u8],
    ) -> HandleResult<()> {
        match (self, resources) {
            (Self::Virgl(context), IrResources::Virgl(resources)) => {
                context.readback_ir_texture(resources, texture, pixels)
            }
        }
    }

    pub(crate) fn create_image(&self, width: u32, height: u32) -> HandleResult<Image> {
        match self {
            Self::Virgl(context) => Ok(Image::Virgl(context.create_image(width, height)?)),
        }
    }

    pub(crate) fn create_shared_image(&self, width: u32, height: u32) -> HandleResult<Image> {
        match self {
            Self::Virgl(context) => Ok(Image::Virgl(context.create_shared_image(width, height)?)),
        }
    }

    pub(crate) fn create_sampled_bgra_texture(
        &self,
        width: u32,
        height: u32,
    ) -> HandleResult<Texture> {
        match self {
            Self::Virgl(context) => Ok(Texture::Virgl(
                context.create_sampled_bgra_texture(width, height)?,
            )),
        }
    }

    pub(crate) fn create_imported_bgra_texture(
        &self,
        shared_memory: &SharedMemory,
        width: u32,
        height: u32,
        shm_offset: usize,
        source_stride: u32,
    ) -> HandleResult<Texture> {
        match self {
            Self::Virgl(context) => Ok(Texture::Virgl(context.create_imported_bgra_texture(
                shared_memory,
                width,
                height,
                shm_offset,
                source_stride,
            )?)),
        }
    }

    pub(crate) fn import_shared_bgra_texture(
        &self,
        handle: Handle,
    ) -> HandleResult<(Texture, u32, u32)> {
        match self {
            Self::Virgl(context) => {
                let (texture, width, height) = context.import_shared_bgra_texture(handle)?;
                Ok((Texture::Virgl(texture), width, height))
            }
        }
    }

    pub(crate) fn upload_texture_bgra(
        &self,
        texture: &Texture,
        pixels: &[u8],
        source_stride: u32,
        damage: PixelRect,
    ) -> HandleResult<()> {
        match (self, texture) {
            (Self::Virgl(context), Texture::Virgl(texture)) => {
                context.upload_texture_bgra(texture, pixels, source_stride, damage)
            }
        }
    }

    pub(crate) fn transfer_imported_bgra_rect(
        &self,
        texture: &Texture,
        damage: PixelRect,
    ) -> HandleResult<()> {
        match (self, texture) {
            (Self::Virgl(context), Texture::Virgl(texture)) => {
                context.transfer_imported_bgra_rect(texture, damage)
            }
        }
    }

    pub(crate) fn readback_image_bgra(
        &self,
        image: &Image,
        destination: &mut [u8],
        destination_stride: u32,
        rect: PixelRect,
    ) -> HandleResult<()> {
        match (self, image) {
            (Self::Virgl(context), Image::Virgl(image)) => {
                context.readback_image_bgra(image, destination, destination_stride, rect)
            }
        }
    }

    pub(crate) fn release_texture(&self, texture: Texture) -> HandleResult<()> {
        match (self, texture) {
            (Self::Virgl(context), Texture::Virgl(texture)) => context.release_texture(texture),
        }
    }

    pub(crate) fn release_image(&self, image: Image) -> HandleResult<()> {
        match (self, image) {
            (Self::Virgl(context), Image::Virgl(image)) => context.release_image(image),
        }
    }

    pub(crate) fn create_pipeline(
        &self,
        image: &Image,
        description: PipelineDesc,
    ) -> HandleResult<Pipeline> {
        match (self, image) {
            (Self::Virgl(context), Image::Virgl(image)) => Ok(Pipeline::Virgl(
                context.create_pipeline(image, description)?,
            )),
        }
    }

    pub(crate) fn create_queue(&self) -> HandleResult<Queue> {
        match self {
            Self::Virgl(context) => Ok(Queue::Virgl(context.create_queue()?)),
        }
    }
}

pub(crate) enum Queue {
    Virgl(virgl::Queue),
}

impl Queue {
    #[cfg(feature = "backend-abi")]
    pub(crate) fn is_idle(&self) -> HandleResult<bool> {
        match self {
            Self::Virgl(queue) => queue.is_idle(),
        }
    }

    pub(crate) fn wait_idle(&self) -> HandleResult<()> {
        match self {
            Self::Virgl(queue) => queue.wait_idle(),
        }
    }
    pub(crate) fn materialize_ir_texture(
        &self,
        context: &Context,
        resources: &mut IrResources,
        texture: IrTextureSpec,
    ) -> HandleResult<()> {
        match (self, context, resources) {
            (Self::Virgl(queue), Context::Virgl(context), IrResources::Virgl(resources)) => {
                queue.materialize_ir_texture(context, resources, texture)
            }
        }
    }

    pub(crate) fn materialize_ir_pass(
        &self,
        context: &Context,
        resources: &mut IrResources,
        submission: &IrSubmission,
    ) -> HandleResult<()> {
        match (self, context, resources) {
            (Self::Virgl(queue), Context::Virgl(context), IrResources::Virgl(resources)) => {
                queue.materialize_ir_pass(context, resources, submission)
            }
        }
    }

    pub(crate) fn check_async_support(&self) -> HandleResult<usize> {
        match self {
            Self::Virgl(queue) => queue.check_async_support(),
        }
    }

    pub(crate) fn checkpoint(&self, mode: &mut SubmitMode) -> HandleResult<()> {
        match self {
            Self::Virgl(queue) => queue.checkpoint(mode),
        }
    }

    pub(crate) fn context_id(&self) -> i32 {
        match self {
            Self::Virgl(queue) => queue.context_id(),
        }
    }

    pub(crate) fn submit_ir(
        &self,
        context: &Context,
        resources: &mut IrResources,
        image: &Image,
        submission: &IrSubmission,
        mode: &mut SubmitMode,
    ) -> HandleResult<()> {
        match (self, context, resources, image) {
            (
                Self::Virgl(queue),
                Context::Virgl(context),
                IrResources::Virgl(resources),
                Image::Virgl(image),
            ) => queue.submit_ir(context, resources, image, submission, mode),
        }
    }

    pub(crate) fn submit_ir_internal(
        &self,
        context: &Context,
        resources: &mut IrResources,
        target: IrTextureSpec,
        submission: &IrSubmission,
        mode: &mut SubmitMode,
    ) -> HandleResult<()> {
        match (self, context, resources) {
            (Self::Virgl(queue), Context::Virgl(context), IrResources::Virgl(resources)) => {
                queue.submit_ir_internal(context, resources, target, submission, mode)
            }
        }
    }

    pub(crate) fn upload_ir_texture(
        &self,
        context: &Context,
        resources: &mut IrResources,
        texture: IrTextureSpec,
        upload: &IrTextureUpload,
        mode: &mut SubmitMode,
    ) -> HandleResult<()> {
        match (self, context, resources) {
            (Self::Virgl(queue), Context::Virgl(context), IrResources::Virgl(resources)) => {
                queue.upload_ir_texture(context, resources, texture, upload, mode)
            }
        }
    }

    pub(crate) fn prepare_ir_buffer(
        &self,
        context: &Context,
        resources: &mut IrResources,
        buffer: IrBufferSpec,
        source: &IrBufferSource<'_>,
        update: Option<&IrBufferUpdate>,
        mode: &mut SubmitMode,
    ) -> HandleResult<()> {
        match (self, context, resources) {
            (Self::Virgl(queue), Context::Virgl(context), IrResources::Virgl(resources)) => {
                queue.prepare_ir_buffer(context, resources, buffer, source, update, mode)
            }
        }
    }

    pub(crate) fn copy_ir_texture(
        &self,
        context: &Context,
        resources: &mut IrResources,
        copy: IrTextureCopy,
        mode: &mut SubmitMode,
    ) -> HandleResult<()> {
        match (self, context, resources) {
            (Self::Virgl(queue), Context::Virgl(context), IrResources::Virgl(resources)) => {
                queue.copy_ir_texture(context, resources, copy, mode)
            }
        }
    }

    pub(crate) fn submit(
        &self,
        image: &Image,
        viewport: Viewport,
        clear_color: Color,
        pipeline: &Pipeline,
        vertices: &[VertexClip4Color3],
    ) -> HandleResult<()> {
        match (self, image, pipeline) {
            (Self::Virgl(queue), Image::Virgl(image), Pipeline::Virgl(pipeline)) => {
                queue.submit(image, viewport, clear_color, pipeline, vertices)
            }
        }
    }

    pub(crate) fn submit_composition(
        &self,
        image: &Image,
        clear_color: Option<Color>,
        operations: &[CompositionOperation<'_>],
    ) -> HandleResult<()> {
        match (self, image) {
            (Self::Virgl(queue), Image::Virgl(image)) => {
                let mut backend_operations = Vec::new();
                backend_operations
                    .try_reserve(operations.len())
                    .map_err(|_| HandleError::OutOfResources)?;
                for operation in operations {
                    match operation {
                        CompositionOperation::Textured {
                            texture: Texture::Virgl(texture),
                            destination,
                            source,
                            opacity,
                            source_alpha,
                            clip,
                        } => backend_operations.push(virgl::CompositionOperation::Textured {
                            texture,
                            destination: *destination,
                            source: *source,
                            opacity: *opacity,
                            source_alpha: *source_alpha,
                            clip: *clip,
                        }),
                        CompositionOperation::Solid {
                            destination,
                            color,
                            clip,
                        } => backend_operations.push(virgl::CompositionOperation::Solid {
                            destination: *destination,
                            color: *color,
                            clip: *clip,
                        }),
                    }
                }
                queue.submit_composition(image, clear_color, &backend_operations)
            }
        }
    }
}

pub(crate) enum Image {
    Virgl(virgl::Image),
}

impl Image {
    pub(crate) fn ir_surface_initialized(&self) -> bool {
        match self {
            Self::Virgl(image) => image.ir_surface_initialized.get(),
        }
    }

    pub(crate) fn restore_ir_surface(&self, initialized: bool) {
        match self {
            Self::Virgl(image) => image.ir_surface_initialized.set(initialized),
        }
    }
    pub(crate) fn context_id(&self) -> i32 {
        match self {
            Self::Virgl(image) => image.context_id(),
        }
    }

    pub(crate) fn shared_handle(&self) -> &Handle {
        match self {
            Self::Virgl(image) => image.shared_handle(),
        }
    }
}

/// Maximum canonical vertices accepted in one persistent IR submission.
// Keep the expanded 40-byte canonical vertex stream below VirGL's 64 KiB
// opaque command limit after the inline write and draw state are included.
pub(crate) const MAX_IR_VERTICES: usize = 1_440;

/// Canonical vertex representation shared by portable IR lowering and drivers.
#[derive(Clone, Copy)]
pub(crate) struct IrVertex {
    pub(crate) position: [f32; 4],
    pub(crate) secondary: [f32; 4],
    pub(crate) tertiary: [f32; 2],
}

/// Persistent canonical vertex-buffer requirements for one logical IR slot.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct IrBufferSpec {
    pub(crate) slot: usize,
    pub(crate) size: u64,
    pub(crate) revision: u64,
}

/// Cumulative byte changes since `previous_revision`, an exact physical base.
/// `range` is the compatible envelope. Optional intervals omit unwritten gaps;
/// an empty interval list requires full repair after tracking overflows.
#[derive(Clone)]
pub(crate) struct IrBufferUpdate {
    pub(crate) previous_revision: u64,
    pub(crate) range: core::ops::Range<usize>,
    pub(crate) intervals: Option<IrBufferIntervals>,
}

/// One owned change layered over a buffer shadow's accepted base.
pub(crate) struct IrBufferOverlay<'a> {
    pub(crate) range: core::ops::Range<usize>,
    pub(crate) bytes: &'a [u8],
}

/// Read-only source for a single logical buffer preparation. Sparse sources
/// keep the accepted shadow untouched and materialize only the requested
/// upload range into the backend's bounded scratch buffer.
pub(crate) enum IrBufferSource<'a> {
    Contiguous(&'a [u8]),
    Sparse {
        base: Option<&'a [u8]>,
        overlays: &'a [IrBufferOverlay<'a>],
        len: usize,
    },
}

impl IrBufferSource<'_> {
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Contiguous(bytes) => bytes.len(),
            Self::Sparse { len, .. } => *len,
        }
    }

    /// Copy an exact logical range, applying overlays in write order.
    pub(crate) fn copy_range(
        &self,
        range: core::ops::Range<usize>,
        destination: &mut [u8],
    ) -> HandleResult<()> {
        if range.len() != destination.len() || range.end > self.len() {
            return Err(HandleError::InvalidParameter);
        }
        destination.fill(0);
        match self {
            Self::Contiguous(bytes) => {
                let source = bytes
                    .get(range.clone())
                    .ok_or(HandleError::InvalidParameter)?;
                destination.copy_from_slice(source);
            }
            Self::Sparse { base, overlays, .. } => {
                if let Some(base) = base {
                    copy_intersection(base, 0, range.clone(), destination);
                }
                for overlay in *overlays {
                    copy_intersection(
                        overlay.bytes,
                        overlay.range.start,
                        range.clone(),
                        destination,
                    );
                }
            }
        }
        Ok(())
    }

    pub(crate) fn direct_range(&self, range: core::ops::Range<usize>) -> Option<&[u8]> {
        let Self::Contiguous(bytes) = self else {
            return None;
        };
        bytes.get(range)
    }
}

fn copy_intersection(
    source: &[u8],
    source_start: usize,
    requested: core::ops::Range<usize>,
    destination: &mut [u8],
) {
    let first = requested.start.max(source_start);
    let last = requested.end.min(source_start.saturating_add(source.len()));
    if first >= last {
        return;
    }
    let destination_start = first - requested.start;
    let source_offset = first - source_start;
    let count = last - first;
    destination[destination_start..destination_start + count]
        .copy_from_slice(&source[source_offset..source_offset + count]);
}

pub(crate) const MAX_BUFFER_DIRTY_INTERVALS: usize = 128;

/// Nonempty lists always reserve the complete bound before admission. This
/// permits cumulative merging at accepted publication without allocating.
pub(crate) struct IrBufferIntervals {
    ranges: Vec<core::ops::Range<usize>>,
}

impl Clone for IrBufferIntervals {
    fn clone(&self) -> Self {
        let mut ranges = Vec::new();
        if !self.ranges.is_empty() && ranges.try_reserve_exact(MAX_BUFFER_DIRTY_INTERVALS).is_ok() {
            ranges.extend_from_slice(&self.ranges);
        }
        // Allocation failure conservatively produces the full-repair marker.
        Self { ranges }
    }
}

impl IrBufferIntervals {
    fn new(range: core::ops::Range<usize>) -> Self {
        let mut intervals = Self { ranges: Vec::new() };
        if intervals
            .ranges
            .try_reserve_exact(MAX_BUFFER_DIRTY_INTERVALS)
            .is_ok()
            && let Some(range) = Self::aligned(range)
        {
            intervals.ranges.push(range);
        }
        intervals
    }

    fn aligned(range: core::ops::Range<usize>) -> Option<core::ops::Range<usize>> {
        (range.start < range.end).then_some(())?;
        Some((range.start & !3)..(range.end.checked_add(3)? & !3))
    }

    pub(crate) fn ranges(&self) -> &[core::ops::Range<usize>] {
        &self.ranges
    }

    fn include(&mut self, range: core::ops::Range<usize>) {
        if self.ranges.is_empty() {
            return; // Full repair remains sticky until accepted preparation.
        }
        let Some(mut range) = Self::aligned(range) else {
            self.ranges.clear();
            return;
        };
        let first = self.ranges.partition_point(|old| old.end < range.start);
        let last = first + self.ranges[first..].partition_point(|old| old.start <= range.end);
        if first < last {
            range.start = range.start.min(self.ranges[first].start);
            range.end = range.end.max(self.ranges[last - 1].end);
            self.ranges.drain(first..last);
        } else if self.ranges.len() == MAX_BUFFER_DIRTY_INTERVALS {
            self.ranges.clear();
            return;
        }
        debug_assert!(self.ranges.capacity() >= MAX_BUFFER_DIRTY_INTERVALS);
        self.ranges.insert(first, range);
    }
}

impl IrBufferUpdate {
    pub(crate) fn sparse(previous_revision: u64, range: core::ops::Range<usize>) -> Self {
        Self {
            previous_revision,
            intervals: Some(IrBufferIntervals::new(range.clone())),
            range,
        }
    }

    pub(crate) fn include_range(&mut self, range: core::ops::Range<usize>) {
        self.range.start = self.range.start.min(range.start);
        self.range.end = self.range.end.max(range.end);
        if let Some(intervals) = &mut self.intervals {
            intervals.include(range);
        }
    }

    /// Called only when the earlier delta ends at our immediate predecessor.
    /// All storage is already reserved, including on the publication path.
    pub(crate) fn merge_previous(&mut self, previous: &Self) {
        self.previous_revision = previous.previous_revision;
        self.range.start = self.range.start.min(previous.range.start);
        self.range.end = self.range.end.max(previous.range.end);
        if previous
            .intervals
            .as_ref()
            .is_some_and(|ranges| ranges.ranges.is_empty())
        {
            if let Some(intervals) = &mut self.intervals {
                intervals.ranges.clear();
            } else {
                self.intervals = Some(IrBufferIntervals { ranges: Vec::new() });
            }
        } else if let Some(intervals) = &mut self.intervals {
            if let Some(earlier) = &previous.intervals {
                for range in &earlier.ranges {
                    intervals.include(range.clone());
                }
            } else {
                intervals.include(previous.range.clone());
            }
        }
        // A general planner's absent interval list keeps the envelope fallback.
    }
}

/// One draw's binding into a persistent canonical vertex buffer.
#[derive(Clone, Copy)]
pub(crate) struct IrVertexBufferBinding {
    pub(crate) buffer: IrBufferSpec,
    pub(crate) offset: u32,
}

/// Backend-neutral rectangle used by the private IR execution plan.
#[derive(Clone, Copy)]
pub(crate) struct IrRect {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

/// Portable fragment operation lowered by the IR executor.
#[derive(Clone, Copy)]
pub(crate) enum IrFragmentProgram {
    Solid,
    VertexColor,
    TextureRgba,
    TextureRgbIgnoreAlpha,
    TextureAlphaMask,
    TextureVertexColorRgba,
    TextureVertexColorRgbIgnoreAlpha,
    TextureVertexColorAlphaMask,
}

/// Portable blend factor carried without backend protocol constants.
#[derive(Clone, Copy)]
pub(crate) enum IrBlendFactor {
    Zero,
    One,
    SourceAlpha,
    OneMinusSourceAlpha,
    DestinationAlpha,
    OneMinusDestinationAlpha,
    SourceColor,
    OneMinusSourceColor,
    DestinationColor,
    OneMinusDestinationColor,
}

/// Portable blend arithmetic operation carried without backend protocol constants.
#[derive(Clone, Copy)]
pub(crate) enum IrBlendOp {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

/// Independent portable blend component.
#[derive(Clone, Copy)]
pub(crate) struct IrBlendComponent {
    pub(crate) source_factor: IrBlendFactor,
    pub(crate) destination_factor: IrBlendFactor,
    pub(crate) operation: IrBlendOp,
}

/// Exact independent color and alpha blending state.
#[derive(Clone, Copy)]
pub(crate) struct IrBlendState {
    pub(crate) color: IrBlendComponent,
    pub(crate) alpha: IrBlendComponent,
}

/// Portable culling selection used by the IR executor.
#[derive(Clone, Copy)]
pub(crate) enum IrCullMode {
    None,
    Front,
    Back,
}

/// Portable front-face selection used by the IR executor.
#[derive(Clone, Copy)]
pub(crate) enum IrFrontFace {
    Clockwise,
    CounterClockwise,
}

/// Portable depth comparison used by the IR executor.
#[derive(Clone, Copy)]
pub(crate) enum IrCompareFunction {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

/// Portable depth-test state used by the IR executor.
#[derive(Clone, Copy)]
pub(crate) struct IrDepthState {
    pub(crate) compare: IrCompareFunction,
    pub(crate) write_enabled: bool,
}

/// Persistent pipeline slot and its immutable portable state.
#[derive(Clone, Copy)]
pub(crate) struct IrPipelineState {
    pub(crate) depth_bias: [u32; 2],
    pub(crate) slot: usize,
    pub(crate) fragment: IrFragmentProgram,
    pub(crate) blend: IrBlendState,
    pub(crate) color_write_mask: u8,
    pub(crate) additional_color_blends: [Option<(IrBlendState, u8)>; 7],
    pub(crate) cull_mode: IrCullMode,
    pub(crate) front_face: IrFrontFace,
    pub(crate) depth: Option<IrDepthState>,
}

/// Portable sampler filtering mode.
#[derive(Clone, Copy)]
pub(crate) enum IrFilterMode {
    Nearest,
    Linear,
}

/// Portable sampler coordinate addressing mode.
#[derive(Clone, Copy)]
pub(crate) enum IrAddressMode {
    ClampToEdge,
    Repeat,
    MirrorRepeat,
    ClampToBorder,
}

/// Persistent sampler slot and its immutable portable state.
#[derive(Clone, Copy)]
pub(crate) struct IrSamplerState {
    pub(crate) slot: usize,
    pub(crate) min_filter: IrFilterMode,
    pub(crate) mag_filter: IrFilterMode,
    pub(crate) address_u: IrAddressMode,
    pub(crate) address_v: IrAddressMode,
    pub(crate) mip_filter: IrFilterMode,
    pub(crate) min_lod: f32,
    pub(crate) max_lod: f32,
    pub(crate) compare: Option<crate::ir::CompareFunction>,
    pub(crate) anisotropy: u8,
}

/// Draw-uniform constants sent to the GPU without CPU vertex transformation.
#[derive(Clone, Copy)]
pub(crate) struct IrUniforms {
    pub(crate) transform: [f32; 16],
    pub(crate) color: [f32; 4],
}

/// One private non-indexed draw in an ordered IR submission.
#[derive(Clone)]
pub(crate) struct IrDraw {
    pub(crate) draw_index: u32,
    pub(crate) programmable: Option<Rc<IrProgrammableDraw>>,
    pub(crate) start_vertex: usize,
    pub(crate) instance_count: u32,
    pub(crate) first_instance: u32,
    pub(crate) vertex_buffers: [Option<IrVertexBufferBinding>; 8],
    pub(crate) vertex_count: usize,
    pub(crate) vertex_buffer: Option<IrVertexBufferBinding>,
    pub(crate) pipeline: IrPipelineState,
    pub(crate) texture: Option<IrTextureSpec>,
    pub(crate) sampler: Option<IrSamplerState>,
    pub(crate) uniforms: IrUniforms,
    pub(crate) scissor: IrRect,
    pub(crate) viewport: Option<[f32; 6]>,
}

pub(crate) fn draw_count_valid(draw: &IrDraw) -> bool {
    match draw
        .programmable
        .as_ref()
        .map(|draw| draw.pipeline.topology)
    {
        Some(
            crate::ir::PrimitiveTopology::TriangleStrip | crate::ir::PrimitiveTopology::TriangleFan,
        ) => draw.vertex_count >= 3,
        Some(crate::ir::PrimitiveTopology::LineList) => {
            draw.vertex_count > 0 && draw.vertex_count.is_multiple_of(2)
        }
        Some(crate::ir::PrimitiveTopology::LineStrip) => draw.vertex_count >= 2,
        Some(crate::ir::PrimitiveTopology::PointList) => draw.vertex_count != 0,
        _ => draw.vertex_count > 0 && draw.vertex_count.is_multiple_of(3),
    }
}

/// Immutable compiled stages and vertex-fetch layout for one programmable pipeline.
pub(crate) struct IrProgrammablePipeline {
    pub(crate) slot: usize,
    #[cfg(feature = "programmable")]
    pub(crate) vertex: sgfx_codegen_virgl::programmable::CompiledShader,
    #[cfg(feature = "programmable")]
    pub(crate) fragment: sgfx_codegen_virgl::programmable::CompiledShader,
    pub(crate) vertex_buffers: Vec<crate::ir::VertexBufferLayout>,
    pub(crate) topology: crate::ir::PrimitiveTopology,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct IrIndexBufferBinding {
    pub(crate) buffer: IrBufferSpec,
    pub(crate) offset: u32,
    pub(crate) format: crate::ir::IndexFormat,
    pub(crate) base_vertex: i32,
}

/// Small constant banks are copied with a draw snapshot without a separate
/// allocation. Larger uniform blocks keep immutable shared storage.
#[derive(Clone, Debug)]
pub(crate) enum IrConstantWords {
    Inline { words: [u32; 32], len: usize },
    Shared(Rc<[u32]>),
}

impl IrConstantWords {
    pub(crate) fn as_slice(&self) -> &[u32] {
        match self {
            Self::Inline { words, len } => &words[..*len],
            Self::Shared(words) => words,
        }
    }
}

impl core::ops::Deref for IrConstantWords {
    type Target = [u32];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl PartialEq for IrConstantWords {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for IrConstantWords {}

#[cfg(test)]
impl From<Vec<u32>> for IrConstantWords {
    fn from(values: Vec<u32>) -> Self {
        if values.len() <= 32 {
            let mut words = [0; 32];
            words[..values.len()].copy_from_slice(&values);
            Self::Inline {
                words,
                len: values.len(),
            }
        } else {
            Self::Shared(values.into())
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct IrConstantBuffer {
    pub(crate) stage: crate::ir::ShaderStage,
    pub(crate) first_register: u32,
    pub(crate) words: IrConstantWords,
}

pub(crate) struct IrProgrammableDraw {
    pub(crate) pipeline: Rc<IrProgrammablePipeline>,
    pub(crate) index_buffer: Option<IrIndexBufferBinding>,
    pub(crate) constants: Rc<[IrConstantBuffer]>,
    pub(crate) textures: Rc<[IrTextureBinding]>,
    pub(crate) storage_buffers: Rc<[IrStorageBufferBinding]>,
}

#[derive(Clone, Copy)]
pub(crate) struct IrStorageBufferBinding {
    pub(crate) stage: crate::ir::ShaderStage,
    pub(crate) slot: u32,
    pub(crate) buffer: IrBufferSpec,
}

#[derive(Clone, Copy)]
pub(crate) struct IrTextureBinding {
    pub(crate) stage: crate::ir::ShaderStage,
    pub(crate) slot: u32,
    pub(crate) texture: IrTextureSpec,
    pub(crate) sampler: Option<IrSamplerState>,
}

/// Tightly packed physical texture upload retained until stream validation succeeds.
#[derive(Clone)]
pub(crate) struct IrTextureUpload {
    pub(crate) texture: IrTextureSpec,
    pub(crate) destination: IrRect,
    pub(crate) pixels: Vec<u8>,
    pub(crate) storage_format: IrTextureFormat,
    pub(crate) mip_level: u32,
    pub(crate) array_layer: u32,
}

/// Logical texture materialization requirements without backend identifiers.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct IrTextureSpec {
    pub(crate) slot: usize,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) mip_levels: u32,
    pub(crate) array_layers: u32,
    pub(crate) cube: bool,
    pub(crate) sampled: bool,
    pub(crate) render_attachment: bool,
    pub(crate) copy_destination: bool,
    pub(crate) present: bool,
    pub(crate) format: IrTextureFormat,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum IrTextureFormat {
    Bgra8,
    Rgba8,
    R8,
    Rg8,
    Depth32Float,
    Rgba16Float,
    Rgba32Float,
}

impl IrTextureFormat {
    pub(crate) const fn storage_format(self, native_narrow: bool) -> Self {
        match self {
            Self::R8 | Self::Rg8 if native_narrow => self,
            Self::Depth32Float | Self::Rgba16Float | Self::Rgba32Float => self,
            _ => Self::Bgra8,
        }
    }

    pub(crate) const fn bytes_per_texel(self) -> usize {
        match self {
            Self::R8 => 1,
            Self::Rg8 => 2,
            Self::Rgba16Float => 8,
            Self::Rgba32Float => 16,
            _ => 4,
        }
    }
}

/// One validated texture copy request without backend identifiers.
#[derive(Clone, Copy)]
pub(crate) struct IrTextureCopy {
    pub(crate) source: IrTextureSpec,
    pub(crate) source_rect: IrRect,
    pub(crate) destination: IrTextureSpec,
    pub(crate) destination_rect: IrRect,
    pub(crate) source_mip: u32,
    pub(crate) destination_mip: u32,
    /// None requests an exact copy; Some requests a filtered GPU blit.
    pub(crate) filter: Option<IrFilterMode>,
    pub(crate) flips: [bool; 2],
}

/// Complete backend-neutral render submission for one mapped presentation target.
pub(crate) struct IrSubmission {
    pub(crate) clear_color: Option<[f32; 4]>,
    pub(crate) additional_colors: Vec<IrColorAttachment>,
    pub(crate) depth_attachment: Option<IrTextureSpec>,
    pub(crate) depth_read_only: bool,
    pub(crate) clear_depth: Option<f32>,
    pub(crate) render_area: IrRect,
    pub(crate) vertices: Vec<IrVertex>,
    pub(crate) draws: Vec<IrDraw>,
    pub(crate) texture_uploads: Vec<IrTextureUpload>,
}

/// One additional color output in a render pass.
#[derive(Clone, Copy)]
pub(crate) struct IrColorAttachment {
    pub(crate) texture: IrTextureSpec,
    pub(crate) clear: Option<[f32; 4]>,
}

/// Private persistent materialization cache owned by the creating context.
pub(crate) enum IrResources {
    Virgl(virgl::IrResources),
}

impl IrResources {
    pub(crate) fn release_texture(&mut self, context: &Context, slot: usize) -> HandleResult<()> {
        match (self, context) {
            (Self::Virgl(resources), Context::Virgl(context)) => {
                resources.release_texture(context, slot)
            }
        }
    }

    pub(crate) fn release_buffer(&mut self, context: &Context, slot: usize) -> HandleResult<()> {
        match (self, context) {
            (Self::Virgl(resources), Context::Virgl(context)) => {
                resources.release_buffer(context, slot)
            }
        }
    }

    pub(crate) fn release_programmable_render_pipeline(
        &mut self,
        context: &Context,
        slot: usize,
    ) -> HandleResult<()> {
        match (self, context) {
            (Self::Virgl(resources), Context::Virgl(context)) => {
                resources.release_programmable_render_pipeline(context, slot)
            }
        }
    }

    pub(crate) fn unmap_ir_image(
        &mut self,
        context: &Context,
        texture: IrTextureSpec,
        image: &Image,
    ) -> HandleResult<()> {
        match (self, context, image) {
            (Self::Virgl(resources), Context::Virgl(context), Image::Virgl(image)) => {
                resources.unmap_ir_image(context, texture, image)
            }
        }
    }

    pub(crate) fn snapshot(&self) -> HandleResult<virgl::IrStateSnapshot> {
        match self {
            Self::Virgl(resources) => resources.snapshot(),
        }
    }

    pub(crate) fn restore(&mut self, snapshot: virgl::IrStateSnapshot) {
        match self {
            Self::Virgl(resources) => resources.restore(snapshot),
        }
    }
}

pub(crate) enum Texture {
    Virgl(virgl::Texture),
}

pub(crate) enum CompositionOperation<'a> {
    Textured {
        texture: &'a Texture,
        destination: PixelRect,
        source: PixelRect,
        opacity: f32,
        source_alpha: SourceAlpha,
        clip: Option<PixelRect>,
    },
    Solid {
        destination: PixelRect,
        color: Color,
        clip: Option<PixelRect>,
    },
}

pub(crate) enum Pipeline {
    Virgl(virgl::Pipeline),
}

#[cfg(test)]
mod tests {
    use gpu_raw::{
        GPU_DEVICE_STATE_READY, GPU_EXECUTION_SUPPORT_QUEUE, GPU_RESULT_SUCCESS, GpuQueryInfo,
    };

    use super::{Device, IrBufferUpdate, MAX_BUFFER_DIRTY_INTERVALS};
    use crate::{BACKEND_ID, matches_backend_id};

    #[test]
    fn matches_only_the_exact_virtio_gpu_backend_id() {
        assert!(matches_backend_id(b"virtio-gpu"));
        assert!(!matches_backend_id(b"virtio-gpu-extra"));
        assert!(!matches_backend_id(b"apple-agx"));
        assert!(!matches_backend_id(b""));
        assert!(!matches_backend_id(b"software"));
    }

    #[test]
    fn dirty_intervals_align_coalesce_and_merge_without_allocating() {
        let mut update = IrBufferUpdate::sparse(7, 5..6);
        let pointer = update.intervals.as_ref().unwrap().ranges.as_ptr();
        update.include_range(8..9);
        update.include_range(20..21);
        assert_eq!(
            update.intervals.as_ref().unwrap().ranges(),
            &[4..12, 20..24]
        );
        update.include_range(13..17);
        assert_eq!(update.intervals.as_ref().unwrap().ranges().len(), 1);
        assert_eq!(update.intervals.as_ref().unwrap().ranges()[0], 4..24);
        assert_eq!(update.range, 5..21);
        let earlier = IrBufferUpdate::sparse(5, 40..44);
        update.merge_previous(&earlier);
        assert_eq!(update.previous_revision, 5);
        assert_eq!(
            update.intervals.as_ref().unwrap().ranges(),
            &[4..24, 40..44]
        );
        assert_eq!(update.intervals.as_ref().unwrap().ranges.as_ptr(), pointer);
        let mut cloned = update.clone();
        let pointer = cloned.intervals.as_ref().unwrap().ranges.as_ptr();
        cloned.include_range(100..104);
        assert_eq!(cloned.intervals.as_ref().unwrap().ranges.as_ptr(), pointer);
        assert_eq!(cloned.intervals.as_ref().unwrap().ranges().len(), 3);
        assert_eq!(update.intervals.as_ref().unwrap().ranges().len(), 2);
    }

    #[test]
    fn dirty_interval_overflow_is_sticky_and_general_plans_keep_the_envelope() {
        let mut update = IrBufferUpdate::sparse(7, 0..4);
        for index in 1..MAX_BUFFER_DIRTY_INTERVALS {
            update.include_range(index * 8..index * 8 + 4);
        }
        assert_eq!(
            update.intervals.as_ref().unwrap().ranges().len(),
            MAX_BUFFER_DIRTY_INTERVALS
        );
        let pointer = update.intervals.as_ref().unwrap().ranges.as_ptr();
        // A touching interval at capacity still coalesces without overflow.
        update.include_range(1020..1024);
        assert_eq!(
            update.intervals.as_ref().unwrap().ranges().len(),
            MAX_BUFFER_DIRTY_INTERVALS
        );
        update.include_range(1032..1036);
        assert!(update.intervals.as_ref().unwrap().ranges().is_empty());
        update.include_range(0..2048);
        assert!(update.intervals.as_ref().unwrap().ranges().is_empty());
        assert_eq!(update.intervals.as_ref().unwrap().ranges.as_ptr(), pointer);
        let mut next = IrBufferUpdate::sparse(8, 4096..4100);
        next.merge_previous(&update);
        assert_eq!(next.previous_revision, 7);
        assert!(next.intervals.as_ref().unwrap().ranges().is_empty());

        let mut general = IrBufferUpdate {
            previous_revision: 9,
            range: 5000..5004,
            intervals: None,
        };
        general.merge_previous(&IrBufferUpdate::sparse(8, 4000..4004));
        assert!(general.intervals.is_none());
        assert_eq!(general.range, 4000..5004);
        general.merge_previous(&update);
        assert!(general.intervals.as_ref().unwrap().ranges().is_empty());
    }

    #[test]
    fn supports_requires_a_ready_exact_match_with_queue_execution() {
        let mut info = GpuQueryInfo::new();
        info.result = GPU_RESULT_SUCCESS;
        info.device_state = GPU_DEVICE_STATE_READY;
        info.execution_support = GPU_EXECUTION_SUPPORT_QUEUE;
        info.backend_id_len = BACKEND_ID.len() as u32;
        info.backend_id[..BACKEND_ID.len()].copy_from_slice(BACKEND_ID);
        assert!(Device::supports(&info));

        info.execution_support = 0;
        assert!(!Device::supports(&info));
    }
}
