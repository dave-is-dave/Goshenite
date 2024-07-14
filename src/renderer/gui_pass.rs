//! shout out to https://github.com/hakolao/egui_winit_vulkano for a lot of this code

use super::{
    config_renderer::{SHADER_ENTRY_POINT, TIMEOUT_NANOSECS},
    shader_interfaces::{
        push_constants::GuiPushConstant,
        vertex_inputs::{EguiVertex, VulkanVertex},
    },
    vulkan_init::render_pass_indices,
};
use ahash::AHashMap;
use anyhow::Context;
use ash::{khr::synchronization2, prelude::VkResult, vk};
use bort_vk::{
    allocation_info_cpu_accessible, allocation_info_from_flags, default_subresource_layers,
    AllocationAccess, AllocatorAccess, Buffer, BufferProperties, ColorBlendState, CommandBuffer,
    CommandPool, DescriptorPool, DescriptorPoolProperties, DescriptorSet, DescriptorSetLayout,
    DescriptorSetLayoutBinding, DescriptorSetLayoutProperties, Device, DeviceOwned, DynamicState,
    Fence, GraphicsPipeline, GraphicsPipelineProperties, Image, ImageAccess, ImageDimensions,
    ImageProperties, ImageView, ImageViewAccess, ImageViewProperties, MemoryAllocator, MemoryPool,
    MemoryPoolPropeties, PipelineAccess, PipelineLayout, PipelineLayoutProperties, Queue,
    RenderPass, Sampler, SamplerProperties, Semaphore, ShaderModule, ShaderStage, ViewportState,
};
use bort_vma::Alloc;
use egui::{
    epaint::Primitive, ClippedPrimitive, Mesh, Rect, TextureFilter, TextureId, TextureOptions,
    TexturesDelta,
};
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use std::{
    ffi::CString,
    fmt::{self, Display},
    mem,
    sync::Arc,
};

/// Estimate of pretty busy gui: 8192 vertices and 16384 indices
const BUFFER_POOL_UPPER_SIZE: vk::DeviceSize =
    (8192 * mem::align_of::<egui::epaint::Vertex>() + 16384 * 4) as vk::DeviceSize;
const TEXTURE_FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;
const MAX_DESC_SETS_PER_POOL: u32 = 64;

mod descriptor {
    pub const SET_FONT_TEXTURE: usize = 0;
    pub const BINDING_FONT_TEXTURE: u32 = 0;
}

pub struct GuiPass {
    device: Arc<Device>,
    synchronization_2_functions: synchronization2::Device,

    memory_allocator: Arc<MemoryAllocator>,
    pipeline: GraphicsPipeline,

    /// Used for data transfers asynchronous to the rendering queue
    transfer_command_buffer: CommandBuffer,
    /// Used for pipeline barriers to sync with the render queue family
    render_sync_command_buffer: CommandBuffer,
    /// Used for data transfers on the rendering queue
    render_queue_command_buffer: CommandBuffer,

    texture_create_fence: Fence,
    texture_update_fence: Fence,
    render_sync_semaphore: Semaphore,

    descriptor_pools: Vec<Arc<DescriptorPool>>,
    unused_texture_desc_sets: Vec<DescriptorSet>,

    texture_samplers: SamplerVariations,
    texture_desc_sets_and_images:
        AHashMap<egui::TextureId, (DescriptorSet, ImageView<Image>, TextureOptions)>,

    buffer_pools: Vec<Arc<MemoryPool>>,
    /// Indicates which buffer pool to use next. E.g. two buffer pools have been created, but
    /// all the buffers have just been freed, so we'll start from the first pool again.
    current_buffer_pool_index: usize,
    vertex_buffers: Vec<Buffer>,
    index_buffers: Vec<Buffer>,
    texture_upload_buffers: Vec<Buffer>,

    // gui state
    scale_factor: f32,
    gui_primitives: Vec<ClippedPrimitive>,
}

// Public functions

impl GuiPass {
    /// Initializes the gui renderer
    pub fn new(
        memory_allocator: Arc<MemoryAllocator>,
        render_pass: &RenderPass,
        render_command_pool: Arc<CommandPool>,
        transfer_command_pool: Arc<CommandPool>,
        scale_factor: f32,
    ) -> anyhow::Result<Self> {
        let device = memory_allocator.device().clone();
        let synchronization_2_functions =
            synchronization2::Device::new(&device.instance().inner(), &device.inner());

        let transfer_command_buffer =
            CommandBuffer::new(transfer_command_pool, vk::CommandBufferLevel::PRIMARY)
                .context("allocating transfer command buffer")?;
        let render_sync_command_buffer =
            CommandBuffer::new(render_command_pool.clone(), vk::CommandBufferLevel::PRIMARY)
                .context("allocating render command buffer")?;
        let render_queue_command_buffer =
            CommandBuffer::new(render_command_pool, vk::CommandBufferLevel::PRIMARY)
                .context("allocating render command buffer")?;

        let descriptor_pool = create_descriptor_pool(device.clone())?;
        let desc_set_layout = create_descriptor_layout(device.clone())?;

        let pipeline_layout = create_pipeline_layout(device.clone(), desc_set_layout)?;
        let pipeline = create_pipeline(pipeline_layout, render_pass)?;

        let texture_samplers =
            SamplerVariations::new(device.clone()).context("creating gui texture samplers")?;
        let initial_buffer_pool = create_buffer_pool(memory_allocator.clone())?;

        let texture_create_fence =
            Fence::new_signalled(device.clone()).context("creating fence")?;
        let texture_update_fence =
            Fence::new_signalled(device.clone()).context("creating fence")?;
        let render_sync_semaphore =
            Semaphore::new(device.clone()).context("creating render sync semaphore")?;

        Ok(Self {
            device,
            synchronization_2_functions,

            memory_allocator,
            pipeline,

            transfer_command_buffer,
            render_sync_command_buffer,
            render_queue_command_buffer,

            texture_create_fence,
            texture_update_fence,
            render_sync_semaphore,

            descriptor_pools: vec![descriptor_pool],
            unused_texture_desc_sets: Vec::new(),

            texture_samplers,
            texture_desc_sets_and_images: AHashMap::default(),

            buffer_pools: vec![initial_buffer_pool],
            current_buffer_pool_index: 0,
            vertex_buffers: Vec::new(),
            index_buffers: Vec::new(),
            texture_upload_buffers: Vec::new(),

            scale_factor,
            gui_primitives: Vec::new(),
        })
    }

    /// Creates and/or removes texture resources as required by [`TexturesDelta`](epaint::Textures::TexturesDelta)
    /// output by [`egui::end_frame`](egui::context::Context::end_frame).
    ///
    /// New images are uploaded on the async transfer queue. Updating of existing images is done on
    /// the render queue because synchronization is required with the render queue before and after
    /// transfer anyway.
    pub fn update_textures(
        &mut self,
        textures_delta: Vec<TexturesDelta>,
        transfer_queue: &Queue,
        render_queue: &Queue,
    ) -> anyhow::Result<()> {
        if textures_delta.is_empty() {
            return Ok(());
        }

        // wait for previous texture uploads to complete execution so command buffers can be used
        self.wait_on_upload_fences()
            .context("waiting for texture upload fences")?;
        // we're sure that the previous texture uploads have completed so we can free old staging buffers
        self.texture_upload_buffers.clear();

        // begin command buffers
        let begin_info = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        self.transfer_command_buffer
            .begin(&begin_info)
            .context("beginning gui texture upload command buffer")?;
        self.render_sync_command_buffer
            .begin(&begin_info)
            .context("beginning gui render sync command buffer")?;
        self.render_queue_command_buffer
            .begin(&begin_info)
            .context("beginning gui texture update command buffer")?;

        let mut new_image_commands_recorded = false;
        let mut existing_image_commands_recorded = false;
        let mut upload_buffers = Vec::<Buffer>::new();

        // loop through texture update commands
        for textures_delta in textures_delta {
            // release unused texture resources
            for &id in &textures_delta.free {
                self.unregister_image(id);
            }

            // create new images and record upload commands
            for (id, image_delta) in textures_delta.set {
                let process_texture_data_res = self.process_texture_data(id, image_delta)?;

                new_image_commands_recorded |= process_texture_data_res.new_image_commands_recorded;
                existing_image_commands_recorded |=
                    process_texture_data_res.existing_image_commands_recorded;

                if let Some(upload_buffer) = process_texture_data_res.texture_staging_buffer {
                    upload_buffers.push(upload_buffer);
                }
            }
        }

        self.texture_upload_buffers.append(&mut upload_buffers);

        // end command buffers
        self.transfer_command_buffer
            .end()
            .context("ending gui texture upload command buffer")?;
        self.render_sync_command_buffer
            .end()
            .context("ending gui render sync command buffer")?;
        self.render_queue_command_buffer
            .end()
            .context("ending gui texture update command buffer")?;

        if new_image_commands_recorded {
            let queue_ownership_required =
                transfer_queue.family_index() != render_queue.family_index();

            self.submit_texture_creation_commands(
                queue_ownership_required,
                transfer_queue,
                render_queue,
            )?;
        }

        if existing_image_commands_recorded {
            self.submit_texture_update_commands(render_queue)?;
        }

        Ok(())
    }

    pub fn set_scale_factor(&mut self, scale_factor: f32) {
        self.scale_factor = scale_factor;
    }

    pub fn set_gui_primitives(&mut self, gui_primitives: Vec<ClippedPrimitive>) {
        self.gui_primitives = gui_primitives;
    }

    pub fn record_render_commands(
        &mut self,
        command_buffer: &CommandBuffer,
        write_linear_color: bool,
        framebuffer_dimensions: [f32; 2],
    ) -> anyhow::Result<()> {
        let push_constant_data = GuiPushConstant::new(
            [
                framebuffer_dimensions[0] / self.scale_factor,
                framebuffer_dimensions[1] / self.scale_factor,
            ],
            write_linear_color,
        );
        let push_constant_bytes = bytemuck::bytes_of(&push_constant_data);

        let viewport = vk::Viewport {
            x: 0.,
            y: 0.,
            width: framebuffer_dimensions[0],
            height: framebuffer_dimensions[1],
            min_depth: 0.,
            max_depth: 1.,
        };

        command_buffer.bind_pipeline(&self.pipeline);
        command_buffer.push_constants(
            self.pipeline.pipeline_layout().as_ref(),
            vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::VERTEX,
            0,
            push_constant_bytes,
        );
        command_buffer.set_viewport(0, &[viewport]);

        for ClippedPrimitive {
            clip_rect,
            primitive,
        } in self.gui_primitives.clone()
        {
            match primitive {
                Primitive::Mesh(mesh) => {
                    // nothing to draw if we don't have vertices & indices
                    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                        continue;
                    }

                    self.record_mesh_commands(
                        command_buffer,
                        mesh,
                        self.scale_factor,
                        framebuffer_dimensions,
                        clip_rect.clone(),
                    )?;
                }
                Primitive::Callback(_) => continue, // we don't need to support Primitive::Callback
            }
        }
        Ok(())
    }

    /// Fress vertex and index buffers created in previous calls to `record_render_commands`.
    /// Call this when gui rendering commands from the previous frame have finished.
    pub fn free_previous_vertex_and_index_buffers(&mut self) {
        self.vertex_buffers.clear();
        self.index_buffers.clear();
        self.current_buffer_pool_index = 0;
    }
}

impl Drop for GuiPass {
    fn drop(&mut self) {
        trace!("dropping gui pass...");
    }
}

// Private functions

impl GuiPass {
    fn wait_on_upload_fences(&mut self) -> VkResult<()> {
        self.device.wait_for_fences(
            [&self.texture_create_fence, &self.texture_update_fence],
            true,
            TIMEOUT_NANOSECS,
        )
    }

    fn submit_texture_creation_commands(
        &mut self,
        queue_ownership_transfer_required: bool,
        transfer_queue: &Queue,
        render_queue: &Queue,
    ) -> Result<(), anyhow::Error> {
        self.texture_create_fence
            .reset()
            .context("reseting gui texture creation fence")?;

        let sync_semaphores = if queue_ownership_transfer_required {
            vec![self.render_sync_semaphore.handle()] // sync with render queue
        } else {
            Vec::new() // implicit sync
        };

        let transfer_fence = if queue_ownership_transfer_required {
            None // fence signalled by render sync command buffer instead
        } else {
            Some(&self.texture_create_fence)
        };

        let transfer_command_buffers = [self.transfer_command_buffer.handle()];
        let transfer_submit_info = vk::SubmitInfo::default()
            .command_buffers(&transfer_command_buffers)
            .signal_semaphores(&sync_semaphores);

        transfer_queue
            .submit(&[transfer_submit_info], transfer_fence)
            .context("submitting gui texture creation commands")?;

        // todo this syncs with ALL fragment shader operations, but we only care about the gui fragment shader!
        // todo record as secondary command buffer and submit in record_render_commands?
        if queue_ownership_transfer_required {
            let render_sync_command_buffers = [self.render_sync_command_buffer.handle()];
            let render_sync_submit_info = vk::SubmitInfo::default()
                .command_buffers(&render_sync_command_buffers)
                .wait_semaphores(&sync_semaphores)
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::FRAGMENT_SHADER]);

            render_queue
                .submit(&[render_sync_submit_info], Some(&self.texture_create_fence))
                .context("submitting texture creation render queue sync commands")?;
        }

        Ok(())
    }

    fn submit_texture_update_commands(
        &mut self,
        render_queue: &Queue,
    ) -> Result<(), anyhow::Error> {
        self.texture_update_fence
            .reset()
            .context("reseting gui texture update fence")?;

        let command_buffers = [self.render_queue_command_buffer.handle()];
        let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);

        render_queue
            .submit(&[submit_info], Some(&self.texture_update_fence))
            .context("submitting gui texture update commands")?;

        Ok(())
    }

    /// Either updates an existing texture or creates a new one as required for `texture_id` with the
    /// data in `delta`. If commands were recorded to `command_buffer`, returns the buffer that will
    /// be used to upload the texture data. Otherwise returns `Ok(None)` if this update was skipped
    /// for some reason.
    fn process_texture_data(
        &mut self,
        texture_id: egui::TextureId,
        delta: egui::epaint::ImageDelta,
    ) -> anyhow::Result<ProcessTextureDataReturn> {
        // todo delta.options: TextureOptions mag/min filter for sampler

        let mut ret = ProcessTextureDataReturn {
            texture_staging_buffer: None,
            existing_image_commands_recorded: false,
            new_image_commands_recorded: false,
        };

        let image_bytes = egui_image_bytes(&delta.image, texture_id);

        if image_bytes.len() == 0 {
            info!(
                "attempted to create gui texture with no data! skipping... texture_id = {:?}",
                texture_id
            );
            return Ok(ret);
        }

        let upload_data_dimensions: [usize; 2] = match &delta.image {
            egui::ImageData::Color(image) => [image.width(), image.height()],
            egui::ImageData::Font(image) => [image.width(), image.height()],
        };

        // create buffer to be copied to the image
        let mut texture_staging_buffer = create_texture_staging_buffer(
            self.memory_allocator.clone(),
            std::mem::size_of_val(image_bytes.as_slice()) as u64,
        )?;
        texture_staging_buffer
            .write_slice(&image_bytes, 0)
            .context("uploading gui texture data to staging buffer")?;

        if let Some(update_pos) = delta.pos {
            if let Some((_, existing_image_view, _)) =
                self.texture_desc_sets_and_images.get(&texture_id)
            {
                // a subregion of an already allocated texture needs to be updated e.g. when a font size is changed

                let copy_region = vk::BufferImageCopy {
                    image_subresource: default_subresource_layers(vk::ImageAspectFlags::COLOR),
                    image_offset: vk::Offset3D {
                        x: update_pos[0] as i32,
                        y: update_pos[1] as i32,
                        z: 0,
                    },
                    image_extent: vk::Extent3D {
                        width: upload_data_dimensions[0] as u32,
                        height: upload_data_dimensions[1] as u32,
                        depth: 1,
                    },
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                };
                trace!(
                    "updating existing gui texture. id = {:?}, region offset = {:?}, region extent = {:?}",
                    texture_id, copy_region.image_offset, copy_region.image_extent
                );

                self.upload_existing_font_texture(
                    existing_image_view,
                    &texture_staging_buffer,
                    copy_region,
                );
                self.check_and_update_sampler(delta.options, texture_id)?;

                ret.existing_image_commands_recorded = true;
            }
        } else {
            // but usually `ImageDelta.pos` is `None` meaning a new image needs to be created
            trace!("creating new gui texture. id = {:?}", texture_id);

            self.create_new_texture(&texture_staging_buffer, delta, texture_id)?;

            ret.new_image_commands_recorded = true;
        }

        ret.texture_staging_buffer = Some(texture_staging_buffer);
        Ok(ret)
    }

    /// Unregister a texture that is no longer required by the gui.
    ///
    /// Helper function for [`Self::update_textures`]
    fn unregister_image(&mut self, texture_id: egui::TextureId) {
        trace!("removing unneeded gui texture id = {:?}", texture_id);
        let unused_desc_set = self.texture_desc_sets_and_images.remove(&texture_id);
        if let Some((unused_desc_set, _, _)) = unused_desc_set {
            self.unused_texture_desc_sets.push(unused_desc_set);
        }
    }

    /// Note: staging buffer commands are always used regardless of memory type because the image
    /// has optimal tiling.
    fn create_new_texture(
        &mut self,
        texture_staging_buffer: &Buffer,
        delta: egui::epaint::ImageDelta,
        texture_id: TextureId,
    ) -> anyhow::Result<()> {
        let transfer_queue_family_index = self
            .transfer_command_buffer
            .command_pool()
            .properties()
            .queue_family_index;
        let render_queue_family_index = self
            .render_sync_command_buffer
            .command_pool()
            .properties()
            .queue_family_index;

        let different_queue_family_indices =
            render_queue_family_index != transfer_queue_family_index;

        // create new image

        let new_image_properties = ImageProperties {
            format: TEXTURE_FORMAT,
            dimensions: ImageDimensions::new_2d(
                delta.image.width() as u32,
                delta.image.height() as u32,
            ),
            usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            sharing_mode: vk::SharingMode::EXCLUSIVE, // better performance depending (device dependant) particularly on mobile. queue ownership transfer only happens at image creation
            ..Default::default()
        };
        let new_image_allocation_info = allocation_info_from_flags(
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::empty(),
        );
        let new_image = Arc::new(
            Image::new(
                self.memory_allocator.clone(),
                new_image_properties.clone(),
                new_image_allocation_info,
            )
            .context("creating image for new egui texture")?,
        );

        let new_image_view_properties =
            ImageViewProperties::from_image_properties_default(&new_image_properties);
        let new_image_view = ImageView::new(new_image.clone(), new_image_view_properties)
            .context("creating image view for new egui texture")?;

        // copy data from staging buffer to image

        let copy_region = vk::BufferImageCopy {
            image_subresource: default_subresource_layers(vk::ImageAspectFlags::COLOR),
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::Extent3D {
                width: delta.image.width() as u32,
                height: delta.image.height() as u32,
                depth: 1,
            },
            ..Default::default()
        };

        // we need to transition the image layout to vk::ImageLayout::TRANSFER_DST_OPTIMAL
        let before_transfer_image_barrier = vk::ImageMemoryBarrier2 {
            src_stage_mask: vk::PipelineStageFlags2::empty(),
            dst_stage_mask: vk::PipelineStageFlags2::TRANSFER,
            src_access_mask: vk::AccessFlags2::empty(),
            dst_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            image: new_image_view.image().handle(),
            subresource_range: new_image_view.properties().subresource_range,
            ..Default::default()
        };

        let before_transfer_barriers = [before_transfer_image_barrier];
        let before_transfer_dependency =
            vk::DependencyInfo::default().image_memory_barriers(&before_transfer_barriers);

        // then transition to vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL (and perform queue release if transfer and render queue families are different)
        let after_transfer_image_barrier = {
            let mut dst_stage_mask = vk::PipelineStageFlags2::FRAGMENT_SHADER;
            let mut dst_access_mask = vk::AccessFlags2::SHADER_READ;

            if different_queue_family_indices {
                // this is a queue release operation https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VkImageMemoryBarrier
                // these values will be ignored by the driver, but we set them to null to stop the validation layers from freaking out
                dst_stage_mask = vk::PipelineStageFlags2::empty();
                dst_access_mask = vk::AccessFlags2::empty();
            }

            vk::ImageMemoryBarrier2 {
                src_stage_mask: vk::PipelineStageFlags2::TRANSFER,
                dst_stage_mask: dst_stage_mask,
                src_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                dst_access_mask: dst_access_mask,
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                image: new_image_view.image().handle(),
                subresource_range: new_image_view.properties().subresource_range,
                src_queue_family_index: transfer_queue_family_index,
                dst_queue_family_index: render_queue_family_index,
                ..Default::default()
            }
        };

        let after_transfer_barriers = [after_transfer_image_barrier];
        let after_transfer_dependency =
            vk::DependencyInfo::default().image_memory_barriers(&after_transfer_barriers);

        unsafe {
            self.synchronization_2_functions.cmd_pipeline_barrier2(
                self.transfer_command_buffer.handle(),
                &before_transfer_dependency,
            );
        }

        self.transfer_command_buffer.copy_buffer_to_image(
            texture_staging_buffer,
            new_image.as_ref(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[copy_region],
        );

        unsafe {
            self.synchronization_2_functions.cmd_pipeline_barrier2(
                self.transfer_command_buffer.handle(),
                &after_transfer_dependency,
            );
        }

        // sync with render queue (if necessary)

        if different_queue_family_indices {
            // an identical queue aquire operation is required to complete the layout transition https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#synchronization-queue-transfers-acquire
            let before_render_image_barrier = vk::ImageMemoryBarrier2 {
                src_stage_mask: vk::PipelineStageFlags2::empty(),
                dst_stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                src_access_mask: vk::AccessFlags2::empty(),
                dst_access_mask: vk::AccessFlags2::SHADER_READ,
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                image: new_image_view.image().handle(),
                subresource_range: new_image_view.properties().subresource_range,
                src_queue_family_index: transfer_queue_family_index,
                dst_queue_family_index: render_queue_family_index,
                ..Default::default()
            };

            let before_render_barriers = [before_render_image_barrier];
            let before_render_dependency =
                vk::DependencyInfo::default().image_memory_barriers(&before_render_barriers);

            unsafe {
                self.synchronization_2_functions.cmd_pipeline_barrier2(
                    self.render_sync_command_buffer.handle(),
                    &before_render_dependency,
                );
            }
        }

        // new descriptor set

        let sampler = self.texture_samplers.get_sampler(delta.options);
        let font_desc_set = self.get_new_font_texture_desc_set()?;
        write_desc_set_font_texture(&font_desc_set, &new_image_view, &sampler)?;

        self.texture_desc_sets_and_images
            .insert(texture_id, (font_desc_set, new_image_view, delta.options));

        Ok(())
    }

    fn upload_existing_font_texture(
        &self,
        existing_image_view: &ImageView<Image>,
        texture_data_buffer: &Buffer,
        copy_region: vk::BufferImageCopy,
    ) {
        // we need to transition the image layout to vk::ImageLayout::TRANSFER_DST_OPTIMAL
        let to_general_image_barrier = vk::ImageMemoryBarrier2 {
            src_stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
            dst_stage_mask: vk::PipelineStageFlags2::TRANSFER,
            src_access_mask: vk::AccessFlags2::SHADER_READ,
            dst_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
            old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            image: existing_image_view.image().handle(),
            subresource_range: existing_image_view.properties().subresource_range,
            ..Default::default()
        };

        let to_general_barriers = [to_general_image_barrier];
        let to_general_dependency =
            vk::DependencyInfo::default().image_memory_barriers(&to_general_barriers);

        // then transition back to vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        let to_shader_read_image_barrier = vk::ImageMemoryBarrier2 {
            src_stage_mask: vk::PipelineStageFlags2::TRANSFER,
            dst_stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
            src_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags2::SHADER_READ,
            old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            image: existing_image_view.image().handle(),
            subresource_range: existing_image_view.properties().subresource_range,
            ..Default::default()
        };

        let to_shader_read_barriers = [to_shader_read_image_barrier];
        let to_shader_read_dependency =
            vk::DependencyInfo::default().image_memory_barriers(&to_shader_read_barriers);

        // copy buffer to image
        unsafe {
            self.synchronization_2_functions.cmd_pipeline_barrier2(
                self.render_queue_command_buffer.handle(),
                &to_general_dependency,
            );
        }

        self.render_queue_command_buffer.copy_buffer_to_image(
            texture_data_buffer,
            existing_image_view.image().as_ref(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[copy_region],
        );

        unsafe {
            self.synchronization_2_functions.cmd_pipeline_barrier2(
                self.render_queue_command_buffer.handle(),
                &to_shader_read_dependency,
            );
        }
    }

    fn get_new_font_texture_desc_set(&mut self) -> anyhow::Result<DescriptorSet> {
        if let Some(existing_desc_set) = self.unused_texture_desc_sets.pop() {
            // reuse old desc set
            return Ok(existing_desc_set);
        }
        // else allocate new one
        return self.allocate_font_texture_desc_set();
    }

    fn allocate_font_texture_desc_set(&mut self) -> anyhow::Result<DescriptorSet> {
        let set_layout = self
            .pipeline
            .pipeline_layout()
            .properties()
            .set_layouts
            .get(descriptor::SET_FONT_TEXTURE)
            .context("indexing gui pipeline descriptor set layout")?;

        let allocate_res = self
            .descriptor_pools
            .get(self.descriptor_pools.len() - 1)
            .expect("should always be at least one descriptor pool")
            .allocate_descriptor_set(set_layout.clone());

        let desc_set = match allocate_res {
            Err(allocate_error) => match allocate_error {
                vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL => {
                    self.allocate_font_texture_desc_set_from_new_pool(set_layout.clone())?
                }
                _ => {
                    return Err(allocate_error)
                        .context("allocating descriptor set for new egui texture")
                }
            },
            Ok(desc_set) => desc_set,
        };

        Ok(desc_set)
    }

    fn allocate_font_texture_desc_set_from_new_pool(
        &mut self,
        set_layout: Arc<DescriptorSetLayout>,
    ) -> anyhow::Result<DescriptorSet> {
        // create a new descriptor pool
        let new_pool = create_descriptor_pool(self.device.clone())?;
        self.descriptor_pools.push(new_pool.clone());

        // try allocation again
        new_pool
            .allocate_descriptor_set(set_layout)
            .context("allocating descriptor set for new egui texture")
    }

    fn check_and_update_sampler(
        &mut self,
        texture_options: TextureOptions,
        texture_id: TextureId,
    ) -> anyhow::Result<()> {
        let Some((descriptor_set_ref, image_view, previous_texture_options)) =
            self.texture_desc_sets_and_images.get_mut(&texture_id)
        else {
            return Ok(());
        };
        if texture_options == *previous_texture_options {
            return Ok(());
        }
        let sampler = self.texture_samplers.get_sampler(texture_options);
        write_desc_set_font_texture(descriptor_set_ref, image_view, &sampler)
    }

    fn record_mesh_commands(
        &mut self,
        command_buffer: &CommandBuffer,
        mesh: Mesh,
        scale_factor: f32,
        framebuffer_dimensions: [f32; 2],
        clip_rect: Rect,
    ) -> anyhow::Result<()> {
        let index_count = mesh.indices.len() as u32;
        let texture_id = mesh.texture_id;

        let (vertex_buffer, index_buffer) = self.create_vertex_and_index_buffers(mesh)?;

        let scissor =
            calculate_gui_element_scissor(scale_factor, framebuffer_dimensions, clip_rect);

        let (desc_set, _, _) = self
            .texture_desc_sets_and_images
            .get(&texture_id)
            .ok_or(GuiRendererError::TextureDescSetMissing { id: texture_id })
            .context("recording gui render commands")?;

        command_buffer.set_scissor(0, &[scissor]);
        command_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline.pipeline_layout().as_ref(),
            0,
            [desc_set],
            &[],
        );
        command_buffer.bind_vertex_buffers(0, [&vertex_buffer], &[0]);
        command_buffer.bind_index_buffer(&index_buffer, 0, vk::IndexType::UINT32);

        command_buffer.draw_indexed(index_count, 1, 0, 0, 0);

        self.vertex_buffers.push(vertex_buffer);
        self.index_buffers.push(index_buffer);

        Ok(())
    }

    fn create_vertex_and_index_buffers(&mut self, mesh: Mesh) -> anyhow::Result<(Buffer, Buffer)> {
        let vertices = mesh
            .vertices
            .into_iter()
            .map(|egui_vertex| EguiVertex::from_egui_vertex(&egui_vertex))
            .collect::<Vec<_>>();

        let vertex_buffer_props = BufferProperties::new_default(
            mem::size_of_val(vertices.as_slice()) as vk::DeviceSize,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        );

        let index_buffer_props = BufferProperties::new_default(
            mem::size_of_val(mesh.indices.as_slice()) as vk::DeviceSize,
            vk::BufferUsageFlags::INDEX_BUFFER,
        );

        // create buffers

        let mut vertex_buffer = self.create_buffer_from_pools(vertex_buffer_props)?;

        let mut index_buffer = self.create_buffer_from_pools(index_buffer_props)?;

        // upload data

        // todo can avoid the vec clones here with moves! look at `gui::mesh_primitives` and `free_previous_vertex_and_index_buffers`

        vertex_buffer
            .write_slice(&vertices, 0)
            .context("uploading gui pass vertices")?;

        index_buffer
            .write_slice(&mesh.indices, 0)
            .context("uploading gui pass indices")?;

        Ok((vertex_buffer, index_buffer))
    }

    fn create_buffer_from_pools(
        &mut self,
        buffer_props: BufferProperties,
    ) -> anyhow::Result<Buffer> {
        // note: I think this ends up getting ignored anyway because we're allocating from a pool, not sure though... (https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/custom_memory_pools.html#choosing_memory_type_index)
        let buffer_alloc_info = allocation_info_from_flags(
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::empty(),
        );

        // a buffer pool may no longer have enough memory for the buffer allocation so we may need to try with other pools...
        loop {
            // try creating a buffer with an existing buffer pool
            let buffer_res = Buffer::new(
                self.buffer_pools[self.current_buffer_pool_index].clone(),
                buffer_props.clone(),
                buffer_alloc_info.clone(),
            );

            // `VK_ERROR_OUT_OF_DEVICE_MEMORY` means the vma pool has run out of space!
            if let Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) = buffer_res {
                debug!("creating new gui pass buffer pool");

                self.current_buffer_pool_index += 1;

                if self.current_buffer_pool_index < self.buffer_pools.len() {
                    // there are more buffer pools, lets try those...
                    continue;
                }

                // run out of existing pools, need to make a new one!
                let new_buffer_pool = create_buffer_pool(self.memory_allocator.clone())?;
                self.buffer_pools.push(new_buffer_pool.clone());

                // try one final time with new buffer pool
                break Buffer::new(new_buffer_pool, buffer_props, buffer_alloc_info)
                    .context("creating gui pass vertex/index buffer");
            }

            // stop loop at other errors or `Ok`
            break buffer_res.context("creating gui pass vertex/index buffer");
        }
    }
}

fn egui_image_bytes(image_data: &egui::ImageData, texture_id: TextureId) -> Vec<u8> {
    match image_data {
        egui::ImageData::Color(image) => {
            if image.width() * image.height() != image.pixels.len() {
                warn!(
                    "mismatch between gui color texture size and texel count. texture_id = {:?}",
                    texture_id
                );
            }
            image
                .pixels
                .iter()
                .flat_map(|color| color.to_array())
                .collect()
        }
        egui::ImageData::Font(image) => {
            if image.width() * image.height() != image.pixels.len() {
                warn!(
                    "mismatch between gui font texture size and texel count. texture_id = {:?}",
                    texture_id
                );
            }
            image
                .srgba_pixels(None)
                .flat_map(|color| color.to_array())
                .collect()
        }
    }
}

fn write_desc_set_font_texture(
    desc_set: &DescriptorSet,
    image_view: &ImageView<Image>,
    sampler: &Sampler,
) -> anyhow::Result<()> {
    let texture_info = vk::DescriptorImageInfo {
        image_view: image_view.handle(),
        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        sampler: sampler.handle(),
    };
    let texture_infos = [texture_info];

    let descriptor_write = vk::WriteDescriptorSet::default()
        .dst_set(desc_set.handle())
        .dst_binding(descriptor::BINDING_FONT_TEXTURE)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(&texture_infos);

    desc_set
        .device()
        .update_descriptor_sets([descriptor_write], []);

    Ok(())
}

fn create_descriptor_pool(device: Arc<Device>) -> anyhow::Result<Arc<DescriptorPool>> {
    let descriptor_pool_props = DescriptorPoolProperties {
        max_sets: MAX_DESC_SETS_PER_POOL,
        pool_sizes: vec![vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: MAX_DESC_SETS_PER_POOL,
        }],
        ..Default::default()
    };

    let descriptor_pool = DescriptorPool::new(device, descriptor_pool_props)
        .context("creating gui renderer descriptor pool")?;

    Ok(Arc::new(descriptor_pool))
}

fn create_buffer_pool(memory_allocator: Arc<MemoryAllocator>) -> anyhow::Result<Arc<MemoryPool>> {
    // may consider use device local with staging buffers because host visible + device local is a relatively
    // scarce resource on discrete cards https://asawicki.info/news_1740_vulkan_memory_types_on_pc_and_how_to_use_them
    // on the other hand we are writing to these buffers each frame so this is probably the optimal path
    let buffer_alloc_info = allocation_info_from_flags(
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::MemoryPropertyFlags::empty(),
    );

    let buffer_info = vk::BufferCreateInfo {
        size: BUFFER_POOL_UPPER_SIZE,
        usage: vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
        ..Default::default()
    };

    let memory_type_index = unsafe {
        memory_allocator
            .inner()
            .find_memory_type_index_for_buffer_info(&buffer_info, &buffer_alloc_info)
    }
    .context("finding memory type index for gui pass buffer pool")?;

    // using linear algorithm because we're creating new buffers each frame and freeing them all at once
    // https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/custom_memory_pools.html#linear_algorithm_free_at_once
    let pool_props = MemoryPoolPropeties {
        flags: bort_vma::AllocatorPoolCreateFlags::LINEAR_ALGORITHM,
        memory_type_index,
        ..Default::default()
    };

    let memory_pool = MemoryPool::new(memory_allocator, pool_props)
        .context("creating gui pass vertex/index buffer pool")?;
    Ok(Arc::new(memory_pool))
}

fn create_descriptor_layout(device: Arc<Device>) -> anyhow::Result<Arc<DescriptorSetLayout>> {
    let layout_props =
        DescriptorSetLayoutProperties::new_default(vec![DescriptorSetLayoutBinding {
            binding: descriptor::BINDING_FONT_TEXTURE,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        }]);

    let desc_layout = DescriptorSetLayout::new(device, layout_props)
        .context("creating gui pass descriptor set layout")?;
    Ok(Arc::new(desc_layout))
}

fn create_pipeline_layout(
    device: Arc<Device>,
    desc_set_layout_texture: Arc<DescriptorSetLayout>,
) -> anyhow::Result<Arc<PipelineLayout>> {
    let push_constant_range = vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::VERTEX,
        offset: 0,
        size: std::mem::size_of::<GuiPushConstant>() as u32,
    };

    let pipeline_layout_props =
        PipelineLayoutProperties::new(vec![desc_set_layout_texture], vec![push_constant_range]);

    let pipeline_layout = PipelineLayout::new(device, pipeline_layout_props)
        .context("creating gui pass pipeline layout")?;
    Ok(Arc::new(pipeline_layout))
}

fn create_pipeline(
    pipeline_layout: Arc<PipelineLayout>,
    render_pass: &RenderPass,
) -> anyhow::Result<GraphicsPipeline> {
    let (vert_stage, frag_stage) = create_shader_stages(pipeline_layout.device().clone())?;

    let dynamic_state =
        DynamicState::new_default(vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

    let viewport_state = ViewportState::new_dynamic(1, 1);

    let color_blend_attachment_state = vk::PipelineColorBlendAttachmentState {
        blend_enable: 1,
        src_color_blend_factor: vk::BlendFactor::ONE,
        dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::SRC_ALPHA,
        dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        alpha_blend_op: vk::BlendOp::ADD,
        color_write_mask: vk::ColorComponentFlags::RGBA,
        ..Default::default()
    };
    let color_blend_state = ColorBlendState::new_default(vec![color_blend_attachment_state]);

    let vertex_input_state = EguiVertex::vertex_input_state();

    let pipeline_properties = GraphicsPipelineProperties {
        subpass_index: render_pass_indices::SUBPASS_DEFERRED as u32,
        dynamic_state,
        color_blend_state,
        vertex_input_state,
        viewport_state,
        ..Default::default()
    };

    let pipeline = GraphicsPipeline::new(
        pipeline_layout,
        pipeline_properties,
        &[vert_stage, frag_stage],
        render_pass,
        None,
    )
    .context("creating gui pass pipeline")?;

    Ok(pipeline)
}

#[cfg(feature = "include-spirv-bytes")]
fn create_shader_stages<'a>(
    device: Arc<Device>,
) -> anyhow::Result<(ShaderStage<'a>, ShaderStage<'a>)> {
    let mut vertex_spv_file =
        std::io::Cursor::new(&include_bytes!("../../assets/shader_binaries/gui.vert.spv")[..]);
    let vert_shader = Arc::new(
        ShaderModule::new_from_spirv(device.clone(), &mut vertex_spv_file)
            .context("creating lighting pass vertex shader")?,
    );
    let vert_stage = ShaderStage::new(
        vk::ShaderStageFlags::VERTEX,
        vert_shader,
        CString::new(SHADER_ENTRY_POINT).context("converting shader entry point to c-string")?,
        None,
    );

    let mut frag_spv_file =
        std::io::Cursor::new(&include_bytes!("../../assets/shader_binaries/gui.frag.spv")[..]);
    let frag_shader = Arc::new(
        ShaderModule::new_from_spirv(device, &mut frag_spv_file)
            .context("creating lighting pass fragment shader")?,
    );
    let frag_stage = ShaderStage::new(
        vk::ShaderStageFlags::FRAGMENT,
        frag_shader,
        CString::new(SHADER_ENTRY_POINT).context("converting shader entry point to c-string")?,
        None,
    );

    Ok((vert_stage, frag_stage))
}

#[cfg(not(feature = "include-spirv-bytes"))]
fn create_shader_stages(device: &Arc<Device>) -> anyhow::Result<(ShaderStage, ShaderStage)> {
    const VERT_SHADER_PATH: &str = "assets/shader_binaries/gui.vert.spv";
    const FRAG_SHADER_PATH: &str = "assets/shader_binaries/gui.frag.spv";

    let vert_shader = Arc::new(
        ShaderModule::new_from_file(device.clone(), VERT_SHADER_PATH)
            .context("creating lighting pass vertex shader")?,
    );
    let vert_stage = ShaderStage::new(
        vk::ShaderStageFlags::VERTEX,
        vert_shader,
        CString::new(SHADER_ENTRY_POINT).context("converting shader entry point to c-string")?,
        None,
    );

    let frag_shader = Arc::new(
        ShaderModule::new_from_file(device.clone(), FRAG_SHADER_PATH)
            .context("creating lighting pass fragment shader")?,
    );
    let frag_stage = ShaderStage::new(
        vk::ShaderStageFlags::FRAGMENT,
        frag_shader,
        CString::new(SHADER_ENTRY_POINT).context("converting shader entry point to c-string")?,
        None,
    );

    Ok((vert_stage, frag_stage))
}

/// Caclulates the region of the framebuffer to render a gui element
fn calculate_gui_element_scissor(
    scale_factor: f32,
    framebuffer_dimensions: [f32; 2],
    rect: Rect,
) -> vk::Rect2D {
    let min = egui::Pos2 {
        x: rect.min.x * scale_factor,
        y: rect.min.y * scale_factor,
    };
    let min = egui::Pos2 {
        x: min.x.clamp(0.0, framebuffer_dimensions[0]),
        y: min.y.clamp(0.0, framebuffer_dimensions[1]),
    };
    let max = egui::Pos2 {
        x: rect.max.x * scale_factor,
        y: rect.max.y * scale_factor,
    };
    let max = egui::Pos2 {
        x: max.x.clamp(min.x, framebuffer_dimensions[0]),
        y: max.y.clamp(min.y, framebuffer_dimensions[1]),
    };
    vk::Rect2D {
        offset: vk::Offset2D {
            x: min.x.round() as i32,
            y: min.y.round() as i32,
        },
        extent: vk::Extent2D {
            width: (max.x.round() - min.x) as u32,
            height: (max.y.round() - min.y) as u32,
        },
    }
}

fn create_texture_staging_buffer(
    memory_allocator: Arc<MemoryAllocator>,
    size: vk::DeviceSize,
) -> anyhow::Result<Buffer> {
    let buffer_props = BufferProperties::new_default(size, vk::BufferUsageFlags::TRANSFER_SRC);
    let alloc_info = allocation_info_cpu_accessible();

    Buffer::new(memory_allocator, buffer_props, alloc_info).context("creating texture data buffer")
}

// ~~ Other stucts ~~

struct SamplerVariations {
    /// magnification = linear, minificaiton = linear
    pub l_mag_l_min: Arc<Sampler>,
    /// magnification = linear, minificaiton = nearest
    pub l_mag_n_min: Arc<Sampler>,
    /// magnification = nearest, minificaiton = linear
    pub n_mag_l_min: Arc<Sampler>,
    /// magnification = nearest, minificaiton = nearest
    pub n_mag_n_min: Arc<Sampler>,
}

impl SamplerVariations {
    pub fn new(device: Arc<Device>) -> VkResult<Self> {
        let mut sampler_props = SamplerProperties {
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            address_mode: [vk::SamplerAddressMode::CLAMP_TO_EDGE; 3],
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            ..Default::default()
        };
        let l_mag_l_min = Arc::new(Sampler::new(device.clone(), sampler_props)?);

        sampler_props.mag_filter = vk::Filter::LINEAR;
        sampler_props.min_filter = vk::Filter::NEAREST;
        let l_mag_n_min = Arc::new(Sampler::new(device.clone(), sampler_props)?);

        sampler_props.mag_filter = vk::Filter::NEAREST;
        sampler_props.min_filter = vk::Filter::LINEAR;
        let n_mag_l_min = Arc::new(Sampler::new(device.clone(), sampler_props)?);

        sampler_props.mag_filter = vk::Filter::NEAREST;
        sampler_props.min_filter = vk::Filter::NEAREST;
        let n_mag_n_min = Arc::new(Sampler::new(device.clone(), sampler_props)?);

        Ok(Self {
            l_mag_l_min,
            l_mag_n_min,
            n_mag_l_min,
            n_mag_n_min,
        })
    }

    pub fn get_sampler(&self, options: TextureOptions) -> Arc<Sampler> {
        match options.magnification {
            TextureFilter::Linear => match options.minification {
                TextureFilter::Linear => self.l_mag_l_min.clone(),
                TextureFilter::Nearest => self.l_mag_n_min.clone(),
            },
            TextureFilter::Nearest => match options.minification {
                TextureFilter::Linear => self.n_mag_l_min.clone(),
                TextureFilter::Nearest => self.n_mag_n_min.clone(),
            },
        }
    }
}

struct ProcessTextureDataReturn {
    texture_staging_buffer: Option<Buffer>,
    new_image_commands_recorded: bool,
    existing_image_commands_recorded: bool,
}

// ~~~ Errors ~~~

#[derive(Debug)]
pub enum GuiRendererError {
    /// Mesh requires a texture which doesn't exist (may have been prematurely destroyed or not yet created...)
    TextureDescSetMissing { id: TextureId },
}
impl std::error::Error for GuiRendererError {}
impl Display for GuiRendererError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::TextureDescSetMissing{id} =>
                write!(f, "Mesh requires texture [{:?}] which doesn't exist (may have been prematurely destroyed or not yet created...)", *id),
        }
    }
}

// ~~ Notes ~~

/*
typical gui vertex/index buffer composition:
index size = 4
indices = 3942
                    (546)  (1377)
indices mem total = 2184 + 5508 + 1080 + 456 + 5508 + 456 + 576
                  = 15768
vertex size = 32
vertices = 1264
                   (260)  (322)
vertex mem total = 8320 + 10304 + 4352 + 2048 + 10304 + 2432 + 2688
                 = 40448
*/
