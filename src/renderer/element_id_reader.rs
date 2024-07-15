use super::{
    config_renderer::TIMEOUT_NANOSECS,
    shader_interfaces::primitive_op_buffer::{PRIMITIVE_ID_BACKGROUND, PRIMITIVE_ID_BLEND},
    vulkan_init::create_cpu_read_staging_buffer,
};
use crate::engine::object::object::ObjectId;
use anyhow::Context;
use ash::{khr::synchronization2, vk};
use bort_vk::{
    AllocationAccess, Buffer, CommandBuffer, CommandPool, Device, DeviceOwned, Fence, Image,
    ImageAccess, ImageView, MemoryAllocator, Queue, Semaphore,
};
use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
pub enum ElementAtPoint {
    Object {
        object_id: ObjectId,
        primitive_op_index: usize,
    },
    Background,
    BlendArea {
        object_id: ObjectId,
    },
    // X, Y, Z manipulation ui elements
}

impl ElementAtPoint {
    pub fn from_rendered_id(rendered_id: u32) -> Self {
        match rendered_id {
            PRIMITIVE_ID_BACKGROUND => Self::Background,
            encoded_id => {
                let object_id_u32 = encoded_id >> 16;
                let object_id = ObjectId::from(object_id_u32 as u16);
                let primitive_op_index = (encoded_id & 0x0000FFFF) as usize;

                if primitive_op_index == PRIMITIVE_ID_BLEND as usize {
                    Self::BlendArea { object_id }
                } else {
                    Self::Object {
                        object_id,
                        primitive_op_index,
                    }
                }
            }
        }
    }
}

pub(super) struct ElementIdReader {
    device: Arc<Device>,
    synchronization_2_functions: synchronization2::Device,

    transfer_queue: Arc<Queue>,
    render_queue: Arc<Queue>,

    command_buffer_transfer: CommandBuffer,
    command_buffer_post_render_sync: CommandBuffer,
    command_buffer_pre_render_sync: CommandBuffer,

    completion_fence: Fence,
    semaphore_before_transfer: Semaphore,
    semaphore_after_transfer: Semaphore,

    cpu_read_staging_buffer: Buffer,
}

impl ElementIdReader {
    pub fn new(
        transfer_queue: Arc<Queue>,
        render_queue: Arc<Queue>,
        command_pool_transfer_queue: &Arc<CommandPool>,
        command_pool_render_queue: &Arc<CommandPool>,
        memory_allocator: Arc<MemoryAllocator>,
    ) -> anyhow::Result<Self> {
        let device = transfer_queue.device().clone();
        let synchronization_2_functions =
            synchronization2::Device::new(&device.instance().inner(), &device.inner());

        let command_buffer_transfer = command_pool_transfer_queue
            .allocate_command_buffer(vk::CommandBufferLevel::PRIMARY)
            .context("allocating buffer read transfer queue command buffer")?;

        let command_buffer_post_render_sync = command_pool_render_queue
            .allocate_command_buffer(vk::CommandBufferLevel::PRIMARY)
            .context("allocating buffer read render sync command buffer")?;

        let command_buffer_pre_render_sync = command_pool_render_queue
            .allocate_command_buffer(vk::CommandBufferLevel::PRIMARY)
            .context("allocating buffer read render sync command buffer")?;

        let completion_fence = Fence::new_unsignalled(device.clone()).context("creating fence")?;

        let semaphore_before_transfer =
            Semaphore::new(device.clone()).context("creating semaphore")?;
        let semaphore_after_transfer =
            Semaphore::new(device.clone()).context("creating semaphore")?;

        let cpu_read_staging_buffer = create_cpu_read_staging_buffer(memory_allocator)?;

        Ok(Self {
            device,
            synchronization_2_functions,

            transfer_queue,
            render_queue,

            command_buffer_transfer,
            command_buffer_post_render_sync,
            command_buffer_pre_render_sync,

            completion_fence,
            semaphore_before_transfer,
            semaphore_after_transfer,

            cpu_read_staging_buffer,
        })
    }

    /// Assumes that the render and transfer queue families are different, otherwise there's no
    /// need to call this function.
    pub fn record_and_submit_pre_transfer_sync_commands(
        &self,
        last_primitive_id_buffer: Arc<ImageView<Image>>,
    ) -> anyhow::Result<()> {
        let semaphores_before_transfer = [self.semaphore_before_transfer.handle()];

        let image_subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            level_count: 1,
            layer_count: 1,
            ..Default::default()
        };

        // render queue release
        let after_render_image_barrier = vk::ImageMemoryBarrier2 {
            src_stage_mask: vk::PipelineStageFlags2::FRAGMENT_SHADER,
            dst_stage_mask: vk::PipelineStageFlags2::empty(),
            src_access_mask: vk::AccessFlags2::INPUT_ATTACHMENT_READ,
            dst_access_mask: vk::AccessFlags2::empty(),
            old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            image: last_primitive_id_buffer.image().handle(),
            subresource_range: image_subresource_range,
            src_queue_family_index: self.render_queue.family_index(),
            dst_queue_family_index: self.transfer_queue.family_index(),
            ..Default::default()
        };

        let after_render_barriers = [after_render_image_barrier];
        let after_render_dependency =
            vk::DependencyInfo::default().image_memory_barriers(&after_render_barriers);

        let begin_info = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        self.command_buffer_post_render_sync
            .begin(&begin_info)
            .context("beginning command buffer record_and_submit_pre_transfer_sync_commands")?;

        unsafe {
            self.synchronization_2_functions.cmd_pipeline_barrier2(
                self.command_buffer_post_render_sync.handle(),
                &after_render_dependency,
            );
        }

        self.command_buffer_post_render_sync
            .end()
            .context("ending command buffer record_and_submit_pre_transfer_sync_commands")?;

        let submit_command_buffers = [self.command_buffer_post_render_sync.handle()];
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&submit_command_buffers)
            .signal_semaphores(&semaphores_before_transfer);

        unsafe {
            self.device
                .inner()
                .queue_submit(
                    self.render_queue.handle(),
                    &[submit_info],
                    vk::Fence::null(),
                )
                .context("submitting commands to sync reading primitive id at a coordinate")?;
        }

        Ok(())
    }

    pub fn record_primitive_id_copy_commands(
        &self,
        screen_coordinate: [f32; 2],
        last_primitive_id_buffer: Arc<ImageView<Image>>,
    ) -> anyhow::Result<()> {
        let different_queue_family_indices =
            self.render_queue.family_index() != self.transfer_queue.family_index();

        let image_subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            level_count: 1,
            layer_count: 1,
            ..Default::default()
        };

        let image_subresource_layers = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            layer_count: 1,
            ..Default::default()
        };

        let image_offset = vk::Offset3D {
            x: screen_coordinate[0].round() as i32,
            y: screen_coordinate[1].round() as i32,
            z: 0,
        };

        let image_extent = vk::Extent3D {
            width: 1,
            height: 1,
            depth: 1,
        };

        let buffer_image_copy_region = vk::BufferImageCopy {
            buffer_offset: 0,
            image_subresource: image_subresource_layers,
            image_offset,
            image_extent,
            ..Default::default()
        };

        let before_transfer_image_barrier = {
            let mut src_stage_mask = vk::PipelineStageFlags2::FRAGMENT_SHADER;
            let mut src_access_mask = vk::AccessFlags2::INPUT_ATTACHMENT_READ;

            if different_queue_family_indices {
                // this is a queue aquire operation https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VkImageMemoryBarrier
                // these values will be ignored by the driver, but we set them to null to stop the validation layers from freaking out
                src_stage_mask = vk::PipelineStageFlags2::empty();
                src_access_mask = vk::AccessFlags2::empty();
            }

            vk::ImageMemoryBarrier2 {
                src_stage_mask: src_stage_mask,
                dst_stage_mask: vk::PipelineStageFlags2::TRANSFER,
                src_access_mask: src_access_mask,
                dst_access_mask: vk::AccessFlags2::TRANSFER_READ,
                old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                image: last_primitive_id_buffer.image().handle(),
                subresource_range: image_subresource_range,
                src_queue_family_index: self.render_queue.family_index(),
                dst_queue_family_index: self.transfer_queue.family_index(),
                ..Default::default()
            }
        };

        let before_transfer_barriers = [before_transfer_image_barrier];
        let before_transfer_dependency =
            vk::DependencyInfo::default().image_memory_barriers(&before_transfer_barriers);

        let after_transfer_image_barrier = {
            let mut dst_stage_mask = vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT;
            let mut dst_access_mask = vk::AccessFlags2::COLOR_ATTACHMENT_WRITE;

            if different_queue_family_indices {
                // this is a queue release operation https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VkImageMemoryBarrier
                // these values will be ignored by the driver, but we set them to null to stop the validation layers from freaking out
                dst_stage_mask = vk::PipelineStageFlags2::empty();
                dst_access_mask = vk::AccessFlags2::empty();
            }

            vk::ImageMemoryBarrier2 {
                src_stage_mask: vk::PipelineStageFlags2::TRANSFER,
                dst_stage_mask: dst_stage_mask,
                src_access_mask: vk::AccessFlags2::TRANSFER_READ,
                dst_access_mask: dst_access_mask,
                old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                image: last_primitive_id_buffer.image().handle(),
                subresource_range: image_subresource_range,
                src_queue_family_index: self.transfer_queue.family_index(),
                dst_queue_family_index: self.render_queue.family_index(),
                ..Default::default()
            }
        };

        let after_transfer_barriers = [after_transfer_image_barrier];
        let after_transfer_dependency =
            vk::DependencyInfo::default().image_memory_barriers(&after_transfer_barriers);

        let begin_info = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        self.command_buffer_transfer
            .begin(&begin_info)
            .context("beginning command buffer get_element_at_screen_coordinate")?;

        unsafe {
            let command_buffer_handle = self.command_buffer_transfer.handle();

            self.synchronization_2_functions
                .cmd_pipeline_barrier2(command_buffer_handle, &before_transfer_dependency);

            self.device.inner().cmd_copy_image_to_buffer(
                command_buffer_handle,
                last_primitive_id_buffer.image().handle(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.cpu_read_staging_buffer.handle(),
                &[buffer_image_copy_region],
            );

            self.synchronization_2_functions
                .cmd_pipeline_barrier2(command_buffer_handle, &after_transfer_dependency);
        }

        self.command_buffer_transfer
            .end()
            .context("ending command buffer get_element_at_screen_coordinate")?;

        Ok(())
    }

    pub fn submit_primitive_id_copy_commands(&self) -> anyhow::Result<()> {
        let different_queue_family_indices =
            self.render_queue.family_index() != self.transfer_queue.family_index();

        let mut semaphores_before_transfer = Vec::<vk::Semaphore>::new();
        let mut semaphores_after_transfer = Vec::<vk::Semaphore>::new();
        if different_queue_family_indices {
            // only need semaphores to sync with other queue families
            semaphores_before_transfer.push(self.semaphore_before_transfer.handle());
            semaphores_after_transfer.push(self.semaphore_after_transfer.handle());
        }

        let transfer_submit_command_buffers = [self.command_buffer_transfer.handle()];

        let mut transfer_submit_info =
            vk::SubmitInfo::default().command_buffers(&transfer_submit_command_buffers);

        let different_queue_family_indices =
            self.render_queue.family_index() != self.transfer_queue.family_index();
        if different_queue_family_indices {
            transfer_submit_info = transfer_submit_info
                .wait_semaphores(&semaphores_before_transfer)
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::TRANSFER])
                .signal_semaphores(&semaphores_after_transfer);
        }

        self.completion_fence
            .reset()
            .context("resetting primitive id buffer reset fn")?;

        unsafe {
            self.device
                .inner()
                .queue_submit(
                    self.transfer_queue.handle(),
                    &[transfer_submit_info],
                    self.completion_fence.handle(),
                )
                .context("submitting commands to read primitive id at coordinate")?;
        }

        Ok(())
    }

    /// Assumes that the render and transfer queue families are different, otherwise there's no
    /// need to call this function.
    pub fn record_and_submit_post_transfer_sync_commands(
        &self,
        last_primitive_id_buffer: Arc<ImageView<Image>>,
    ) -> anyhow::Result<()> {
        let semaphores_after_transfer = [self.semaphore_after_transfer.handle()];

        let image_subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            level_count: 1,
            layer_count: 1,
            ..Default::default()
        };

        // render queue aquire
        let before_render_image_barrier = vk::ImageMemoryBarrier2 {
            src_stage_mask: vk::PipelineStageFlags2::TRANSFER,
            dst_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags2::TRANSFER_READ,
            dst_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            image: last_primitive_id_buffer.image().handle(),
            subresource_range: image_subresource_range,
            src_queue_family_index: self.transfer_queue.family_index(),
            dst_queue_family_index: self.render_queue.family_index(),
            ..Default::default()
        };

        let before_render_barriers = [before_render_image_barrier];
        let before_render_dependency =
            vk::DependencyInfo::default().image_memory_barriers(&before_render_barriers);

        let begin_info = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        self.command_buffer_pre_render_sync
            .begin(&begin_info)
            .context("beginning command buffer record_and_submit_pre_transfer_sync_commands")?;

        unsafe {
            self.synchronization_2_functions.cmd_pipeline_barrier2(
                self.command_buffer_pre_render_sync.handle(),
                &before_render_dependency,
            );
        }

        self.command_buffer_pre_render_sync
            .end()
            .context("ending command buffer record_and_submit_pre_transfer_sync_commands")?;

        let submit_command_buffers = [self.command_buffer_pre_render_sync.handle()];
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&submit_command_buffers)
            .wait_semaphores(&semaphores_after_transfer)
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT]);

        unsafe {
            self.device
                .inner()
                .queue_submit(
                    self.render_queue.handle(),
                    &[submit_info],
                    vk::Fence::null(),
                )
                .context("submitting commands to sync reading primitive id at a coordinate")?;
        }

        Ok(())
    }

    pub fn read_object_id_from_buffer(&mut self) -> anyhow::Result<ElementAtPoint> {
        self.completion_fence
            .wait(TIMEOUT_NANOSECS)
            .context("waiting for render id buffer copy fence")?;

        let rendered_id = self
            .cpu_read_staging_buffer
            .read_struct::<u32>(0)
            .context("reading render id")?;

        Ok(ElementAtPoint::from_rendered_id(rendered_id))
    }
}
