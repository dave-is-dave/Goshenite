use super::geometry_pass::descriptor;
use crate::engine::{
    aabb::AABB_VERTEX_COUNT,
    object::{
        object::{Object, ObjectId},
        objects_delta::{ObjectDeltaOperation, ObjectsDelta},
    },
};
use anyhow::Context;
use ash::{khr::synchronization2, prelude::VkResult, vk};
use bort_vk::{
    allocation_info_from_flags, AllocationAccess, AllocatorAccess, Buffer, BufferProperties,
    CommandBuffer, CommandPool, CommandPoolProperties, DescriptorPool, DescriptorPoolProperties,
    DescriptorSet, DescriptorSetLayout, Device, DeviceOwned, Fence, GraphicsPipeline,
    MemoryAllocator, PipelineAccess, Queue, Semaphore,
};
use bytemuck::NoUninit;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use std::sync::Arc;

const DESCRIPTOR_POOL_SIZE: u32 = 256;

/// Time to wait when we're just checking the status of a fence.
const CHECK_FENCE_TIMEOUT_NANOSECONDS: u64 = 100;

/// Manages per-object resources for the geometry pass
pub struct ObjectResourceManager {
    device: Arc<Device>,
    synchronization_2_functions: synchronization2::Device,

    memory_allocator: Arc<MemoryAllocator>,
    command_pool_transfer_queue: Arc<CommandPool>,
    command_pool_render_queue: Arc<CommandPool>,

    pending_upload_resources: Vec<BufferUploadResources>,
    available_upload_resources: Vec<BufferUploadResources>,

    objects_buffers: Vec<PerObjectResources>,

    descriptor_pools: Vec<Arc<DescriptorPool>>,
    primitive_ops_desc_set_layout: Arc<DescriptorSetLayout>,
}

impl ObjectResourceManager {
    pub fn new(
        memory_allocator: Arc<MemoryAllocator>,
        primitive_ops_desc_set_layout: Arc<DescriptorSetLayout>,
        transfer_queue_family_index: u32,
        render_queue_family_index: u32,
    ) -> anyhow::Result<Self> {
        let device = memory_allocator.device().clone();
        let synchronization_2_functions =
            synchronization2::Device::new(&device.instance().inner(), &device.inner());

        let initial_descriptor_pool = create_descriptor_pool(device.clone())?;
        let command_pool_transfer_queue =
            create_command_pool(device.clone(), transfer_queue_family_index)?;
        let command_pool_render_queue =
            create_command_pool(device.clone(), render_queue_family_index)?;

        Ok(Self {
            device,
            synchronization_2_functions,

            memory_allocator,
            command_pool_transfer_queue,
            command_pool_render_queue,

            pending_upload_resources: Vec::new(),
            available_upload_resources: Vec::new(),

            objects_buffers: Vec::new(),

            descriptor_pools: vec![initial_descriptor_pool],
            primitive_ops_desc_set_layout,
        })
    }

    pub fn update_objects(
        &mut self,
        objects_delta: ObjectsDelta,
        transfer_queue: &Queue,
        render_queue: &Queue,
    ) -> anyhow::Result<()> {
        let mut transfer_operation_resources = self.get_transfer_command_resources()?;

        transfer_operation_resources.begin_command_buffers()?;

        for (object_id, object_delta) in objects_delta {
            match object_delta {
                ObjectDeltaOperation::Add(object) => {
                    trace!("adding object id = {:?} to gpu buffer", object_id);
                    self.update_or_push(object_id, &object, &mut transfer_operation_resources)?;
                }
                ObjectDeltaOperation::Update(object) => {
                    trace!("updating object id = {:?} in gpu buffer", object_id);
                    self.update_or_push(object_id, &object, &mut transfer_operation_resources)?;
                }
                ObjectDeltaOperation::Remove => {
                    if let Some(_removed_index) = self.remove(object_id) {
                        trace!("removing object buffer id = {:?}", object_id);
                    } else {
                        info!(
                            "attempted to remove object id = {:?} from gpu buffer but not found!",
                            object_id
                        );
                    }
                }
            }
        }

        transfer_operation_resources.end_command_buffers()?;

        transfer_operation_resources
            .submit_upload_and_sync_commands(transfer_queue, render_queue)?;

        self.pending_upload_resources
            .push(transfer_operation_resources);

        Ok(())
    }

    pub fn draw_commands(&self, command_buffer: &CommandBuffer, pipeline: &GraphicsPipeline) {
        for per_object_buffers in self.objects_buffers.iter() {
            command_buffer.bind_descriptor_sets(
                vk::PipelineBindPoint::GRAPHICS,
                &pipeline.pipeline_layout(),
                1,
                [per_object_buffers.primitive_ops_descriptor_set.as_ref()],
                &[],
            );
            command_buffer.bind_vertex_buffers(0, [&per_object_buffers.bounding_mesh_buffer], &[0]);
            command_buffer.draw(per_object_buffers.bounding_mesh_vertex_count, 1, 0, 0);
        }
    }

    pub fn draw_bounding_box_commands(&self, command_buffer: &CommandBuffer) {
        for per_object_buffers in self.objects_buffers.iter() {
            command_buffer.bind_vertex_buffers(0, [&per_object_buffers.bounding_mesh_buffer], &[0]);
            command_buffer.draw(per_object_buffers.bounding_mesh_vertex_count, 1, 0, 0);
        }
    }

    /// Returns the vec index if the id was found and removed.
    pub fn remove(&mut self, id: ObjectId) -> Option<usize> {
        let index_res = self.get_index(id);
        if let Some(index) = index_res {
            self.objects_buffers.remove(index);
        }
        index_res
    }

    pub fn get_index(&self, object_id: ObjectId) -> Option<usize> {
        self.objects_buffers
            .iter()
            .position(|o| o.object_id == object_id)
    }

    pub fn object_count(&self) -> usize {
        self.objects_buffers.len()
    }
}

impl Drop for ObjectResourceManager {
    fn drop(&mut self) {
        trace!("dropping object resource manager...");
    }
}

// Private functions

impl ObjectResourceManager {
    fn allocate_primitive_ops_descriptor_set(&mut self) -> anyhow::Result<Arc<DescriptorSet>> {
        let descriptor_pool = self.descriptor_pools[self.descriptor_pools.len() - 1].clone();

        let alloc_res =
            DescriptorSet::new(descriptor_pool, self.primitive_ops_desc_set_layout.clone());

        let desc_set = match alloc_res {
            Err(alloc_err) => {
                if alloc_err == vk::Result::ERROR_OUT_OF_POOL_MEMORY {
                    // old pool ran out of space -> create a new descriptor pool
                    let new_descriptor_pool = create_descriptor_pool(self.device.clone())?;
                    self.descriptor_pools.push(new_descriptor_pool.clone());

                    DescriptorSet::new(
                        new_descriptor_pool,
                        self.primitive_ops_desc_set_layout.clone(),
                    )
                    .context("allocating primitive ops desc set after creating new pool")?
                } else {
                    return Err(alloc_err).context("allocating primitive ops desc set");
                }
            }
            Ok(desc_set) => desc_set,
        };

        Ok(Arc::new(desc_set))
    }

    fn get_transfer_command_resources(&mut self) -> anyhow::Result<BufferUploadResources> {
        self.check_and_reuse_pending_resources()?;

        let returned_resources =
            self.available_upload_resources
                .pop()
                .unwrap_or(BufferUploadResources::new(
                    self.device.clone(),
                    &self.command_pool_transfer_queue,
                    &self.command_pool_render_queue,
                )?);
        Ok(returned_resources)
    }

    fn check_and_reuse_pending_resources(&mut self) -> anyhow::Result<()> {
        let mut i: usize = 0;
        while i < self.pending_upload_resources.len() {
            let wait_res = self.pending_upload_resources[i]
                .completion_fence
                .wait(CHECK_FENCE_TIMEOUT_NANOSECONDS);
            if let Err(vk_res) = wait_res {
                match vk_res {
                    // timeout means commands are still executing, can't use this one so we move on.
                    vk::Result::TIMEOUT => {
                        i = i + 1;
                        continue;
                    }
                    // error!
                    _ => anyhow::bail!(vk_res),
                };
            }

            // else: commands finished executing, can reuse these resources.
            let mut command_resources = self.pending_upload_resources.remove(i);
            command_resources
                .reset()
                .context("resetting object buffer upload fence")?;
            self.available_upload_resources.push(command_resources);
        }
        Ok(())
    }

    fn update_or_push(
        &mut self,
        object_id: ObjectId,
        object: &Object,
        transfer_resources: &mut BufferUploadResources,
    ) -> anyhow::Result<()> {
        let primitive_ops_buffer = self
            .upload_primitive_ops(object_id, &object, transfer_resources)
            .context("initial upload object to buffer")?;

        if let Some(index) = self.get_index(object_id) {
            let bounding_mesh_buffer =
                self.upload_bounding_mesh(object_id, &object, transfer_resources)?;

            write_desc_set_primitive_ops(
                &self.objects_buffers[index].primitive_ops_descriptor_set,
                &primitive_ops_buffer,
            )?;

            self.objects_buffers[index].bounding_mesh_buffer = bounding_mesh_buffer;
            self.objects_buffers[index].bounding_mesh_vertex_count = AABB_VERTEX_COUNT as u32;
            self.objects_buffers[index].primitive_ops_buffer = primitive_ops_buffer;
        } else {
            let bounding_mesh_buffer =
                self.upload_bounding_mesh(object_id, &object, transfer_resources)?;

            let primitive_ops_descriptor_set = self.allocate_primitive_ops_descriptor_set()?;
            write_desc_set_primitive_ops(&primitive_ops_descriptor_set, &primitive_ops_buffer)?;

            let new_object = PerObjectResources {
                object_id,
                bounding_mesh_buffer,
                bounding_mesh_vertex_count: AABB_VERTEX_COUNT as u32,
                primitive_ops_buffer,
                primitive_ops_descriptor_set,
            };
            self.objects_buffers.push(new_object);
        }

        Ok(())
    }

    fn upload_bounding_mesh(
        &mut self,
        object_id: ObjectId,
        object: &Object,
        transfer_resources: &mut BufferUploadResources,
    ) -> anyhow::Result<Buffer> {
        trace!(
            "uploading bounding box vertices for object id = {:?} to gpu buffer",
            object_id
        );

        let data = object.aabb().vertices(object_id);

        self.upload_via_staging_buffer(
            transfer_resources,
            &data,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::PipelineStageFlags2::VERTEX_INPUT,
            vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
        )
    }

    fn upload_primitive_ops(
        &mut self,
        object_id: ObjectId,
        object: &Object,
        transfer_resources: &mut BufferUploadResources,
    ) -> anyhow::Result<Buffer> {
        trace!(
            "uploading primitive ops for object id = {:?} to gpu buffer",
            object_id
        );

        let data = object.encoded_primitive_ops(object_id);

        self.upload_via_staging_buffer(
            transfer_resources,
            &data,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_READ,
        )
    }

    fn upload_via_staging_buffer<I>(
        &mut self,
        transfer_resources: &mut BufferUploadResources,
        upload_data: &[I],
        buffer_usage_during_render: vk::BufferUsageFlags,
        render_dst_stage: vk::PipelineStageFlags2,
        render_dst_access_flags: vk::AccessFlags2,
    ) -> anyhow::Result<Buffer>
    where
        I: NoUninit,
    {
        let upload_data_size = std::mem::size_of_val(upload_data) as vk::DeviceSize;

        let new_buffer = {
            let buffer_props = BufferProperties::new_default(
                upload_data_size,
                buffer_usage_during_render | vk::BufferUsageFlags::TRANSFER_DST,
            );

            let alloc_info = allocation_info_from_flags(
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::MemoryPropertyFlags::empty(),
            );

            Buffer::new(self.memory_allocator.clone(), buffer_props, alloc_info)
                .context("creating geometry pass object data buffer")?
        };

        let mut staging_buffer = {
            let buffer_props =
                BufferProperties::new_default(upload_data_size, vk::BufferUsageFlags::TRANSFER_SRC);

            let alloc_info = allocation_info_from_flags(
                vk::MemoryPropertyFlags::HOST_VISIBLE,
                vk::MemoryPropertyFlags::HOST_COHERENT,
            );

            Buffer::new(self.memory_allocator.clone(), buffer_props, alloc_info)
                .context("creating geometry pass object staging buffer")?
        };

        staging_buffer
            .write_slice(upload_data, 0)
            .context("uploading geometry pass object data to staging buffer")?;

        self.record_buffer_copy_commands(
            transfer_resources,
            &new_buffer,
            &staging_buffer,
            upload_data_size,
            render_dst_stage,
            render_dst_access_flags,
        );

        transfer_resources.staging_buffers.push(staging_buffer);

        Ok(new_buffer)
    }

    fn record_buffer_copy_commands(
        &mut self,
        transfer_resources: &mut BufferUploadResources,
        new_buffer: &Buffer,
        staging_buffer: &Buffer,
        upload_data_size: vk::DeviceSize,
        render_dst_stage: vk::PipelineStageFlags2,
        render_dst_access_flags: vk::AccessFlags2,
    ) {
        let after_transfer_barrier = {
            let mut dst_stage_mask = render_dst_stage;
            let mut dst_access_mask = render_dst_access_flags;

            if transfer_resources.queue_ownership_transfer_required() {
                // this is a queue release operation https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VkImageMemoryBarrier
                // these values will be ignored by the driver, but we set them to null to stop the validation layers from freaking out
                dst_stage_mask = vk::PipelineStageFlags2::empty();
                dst_access_mask = vk::AccessFlags2::empty();
            }

            vk::BufferMemoryBarrier2 {
                src_stage_mask: vk::PipelineStageFlags2::TRANSFER,
                dst_stage_mask: dst_stage_mask,
                src_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                dst_access_mask: dst_access_mask,
                buffer: new_buffer.handle(),
                size: upload_data_size,
                offset: 0,
                src_queue_family_index: transfer_resources.transfer_queue_family_index,
                dst_queue_family_index: transfer_resources.render_queue_family_index,
                ..Default::default()
            }
        };

        let after_transfer_barriers = [after_transfer_barrier];
        let after_transfer_dependency =
            vk::DependencyInfo::default().buffer_memory_barriers(&after_transfer_barriers);

        let copy_region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: upload_data_size,
        };

        transfer_resources.command_buffer_transfer.copy_buffer(
            &staging_buffer,
            &new_buffer,
            &[copy_region],
        );

        unsafe {
            self.synchronization_2_functions.cmd_pipeline_barrier2(
                transfer_resources.command_buffer_transfer.handle(),
                &after_transfer_dependency,
            );
        }

        // sync with render queue (if necessary)

        if transfer_resources.queue_ownership_transfer_required() {
            // an identical queue aquire operation is required to complete the layout transition https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#synchronization-queue-transfers-acquire
            let before_render_barrier = vk::BufferMemoryBarrier2 {
                src_stage_mask: vk::PipelineStageFlags2::empty(),
                dst_stage_mask: render_dst_stage,
                src_access_mask: vk::AccessFlags2::empty(),
                dst_access_mask: render_dst_access_flags,
                buffer: new_buffer.handle(),
                size: upload_data_size,
                offset: 0,
                src_queue_family_index: transfer_resources.transfer_queue_family_index,
                dst_queue_family_index: transfer_resources.render_queue_family_index,
                ..Default::default()
            };

            let before_render_barriers = [before_render_barrier];
            let before_render_dependency =
                vk::DependencyInfo::default().buffer_memory_barriers(&before_render_barriers);

            unsafe {
                self.synchronization_2_functions.cmd_pipeline_barrier2(
                    transfer_resources.command_buffer_render_sync.handle(),
                    &before_render_dependency,
                );
            }
        }
    }
}

fn create_command_pool(
    device: Arc<Device>,
    queue_family_index: u32,
) -> anyhow::Result<Arc<CommandPool>> {
    let command_pool_props = CommandPoolProperties {
        flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        queue_family_index,
    };

    let command_pool = CommandPool::new(device, command_pool_props)
        .context("creating gui renderer command pool")?;

    Ok(Arc::new(command_pool))
}

fn create_descriptor_pool(device: Arc<Device>) -> anyhow::Result<Arc<DescriptorPool>> {
    let descriptor_pool_props = DescriptorPoolProperties {
        max_sets: DESCRIPTOR_POOL_SIZE,
        pool_sizes: vec![vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: DESCRIPTOR_POOL_SIZE,
        }],
        ..Default::default()
    };

    let descriptor_pool = DescriptorPool::new(device, descriptor_pool_props)
        .context("creating geometry pass descriptor pool")?;

    Ok(Arc::new(descriptor_pool))
}

fn write_desc_set_primitive_ops(
    descriptor_set: &DescriptorSet,
    primitive_op_buffer: &Buffer,
) -> anyhow::Result<()> {
    let primitive_ops_buffer_info = vk::DescriptorBufferInfo {
        buffer: primitive_op_buffer.handle(),
        offset: 0,
        range: primitive_op_buffer.properties().size,
    };
    let buffer_infos = [primitive_ops_buffer_info];

    let descriptor_write = vk::WriteDescriptorSet::default()
        .dst_set(descriptor_set.handle())
        .dst_binding(descriptor::BINDING_PRIMITIVE_OPS)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(&buffer_infos);

    descriptor_set
        .device()
        .update_descriptor_sets([descriptor_write], []);

    Ok(())
}

// ~~ Helper Structs ~~

struct PerObjectResources {
    pub object_id: ObjectId,
    pub bounding_mesh_buffer: Buffer,
    pub bounding_mesh_vertex_count: u32,
    pub primitive_ops_buffer: Buffer,
    pub primitive_ops_descriptor_set: Arc<DescriptorSet>,
}

struct BufferUploadResources {
    pub command_buffer_transfer: Arc<CommandBuffer>,
    pub command_buffer_render_sync: Arc<CommandBuffer>,
    pub transfer_queue_family_index: u32,
    pub render_queue_family_index: u32,
    pub completion_fence: Arc<Fence>,
    pub semaphore_queue_sync: Arc<Semaphore>,
    pub staging_buffers: Vec<Buffer>,
}

impl BufferUploadResources {
    pub fn new(
        device: Arc<Device>,
        command_pool_transfer_queue: &Arc<CommandPool>,
        command_pool_render_queue: &Arc<CommandPool>,
    ) -> anyhow::Result<Self> {
        let command_buffer_transfer = Arc::new(
            command_pool_transfer_queue
                .allocate_command_buffer(vk::CommandBufferLevel::PRIMARY)
                .context("allocating object upload transfer command buffer")?,
        );

        let command_buffer_render_sync = Arc::new(
            command_pool_render_queue
                .allocate_command_buffer(vk::CommandBufferLevel::PRIMARY)
                .context("allocating object upload sync command buffer")?,
        );

        let completion_fence =
            Arc::new(Fence::new_unsignalled(device.clone()).context("creating fence")?);
        let semaphore_queue_sync =
            Arc::new(Semaphore::new(device.clone()).context("creating semaphore")?);

        let transfer_queue_family_index =
            command_pool_transfer_queue.properties().queue_family_index;
        let render_queue_family_index = command_pool_render_queue.properties().queue_family_index;

        Ok(Self {
            command_buffer_transfer,
            command_buffer_render_sync,
            transfer_queue_family_index,
            render_queue_family_index,
            completion_fence,
            semaphore_queue_sync,
            staging_buffers: Default::default(),
        })
    }

    /// cleans up buffers and resets the fence
    pub fn reset(&mut self) -> VkResult<()> {
        self.staging_buffers.clear();
        self.completion_fence.reset()?;
        Ok(())
    }

    pub fn queue_ownership_transfer_required(&self) -> bool {
        self.render_queue_family_index != self.transfer_queue_family_index
    }

    pub fn begin_command_buffers(&self) -> anyhow::Result<()> {
        let begin_info = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };

        self.command_buffer_transfer
            .begin(&begin_info)
            .context("beginning object transfer command buffer")?;
        self.command_buffer_render_sync
            .begin(&begin_info)
            .context("beginning object sync command buffer")?;

        Ok(())
    }

    pub fn end_command_buffers(&self) -> anyhow::Result<()> {
        self.command_buffer_transfer
            .end()
            .context("ending object transfer command buffer")?;
        self.command_buffer_render_sync
            .end()
            .context("ending object sync command buffer")?;

        Ok(())
    }

    pub fn submit_upload_and_sync_commands(
        &self,
        transfer_queue: &Queue,
        render_queue: &Queue,
    ) -> anyhow::Result<()> {
        let queue_ownership_transfer_required =
            transfer_queue.family_index() != render_queue.family_index();

        let sync_semaphores = if queue_ownership_transfer_required {
            vec![self.semaphore_queue_sync.handle()]
        } else {
            Vec::new()
        };

        let transfer_fence = if queue_ownership_transfer_required {
            None // fence signalled by render sync command buffer instead
        } else {
            Some(self.completion_fence.as_ref())
        };

        let transfer_command_buffers = [self.command_buffer_transfer.handle()];

        let transfer_submit_info = vk::SubmitInfo::default()
            .command_buffers(&transfer_command_buffers)
            .signal_semaphores(&sync_semaphores);

        transfer_queue
            .submit(&[transfer_submit_info], transfer_fence)
            .context("submitting geometry buffer upload commands")?;

        if queue_ownership_transfer_required {
            let render_sync_command_buffers = [self.command_buffer_render_sync.handle()];

            let render_sync_submit_info = vk::SubmitInfo::default()
                .command_buffers(&render_sync_command_buffers)
                .wait_semaphores(&sync_semaphores)
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::FRAGMENT_SHADER]);

            render_queue
                .submit(
                    &[render_sync_submit_info],
                    Some(self.completion_fence.as_ref()),
                )
                .context("submitting object data upload render sync commands")?;
        }

        Ok(())
    }
}
