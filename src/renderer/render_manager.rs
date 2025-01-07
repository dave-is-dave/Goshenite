use super::{
    config_renderer::{RenderOptions, TIMEOUT_NANOSECS},
    element_id_reader::{ElementAtPoint, ElementIdReader},
    geometry_pass::GeometryPass,
    gizmo_pass::GizmoPass,
    gui_pass::GuiPass,
    lighting_pass::LightingPass,
    overlay_pass::OverlayPass,
    shader_interfaces::camera_uniform_buffer::CameraUniformBuffer,
    vulkan_init::{
        choose_physical_device_and_queue_families, create_camera_ubo, create_clear_values,
        create_depth_buffer, create_framebuffers, create_normal_buffer, create_render_pass,
        create_swapchain, create_swapchain_image_views, swapchain_properties,
        ChoosePhysicalDeviceReturn, CreateDeviceAndQueuesReturn,
    },
};
use crate::{
    engine::object::objects_delta::ObjectsDelta,
    helper::anyhow_panic::log_anyhow_error_and_sources,
    renderer::vulkan_init::{
        choose_depth_buffer_format, create_albedo_buffer, create_command_pool,
        create_debug_callback, create_device_and_queue, create_entry, create_instance,
        create_primitive_id_buffers, create_render_command_buffers, get_display_handle,
        get_window_handle, shaders_should_write_linear_color,
    },
    user_interface::camera::Camera,
};
use anyhow::Context;
use ash::vk;
use bort_vk::{
    AllocationAccess, Buffer, CommandBuffer, CommandPool, DebugCallback, Device, Fence,
    Framebuffer, Image, ImageView, Instance, MemoryAllocator, Queue, RenderPass, Semaphore,
    Surface, Swapchain, SwapchainImage,
};
use egui::{ClippedPrimitive, TexturesDelta};
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use std::sync::Arc;
use winit::window::Window;

/// Contains Vulkan resources and methods to manage rendering
pub struct RenderManager {
    instance: Arc<Instance>,
    debug_callback: Option<Arc<DebugCallback>>,
    device: Arc<Device>,

    render_queue: Arc<Queue>,
    transfer_queue: Arc<Queue>,

    memory_allocator: Arc<MemoryAllocator>,
    command_pool_render: Arc<CommandPool>,
    command_pool_transfer: Arc<CommandPool>,

    window: Arc<Window>,
    surface: Arc<Surface>,
    swapchain: Arc<Swapchain>,
    swapchain_image_views: Vec<Arc<ImageView<SwapchainImage>>>,
    shaders_write_linear_color: bool,

    render_pass: Arc<RenderPass>,
    /// One per swapchain image, or two if there's only one swapchain image so we can store some
    /// images written to in the previous render
    framebuffers: Vec<Framebuffer>,
    /// One for each framebuffer attachment
    clear_values: Vec<vk::ClearValue>,

    depth_buffer: Arc<ImageView<Image>>,
    normal_buffer: Arc<ImageView<Image>>,
    albedo_buffer: Arc<ImageView<Image>>,
    /// One per framebuffer
    primitive_id_buffers: Vec<Arc<ImageView<Image>>>,
    camera_ubo: Buffer,

    geometry_pass: GeometryPass,
    gizmo_pass: GizmoPass,
    lighting_pass: LightingPass,
    overlay_pass: OverlayPass,
    gui_pass: GuiPass,

    object_id_reader: ElementIdReader,

    /// One per framebuffer
    render_command_buffers: Vec<CommandBuffer>,
    previous_render_fence: Fence,
    next_frame_wait_semaphore: Semaphore,
    swapchain_image_available_semaphore: Semaphore,

    renderer_state: RendererState,
    /// Indicates which framebuffer is being processed right now.
    framebuffer_index_currently_rendering: usize,
    /// Indicates which framebuffer was rendered to in the previous frame.
    framebuffer_index_last_rendered_to: usize,
    /// Can be set to true with [`Self::set_window_just_resized_flag`] and set to false in [`Self::render_frame`]
    window_just_resized: bool,
}

// Public functions

impl RenderManager {
    /// Initializes Vulkan resources. If renderer fails to initiver_minoralize, returns a string explanation.
    pub fn new(window: Arc<Window>, scale_factor: f32) -> anyhow::Result<Self> {
        let entry = create_entry()?;

        let display_handle = get_display_handle(&window)?;
        let window_handle = get_window_handle(&window)?;

        let instance = create_instance(entry.clone(), &display_handle)?;
        let debug_callback = create_debug_callback(&instance);

        let surface = Arc::new(
            Surface::new(
                &entry,
                instance.clone(),
                display_handle.as_raw(),
                window_handle.as_raw(),
            )
            .context("creating vulkan surface")?,
        );

        let ChoosePhysicalDeviceReturn {
            physical_device,
            render_queue_family_index,
            transfer_queue_family_index,
        } = choose_physical_device_and_queue_families(instance.clone(), &surface)?;
        let physical_device = Arc::new(physical_device);

        let CreateDeviceAndQueuesReturn {
            device,
            render_queue,
            transfer_queue,
        } = create_device_and_queue(
            physical_device.clone(),
            debug_callback.clone(),
            render_queue_family_index,
            transfer_queue_family_index,
        )?;

        let command_pool_render = create_command_pool(device.clone(), &render_queue)?;
        let command_pool_transfer = create_command_pool(device.clone(), &transfer_queue)?;

        let memory_allocator = Arc::new(MemoryAllocator::new(device.clone())?);

        let swapchain = create_swapchain(device.clone(), surface.clone(), &window)?;
        let shaders_write_linear_color =
            shaders_should_write_linear_color(swapchain.properties().surface_format);

        let swapchain_image_views = create_swapchain_image_views(&swapchain)?;
        let framebuffer_count = swapchain_image_views.len();
        debug!("swapchain image count = {}", framebuffer_count);

        let depth_buffer_format = choose_depth_buffer_format(&physical_device)?;

        let render_pass =
            create_render_pass(device.clone(), swapchain.properties(), depth_buffer_format)?;

        let framebuffer_dimensions = swapchain.properties().dimensions();
        let depth_buffer = create_depth_buffer(
            memory_allocator.clone(),
            framebuffer_dimensions,
            depth_buffer_format,
        )?;
        let normal_buffer = create_normal_buffer(memory_allocator.clone(), framebuffer_dimensions)?;
        let albedo_buffer = create_albedo_buffer(memory_allocator.clone(), framebuffer_dimensions)?;
        let primitive_id_buffers = create_primitive_id_buffers(
            framebuffer_count,
            memory_allocator.clone(),
            framebuffer_dimensions,
        )?;

        let camera_ubo = create_camera_ubo(memory_allocator.clone())?;

        let framebuffers = create_framebuffers(
            render_pass.clone(),
            &swapchain_image_views,
            normal_buffer.clone(),
            albedo_buffer.clone(),
            &primitive_id_buffers,
            depth_buffer.clone(),
        )?;

        let clear_values = create_clear_values();

        let geometry_pass = GeometryPass::new(
            device.clone(),
            memory_allocator.clone(),
            &render_pass,
            &camera_ubo,
            transfer_queue_family_index,
            render_queue_family_index,
        )?;

        let gizmo_pass = GizmoPass::new(memory_allocator.clone(), &render_pass, &camera_ubo)?;

        let lighting_pass = LightingPass::new(
            device.clone(),
            &render_pass,
            &camera_ubo,
            &normal_buffer,
            &albedo_buffer,
            &primitive_id_buffers,
        )?;

        let overlay_pass = OverlayPass::new(&render_pass, &camera_ubo)?;

        let gui_pass = GuiPass::new(
            memory_allocator.clone(),
            &render_pass,
            command_pool_render.clone(),
            command_pool_transfer.clone(),
            scale_factor,
        )?;

        let render_command_buffers = create_render_command_buffers(
            command_pool_render.clone(),
            swapchain_image_views.len() as u32,
        )?;

        let previous_render_fence =
            Fence::new_signalled(device.clone()).context("creating fence")?;
        let next_frame_wait_semaphore =
            Semaphore::new(device.clone()).context("creating per-frame semaphore")?;
        let swapchain_image_available_semaphore =
            Semaphore::new(device.clone()).context("creating per-swapchain-image semaphore")?;

        let object_id_reader = ElementIdReader::new(
            transfer_queue.clone(),
            render_queue.clone(),
            &command_pool_transfer,
            &command_pool_render,
            memory_allocator.clone(),
        )?;

        Ok(Self {
            instance,
            debug_callback,
            device,

            render_queue,
            transfer_queue,

            memory_allocator,
            command_pool_render,
            command_pool_transfer,

            window,
            surface,
            swapchain,
            swapchain_image_views,
            shaders_write_linear_color,

            render_pass,
            framebuffers,
            clear_values,

            depth_buffer,
            normal_buffer,
            albedo_buffer,
            primitive_id_buffers,
            camera_ubo,

            geometry_pass,
            gizmo_pass,
            lighting_pass,
            overlay_pass,
            gui_pass,

            object_id_reader,

            render_command_buffers,
            previous_render_fence,
            next_frame_wait_semaphore,
            swapchain_image_available_semaphore,

            renderer_state: RendererState::Initialized,
            framebuffer_index_currently_rendering: 0,
            framebuffer_index_last_rendered_to: 0,
            window_just_resized: false,
        })
    }

    pub fn update_camera(&mut self, camera: &Camera) -> anyhow::Result<()> {
        self.wait_idle_device()?;

        let dimensions = self.swapchain.properties().width_height;
        let camera_data = CameraUniformBuffer::from_camera(
            camera,
            [dimensions[0] as f32, dimensions[1] as f32],
            self.shaders_write_linear_color,
        );

        self.camera_ubo
            .write_struct(camera_data, 0)
            .context("uploading camera ubo data")?;

        Ok(())
    }

    #[inline]
    pub fn update_objects(&mut self, objects_delta: ObjectsDelta) -> anyhow::Result<()> {
        self.geometry_pass
            .update_objects(objects_delta, &self.transfer_queue, &self.render_queue)
    }

    pub fn update_gui_textures(
        &mut self,
        textures_delta: Vec<TexturesDelta>,
    ) -> anyhow::Result<()> {
        self.wait_for_previous_frame_fence()?;

        self.gui_pass
            .update_textures(textures_delta, &self.transfer_queue, &self.render_queue)?;

        Ok(())
    }

    pub fn set_scale_factor(&mut self, scale_factor: f32) {
        self.gui_pass.set_scale_factor(scale_factor);
    }

    pub fn set_gui_primitives(&mut self, gui_primitives: Vec<ClippedPrimitive>) {
        self.gui_pass.set_gui_primitives(gui_primitives);
    }

    pub fn set_window_just_resized_flag(&mut self) {
        self.window_just_resized = true;
    }

    /// Submits Vulkan commands for rendering a frame.
    pub fn render_frame(&mut self, overlay_options: RenderOptions) -> anyhow::Result<()> {
        // wait for previous frame render/resource upload to finish

        self.wait_for_previous_frame_fence()?;
        // previous frame confirmed finished rendering
        self.framebuffer_index_last_rendered_to = self.framebuffer_index_currently_rendering;

        self.gui_pass.free_previous_vertex_and_index_buffers();

        // note: I found that this check is needed on wayland because the later commands weren't returning 'out of date'...
        if self.window_just_resized {
            self.window_just_resized = false;
            self.recreate_swapchain()?;
        }

        // aquire next swapchain image

        let aquire_res = self.swapchain.aquire_next_image(
            TIMEOUT_NANOSECS,
            Some(&self.swapchain_image_available_semaphore),
            None,
        );

        if let Err(aquire_err) = aquire_res {
            if aquire_err == vk::Result::ERROR_OUT_OF_DATE_KHR {
                debug!("out of date swapchain on aquire");
                return self.recreate_swapchain();
            } else {
                return Err(aquire_err).context("calling vkAcquireNextImageKHR");
            }
        }

        let (swapchain_index, swapchain_is_suboptimal) =
            aquire_res.expect("handled err case in previous lines");
        let swapchain_index = swapchain_index as usize;
        if swapchain_is_suboptimal {
            debug!("suboptimal swapchain");
            return self.recreate_swapchain();
        }

        let framebuffer_index = self
            .current_framebuffer_index(self.framebuffer_index_last_rendered_to, swapchain_index);

        // record commands

        self.record_render_commands(framebuffer_index, overlay_options)?;

        // submit commands

        self.previous_render_fence
            .reset()
            .context("reseting previous render fence")?;

        let submit_command_buffers = [self.render_command_buffers[framebuffer_index].handle()];

        let wait_semaphores = [self.swapchain_image_available_semaphore.handle()];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

        let signal_semaphores = [self.next_frame_wait_semaphore.handle()];

        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&submit_command_buffers)
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .signal_semaphores(&signal_semaphores);

        self.render_queue
            .submit(&[submit_info], Some(&self.previous_render_fence))
            .context("submitting render commands")?;

        self.framebuffer_index_currently_rendering = swapchain_index;

        // submit present instruction

        let swapchain_present_indices = [swapchain_index as u32];
        let swapchain_handles = [self.swapchain.handle()];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .image_indices(&swapchain_present_indices)
            .swapchains(&swapchain_handles);

        let present_res = self
            .swapchain
            .queue_present(&self.render_queue, &present_info);

        if let Err(present_err) = present_res {
            if present_err == vk::Result::ERROR_OUT_OF_DATE_KHR
                || present_err == vk::Result::SUBOPTIMAL_KHR
            {
                debug!("out of date or suboptimal swapchain upon present");
                self.recreate_swapchain()?;
            } else {
                return Err(present_err).context("submitting swapchain present instruction")?;
            }
        }

        self.renderer_state = RendererState::Rendering;
        Ok(())
    }

    pub fn get_element_at_screen_coordinate(
        &mut self,
        screen_coordinate: [f32; 2],
    ) -> anyhow::Result<Option<ElementAtPoint>> {
        if let RendererState::Initialized = self.renderer_state {
            // buffer data and sync is undefined as no render commands have been submitted yet
            warn!("element at screen coordinate queried but renderer is in {:?} state. ignoring request.", self.renderer_state);
            return Ok(None);
        }

        let last_primitive_id_buffer =
            self.primitive_id_buffers[self.framebuffer_index_last_rendered_to].clone();

        let different_queue_family_indices =
            self.render_queue.family_index() != self.transfer_queue.family_index();

        if different_queue_family_indices {
            // render queue release operation
            self.object_id_reader
                .record_and_submit_pre_transfer_sync_commands(last_primitive_id_buffer.clone())?;
        }

        self.object_id_reader.record_primitive_id_copy_commands(
            screen_coordinate,
            last_primitive_id_buffer.clone(),
        )?;
        self.object_id_reader.submit_primitive_id_copy_commands()?;

        if different_queue_family_indices {
            // render queue release operation
            self.object_id_reader
                .record_and_submit_post_transfer_sync_commands(last_primitive_id_buffer)?;
        }

        let element_at_point = self.object_id_reader.read_object_id_from_buffer()?;
        Ok(Some(element_at_point))
    }

    pub fn wait_idle_device(&self) -> anyhow::Result<()> {
        self.device.wait_idle().context("calling vkDeviceWaitIdle")
    }
}

// Private functions

impl RenderManager {
    /// Recreates the swapchain, g-buffers and assiciated descriptor sets, then unsets `recreate_swapchain` trigger.
    fn recreate_swapchain(&mut self) -> anyhow::Result<()> {
        trace!("recreating swapchain...");

        // do host-device sync and reset command buffers
        self.reset_render_command_buffers()?;

        // clean up resources depending on the swapchain
        self.framebuffers.clear();
        self.swapchain_image_views.clear();

        // recreate the swapchain
        let swapchain_properties = swapchain_properties(&self.device, &self.surface, &self.window)?;
        trace!(
            "creating swapchain with dimensions: {:?}",
            swapchain_properties.width_height
        );
        self.swapchain = self
            .swapchain
            .recreate_replace(swapchain_properties)
            .context("recreating swapchain")?;

        // reinitialize related resources
        self.shaders_write_linear_color =
            shaders_should_write_linear_color(self.swapchain.properties().surface_format);
        self.swapchain_image_views = create_swapchain_image_views(&self.swapchain)?;
        let framebuffer_count = self.swapchain_image_views.len();
        trace!("swapchain image count: {}", framebuffer_count);

        let depth_buffer_format = self.depth_buffer.image().properties().format;

        self.render_pass = create_render_pass(
            self.device.clone(),
            self.swapchain.properties(),
            depth_buffer_format,
        )?;

        let framebuffer_dimensions = self.swapchain.properties().dimensions();
        self.normal_buffer =
            create_normal_buffer(self.memory_allocator.clone(), framebuffer_dimensions)?;
        self.albedo_buffer =
            create_albedo_buffer(self.memory_allocator.clone(), framebuffer_dimensions)?;
        self.primitive_id_buffers = create_primitive_id_buffers(
            framebuffer_count,
            self.memory_allocator.clone(),
            framebuffer_dimensions,
        )?;
        self.depth_buffer = create_depth_buffer(
            self.memory_allocator.clone(),
            framebuffer_dimensions,
            depth_buffer_format,
        )?;

        self.framebuffers = create_framebuffers(
            self.render_pass.clone(),
            &self.swapchain_image_views,
            self.normal_buffer.clone(),
            self.albedo_buffer.clone(),
            &self.primitive_id_buffers,
            self.depth_buffer.clone(),
        )?;

        self.lighting_pass.update_g_buffers(
            &self.normal_buffer,
            &self.albedo_buffer,
            &self.primitive_id_buffers,
        )?;

        Ok(())
    }

    fn reset_render_command_buffers(&self) -> anyhow::Result<()> {
        self.render_queue
            .wait_idle()
            .context("calling vkQueueWaitIdle for render queue")?;

        for command_buffer in &self.render_command_buffers {
            command_buffer
                .reset(vk::CommandBufferResetFlags::empty())
                .context("resetting render command buffers")?;
        }
        Ok(())
    }

    fn wait_for_previous_frame_fence(&mut self) -> anyhow::Result<()> {
        let fence_wait_res = self.previous_render_fence.wait(TIMEOUT_NANOSECS);

        if let Err(fence_wait_err) = fence_wait_res {
            if fence_wait_err == vk::Result::TIMEOUT {
                error!(
                    "previous render fence timed out! timeout set to {}ns",
                    TIMEOUT_NANOSECS
                );
                return Err(fence_wait_err)
                    .context("timeout while waiting for previous frame fence");
            } else {
                return Err(fence_wait_err).context("waiting for previous frame fence");
            }
        }
        Ok(())
    }

    fn record_render_commands(
        &mut self,
        framebuffer_index: usize,
        overlay_options: RenderOptions,
    ) -> anyhow::Result<()> {
        let viewport = self.framebuffers[framebuffer_index].whole_viewport();
        let render_area = self.framebuffers[framebuffer_index].whole_rect();
        let command_buffer = &self.render_command_buffers[framebuffer_index];

        let begin_info = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        command_buffer
            .begin(&begin_info)
            .context("beinning render command buffer recording")?;

        let render_pass_begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass.handle())
            .framebuffer(self.framebuffers[framebuffer_index].handle())
            .render_area(render_area)
            .clear_values(self.clear_values.as_slice());
        command_buffer.begin_render_pass(&render_pass_begin, vk::SubpassContents::INLINE);

        self.geometry_pass
            .record_commands(command_buffer, viewport, render_area);
        /*
        self.gizmo_pass
            .record_commands(command_buffer, viewport, render_area);
        */
        command_buffer.next_subpass(vk::SubpassContents::INLINE);

        self.lighting_pass.record_commands(
            framebuffer_index,
            command_buffer,
            viewport,
            render_area,
        );

        if overlay_options.enable_aabb_wire_display {
            self.overlay_pass.record_aabb_overlay_commands(
                command_buffer,
                self.geometry_pass.object_buffer_manager(),
                viewport,
                render_area,
            );
        }

        self.gui_pass.record_render_commands(
            command_buffer,
            self.shaders_write_linear_color,
            [viewport.width, viewport.height],
        )?;

        command_buffer.end_render_pass();

        command_buffer
            .end()
            .context("ending render command buffer recording")?;

        Ok(())
    }

    /// Determines the new framebuffer index
    #[inline]
    fn current_framebuffer_index(
        &self,
        _previous_framebuffer_index: usize,
        swapchain_index: usize,
    ) -> usize {
        swapchain_index
    }
}

impl Drop for RenderManager {
    fn drop(&mut self) {
        debug!("dropping render manager...");

        let wait_res = self.wait_idle_device();
        if let Err(e) = wait_res {
            log_anyhow_error_and_sources(&e, "renderer clean up");
        }
    }
}

// ~~ Helper structs ~~

#[derive(Debug, Clone, Copy)]
enum RendererState {
    /// Before rendering has started. Undefined rendering data.
    Initialized,
    /// Rendering commands have been submitted since initialization.
    Rendering,
}
