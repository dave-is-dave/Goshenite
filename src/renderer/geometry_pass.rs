use super::{
    object_resource_manager::ObjectResourceManager,
    shader_interfaces::vertex_inputs::BoundingBoxVertex,
    vulkan_init::{
        create_camera_descriptor_set_with_binding, render_pass_indices, write_camera_descriptor_set,
    },
};
use crate::{
    engine::object::objects_delta::ObjectsDelta,
    renderer::shader_interfaces::vertex_inputs::VulkanVertex,
};
use anyhow::Context;
use ash::vk;
use bort_vk::{
    Buffer, ColorBlendState, CommandBuffer, DepthStencilState, DescriptorPool,
    DescriptorPoolProperties, DescriptorSet, DescriptorSetLayout, DescriptorSetLayoutBinding,
    DescriptorSetLayoutProperties, Device, DeviceOwned, DynamicState, GraphicsPipeline,
    GraphicsPipelineProperties, MemoryAllocator, PipelineAccess, PipelineLayout,
    PipelineLayoutProperties, Queue, RasterizationState, RenderPass, ShaderStage, ViewportState,
};
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use std::sync::Arc;

// descriptor set and binding indices
pub(super) mod descriptor {
    pub const SET_CAMERA: usize = 0;
    pub const BINDING_CAMERA: u32 = 0;

    pub const SET_PRIMITIVE_OPS: usize = 1;
    pub const BINDING_PRIMITIVE_OPS: u32 = 0;
}

/// Render the scene geometry and write to g-buffers
pub struct GeometryPass {
    device: Arc<Device>,

    desc_set_camera: DescriptorSet,

    pipeline: GraphicsPipeline,
    object_buffer_manager: ObjectResourceManager,
}

// Public functions
impl GeometryPass {
    pub fn new(
        device: Arc<Device>,
        memory_allocator: Arc<MemoryAllocator>,
        render_pass: &RenderPass,
        camera_buffer: &Buffer,
        transfer_queue_family_index: u32,
        render_queue_family_index: u32,
    ) -> anyhow::Result<Self> {
        let descriptor_pool = create_descriptor_pool(device.clone())?;

        let desc_set_camera = create_desc_set_camera(descriptor_pool.clone())?;
        write_camera_descriptor_set(&desc_set_camera, camera_buffer, descriptor::BINDING_CAMERA);

        let primitive_ops_desc_set_layout = create_primitive_ops_desc_set_layout(device.clone())?;

        let pipeline_layout = create_pipeline_layout(
            device.clone(),
            desc_set_camera.layout().clone(),
            primitive_ops_desc_set_layout.clone(),
        )?;
        let pipeline = create_pipeline(pipeline_layout, render_pass)?;

        let object_buffer_manager = ObjectResourceManager::new(
            memory_allocator,
            primitive_ops_desc_set_layout,
            transfer_queue_family_index,
            render_queue_family_index,
        )?;

        Ok(Self {
            device,
            desc_set_camera,
            pipeline,
            object_buffer_manager,
        })
    }

    #[inline]
    pub fn update_objects(
        &mut self,
        objects_delta: ObjectsDelta,
        transfer_queue: &Queue,
        render_queue: &Queue,
    ) -> anyhow::Result<()> {
        self.object_buffer_manager
            .update_objects(objects_delta, transfer_queue, render_queue)
    }

    pub fn record_commands(
        &self,
        command_buffer: &CommandBuffer,
        viewport: vk::Viewport,
        scissor: vk::Rect2D,
    ) {
        if self.object_buffer_manager.object_count() == 0 {
            trace!("no object buffers found. skipping geometry pass commands...");
            return;
        }

        command_buffer.bind_pipeline(&self.pipeline);
        command_buffer.set_viewport(0, &[viewport]);
        command_buffer.set_scissor(0, &[scissor]);
        command_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline.pipeline_layout().as_ref(),
            0,
            [&self.desc_set_camera],
            &[],
        );

        self.object_buffer_manager
            .draw_commands(command_buffer, &self.pipeline);
    }

    #[inline]
    pub fn object_buffer_manager(&self) -> &ObjectResourceManager {
        &self.object_buffer_manager
    }
}

impl Drop for GeometryPass {
    fn drop(&mut self) {
        trace!("dropping geometry pass...");
    }
}

fn create_descriptor_pool(device: Arc<Device>) -> anyhow::Result<Arc<DescriptorPool>> {
    let descriptor_pool_props = DescriptorPoolProperties {
        max_sets: 1,
        pool_sizes: vec![vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
        }],
        ..Default::default()
    };

    let descriptor_pool = DescriptorPool::new(device, descriptor_pool_props)
        .context("creating geometry pass descriptor pool")?;
    Ok(Arc::new(descriptor_pool))
}

fn create_desc_set_camera(descriptor_pool: Arc<DescriptorPool>) -> anyhow::Result<DescriptorSet> {
    create_camera_descriptor_set_with_binding(descriptor_pool, descriptor::BINDING_CAMERA)
        .context("creating geometry pass descriptor set")
}

fn create_primitive_ops_desc_set_layout(
    device: Arc<Device>,
) -> anyhow::Result<Arc<DescriptorSetLayout>> {
    let mut desc_set_layout_props = DescriptorSetLayoutProperties::default();
    desc_set_layout_props.bindings = vec![DescriptorSetLayoutBinding {
        binding: descriptor::BINDING_PRIMITIVE_OPS,
        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        ..Default::default()
    }];

    let desc_set_layout = DescriptorSetLayout::new(device, desc_set_layout_props)
        .context("creating geometry pass primitive-ops descriptor set layout")?;

    Ok(Arc::new(desc_set_layout))
}

fn create_pipeline_layout(
    device: Arc<Device>,
    desc_set_layout_camera: Arc<DescriptorSetLayout>,
    desc_set_layout_primitive_ops: Arc<DescriptorSetLayout>,
) -> anyhow::Result<Arc<PipelineLayout>> {
    let pipeline_layout_props = PipelineLayoutProperties::new(
        vec![desc_set_layout_camera, desc_set_layout_primitive_ops],
        Vec::new(),
    );

    let pipeline_layout = PipelineLayout::new(device, pipeline_layout_props)
        .context("creating geometry pass pipeline layout")?;

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

    let color_blend_state = ColorBlendState::new_default(vec![
        ColorBlendState::blend_state_disabled(
        );
        render_pass_indices::GBUFFER_COLOR_ATTACHMENT_COUNT
    ]);

    let depth_stencil_state = DepthStencilState {
        depth_test_enable: true,
        depth_write_enable: true,
        depth_compare_op: vk::CompareOp::GREATER_OR_EQUAL,
        depth_bounds_test_enable: false,
        stencil_test_enable: false,
        ..Default::default()
    };

    let rasterization_state = RasterizationState {
        // makes sure our fragments are always the far end of the bounding meshes,
        // which allows for a path-tracing miss condition optimization.
        cull_mode: vk::CullModeFlags::FRONT,
        ..Default::default()
    };

    let vertex_input_state = BoundingBoxVertex::vertex_input_state();

    let pipeline_properties = GraphicsPipelineProperties {
        color_blend_state,
        depth_stencil_state,
        dynamic_state,
        rasterization_state,
        subpass_index: render_pass_indices::SUBPASS_GBUFFER as u32,
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
    .context("creating geometry pass pipeline")?;

    Ok(pipeline)
}

#[cfg(feature = "include-spirv-bytes")]
fn create_shader_stages<'a>(
    device: Arc<Device>,
) -> anyhow::Result<(ShaderStage<'a>, ShaderStage<'a>)> {
    use super::vulkan_init::create_shader_stages_from_bytes;

    let vertex_spv_file = std::io::Cursor::new(
        &include_bytes!("../../assets/shader_binaries/bounding_mesh.vert.spv")[..],
    );
    let frag_spv_file = std::io::Cursor::new(
        &include_bytes!("../../assets/shader_binaries/scene_geometry.frag.spv")[..],
    );

    create_shader_stages_from_bytes(device, vertex_spv_file, frag_spv_file)
        .context("creating geoemetry pass shaders")
}

#[cfg(not(feature = "include-spirv-bytes"))]
fn create_shader_stages<'a>(
    device: Arc<Device>,
) -> anyhow::Result<(ShaderStage<'a>, ShaderStage<'a>)> {
    use crate::renderer::vulkan_init::create_shader_stages_from_path;

    const VERT_SHADER_PATH: &str = "assets/shader_binaries/bounding_mesh.vert.spv";
    const FRAG_SHADER_PATH: &str = "assets/shader_binaries/scene_geometry.frag.spv";

    create_shader_stages_from_path(device, VERT_SHADER_PATH, FRAG_SHADER_PATH)
        .context("creating geometry pass shaders")
}
