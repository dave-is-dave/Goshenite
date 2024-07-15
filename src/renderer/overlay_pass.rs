use super::{
    object_resource_manager::ObjectResourceManager,
    shader_interfaces::vertex_inputs::{BoundingBoxVertex, VulkanVertex},
    vulkan_init::{create_camera_descriptor_set_with_binding, render_pass_indices},
};
use crate::renderer::vulkan_init::write_camera_descriptor_set;
use anyhow::Context;
use ash::vk;
use bort_vk::{
    Buffer, ColorBlendState, CommandBuffer, DescriptorPool, DescriptorPoolProperties,
    DescriptorSet, DescriptorSetLayout, Device, DeviceOwned, DynamicState, GraphicsPipeline,
    GraphicsPipelineProperties, InputAssemblyState, PipelineAccess, PipelineLayout,
    PipelineLayoutProperties, RasterizationState, RenderPass, ShaderStage, ViewportState,
};
use std::sync::Arc;

mod descriptor {
    pub const SET_CAMERA: usize = 0;
    pub const BINDING_CAMERA: u32 = 0;
}

pub struct OverlayPass {
    desc_set_camera: DescriptorSet,
    pipeline_aabb: GraphicsPipeline,
}

impl OverlayPass {
    pub fn new(render_pass: &RenderPass, camera_buffer: &Buffer) -> anyhow::Result<Self> {
        let device = render_pass.device().clone();

        let descriptor_pool = create_descriptor_pool(device.clone())?;
        let desc_set_camera = create_descriptor_set_camera(descriptor_pool)?;
        write_camera_descriptor_set(&desc_set_camera, camera_buffer, descriptor::BINDING_CAMERA);

        let pipeline_layout_aabb =
            create_aabb_pipeline_layout(device.clone(), desc_set_camera.layout().clone())?;
        let pipeline_aabb =
            create_aabb_pipeline(device.clone(), pipeline_layout_aabb.clone(), render_pass)?;

        Ok(Self {
            desc_set_camera,
            pipeline_aabb,
        })
    }

    pub fn record_aabb_overlay_commands(
        &self,
        command_buffer: &CommandBuffer,
        object_resource_manager: &ObjectResourceManager,
        viewport: vk::Viewport,
        scissor: vk::Rect2D,
    ) {
        if object_resource_manager.object_count() == 0 {
            return;
        }

        command_buffer.bind_pipeline(&self.pipeline_aabb);
        command_buffer.set_viewport(0, &[viewport]);
        command_buffer.set_scissor(0, &[scissor]);
        command_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline_aabb.pipeline_layout().as_ref(),
            0,
            [&self.desc_set_camera],
            &[],
        );

        object_resource_manager.draw_bounding_box_commands(command_buffer);
    }
}

fn create_descriptor_pool(device: Arc<Device>) -> anyhow::Result<Arc<DescriptorPool>> {
    let descriptor_pool_properties = DescriptorPoolProperties {
        max_sets: 1,
        pool_sizes: vec![vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
        }],
        ..Default::default()
    };

    let descriptor_pool = DescriptorPool::new(device, descriptor_pool_properties)
        .context("creating overlay pass descriptor pool")?;
    Ok(Arc::new(descriptor_pool))
}

fn create_descriptor_set_camera(
    descriptor_pool: Arc<DescriptorPool>,
) -> anyhow::Result<DescriptorSet> {
    create_camera_descriptor_set_with_binding(descriptor_pool, descriptor::BINDING_CAMERA)
        .context("creating overlay pass descriptor set")
}

fn create_aabb_pipeline_layout(
    device: Arc<Device>,
    desc_set_layout_camera: Arc<DescriptorSetLayout>,
) -> anyhow::Result<Arc<PipelineLayout>> {
    let pipeline_layout_props =
        PipelineLayoutProperties::new(vec![desc_set_layout_camera], Vec::new());

    let pipeline_layout = PipelineLayout::new(device, pipeline_layout_props)
        .context("creating overlay pass aabb pipeline layout")?;

    Ok(Arc::new(pipeline_layout))
}

fn create_aabb_pipeline(
    device: Arc<Device>,
    pipeline_layout: Arc<PipelineLayout>,
    render_pass: &RenderPass,
) -> anyhow::Result<GraphicsPipeline> {
    let (vert_stage, frag_stage) = create_aabb_shader_stages(device)?;

    let dynamic_state =
        DynamicState::new_default(vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);
    let viewport_state = ViewportState::new_dynamic(1, 1);

    let color_blend_state =
        ColorBlendState::new_default(vec![ColorBlendState::blend_state_disabled()]);

    let rasterization_state = RasterizationState {
        polygon_mode: vk::PolygonMode::LINE,
        line_width: 1_f32,
        ..Default::default()
    };

    let input_assembly_state = InputAssemblyState {
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        ..Default::default()
    };

    let vertex_input_state = BoundingBoxVertex::vertex_input_state();

    let pipeline_properties = GraphicsPipelineProperties {
        color_blend_state,
        dynamic_state,
        input_assembly_state,
        rasterization_state,
        subpass_index: render_pass_indices::SUBPASS_DEFERRED as u32,
        vertex_input_state,
        viewport_state,
        ..Default::default()
    };

    let pipeline_aabb = GraphicsPipeline::new(
        pipeline_layout,
        pipeline_properties,
        &[vert_stage, frag_stage],
        render_pass,
        None,
    )
    .context("creating overlay pass aabb pipeline")?;

    Ok(pipeline_aabb)
}

#[cfg(feature = "include-spirv-bytes")]
fn create_aabb_shader_stages<'a>(
    device: Arc<Device>,
) -> anyhow::Result<(ShaderStage<'a>, ShaderStage<'a>)> {
    use super::vulkan_init::create_shader_stages_from_bytes;

    let vertex_spv_file =
        std::io::Cursor::new(&include_bytes!("../../assets/shader_binaries/outlines.vert.spv")[..]);
    let frag_spv_file =
        std::io::Cursor::new(&include_bytes!("../../assets/shader_binaries/outlines.frag.spv")[..]);

    create_shader_stages_from_bytes(device, vertex_spv_file, frag_spv_file)
        .context("creating overlay pass aabb shaders")
}

#[cfg(not(feature = "include-spirv-bytes"))]
fn create_aabb_shader_stages<'a>(
    device: Arc<Device>,
) -> anyhow::Result<(ShaderStage<'a>, ShaderStage<'a>)> {
    use crate::renderer::vulkan_init::create_shader_stages_from_path;

    const VERT_SHADER_PATH: &str = "assets/shader_binaries/outlines.vert.spv";
    const FRAG_SHADER_PATH: &str = "assets/shader_binaries/outlines.frag.spv";

    create_shader_stages_from_path(device, VERT_SHADER_PATH, FRAG_SHADER_PATH)
        .context("creating overlay pass aabb shaders")
}
