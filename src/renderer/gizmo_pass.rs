use super::{
    config_renderer::GIZMO_ARROW_STL_PATH,
    shader_interfaces::{
        camera_uniform_buffer::CameraUniformBuffer,
        primitive_op_buffer::PRIMITIVE_ID_GIZMO,
        vertex_inputs::{GizmoVertex, VulkanVertex},
    },
    vulkan_init::{create_camera_descriptor_set_with_binding, render_pass_indices},
};
use crate::{
    helper::more_errors::IoError, renderer::shader_interfaces::push_constants::GizmosPushConstant,
};
use anyhow::Context;
use ash::vk::{self, BufferUsageFlags};
use bort_vk::{
    AllocationAccess, Buffer, BufferProperties, ColorBlendState, CommandBuffer, DepthStencilState,
    DescriptorPool, DescriptorPoolProperties, DescriptorSet, DescriptorSetLayout, Device,
    DeviceOwned, DynamicState, GraphicsPipeline, GraphicsPipelineProperties, MemoryAllocator,
    PipelineAccess, PipelineLayout, PipelineLayoutProperties, RasterizationState, RenderPass,
    ShaderStage, ViewportState,
};
use bort_vma::AllocationCreateInfo;
use std::{fs::OpenOptions, mem::size_of, path::Path};
use std::{mem, sync::Arc};

mod descriptor {
    pub const SET_CAMERA: usize = 0;
    pub const BINDING_CAMERA: u32 = 0;
}

pub struct GizmoPass {
    desc_set_camera: DescriptorSet,
    pipeline: GraphicsPipeline,

    arrow_gizmo_vertex_buffer: Buffer,
    arrow_gizmo_index_buffer: Buffer,
    arrow_gizmo_index_count: u32,
}

impl GizmoPass {
    pub fn new(
        memory_allocator: Arc<MemoryAllocator>,
        render_pass: &RenderPass,
        camera_buffer: &Buffer,
    ) -> anyhow::Result<Self> {
        let device = render_pass.device().clone();

        let descriptor_pool = create_descriptor_pool(device.clone())?;
        let desc_set_camera = create_descriptor_set_camera(descriptor_pool)?;
        write_camera_descriptor_set(&desc_set_camera, camera_buffer, descriptor::BINDING_CAMERA);

        let pipeline_layout =
            create_pipeline_layout(device.clone(), desc_set_camera.layout().clone())?;
        let pipeline = create_pipeline(device.clone(), pipeline_layout, render_pass)?;

        let (arrow_gizmo_vertex_buffer, arrow_gizmo_index_buffer, arrow_gizmo_index_count) =
            create_and_upload_gizmo_buffers(memory_allocator)?;

        Ok(Self {
            desc_set_camera,
            pipeline,
            arrow_gizmo_vertex_buffer,
            arrow_gizmo_index_buffer,
            arrow_gizmo_index_count,
        })
    }

    pub fn record_commands(
        &self,
        command_buffer: &CommandBuffer,
        viewport: vk::Viewport,
        scissor: vk::Rect2D,
    ) {
        let color: [f32; 3] = [0.8, 0.1, 0.1];
        let object_id = PRIMITIVE_ID_GIZMO;
        let gizmo_push_constant = GizmosPushConstant { color, object_id };
        let gizmo_push_constant_bytes = bytemuck::bytes_of(&gizmo_push_constant);

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
        command_buffer.push_constants(
            self.pipeline.pipeline_layout().as_ref(),
            vk::ShaderStageFlags::FRAGMENT,
            0,
            gizmo_push_constant_bytes,
        );
        command_buffer.bind_vertex_buffers(0, [&self.arrow_gizmo_vertex_buffer], &[0]);
        command_buffer.bind_index_buffer(&self.arrow_gizmo_index_buffer, 0, vk::IndexType::UINT32);
        command_buffer.draw_indexed(self.arrow_gizmo_index_count, 1, 0, 0, 0);
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
        .context("creating gizmo pass descriptor set")
}

pub fn write_camera_descriptor_set(
    desc_set_camera: &DescriptorSet,
    camera_buffer: &Buffer,
    binding: u32,
) {
    let camera_buffer_info = vk::DescriptorBufferInfo {
        buffer: camera_buffer.handle(),
        offset: 0,
        range: mem::size_of::<CameraUniformBuffer>() as vk::DeviceSize,
    };
    let camera_buffer_infos = [camera_buffer_info];

    let descriptor_write = vk::WriteDescriptorSet::default()
        .dst_set(desc_set_camera.handle())
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .buffer_info(&camera_buffer_infos);

    desc_set_camera
        .device()
        .update_descriptor_sets([descriptor_write], []);
}

/// Returns `(vertex_buffer, index_buffer, index_count)`
fn create_and_upload_gizmo_buffers(
    memory_allocator: Arc<MemoryAllocator>,
) -> anyhow::Result<(Buffer, Buffer, u32)> {
    let arrow_stl = load_arrow_stl().context("loading gizmo arrow stl")?;

    let vertices: Vec<GizmoVertex> = arrow_stl
        .vertices
        .iter()
        .map(|vertex| GizmoVertex {
            in_position: [vertex[0], vertex[1], vertex[2], 1.0],
        })
        .collect();

    let indices: Vec<u32> = arrow_stl
        .faces
        .iter()
        .flat_map(|indexed_triangle| {
            [
                // these are called vertices but are actually indices
                indexed_triangle.vertices[0] as u32,
                indexed_triangle.vertices[1] as u32,
                indexed_triangle.vertices[2] as u32,
            ]
        })
        .collect();

    let buffer_allocation_info = AllocationCreateInfo {
        required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE, // todo staging buffer
        preferred_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ..AllocationCreateInfo::default()
    };

    let vertex_buffer_properties = BufferProperties::new_default(
        vertices.len() as u64 * size_of::<GizmoVertex>() as u64, // 4 f32 vertices
        BufferUsageFlags::VERTEX_BUFFER | BufferUsageFlags::TRANSFER_DST,
    );
    let mut vertex_buffer = Buffer::new(
        memory_allocator.clone(),
        vertex_buffer_properties,
        buffer_allocation_info.clone(),
    )
    .context("creating gizmo vertex buffer")?;

    vertex_buffer
        .write_iter(vertices, 0)
        .context("uploading gizmo vertices")?;

    let index_count = indices.len();
    let index_buffer_properties = BufferProperties::new_default(
        index_count as u64 * 4, // u32 indices
        BufferUsageFlags::INDEX_BUFFER | BufferUsageFlags::TRANSFER_DST,
    );
    let mut index_buffer = Buffer::new(
        memory_allocator,
        index_buffer_properties,
        buffer_allocation_info,
    )
    .context("creating gizmo vertex buffer")?;

    index_buffer
        .write_iter(indices, 0)
        .context("uploading gizmo indices")?;

    Ok((vertex_buffer, index_buffer, index_count as u32))
}

fn load_arrow_stl() -> Result<stl_io::IndexedMesh, IoError> {
    let gizmo_arrow_stl_path = Path::new(GIZMO_ARROW_STL_PATH);
    let mut arrow_stl_file = OpenOptions::new()
        .read(true)
        .open(gizmo_arrow_stl_path)
        .map_err(|e| IoError::read_file_error(e, GIZMO_ARROW_STL_PATH.to_string()))?;
    stl_io::read_stl(&mut arrow_stl_file).map_err(IoError::ReadBufferFailed)
}

fn create_pipeline_layout(
    device: Arc<Device>,
    desc_set_layout_camera: Arc<DescriptorSetLayout>,
) -> anyhow::Result<Arc<PipelineLayout>> {
    let push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(std::mem::size_of::<GizmosPushConstant>() as u32);

    let pipeline_layout_props =
        PipelineLayoutProperties::new(vec![desc_set_layout_camera], vec![push_constant_range]);

    let pipeline_layout = PipelineLayout::new(device, pipeline_layout_props)
        .context("creating overlay pass gizmos pipeline layout")?;

    Ok(Arc::new(pipeline_layout))
}

fn create_pipeline(
    device: Arc<Device>,
    pipeline_layout: Arc<PipelineLayout>,
    render_pass: &RenderPass,
) -> anyhow::Result<GraphicsPipeline> {
    let (vert_stage, frag_stage) = create_shader_stages(&device)?;

    let color_blend_state = ColorBlendState::new_default(vec![
        ColorBlendState::blend_state_disabled(
        );
        render_pass_indices::GBUFFER_COLOR_ATTACHMENT_COUNT
    ]);

    let depth_stencil_state = DepthStencilState {
        depth_test_enable: false,
        depth_write_enable: true,
        ..Default::default()
    };

    let dynamic_state =
        DynamicState::new_default(vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

    let rasterization_state = RasterizationState {
        cull_mode: vk::CullModeFlags::BACK,
        ..Default::default()
    };

    let vertex_input_state = GizmoVertex::vertex_input_state();

    let viewport_state = ViewportState::new_dynamic(1, 1);

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
    .context("creating overlay pass gizmos pipeline")?;

    Ok(pipeline)
}

#[cfg(feature = "include-spirv-bytes")]
fn create_shader_stages(device: &Arc<Device>) -> anyhow::Result<(ShaderStage, ShaderStage)> {
    use super::vulkan_init::create_shader_stages_from_bytes;

    let vertex_spv_file =
        std::io::Cursor::new(&include_bytes!("../../assets/shader_binaries/gizmos.vert.spv")[..]);
    let frag_spv_file =
        std::io::Cursor::new(&include_bytes!("../../assets/shader_binaries/gizmos.frag.spv")[..]);

    create_shader_stages_from_bytes(device, vertex_spv_file, frag_spv_file)
        .context("creating overlay pass gizmos shaders")
}

#[cfg(not(feature = "include-spirv-bytes"))]
fn create_shader_stages(device: &Arc<Device>) -> anyhow::Result<(ShaderStage, ShaderStage)> {
    use crate::renderer::vulkan_init::create_shader_stages_from_path;

    const VERT_SHADER_PATH: &str = "assets/shader_binaries/gizmos.vert.spv";
    const FRAG_SHADER_PATH: &str = "assets/shader_binaries/gizmos.frag.spv";

    create_shader_stages_from_path(device, VERT_SHADER_PATH, FRAG_SHADER_PATH)
        .context("creating overlay pass gizmos shaders")
}
