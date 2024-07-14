use super::vulkan_init::{
    create_camera_descriptor_set_with_binding, render_pass_indices, write_camera_descriptor_set,
};
use anyhow::Context;
use ash::vk;
use bort_vk::{
    Buffer, ColorBlendState, CommandBuffer, DescriptorPool, DescriptorPoolProperties,
    DescriptorSet, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutProperties,
    Device, DeviceOwned, DynamicState, GraphicsPipeline, GraphicsPipelineProperties, Image,
    ImageView, ImageViewAccess, PipelineAccess, PipelineLayout, PipelineLayoutProperties,
    RenderPass, ShaderStage, ViewportState,
};
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use std::sync::Arc;

/// Describes descriptor set indices
mod descriptor {
    pub const SET_G_BUFFERS: usize = 0;
    pub const BINDING_NORMAL: u32 = 0;
    pub const BINDING_ALBEDO: u32 = 1;
    pub const BINDING_PRIMITIVE_ID: u32 = 2;

    pub const SET_CAMERA: usize = 1;
    pub const BINDING_CAMERA: u32 = 0;
}

/// Defines functionality for reading the g-buffers and calculating the scene color values
pub struct LightingPass {
    device: Arc<Device>,

    desc_set_camera: DescriptorSet,
    /// One per framebuffer
    desc_sets_g_buffer: Vec<DescriptorSet>,

    pipeline: GraphicsPipeline,
}

impl LightingPass {
    pub fn new(
        device: Arc<Device>,
        render_pass: &RenderPass,
        camera_buffer: &Buffer,
        normal_buffer: &ImageView<Image>,
        albedo_buffer: &ImageView<Image>,
        primitive_id_buffers: &Vec<Arc<ImageView<Image>>>,
    ) -> anyhow::Result<Self> {
        let framebuffer_count = primitive_id_buffers.len();
        let descriptor_pool = create_descriptor_pool(device.clone(), framebuffer_count)?;

        let desc_set_camera = create_desc_set_camera(descriptor_pool.clone())?;
        write_camera_descriptor_set(&desc_set_camera, camera_buffer, descriptor::BINDING_CAMERA);

        let desc_sets_g_buffer =
            create_desc_sets_gbuffer(descriptor_pool.clone(), framebuffer_count)?;
        write_desc_sets_gbuffer(
            &desc_sets_g_buffer,
            normal_buffer,
            albedo_buffer,
            primitive_id_buffers,
        )?;

        let pipeline_layout = create_pipeline_layout(
            device.clone(),
            desc_set_camera.layout().clone(),
            desc_sets_g_buffer[0].layout().clone(),
        )?;
        let pipeline = create_pipeline(device.clone(), pipeline_layout.clone(), render_pass)?;

        Ok(Self {
            device,
            desc_sets_g_buffer,
            desc_set_camera,
            pipeline,
        })
    }

    /// Call whenever the g-buffers change
    pub fn update_g_buffers(
        &mut self,
        normal_buffer: &ImageView<Image>,
        albedo_buffer: &ImageView<Image>,
        primitive_id_buffers: &Vec<Arc<ImageView<Image>>>,
    ) -> anyhow::Result<()> {
        write_desc_sets_gbuffer(
            &self.desc_sets_g_buffer,
            normal_buffer,
            albedo_buffer,
            primitive_id_buffers,
        )
    }

    /// Records draw commands to a command buffer.
    ///
    /// **Assumes that the command buffer is already in a render pass state.**
    pub fn record_commands(
        &self,
        frame_index: usize,
        command_buffer: &CommandBuffer,
        viewport: vk::Viewport,
        scissor: vk::Rect2D,
    ) {
        command_buffer.bind_pipeline(&self.pipeline);
        command_buffer.set_viewport(0, &[viewport]);
        command_buffer.set_scissor(0, &[scissor]);
        command_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline.pipeline_layout().as_ref(),
            0,
            [&self.desc_sets_g_buffer[frame_index], &self.desc_set_camera],
            &[],
        );
        command_buffer.draw(3, 1, 0, 0);
    }
}

fn create_descriptor_pool(
    device: Arc<Device>,
    framebuffer_count: usize,
) -> anyhow::Result<Arc<DescriptorPool>> {
    let descriptor_pool_props = DescriptorPoolProperties {
        max_sets: 8,
        pool_sizes: vec![
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::INPUT_ATTACHMENT,
                descriptor_count: 3 * framebuffer_count as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
            },
        ],
        ..Default::default()
    };

    let descriptor_pool = DescriptorPool::new(device, descriptor_pool_props)
        .context("creating lighting pass descriptor pool")?;
    Ok(Arc::new(descriptor_pool))
}

fn create_desc_set_camera(descriptor_pool: Arc<DescriptorPool>) -> anyhow::Result<DescriptorSet> {
    create_camera_descriptor_set_with_binding(descriptor_pool, descriptor::BINDING_CAMERA)
        .context("creating geometry pass descriptor set")
}

fn create_desc_sets_gbuffer(
    descriptor_pool: Arc<DescriptorPool>,
    framebuffer_count: usize,
) -> anyhow::Result<Vec<DescriptorSet>> {
    (0..framebuffer_count)
        .into_iter()
        .map(|_| create_desc_set_gbuffer(descriptor_pool.clone()))
        .collect::<anyhow::Result<Vec<_>>>()
}

fn create_desc_set_gbuffer(descriptor_pool: Arc<DescriptorPool>) -> anyhow::Result<DescriptorSet> {
    let mut desc_set_layout_props = DescriptorSetLayoutProperties::default();
    desc_set_layout_props.bindings = vec![
        DescriptorSetLayoutBinding {
            binding: descriptor::BINDING_NORMAL,
            descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        },
        DescriptorSetLayoutBinding {
            binding: descriptor::BINDING_ALBEDO,
            descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        },
        DescriptorSetLayoutBinding {
            binding: descriptor::BINDING_PRIMITIVE_ID,
            descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        },
    ];

    let desc_set_layout = Arc::new(
        DescriptorSetLayout::new(descriptor_pool.device().clone(), desc_set_layout_props)
            .context("creating lighting pass g-buffer descriptor set layout")?,
    );

    let desc_set = descriptor_pool
        .allocate_descriptor_set(desc_set_layout)
        .context("allocating lighting pass g-buffer descriptor set")?;

    Ok(desc_set)
}

fn write_desc_sets_gbuffer(
    desc_sets_gbuffer: &Vec<DescriptorSet>,
    normal_buffer: &ImageView<Image>,
    albedo_buffer: &ImageView<Image>,
    primitive_id_buffers: &Vec<Arc<ImageView<Image>>>,
) -> anyhow::Result<()> {
    for i in 0..desc_sets_gbuffer.len() {
        write_desc_set_gbuffer(
            &desc_sets_gbuffer[i],
            normal_buffer,
            albedo_buffer,
            primitive_id_buffers[i].as_ref(),
        )?;
    }
    Ok(())
}

fn write_desc_set_gbuffer(
    desc_set_gbuffer: &DescriptorSet,
    normal_buffer: &impl ImageViewAccess,
    albedo_buffer: &impl ImageViewAccess,
    primitive_id_buffer: &impl ImageViewAccess,
) -> anyhow::Result<()> {
    let normal_buffer_info = vk::DescriptorImageInfo {
        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        image_view: normal_buffer.handle(),
        ..Default::default()
    };
    let normal_buffer_infos = [normal_buffer_info];

    let albedo_buffer_info = vk::DescriptorImageInfo {
        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        image_view: albedo_buffer.handle(),
        ..Default::default()
    };
    let albedo_buffer_infos = [albedo_buffer_info];

    let primitive_id_buffer_info = vk::DescriptorImageInfo {
        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        image_view: primitive_id_buffer.handle(),
        ..Default::default()
    };
    let primitive_id_buffer_infos = [primitive_id_buffer_info];

    let descriptor_write_normal_buffer = vk::WriteDescriptorSet::default()
        .dst_set(desc_set_gbuffer.handle())
        .dst_binding(descriptor::BINDING_NORMAL)
        .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
        .image_info(&normal_buffer_infos);
    let descriptor_write_albedo_buffer = vk::WriteDescriptorSet::default()
        .dst_set(desc_set_gbuffer.handle())
        .dst_binding(descriptor::BINDING_ALBEDO)
        .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
        .image_info(&albedo_buffer_infos);
    let descriptor_write_primitive_id_buffer = vk::WriteDescriptorSet::default()
        .dst_set(desc_set_gbuffer.handle())
        .dst_binding(descriptor::BINDING_PRIMITIVE_ID)
        .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
        .image_info(&primitive_id_buffer_infos);

    desc_set_gbuffer.device().update_descriptor_sets(
        [
            descriptor_write_normal_buffer,
            descriptor_write_albedo_buffer,
            descriptor_write_primitive_id_buffer,
        ],
        [],
    );

    Ok(())
}

fn create_pipeline_layout(
    device: Arc<Device>,
    desc_set_layout_camera: Arc<DescriptorSetLayout>,
    desc_set_layout_g_buffers: Arc<DescriptorSetLayout>,
) -> anyhow::Result<Arc<PipelineLayout>> {
    let pipeline_layout_props = PipelineLayoutProperties::new(
        vec![desc_set_layout_g_buffers, desc_set_layout_camera],
        Vec::new(),
    );

    let pipeline_layout = PipelineLayout::new(device, pipeline_layout_props)
        .context("creating lighting pass pipeline layout")?;

    Ok(Arc::new(pipeline_layout))
}

fn create_pipeline(
    device: Arc<Device>,
    pipeline_layout: Arc<PipelineLayout>,
    render_pass: &RenderPass,
) -> anyhow::Result<GraphicsPipeline> {
    let (vert_stage, frag_stage) = create_shader_stages(&device)?;

    let dynamic_state =
        DynamicState::new_default(vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

    let viewport_state = ViewportState::new_dynamic(1, 1);

    let color_blend_state =
        ColorBlendState::new_default(vec![ColorBlendState::blend_state_disabled()]);

    let pipeline_properties = GraphicsPipelineProperties {
        color_blend_state,
        dynamic_state,
        subpass_index: render_pass_indices::SUBPASS_DEFERRED as u32,
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
    .context("creating lighting pass pipeline")?;

    Ok(pipeline)
}

#[cfg(feature = "include-spirv-bytes")]
fn create_shader_stages(device: &Arc<Device>) -> anyhow::Result<(ShaderStage, ShaderStage)> {
    use super::vulkan_init::create_shader_stages_from_bytes;

    let vertex_spv_file = std::io::Cursor::new(
        &include_bytes!("../../assets/shader_binaries/full_screen.vert.spv")[..],
    );
    let frag_spv_file = std::io::Cursor::new(
        &include_bytes!("../../assets/shader_binaries/scene_lighting.frag.spv")[..],
    );

    create_shader_stages_from_bytes(device, vertex_spv_file, frag_spv_file)
        .context("creating lighting pass shaders")
}

#[cfg(not(feature = "include-spirv-bytes"))]
fn create_shader_stages(device: &Arc<Device>) -> anyhow::Result<(ShaderStage, ShaderStage)> {
    use crate::renderer::vulkan_init::create_shader_stages_from_path;

    const VERT_SHADER_PATH: &str = "assets/shader_binaries/full_screen.vert.spv";
    const FRAG_SHADER_PATH: &str = "assets/shader_binaries/scene_lighting.frag.spv";

    create_shader_stages_from_path(device, VERT_SHADER_PATH, FRAG_SHADER_PATH)
        .context("creating lighting pass shaders")
}

impl Drop for LightingPass {
    fn drop(&mut self) {
        trace!("dropping lighting pass...");
    }
}
