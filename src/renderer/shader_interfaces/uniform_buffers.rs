use crate::user_interface::camera::Camera;
use glam::{Mat4, Vec3};

/// Camera data read by GPU shaders
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct CameraUniformBuffer {
    pub view_inverse: [f32; 16],
    pub proj_inverse: [f32; 16],
    /// Camera position in world space (w component unused)
    pub position: [f32; 4],
    pub framebuffer_dims: [f32; 2],
    pub near: f32,
    pub far: f32,
    /// 0 if false, 1 if true
    pub write_linear_color: u32,
}

impl CameraUniformBuffer {
    #[inline]
    pub fn new(
        view_inverse: Mat4,
        proj_inverse: Mat4,
        position: Vec3,
        framebuffer_dimensions: [f32; 2],
        near: f32,
        far: f32,
        write_linear_color: bool,
    ) -> Self {
        Self {
            view_inverse: view_inverse.to_cols_array(),
            proj_inverse: proj_inverse.to_cols_array(),
            position: [position.x, position.y, position.z, 0.0],
            framebuffer_dims: framebuffer_dimensions,
            near,
            far,
            write_linear_color: write_linear_color as u32,
        }
    }

    pub fn from_camera(
        camera: &Camera,
        framebuffer_dimensions: [f32; 2],
        write_linear_color: bool,
    ) -> Self {
        let proj_inverse = camera.projection_matrix_inverse();
        let view_inverse = camera.view_matrix().inverse();
        Self::new(
            view_inverse,
            proj_inverse,
            camera.position().as_vec3(),
            framebuffer_dimensions,
            camera.near_plane() as f32,
            camera.far_plane() as f32,
            write_linear_color,
        )
    }
}
