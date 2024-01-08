use glam::Vec3;

pub const RENDER_THREAD_WAIT_TIMEOUT_SECONDS: f64 = 2.;

pub const DEFAULT_RADIUS: f32 = 0.5;
pub const DEFAULT_DIMENSIONS: Vec3 = Vec3::ONE;

pub mod primitive_names {
    pub const SPHERE: &'static str = "Sphere";
    pub const CUBE: &'static str = "Cube";
    pub const UBER_PRIMITIVE: &'static str = "Uber Primitive";
}

pub const AABB_EDGE: Vec3 = Vec3::splat(0.1);

pub const LOCAL_STORAGE_DIR: &'static str = ".goshenite";
pub const SAVE_STATE_FILENAME_CAMERA: &'static str = "camera.gsave";
pub const SAVE_STATE_FILENAME_OBJECTS: &'static str = "objects.gsave";
