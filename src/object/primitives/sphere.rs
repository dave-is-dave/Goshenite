use super::primitive::Primitive;
use crate::shaders::object_buffer::{primitive_codes, PrimitiveDataSlice};
use glam::Vec3;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}
impl Sphere {
    pub const fn new(center: Vec3, radius: f32) -> Self {
        Self { center, radius }
    }
}
impl Primitive for Sphere {
    fn encode(&self) -> PrimitiveDataSlice {
        [
            primitive_codes::SPHERE,
            self.center.x.to_bits(),
            self.center.y.to_bits(),
            self.center.z.to_bits(),
            self.radius.to_bits(),
            // padding
            primitive_codes::NULL,
            primitive_codes::NULL,
        ]
    }

    fn center(&self) -> Vec3 {
        self.center
    }

    fn type_name(&self) -> &'static str {
        "Sphere"
    }
}
impl Default for Sphere {
    fn default() -> Self {
        Self {
            center: Vec3::ZERO,
            radius: 1.0,
        }
    }
}
