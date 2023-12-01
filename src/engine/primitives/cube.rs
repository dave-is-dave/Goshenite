use super::{
    primitive::EncodablePrimitive,
    primitive_transform::{PrimitiveTransform, DEFAULT_PRIMITIVE_TRANSFORM},
};
use crate::{
    engine::{
        aabb::Aabb,
        config_engine::{primitive_names, DEFAULT_DIMENSIONS},
    },
    renderer::shader_interfaces::primitive_op_buffer::PrimitivePropsSlice,
};
use glam::{Quat, Vec2, Vec3};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cube {
    pub transform: PrimitiveTransform,
    pub dimensions: Vec3,
}

impl Cube {
    pub const fn new(center: Vec3, rotation: Quat, dimensions: Vec3) -> Self {
        let transform = PrimitiveTransform::new(center, rotation);
        Self {
            transform,
            dimensions,
        }
    }
}

pub const DEFAULT_CUBE: Cube = Cube {
    transform: DEFAULT_PRIMITIVE_TRANSFORM,
    dimensions: DEFAULT_DIMENSIONS,
};

impl Default for Cube {
    fn default() -> Self {
        DEFAULT_CUBE
    }
}

impl EncodablePrimitive for Cube {
    fn type_name(&self) -> &'static str {
        primitive_names::CUBE
    }

    fn encoded_props(&self) -> PrimitivePropsSlice {
        let width = self.dimensions.x / 2.0;
        let depth = self.dimensions.y / 2.0;
        let height = self.dimensions.z / 2.0;
        let thickness = 0.5_f32;
        let corner_radius = Vec2::new(-1.0, 0.0);
        [
            width.to_bits(),
            depth.to_bits(),
            height.to_bits(),
            thickness.to_bits(),
            corner_radius.x.to_bits(),
            corner_radius.y.to_bits(),
        ]
    }

    fn transform(&self) -> &PrimitiveTransform {
        &self.transform
    }

    fn aabb(&self) -> Aabb {
        // todo calculate only when props/transform changed!
        //todo!("dimensions need to be adjusted for rotation!");
        Aabb::new(self.transform.center, self.dimensions + Vec3::splat(0.1))
    }
}
