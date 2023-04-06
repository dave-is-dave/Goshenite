use super::{
    primitive::{default_dimensions, Primitive, PrimitiveId},
    primitive_ref_types::primitive_names,
    primitive_transform::PrimitiveTransform,
};
use crate::{
    engine::aabb::Aabb,
    renderer::shader_interfaces::primitive_op_buffer::{
        primitive_type_codes, PrimitiveOpBufferUnit, PrimitivePropsSlice,
    },
};
use glam::{Quat, Vec3};

#[derive(Debug, Clone, PartialEq)]
pub struct Cube {
    id: PrimitiveId,
    pub transform: PrimitiveTransform,
    pub dimensions: Vec3,
}

impl Cube {
    pub const fn new_default(id: PrimitiveId) -> Self {
        Self {
            id,
            transform: PrimitiveTransform::new_default(),
            dimensions: default_dimensions(),
        }
    }

    pub const fn new(id: PrimitiveId, center: Vec3, rotation: Quat, dimensions: Vec3) -> Self {
        let transform = PrimitiveTransform { center, rotation };
        Self {
            id,
            transform,
            dimensions,
        }
    }
}

impl Primitive for Cube {
    fn id(&self) -> PrimitiveId {
        self.id
    }

    fn type_code(&self) -> PrimitiveOpBufferUnit {
        primitive_type_codes::CUBE
    }

    fn type_name(&self) -> &'static str {
        primitive_names::CUBE
    }

    fn encoded_props(&self) -> PrimitivePropsSlice {
        [
            self.dimensions.x.to_bits(),
            self.dimensions.y.to_bits(),
            self.dimensions.z.to_bits(),
            // padding
            0,
            0,
            0,
        ]
    }

    fn transform(&self) -> &PrimitiveTransform {
        &self.transform
    }

    fn aabb(&self) -> Aabb {
        // todo calculate only when props/transform changed!
        //todo!("dimensions need to ba adjusted for rotation!");
        Aabb::new(self.transform, self.dimensions)
    }
}
