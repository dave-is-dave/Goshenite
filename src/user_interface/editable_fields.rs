use super::{gui::EditState, gui_state::DRAG_INC};
use crate::{
    config,
    engine::{
        object::{object::ObjectId, operation::Operation, primitive_op::PrimitiveOp},
        primitives::{
            cube::Cube, primitive_transform::PrimitiveTransform, sphere::Sphere,
            uber_primitive::UberPrimitive,
        },
    },
    helper::{
        angle::Angle,
        axis::{Axis, CartesianAxis},
    },
};
use egui::{
    color_picker::{color_edit_button_hsva, Alpha},
    epaint::Hsva,
    ComboBox, DragValue,
};
use egui_dnd::DragableItem;
use glam::{Vec2, Vec3, Vec4};

/// Returns a new operation if a different one is selected
pub fn op_drop_down(
    ui: &mut egui::Ui,
    original_op: Operation,
    object_id: ObjectId,
) -> Option<Operation> {
    let mut new_op = original_op.clone();

    ComboBox::from_id_salt(format!("op drop down {:?}", object_id))
        .width(0_f32)
        .selected_text(original_op.name())
        .show_ui(ui, |ui_op| {
            for (op, op_name) in Operation::variants_with_names() {
                ui_op.selectable_value(&mut new_op, op, op_name);
            }
        });

    if original_op != new_op {
        return Some(new_op);
    } else {
        None
    }
}

pub fn sphere_editor_ui(ui: &mut egui::Ui, sphere: &mut Sphere) -> EditState {
    let new_radius = editable_radius_ui(ui, sphere.radius);

    if let Some(some_new_radius) = new_radius {
        sphere.radius = some_new_radius;
        EditState::Modified
    } else {
        EditState::NoChange
    }
}

pub fn cube_editor_ui(ui: &mut egui::Ui, cube: &mut Cube) -> EditState {
    editable_dimensions_ui(ui, &mut cube.dimensions)
}

pub fn uber_primitive_editor_ui(
    ui: &mut egui::Ui,
    uber_primitive: &mut UberPrimitive,
) -> EditState {
    editable_uber_parameters_ui(
        ui,
        &mut uber_primitive.dimensions,
        &mut uber_primitive.corner_radius,
    )
}

pub fn blend_editor_ui(ui: &mut egui::Ui, blend: &mut f32) -> EditState {
    let original_blend = *blend;
    ui.horizontal(|ui_h| {
        ui_h.label("Blend:");
        ui_h.add(DragValue::new(blend).speed(DRAG_INC));
    });
    if original_blend != *blend {
        EditState::Modified
    } else {
        EditState::NoChange
    }
}

pub fn color_specular_editor_ui(
    ui: &mut egui::Ui,
    color: &mut Vec3,
    specular: &mut f32,
) -> EditState {
    let original_color = *color;
    let original_specular = *specular;

    let mut hsva_color = Hsva::from_rgb(color.to_array());
    ui.label("Color:");
    color_edit_button_hsva(ui, &mut hsva_color, Alpha::Opaque);
    *color = Vec3::from_array(hsva_color.to_rgb());

    ui.horizontal(|ui_h| {
        ui_h.label("Specular:");
        ui_h.add(DragValue::new(specular).speed(DRAG_INC).range(0..=1));
    });

    if original_color != *color || original_specular != *specular {
        EditState::Modified
    } else {
        EditState::NoChange
    }
}

pub fn primitive_transform_editor_ui(
    ui: &mut egui::Ui,
    primitive_transform: &mut PrimitiveTransform,
) -> EditState {
    let mut edit_state = EditState::NoChange;

    let edited_center = editable_center_ui(ui, primitive_transform.center);
    if let Some(some_new_center) = edited_center {
        primitive_transform.center = some_new_center;
        edit_state = EditState::Modified;
    }

    let edited_axis = editable_axis_ui(ui, primitive_transform.rotation_tentative_append().axis);
    if let Some(changed_axis) = edited_axis {
        // axis changed -> old tentative rotation invalid -> need to commit it before changing it
        primitive_transform.commit_tentative_rotation();
        primitive_transform.set_tentative_rotation_axis(changed_axis);
        edit_state = EditState::Modified;
    }

    let edited_angle = editable_angle_ui(ui, primitive_transform.rotation_tentative_append().angle);
    if let Some(change_angle) = edited_angle {
        primitive_transform.set_tentative_rotation_angle(change_angle);
        edit_state = EditState::Modified;
    }

    edit_state
}

/// Returns `Some` new center if the value was modified by the gui
pub fn editable_center_ui(ui: &mut egui::Ui, original_center: Vec3) -> Option<Vec3> {
    let mut new_center = original_center.clone();

    ui.horizontal(|ui_h| {
        ui_h.label("Center:");
        ui_h.add(DragValue::new(&mut new_center.x).speed(DRAG_INC));
        ui_h.add(DragValue::new(&mut new_center.y).speed(DRAG_INC));
        ui_h.add(DragValue::new(&mut new_center.z).speed(DRAG_INC));
    });

    if new_center != original_center {
        Some(new_center)
    } else {
        None
    }
}

/// Returns `Some` new axis if the value was modified by the gui
pub fn editable_axis_ui(ui: &mut egui::Ui, original_axis: Axis) -> Option<Axis> {
    let mut new_axis = original_axis.clone();

    ui.horizontal(|ui_h| {
        ui_h.label("Rotation axis:");

        ComboBox::new("Axis type", "")
            .width(0_f32)
            .selected_text(new_axis.type_name())
            .show_ui(ui_h, |ui_op| {
                ui_op.selectable_value(
                    &mut new_axis,
                    Axis::DEFAULT_CARTESIAN,
                    Axis::CARTESIAN_VARIANT_NAME,
                );
                ui_op.selectable_value(
                    &mut new_axis,
                    Axis::DEFAULT_DIRECION,
                    Axis::DIRECTION_VARIANT_NAME,
                );
            });

        match &mut new_axis {
            Axis::Cartesian(c_axis_mut) => {
                ComboBox::new("Cartesian axis", "")
                    .width(0_f32)
                    .selected_text(c_axis_mut.as_str())
                    .show_ui(ui_h, |ui_op| {
                        for (c_axis_variant, c_axis_name) in CartesianAxis::variants_with_names() {
                            ui_op.selectable_value(c_axis_mut, c_axis_variant, c_axis_name);
                        }
                    });
            }
            Axis::Direction(direction) => {
                ui_h.label("Direction:");
                ui_h.label(format!("X = {}", direction.x));
                ui_h.label(format!("Y = {}", direction.y));
                ui_h.label(format!("Z = {}", direction.z));
            }
        }
    });

    if new_axis != original_axis {
        Some(new_axis)
    } else {
        None
    }
}

/// Returns `Some` new angle if the value was modified by the gui
pub fn editable_angle_ui(ui: &mut egui::Ui, original_angle: Angle) -> Option<Angle> {
    let mut new_angle = original_angle;

    ui.horizontal(|ui_h| match &mut new_angle {
        Angle::Degrees(degrees) => {
            ui_h.label("Angle (degrees):");
            ui_h.add(DragValue::new(degrees).speed(DRAG_INC));
        }
        Angle::Radians(radians) => {
            ui_h.label("Angle (radians):");
            ui_h.add(DragValue::new(radians).speed(DRAG_INC));
        }
    });

    if new_angle != original_angle {
        Some(new_angle)
    } else {
        None
    }
}

/// Returns `Some` new radius if the value was modified by the gui
pub fn editable_radius_ui(ui: &mut egui::Ui, original_radius: f32) -> Option<f32> {
    let mut new_radius = original_radius;

    ui.horizontal(|ui| {
        ui.label("Radius:");
        ui.add(
            DragValue::new(&mut new_radius)
                .speed(DRAG_INC)
                .range(0..=config::MAX_SPHERE_RADIUS),
        );
    });

    if new_radius != original_radius {
        Some(new_radius)
    } else {
        None
    }
}

pub fn editable_dimensions_ui(ui: &mut egui::Ui, dimensions: &mut Vec3) -> EditState {
    let mut something_changed: bool = false;

    ui.horizontal(|ui| {
        ui.label("Dimensions:");
        something_changed |= ui
            .add(DragValue::new(&mut dimensions.x).speed(DRAG_INC))
            .changed();
        something_changed |= ui
            .add(DragValue::new(&mut dimensions.y).speed(DRAG_INC))
            .changed();
        something_changed |= ui
            .add(DragValue::new(&mut dimensions.z).speed(DRAG_INC))
            .changed();
    });

    if something_changed {
        EditState::Modified
    } else {
        EditState::NoChange
    }
}

pub fn editable_uber_parameters_ui(
    ui: &mut egui::Ui,
    dimensions: &mut Vec4,
    corner_radius: &mut Vec2,
) -> EditState {
    let mut something_changed: bool = false;

    ui.horizontal(|ui| {
        ui.label("Dimensions:");
        something_changed |= ui
            .add(DragValue::new(&mut dimensions.x).speed(DRAG_INC))
            .changed();
        something_changed |= ui
            .add(DragValue::new(&mut dimensions.y).speed(DRAG_INC))
            .changed();
        something_changed |= ui
            .add(DragValue::new(&mut dimensions.z).speed(DRAG_INC))
            .changed();
        something_changed |= ui
            .add(DragValue::new(&mut dimensions.w).speed(DRAG_INC))
            .changed();
    });

    ui.horizontal(|ui| {
        ui.label("Corner radius:");
        something_changed |= ui
            .add(DragValue::new(&mut corner_radius.x).speed(DRAG_INC))
            .changed();
        something_changed |= ui
            .add(DragValue::new(&mut corner_radius.y).speed(DRAG_INC))
            .changed();
    });

    if something_changed {
        EditState::Modified
    } else {
        EditState::NoChange
    }
}

impl DragableItem for PrimitiveOp {
    fn drag_id(&self) -> egui::Id {
        egui::Id::new(format!("p-op-drag{}", self.id()))
    }
}
