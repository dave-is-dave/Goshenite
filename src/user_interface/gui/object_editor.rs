use super::Gui;
use crate::{
    engine::{
        commands::{Command, TargetPrimitiveOp, ValidationCommand},
        object::{
            object::{Object, ObjectId},
            object_collection::ObjectCollection,
            primitive_op::{PrimitiveOp, PrimitiveOpId},
        },
        primitives::primitive::{EncodablePrimitive, Primitive},
    },
    user_interface::{
        config_ui,
        editable_fields::{
            blend_editor_ui, color_specular_editor_ui, cube_editor_ui, op_drop_down,
            primitive_transform_editor_ui, sphere_editor_ui, uber_primitive_editor_ui,
        },
        gui::EditState,
        gui_state::{GuiState, DRAG_INC},
    },
};
use egui::{ComboBox, DragValue, RichText, TextStyle};
use egui_dnd::DragDropResponse;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use std::mem::discriminant;

impl Gui {
    pub(super) fn draw_object_editor_window(
        &mut self,
        object_collection: &ObjectCollection,
        selected_object_id: Option<ObjectId>,
        selected_primitive_op_id: Option<PrimitiveOpId>,
    ) -> Vec<Command> {
        let mut commands = Vec::<Command>::new();

        let add_contents = |ui: &mut egui::Ui| {
            commands = layout_object_editor(
                ui,
                &mut self.gui_state,
                &self.egui_context,
                object_collection,
                selected_object_id,
                selected_primitive_op_id,
            );
        };
        egui::Window::new("Object Editor")
            .open(&mut self.sub_window_states.object_editor)
            .resizable(true)
            .vscroll(true)
            .hscroll(true)
            .show(&self.egui_context, add_contents);

        commands
    }
}

fn layout_object_editor(
    ui: &mut egui::Ui,
    gui_state: &mut GuiState,
    context: &egui::Context,
    object_collection: &ObjectCollection,
    selected_object_id: Option<ObjectId>,
    selected_primitive_op_id: Option<PrimitiveOpId>,
) -> Vec<Command> {
    let mut commands = Vec::<Command>::new();

    // selected object name
    let (selected_object, some_selected_object_id) = match label_and_get_selected_object(
        ui,
        &mut commands,
        object_collection,
        selected_object_id,
    ) {
        Some(value) => value,
        None => return commands,
    };

    object_properties_editor(ui, &mut commands, selected_object, some_selected_object_id);

    primitive_op_editor(
        ui,
        &mut commands,
        gui_state,
        selected_object,
        some_selected_object_id,
        selected_primitive_op_id,
    );

    primitive_op_list(
        ui,
        &mut commands,
        gui_state,
        context,
        selected_object,
        some_selected_object_id,
        selected_primitive_op_id,
    );

    commands
}

fn label_and_get_selected_object<'a>(
    ui: &mut egui::Ui,
    commands: &mut Vec<Command>,
    object_collection: &'a ObjectCollection,
    selected_object_id: Option<ObjectId>,
) -> Option<(&'a Object, ObjectId)> {
    let no_object_text = RichText::new("No object selected...").italics();

    let some_selected_object_id = match selected_object_id {
        Some(id) => id,
        None => {
            ui.label(no_object_text);
            return None;
        }
    };

    let selected_object = match object_collection.get_object(some_selected_object_id) {
        Some(o) => o,
        None => {
            // invalid object id
            debug!("selected object {} dropped", some_selected_object_id);
            commands.push(ValidationCommand::SelectedObject().into());

            ui.label(no_object_text);
            return None;
        }
    };

    let mut new_name = selected_object.name.clone();
    ui.horizontal(|ui_h| {
        ui_h.label("Name:");
        ui_h.text_edit_singleline(&mut new_name);
        let id_label = format!("id: {}", some_selected_object_id);
        ui_h.label(&id_label);
    });
    if new_name != selected_object.name {
        commands.push(Command::SetObjectName {
            object_id: some_selected_object_id,
            new_name,
        });
    }

    Some((selected_object, some_selected_object_id))
}

fn object_properties_editor(
    ui: &mut egui::Ui,
    commands: &mut Vec<Command>,
    object: &Object,
    object_id: ObjectId,
) {
    ui.separator();

    let original_origin = object.origin;
    let mut new_origin = original_origin;

    ui.horizontal(|ui| {
        ui.label("Origin:");
        ui.add(DragValue::new(&mut new_origin.x).speed(DRAG_INC))
            .changed();
        ui.add(DragValue::new(&mut new_origin.y).speed(DRAG_INC));
        ui.add(DragValue::new(&mut new_origin.z).speed(DRAG_INC));
    });

    if original_origin != new_origin {
        commands.push(Command::SetObjectOrigin {
            object_id: object_id,
            origin: new_origin,
        });
    }
}

fn primitive_op_editor(
    ui: &mut egui::Ui,
    commands: &mut Vec<Command>,
    gui_state: &mut GuiState,
    selected_object: &Object,
    selected_object_id: ObjectId,
    selected_primitive_op_id: Option<PrimitiveOpId>,
) {
    if let Some(selected_prim_op_id) = selected_primitive_op_id {
        existing_primitive_op_editor(
            ui,
            commands,
            gui_state,
            selected_object,
            selected_object_id,
            selected_prim_op_id,
        );
    } else {
        new_primitive_op_editor(ui, commands, gui_state, selected_object_id);
    }
}

fn existing_primitive_op_editor(
    ui: &mut egui::Ui,
    commands: &mut Vec<Command>,
    gui_state: &mut GuiState,
    selected_object: &Object,
    selected_object_id: ObjectId,
    selected_prim_op_id: PrimitiveOpId,
) {
    let mut primitive_op_edit_state = EditState::NoChange;

    let selected_object_id = selected_object_id;
    let (selected_primitive_op, selected_primitive_op_index) =
        match selected_object.get_primitive_op_and_index(selected_prim_op_id) {
            Some(primitive_op_and_index) => primitive_op_and_index,
            None => {
                // selected_prim_op_id not in selected_obejct -> invalid id
                debug!("selected object {} dropped", selected_object_id);
                commands.push(ValidationCommand::SelectedObject().into());

                new_primitive_op_editor(ui, commands, gui_state, selected_object_id);
                return;
            }
        };

    gui_state.set_primitive_op_edit_state(selected_primitive_op);

    ui.separator();

    ui.label(format!("Primitive op {}:", selected_primitive_op_index));

    // primitive type/op selection

    ui.horizontal(|ui_h| {
        // op drop down menu
        let possible_updated_op = op_drop_down(ui_h, gui_state.op_edit, selected_object_id);
        if let Some(updated_op) = possible_updated_op {
            // user edited the op via drop-down menu
            gui_state.op_edit = updated_op;
            primitive_op_edit_state = EditState::Modified;
        }

        // primitive type drop down menu
        let primitive_type_changed = primitive_type_drop_down(ui_h, gui_state, selected_object_id);
        primitive_op_edit_state = primitive_op_edit_state.combine(primitive_type_changed);
    });

    // primitive editor

    let primitive_edit_state = primitive_editor_ui(ui, gui_state);
    primitive_op_edit_state = primitive_op_edit_state.combine(primitive_edit_state);

    // delete button

    let delete_clicked = ui.button("Delete").clicked();
    if delete_clicked {
        let target_primitive_op = TargetPrimitiveOp::Id(selected_object_id, selected_prim_op_id);
        commands.push(Command::RemovePrimitiveOp(target_primitive_op));
        return;
    }

    match primitive_op_edit_state {
        EditState::Modified => {
            // update the primitive op data with what we've been using
            let target_primitive_op =
                TargetPrimitiveOp::Id(selected_object_id, selected_prim_op_id);
            commands.push(Command::SetPrimitiveOp {
                target_primitive_op,
                new_primitive: gui_state.primitive_edit,
                new_transform: gui_state.transform_edit,
                new_operation: gui_state.op_edit,
                new_blend: gui_state.blend_edit,
                new_albedo: gui_state.albedo_edit,
                new_specular: gui_state.specular_edit,
            });
        }
        EditState::NoChange => (),
    }
}

/// Returns wherever the object has been edited
fn new_primitive_op_editor(
    ui: &mut egui::Ui,
    commands: &mut Vec<Command>,
    gui_state: &mut GuiState,
    selected_object_id: ObjectId,
) {
    ui.separator();
    ui.label("New primitive");

    ui.horizontal(|ui_h| {
        // op drop down menu
        let possible_updated_op = op_drop_down(ui_h, gui_state.op_edit, selected_object_id);
        if let Some(updated_op) = possible_updated_op {
            // user edited the op via drop-down menu
            gui_state.op_edit = updated_op;
        }

        // primitive type drop down menu
        primitive_type_drop_down(ui_h, gui_state, selected_object_id);
    });

    // primitive editor

    primitive_editor_ui(ui, gui_state);

    // Add and Reset buttons

    let mut clicked_add = false;
    let mut clicked_reset = false;
    ui.horizontal(|ui_h| {
        clicked_add = ui_h.button("Add").clicked();
        clicked_reset = ui_h.button("Reset").clicked();
    });
    if clicked_add {
        if config_ui::SELECT_PRIMITIVE_OP_AFTER_ADD {
            commands.push(Command::PushPrimitiveOpAndSelect {
                object_id: selected_object_id,
                primitive: gui_state.primitive_edit.clone(),
                transform: gui_state.transform_edit,
                operation: gui_state.op_edit,
                blend: gui_state.blend_edit,
                albedo: gui_state.albedo_edit,
                specular: gui_state.specular_edit,
            });
        } else {
            commands.push(Command::PushPrimitiveOp {
                object_id: selected_object_id,
                primitive: gui_state.primitive_edit.clone(),
                transform: gui_state.transform_edit,
                operation: gui_state.op_edit,
                blend: gui_state.blend_edit,
                albedo: gui_state.albedo_edit,
                specular: gui_state.specular_edit,
            });
        }
    }
    if clicked_reset {
        gui_state.reset_primitive_op_fields();
    }
}

/// Returns true if the primitive type was changed. If this happens, gui_state.primitive_edit_state
/// gets set to the default of the chosen type.
fn primitive_type_drop_down(
    ui: &mut egui::Ui,
    gui_state: &mut GuiState,
    selected_object_id: ObjectId,
) -> EditState {
    let selected_primitive_type_name: &str = gui_state.primitive_edit.type_name();
    let mut type_has_changed = EditState::NoChange;

    ComboBox::from_id_salt(format!("primitive type drop down {:?}", selected_object_id))
        .width(0_f32)
        .selected_text(selected_primitive_type_name)
        .show_ui(ui, |ui_p| {
            for (variant_default_primitive, variant_type_name) in Primitive::variants_with_names() {
                // drop-down option for each primitive type
                let this_is_selected = discriminant(&gui_state.primitive_edit)
                    == discriminant(&variant_default_primitive);
                let label_clicked = ui_p
                    .selectable_label(this_is_selected, variant_type_name)
                    .clicked();

                if label_clicked & !this_is_selected {
                    // new primitive type was selected
                    type_has_changed = EditState::Modified;
                    gui_state.primitive_edit = variant_default_primitive;
                }
            }
        });

    type_has_changed
}

/// Draw the primitive op list. each list element can be dragged/dropped elsewhere in the list,
/// or selected with a button for editing.
fn primitive_op_list(
    ui: &mut egui::Ui,
    commands: &mut Vec<Command>,
    gui_state: &mut GuiState,
    context: &egui::Context,
    selected_object: &Object,
    selected_object_id: ObjectId,
    selected_primitive_op_id: Option<PrimitiveOpId>,
) {
    ui.separator();

    // new primitive op button
    let new_op_response =
        ui.selectable_label(selected_primitive_op_id.is_none(), "New primitive op");
    if new_op_response.clicked() {
        commands.push(Command::DeselectPrimtiveOp());
    }

    let selected_prim_op = match selected_primitive_op_id {
        Some(selected_prim_op_id) => {
            match selected_object.get_primitive_op(selected_prim_op_id) {
                Some(found_prim_op) => Some(found_prim_op),
                None => {
                    // selected_prim_op_id not in selected_obejct! invalid id so we should deselect
                    debug!("primitive op id not found in selected object!");
                    commands.push(Command::DeselectPrimtiveOp());
                    None
                }
            }
        }
        None => None,
    };

    // draw each item in the primitive op list
    let mut primitive_op_list_drag_state = gui_state.primitive_op_list_drag.clone();
    let drag_drop_response = primitive_op_list_drag_state.list_ui::<PrimitiveOp>(
        context,
        ui,
        selected_object.primitive_ops.iter(),
        // function to draw a single primitive op entry in the list
        |ui, drag_handle, index, primitive_op| {
            primitive_op_list_item(
                ui,
                commands,
                primitive_op,
                index,
                selected_prim_op,
                drag_handle,
                selected_object_id,
            );
        },
    );
    gui_state.primitive_op_list_drag = primitive_op_list_drag_state;

    // if an item has been dropped after being dragged, re-arrange the primtive ops list
    if let DragDropResponse::Completed(drag_indices) = drag_drop_response {
        commands.push(Command::ShiftPrimitiveOps {
            object_id: selected_object_id,
            source_index: drag_indices.source,
            target_index: drag_indices.target,
        });
    }
}

fn primitive_op_list_item(
    ui: &mut egui::Ui,
    commands: &mut Vec<Command>,
    primitive_op: &PrimitiveOp,
    index: usize,
    selected_prim_op: Option<&PrimitiveOp>,
    drag_handle: egui_dnd::handle::DragHandle<'_>,
    selected_object_id: ObjectId,
) {
    let draggable_text = RichText::new(format!("{}", index)).text_style(TextStyle::Monospace);

    // label text
    let primitive_op_text = RichText::new(format!(
        "{} {}",
        primitive_op.op.name(),
        primitive_op.primitive.type_name()
    ))
    .text_style(TextStyle::Monospace);

    // check if this primitive op is selected
    let is_selected = match selected_prim_op {
        Some(some_selected_prim_op) => some_selected_prim_op.id() == primitive_op.id(),
        None => false,
    };

    // draw ui for this primitive op
    ui.horizontal(|ui_h| {
        // anything inside the handle can be used to drag the item
        drag_handle.ui(ui_h, primitive_op, |handle_ui| {
            handle_ui.label(draggable_text);
        });

        // label to select this primitive op
        let prim_op_res = ui_h.selectable_label(is_selected, primitive_op_text);

        // primitive op selected
        if prim_op_res.clicked() {
            let target_primitive_op = TargetPrimitiveOp::Id(selected_object_id, primitive_op.id());
            commands.push(Command::SelectPrimitiveOp(target_primitive_op))
        }
    });
}

fn primitive_editor_ui(ui: &mut egui::Ui, gui_state: &mut GuiState) -> EditState {
    let primitive_edit_state = match &mut gui_state.primitive_edit {
        Primitive::Sphere(p) => sphere_editor_ui(ui, p),
        Primitive::Cube(p) => cube_editor_ui(ui, p),
        Primitive::UberPrimitive(p) => uber_primitive_editor_ui(ui, p),
    };
    let transform_edit_state = primitive_transform_editor_ui(ui, &mut gui_state.transform_edit);
    let blend_edit_state = blend_editor_ui(ui, &mut gui_state.blend_edit);
    let color_specular_edit_state =
        color_specular_editor_ui(ui, &mut gui_state.albedo_edit, &mut gui_state.specular_edit);

    transform_edit_state
        .combine(primitive_edit_state)
        .combine(blend_edit_state)
        .combine(color_specular_edit_state)
}
