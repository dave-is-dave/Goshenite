use super::gui_state::WindowStates;
use egui::Ui;

pub fn top_panel_layout(ui: &mut Ui, window_states: &mut WindowStates) {
    ui.horizontal_wrapped(|ui| {
        ui.visuals_mut().button_frame = false; // idk what this does tbh

        // light/dark theme toggle
        egui::widgets::global_dark_light_mode_switch(ui);

        ui.separator();

        // window toggles
        ui.toggle_value(&mut window_states.object_list, "Object List");
        ui.toggle_value(&mut window_states.object_editor, "Object Editor");
    });
}
