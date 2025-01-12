use super::Gui;
use crate::user_interface::gui_state::SubWindowStates;
use egui::Ui;

impl Gui {
    pub(super) fn draw_bottom_panel(&mut self) {
        egui::TopBottomPanel::bottom("main top panel").show(&self.egui_context, |ui| {
            bottom_panel_layout(ui, &mut self.sub_window_states);
        });
    }
}

fn bottom_panel_layout(ui: &mut Ui, window_states: &mut SubWindowStates) {
    ui.horizontal_wrapped(|ui| {
        ui.visuals_mut().button_frame = false; // idk what this does tbh

        // light/dark theme toggle
        egui::widgets::global_theme_preference_switch(ui);

        ui.separator();

        // window toggles
        ui.toggle_value(&mut window_states.object_list, "Object List");
        ui.toggle_value(&mut window_states.object_editor, "Object Editor");
        ui.toggle_value(&mut window_states.camera_control, "Camera Control");

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            egui::warn_if_debug_build(ui);
            #[cfg(debug_assertions)]
            ui.toggle_value(&mut window_states.debug_options, "Debug Options");
        });
    });
}
