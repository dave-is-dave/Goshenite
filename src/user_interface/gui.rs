use self::command_palette::GuiStateCommandPalette;
use super::{
    camera::Camera,
    gui_state::{GuiState, SubWindowStates},
};
use crate::{
    engine::{
        commands::{Command, CommandWithSource},
        object::{
            object::ObjectId,
            object_collection::ObjectCollection,
            primitive_op::{PrimitiveOp, PrimitiveOpId},
        },
    },
    renderer::config_renderer::RenderOptions,
};
use egui::{TextWrapMode, TexturesDelta};
use egui_winit::EventResponse;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use std::sync::Arc;
use winit::window::Window;

// various gui sections
mod bottom_panel;
mod camera_control;
mod command_palette;
mod debug_options;
mod object_editor;
mod object_list;

/// Describes how something has been edited/added/removed by a function
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum EditState {
    NoChange,
    Modified,
}

impl EditState {
    pub fn combine(self, other: Self) -> Self {
        self.max(other)
    }
}

/// Controller for an [`egui`] immediate-mode gui
pub struct Gui {
    egui_context: egui::Context,
    window: Arc<Window>,
    winit_state: egui_winit::State,
    mesh_primitives: Vec<egui::ClippedPrimitive>,
    sub_window_states: SubWindowStates,
    gui_state: GuiState,
    command_palette_state: GuiStateCommandPalette,
    textures_delta_accumulation: Vec<TexturesDelta>,
}

// Public functions

impl Gui {
    /// Creates a new [`Gui`].
    /// * `window`: [`winit`] window
    /// * `max_texture_side`: maximum size of a texture. Query from graphics driver using
    /// [`crate::renderer::render_manager::RenderManager::max_image_array_layers`]
    pub fn new(window: Arc<Window>, scale_factor: f32) -> Self {
        let egui_context = egui::Context::default();
        egui_context.set_style(egui::Style {
            // disable sentance wrap by default (horizontal scroll instead)
            wrap_mode: Some(TextWrapMode::Extend),
            ..Default::default()
        });

        // todo max_texture_side?
        let winit_state = egui_winit::State::new(
            egui_context.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(scale_factor),
            None,
            None,
        );

        Self {
            egui_context,
            window,
            winit_state,
            mesh_primitives: Default::default(),
            sub_window_states: Default::default(),
            gui_state: Default::default(),
            command_palette_state: Default::default(),
            textures_delta_accumulation: Default::default(),
        }
    }

    /// Updates egui_context state by winit window event.
    /// Returns `true` if egui wants exclusive use of this event
    /// (e.g. a mouse click on an egui window, or entering text into a text field).
    /// For instance, if you use egui for a game, you want to first call this
    /// and only when this returns `false` pass on the events to your game.
    ///
    /// Note that egui uses `tab` to move focus between elements, so this will always return `true` for tabs.
    pub fn process_event(&mut self, event: &winit::event::WindowEvent) -> EventResponse {
        self.winit_state
            .on_window_event(self.window.as_ref(), event)
    }

    /// Get a reference to the clipped meshes required for rendering
    pub fn mesh_primitives(&self) -> &Vec<egui::ClippedPrimitive> {
        &self.mesh_primitives
    }

    pub fn scale_factor(&self, window: &Window) -> f32 {
        egui_winit::pixels_per_point(&self.egui_context, window)
    }

    pub fn set_scale_factor(&mut self, scale_factor: f32) {
        self.egui_context.set_pixels_per_point(scale_factor);
    }

    /// Call this when the selected object is changed
    pub fn selected_object_changed(&mut self) {
        self.gui_state.primitive_op_list_drag = Default::default();
    }

    /// Call this when a primitive op is selected
    pub fn primitive_op_selected(&mut self, selected_primitive_op: &PrimitiveOp) {
        self.gui_state
            .set_selected_primitive_op(selected_primitive_op);
    }

    pub fn update_gui(
        &mut self,
        object_collection: &ObjectCollection,
        window: &Window,
        camera: Camera,
        selected_object_id: Option<ObjectId>,
        selected_primitive_op_id: Option<PrimitiveOpId>,
        render_options: RenderOptions,
    ) -> anyhow::Result<Vec<CommandWithSource>> {
        let mut commands = Vec::<Command>::new();

        // begin frame
        let raw_input = self.winit_state.take_egui_input(window);
        self.egui_context.begin_pass(raw_input);

        // draw

        self.draw_bottom_panel();

        if self.sub_window_states.object_list {
            let mut new_commands =
                self.draw_object_list_window(object_collection, selected_object_id);
            commands.append(&mut new_commands);
        }

        if self.sub_window_states.object_editor {
            let mut new_commands = self.draw_object_editor_window(
                object_collection,
                selected_object_id,
                selected_primitive_op_id,
            );
            commands.append(&mut new_commands);
        }

        if self.sub_window_states.camera_control {
            let mut new_commands = self.draw_camera_control_window(camera);
            commands.append(&mut new_commands);
        }

        if self.sub_window_states.command_palette {
            let new_command = self.draw_command_palette(window);
            if let Some(some_command) = new_command {
                commands.push(some_command);
                // close command palette after command has been selected
                self.sub_window_states.command_palette = false;
            }
        }

        if self.sub_window_states.debug_options {
            let mut new_commands = self.draw_debug_options_window(render_options);
            commands.append(&mut new_commands);
        }

        // end frame
        let egui::FullOutput {
            platform_output,
            textures_delta,
            shapes,
            pixels_per_point,
            viewport_output: _,
        } = self.egui_context.end_pass();
        self.winit_state
            .handle_platform_output(&self.window, platform_output);

        // store clipped primitive data for use by the renderer
        self.mesh_primitives = self.egui_context.tessellate(shapes, pixels_per_point);

        // store required texture changes for the renderer to apply updates
        if !textures_delta.is_empty() {
            self.textures_delta_accumulation.push(textures_delta);
        }

        Ok(commands
            .into_iter()
            .map(|command| CommandWithSource::new_from_gui(command))
            .collect())
    }

    pub fn set_cursor_icon(&self, cursor_icon: egui::CursorIcon) {
        self.egui_context.set_cursor_icon(cursor_icon);
    }

    /// Returns texture update info accumulated since the last call to this function.
    pub fn get_and_clear_textures_delta(&mut self) -> Vec<TexturesDelta> {
        std::mem::take(&mut self.textures_delta_accumulation)
    }

    pub fn set_command_palette_visability(&mut self, is_open: bool) {
        self.sub_window_states.command_palette = is_open;
    }
}
