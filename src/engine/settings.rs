use crate::user_interface::{
    camera_control::CameraControlMappings, config_ui::DEFAULT_SCROLL_ZOOM_SENSITIVITY, theme::Theme,
};

// ~~ Json Setting Names ~~

pub const SETTING_NAME_LOOK_MAPPING: &str = "cameraLookMapping";
pub const SETTING_NAME_LOOK_MAPPING_2: &str = "cameraLookMapping2";
pub const SETTING_NAME_PAN_MAPPING: &str = "cameraPanMapping";
pub const SETTING_NAME_PAN_MAPPING_2: &str = "cameraPanMapping2";
pub const SETTING_NAME_ZOOM_MAPPING: &str = "cameraZoomMapping";
pub const SETTING_NAME_ZOOM_MAPPING_2: &str = "cameraZoomMapping2";

pub const SETTING_NAME_MOUSE_BUTTON: &str = "mouseButton";
pub const SETTING_NAME_MODIFIERS: &str = "modifiers";

pub const SETTING_NAME_MOUSE_LEFT: &str = "left";
pub const SETTING_NAME_MOUSE_RIGHT: &str = "right";
pub const SETTING_NAME_MOUSE_MIDDLE: &str = "middle";
pub const SETTING_NAME_MOUSE_BACK: &str = "back";
pub const SETTING_NAME_MOUSE_FORWARD: &str = "forward";

pub const SETTING_NAME_SHIFT: &str = "shift";
pub const SETTING_NAME_CONTROL: &str = "control";
pub const SETTING_NAME_ALT: &str = "alt";

pub const SETTING_NAME_SCROLL_ZOOM_SENSITIVITY: &str = "scrollZoomSensitivity";

// ~~ Settings Struct ~~

#[derive(Clone, Copy)]
pub struct Settings {
    pub scroll_zoom_sensitivity: f64,
    pub theme: Theme,
    pub camera_control_mappings: CameraControlMappings,
}

impl Settings {
    pub fn reset_all(&mut self) {
        *self = Self::default()
    }

    pub fn set_theme(&mut self, theme: Theme) {
        self.theme = theme;
        // todo does egui update automatically from winit::event?
        // may add color schemes in future -> egui::set_visuals_of
    }

    pub fn reset_scroll_zoom_sensitivity(&mut self) {
        self.scroll_zoom_sensitivity = DEFAULT_SCROLL_ZOOM_SENSITIVITY;
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            scroll_zoom_sensitivity: DEFAULT_SCROLL_ZOOM_SENSITIVITY,
            theme: Default::default(),
            camera_control_mappings: Default::default(),
        }
    }
}

// ~~ Json Encoding/Decoding Functions ~~

impl Settings {
    pub fn update_from_json(&mut self) {}
}
