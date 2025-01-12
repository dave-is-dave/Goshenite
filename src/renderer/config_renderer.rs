use ash::vk::{self, KHR_MAINTENANCE4_NAME, KHR_SWAPCHAIN_NAME, KHR_SYNCHRONIZATION2_NAME};
use bort_vk::ApiVersion;
use std::ffi::CString;

pub fn required_device_extensions() -> [CString; 3] {
    // VK_KHR_swapchain, VK_KHR_synchronization2
    [
        KHR_SWAPCHAIN_NAME.to_owned(),
        KHR_SYNCHRONIZATION2_NAME.to_owned(),
        KHR_MAINTENANCE4_NAME.to_owned(), // core in 1.3
    ]
}

pub fn supports_required_features_1_0(supported_features: vk::PhysicalDeviceFeatures) -> bool {
    supported_features.fill_mode_non_solid != vk::FALSE
}
pub fn required_features_1_0() -> vk::PhysicalDeviceFeatures {
    vk::PhysicalDeviceFeatures {
        fill_mode_non_solid: vk::TRUE,
        ..Default::default()
    }
}

pub const MAX_VULKAN_VER: ApiVersion = ApiVersion::V1_2;
pub const MIN_VULKAN_VER: ApiVersion = ApiVersion::V1_2;
/// If true, the renderer will attempt to enable khronos valication layer. If VK_LAYER_KHRONOS_validation
/// is installed on the system, a debug callback will be created to log layer messages.
pub const ENABLE_VULKAN_VALIDATION: bool = cfg!(debug_assertions);

/// Function name of the entry point for shaders
pub const SHADER_ENTRY_POINT: &str = "main";

// G-buffer formats. Note that the combined bit total of these should be under 128bits to fit in tile storage on many tile-based architectures.
pub const FORMAT_NORMAL_BUFFER: vk::Format = vk::Format::R8G8B8A8_SNORM;
pub const FORMAT_ALBEDO_BUFFER: vk::Format = vk::Format::R8G8B8A8_UNORM;
pub const FORMAT_PRIMITIVE_ID_BUFFER: vk::Format = vk::Format::R32_UINT;

/// 1 second
pub const TIMEOUT_NANOSECS: u64 = 1_000_000_000;

/// Double-buffering
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub const CPU_ACCESS_BUFFER_SIZE: vk::DeviceSize = 1024;

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderOptions {
    pub enable_aabb_wire_display: bool,
}

pub const GIZMO_ARROW_STL_PATH: &str = "./assets/models/gizmo-arrow.stl";

pub const DISPLAY_UNAVAILABLE_TIMEOUT_NANOSECONDS: i32 = 10000;
