use crate::{
    common::is_format_srgb, device::Device, instance::Instance, physical_device::PhysicalDevice,
    surface::Surface, ALLOCATION_CALLBACK,
};
use anyhow::Context;
use ash::{extensions::khr, prelude::VkResult, vk};
use core::slice::SlicePattern;
use std::cmp::{max, min};

/// Checks surface support for the first compositie alpha flag in order of preference:
/// 1. `vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED`
/// 2. `vk::CompositeAlphaFlagsKHR::OPAQUE`
/// 3. `vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED` (because cbf implimenting the logic for this)
/// 4. `vk::CompositeAlphaFlagsKHR::INHERIT` (noooope)
pub fn choose_composite_alpha(
    surface_capabilities: vk::SurfaceCapabilitiesKHR,
) -> vk::CompositeAlphaFlagsKHR {
    let supported_composite_alpha = surface_capabilities.supported_composite_alpha;
    let composite_alpha_preference_order = [
        vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
        vk::CompositeAlphaFlagsKHR::OPAQUE,
        vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
        vk::CompositeAlphaFlagsKHR::INHERIT,
    ];
    composite_alpha_preference_order
        .into_iter()
        .find(|&ca| supported_composite_alpha.contains(ca))
        .expect("driver should support at least one type of composite alpha!")
}

/// Returns the first SRGB surface format in the vec.
pub fn get_first_srgb_surface_format(
    surface_formats: &Vec<vk::SurfaceFormatKHR>,
) -> vk::SurfaceFormatKHR {
    surface_formats
        .iter()
        .cloned()
        // use the first SRGB format we find
        .find(|vk::SurfaceFormatKHR { format, .. }| is_format_srgb(*format))
        // otherwise just go with the first format
        .unwrap_or(surface_formats[0])
}

#[derive(Debug, Clone)]
pub struct SwapchainProperties {
    pub create_flags: vk::SwapchainCreateFlagsKHR,
    pub image_count: u32,

    pub surface_format: vk::SurfaceFormatKHR,
    pub dimensions: [u32; 2],
    pub array_layers: u32,
    pub image_usage: vk::ImageUsageFlags,
    pub sharing_mode: vk::SharingMode,
    pub queue_family_indices: Option<Vec<u32>>,

    pub pre_transform: vk::SurfaceTransformFlagsKHR,
    pub composite_alpha: vk::CompositeAlphaFlagsKHR,
    pub present_mode: vk::PresentModeKHR,
    pub clipping_enabled: bool,
}

impl SwapchainProperties {
    pub fn create_info_builder(
        &self,
        surface_handle: vk::SurfaceKHR,
        old_swapchain_handle: vk::SwapchainKHR,
    ) -> vk::SwapchainCreateInfoKHRBuilder {
        let mut builder = vk::SwapchainCreateInfoKHR::builder()
            .flags(self.create_flags)
            .surface(surface_handle)
            .min_image_count(self.image_count)
            .image_format(self.surface_format.format)
            .image_color_space(self.surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(self.array_layers)
            .image_usage(self.image_usage)
            .image_sharing_mode(self.sharing_mode)
            .pre_transform(self.pre_transform)
            .composite_alpha(self.composite_alpha)
            .present_mode(self.present_mode)
            .clipped(self.clipping_enabled)
            .old_swapchain(old_swapchain_handle);
        if let Some(queue_family_indices) = self.queue_family_indices {
            builder = builder.queue_family_indices(queue_family_indices.as_slice());
        }

        builder
    }
}

pub struct Swapchain {
    handle: vk::SwapchainKHR,
    swapchain_loader: khr::Swapchain,
    properties: SwapchainProperties,
}

impl Swapchain {
    /// Prefers the following settings:
    /// - present mode = `vk::PresentModeKHR::MAILBOX`
    /// - pre-transform = `vk::SurfaceTransformFlagsKHR::IDENTITY`
    ///
    /// `preferred_image_count` is clamped based on `vk::SurfaceCapabilitiesKHR`.
    ///
    /// `surface_format`, `composite_alpha` and `image_usage` are unchecked.
    ///
    /// Sharing mode is set to `vk::SharingMode::EXCLUSIVE`, only 1 array layer, and clipping is enabled.
    pub fn new(
        instance: &Instance,
        device: &Device,
        surface: &Surface,
        physical_device: &PhysicalDevice,

        preferred_image_count: u32,
        surface_format: vk::SurfaceFormatKHR,
        composite_alpha: vk::CompositeAlphaFlagsKHR,
        image_usage: vk::ImageUsageFlags,
        window_dimensions: [u32; 2],
    ) -> anyhow::Result<Self> {
        let swapchain_loader = khr::Swapchain::new(instance.inner(), device.inner());

        let surface_capabilities = surface
            .get_physical_device_surface_capabilities(physical_device)
            .context("get_physical_device_surface_capabilities")?;

        let image_count = max(
            min(preferred_image_count, surface_capabilities.max_image_count),
            surface_capabilities.min_image_count,
        );

        let extent = match surface_capabilities.current_extent.width {
            std::u32::MAX => vk::Extent2D {
                width: window_dimensions[0],
                height: window_dimensions[1],
            },
            _ => surface_capabilities.current_extent,
        };

        let present_modes = surface
            .get_physical_device_surface_present_modes(physical_device)
            .context("get_physical_device_surface_present_modes")?;
        let present_mode = present_modes
            .into_iter()
            .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO);

        // should think more about this if targeting mobile in the future...
        let pre_transform = if surface_capabilities
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            surface_capabilities.current_transform
        };

        let properties = SwapchainProperties {
            image_count,
            surface_format,
            dimensions: [extent.width, extent.height],
            image_usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            pre_transform,
            composite_alpha,
            present_mode,
            clipping_enabled: true,
            ..SwapchainProperties::default()
        };

        let swapchain_create_info_builder =
            properties.create_info_builder(surface.handle(), vk::SwapchainKHR::null());
        let handle = unsafe {
            swapchain_loader.create_swapchain(&swapchain_create_info_builder, ALLOCATION_CALLBACK)
        }
        .context("creating swapchain")?;

        Ok(Self {
            handle,
            swapchain_loader,
            properties,
        })
    }

    pub fn get_swapchain_images(&self) -> VkResult<Vec<vk::Image>> {
        unsafe { self.swapchain_loader.get_swapchain_images(self.handle) }
    }

    // Getters

    pub fn handle(&self) -> vk::SwapchainKHR {
        self.handle
    }

    pub fn swapchain_loader(&self) -> &khr::Swapchain {
        &self.swapchain_loader
    }

    pub fn properties(&self) -> &SwapchainProperties {
        &self.properties
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            self.swapchain_loader
                .destroy_swapchain(self.handle, ALLOCATION_CALLBACK)
        };
    }
}
