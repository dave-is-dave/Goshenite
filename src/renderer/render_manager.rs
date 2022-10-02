use super::blit_pass::BlitPass;
use super::gui_renderer::GuiRenderer;
use super::scene_pass::ScenePass;
use crate::camera::Camera;
use crate::config;
use crate::gui::Gui;
use crate::primitives::primitives::PrimitiveCollection;
use crate::shaders::shader_interfaces::CameraPushConstant;
use anyhow::{anyhow, bail, Context};
use log::{debug, error, info, warn};
use std::sync::Arc;
use vulkano::{
    command_buffer,
    device::{
        self,
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo,
    },
    format::Format,
    image::{view::ImageView, ImageAccess, ImageUsage, StorageImage, SwapchainImage},
    instance::debug::{
        DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessenger,
        DebugUtilsMessengerCreateInfo,
    },
    instance::{Instance, InstanceCreateInfo, InstanceExtensions},
    pipeline::graphics::viewport::Viewport,
    render_pass::{LoadOp, StoreOp},
    swapchain::{self, PresentInfo, Surface, Swapchain},
    sync::{self, FlushError, GpuFuture},
    VulkanLibrary,
};
use winit::window::Window;

/// Contains Vulkan resources and methods to manage rendering
pub struct RenderManager {
    device: Arc<Device>,
    render_queue: Arc<Queue>,
    _transfer_queue: Arc<Queue>,
    _debug_callback: Option<DebugUtilsMessenger>,

    surface: Arc<Surface<Arc<Window>>>,
    swapchain: Arc<Swapchain<Arc<Window>>>,
    swapchain_image_views: Vec<Arc<ImageView<SwapchainImage<Arc<Window>>>>>,
    viewport: Viewport,
    render_image: Arc<ImageView<StorageImage>>,

    scene_pass: ScenePass,
    blit_pass: BlitPass,
    gui_pass: GuiRenderer,

    future_previous_frame: Option<Box<dyn GpuFuture>>, // todo description
    /// indicates that the swapchain needs to be recreated next frame
    recreate_swapchain: bool,
}

/// Indicates a queue family index
pub type QueueFamilyIndex = u32;

// ~~~ Public functions ~~~

impl RenderManager {
    /// Initializes Vulkan resources. If renderer fails to initialize, returns a string explanation.
    pub fn new(window: Arc<Window>, primitives: &PrimitiveCollection) -> anyhow::Result<Self> {
        // load vulkan library
        let vulkan_library = VulkanLibrary::new().context("loading vulkan library")?;
        info!(
            "loaded vulkan library, api version = {}",
            vulkan_library.api_version()
        );

        // required instance extensions for platform surface rendering
        let mut instance_extensions = vulkano_win::required_extensions(&vulkan_library);
        let mut instance_layers: Vec<String> = Vec::new();

        // check for validation layer/debug callback support
        let enable_debug_callback = if config::ENABLE_VULKAN_VALIDATION {
            if add_debug_validation(
                vulkan_library.clone(),
                &mut instance_extensions,
                &mut instance_layers,
            )
            .is_ok()
            {
                info!("enabling Vulkan validation layers and debug callback");
                true
            } else {
                warn!("validation layer debug callback requested but cannot be enabled");
                false
            }
        } else {
            debug!("Vulkan validation layers disabled by config");
            false
        };

        // create instance
        debug!("enabling instance extensions: {:?}", instance_extensions);
        debug!("enabling vulkan layers: {:?}", instance_layers);
        let instance = Instance::new(
            vulkan_library.clone(),
            InstanceCreateInfo {
                enabled_extensions: instance_extensions,
                enumerate_portability: true, // enable enumerating devices that use non-conformant vulkan implementations. (ex. MoltenVK)
                enabled_layers: instance_layers,
                ..Default::default()
            },
        )
        .context("creating vulkan instance")?;

        // setup debug callback
        let debug_callback = if enable_debug_callback {
            setup_debug_callback(instance.clone())
        } else {
            None
        };

        // create surface
        let surface = vulkano_win::create_surface_from_winit(window.clone(), instance.clone())
            .context("creating vulkan surface")?;

        // required device extensions
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };
        debug!("required vulkan device extensions: {:?}", device_extensions);

        // print available physical devices
        debug!("Available Vulkan physical devices:");
        for pd in instance
            .enumerate_physical_devices()
            .context("enumerating physical devices")?
        {
            debug!("\t{}", pd.properties().device_name);
        }
        // choose physical device and queue families
        let ChoosePhysicalDeviceReturn {
            physical_device,
            render_queue_family,
            transfer_queue_family,
        } = choose_physical_device(instance.clone(), &device_extensions, &surface)?;
        info!(
            "Using Vulkan device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );
        debug!("render queue family index = {}", render_queue_family);
        debug!("transfer queue family index = {}", transfer_queue_family);

        // queue create info(s) for creating render and transfer queues
        let single_queue = (render_queue_family == transfer_queue_family)
            && (physical_device.queue_family_properties()[render_queue_family as usize]
                .queue_count
                == 1);
        let queue_create_infos = if render_queue_family == transfer_queue_family {
            vec![QueueCreateInfo {
                queue_family_index: render_queue_family,
                queues: if single_queue {
                    vec![0.5]
                } else {
                    vec![0.5; 2]
                },
                ..Default::default()
            }]
        } else {
            vec![
                QueueCreateInfo {
                    queue_family_index: render_queue_family,
                    ..Default::default()
                },
                QueueCreateInfo {
                    queue_family_index: transfer_queue_family,
                    ..Default::default()
                },
            ]
        };

        // create device and queues
        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                enabled_features: device::Features {
                    dynamic_rendering: true,
                    ..device::Features::empty()
                },
                queue_create_infos,
                ..Default::default()
            },
        )
        .context("creating vulkan device and queues")?;
        let render_queue = queues
            .next()
            .expect("requested 1 queue from render_queue_family");
        let transfer_queue = if single_queue {
            render_queue.clone()
        } else {
            queues.next().expect("requested 1 unique transfer queue")
        };

        // create swapchain and images
        let (swapchain, swapchain_images) =
            create_swapchain(device.clone(), physical_device.clone(), surface.clone())?;
        debug!(
            "initial swapchain image size = {:?}",
            swapchain_images[0].dimensions()
        );

        // init dynamic viewport
        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [
                swapchain_images[0].dimensions().width() as f32,
                swapchain_images[0].dimensions().height() as f32,
            ],
            depth_range: 0.0..1.0,
        };

        // create swapchain image views
        let swapchain_image_views = swapchain_images
            .iter()
            .map(|image| ImageView::new_default(image.clone()))
            .collect::<Result<Vec<_>, _>>()
            .context("creating swapchain image views")?;

        // scene render target
        let render_image = create_render_image(
            render_queue.clone(),
            swapchain_images[0].dimensions().width_height(),
        )?;

        // init compute shader scene pass
        let scene_pass = ScenePass::new(
            device.clone(),
            primitives,
            swapchain_images[0].dimensions().width_height(),
            render_image.clone(),
        )?;

        // init blit pass
        let blit_pass = BlitPass::new(
            device.clone(),
            swapchain.image_format(),
            render_image.clone(),
        )?;

        // init gui renderer
        let gui_pass = GuiRenderer::new(
            device.clone(),
            transfer_queue.clone(),
            swapchain.image_format(),
        )?;

        // create futures used for frame synchronization
        let future_previous_frame = Some(sync::now(device.clone()).boxed());
        let recreate_swapchain = false;

        Ok(RenderManager {
            _debug_callback: debug_callback,
            device,
            render_queue,
            _transfer_queue: transfer_queue,
            surface,
            swapchain,
            swapchain_image_views,
            viewport,
            render_image,
            scene_pass,
            blit_pass,
            gui_pass,
            future_previous_frame,
            recreate_swapchain,
        })
    }

    /// Returns a mutable reference to the gui renderer so its resources can be updated by the gui
    pub fn gui_renderer_mut(&mut self) -> &mut GuiRenderer {
        &mut self.gui_pass
    }

    /// Submits Vulkan commands for rendering a frame.
    pub fn render_frame(
        &mut self,
        window_resize: bool,
        primitives: &PrimitiveCollection,
        gui: &Gui,
        camera: Camera,
    ) -> anyhow::Result<()> {
        // checks for submission finish and free locks on gpu resources
        self.future_previous_frame
            .as_mut()
            .unwrap()
            .cleanup_finished();

        self.recreate_swapchain = self.recreate_swapchain || window_resize;
        if self.recreate_swapchain {
            // recreate swapchain and skip frame render
            return self.recreate_swapchain();
        }

        // blocks when no images currently available (all have been submitted already)
        let (image_index, suboptimal, acquire_future) =
            match swapchain::acquire_next_image(self.swapchain.clone(), None) {
                Ok(r) => r,
                Err(swapchain::AcquireError::OutOfDate) => {
                    self.recreate_swapchain = true;
                    // recreate swapchain and skip frame render
                    return self.recreate_swapchain();
                }
                Err(e) => {
                    return Err(anyhow!(e)).context("aquiring swapchain image");
                }
            };
        if suboptimal {
            self.recreate_swapchain = true;
        }

        // todo shouldn't need to recreate each frame?
        self.scene_pass.update_primitives(primitives)?;

        // todo actually set this
        let need_srgb_conv = false;

        // record command buffer
        let mut builder = command_buffer::AutoCommandBufferBuilder::primary(
            self.device.clone(),
            self.render_queue.queue_family_index(),
            command_buffer::CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        // compute shader scene render
        let camera_push_constant = CameraPushConstant::new(
            glam::Mat4::inverse(&(camera.proj_matrix() * camera.view_matrix())),
            camera.position(),
        );
        self.scene_pass
            .record_commands(&mut builder, camera_push_constant)?;
        // begin render pass
        builder
            .begin_rendering(command_buffer::RenderingInfo {
                color_attachments: vec![Some(command_buffer::RenderingAttachmentInfo {
                    load_op: LoadOp::Clear,
                    store_op: StoreOp::Store,
                    clear_value: Some([0.0, 1.0, 0.0, 1.0].into()),
                    ..command_buffer::RenderingAttachmentInfo::image_view(
                        self.swapchain_image_views[image_index as usize].clone(),
                    )
                })],
                ..Default::default()
            })
            .context("recording vkCmdBeginRendering")?;
        // draw render image to screen
        self.blit_pass
            .record_commands(&mut builder, self.viewport.clone())?;
        // render gui todo return error
        self.gui_pass.record_commands(
            &mut builder,
            gui,
            need_srgb_conv,
            [
                self.viewport.dimensions[0] as u32,
                self.viewport.dimensions[1] as u32,
            ],
        )?;
        // end render pass
        builder
            .end_rendering()
            .context("recording vkCmdEndRendering")?;
        let command_buffer = builder.build().context("building frame command buffer")?;

        // submit
        let future = self
            .future_previous_frame
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(self.render_queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.render_queue.clone(),
                PresentInfo {
                    index: image_index,
                    ..PresentInfo::swapchain(self.swapchain.clone())
                },
            )
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                self.future_previous_frame = Some(future.boxed());
            }
            Err(FlushError::OutOfDate) => {
                self.recreate_swapchain = true;
                self.future_previous_frame = Some(sync::now(self.device.clone()).boxed());
            }
            Err(e) => {
                error!("Failed to flush future: {}", e);
                self.future_previous_frame = Some(sync::now(self.device.clone()).boxed());
            }
        }
        Ok(())
    }
}
// Private functions
impl RenderManager {
    /// Recreates the swapchain, render image and assiciated descriptor sets, then unsets `recreate_swapchain` trigger.
    fn recreate_swapchain(&mut self) -> anyhow::Result<()> {
        debug!("recreating swapchain and render targets...");

        let (new_swapchain, swapchain_images) =
            match self.swapchain.recreate(swapchain::SwapchainCreateInfo {
                image_extent: self.surface.window().inner_size().into(),
                ..self.swapchain.create_info()
            }) {
                Ok(r) => r,
                Err(e) => return Err(e.into()),
            };

        self.swapchain = new_swapchain;
        self.swapchain_image_views = swapchain_images
            .iter()
            .map(|image| ImageView::new_default(image.clone()).unwrap())
            .collect::<Vec<_>>();

        // set parameters for new resolution
        let resolution = swapchain_images[0].dimensions().width_height();
        self.viewport.dimensions = [resolution[0] as f32, resolution[1] as f32];

        self.render_image = create_render_image(
            self.render_queue.clone(),
            swapchain_images[0].dimensions().width_height(),
        )?;

        // update scene pass
        self.scene_pass
            .update_render_target(resolution, self.render_image.clone())?;

        // update blit pass
        self.blit_pass
            .update_render_image(self.render_image.clone())?;

        // unset trigger
        self.recreate_swapchain = false;

        Ok(())
    }
}

/// Checks for VK_EXT_debug_utils support and presence khronos validation layers
/// If both can be enabled, adds them to provided extension and layer lists
fn add_debug_validation(
    vulkan_library: Arc<VulkanLibrary>,
    instance_extensions: &mut InstanceExtensions,
    instance_layers: &mut Vec<String>,
) -> anyhow::Result<()> {
    // check debug utils extension support
    if vulkan_library.supported_extensions().ext_debug_utils {
        info!("VK_EXT_debug_utils was requested and is supported");
    } else {
        warn!("VK_EXT_debug_utils was requested but is unsupported");
        bail!(
            "vulkan extension {} was requested but is unsupported",
            "VK_EXT_debug_utils"
        )
    }

    // check validation layers are present
    let validation_layer = "VK_LAYER_KHRONOS_validation";
    if vulkan_library
        .layer_properties()?
        .find(|l| l.name() == validation_layer)
        .is_some()
    {
        info!("{} was requested and found", validation_layer);
    } else {
        warn!(
            "{} was requested but was not found (may not be installed)",
            validation_layer
        );
        bail!(
            "requested vulkan layer {} not found (may not be installed)",
            validation_layer
        )
    }

    // add VK_EXT_debug_utils and VK_LAYER_LUNARG_standard_validation
    instance_extensions.ext_debug_utils = true;
    instance_layers.push(validation_layer.to_owned());
    Ok(())
}

fn setup_debug_callback(instance: Arc<Instance>) -> Option<DebugUtilsMessenger> {
    unsafe {
        match DebugUtilsMessenger::new(
            instance,
            DebugUtilsMessengerCreateInfo {
                message_severity: DebugUtilsMessageSeverity {
                    error: true,
                    warning: true,
                    information: true,
                    verbose: false,
                    ..DebugUtilsMessageSeverity::empty()
                },
                message_type: DebugUtilsMessageType {
                    general: true,
                    validation: true,
                    performance: true,
                    ..DebugUtilsMessageType::empty()
                },
                ..DebugUtilsMessengerCreateInfo::user_callback(Arc::new(|msg| {
                    vulkan_callback::process_debug_callback(msg)
                }))
            },
        ) {
            Ok(x) => Some(x),
            Err(e) => {
                warn!("failed to setup vulkan debug callback: {}", e,);
                None
            }
        }
    }
}

/// Create swapchain and swapchain images
fn create_swapchain(
    device: Arc<Device>,
    physical_device: Arc<PhysicalDevice>,
    surface: Arc<Surface<Arc<Window>>>,
) -> anyhow::Result<(
    Arc<Swapchain<Arc<Window>>>,
    Vec<Arc<SwapchainImage<Arc<Window>>>>,
)> {
    // todo prefer sRGB (linux sRGB)
    let image_format = physical_device
        .surface_formats(&surface, Default::default())
        .context("querying surface formats")?
        .get(0)
        .expect("vulkan driver should support at least 1 surface format... right?")
        .0;
    debug!("swapchain image format = {:?}", image_format);

    let surface_capabilities = physical_device
        .surface_capabilities(&surface, Default::default())
        .context("querying surface capabilities")?;
    let composite_alpha = surface_capabilities
        .supported_composite_alpha
        .iter()
        .max_by_key(|c| match c {
            swapchain::CompositeAlpha::PostMultiplied => 4,
            swapchain::CompositeAlpha::Inherit => 3,
            swapchain::CompositeAlpha::Opaque => 2,
            swapchain::CompositeAlpha::PreMultiplied => 1, // because cbf implimenting this logic
            _ => 0,
        })
        .expect("surface should support at least 1 composite mode... right?");
    debug!("swapchain composite alpha = {:?}", composite_alpha);

    let mut present_modes = physical_device
        .surface_present_modes(&surface)
        .context("querying surface present modes")?;
    let present_mode = present_modes
        .find(|&pm| pm == swapchain::PresentMode::Mailbox)
        .unwrap_or(swapchain::PresentMode::Fifo);
    debug!("swapchain present mode = {:?}", present_mode);

    swapchain::Swapchain::new(
        device.clone(),
        surface.clone(),
        swapchain::SwapchainCreateInfo {
            min_image_count: surface_capabilities.min_image_count,
            image_extent: surface.window().inner_size().into(),
            image_usage: ImageUsage {
                color_attachment: true,
                ..ImageUsage::empty()
            },
            image_format: Some(image_format),
            composite_alpha,
            present_mode,
            ..Default::default()
        },
    )
    .context("creating swapchain")
}

/// Creates the render target for the scene render. _Note that the value of `access_queue` isn't actually used
/// in the vulkan image creation create info._
fn create_render_image(
    access_queue: Arc<Queue>,
    size: [u32; 2],
) -> anyhow::Result<Arc<ImageView<StorageImage>>> {
    // format must match what's specified in the compute shader layout
    let render_image_format = Format::R8G8B8A8_UNORM;
    StorageImage::general_purpose_image_view(
        access_queue,
        size,
        render_image_format,
        ImageUsage {
            storage: true,
            sampled: true,
            ..ImageUsage::empty()
        },
    )
    .context("creating render image")
}

/// Choose physical device and queue families
fn choose_physical_device(
    instance: Arc<Instance>,
    device_extensions: &DeviceExtensions,
    surface: &Arc<Surface<Arc<Window>>>,
) -> anyhow::Result<ChoosePhysicalDeviceReturn> {
    instance
        .enumerate_physical_devices()
        .context("enumerating physical devices")?
        // filter for vulkan version support
        .filter(|p| {
            p.api_version()
                >= vulkano::Version::major_minor(config::VULKAN_VER_MAJ, config::VULKAN_VER_MIN)
        })
        // filter for required device extensions
        .filter(|p| p.supported_extensions().contains(device_extensions))
        // filter for queue support
        .filter_map(|p| {
            // get queue family index for main queue used for rendering
            let render_family = p
                .queue_family_properties()
                .iter()
                // because we want the queue family index
                .enumerate()
                .position(|(i, q)| {
                    // must support our surface and essential operations
                    q.queue_flags.graphics
                        && q.queue_flags.compute
                        && q.queue_flags.transfer
                        && p.surface_support(i as u32, surface).unwrap_or(false)
                });
            if let Some(render_index) = render_family {
                // attempt to find a different queue family that we can use for asynchronous transfer operations
                // e.g. uploading image/buffer data while rendering
                let transfer_family = p
                    .queue_family_properties()
                    .iter()
                    // because we want the queue family index
                    .enumerate()
                    // exclude the queue family we've already found and filter by transfer operation support
                    .filter(|(i, q)| *i != render_index && q.queue_flags.transfer)
                    // some drivers expose a queue that only supports transfer operations (for this very purpose) which is preferable
                    .max_by_key(|(_, q)| {
                        if !q.queue_flags.compute && !q.queue_flags.graphics {
                            1
                        } else {
                            0
                        }
                    })
                    .map(|(i, _)| i);
                Some(ChoosePhysicalDeviceReturn {
                    physical_device: p,
                    render_queue_family: render_index as QueueFamilyIndex,
                    transfer_queue_family: transfer_family.unwrap_or(render_index)
                        as QueueFamilyIndex,
                })
            } else {
                // failed to find suitable main queue
                None
            }
        })
        // preference of device type
        .max_by_key(
            |ChoosePhysicalDeviceReturn {
                 physical_device, ..
             }| match physical_device.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 4,
                PhysicalDeviceType::IntegratedGpu => 3,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 1,
                PhysicalDeviceType::Other => 0,
                _ne => 0,
            },
        )
        .with_context(|| format!("could not find a suitable vulkan physical device. requirements:\n
            \t- must support minimum vulkan version {}.{}\n
            \t- must contain queue family supporting graphics, compute, transfer and surface operations\n
            \t- must support device extensions: {:?}",
            config::VULKAN_VER_MAJ, config::VULKAN_VER_MIN, device_extensions))
}
/// Physical device and queue family indices returned by [`RenderManager::choose_physical_device`]
struct ChoosePhysicalDeviceReturn {
    pub physical_device: Arc<PhysicalDevice>,
    pub render_queue_family: QueueFamilyIndex,
    pub transfer_queue_family: QueueFamilyIndex,
}

/// This mod just makes the module path unique for debug callbacks in the log
mod vulkan_callback {
    use log::{debug, error, warn};
    use vulkano::instance::debug::Message;
    /// Prints/logs a Vulkan validation layer message
    pub fn process_debug_callback(msg: &Message) {
        let ty = if msg.ty.general {
            "GENERAL"
        } else if msg.ty.validation {
            "VALIDATION"
        } else if msg.ty.performance {
            "PERFORMANCE"
        } else {
            "TYPE-UNKNOWN"
        };
        if msg.severity.error {
            error!("Vulkan [{}]:\n{}", ty, msg.description);
        } else if msg.severity.warning {
            warn!("Vulkan [{}]:\n{}", ty, msg.description);
        } else if msg.severity.information {
            debug!("Vulkan [{}]:\n{}", ty, msg.description);
        } else if msg.severity.verbose {
            debug!("Vulkan [{}]:\n{}", ty, msg.description);
        } else {
            debug!("Vulkan [{}] (SEVERITY-UNKONWN):\n{}", ty, msg.description);
        };
    }
}

/*
/// Describes the types of errors encountered by the renderer
// todo handle this stuff
#[derive(Debug)]
pub enum RenderManagerError {
    /// Requested dimensions are not within supported range when attempting to create a render target (swapchain)
    /// This error tends to happen when the user is manually resizing the window.
    /// Simply restarting the loop is the easiest way to fix this issue.
    ///
    /// Equivalent to vulkano [SwapchainCreationError::ImageExtentNotSupported](`vulkano::swapchain::SwapchainCreationError::ImageExtentNotSupported`)
    SurfaceSizeUnsupported {
        provided: [u32; 2],
        min_supported: [u32; 2],
        max_supported: [u32; 2],
    },
    // todo VulkanError recoverable case handling... clamp inner window size in Engine::process_frame()?
    // The window surface is no longer accessible and must be recreated.
    // Invalidates the RenderManger and requires re-initialization.
    //
    // Equivalent to vulkano [SurfacePropertiesError::SurfaceLost](`vulkano::device::physical::SurfacePropertiesError::SurfaceLost`)
    //SurfaceLost,
}
impl fmt::Display for RenderManagerError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            //RenderManagerError::SurfaceLost =>
            //    write!(f, "the Vulkan surface is no longer accessible, thus invalidating this RenderManager instance"),
            RenderManagerError::SurfaceSizeUnsupported{ provided, min_supported, max_supported } =>
                write!(f, "cannot create render target with requested dimensions = {:?}. min size = {:?}, max size = {:?}",
                    provided, min_supported, max_supported),
        }
    }
}
impl From<SwapchainCreationError> for RenderManagerError {
    fn from(error: SwapchainCreationError) -> Self {
        match error {
            // this error tends to happen when the user is manually resizing the window.
            // simply restarting the loop is the easiest way to fix this issue.
            SwapchainCreationError::ImageExtentNotSupported {
                provided,
                min_supported,
                max_supported,
            } => {
                let err = RenderManagerError::SurfaceSizeUnsupported {
                    provided,
                    min_supported,
                    max_supported,
                };
                debug!("cannot create swapchain: {}", err);
                err
            }
        }
    }
}
*/
