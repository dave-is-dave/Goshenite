use super::{
    commands::{CommandWithSource, TargetPrimitiveOp},
    config_engine,
    main_thread::MainThreadChannels,
    object::{
        object::ObjectId, object_collection::ObjectCollection, operation::Operation,
        primitive_op::PrimitiveOpId,
    },
    primitives::{
        cube::Cube, primitive::Primitive, primitive_transform::PrimitiveTransform, sphere::Sphere,
    },
    render_thread::{start_render_thread, RenderThreadChannels, RenderThreadCommand},
    settings::Settings,
};
use crate::{
    config,
    engine::object::object::Object,
    helper::anyhow_panic::anyhow_unwrap,
    renderer::{
        config_renderer::RenderOptions, element_id_reader::ElementAtPoint,
        render_manager::RenderManager,
    },
    user_interface::{
        camera::Camera,
        cursor::{Cursor, CursorEvent},
        gui::Gui,
        keyboard_modifiers::KeyboardModifierStates,
        mouse_button::MouseButton,
    },
};
use glam::Vec3;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use single_value_channel::NoReceiverError;
use std::{
    collections::VecDeque,
    env,
    fmt::Debug,
    sync::{mpsc::SendError, Arc},
    thread::JoinHandle,
    time::Instant,
};
use winit::{
    event::{ElementState, Event, KeyEvent, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

// engine_instance sub-modules (files in engine_instance directory)
mod commands_impl;

#[derive(Clone, Copy)]
pub enum EngineCommand {
    Run,
    Pause,
    Quit,
}

pub struct EngineController {
    window: Arc<Window>,

    // state
    scale_factor: f64,
    object_collection: ObjectCollection, // note: some engine code written on the assumtion that there is only one object collection
    main_thread_frame_number: u64,
    pending_commands: VecDeque<CommandWithSource>,
    selected_object_id: Option<ObjectId>,
    selected_primitive_op_id: Option<PrimitiveOpId>,
    render_options: RenderOptions,
    keyboard_modifier_states: KeyboardModifierStates,

    // controllers
    cursor: Cursor,
    camera: Camera,
    gui: Gui,

    // settings
    settings: Settings,

    // window thread (main thread)
    main_thread_channels: MainThreadChannels,

    // render thread
    render_thread_handle: Option<JoinHandle<()>>, // option so can be consumed by `join`
    render_thread_channels: RenderThreadChannels,
}

// ~~ Public Functions ~~

impl EngineController {
    pub fn new(
        window: Arc<Window>,
        main_thread_channels: MainThreadChannels,
    ) -> anyhow::Result<Self> {
        let scale_factor_override: Option<f64> = match env::var(config::ENV::SCALE_FACTOR) {
            Ok(s) => s.parse::<f64>().ok(),
            _ => None,
        };
        let scale_factor = scale_factor_override.unwrap_or(window.scale_factor());

        let cursor = Cursor::new();

        let camera = Camera::new(window.inner_size().into())?;

        let mut renderer = RenderManager::new(window.clone(), scale_factor as f32)?;
        renderer.update_camera(&camera)?;

        let gui = Gui::new(window.clone(), scale_factor as f32);

        let mut object_collection = ObjectCollection::new();

        // start render thread
        let (render_thread_handle, render_thread_channels) = start_render_thread(renderer);

        // ~~ TESTING OBJECTS START ~~

        object_testing(&mut object_collection);
        //create_default_cube_object(&mut self.object_collection);

        // ~~ TESTING OBJECTS END ~~

        Ok(EngineController {
            window,

            scale_factor,
            object_collection,
            main_thread_frame_number: 0,
            pending_commands: VecDeque::new(),
            selected_object_id: None,
            selected_primitive_op_id: None,
            render_options: RenderOptions::default(),
            keyboard_modifier_states: KeyboardModifierStates::default(),

            cursor,
            camera,
            gui,

            settings: Settings::default(),

            main_thread_channels,

            render_thread_handle: Some(render_thread_handle),
            render_thread_channels,
        })
    }

    pub fn run(&mut self) -> anyhow::Result<()> {
        loop {
            let engine_command = self.main_thread_channels.latest_command();
            match engine_command {
                Some(EngineCommand::Run) => (),
                None => (), // just keep running
                Some(EngineCommand::Pause) => continue,
                Some(EngineCommand::Quit) => {
                    self.shut_down();
                    return Ok(());
                }
            }

            let frame_res = self.run_frame();

            match frame_res {
                Ok(EngineCommand::Quit) => {
                    self.shut_down();
                    return Ok(());
                }
                Err(e) => {
                    self.shut_down();
                    return Err(e);
                }
                _ => (),
            }
        }
    }
}

// ~~ Private Functions ~~

impl EngineController {
    /// The main loop of the engine thread
    fn run_frame(&mut self) -> anyhow::Result<EngineCommand> {
        let events = self.main_thread_channels.get_events()?;

        for event in events {
            match event {
                // process window events and update state
                Event::WindowEvent { event, .. } => {
                    let process_input_res = self.process_window_event(event);

                    if let Err(e) = process_input_res {
                        error!("error while processing input: {}", e);
                    }
                }

                _ => (),
            }
        }

        self.update_engine()?;

        Ok(EngineCommand::Run)
    }

    /// Process window events and update state
    fn process_window_event(&mut self, event: WindowEvent) -> Result<(), EngineError> {
        trace!("winit event: {:?}", event);

        // egui event handling
        let captured_by_gui = self.gui.process_event(&event).consumed;

        // engine event handling
        match event {
            // cursor moved. triggered when cursor is in window or if currently dragging and started in the window (on linux at least)
            WindowEvent::CursorMoved { position, .. } => self.cursor.set_position(position.into()),

            // send mouse button events to cursor state
            WindowEvent::MouseInput { state, button, .. } => {
                self.cursor.set_click_state(button, state, captured_by_gui)
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.cursor.accumulate_scroll_delta(delta, captured_by_gui)
            }

            // cursor entered window
            WindowEvent::CursorEntered { .. } => self.cursor.set_in_window_state(true),

            // cursor left window
            WindowEvent::CursorLeft { .. } => self.cursor.set_in_window_state(false),

            // keyboard
            WindowEvent::KeyboardInput { event, .. } => {
                self.process_keyboard_input(event, captured_by_gui)
            }

            // window resize
            WindowEvent::Resized(new_inner_size) => {
                self.update_window_inner_size(new_inner_size)?;
            }

            // dpi change
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.set_scale_factor(scale_factor)?;
            }

            //WindowEvent::ThemeChanged(winit_theme)
            _ => (),
        }

        Ok(())
    }

    fn update_engine(&mut self) -> anyhow::Result<()> {
        // make sure the render thread is active to receive the upcoming messages
        let thread_send_res = self
            .render_thread_channels
            .set_render_thread_command(RenderThreadCommand::Run(self.render_options));
        check_channel_updater_result(thread_send_res)?;

        // process recieved events for cursor state
        let cursor_event = self.cursor.process_frame();
        if let Some(cursor_icon) = self.cursor.cursor_icon() {
            self.gui.set_cursor_icon(cursor_icon);
        }

        // process gui inputs and update layout
        let update_gui_res = self.gui.update_gui(
            &self.object_collection,
            &self.window,
            self.camera,
            self.selected_object_id,
            self.selected_primitive_op_id,
            self.render_options,
        );
        let commands_from_gui = anyhow_unwrap(update_gui_res, "update gui");
        self.pending_commands.extend(commands_from_gui.into_iter());

        // process commands from gui
        self.execute_engine_commands();

        // update camera
        self.camera.update_camera(
            &mut self.cursor,
            self.settings,
            self.keyboard_modifier_states,
            self.settings.camera_control_mappings,
            &self.object_collection,
        );
        let thread_send_res = self
            .render_thread_channels
            .update_camera(self.camera.clone());
        check_channel_updater_result(thread_send_res)?;

        // submit object buffer updates
        let objects_delta = self.object_collection.get_and_clear_objects_delta();
        if !objects_delta.is_empty() {
            let thread_send_res = self.render_thread_channels.update_objects(objects_delta);
            check_channel_sender_result(thread_send_res)?;
        }

        // submit gui texture updates
        let textures_delta = self.gui.get_and_clear_textures_delta();
        if !textures_delta.is_empty() {
            let thread_send_res = self
                .render_thread_channels
                .update_gui_textures(textures_delta);
            check_channel_sender_result(thread_send_res)?;
        }

        // submit gui primitive updates
        let gui_primitives = self.gui.mesh_primitives().clone();
        if !gui_primitives.is_empty() {
            let thread_send_res = self
                .render_thread_channels
                .set_gui_primitives(gui_primitives);
            check_channel_updater_result(thread_send_res)?;
        }

        // if render clicked, send request to find out which element on scene it is
        if let CursorEvent::ClickInPlace(MouseButton::Left) = cursor_event {
            self.submit_request_for_element_id_at_point()?;
        }

        // receive clicked element response
        self.receive_and_select_element_id_at_point();

        let _latest_render_frame_timestamp = self
            .render_thread_channels
            .get_latest_render_frame_timestamp();

        self.main_thread_frame_number += 1;

        Ok(())
    }

    fn process_keyboard_input(&mut self, key_event: KeyEvent, captured_by_gui: bool) {
        // update modifiers whenever focus is in window
        self.keyboard_modifier_states.set(key_event.clone());

        // todo clean up the ordering of this... move keyboard_modifiers up? think it through...
        if captured_by_gui {
            return;
        }

        let PhysicalKey::Code(key_code) = key_event.physical_key else {
            return;
        };

        match key_code {
            KeyCode::KeyP => {
                if let ElementState::Released = key_event.state {
                    self.gui.set_command_palette_visability(true);
                }
            }
            KeyCode::Escape => {
                if let ElementState::Released = key_event.state {
                    self.gui.set_command_palette_visability(false);
                }
            }
            _ => (),
        }
    }

    fn update_window_inner_size(
        &mut self,
        new_inner_size: winit::dpi::PhysicalSize<u32>,
    ) -> Result<(), EngineError> {
        self.camera.set_aspect_ratio(new_inner_size.into());
        let thread_send_res = self.render_thread_channels.set_window_just_resized_flag();

        check_channel_updater_result(thread_send_res)
    }

    fn set_scale_factor(&mut self, scale_factor: f64) -> Result<(), EngineError> {
        self.scale_factor = scale_factor;
        self.gui.set_scale_factor(self.scale_factor as f32);
        let thread_send_res = self
            .render_thread_channels
            .set_scale_factor(scale_factor as f32);

        check_channel_updater_result(thread_send_res)
    }

    fn submit_request_for_element_id_at_point(&mut self) -> Result<(), EngineError> {
        if let Some(cursor_screen_coordinates_dvec2) = self.cursor.position() {
            let cursor_screen_coordinates = cursor_screen_coordinates_dvec2.as_vec2().to_array();

            // send request
            let thread_send_res = self
                .render_thread_channels
                .request_element_id_at_screen_coordinate(cursor_screen_coordinates);
            check_channel_updater_result(thread_send_res)?;
        }
        Ok(())
    }

    fn receive_and_select_element_id_at_point(&mut self) {
        if let Some(element_at_point) = self
            .render_thread_channels
            .receive_element_id_at_screen_coordinate()
        {
            debug!("element clicked = {:?}", element_at_point);
            match element_at_point {
                ElementAtPoint::Background => self.background_clicked(),
                ElementAtPoint::Object {
                    object_id,
                    primitive_op_index,
                } => self.object_clicked(object_id, Some(primitive_op_index)),
                ElementAtPoint::BlendArea { object_id } => self.object_clicked(object_id, None),
            }
        }
    }

    fn background_clicked(&mut self) {
        self.deselect_primitive_op();
        self.camera.unset_lock_on_target();
    }

    fn object_clicked(&mut self, object_id: ObjectId, primitive_op_index: Option<usize>) {
        if let Some(some_primitive_op_index) = primitive_op_index {
            let target_primitive_op = TargetPrimitiveOp::Index(object_id, some_primitive_op_index);
            self.select_primitive_op_and_object(target_primitive_op, None)
        } else {
            self.select_object(object_id, None);
        }
    }

    fn is_object_id_selected(&self, compare_object_id: ObjectId) -> bool {
        if let Some(some_selected_object_id) = self.selected_object_id {
            some_selected_object_id == compare_object_id
        } else {
            false
        }
    }

    fn shut_down(&mut self) {
        self.request_render_thread_quit();
        self.wait_for_render_thread_quit();
    }

    fn request_render_thread_quit(&self) {
        debug!("sending quit command to render thread...");
        let _render_thread_send_res = self
            .render_thread_channels
            .set_render_thread_command(RenderThreadCommand::Quit);
    }

    fn wait_for_render_thread_quit(&mut self) {
        let Some(render_thread_handle) = self.render_thread_handle.take() else {
            debug!("no render thread handle");
            return;
        };
        debug!(
            "waiting for render thread to quit (timeout = {:.2}s)",
            config_engine::JOIN_THREAD_WAIT_TIMEOUT_SECONDS
        );
        let join_timeout_begin = Instant::now();
        let timeout_millis = (config_engine::JOIN_THREAD_WAIT_TIMEOUT_SECONDS * 1_000.) as u128;
        loop {
            let render_thread_finished = render_thread_handle.is_finished();
            if render_thread_finished {
                break;
            }
            if join_timeout_begin.elapsed().as_millis() > timeout_millis {
                error!(
                    "render thread hanging longer than timeout of {:.2}s. continuing now...",
                    config_engine::JOIN_THREAD_WAIT_TIMEOUT_SECONDS
                );
                return;
            }
        }
        match render_thread_handle.join() {
            Err(e) => error!("thread joined -> render thread paniced: {:?}", e),
            Ok(()) => (),
        }
        debug!("render thread quit.");
    }
}

// ~~ Engine Error ~~

#[derive(Debug)]
pub enum EngineError {
    RenderThreadClosedPrematurely,
    WindowThreadClosedPrematurely,
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::RenderThreadClosedPrematurely => {
                write!(f, "render thread was closed prematurely")
            }
            Self::WindowThreadClosedPrematurely => {
                write!(f, "window thread was closed prematurely")
            }
        }
    }
}

impl std::error::Error for EngineError {}

/// If `thread_send_res` is an error, returns `EngineError::RenderThreadClosedPrematurely`.
/// Otherwise returns `Ok`.
fn check_channel_updater_result<T>(
    thread_send_res: Result<(), NoReceiverError<T>>,
) -> Result<(), EngineError> {
    if let Err(e) = thread_send_res {
        warn!("render thread receiver dropped prematurely ({})", e);
        return Err(EngineError::RenderThreadClosedPrematurely);
    }
    Ok(())
}

/// If `thread_send_res` is an error, returns `EngineError::RenderThreadClosedPrematurely`.
/// Otherwise returns `Ok`.
fn check_channel_sender_result<T>(
    thread_send_res: Result<(), SendError<T>>,
) -> Result<(), EngineError> {
    if let Err(e) = thread_send_res {
        warn!("render thread receiver dropped prematurely ({})", e);
        return Err(EngineError::RenderThreadClosedPrematurely);
    }
    Ok(())
}

// ~~ Testing ~~

fn create_default_cube_object(object_collection: &mut ObjectCollection) {
    let mut object = Object::new(String::from("Cube"), Vec3::ZERO);
    let cube = Cube::new(Vec3::splat(1.));
    _ = object.push_primitive_op(
        cube.into(),
        PrimitiveTransform::default(),
        Operation::Union,
        0.1,
        Vec3::new(0.8, 0.3, 0.1),
        0.5,
    );
    _ = object_collection
        .push_object(object)
        .expect("no where near maxing out unique ids");
}

fn object_testing(object_collection: &mut ObjectCollection) {
    use config_engine::DEFAULT_ALBEDO;
    use glam::Quat;

    let sphere = Sphere::new(0.5);
    let cube = Cube::new(Vec3::splat(0.8));
    let another_sphere = Sphere::new(0.83);

    let mut object = Object::new(String::from("Bruh"), Vec3::new(-0.2, 0.2, 0.));
    _ = object.push_primitive_op(
        Primitive::Cube(cube),
        PrimitiveTransform::new(Vec3::new(-0.2, 0.2, 0.), Quat::IDENTITY),
        Operation::Union,
        0.1,
        Vec3::new(0.1, 0.6, 0.7),
        0.5,
    );
    _ = object.push_primitive_op(
        Primitive::Sphere(sphere.clone()),
        PrimitiveTransform::new(Vec3::new(0., 0., 0.), Quat::IDENTITY),
        Operation::Union,
        0.1,
        Vec3::new(0.7, 0.2, 0.6),
        0.5,
    );
    _ = object.push_primitive_op(
        Primitive::Sphere(another_sphere),
        PrimitiveTransform::new(Vec3::new(0.2, -0.2, 0.), Quat::IDENTITY),
        Operation::Intersection,
        0.1,
        Vec3::new(0.8, 0.5, 0.1),
        0.5,
    );
    _ = object_collection
        .push_object(object)
        .expect("no where near maxing out unique ids");

    let mut another_object = Object::new(String::from("Another Bruh"), Vec3::new(0.2, -0.2, 0.));
    _ = another_object.push_primitive_op(
        Primitive::Sphere(sphere),
        PrimitiveTransform::DEFAULT,
        Operation::Union,
        0.1,
        DEFAULT_ALBEDO,
        0.5,
    );
    _ = object_collection.push_object(another_object);
}
