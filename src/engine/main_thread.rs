use crate::config;
use anyhow::Context;
#[allow(unused_imports)]
use log::{debug, error, info, trace, warn};
use single_value_channel::Updater;
use std::{
    mem::ManuallyDrop,
    sync::{
        mpsc::{self, Sender, TryRecvError},
        Arc,
    },
    thread::{self, JoinHandle},
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes, WindowId},
};

use super::engine_controller::{EngineCommand, EngineController};

pub fn start_main_thread() -> anyhow::Result<()> {
    let event_loop = EventLoop::new().context("creating os event loop")?;

    let window = create_window(&event_loop)?;
    let primary_window_id = window.id();

    let (engine_command_rx, engine_command_tx) = single_value_channel::channel::<EngineCommand>();
    let (window_event_tx, window_event_rx) = mpsc::channel::<WindowEvent>();

    let main_thread_channels = MainThreadChannels {
        engine_command_rx,
        window_event_rx,
    };

    let _ = engine_command_tx.update(Some(EngineCommand::Run));
    let engine_thread_handle = thread::spawn(|| {
        info!("initializing engine instance");
        let mut engine_controller = EngineController::new(window, main_thread_channels)?;

        info!("starting engine loop");
        engine_controller.run()?;

        Ok::<(), anyhow::Error>(())
    });

    let mut main_thread = MainThread {
        primary_window_id,
        engine_thread_handle: ManuallyDrop::new(engine_thread_handle),
        engine_command_tx,
        window_event_tx,
    };

    event_loop.run_app(&mut main_thread)?;

    Ok(())
}

pub struct MainThread {
    primary_window_id: WindowId,
    engine_thread_handle: ManuallyDrop<JoinHandle<Result<(), anyhow::Error>>>,
    engine_command_tx: Updater<Option<EngineCommand>>,
    window_event_tx: Sender<WindowEvent>,
}

impl ApplicationHandler for MainThread {
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        if event == WindowEvent::CloseRequested && window_id == self.primary_window_id {
            let _ = self.engine_command_tx.update(Some(EngineCommand::Quit));
            info!("close requested by window. stopping main thread...");
            event_loop.exit();

            return;
        }

        // send os event to engine thread
        let send_res = self.window_event_tx.send(event.clone());

        // handle premature engine closure
        if let Err(_e) = send_res {
            info!("engine thread disconnected. stopping main thread...");
            event_loop.exit();
        }
    }

    fn resumed(&mut self, _event_loop: &ActiveEventLoop) {}

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Safety: winit guarentees event loop shutdown after returning so this should be
        // functionally the same as Drop
        let engine_thread_handle =
            unsafe { ManuallyDrop::<_>::take(&mut self.engine_thread_handle) };
        wait_for_engine_thread_closure(engine_thread_handle);
    }
}

fn wait_for_engine_thread_closure(
    engine_thread_handle: thread::JoinHandle<Result<(), anyhow::Error>>,
) {
    // check reason for engine thread closure
    let engine_thread_join_res = engine_thread_handle.join();
    if let Err(engine_panic_param) = &engine_thread_join_res {
        error!("panic on engine thread! panic params:");
        error!("{:?}", engine_panic_param);
    }
    if let Ok(engine_thread_res) = engine_thread_join_res {
        match engine_thread_res {
            Ok(()) => info!("engine thread shut down cleanly."),
            Err(engine_thread_err) => error!(
                "engine thread shut down due to error: {}",
                engine_thread_err
            ),
        }
    }
}

fn create_window(event_loop: &EventLoop<()>) -> anyhow::Result<Arc<Window>> {
    info!("creating main window...");
    let window_attributes = WindowAttributes::default().with_title(config::ENGINE_NAME);
    let window = event_loop
        .create_window(window_attributes)
        .context("instanciating initial os window")?;
    Ok(Arc::new(window))
}

pub struct MainThreadChannels {
    /// FIFO queue
    pub engine_command_rx: single_value_channel::Receiver<Option<EngineCommand>>,
    pub window_event_rx: mpsc::Receiver<WindowEvent>,
}

impl MainThreadChannels {
    /// Ordered by time received, i.e. first event in index 0
    pub fn get_events(&self) -> anyhow::Result<Vec<WindowEvent>> {
        let mut events = Vec::<WindowEvent>::new();
        loop {
            let recv_res = self.window_event_rx.try_recv();
            match recv_res {
                Ok(event) => events.push(event),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => anyhow::bail!("window thread disconnected"),
            };
        }
        Ok(events)
    }

    pub fn latest_command(&mut self) -> Option<EngineCommand> {
        *self.engine_command_rx.latest()
    }
}
