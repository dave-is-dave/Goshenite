mod config;
//mod immutable;
mod logger;
mod renderer;

use log::LevelFilter;
use logger::ConsoleLogger;
use renderer::render_manager::RenderManager;
use std::sync::Arc;
use winit::event_loop::EventLoop;
use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::ControlFlow,
    platform::run_return::EventLoopExtRunReturn,
    window::WindowBuilder,
};

const SPLASH: &str = "
     ___        ___        ___        ___        ___        ___        ___       ___        ___     
    /\\  \\      /\\  \\      /\\  \\      /\\__\\      /\\  \\      /\\__\\      /\\  \\     /\\  \\      /\\  \\    
   /  \\  \\    /  \\  \\    /  \\  \\    / /  /     /  \\  \\    / /  /      \\ \\  \\    \\ \\  \\    /  \\  \\   
  / /\\ \\  \\  / /\\ \\  \\  / /\\ \\  \\  / /__/     / /\\ \\  \\  / /  /        \\ \\  \\    \\ \\  \\  / /\\ \\  \\  
 / /  \\ \\  \\/ /  \\ \\  \\_\\ \\ \\ \\  \\/  \\  \\ ___/  \\ \\ \\  \\/ /__/_____ __ /  \\  \\   /  \\  \\/  \\ \\ \\  \\ 
/ /__/ \\ \\__\\/__/ \\ \\__\\ \\ \\ \\ \\__\\/\\ \\  /\\__\\/\\ \\ \\ \\__\\ _____ \\__\\  / /\\ \\__\\ / /\\ \\__\\/\\ \\ \\ \\__\\
\\ \\  /\\ \\/__/\\  \\ / /  /\\ \\ \\ \\/__/__\\ \\/ /  /\\ \\ \\ \\/__/__/  / /  /\\/ /  \\/__// /  \\/__/\\ \\ \\ \\/__/
 \\ \\ \\ \\__\\ \\ \\  / /  /\\ \\ \\ \\__\\     \\  /  /\\ \\ \\ \\__\\      / /  /\\  /__/    / /  /    \\ \\ \\ \\__\\  
  \\ \\/ /  /  \\ \\/ /  /  \\ \\/ /  /     / /  /  \\ \\ \\/__/     / /  /  \\ \\  \\    \\/__/      \\ \\ \\/__/  
   \\  /  /    \\  /  /    \\  /  /     / /  /    \\ \\__\\      / /  /    \\ \\__\\               \\ \\__\\    
    \\/__/      \\/__/      \\/__/      \\/__/      \\/__/      \\/__/      \\/__/                \\/__/    
";

fn main() {
    println!("{}", SPLASH);

    // init logger
    static CONSOLE_LOGGER: ConsoleLogger = ConsoleLogger;
    if let Err(e) = log::set_logger(&CONSOLE_LOGGER) {
        println!("Goshenite ERROR - Failed to initialize logger: {:?}", e);
    };
    log::set_max_level(LevelFilter::Info);

    // todo how default res usually handled?
    let init_resolution = [800, 800];

    // create winit window
    let mut event_loop = EventLoop::new();
    let window = Arc::new(
        WindowBuilder::new()
            .with_title(config::ENGINE_NAME)
            .with_inner_size(winit::dpi::LogicalSize::new(
                f64::from(init_resolution[0]),
                f64::from(init_resolution[1]),
            ))
            .build(&event_loop)
            .unwrap(),
    );

    {
        // init renderer
        let mut renderer = RenderManager::new(window);

        // start render loop
        let mut window_resize: bool = false;
        event_loop.run_return(|event, _, control_flow| {
            *control_flow = ControlFlow::Poll;
            match event {
                Event::WindowEvent {
                    event:
                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput {
                            input:
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::Escape),
                                    ..
                                },
                            ..
                        },
                    ..
                } => *control_flow = ControlFlow::Exit,
                Event::WindowEvent {
                    event: WindowEvent::Resized(_),
                    ..
                } => window_resize = true,
                Event::MainEventsCleared => renderer.render_frame(window_resize),
                Event::RedrawEventsCleared => window_resize = false,
                _ => (),
            }
        });

        // render cleanup on drop
    }
}
