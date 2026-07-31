//! A node editor for `shadergraph`, with the material lit live on a mesh above
//! the canvas.
//!
//! # Why this is a separate binary
//!
//! Not for tidiness. The node library is deliberately permissively licensed and
//! kept free of engine assumptions so it can be reused; an editor that only
//! existed as a debug mode inside one application would drag that application's
//! dependencies — and its licence — into everything it touched. Shipping it as
//! its own binary in the same workspace keeps the compiler reusable and the
//! tool usable by anyone who vendors the crate.
//!
//! It renders through `shadergraph-bevy`, the same material a consuming
//! application uses, for the reason any preview exists: if the preview draws
//! through its own code path, it is a preview of the preview.

mod canvas;
mod compile;
mod preview;
mod ramp_widget;
mod ui;
mod zoom;

use std::time::Duration;

use bevy::asset::io::{AssetSource, AssetSourceBuilder};
use bevy::prelude::*;
use bevy::winit::{UpdateMode, WinitSettings};
use bevy_egui::{EguiPlugin, EguiPrimaryContextPass};

use canvas::GraphDocument;
use compile::{CompilePlugin, CompileState};
use preview::{PreviewPlugin, PreviewSettings, PreviewSubject};
use shadergraph_bevy::GraphMaterialPlugin;
use ui::{draw_ui, UiState};

fn main() {
    let mut app = App::new();

    // An asset source rooted at the filesystem root, so a glTF dragged in from
    // anywhere can be loaded as `abs://path/to/file.glb`. The default source is
    // rooted at `assets/`, which a dropped file will almost never be under.
    //
    // NOTE: Must be registered before `DefaultPlugins`: `AssetPlugin` reads the
    // registered sources when it builds, and one added afterwards is silently
    // never consulted.
    app.register_asset_source(
        "abs",
        AssetSourceBuilder::new(AssetSource::get_default_reader("/".to_string())),
    );

    // Uncapped presentation for shader benchmarking: with vsync on, every
    // frame-time reading is the refresh interval and says nothing about the
    // shader. Opt-in by env var because uncapped is the wrong default for a
    // tool — during a light sweep it would spin the GPU flat out to draw
    // frames nobody can see.
    let present_mode = if std::env::var_os("SHADERGRAPH_UNCAPPED").is_some() {
        bevy::window::PresentMode::AutoNoVsync
    } else {
        bevy::window::PresentMode::AutoVsync
    };

    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "shadergraph editor".to_string(),
            resolution: (1600, 1000).into(),
            present_mode,
            ..default()
        }),
        ..default()
    }))
    // Feeds the frame-time readout in the toolbar. Cheap enough to run
    // unconditionally: it averages a handful of timestamps per frame.
    .add_plugins(bevy::diagnostic::FrameTimeDiagnosticsPlugin::default())
    // Multipass is unconditional in current bevy_egui; the plugin no longer
    // has a mode to choose. The two-contexts-sharing-a-schedule panic does
    // not apply: this app has exactly one context.
    .add_plugins(EguiPlugin::default())
    .add_plugins((
        GraphMaterialPlugin,
        PreviewPlugin,
        CompilePlugin,
        zoom::ScrollZoomPlugin,
    ))
    .insert_resource(WinitSettings::desktop_app())
    .init_resource::<GraphDocument>()
    .init_resource::<UiState>()
    .add_systems(Startup, seed_document)
    // `PostUpdate`, not `Update`. This decides whether the *next* frame is
    // needed, so it has to see the finished state of this one — a drop
    // handled in `Update` starts an asset load, and a scheduler free to run
    // this first would park the loop a fraction of a millisecond before the
    // thing it needed to wait for existed. Running last makes it independent
    // of every ordering decision in `Update` rather than coupled to them.
    .add_systems(PostUpdate, keep_rendering_while_it_matters)
    .add_systems(EguiPrimaryContextPass, draw_ui)
    .run();
}

/// How long the loop may sleep when nothing is happening.
///
/// Reached only when there is genuinely nothing to draw: any input, and any
/// `RequestRedraw` — which bevy_egui raises on egui's behalf, explicitly to
/// support reactive winit modes like this one — wakes it at once.
/// So hover fades and menu animations stay smooth; this is the interval between
/// frames drawn for *no reason at all*.
const IDLE: UpdateMode = UpdateMode::Reactive {
    wait: Duration::from_secs(5),
    react_to_device_events: true,
    react_to_user_events: true,
    react_to_window_events: true,
};

/// Frames to keep drawing after a structural recompile.
///
/// A new shader is not visible the moment the asset is written: `PipelineCache`
/// builds the pipeline across subsequent frames. Park the loop immediately and
/// the compile never finishes, so the mesh keeps the old material until the next
/// stray mouse move — which looks exactly like the editor having missed the
/// edit. Eight is comfortably more than the rebuild needs and costs an eighth of
/// a second of rendering after an edit that was, by definition, deliberate.
const PIPELINE_SETTLE_FRAMES: u32 = 8;

/// Run the loop continuously only while something is actually moving.
///
/// # Why this exists
///
/// Bevy defaults to [`UpdateMode::Continuous`], which is right for a game and
/// wrong for a tool: the editor spends most of its life displaying a motionless
/// cube, and doing that at the refresh rate means running the full schedule —
/// render graph, shadow pass, and a complete graph recompile — over and over to
/// produce identical pixels.
///
/// Gating the *loop* rather than the individual systems is deliberate. It fixes
/// the per-frame recompile as a side effect, without a dirty flag anywhere: the
/// compile still runs unconditionally on every frame that happens, and frames
/// now only happen when something has occurred. `compile`'s "it cannot be wrong
/// because nothing has to remember to mark it dirty" property is preserved
/// exactly.
fn keep_rendering_while_it_matters(
    preview: Res<PreviewSettings>,
    compile: Res<CompileState>,
    assets: Res<AssetServer>,
    subject: Query<&Mesh3d, With<PreviewSubject>>,
    mut winit: ResMut<WinitSettings>,
    mut last_compile: Local<u64>,
    mut settling: Local<u32>,
) {
    if compile.compiles != *last_compile {
        *last_compile = compile.compiles;
        *settling = PIPELINE_SETTLE_FRAMES;
    }

    // NOTE: Asset loading needs frames too, and a dropped glTF is the case that
    // proves it. The drop itself is a window event, so it wakes the loop for one
    // update — but the file is then read, parsed and uploaded across many more,
    // and if the loop parks after that first frame none of them ever happen. The
    // mesh stays unloaded, `frame_subject_on_load` never fires, and the viewport
    // sits empty looking exactly like a failed import.
    //
    // NOTE: Specifically `is_loading`, and not "absent from `Assets<Mesh>`". A file
    // that fails to import — wrong format, or a glTF with no first primitive —
    // never arrives, so the weaker test would spin the loop at full rate forever
    // and quietly undo the whole reason this system exists. `Failed` and
    // `NotLoaded` both end the wait; a broken drop should cost a log line, not a
    // core.
    let waiting_on_mesh = subject
        .single()
        .map(|mesh| assets.load_state(&mesh.0).is_loading())
        .unwrap_or(false);

    // A sweeping light is the one thing here that animates without any input at
    // all, so it is the one thing that genuinely needs a continuous loop.
    let animating = preview.orbit_light || *settling > 0 || waiting_on_mesh;
    *settling = settling.saturating_sub(1);

    let wanted = if animating {
        UpdateMode::Continuous
    } else {
        IDLE
    };
    // Compared before assigning so the resource is not marked changed on every
    // frame. Nothing watches it today; leaving a change flag permanently set is
    // how the next thing that does watch it gets a surprise.
    if winit.focused_mode != wanted {
        winit.focused_mode = wanted;
    }
}

/// Open on a real material rather than an empty canvas.
///
/// An empty graph is not a neutral starting point — it has no output node, so
/// it does not compile, so the first thing the editor would show is an error
/// about a node the author has not been told they need. Loading a preset means
/// the tool opens working, and taking it apart is a far better way to learn the
/// node set than assembling one from a blank page.
fn seed_document(mut document: ResMut<GraphDocument>) {
    match shadergraph::materials::hull_metal() {
        Ok(graph) => document.load(&graph),
        Err(e) => error!("could not build the starting graph `hull_metal`: {e}"),
    }
}
