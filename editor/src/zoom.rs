//! Plain scroll zooms the node canvas, without holding a modifier.
//!
//! # Why this is not a setting
//!
//! `SnarlStyle` exposes `min_scale` and `max_scale` but nothing about which
//! gesture zooms. The canvas (egui-snarl, via egui's `Scene` pan-and-zoom)
//! reads `InputState::zoom_delta()`, and egui derives that from `Event::Zoom`
//! and from **ctrl+scroll** — plain scroll goes to the scroll delta, which the
//! canvas spends on panning instead. `InputState::zoom_factor_delta` is
//! private, so it cannot simply be assigned.
//!
//! What *is* reachable is the raw input before egui reads it. `EguiInput` is a
//! public component holding `egui::RawInput`, and bevy_egui documents the
//! supported hook for mutating it: run after `EguiPreUpdateSet::ProcessInput`
//! and before `EguiPreUpdateSet::BeginPass`.
//!
//! So rather than reimplementing zoom, this claims the gesture: a scroll event
//! over the canvas is rewritten as though ctrl were held, and egui's own
//! scroll-to-zoom path handles it — smoothing, zoom speed and all. Nothing here
//! knows what a zoom level is.
//!
//! The cost is that the canvas can no longer be scrolled vertically by wheel,
//! which is the trade being asked for: on an infinite pannable canvas, dragging
//! the background is the natural pan and the wheel is better spent on zoom.

use bevy::prelude::*;
use bevy_egui::{egui, EguiInput, EguiPreUpdateSet};

/// The node canvas' rectangle, in egui points.
///
/// Written by the UI each frame, read here on the *next* one. A frame of lag on
/// a hit test against a panel the user is already pointing at is not detectable;
/// the alternative is duplicating egui's layout to predict the rect.
#[derive(Resource, Default)]
pub struct CanvasRect(pub Option<egui::Rect>);

pub struct ScrollZoomPlugin;

impl Plugin for ScrollZoomPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CanvasRect>().add_systems(
            PreUpdate,
            scroll_zooms_the_canvas
                .after(EguiPreUpdateSet::ProcessInput)
                .before(EguiPreUpdateSet::BeginPass),
        );
    }
}

fn scroll_zooms_the_canvas(
    canvas: Res<CanvasRect>,
    // The pointer is tracked across frames rather than read out of this frame's
    // events. A wheel event usually arrives with no accompanying motion — the
    // hand is still while the finger scrolls — so requiring a `PointerMoved` in
    // the same frame would make zoom work only while also moving the mouse.
    mut last_pointer: Local<Option<egui::Pos2>>,
    mut contexts: Query<&mut EguiInput>,
) {
    for mut input in &mut contexts {
        // Taken from the event stream rather than from bevy's cursor because
        // these positions are already in egui points, so comparing them against
        // a rect in egui points needs no scale-factor conversion and cannot
        // disagree with egui about where the pointer is.
        for event in &input.0.events {
            match event {
                egui::Event::PointerMoved(position) => *last_pointer = Some(*position),
                egui::Event::PointerButton { pos, .. } => *last_pointer = Some(*pos),
                egui::Event::PointerGone => *last_pointer = None,
                _ => {}
            }
        }

        let (Some(rect), Some(position)) = (canvas.0, *last_pointer) else {
            continue;
        };
        if !rect.contains(position) {
            continue;
        }

        for event in &mut input.0.events {
            if let egui::Event::MouseWheel { modifiers, .. } = event {
                // `command` is the spelling egui checks on macOS; setting both
                // means the gesture behaves the same wherever this runs.
                modifiers.ctrl = true;
                modifiers.command = true;
            }
        }
    }
}
