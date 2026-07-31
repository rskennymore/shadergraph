//! Window layout: controls on top, lit preview in the middle, node canvas below.
//!
//! The 3D view is not drawn by egui — it is the *hole left over* once the
//! panels have taken what they want. The camera's viewport is fitted to
//! `available_rect` each frame, which is what makes the splitter between the
//! preview and the canvas work: drag it and the render target follows.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use egui_snarl::ui::SnarlWidget;
use shadergraph::materials;

use crate::canvas::{latch_menu_anchor, GraphDocument, NodeViewer};
use crate::compile::{CompileState, CompileStatus};
use crate::preview::{
    viewport_from_physical, PreviewMesh, PreviewMeshAttributes, PreviewSettings, PreviewViewport,
};
use crate::zoom::CanvasRect;

/// Width of a light control, in points.
///
/// These sit in a horizontal strip and a `DragValue` sizes itself to whatever it
/// currently reads. That is fine for a value you type into and wrong for one that
/// changes on its own: while the light sweeps, the yaw field grows and shrinks by
/// a character at a time and shoves everything to its right along with it. The
/// toolbar wobbling is a worse problem than a field being wider than it needs.
const LIGHT_FIELD_WIDTH: f32 = 104.0;

/// Add a light control at a fixed size, so a changing value cannot move the
/// toolbar around it.
fn light_field(ui: &mut egui::Ui, drag: egui::DragValue<'_>) -> egui::Response {
    ui.add_sized([LIGHT_FIELD_WIDTH, ui.spacing().interact_size.y], drag)
}

/// What a toolbar preset entry runs to get its graph.
type PresetBuilder = fn() -> Result<shadergraph::Graph, shadergraph::GraphError>;

/// Presets offered in the toolbar.
///
/// Named `fn` pointers rather than pre-built graphs so a failure to build shows
/// up when you ask for it, in this window, rather than at startup in a log.
const PRESETS: &[(&str, PresetBuilder)] = &[
    ("hull_metal", materials::hull_metal),
    ("hull_dark", materials::hull_dark),
    ("copper_accent", materials::copper_accent),
    ("scorched_plating", materials::scorched_plating),
    ("panel_wear", materials::panel_wear),
    ("worn_paint", materials::worn_paint),
    ("glass", || materials::glass([0.1, 0.12, 0.2, 1.0], 0.3)),
];

#[derive(Resource)]
pub struct UiState {
    pub show_shader: bool,
    /// Where "Save WGSL" writes. A plain text field rather than a file dialog,
    /// which would mean a dependency for one button.
    pub export_path: String,
    /// Outcome of the last export, shown next to the button. Errors name the
    /// path, because "failed to save" without one is a message you cannot act on.
    pub export_result: Option<Result<String, String>>,
    /// Where the graph itself is saved and loaded from.
    pub graph_path: String,
    pub graph_result: Option<Result<String, String>>,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            show_shader: false,
            export_path: "shader.wgsl".to_string(),
            export_result: None,
            graph_path: "graph.json".to_string(),
            graph_result: None,
        }
    }
}

/// One vertex-attribute indicator.
///
/// Always shown, so the mesh can be judged before anything is wired to it, but
/// only loud when it matters: a graph that reads an attribute the mesh has not
/// got is the one case that is otherwise completely silent, and it is the one
/// case that turns red.
fn attribute_chip(ui: &mut egui::Ui, label: &str, needed: bool, present: Option<bool>) {
    let (text, color, hint) = match (needed, present) {
        (_, None) => (
            format!("{label} …"),
            egui::Color32::GRAY,
            "the mesh has not finished loading".to_string(),
        ),
        (true, Some(true)) => (
            format!("{label} ✓"),
            egui::Color32::from_rgb(120, 200, 120),
            format!("the graph reads {label} and this mesh has it"),
        ),
        (true, Some(false)) => (
            format!("{label} ✗"),
            egui::Color32::from_rgb(230, 90, 90),
            format!(
                "the graph reads {label} but this mesh has none — it will read \
                 zero everywhere, which looks like a broken material rather than \
                 a missing channel"
            ),
        ),
        (false, Some(has)) => (
            format!("{label} {}", if has { "✓" } else { "✗" }),
            egui::Color32::DARK_GRAY,
            format!("this mesh {} {label}; the graph does not read it", {
                if has {
                    "has"
                } else {
                    "has no"
                }
            }),
        ),
    };
    ui.colored_label(color, text).on_hover_text(hint);
}

/// Report the outcome of a file operation, in the color that says which it was.
fn show_result(ui: &mut egui::Ui, result: &Option<Result<String, String>>) {
    match result {
        Some(Ok(message)) => {
            ui.colored_label(egui::Color32::from_rgb(120, 200, 120), message);
        }
        Some(Err(message)) => {
            ui.colored_label(egui::Color32::from_rgb(230, 90, 90), message);
        }
        None => {}
    }
}

// A Bevy system takes one parameter per resource it touches; that is the
// system signature working as designed, not a function grown out of hand.
#[allow(clippy::too_many_arguments)]
pub fn draw_ui(
    mut contexts: EguiContexts,
    mut ui_state: ResMut<UiState>,
    mut settings: ResMut<PreviewSettings>,
    mut document: ResMut<GraphDocument>,
    mut viewport: ResMut<PreviewViewport>,
    mut canvas_rect: ResMut<CanvasRect>,
    compile: Res<CompileState>,
    attributes: Res<PreviewMeshAttributes>,
    windows: Query<&Window>,
    diagnostics: Res<bevy::diagnostic::DiagnosticsStore>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let (scale_factor, surface) = (
        window.scale_factor(),
        UVec2::new(window.physical_width(), window.physical_height()),
    );

    // `ctx_mut` returns `Err` only when no primary window context exists —
    // during shutdown, or before the first frame — and skipping the UI for
    // those frames is exactly right.
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    let ctx = ctx.clone();

    // Panels draw into a `Ui` rather than onto the `Context`, so the frame
    // starts by making the root they carve up: a background-layer `Ui` covering
    // the whole viewport. Each panel advances this `Ui`'s cursor, which is what
    // makes the leftover rect below mean "whatever the panels did not take".
    let mut root = egui::Ui::new(
        ctx.clone(),
        "root".into(),
        egui::UiBuilder::new()
            .layer_id(egui::LayerId::background())
            .max_rect(ctx.viewport_rect()),
    );

    egui::Panel::top("toolbar").show(&mut root, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.menu_button("Preset", |ui| {
                for (name, build) in PRESETS {
                    if ui.button(*name).clicked() {
                        match build() {
                            Ok(graph) => document.load(&graph),
                            Err(e) => error!("preset `{name}` failed to build: {e}"),
                        }
                        ui.close();
                    }
                }
            });

            ui.separator();

            egui::ComboBox::from_id_salt("mesh")
                .selected_text(settings.mesh.label())
                .show_ui(ui, |ui| {
                    for &mesh in PreviewMesh::ALL {
                        ui.selectable_value(&mut settings.mesh, mesh, mesh.label());
                    }
                });

            attribute_chip(ui, "UV", compile.uses_uv, attributes.uv);
            attribute_chip(
                ui,
                "Color",
                compile.uses_vertex_color,
                attributes.vertex_color,
            );

            // Unresolved texture slots render fallback white, so the mesh
            // looks *plausible* rather than broken — which is exactly why the
            // failure needs a chip. One chip however many slots are unhappy;
            // the hover text carries the details.
            if !compile.texture_warnings.is_empty() {
                ui.colored_label(
                    egui::Color32::from_rgb(230, 90, 90),
                    format!("Tex ✗{}", compile.texture_warnings.len()),
                )
                .on_hover_text(compile.texture_warnings.join("\n"));
            }

            ui.separator();

            ui.label("Light");
            // Fixed decimals as well as a fixed box. The width stops the *field*
            // resizing; the decimal count stops the number inside it flickering
            // between two and five digits, which reads as a glitch even when
            // nothing moves.
            light_field(
                ui,
                egui::DragValue::new(&mut settings.light_yaw)
                    .speed(0.01)
                    .fixed_decimals(2)
                    .prefix("yaw "),
            );
            light_field(
                ui,
                egui::DragValue::new(&mut settings.light_pitch)
                    .speed(0.01)
                    .fixed_decimals(2)
                    .prefix("pitch "),
            );
            ui.checkbox(&mut settings.orbit_light, "sweep");
            light_field(
                ui,
                egui::DragValue::new(&mut settings.illuminance)
                    .speed(50.0)
                    .range(0.0..=100_000.0)
                    .fixed_decimals(0)
                    .prefix("lux "),
            );
            light_field(
                ui,
                egui::DragValue::new(&mut settings.ambient)
                    .speed(5.0)
                    .range(0.0..=5_000.0)
                    .fixed_decimals(0)
                    .prefix("ambient "),
            );

            ui.separator();

            // EXPERIMENT: see `PreviewSettings::fine_derivatives`.
            ui.checkbox(&mut settings.fine_derivatives, "fine dpdx")
                .on_hover_text(
                    "Rewrite dpdx/dpdy to dpdxFine/dpdyFine in the compiled \
                     shader. Bump normals stop being constant across each \
                     2x2 pixel quad, at whatever cost this readout shows.",
                );

            // Meaningful only while the loop runs continuously — turn `sweep`
            // on — and honest only with vsync off (SHADERGRAPH_UNCAPPED=1);
            // capped, every reading is the refresh interval.
            if let Some(ms) = diagnostics
                .get(&bevy::diagnostic::FrameTimeDiagnosticsPlugin::FRAME_TIME)
                .and_then(|d| d.smoothed())
            {
                let fps = diagnostics
                    .get(&bevy::diagnostic::FrameTimeDiagnosticsPlugin::FPS)
                    .and_then(|d| d.smoothed())
                    .unwrap_or_default();
                ui.monospace(format!("{ms:6.3} ms · {fps:5.0} fps"))
                    .on_hover_text(
                        "Smoothed frame time. Turn `sweep` on for continuous \
                         rendering; launch with SHADERGRAPH_UNCAPPED=1 to lift \
                         vsync, or this only ever shows the refresh interval.",
                    );
            }

            ui.separator();

            ui.add(
                egui::TextEdit::singleline(&mut ui_state.graph_path)
                    .desired_width(140.0)
                    .hint_text("graph file"),
            );
            if ui
                .button("Save")
                .on_hover_text("Save the graph, layout included")
                .clicked()
            {
                let path = ui_state.graph_path.clone();
                ui_state.graph_result = Some(
                    document
                        .save(&path)
                        .map(|()| format!("saved {path}"))
                        .map_err(|e| e.to_string()),
                );
            }
            if ui.button("Load").clicked() {
                let path = ui_state.graph_path.clone();
                ui_state.graph_result = Some(
                    document
                        .load_from(&path)
                        .map(|()| format!("loaded {path}"))
                        .map_err(|e| e.to_string()),
                );
            }

            ui.separator();
            ui.toggle_value(&mut ui_state.show_shader, "Shader");

            // Status sits at the far right so it is in the same place whatever
            // the window width, and reads as a light rather than a message.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                match &compile.status {
                    CompileStatus::Ok => ui.colored_label(
                        egui::Color32::from_rgb(120, 200, 120),
                        // The live-write count is the visible proof that a value
                        // edit skipped the compiler entirely.
                        format!("compiled · {} live param writes", compile.live_param_writes),
                    ),
                    CompileStatus::Failed(_) => ui.colored_label(
                        egui::Color32::from_rgb(230, 90, 90),
                        "not compiled — see Shader",
                    ),
                    CompileStatus::Pending => ui.weak("waiting for a graph"),
                }
            });
        });

        show_result(ui, &ui_state.graph_result);
    });

    let canvas_panel = egui::Panel::bottom("canvas")
        .resizable(true)
        .default_size(surface.y as f32 / scale_factor * 0.5)
        .show(&mut root, |ui| {
            // Before the widget is shown, because it records the right-click
            // that *opens* the node menu and `show` is where that menu is drawn.
            latch_menu_anchor(ui);

            let GraphDocument { snarl, style } = &mut *document;
            SnarlWidget::new()
                .id_salt("shadergraph")
                .style(*style)
                .show(snarl, &mut NodeViewer::default(), ui);
        });

    // Handed to the scroll-zoom system, which runs before egui reads its input
    // next frame and so cannot ask for this itself.
    canvas_rect.0 = Some(canvas_panel.response.rect);

    if ui_state.show_shader {
        let mut open = true;
        egui::Window::new("Shader")
            .open(&mut open)
            .default_size([720.0, 520.0])
            .show(&ctx, |ui| {
                if let Some(message) = compile.error() {
                    ui.colored_label(egui::Color32::from_rgb(230, 90, 90), "Compile error");
                    egui::ScrollArea::vertical()
                        .max_height(220.0)
                        .id_salt("error")
                        .show(ui, |ui| {
                            ui.monospace(message);
                        });
                    ui.separator();
                    ui.weak("The mesh is still showing the last shader that compiled.");
                    ui.separator();
                }

                ui.horizontal(|ui| {
                    ui.label("Generated WGSL");
                    ui.add(
                        egui::TextEdit::singleline(&mut ui_state.export_path)
                            .desired_width(240.0)
                            .hint_text("path to write"),
                    );

                    // Exporting an empty or stale shader would write a file that
                    // silently disagrees with the canvas, so the button is only
                    // live when what is on screen is what compiled.
                    let exportable =
                        matches!(compile.status, CompileStatus::Ok) && !compile.wgsl.is_empty();
                    if ui
                        .add_enabled(exportable, egui::Button::new("Save WGSL"))
                        .clicked()
                    {
                        let path = ui_state.export_path.clone();
                        ui_state.export_result = Some(
                            std::fs::write(&path, &compile.wgsl)
                                .map(|()| format!("wrote {path}"))
                                .map_err(|e| format!("could not write {path}: {e}")),
                        );
                    }
                });

                show_result(ui, &ui_state.export_result);

                egui::ScrollArea::vertical().id_salt("wgsl").show(ui, |ui| {
                    ui.monospace(&compile.wgsl);
                });
            });
        ui_state.show_shader = open;
    }

    // Whatever is left after the panels is the 3D view. Read *after* every
    // panel has been added, or the preview is sized to a layout that no longer
    // exists. Each panel shrank the root's cursor as it took its edge, so this
    // is the hole they left.
    let rect = root.available_rect_before_wrap();
    let fitted = viewport_from_physical(
        Vec2::new(rect.min.x, rect.min.y) * scale_factor,
        Vec2::new(rect.max.x, rect.max.y) * scale_factor,
        surface,
    );

    // Only write on a real change. `PreviewViewport` is watched by change
    // detection, and assigning every frame — twice per frame, with multipass —
    // would make that watch meaningless.
    let unchanged = match (&viewport.0, &fitted) {
        (Some(a), Some(b)) => {
            a.physical_position == b.physical_position && a.physical_size == b.physical_size
        }
        (None, None) => true,
        _ => false,
    };
    if !unchanged {
        viewport.0 = fitted;
    }
}
