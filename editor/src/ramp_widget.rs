//! A gradient bar with draggable stops.
//!
//! egui has no such widget, so this is drawn on a raw `Painter`.
//!
//! # color space, again
//!
//! Stops are stored linear (see `shadergraph::ramp`). Two consequences meet
//! here and pull in opposite directions:
//!
//! - `color_edit_button_rgba_unmultiplied` takes **linear** values, so the
//!   picker is wired to the stop directly with no conversion.
//! - `Color32` is **sRGB**, so painting a stop's color has to go through
//!   `Rgba` → `Color32`, which applies the transfer function.
//!
//! Getting the second one wrong produces a preview that is merely too dark
//! rather than obviously broken, and it would be the swatch lying about the
//! surface — the one thing this widget exists to avoid.

use bevy_egui::egui::{self, Color32, Rect, Rgba, Sense, Stroke, Ui, Vec2};
use shadergraph::{ColorRamp, ColorStop, RampInterp};

// # Sizes here are LOGICAL points.
//
// egui-snarl zooms the canvas with an egui layer transform, so everything the
// widget paints — including raw sizes handed to `allocate_exact_size` — is
// scaled with the node around it. Nothing here needs to know the zoom level.

/// Width of the gradient bar.
///
/// Fixed, not derived from `available_width`. This widget lives inside a node on
/// an infinite pannable canvas, where "available width" is a property of the
/// canvas rather than of the node — asking for it yields thousands of pixels and
/// a ramp too wide to use. A node body should size itself to its contents and
/// this is the content.
const BAR_WIDTH: f32 = 200.0;
/// Height of the gradient bar itself, before the handle strip.
const BAR_HEIGHT: f32 = 26.0;
/// Room below the bar for the stop handles.
const HANDLE_STRIP: f32 = 12.0;
const HANDLE_HALF_WIDTH: f32 = 5.0;
/// Width of the interpolation-mode picker.
const INTERP_COMBO_WIDTH: f32 = 120.0;
/// Corner radius of the bar outline, and the width of a stop handle's outline.
const BAR_CORNER_RADIUS: f32 = 2.0;
const HANDLE_STROKE: f32 = 1.0;
const SELECTED_HANDLE_STROKE: f32 = 2.0;
/// How much wider the selected and hovered handles are drawn.
///
/// Size, unlike color, cannot collide with the stop's own fill — which is why
/// selection leans on it rather than on a highlight tint.
const SELECTED_HANDLE_GROWTH: f32 = 1.5;
const HOVERED_HANDLE_GROWTH: f32 = 1.25;
/// How far the selected handle's halo extends past its outline.
const HANDLE_HALO: f32 = 2.0;

/// Draw the ramp editor. Returns whether the ramp changed.
///
/// The return value matters more than it looks: the caller uses it to decide
/// nothing at all, because the compile loop re-derives the graph every frame.
/// It is reported anyway so a future caller that *does* want to know — an undo
/// stack, say — does not have to re-plumb it.
pub fn ramp_editor(ui: &mut Ui, salt: &str, ramp: &mut ColorRamp) -> bool {
    let mut changed = false;

    // Selection is transient view state, not part of the document, so it lives
    // in egui's per-widget memory rather than on the node. A reloaded graph
    // should not remember which stop happened to be selected when it was saved.
    let selection_id = egui::Id::new(("ramp_selection", salt));
    let mut selected: usize = ui
        .data(|d| d.get_temp(selection_id))
        .unwrap_or(0)
        .min(ramp.stops.len().saturating_sub(1));

    // --- row 1: what the whole ramp does -----------------------------------
    // The mode governs every segment, so it sits alone above the bar rather than
    // among the per-stop controls below it. The counter keeps it company because
    // it describes the ramp too, not the selected stop.
    let stop_count = ramp.stops.len();
    ui.horizontal(|ui| {
        egui::ComboBox::from_id_salt(("ramp_interp", salt))
            .width(INTERP_COMBO_WIDTH)
            .selected_text(ramp.interp.label())
            .show_ui(ui, |ui| {
                for &mode in RampInterp::ALL {
                    if ui
                        .selectable_value(&mut ramp.interp, mode, mode.label())
                        .changed()
                    {
                        changed = true;
                    }
                }
            });
        ui.weak(format!("{stop_count} stops"));
    });

    // --- the bar -----------------------------------------------------------
    let (rect, response) = ui.allocate_exact_size(
        Vec2::new(BAR_WIDTH, BAR_HEIGHT + HANDLE_STRIP),
        Sense::click_and_drag(),
    );
    let bar = Rect::from_min_size(rect.min, Vec2::new(rect.width(), BAR_HEIGHT));

    paint_gradient(ui, bar, ramp);
    ui.painter().rect_stroke(
        bar,
        BAR_CORNER_RADIUS,
        ui.visuals().widgets.noninteractive.bg_stroke,
        egui::StrokeKind::Inside,
    );

    // --- dragging ----------------------------------------------------------
    // Double-click inserts a stop under the pointer. The `+` button does the
    // same thing into the widest gap, and keeps working — but a small glyph in a
    // corner is something you have to already know about, which is exactly the
    // complaint Blender's own version attracts. Double-clicking a gradient to add
    // a stop to it is a guess most people make unprompted, and it puts the stop
    // where you were pointing rather than wherever there happened to be room.
    if response.double_clicked() {
        if let Some(pointer) = response.interact_pointer_pos() {
            let t = ((pointer.x - bar.left()) / bar.width().max(1.0)).clamp(0.0, 1.0);
            let color = ramp.sample(t);
            ramp.insert(ColorStop::new(t, color));
            selected = ramp
                .stops
                .iter()
                .position(|s| s.position == t)
                .unwrap_or(selected);
            changed = true;
        }
    } else if let Some(pointer) = response.interact_pointer_pos() {
        let t = ((pointer.x - bar.left()) / bar.width().max(1.0)).clamp(0.0, 1.0);

        if response.drag_started() || response.clicked() {
            selected = nearest_stop(ramp, t);
        }
        // The emptiness check guards a hand-edited or version-skewed save: the
        // widget never produces a stopless ramp, but `ColorRamp` deserialises
        // one without complaint, and indexing it here would take the frame down.
        if response.dragged() && !ramp.stops.is_empty() {
            ramp.stops[selected].position = t;
            // Re-sorting can move the selected stop's index out from under us,
            // so it is re-found by identity afterwards. Without this, dragging
            // one stop past another silently starts dragging the other one.
            let dragged = ramp.stops[selected];
            ramp.normalize();
            selected = ramp
                .stops
                .iter()
                .position(|s| *s == dragged)
                .unwrap_or(selected.min(ramp.stops.len().saturating_sub(1)));
            changed = true;
        }
    }

    // Which stop a click would grab, so the answer is visible *before*
    // committing to it rather than discovered afterwards. This is most of the
    // "hard to get the cursor in the right place" problem: the target was always
    // the whole bar — click anywhere and the nearest stop is taken — but with
    // nothing highlighted there was no way to know which one that was.
    let hovered = response
        .hover_pos()
        .filter(|_| !response.dragged())
        .map(|p| {
            let t = ((p.x - bar.left()) / bar.width().max(1.0)).clamp(0.0, 1.0);
            nearest_stop(ramp, t)
        });

    paint_handles(ui, rect, bar, ramp, selected, hovered);

    // --- row 3: the selected stop ------------------------------------------
    // Everything here acts on one stop, including the add and remove buttons —
    // which is why they moved down from the mode row. `−` deletes the *selected*
    // stop, so sitting it next to that stop's own color and position is what
    // makes which stop it means obvious without a tooltip.
    ui.horizontal(|ui| {
        if let Some(stop) = ramp.stops.get_mut(selected) {
            // Linear in, linear out. See the module docs.
            if ui
                .color_edit_button_rgba_unmultiplied(&mut stop.color)
                .changed()
            {
                changed = true;
            }
            if ui
                .add(
                    egui::DragValue::new(&mut stop.position)
                        .speed(0.005)
                        .range(0.0..=1.0)
                        // Fixed, so dragging a stop does not resize its own
                        // field and shuffle the buttons beside it.
                        .fixed_decimals(3),
                )
                .changed()
            {
                changed = true;
            }
        }

        if ui.small_button("+").on_hover_text("Add a stop").clicked() {
            // Inserted in the widest gap rather than at the pointer, so the
            // result is predictable and never lands on top of a stop that is
            // already there.
            if let Some(position) = widest_gap_midpoint(ramp) {
                let color = ramp.sample(position);
                ramp.insert(ColorStop::new(position, color));
                selected = ramp
                    .stops
                    .iter()
                    .position(|s| s.position == position)
                    .unwrap_or(selected);
                changed = true;
            }
        }

        let can_remove = ramp.stops.len() > 2;
        if ui
            .add_enabled(can_remove, egui::Button::new("−").small())
            .on_hover_text("Remove the selected stop")
            .clicked()
            && ramp.remove(selected)
        {
            selected = selected.saturating_sub(1);
            changed = true;
        }
    });

    if changed {
        ramp.normalize();
    }
    ui.data_mut(|d| d.insert_temp(selection_id, selected));
    changed
}

/// Paint the ramp as a horizontal gradient.
///
/// Sampled roughly per two pixels and interpolated between samples by the GPU.
/// That is exact for the linear mode and a hair soft for the others — including
/// `Constant`, whose hard edges land within a pixel or two of where the shader
/// puts them rather than exactly on them. Sampling per pixel would remove even
/// that, at the cost of a vertex per pixel for a difference nobody can see.
fn paint_gradient(ui: &Ui, bar: Rect, ramp: &ColorRamp) {
    let samples = ((bar.width() / 2.0).ceil() as usize).clamp(2, 512);
    let mut mesh = egui::Mesh::default();

    for i in 0..=samples {
        let t = i as f32 / samples as f32;
        let x = bar.left() + t * bar.width();
        let color = to_srgb(ramp.sample(t));
        mesh.colored_vertex(egui::pos2(x, bar.top()), color);
        mesh.colored_vertex(egui::pos2(x, bar.bottom()), color);

        if i > 0 {
            let base = (i as u32 - 1) * 2;
            mesh.add_triangle(base, base + 1, base + 2);
            mesh.add_triangle(base + 1, base + 2, base + 3);
        }
    }

    ui.painter().add(egui::Shape::mesh(mesh));
}

/// Draw the stop handles, marking the selected one and the one a click would
/// take.
///
/// # Why the selected stop gets three cues and not one
///
/// A slightly thicker outline on its own is invisible in practice — a one-pixel
/// difference between two small triangles, against a fill that is already
/// whatever color the author chose, which may well be the same color as the
/// highlight. Selection has to survive the stop being black, white, or the
/// exact shade of the accent, so it cannot be carried by color alone.
///
/// So: a **guide line** the full height of the bar (visible even where the
/// handle is not), a **larger** handle, and a **halo** drawn under the outline in
/// the opposite luminance to the fill. Any one of them works on its own; the
/// point is that at least one always survives whatever the stop color is.
fn paint_handles(
    ui: &Ui,
    rect: Rect,
    bar: Rect,
    ramp: &ColorRamp,
    selected: usize,
    hovered: Option<usize>,
) {
    let painter = ui.painter();
    let y = bar.bottom() + 1.0;
    let tip = rect.bottom() - 1.0;

    for (index, stop) in ramp.stops.iter().enumerate() {
        let is_selected = index == selected;
        let is_hovered = hovered == Some(index);
        let x = bar.left() + stop.position.clamp(0.0, 1.0) * bar.width();

        // A vertical line through the gradient itself. The handles sit in a
        // 12pt strip below the bar and are easy to lose among their neighbours;
        // a line crossing the color reads at a glance and cannot be confused
        // with the stop next door.
        if is_selected {
            painter.line_segment(
                [egui::pos2(x, bar.top()), egui::pos2(x, bar.bottom())],
                Stroke::new(SELECTED_HANDLE_STROKE, contrasting(stop.color)),
            );
        }

        let half_width = HANDLE_HALF_WIDTH
            * if is_selected {
                SELECTED_HANDLE_GROWTH
            } else if is_hovered {
                HOVERED_HANDLE_GROWTH
            } else {
                1.0
            };

        let points = vec![
            egui::pos2(x, y),
            egui::pos2(x - half_width, tip),
            egui::pos2(x + half_width, tip),
        ];

        // Drawn under the outline and slightly larger, so the selected handle
        // reads against both a light and a dark stop color.
        if is_selected {
            let halo = half_width + HANDLE_HALO;
            painter.add(egui::Shape::convex_polygon(
                vec![
                    egui::pos2(x, y - HANDLE_HALO),
                    egui::pos2(x - halo, tip + HANDLE_HALO),
                    egui::pos2(x + halo, tip + HANDLE_HALO),
                ],
                contrasting(stop.color),
                Stroke::NONE,
            ));
        }

        // The handle is filled with the stop's own color so the bar can be read
        // without selecting anything.
        let stroke = if is_selected {
            Stroke::new(SELECTED_HANDLE_STROKE, ui.visuals().selection.stroke.color)
        } else if is_hovered {
            Stroke::new(
                SELECTED_HANDLE_STROKE,
                ui.visuals().widgets.hovered.fg_stroke.color,
            )
        } else {
            Stroke::new(
                HANDLE_STROKE,
                ui.visuals().widgets.noninteractive.fg_stroke.color,
            )
        };

        painter.add(egui::Shape::convex_polygon(
            points,
            to_srgb(stop.color),
            stroke,
        ));
    }
}

/// Black or white, whichever will be seen against this color.
///
/// The selection cues are drawn on top of a fill the author chose, so a fixed
/// highlight color is guaranteed to vanish against *some* stop. Choosing by
/// luminance means it never does.
///
/// Uses the Rec. 709 luma weights on the linear values, which is the right
/// comparison for "will the eye see this", rather than comparing sRGB bytes.
fn contrasting(linear: [f32; 4]) -> Color32 {
    let luma = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
    if luma > 0.18 {
        Color32::BLACK
    } else {
        Color32::WHITE
    }
}

/// Linear RGBA to a paintable sRGB color.
fn to_srgb(linear: [f32; 4]) -> Color32 {
    Color32::from(Rgba::from_rgba_unmultiplied(
        linear[0], linear[1], linear[2], linear[3],
    ))
}

fn nearest_stop(ramp: &ColorRamp, t: f32) -> usize {
    ramp.stops
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (a.position - t)
                .abs()
                .partial_cmp(&(b.position - t).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Midpoint of the widest gap between stops, or `None` if there is nowhere
/// sensible to put one.
fn widest_gap_midpoint(ramp: &ColorRamp) -> Option<f32> {
    ramp.stops
        .windows(2)
        .map(|w| {
            (
                w[1].position - w[0].position,
                (w[0].position + w[1].position) / 2.0,
            )
        })
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .filter(|(gap, _)| *gap > 1e-3)
        .map(|(_, mid)| mid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_stop_lands_in_the_widest_gap() {
        let ramp = ColorRamp {
            stops: vec![
                ColorStop::new(0.0, [0.0; 4]),
                ColorStop::new(0.1, [0.0; 4]),
                ColorStop::new(1.0, [1.0; 4]),
            ],
            interp: RampInterp::Linear,
        };
        let mid = widest_gap_midpoint(&ramp).expect("there is a gap");
        assert!(
            (mid - 0.55).abs() < 1e-5,
            "expected the 0.1..1.0 gap, got {mid}"
        );
    }

    /// The selection highlight has to be visible against any stop color.
    ///
    /// The failure it prevents is specific: a fixed highlight tint disappears
    /// entirely when the author picks that same color for a stop, and the stop
    /// most likely to be picked is pure black or pure white.
    #[test]
    fn the_selection_highlight_flips_against_light_and_dark_stops() {
        assert_eq!(contrasting([0.0, 0.0, 0.0, 1.0]), Color32::WHITE);
        assert_eq!(contrasting([1.0, 1.0, 1.0, 1.0]), Color32::BLACK);

        // Green carries most of the luma; blue almost none. A naive average
        // would call these two the same brightness and get one of them wrong.
        assert_eq!(contrasting([0.0, 1.0, 0.0, 1.0]), Color32::BLACK);
        assert_eq!(contrasting([0.0, 0.0, 1.0, 1.0]), Color32::WHITE);
    }

    #[test]
    fn nearest_stop_picks_the_closest_not_the_first() {
        let ramp = ColorRamp {
            stops: vec![
                ColorStop::new(0.0, [0.0; 4]),
                ColorStop::new(0.5, [0.0; 4]),
                ColorStop::new(1.0, [1.0; 4]),
            ],
            interp: RampInterp::Linear,
        };
        assert_eq!(nearest_stop(&ramp, 0.05), 0);
        assert_eq!(nearest_stop(&ramp, 0.45), 1);
        assert_eq!(nearest_stop(&ramp, 0.99), 2);
    }

    /// A stop dragged past its neighbour keeps being the stop you grabbed.
    ///
    /// `normalize` re-sorts, so the index moves. Following it by identity is
    /// what stops a drag from silently transferring to the neighbour halfway
    /// through — which reads as the widget fighting you.
    #[test]
    fn a_dragged_stop_is_followed_across_a_reorder() {
        let mut ramp = ColorRamp {
            stops: vec![
                ColorStop::new(0.0, [0.0, 0.0, 0.0, 1.0]),
                ColorStop::new(0.4, [1.0, 0.0, 0.0, 1.0]),
                ColorStop::new(0.6, [0.0, 1.0, 0.0, 1.0]),
            ],
            interp: RampInterp::Linear,
        };

        // Drag the red stop (index 1) past the green one.
        ramp.stops[1].position = 0.9;
        let dragged = ramp.stops[1];
        ramp.normalize();
        let followed = ramp.stops.iter().position(|s| *s == dragged).unwrap();

        assert_eq!(followed, 2, "the red stop should now be last");
        assert_eq!(ramp.stops[followed].color, [1.0, 0.0, 0.0, 1.0]);
    }
}
