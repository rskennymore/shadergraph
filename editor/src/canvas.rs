//! The node canvas: `egui-snarl` driven directly off `shadergraph`'s node
//! vocabulary.
//!
//! # Why the document is a `Snarl`, not a `Graph`
//!
//! `shadergraph::Graph` is a *compile-time* IR. It is append-only — `NodeId` is
//! a position in its node list, so there is no `remove_node` and no `unlink` —
//! and that is a deliberate property, not an omission: it keeps evaluation
//! order, dead-code elimination and the generated binding names simple enough
//! to be obviously correct.
//!
//! An editor needs the opposite. So the editable document is the `Snarl`, which
//! already stores positions, supports deletion, and is what the widget draws;
//! and a fresh `Graph` is built from it on every change. That costs microseconds
//! for a graph of this size and it means nothing about editing ever leaks into
//! the compiler.

use bevy::prelude::*;
use bevy_egui::egui::{self, emath::TSTransform, Color32, Pos2, Ui};
use egui_snarl::ui::{PinInfo, SnarlPin, SnarlStyle, SnarlViewer};
use egui_snarl::{InPin, NodeId as SnarlNodeId, OutPin, Snarl};
use shadergraph::{
    BlendMode, ColorToFloatOp, GradientType, Graph, MathOp, NodeKind, NoiseOctaves, SocketDefault,
    Value, ValueType, VectorOp, VectorToFloatOp, MAX_NOISE_OCTAVES,
};

/// Width of an exposed parameter's name field, in logical points.
///
/// The canvas zoom is an egui layer transform, so raw sizes scale with the node
/// around them and nothing here needs to know the zoom level.
const NAME_FIELD_WIDTH: f32 = 120.0;

/// The graph being authored.
#[derive(Resource)]
pub struct GraphDocument {
    pub snarl: Snarl<NodeKind>,
    pub style: SnarlStyle,
}

impl Default for GraphDocument {
    fn default() -> Self {
        Self {
            snarl: Snarl::new(),
            style: SnarlStyle::new(),
        }
    }
}

impl GraphDocument {
    /// Replace the document with a laid-out copy of an existing graph.
    ///
    /// This is how the editor opens on something rather than an empty canvas,
    /// and it is the only way to get a hand-written preset — `hull_metal` and
    /// friends — onto the canvas to be taken apart.
    pub fn load(&mut self, graph: &Graph) {
        let positions = layout(graph);
        let mut snarl = Snarl::new();

        // `shadergraph::NodeId` is a dense index and snarl's is a slab key, so
        // the two are related only by this map. Building it as we insert avoids
        // ever assuming they coincide — they do today, and would stop the first
        // time a node was deleted.
        let mut ids = vec![SnarlNodeId(0); graph.len()];
        for (id, kind) in graph.nodes() {
            let pos = positions
                .get(id.index())
                .copied()
                .unwrap_or(Pos2::new(0.0, 0.0));
            ids[id.index()] = snarl.insert_node(pos, kind.clone());
        }

        for ((from, from_idx), (to, to_idx)) in graph.links() {
            snarl.connect(
                egui_snarl::OutPinId {
                    node: ids[from.index()],
                    output: from_idx,
                },
                egui_snarl::InPinId {
                    node: ids[to.index()],
                    input: to_idx,
                },
            );
        }

        self.snarl = snarl;
    }

    /// Write the document to a file, layout and all.
    ///
    /// `Snarl` stores node positions, so a saved graph comes back arranged the
    /// way it was left rather than re-laid-out from scratch. That is most of the
    /// point of saving one — the wiring can always be rebuilt, the arrangement
    /// is the part that took thought.
    ///
    /// Errors name the path, because a save that reports only "failed" leaves
    /// nothing to act on.
    pub fn save(&self, path: &str) -> Result<(), String> {
        let json = serde_json::to_string_pretty(&self.snarl)
            .map_err(|e| format!("could not serialise the graph: {e}"))?;
        std::fs::write(path, json).map_err(|e| format!("could not write {path}: {e}"))
    }

    /// Replace the document with one read from a file.
    ///
    /// On any failure the current document is left untouched — a mistyped path
    /// should not cost you what is on the canvas.
    pub fn load_from(&mut self, path: &str) -> Result<(), String> {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("could not read {path}: {e}"))?;
        let snarl: Snarl<NodeKind> = serde_json::from_str(&text)
            .map_err(|e| format!("{path} is not a graph this version understands: {e}"))?;
        self.snarl = snarl;
        Ok(())
    }

    /// Compile the document back into an IR graph.
    ///
    /// Returns the same errors the compiler would raise on a hand-written
    /// graph, which is what we want: an editor that silently repaired a type
    /// mismatch would be hiding the one message that says what is wrong.
    pub fn to_graph(&self) -> Result<Graph, shadergraph::GraphError> {
        let mut graph = Graph::new();
        let mut ids = std::collections::HashMap::new();

        for (snarl_id, kind) in self.snarl.node_ids() {
            ids.insert(snarl_id, graph.add(kind.clone()));
        }

        for (out, in_) in self.snarl.wires() {
            // A wire whose endpoint is missing cannot happen through the UI —
            // snarl drops wires with the node — but skipping is still the right
            // response, because failing the whole compile over it would leave
            // the author with an error about a node they cannot see.
            let (Some(&from), Some(&to)) = (ids.get(&out.node), ids.get(&in_.node)) else {
                continue;
            };
            graph.link_by_index(from, out.output, to, in_.input)?;
        }

        Ok(graph)
    }
}

/// Column-and-row positions for an existing graph's nodes.
///
/// Depth is "one past the deepest input", walked in evaluation order so every
/// input is already resolved by the time a node is reached. Without this a
/// loaded preset arrives as a pile of overlapping boxes at the origin, which is
/// technically the same graph and completely unreadable.
fn layout(graph: &Graph) -> Vec<Pos2> {
    const COLUMN: f32 = 260.0;
    const ROW: f32 = 150.0;

    let mut incoming: Vec<Vec<usize>> = vec![Vec::new(); graph.len()];
    for ((from, _), (to, _)) in graph.links() {
        incoming[to.index()].push(from.index());
    }

    let mut depth = vec![0usize; graph.len()];
    if let Ok(order) = graph.evaluation_order() {
        for node in order {
            let d = incoming[node.index()]
                .iter()
                .map(|&src| depth[src] + 1)
                .max()
                .unwrap_or(0);
            depth[node.index()] = d;
        }
    }

    let mut next_row = std::collections::HashMap::new();
    depth
        .iter()
        .map(|&d| {
            let row = next_row.entry(d).or_insert(0usize);
            let pos = Pos2::new(d as f32 * COLUMN, *row as f32 * ROW);
            *row += 1;
            pos
        })
        .collect()
}

/// Where the pointer was when the add-node menu was opened, in screen points.
///
/// Lives in egui's own memory rather than on [`NodeViewer`], which is a unit
/// struct rebuilt every frame, or on a resource, which would put a piece of
/// pointer state outside the only code that reads it.
fn menu_anchor_id() -> egui::Id {
    egui::Id::new("shadergraph::graph_menu_anchor")
}

/// Record the pointer position on every right-click, for the node menu to spawn
/// at.
///
/// Must be called before the snarl widget is shown, from a `Ui` on the same
/// context.
///
/// # Why this is needed at all
///
/// egui-snarl (still, as of 0.11) hands `show_graph_menu` a position derived
/// from `ui.cursor().min` — the top-left of the *menu's own content*, not the
/// pointer. Those coincide only while the menu is small enough to be drawn
/// where it was asked for. A palette past about twenty entries makes the menu
/// tall enough that egui shifts it up the screen to keep it visible, and the
/// spawn point moves with it — so new nodes land at the top of the canvas
/// regardless of where the click was.
///
/// The pointer is captured here, at press time, because by the time a palette
/// button is clicked the pointer is inside the menu and no longer says anything
/// about where the node was wanted.
pub fn latch_menu_anchor(ui: &egui::Ui) {
    if !ui.input(|i| i.pointer.secondary_pressed()) {
        return;
    }
    if let Some(pos) = ui.input(|i| i.pointer.interact_pos()) {
        ui.data_mut(|d| d.insert_temp(menu_anchor_id(), pos));
    }
}

/// Draws the graph and enforces the compiler's linking rules at the point of
/// the gesture.
///
/// Rebuilt every frame; the one field is per-frame state, not document state.
#[derive(Default)]
pub struct NodeViewer {
    /// The graph→screen transform snarl reports through `current_transform`,
    /// kept so the menu handlers can map the latched pointer (a screen point)
    /// back into graph space. Identity until snarl calls in, which it does
    /// before any menu can open.
    to_global: TSTransform,
}

impl SnarlViewer<NodeKind> for NodeViewer {
    fn title(&mut self, node: &NodeKind) -> String {
        match node {
            NodeKind::Math(op) => format!("Math · {}", op.label()),
            NodeKind::VectorMath(op) => format!("Vector · {}", op.label()),
            NodeKind::VectorToFloat(op) => format!("Vector → Float · {}", op.label()),
            other => other.label(),
        }
    }

    fn inputs(&mut self, node: &NodeKind) -> usize {
        node.inputs().len()
    }

    fn outputs(&mut self, node: &NodeKind) -> usize {
        node.outputs().len()
    }

    fn show_input(
        &mut self,
        pin: &InPin,
        ui: &mut Ui,
        snarl: &mut Snarl<NodeKind>,
    ) -> impl SnarlPin + 'static {
        let Some(kind) = snarl.get_node(pin.id.node) else {
            return PinInfo::circle();
        };
        let Some(def) = kind.inputs().get(pin.id.input) else {
            return PinInfo::circle();
        };

        ui.label(def.name);

        // What an unwired socket falls back to is not visible anywhere else,
        // and it is frequently the answer to "why is this black". Shown only
        // while unwired, because once a wire arrives the default is irrelevant.
        if pin.remotes.is_empty() {
            ui.weak(match def.default {
                SocketDefault::Value(value) => describe_value(value),
                SocketDefault::Required => "required".to_string(),
                SocketDefault::Context(expr) => expr.to_string(),
            });
        }

        pin_for(
            def.ty,
            matches!(def.default, SocketDefault::Required) && pin.remotes.is_empty(),
        )
    }

    fn show_output(
        &mut self,
        pin: &OutPin,
        ui: &mut Ui,
        snarl: &mut Snarl<NodeKind>,
    ) -> impl SnarlPin + 'static {
        let Some(kind) = snarl.get_node(pin.id.node) else {
            return PinInfo::circle();
        };
        let Some(def) = kind.outputs().get(pin.id.output) else {
            return PinInfo::circle();
        };

        ui.label(def.name);
        pin_for(def.ty, false)
    }

    /// Reject a wire the compiler would reject, and keep inputs single-valued.
    ///
    /// Both rules are the IR's, restated here because snarl asks before it
    /// connects. Letting the wire land and failing at compile time would be
    /// worse than useless: the canvas would show a graph that cannot exist.
    fn connect(&mut self, from: &OutPin, to: &InPin, snarl: &mut Snarl<NodeKind>) {
        let (Some(from_kind), Some(to_kind)) =
            (snarl.get_node(from.id.node), snarl.get_node(to.id.node))
        else {
            return;
        };

        // Both are `&'static`, so the borrows of `snarl` end here and it is free
        // to be mutated below.
        let (Some(out_def), Some(in_def)) = (
            from_kind.outputs().get(from.id.output),
            to_kind.inputs().get(to.id.input),
        ) else {
            return;
        };

        // Both are `&'static`, so the borrows above have ended and these are
        // plain copies — snarl is free to be mutated from here on.
        let (mut out_ty, mut in_ty) = (out_def.ty, in_def.ty);

        // A reroute takes the type of the first wire it is given, which is what
        // makes it feel like Blender's: you place one and connect it, rather than
        // deciding up front what it will carry. Only ever done to a reroute with
        // nothing attached — retyping one that is already passing a value through
        // would silently break the wire at its other end.
        //
        // Tried in both directions because a reroute is as often dragged *from*
        // as dragged *to*, and only one can succeed: after the first branch the
        // types agree, so the second's guard is false.
        if out_ty != in_ty {
            if adopt_type(snarl, to.id.node, out_ty) {
                in_ty = out_ty;
            } else if adopt_type(snarl, from.id.node, in_ty) {
                out_ty = in_ty;
            }
        }

        if out_ty != in_ty {
            return;
        }

        // An input accepts at most one wire, so a new one replaces rather than
        // joins. snarl allows many-to-one by default; the compiler does not.
        for &remote in &to.remotes {
            snarl.disconnect(remote, to.id);
        }

        snarl.connect(from.id, to.id);
    }

    /// Label a `Param` with the socket it feeds.
    ///
    /// A graph full of nodes called "Param (Float)" tells you nothing; the same
    /// graph with "→ Roughness", "→ Randomness", "→ Anisotropy Strength" reads at
    /// a glance. The name is *derived*, never stored, so rewiring a param
    /// relabels it and there is no second source of truth to go stale.
    ///
    /// This has to be `show_header` rather than `title`, which is where you
    /// would look first: `title` receives only the node's value and cannot see
    /// the graph, so it has no way to know what the node is wired to.
    fn show_header(
        &mut self,
        node: SnarlNodeId,
        _inputs: &[InPin],
        outputs: &[OutPin],
        ui: &mut Ui,
        snarl: &mut Snarl<NodeKind>,
    ) {
        let Some(kind) = snarl.get_node(node) else {
            return;
        };

        if let NodeKind::Param { value, name } = kind {
            match (name, destination_label(outputs, snarl)) {
                // An exposed param's name outranks what it is wired to: the name
                // is a contract with whatever drives the material, and it is the
                // thing you are scanning the graph for.
                (Some(exposed), destination) => {
                    ui.label(format!("⇢ {exposed}"));
                    match destination {
                        Some(d) => ui.weak(format!("{} → {d}", value.ty())),
                        None => ui.weak(value.ty().to_string()),
                    };
                }
                (None, Some(destination)) => {
                    ui.label(format!("→ {destination}"));
                    // The type stays visible but demoted: it matters when wiring
                    // and not when reading.
                    ui.weak(value.ty().to_string());
                }
                // Nothing downstream yet, so there is no better name than the
                // one it was created with.
                (None, None) => {
                    ui.label(kind.label());
                }
            }
            return;
        }

        ui.label(self.title(kind));
    }

    fn has_body(&mut self, node: &NodeKind) -> bool {
        matches!(
            node,
            NodeKind::Param { .. }
                | NodeKind::TexImage { .. }
                | NodeKind::Noise(_)
                | NodeKind::MixColor(_)
                | NodeKind::Math(_)
                | NodeKind::VectorMath(_)
                | NodeKind::VectorToFloat(_)
                | NodeKind::ColorToFloat(_)
                | NodeKind::ColorRamp(_)
                | NodeKind::Gradient(_)
        )
    }

    fn show_body(
        &mut self,
        node: SnarlNodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut Ui,
        snarl: &mut Snarl<NodeKind>,
    ) {
        let Some(kind) = snarl.get_node_mut(node) else {
            return;
        };

        match kind {
            // The only editable values in the graph. An unwired socket's default
            // is a compile-time constant of the node *kind* and cannot be tuned
            // — to vary something you wire a Param to it, exactly as in Blender
            // you would drive a socket from a Value node.
            NodeKind::Param { value, name } => {
                match value {
                    Value::Float(v) => {
                        ui.add(egui::DragValue::new(v).speed(0.01));
                    }
                    Value::Vec3(v) => {
                        ui.horizontal(|ui| {
                            for component in v.iter_mut() {
                                ui.add(egui::DragValue::new(component).speed(0.01));
                            }
                        });
                    }
                    Value::Color(c) => {
                        ui.color_edit_button_rgba_unmultiplied(c);
                    }
                }

                // Only exposed params get a name field, and only because the
                // name has to be typeable: it must match a custom property
                // authored in Blender exactly, and guessing at it from a menu
                // label would be worse than showing the string.
                if let Some(exposed) = name {
                    ui.add(
                        egui::TextEdit::singleline(exposed)
                            .desired_width(NAME_FIELD_WIDTH)
                            .hint_text("input name"),
                    );
                }
            }
            // The path is this node's whole payload, so it is always editable
            // — unlike a param's name, which appears only once exposed.
            // Dropping an image file onto the window fills it too; see
            // `preview::receive_dropped_files`.
            NodeKind::TexImage { path, name } => {
                ui.add(
                    egui::TextEdit::singleline(path)
                        .desired_width(NAME_FIELD_WIDTH)
                        .hint_text("image path"),
                );
                if let Some(exposed) = name {
                    ui.add(
                        egui::TextEdit::singleline(exposed)
                            .desired_width(NAME_FIELD_WIDTH)
                            .hint_text("texture name"),
                    );
                }
            }
            NodeKind::Math(op) => {
                egui::ComboBox::from_id_salt(("math", node))
                    .selected_text(op.label())
                    .show_ui(ui, |ui| {
                        for &candidate in MathOp::ALL {
                            ui.selectable_value(op, candidate, candidate.label());
                        }
                    });
            }
            NodeKind::VectorMath(op) => {
                egui::ComboBox::from_id_salt(("vector", node))
                    .selected_text(op.label())
                    .show_ui(ui, |ui| {
                        for &candidate in VectorOp::ALL {
                            ui.selectable_value(op, candidate, candidate.label());
                        }
                    });
            }
            NodeKind::VectorToFloat(op) => {
                egui::ComboBox::from_id_salt(("vecf", node))
                    .selected_text(op.label())
                    .show_ui(ui, |ui| {
                        for &candidate in VectorToFloatOp::ALL {
                            ui.selectable_value(op, candidate, candidate.label());
                        }
                    });
            }
            // The whole point of carrying two variants is being able to flip
            // between them on one graph, so the control sits on the node rather
            // than requiring a different node to be placed.
            NodeKind::Noise(octaves) => {
                egui::ComboBox::from_id_salt(("noise", node))
                    .selected_text(octaves.label())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(octaves, NoiseOctaves::Live, "Live");
                        // Switching to unrolled keeps whatever count was last
                        // set, or starts at 4 — a middle value, so the first
                        // comparison is against something representative rather
                        // than a degenerate one-octave shader.
                        let current = octaves.unrolled_count().max(1);
                        ui.selectable_value(
                            octaves,
                            NoiseOctaves::Unrolled(if current > 1 { current } else { 4 }),
                            "Unrolled",
                        );
                    });

                if let NoiseOctaves::Unrolled(count) = octaves {
                    ui.add(
                        egui::DragValue::new(count)
                            .speed(0.1)
                            .range(1..=MAX_NOISE_OCTAVES)
                            .prefix("octaves "),
                    );
                }
            }
            NodeKind::MixColor(mode) => {
                egui::ComboBox::from_id_salt(("blend", node))
                    .selected_text(mode.label())
                    .show_ui(ui, |ui| {
                        for &candidate in BlendMode::ALL {
                            ui.selectable_value(mode, candidate, candidate.label());
                        }
                    });
            }
            NodeKind::Gradient(kind) => {
                egui::ComboBox::from_id_salt(("grad", node))
                    .selected_text(kind.label())
                    .show_ui(ui, |ui| {
                        for &candidate in GradientType::ALL {
                            ui.selectable_value(kind, candidate, candidate.label());
                        }
                    });
            }
            NodeKind::ColorToFloat(op) => {
                egui::ComboBox::from_id_salt(("colf", node))
                    .selected_text(op.label())
                    .show_ui(ui, |ui| {
                        for &candidate in ColorToFloatOp::ALL {
                            ui.selectable_value(op, candidate, candidate.label());
                        }
                    });
            }
            NodeKind::ColorRamp(ramp) => {
                crate::ramp_widget::ramp_editor(ui, &format!("ramp{}", node.0), ramp);
            }
            _ => {}
        }
    }

    fn has_graph_menu(&mut self, _pos: Pos2, _snarl: &mut Snarl<NodeKind>) -> bool {
        true
    }

    fn show_graph_menu(&mut self, pos: Pos2, ui: &mut Ui, snarl: &mut Snarl<NodeKind>) {
        let spawn = spawn_position(self.to_global, pos, ui);

        ui.label("Add node");
        ui.separator();
        // Driven off the compiler's own vocabulary. A hand-written menu is one
        // that stops matching the node set the first time a node is added.
        for kind in NodeKind::palette() {
            if ui.button(kind.label()).clicked() {
                snarl.insert_node(spawn, kind);
                ui.close();
            }
        }
    }

    fn has_node_menu(&mut self, _node: &NodeKind) -> bool {
        true
    }

    fn show_node_menu(
        &mut self,
        node: SnarlNodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut Ui,
        snarl: &mut Snarl<NodeKind>,
    ) {
        // A graph must contain exactly one output node, so deleting it would
        // put the document in a state with no legal way back except undo, which
        // does not exist yet. Refusing is kinder than allowing and then
        // reporting `NoOutputNode` on every frame afterwards.
        if matches!(snarl.get_node(node), Some(NodeKind::PbrOutput)) {
            ui.weak("The output node cannot be removed.");
            return;
        }

        // Exposing lives in the menu rather than as a field on every Param,
        // because most params are internal constants and a name box on each of
        // them would be ten boxes of clutter to serve one.
        //
        // The starting name is taken from whatever the param drives, which is
        // almost always the name you wanted — a param feeding Perceptual
        // Roughness exposed as `perceptual_roughness` needs no typing at all.
        if let Some(NodeKind::Param { name, .. }) = snarl.get_node(node) {
            let exposed = name.is_some();
            let suggestion = destination_label(_outputs, snarl)
                .map(|label| slugify(&label))
                .unwrap_or_else(|| "input".to_string());

            let action = if exposed {
                "Make internal again"
            } else {
                "Expose as material input"
            };
            if ui.button(action).clicked() {
                if let Some(NodeKind::Param { name, .. }) = snarl.get_node_mut(node) {
                    *name = if exposed { None } else { Some(suggestion) };
                }
                ui.close();
            }
        }

        // The texture twin of the Param entry above, with the suggestion drawn
        // from the file rather than the destination: `decals/insignia.png`
        // exposed as `insignia` is almost always the name wanted, and the
        // destination socket ("Base Color") almost never is.
        if let Some(NodeKind::TexImage { path, name }) = snarl.get_node(node) {
            let exposed = name.is_some();
            let suggestion = std::path::Path::new(path)
                .file_stem()
                .and_then(|stem| stem.to_str())
                .map(slugify)
                .filter(|slug| !slug.is_empty())
                .unwrap_or_else(|| "texture".to_string());

            let action = if exposed {
                "Make internal again"
            } else {
                "Expose as texture input"
            };
            if ui.button(action).clicked() {
                if let Some(NodeKind::TexImage { name, .. }) = snarl.get_node_mut(node) {
                    *name = if exposed { None } else { Some(suggestion) };
                }
                ui.close();
            }
        }

        if ui.button("Delete").clicked() {
            snarl.remove_node(node);
            ui.close();
        }
    }

    /// Capture the frame's graph→screen transform for the menu handlers.
    ///
    /// Snarl calls this at the start of rendering, before any menu can open, so
    /// by the time `show_graph_menu` needs it the field is this frame's truth.
    fn current_transform(&mut self, to_global: &mut TSTransform, _snarl: &mut Snarl<NodeKind>) {
        self.to_global = *to_global;
    }
}

/// Turn a socket's display name into a key an author would type in Blender.
///
/// `Perceptual Roughness` becomes `perceptual_roughness`. Not a general slug
/// function — it handles the only alphabet socket names use, and anything else
/// is dropped rather than transliterated, because a name containing a stray
/// character would fail to match its custom property with nothing to show why.
fn slugify(label: &str) -> String {
    let mut out = String::with_capacity(label.len());
    for ch in label.chars() {
        if ch.is_ascii_alphanumeric() {
            out.extend(ch.to_lowercase());
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    out.trim_matches('_').to_string()
}

/// Where a new node should be placed, in graph space.
///
/// The latched pointer is a screen point; `to_global` is the frame's
/// graph→screen transform, so its inverse maps the pointer straight into graph
/// space — no reference pair needed, unlike the pre-0.11 workaround this
/// replaces.
///
/// Falls back to `menu_graph` — the position snarl derived from the menu
/// itself — when there is no latched pointer or the transform is degenerate. A
/// node in a slightly wrong place is a nuisance; refusing to place one is a
/// broken editor.
fn spawn_position(to_global: TSTransform, menu_graph: Pos2, ui: &Ui) -> Pos2 {
    if to_global.scaling <= f32::EPSILON {
        return menu_graph;
    }
    match ui.data(|d| d.get_temp::<Pos2>(menu_anchor_id())) {
        Some(pointer) => to_global.inverse() * pointer,
        None => menu_graph,
    }
}

/// Retype an unwired reroute to `ty`, reporting whether it happened.
///
/// Returns `false` for anything that is not a reroute, and for a reroute that
/// already carries the type asked for — so a caller can use it as "did this
/// resolve the mismatch?" without checking the node kind itself.
///
/// The "nothing attached" test is the whole safety argument. A reroute's input
/// and output are the same socket, so changing its type changes both ends at
/// once; do that to one that is wired and the far end becomes a link between
/// mismatched sockets that the canvas would happily draw and the compiler would
/// then refuse.
fn adopt_type(snarl: &mut Snarl<NodeKind>, node: SnarlNodeId, ty: ValueType) -> bool {
    match snarl.get_node(node) {
        Some(NodeKind::Reroute(current)) if *current != ty => {}
        _ => return false,
    }

    let attached = snarl
        .wires()
        .any(|(out, in_)| out.node == node || in_.node == node);
    if attached {
        return false;
    }

    if let Some(NodeKind::Reroute(current)) = snarl.get_node_mut(node) {
        *current = ty;
        return true;
    }
    false
}

/// Socket color by data type, following Blender's convention closely enough to
/// be readable by anyone who has used it: grey scalars, blue vectors, yellow
/// colors, green surfaces. The point is to make an illegal connection obvious
/// before it is attempted rather than after it is refused.
fn type_color(ty: ValueType) -> Color32 {
    match ty {
        ValueType::Float => Color32::from_rgb(160, 160, 160),
        ValueType::Vec3 => Color32::from_rgb(100, 130, 235),
        ValueType::Color => Color32::from_rgb(220, 200, 90),
        // Green, which is where Blender puts its Shader sockets — the closest
        // thing it has to this, and the association worth borrowing even though
        // the semantics deliberately differ (see `ValueType::Surface`).
        ValueType::Surface => Color32::from_rgb(110, 200, 120),
    }
}

/// A pin, drawn hollow with a warning outline when it is a required socket that
/// nothing is wired into — the one graph error you can see coming.
fn pin_for(ty: ValueType, missing: bool) -> PinInfo {
    let pin = PinInfo::circle().with_fill(type_color(ty));
    if missing {
        pin.with_stroke(egui::Stroke::new(2.0_f32, Color32::from_rgb(230, 90, 90)))
    } else {
        pin
    }
}

/// Name the socket a node's output feeds, for labelling.
///
/// A node may feed several sockets. Naming the first and counting the rest is a
/// deliberate choice over listing them all: a header that grows with the graph
/// stops being a label. Sockets are named by the socket alone when that reads
/// unambiguously, and qualified with the node when it does not — "Value_001"
/// says nothing, "Math · Value_001" at least says where to look.
fn destination_label(outputs: &[OutPin], snarl: &Snarl<NodeKind>) -> Option<String> {
    let mut destinations = outputs.iter().flat_map(|pin| pin.remotes.iter());
    let first = destinations.next()?;
    let extra = destinations.count();

    let target = snarl.get_node(first.node)?;
    let socket = target.inputs().get(first.input)?;

    // Sockets whose names are positional rather than descriptive get qualified.
    let generic = socket.name.starts_with("Value") || socket.name.starts_with("Vector");
    let name = if generic {
        format!("{} · {}", target.name(), socket.name)
    } else {
        socket.name.to_string()
    };

    Some(if extra > 0 {
        format!("{name} +{extra}")
    } else {
        name
    })
}

fn describe_value(value: Value) -> String {
    match value {
        Value::Float(v) => format!("{v}"),
        Value::Vec3(v) => format!("({}, {}, {})", v[0], v[1], v[2]),
        Value::Color(c) => format!("({}, {}, {}, {})", c[0], c[1], c[2], c[3]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A new node lands under the pointer, not under the menu.
    ///
    /// `spawn_position` maps the latched screen point into graph space through
    /// the frame's transform. These cases pin the two ways that goes wrong:
    /// forgetting the scale, and applying the pan backwards.
    #[test]
    fn a_new_node_is_placed_where_the_pointer_was() {
        let ctx = egui::Context::default();
        let anchor = Pos2::new(500.0, 400.0);
        ctx.data_mut(|d| d.insert_temp(menu_anchor_id(), anchor));

        // Snarl's own answer, deliberately far from the pointer — egui shifting
        // a tall popup up the screen, which is the bug being corrected.
        let menu_graph = Pos2::new(20.0, 10.0);

        let _ = ctx.run_ui(Default::default(), |ui| {
            // Identity transform: graph space IS screen space.
            let at_identity = spawn_position(TSTransform::IDENTITY, menu_graph, ui);
            assert_eq!(at_identity, anchor);

            // Zoomed to 2x with a pan: screen = graph * 2 + (100, 60), so
            // the pointer maps back to ((500-100)/2, (400-60)/2).
            let to_global = TSTransform::new(egui::Vec2::new(100.0, 60.0), 2.0);
            let at_2x = spawn_position(to_global, menu_graph, ui);
            assert_eq!(at_2x, Pos2::new(200.0, 170.0));

            // Degenerate scale falls back rather than inverting a
            // non-invertible transform.
            assert_eq!(
                spawn_position(TSTransform::new(egui::Vec2::ZERO, 0.0), menu_graph, ui),
                menu_graph
            );
        });
    }

    /// With nothing latched, placement is exactly what snarl asked for.
    ///
    /// The menu can be opened by paths that never went through a right-click,
    /// and those must still place a node somewhere reasonable.
    #[test]
    fn without_a_latched_pointer_the_menu_position_is_used_unchanged() {
        let ctx = egui::Context::default();
        let menu_graph = Pos2::new(7.0, 9.0);
        let _ = ctx.run_ui(Default::default(), |ui| {
            assert_eq!(
                spawn_position(TSTransform::IDENTITY, menu_graph, ui),
                menu_graph
            );
        });
    }

    /// A fresh reroute takes the type of whatever is plugged into it.
    ///
    /// Tested at the `adopt_type` level rather than through `connect`, which
    /// needs snarl's pin structs and therefore a live UI. The rule this encodes
    /// is the interesting part and it is entirely here: retype only a reroute,
    /// only one that carries nothing.
    #[test]
    fn an_unwired_reroute_adopts_the_type_it_is_given() {
        let mut snarl = Snarl::new();
        let via = snarl.insert_node(Pos2::ZERO, NodeKind::Reroute(ValueType::Float));

        assert!(adopt_type(&mut snarl, via, ValueType::Color));
        assert!(matches!(
            snarl.get_node(via),
            Some(NodeKind::Reroute(ValueType::Color))
        ));

        // Asking for what it already is reports "nothing to resolve", so a
        // caller can treat the return as "did this fix the mismatch".
        assert!(!adopt_type(&mut snarl, via, ValueType::Color));

        // Anything that is not a reroute is left alone.
        let math = snarl.insert_node(Pos2::ZERO, NodeKind::Math(MathOp::Add));
        assert!(!adopt_type(&mut snarl, math, ValueType::Color));
    }

    /// Once a reroute is carrying a value its type is fixed.
    ///
    /// The guard that keeps adoption safe: a reroute's input and output are the
    /// same socket, so retyping changes both ends at once. Doing that to a wired
    /// one would leave a link between mismatched sockets that the canvas draws
    /// and the compiler then refuses.
    #[test]
    fn a_wired_reroute_keeps_its_type() {
        let mut snarl = Snarl::new();
        let source = snarl.insert_node(
            Pos2::ZERO,
            NodeKind::Param {
                value: Value::Float(1.0),
                name: None,
            },
        );
        let via = snarl.insert_node(Pos2::ZERO, NodeKind::Reroute(ValueType::Float));

        snarl.connect(
            egui_snarl::OutPinId {
                node: source,
                output: 0,
            },
            egui_snarl::InPinId {
                node: via,
                input: 0,
            },
        );

        assert!(!adopt_type(&mut snarl, via, ValueType::Color));
        assert!(matches!(
            snarl.get_node(via),
            Some(NodeKind::Reroute(ValueType::Float))
        ));
    }

    /// The bridge between the canvas and the compiler, in both directions.
    ///
    /// This is the riskiest code in the editor and the cheapest to get subtly
    /// wrong — a socket index off by one still produces a graph that compiles,
    /// just not the one on screen. Comparing emitted WGSL rather than graph
    /// structure is deliberate: it is the thing that actually has to match.
    #[test]
    fn a_loaded_preset_compiles_back_to_itself() {
        for (name, graph) in [
            ("hull_metal", materials_hull_metal()),
            ("glass", materials_glass()),
        ] {
            let mut document = GraphDocument::default();
            document.load(&graph);

            let rebuilt = document
                .to_graph()
                .unwrap_or_else(|e| panic!("{name} did not survive the round trip: {e}"));

            assert_eq!(
                shadergraph::emit(&rebuilt).unwrap(),
                shadergraph::emit(&graph).unwrap(),
                "{name} changed on the way through the canvas"
            );
        }
    }

    /// Every node gets its own spot.
    ///
    /// Overlapping nodes are not a cosmetic problem — a node hidden underneath
    /// another cannot be selected, so a preset that lays out badly is a preset
    /// you cannot edit.
    #[test]
    fn layout_does_not_stack_nodes() {
        let graph = materials_hull_metal();
        let positions = layout(&graph);
        assert_eq!(positions.len(), graph.len());

        for (i, a) in positions.iter().enumerate() {
            for b in &positions[i + 1..] {
                assert!(a != b, "two nodes were placed at {a:?}");
            }
        }
    }

    /// A saved graph comes back as the same graph.
    ///
    /// The serialised form crosses a version boundary — a file written today is
    /// read by a build from months from now — so this is the one place where a
    /// silent format change costs real work. Comparing emitted WGSL rather than
    /// the document checks the thing that matters and ignores the thing that
    /// does not (node positions are preserved, but a difference in them is not a
    /// correctness failure).
    #[test]
    fn a_saved_graph_reloads_as_the_same_graph() {
        let path = std::env::temp_dir().join("shadergraph_roundtrip_test.json");
        let path = path.to_str().expect("a utf-8 temp path");

        let mut original = GraphDocument::default();
        original.load(&materials_hull_metal());
        original.save(path).expect("the graph saves");

        let mut reloaded = GraphDocument::default();
        reloaded.load_from(path).expect("the graph loads");

        assert_eq!(
            shadergraph::emit(&reloaded.to_graph().unwrap()).unwrap(),
            shadergraph::emit(&original.to_graph().unwrap()).unwrap(),
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn loading_a_bad_file_leaves_the_canvas_alone() {
        // A mistyped path or a stale file should cost a message, not the work
        // currently on screen.
        let mut document = GraphDocument::default();
        document.load(&materials_hull_metal());
        let before = document.to_graph().unwrap();

        let error = document
            .load_from("/definitely/not/a/graph/file.json")
            .expect_err("a missing file is an error");
        assert!(
            error.contains("/definitely/not/a/graph/file.json"),
            "{error}"
        );

        assert_eq!(
            shadergraph::emit(&document.to_graph().unwrap()).unwrap(),
            shadergraph::emit(&before).unwrap(),
        );
    }

    fn materials_hull_metal() -> Graph {
        shadergraph::materials::hull_metal().unwrap()
    }

    fn materials_glass() -> Graph {
        shadergraph::materials::glass([0.1, 0.12, 0.2, 1.0], 0.3).unwrap()
    }
}
