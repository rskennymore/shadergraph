//! Where a graph's `Param` values live once the shader is compiled.
//!
//! # Why they are not literals
//!
//! Baking a `Param` into the emitted WGSL is the obvious thing and it is wrong
//! for two reasons that compound.
//!
//! The first is cost per edit. A value in the source means *changing* the value
//! changes the source, so nudging a slider recompiles a shader and rebuilds a
//! pipeline. That is survivable in a batch build and intolerable in an editor,
//! where the whole point is that dragging a number moves the picture.
//!
//! The second is cost per material. Two materials that differ only in color
//! compile to two shaders and occupy two pipelines. Read the other way round:
//! moving values out of the source means every material sharing a graph shares
//! one compiled pipeline, however many of them there are.
//!
//! # The layout
//!
//! Each `Param` node is assigned a slot in a `vec4` array, and the emitter
//! writes `params.slots[n]` where it used to write a number. One param per slot
//! — a float uses `.x` and wastes three components — because packing four
//! floats into a slot buys capacity nobody is short of yet, at the cost of a
//! layout that has to be reasoned about every time it is read. [`ParamLayout`]
//! is the seam that keeps that decision cheap to revisit.
//!
//! NOTE: A `vec4` array is also the alignment-safe choice. A `vec3` in a uniform
//! occupies 16 bytes, not 12, and packing three floats where the shader expects
//! four is the classic way to get a shader that is quietly, subtly wrong rather
//! than broken.
//!
//! # Blocks, not just params
//!
//! A `Param` takes one slot, but nothing here requires that. A color ramp takes
//! a *run* of consecutive slots holding its per-segment polynomial coefficients,
//! and the same argument that moved params out of the source applies to it
//! unchanged: dragging a stop should be an upload, not a recompile.
//!
//! What stays in the source is only what changes the *shape* of the generated
//! code — for a ramp, the number of segments. Everything that is merely a number
//! goes in a slot. [`SlotValue`] is the distinction: `Param` is a value a person
//! typed, `Raw` is a row of a generated block that nobody edits directly.

use std::collections::BTreeMap;

use crate::graph::NodeId;
use crate::value::{Value, ValueType};

/// Number of `vec4` slots a compiled graph may use.
///
/// Shared by the emitter (which writes this number into the generated WGSL) and
/// by the renderer-side uniform struct, so the two cannot disagree about the
/// size of the buffer. 128 slots is 2 KiB — nothing against a uniform buffer
/// budget measured in tens of kilobytes.
///
/// It was 32 while only `Param` nodes consumed slots. A color ramp consumes
/// seven per segment, so a four-stop ramp alone wants 22, and a graph with two
/// of them plus its ordinary parameters would have hit the old ceiling.
pub const PARAM_SLOTS: usize = 128;

/// Number of texture bindings a compiled graph may use.
///
/// Small and fixed where [`PARAM_SLOTS`] is generous, because the two scale
/// differently: a uniform slot is sixteen bytes in one buffer, while a texture
/// slot is a binding the renderer's material type has to declare as a field —
/// bindings cannot be conjured at runtime, so unused slots still occupy the
/// bind group layout of every material. Four sampled images is already a lot
/// for a pipeline whose premise is procedural surfaces; raising this is one
/// constant here plus one texture/sampler field pair per new slot on the
/// renderer side.
pub const TEXTURE_SLOTS: usize = 4;

/// What a slot holds.
///
/// The variants differ in provenance, not in representation — both end up as
/// four floats. The distinction is what lets an editor tell "a value the author
/// set" from "a number the compiler computed", and it keeps [`ParamLayout::packed`]
/// honest about which slots have a type and which are just bits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SlotValue {
    /// A `Param` node's value, padded to `vec4` according to its type.
    Param(Value),
    /// One row of a generated block — a ramp's segment coefficients today, a
    /// curve's control points later. Written by the node that owns the block and
    /// read back by generated code that knows the row order.
    Raw([f32; 4]),
}

impl SlotValue {
    /// The four floats this slot occupies in the buffer.
    ///
    /// Trailing components of a float or vector param are zero rather than
    /// undefined, so a shader reading `.xyz` from a slot written as a float gets
    /// a defined answer instead of whatever was in the buffer before.
    pub fn packed(&self) -> [f32; 4] {
        match self {
            SlotValue::Param(Value::Float(v)) => [*v, 0.0, 0.0, 0.0],
            SlotValue::Param(Value::Vec3(v)) => [v[0], v[1], v[2], 0.0],
            SlotValue::Param(Value::Color(v)) => *v,
            SlotValue::Raw(v) => *v,
        }
    }

    /// Whether two slots are the same *kind* of slot, values aside.
    ///
    /// A `Param` slot's type is part of its shape, because changing a float param
    /// to a color changes the swizzle the shader reads it with. A `Raw` slot has
    /// no type to change — its meaning comes from its position in a block.
    fn same_shape(&self, other: &SlotValue) -> bool {
        match (self, other) {
            (SlotValue::Param(a), SlotValue::Param(b)) => a.ty() == b.ty(),
            (SlotValue::Raw(_), SlotValue::Raw(_)) => true,
            _ => false,
        }
    }
}

/// One slot, and the node it belongs to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParamSlot {
    /// The node this slot holds data for. Several consecutive slots may name the
    /// same node when it owns a block.
    pub node: NodeId,
    /// The contents at the time the graph was compiled.
    ///
    /// A snapshot, not a live view. It is what the uniform should be filled
    /// with if nothing has changed since, which makes it the natural thing to
    /// compare against to decide whether an upload is needed at all.
    pub value: SlotValue,
}

/// One texture binding: which node it samples for, and where the image lives.
///
/// The texture counterpart of [`ParamSlot`], with a reference where that has a
/// value: the compiler cannot hold image data any more than it should, so what
/// it carries is the node's path for the host to resolve, load and bind.
#[derive(Clone, Debug, PartialEq)]
pub struct TextureSlot {
    /// The `TexImage` node this slot binds for.
    pub node: NodeId,
    /// The image reference the node carried at compile time. Opaque to the
    /// compiler; may be empty while a graph is being authored.
    pub path: String,
    /// The name the slot is exposed under, if any — the texture analogue of an
    /// exposed parameter, in its own namespace.
    pub name: Option<String>,
}

/// Which slots each slot-consuming node was assigned.
///
/// Returned alongside the WGSL by [`crate::emit::emit_with_layout`]. A caller
/// needs it for exactly two things: filling the uniform buffer, in slot order,
/// and binding an image to each texture slot, in slot order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParamLayout {
    slots: Vec<ParamSlot>,
    /// Slot index for each exposed parameter name.
    ///
    /// A `BTreeMap` so that listing the known names — which is what an error
    /// message does when something asks for one that does not exist — comes out
    /// in a stable order rather than a different one each run.
    named: BTreeMap<String, usize>,
    /// Texture bindings in assignment order. No parallel name map: with at
    /// most [`TEXTURE_SLOTS`] entries a scan is not worth an index.
    textures: Vec<TextureSlot>,
}

impl ParamLayout {
    /// Append a node's slots and return the index of the first one.
    ///
    /// The returned base is what the emitter writes into the generated code, so
    /// a block's rows are addressed `base + k` and only the base ever appears in
    /// the source. That is the whole reason editing a coefficient does not
    /// change the shader.
    ///
    /// `name` exposes the block under a name for [`ParamLayout::slot_of`].
    /// Duplicates are rejected by `Graph::validate` before this is reached.
    pub(crate) fn push_block(
        &mut self,
        node: NodeId,
        name: Option<&str>,
        rows: Vec<SlotValue>,
    ) -> usize {
        let base = self.slots.len();
        self.slots
            .extend(rows.into_iter().map(|value| ParamSlot { node, value }));
        if let Some(name) = name {
            self.named.insert(name.to_string(), base);
        }
        base
    }

    /// Record a texture slot and return its index.
    ///
    /// Crate-private like [`ParamLayout::push_block`], and for the same
    /// reason: assignment order is the emitter's contract with the generated
    /// source — the index returned here is baked into a `sg_texture_{k}`
    /// declaration — so nothing outside the emitter may append.
    pub(crate) fn push_texture(&mut self, slot: TextureSlot) -> usize {
        self.textures.push(slot);
        self.textures.len() - 1
    }

    /// Which slot an exposed parameter lives in.
    ///
    /// This is the entire runtime interface for driving a material from outside:
    /// look the name up once when the material is built, then write that slot.
    pub fn slot_of(&self, name: &str) -> Option<usize> {
        self.named.get(name).copied()
    }

    /// Every texture binding, in assignment order. The position in this slice
    /// *is* the slot index the shader's `sg_texture_{k}` declaration carries,
    /// so binding image `k` to slot `k` is all a host has to do.
    pub fn textures(&self) -> &[TextureSlot] {
        &self.textures
    }

    /// Which texture slot an exposed texture name refers to.
    ///
    /// The texture half of [`ParamLayout::slot_of`]: look the name up once
    /// when the material is built, then rebind that slot.
    pub fn texture_slot_of(&self, name: &str) -> Option<usize> {
        self.textures
            .iter()
            .position(|slot| slot.name.as_deref() == Some(name))
    }

    /// The type an exposed parameter expects.
    ///
    /// `None` if the name is unknown. Also `None` for a name attached to a
    /// generated block, which has no single type — nothing can name one today,
    /// and if something ever does, driving it by hand is not what it is for.
    pub fn slot_type(&self, name: &str) -> Option<ValueType> {
        match self.slots.get(self.slot_of(name)?)?.value {
            SlotValue::Param(value) => Some(value.ty()),
            SlotValue::Raw(_) => None,
        }
    }

    /// Overwrite one slot's contents.
    ///
    /// Deliberately narrow: this is how an outside value reaches the buffer, and
    /// the only legitimate caller already knows the slot from a name lookup.
    /// Returns `false` for an out-of-range index rather than panicking, because
    /// the index came from somewhere and the caller deserves to hear about it.
    pub fn set(&mut self, slot: usize, value: SlotValue) -> bool {
        match self.slots.get_mut(slot) {
            Some(existing) => {
                existing.value = value;
                true
            }
            None => false,
        }
    }

    /// Every exposed parameter name, in sorted order.
    ///
    /// For diagnostics. "no input called `wear_lvl`" is a bad message; the same
    /// message listing `wear_level, damage_level` is one you can act on without
    /// opening the graph.
    pub fn exposed_names(&self) -> impl Iterator<Item = &str> {
        self.named.keys().map(String::as_str)
    }

    /// Slots in assignment order. The position in this slice *is* the slot
    /// index the shader reads.
    pub fn slots(&self) -> &[ParamSlot] {
        &self.slots
    }

    /// Number of slots the graph occupies.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether the graph exposes no parameters at all.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Every slot's contents, padded to `vec4` the way the shader expects.
    pub fn packed(&self) -> Vec<[f32; 4]> {
        self.slots.iter().map(|slot| slot.value.packed()).collect()
    }

    /// Whether this layout describes the same graph as `other` — same nodes in
    /// the same slots — ignoring the values.
    ///
    /// The distinction matters: if only values differ, the compiled shader is
    /// byte-identical and the correct response is to upload a uniform. If the
    /// *shape* differs, a recompile happened and the old slot indices mean
    /// nothing.
    ///
    /// Block sizes are covered without a special case: a three-segment ramp
    /// contributes 22 consecutive slots naming the same node, so a ramp that
    /// gained a stop compares unequal on length alone.
    pub fn same_shape(&self, other: &ParamLayout) -> bool {
        // Texture slots compare by count and node only: the compiled source
        // names slots by index, so editing a path — or naming a slot — is a
        // rebind against the same shader, exactly as a param edit is an
        // upload. A texture gained or lost, or slots reshuffled between
        // nodes, is a recompile.
        self.slots.len() == other.slots.len()
            && self
                .slots
                .iter()
                .zip(&other.slots)
                .all(|(a, b)| a.node == b.node && a.value.same_shape(&b.value))
            && self.textures.len() == other.textures.len()
            && self
                .textures
                .iter()
                .zip(&other.textures)
                .all(|(a, b)| a.node == b.node)
    }
}

/// The WGSL expression that reads a slot as a given type.
///
/// Swizzles rather than a conversion, so a float param costs nothing at runtime
/// for having been stored in a `vec4`.
pub fn slot_expr(slot: usize, ty: ValueType) -> String {
    match ty {
        ValueType::Float => format!("params.slots[{slot}].x"),
        ValueType::Vec3 => format!("params.slots[{slot}].xyz"),
        ValueType::Color => format!("params.slots[{slot}]"),
        // Not an oversight, and not something to implement later: a slot is one
        // `vec4` and a surface is ten values, so there is nothing sensible to
        // read here. Reaching this means a `Param` was somehow given a Surface
        // output type, which `NodeKind::outputs` cannot produce — `Value` has no
        // Surface variant to select it.
        ValueType::Surface => unreachable!(
            "a surface cannot live in a parameter slot; \
             it is ten values and a slot is one vec4"
        ),
    }
}

/// The uniform declaration a generated shader carries.
///
/// The bind group is the `#{MATERIAL_BIND_GROUP}` placeholder rather than a
/// number: it is how Bevy's own `pbr_bindings.wgsl` declares material bindings,
/// and it is the spelling that survives the engine renumbering the group (as it
/// did moving from 2 to 3 between Bevy 0.16 and 0.17). naga_oil substitutes it
/// at pipeline build; the headless validator substitutes it in
/// [`crate::standalone::to_standalone_wgsl`], which is the one other place that
/// has to know the placeholder exists.
pub fn uniform_declaration() -> String {
    format!(
        "struct SgParams {{\n    slots: array<vec4<f32>, {PARAM_SLOTS}>,\n}}\n\
         @group(#{{MATERIAL_BIND_GROUP}}) @binding(0) var<uniform> params: SgParams;\n"
    )
}

/// The texture and sampler declarations for one slot.
///
/// Each texture binds its **own** sampler rather than sharing one, because
/// wrap and filter settings ride with the image it samples — a shared sampler
/// would silently take its settings from whichever slot happened to supply it,
/// which is the sort of bug that reads as a blurry texture rather than a
/// binding one.
///
/// Bindings interleave after the uniform at `@binding(0)`: slot `k` is texture
/// `@binding(1 + 2k)`, sampler `@binding(2 + 2k)`. The renderer-side material
/// struct derives its binding attributes from the same arithmetic, so the two
/// cannot disagree without one of them changing this formula.
///
/// Same `#{MATERIAL_BIND_GROUP}` placeholder as [`uniform_declaration`], and
/// substituted in the same one place — see that function's docs.
pub fn texture_declaration(slot: usize) -> String {
    format!(
        "@group(#{{MATERIAL_BIND_GROUP}}) @binding({texture}) var sg_texture_{slot}: texture_2d<f32>;\n\
         @group(#{{MATERIAL_BIND_GROUP}}) @binding({sampler}) var sg_sampler_{slot}: sampler;\n",
        texture = 1 + 2 * slot,
        sampler = 2 + 2 * slot,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;

    /// Two node ids from a real graph, since ids are only meaningful when a
    /// graph issued them.
    fn ids() -> (Graph, NodeId, NodeId) {
        let mut g = Graph::new();
        let a = g.param_float(0.0);
        let b = g.param_float(0.0);
        (g, a, b)
    }

    #[test]
    fn a_float_slot_pads_with_zeros_not_garbage() {
        let (_g, a, b) = ids();
        let mut layout = ParamLayout::default();
        layout.push_block(a, None, vec![SlotValue::Param(Value::Float(0.5))]);
        layout.push_block(
            b,
            None,
            vec![SlotValue::Param(Value::Vec3([1.0, 2.0, 3.0]))],
        );
        assert_eq!(
            layout.packed(),
            vec![[0.5, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 0.0]]
        );
    }

    #[test]
    fn same_shape_ignores_values_but_not_types() {
        let (_g, a, _) = ids();

        let mut x = ParamLayout::default();
        x.push_block(a, None, vec![SlotValue::Param(Value::Float(0.5))]);

        let mut y = ParamLayout::default();
        y.push_block(a, None, vec![SlotValue::Param(Value::Float(9.9))]);
        assert!(x.same_shape(&y), "only the value differs");

        let mut z = ParamLayout::default();
        z.push_block(a, None, vec![SlotValue::Param(Value::Color([0.0; 4]))]);
        assert!(!x.same_shape(&z), "the slot's type differs");
    }

    #[test]
    fn a_block_that_grew_is_a_different_shape() {
        // The structural test for ramps, stated on the layout rather than left
        // implicit in the emitter: more rows for the same node means the
        // generated function changed and the old slot indices are meaningless.
        let (_g, a, _) = ids();

        let mut two = ParamLayout::default();
        two.push_block(a, None, vec![SlotValue::Raw([0.0; 4]); 2]);

        let mut also_two = ParamLayout::default();
        also_two.push_block(a, None, vec![SlotValue::Raw([9.0; 4]); 2]);
        assert!(two.same_shape(&also_two), "only the numbers differ");

        let mut three = ParamLayout::default();
        three.push_block(a, None, vec![SlotValue::Raw([0.0; 4]); 3]);
        assert!(!two.same_shape(&three), "a block gained a row");
    }

    #[test]
    fn base_indices_follow_block_sizes() {
        let (_g, a, b) = ids();
        let mut layout = ParamLayout::default();
        assert_eq!(
            layout.push_block(a, None, vec![SlotValue::Raw([0.0; 4]); 7]),
            0
        );
        assert_eq!(
            layout.push_block(b, Some("wear"), vec![SlotValue::Param(Value::Float(1.0))]),
            7
        );
        assert_eq!(layout.len(), 8);

        // A name resolves to the block's *base*, which for a one-slot param is
        // the slot itself — and for anything larger is where its data starts.
        assert_eq!(layout.slot_of("wear"), Some(7));
        assert_eq!(layout.slot_of("nonexistent"), None);
        assert_eq!(layout.exposed_names().collect::<Vec<_>>(), vec!["wear"]);
    }

    #[test]
    fn slot_expressions_swizzle_by_type() {
        assert_eq!(slot_expr(3, ValueType::Float), "params.slots[3].x");
        assert_eq!(slot_expr(3, ValueType::Vec3), "params.slots[3].xyz");
        assert_eq!(slot_expr(3, ValueType::Color), "params.slots[3]");
    }

    /// The binding arithmetic the renderer-side material struct mirrors.
    ///
    /// Pinned as literal text because the other half of the contract lives in
    /// another crate's attribute macros, where nothing can assert against it.
    #[test]
    fn texture_bindings_interleave_after_the_uniform() {
        let slot0 = texture_declaration(0);
        assert!(slot0.contains("@binding(1) var sg_texture_0: texture_2d<f32>;"));
        assert!(slot0.contains("@binding(2) var sg_sampler_0: sampler;"));

        let slot3 = texture_declaration(3);
        assert!(slot3.contains("@binding(7) var sg_texture_3: texture_2d<f32>;"));
        assert!(slot3.contains("@binding(8) var sg_sampler_3: sampler;"));
    }

    fn tex(node: NodeId, path: &str, name: Option<&str>) -> TextureSlot {
        TextureSlot {
            node,
            path: path.to_string(),
            name: name.map(str::to_string),
        }
    }

    #[test]
    fn texture_names_resolve_to_their_slot() {
        let (_g, a, b) = ids();
        let mut layout = ParamLayout::default();
        layout.push_texture(tex(a, "hull.png", None));
        layout.push_texture(tex(b, "decal.png", Some("insignia")));

        assert_eq!(layout.texture_slot_of("insignia"), Some(1));
        assert_eq!(layout.texture_slot_of("nonexistent"), None);
        assert_eq!(layout.textures().len(), 2);
    }

    /// A path edit is a rebind; a texture gained is a recompile.
    ///
    /// The texture half of the shape/value distinction: the shader names slots
    /// by index only, so nothing about *which image* fills one can change the
    /// compiled source.
    #[test]
    fn a_path_edit_is_the_same_shape_and_a_new_texture_is_not() {
        let (_g, a, b) = ids();

        let mut x = ParamLayout::default();
        x.push_texture(tex(a, "hull.png", None));

        let mut y = ParamLayout::default();
        y.push_texture(tex(a, "repainted.png", Some("skin")));
        assert!(x.same_shape(&y), "only the reference and name differ");
        assert_ne!(x, y, "the edit has to reach the layout, or nothing happens");

        let mut z = ParamLayout::default();
        z.push_texture(tex(a, "hull.png", None));
        z.push_texture(tex(b, "decal.png", None));
        assert!(!x.same_shape(&z), "a texture slot was gained");
    }
}
