//! WGSL emission.
//!
//! The compiler is a single backwards walk from the output node, turning each
//! reached node into one SSA `let` binding. There is no optimizer and no
//! register allocation: WGSL is compiled again downstream by the driver, which
//! is far better at both than anything worth writing here. The only
//! optimization this pass performs is the one the driver cannot — not emitting
//! nodes and support functions that the graph never reaches.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use crate::graph::{Graph, GraphError, NodeId};
use crate::node::{
    Emission, NodeKind, Snippet, SocketDefault, GEOMETRY_UV_OUTPUT, GEOMETRY_VERTEX_COLOR_OUTPUT,
    SURFACE_IN,
};
use crate::params::{
    slot_expr, texture_declaration, uniform_declaration, ParamLayout, TextureSlot, PARAM_SLOTS,
    TEXTURE_SLOTS,
};

/// Bindings the generated fragment function always establishes, and which
/// [`SocketDefault::Context`] expressions and `Geometry` outputs refer to.
///
/// These three come off the mesh unconditionally — every mesh Bevy draws has a
/// position and a normal. The vertex attributes below do not.
const PRELUDE: &str = "    \
let sg_position = in.world_position.xyz;
    let sg_normal = normalize(in.world_normal);
    let sg_view = normalize(view.world_position.xyz - in.world_position.xyz);";

/// UV binding, emitted only when a graph reads it.
///
/// `VERTEX_UVS_A` is pushed from the mesh's own vertex layout
/// (`bevy_pbr::render::mesh`), and `VertexOutput` only declares `uv` under it —
/// so an unguarded `in.uv` is a shader that compiles on a UV-mapped mesh and
/// fails to resolve on everything else. The fallback is zero rather than an
/// error because a material should still draw on a mesh that lacks the channel;
/// an editor is the right place to point out that it will look wrong.
const UV_PRELUDE: &str = "\
#ifdef VERTEX_UVS_A
    let sg_uv = vec3<f32>(in.uv, 0.0);
#else
    let sg_uv = vec3<f32>(0.0);
#endif";

/// Vertex color binding, emitted only when a graph reads it.
///
/// NOTE: The fallback is **white, not black**. Vertex color is conventionally
/// multiplicative, and the channel's most useful payload here is baked
/// occlusion from Blender — which needs no UV unwrap and arrives as `COLOR_0`.
/// A mesh carrying none must read as "unoccluded, untinted", and black would
/// mean every surface without the attribute rendered dark for no visible reason.
const VERTEX_COLOR_PRELUDE: &str = "\
#ifdef VERTEX_COLORS
    let sg_vertex_color = in.color;
#else
    let sg_vertex_color = vec4<f32>(1.0);
#endif";

/// Compile a graph to a complete Bevy-compatible WGSL fragment shader.
///
/// Discards the parameter layout. Correct for a caller that only wants the
/// source — a golden test, a dump to stdout — and wrong for anything that
/// intends to *render* it, which needs [`emit_with_layout`] to know how to fill
/// the uniform.
pub fn emit(graph: &Graph) -> Result<String, GraphError> {
    emit_with_layout(graph).map(|(wgsl, _)| wgsl)
}

/// Compile a graph, and report where each `Param` node's value has to be written
/// at runtime.
///
/// The two results are a pair for a reason: the slot indices are baked into the
/// returned WGSL, so a layout is only meaningful alongside the exact source it
/// was produced with. Keeping them together makes it awkward to hold a stale one.
pub fn emit_with_layout(graph: &Graph) -> Result<(String, ParamLayout), GraphError> {
    graph.validate()?;
    let order = graph.evaluation_order()?;

    // Binding name per node. Built from the node's own index rather than its
    // position in evaluation order, so that adding an unrelated branch to a
    // graph does not renumber every binding downstream of it — which would turn
    // a one-line graph change into a whole-file golden diff.
    let mut names: BTreeMap<NodeId, String> = BTreeMap::new();
    for id in &order {
        names.insert(*id, format!("n{}_{}", id.index(), graph.kind(*id).tag()));
    }

    // Slots are assigned in evaluation order, so the same graph always produces
    // the same assignment — which is what lets a caller compare two layouts and
    // conclude that only values changed.
    //
    // Collected before any are placed so the overflow message can report the
    // real total rather than "it ran out at node seventeen".
    let mut blocks: Vec<(NodeId, Vec<crate::params::SlotValue>)> = Vec::new();
    let mut needed = 0usize;
    for id in &order {
        let rows = graph.kind(*id).slots();
        if rows.is_empty() {
            continue;
        }
        needed += rows.len();
        blocks.push((*id, rows));
    }
    if needed > PARAM_SLOTS {
        return Err(GraphError::TooManyParams {
            needed,
            limit: PARAM_SLOTS,
        });
    }

    // Base slot per slot-consuming node. Not derived from the layout afterwards:
    // a node owning several consecutive slots would collapse to whichever one
    // was seen last, which is the *end* of its block rather than the start.
    let mut layout = ParamLayout::default();
    let mut bases: BTreeMap<NodeId, usize> = BTreeMap::new();
    for (id, rows) in blocks {
        let base = layout.push_block(id, graph.kind(id).param_name(), rows);
        bases.insert(id, base);
    }

    // Texture slots, assigned in evaluation order like uniform slots and for
    // the same reason: the same graph must always bind the same image to the
    // same index, or comparing two layouts would mean nothing. Counted before
    // any are placed so the overflow message reports the real total.
    let tex_nodes: Vec<NodeId> = order
        .iter()
        .copied()
        .filter(|id| matches!(graph.kind(*id), NodeKind::TexImage { .. }))
        .collect();
    if tex_nodes.len() > TEXTURE_SLOTS {
        return Err(GraphError::TooManyTextures {
            needed: tex_nodes.len(),
            limit: TEXTURE_SLOTS,
        });
    }
    let mut tex_slots: BTreeMap<NodeId, usize> = BTreeMap::new();
    for id in tex_nodes {
        let NodeKind::TexImage { path, name } = graph.kind(id) else {
            unreachable!("tex_nodes was filtered to TexImage nodes");
        };
        let slot = layout.push_texture(TextureSlot {
            node: id,
            path: path.clone(),
            name: name.clone(),
        });
        tex_slots.insert(id, slot);
    }

    let mut needed: BTreeSet<Snippet> = BTreeSet::new();
    for id in &order {
        needed.extend(graph.kind(*id).snippets());
    }
    // Emitted in `Snippet::ALL` order rather than set order, because that order
    // is dependency order and WGSL requires a function to be declared before
    // use. Relying on the set's own ordering would work only for as long as the
    // enum happens to be declared dependency-first.
    let snippets: Vec<Snippet> = Snippet::ALL
        .iter()
        .copied()
        .filter(|s| needed.contains(s))
        .collect();

    let mut body = String::new();
    for id in &order {
        let kind = graph.kind(*id);
        match kind.emission() {
            // Reads the prelude directly; nothing to bind.
            Emission::Inline(_) => {}
            Emission::Param => {
                let NodeKind::Param { value, .. } = kind else {
                    unreachable!("only Param reads a parameter slot");
                };
                // Still bound to a name rather than read inline, so that a param
                // feeding several sockets is one buffer read instead of several
                // and the generated body stays readable.
                let _ = writeln!(
                    body,
                    "    let {} = {};",
                    names[id],
                    slot_expr(bases[id], value.ty())
                );
            }
            Emission::Texture => {
                // Written here rather than by `NodeKind::expr`, because the
                // slot index is emitter state — the same division of labor
                // as a Param's slot read. Only `.xy` of the coordinate
                // reaches the sampler; the third component exists so every
                // vector-producing node can feed a texture directly.
                let args = input_exprs(graph, *id, &names);
                let slot = tex_slots[id];
                let _ = writeln!(
                    body,
                    "    let {} = textureSample(sg_texture_{slot}, sg_sampler_{slot}, ({}).xy);",
                    names[id], args[0]
                );
            }
            Emission::Expr | Emission::Struct => {
                let args = input_exprs(graph, *id, &names);
                let _ = writeln!(body, "    let {} = {};", names[id], kind.expr(&args));
            }
            Emission::Function => {
                let args = input_exprs(graph, *id, &names);
                let _ = writeln!(
                    body,
                    "    let {} = {}({});",
                    names[id],
                    function_name(*id, kind),
                    args.join(", ")
                );
            }
            Emission::Root => {
                body.push_str(&emit_root(graph, *id, &names));
            }
        }
    }

    let mut out = String::new();
    out.push_str(HEADER);
    // Unconditional for the same reason as the uniform below: every graph
    // terminates in a `Surface`, so gating this would be a condition that is
    // always true. Generated from the socket table rather than written out here,
    // so the struct and the node that constructs it cannot disagree about field
    // order — and a positional constructor makes disagreement silent.
    out.push('\n');
    out.push_str(&surface_struct());
    // Declared unconditionally, even by a graph with no params. The material's
    // bind group always carries the buffer, and emitting the declaration either
    // way keeps every generated shader the same shape — which matters when the
    // things being read are golden-file diffs.
    out.push('\n');
    out.push_str(&uniform_declaration());
    // One declaration pair per texture the graph actually samples —
    // conditional like the attribute preludes, and unlike the uniform above:
    // the uniform is on every material's bind group unconditionally, but a
    // texture binding only exists where the renderer has an image field to
    // fill it, so a texture-free graph must not declare one. It also keeps
    // the majority of generated shaders byte-identical to what they were
    // before textures existed.
    for slot in 0..tex_slots.len() {
        out.push('\n');
        out.push_str(&texture_declaration(slot));
    }
    for snippet in &snippets {
        out.push('\n');
        out.push_str(snippet.source());
    }
    // Per-node functions come after the shared snippets, because a generated
    // function may call one and WGSL requires declaration before use. Emitted in
    // evaluation order so the output is stable run to run.
    for id in &order {
        let kind = graph.kind(*id);
        // Not every function-generating node carries uniform data: a color ramp
        // reads its coefficients from slots, while an unrolled noise node's code
        // depends only on an octave count it holds itself. Nodes with no block
        // ignore the base, so defaulting is correct rather than a fallback —
        // skipping them here would silently drop their function instead.
        let base = bases.get(id).copied().unwrap_or(0);
        if let Some(function) = kind.function(&function_name(*id, kind), base) {
            out.push('\n');
            out.push_str(&function);
        }
    }
    let prelude = prelude(graph, &order);
    let _ = write!(
        out,
        "\n{FRAGMENT_OPEN}\n{prelude}\n\n{CONTEXT}\n\n{body}{FRAGMENT_CLOSE}"
    );
    Ok((out, layout))
}

/// The `SgSurface` declaration, generated from the `Surface` node's socket table.
///
/// Generated rather than written as a literal because the `Surface` node builds
/// this struct with a **positional** constructor. If the field order here and the
/// socket order there ever diverged, the result would still be valid WGSL
/// whenever two adjacent fields shared a type — metallic and roughness, say —
/// and the only symptom would be a material that is inexplicably shiny. Deriving
/// both from one table makes that failure impossible rather than unlikely.
fn surface_struct() -> String {
    let mut s = String::from("struct SgSurface {\n");
    for socket in SURFACE_IN {
        let _ = writeln!(s, "    {}: {},", socket.ident, socket.ty.wgsl());
    }
    s.push_str("}\n");
    s
}

/// The fragment prelude, with conditional attribute bindings appended for the
/// attributes this graph actually reads.
///
/// Emitting them unconditionally would work and would be simpler. It is not
/// done because it would put a `#ifdef` in every generated shader — including
/// the majority that never touch a UV — which means every shader would have to
/// be validated twice, and every golden file would carry a branch that is dead
/// for it. Generated code should contain what the graph asked for.
/// Asked of the *links* rather than of the Geometry node's presence: a graph can
/// carry a Geometry node for its normal alone, and binding a UV it never reads
/// would emit a conditional for nothing.
fn prelude(graph: &Graph, order: &[NodeId]) -> String {
    let mut s = String::from(PRELUDE);
    if graph.consumes_geometry_output(order, GEOMETRY_UV_OUTPUT) {
        s.push('\n');
        s.push_str(UV_PRELUDE);
    }
    if graph.consumes_geometry_output(order, GEOMETRY_VERTEX_COLOR_OUTPUT) {
        s.push('\n');
        s.push_str(VERTEX_COLOR_PRELUDE);
    }
    s
}

/// Name of the function a [`Emission::Function`] node generates.
///
/// Built from the node's index for the same reason binding names are: an
/// unrelated edit elsewhere in the graph must not renumber it and turn a
/// one-node change into a whole-file golden diff.
fn function_name(id: NodeId, kind: &NodeKind) -> String {
    format!("sg_n{}_{}", id.index(), kind.tag())
}

/// Resolve every input socket of a node to a WGSL expression, in socket order.
fn input_exprs(graph: &Graph, node: NodeId, names: &BTreeMap<NodeId, String>) -> Vec<String> {
    graph
        .kind(node)
        .inputs()
        .iter()
        .enumerate()
        .map(|(idx, socket)| match graph.source_of(node, idx) {
            Some((src, src_socket)) => output_expr(graph, src, src_socket, names),
            None => match socket.default {
                SocketDefault::Value(v) => v.wgsl_literal(),
                SocketDefault::Context(expr) => expr.to_string(),
                // `Graph::validate` runs first and rejects this, so reaching it
                // means the two have drifted apart rather than that a user did
                // something wrong.
                SocketDefault::Required => unreachable!(
                    "required socket `{}.{}` unwired despite validation",
                    graph.kind(node).name(),
                    socket.name
                ),
            },
        })
        .collect()
}

/// The expression that reads one output socket of an already-emitted node.
fn output_expr(
    graph: &Graph,
    node: NodeId,
    socket: usize,
    names: &BTreeMap<NodeId, String>,
) -> String {
    let kind = graph.kind(node);
    match kind.emission() {
        Emission::Inline(exprs) => exprs[socket].to_string(),
        Emission::Param | Emission::Expr => names[&node].clone(),
        // A generated function returns exactly one value. With a single output
        // socket that value *is* the output; with several, each is read off it
        // by swizzle using the socket's ident — the same contract `Struct` uses,
        // and the reason `SocketDef::ident` is required to be WGSL-safe.
        //
        // Gated on the count rather than applied always, because a single-output
        // function returns a scalar and there is no swizzle that means "all of
        // an f32". `ColorRamp` is the node this exists for.
        Emission::Function if kind.outputs().len() > 1 => {
            format!("{}.{}", names[&node], kind.outputs()[socket].ident)
        }
        Emission::Function => names[&node].clone(),
        // The binding is the sampled vec4; each output reads off it by
        // swizzle, using idents pinned by `TEX_IMAGE_OUT` — the same contract
        // a multi-output Function uses.
        Emission::Texture => format!("{}.{}", names[&node], kind.outputs()[socket].ident),
        Emission::Struct => format!("{}.{}", names[&node], kind.outputs()[socket].ident),
        Emission::Root => unreachable!("the output node has no outputs to read"),
    }
}

/// The terminating block: graph results into `PbrInput`, then lighting.
fn emit_root(graph: &Graph, node: NodeId, names: &BTreeMap<NodeId, String>) -> String {
    // One input now — the surface — so the ten properties are read as fields off
    // it rather than resolved as ten separate socket expressions. Field names
    // come from `SURFACE_IN`'s idents, the same table `surface_struct` declares.
    let args = input_exprs(graph, node, names);
    let surface = &args[0];
    let field = |ident: &str| format!("{surface}.{ident}");
    let (
        base_color,
        metallic,
        roughness,
        reflectance,
        emissive,
        occlusion,
        normal,
        aniso_strength,
        aniso_tangent,
        alpha,
    ) = (
        field("base_color"),
        field("metallic"),
        field("perceptual_roughness"),
        field("reflectance"),
        field("emissive"),
        field("occlusion"),
        field("normal"),
        field("anisotropy_strength"),
        field("anisotropy_tangent"),
        field("alpha"),
    );

    let mut s = String::new();
    s.push_str("\n    // --- surface properties ---\n");
    let _ = writeln!(s, "    pbr_input.material.base_color = {base_color};");
    let _ = writeln!(s, "    pbr_input.material.base_color.a = {alpha};");
    let _ = writeln!(s, "    pbr_input.material.metallic = {metallic};");
    let _ = writeln!(
        s,
        "    pbr_input.material.perceptual_roughness = {roughness};"
    );
    let _ = writeln!(
        s,
        "    pbr_input.material.reflectance = vec3<f32>({reflectance});"
    );
    let _ = writeln!(s, "    pbr_input.material.emissive = {emissive};");
    // Occlusion is a vec3 on PbrInput because it can be tinted; a scalar socket
    // broadcasts to grey, which is what every non-exotic AO source produces.
    let _ = writeln!(
        s,
        "    pbr_input.diffuse_occlusion = vec3<f32>({occlusion});"
    );
    // Re-normalised because a graph is free to compute a normal by any means,
    // and lighting on an un-normalised normal is wrong in a way that reads as a
    // material problem rather than a graph one.
    let _ = writeln!(s, "    pbr_input.N = normalize({normal});");

    s.push_str("\n    // --- anisotropy ---\n");
    let _ = writeln!(s, "    pbr_input.anisotropy_strength = {aniso_strength};");
    // The tangent is rejected against the shading normal before use. A brush
    // direction derived from surrounding geometry (a Voronoi cell center, say)
    // is not tangent to the surface in general, and the renderer assumes an
    // orthonormal (N, T, B) frame — feeding it a tangent with a normal-parallel
    // component skews the highlight off the brush direction.
    let _ = writeln!(
        s,
        "    let sg_aniso_t = sg_reject_root({aniso_tangent}, pbr_input.N);"
    );
    // Guard the degenerate case: if the tangent is zero, or exactly parallel to
    // the normal, the rejection is zero-length and normalising it yields NaN.
    let _ = writeln!(s, "    if (dot(sg_aniso_t, sg_aniso_t) > 1e-12) {{");
    let _ = writeln!(s, "        pbr_input.anisotropy_T = normalize(sg_aniso_t);");
    let _ = writeln!(
        s,
        "        pbr_input.anisotropy_B = normalize(cross(pbr_input.N, pbr_input.anisotropy_T));"
    );
    let _ = writeln!(s, "    }} else {{");
    let _ = writeln!(s, "        pbr_input.anisotropy_strength = 0.0;");
    let _ = writeln!(s, "    }}");

    s.push_str(
        "
    var out = apply_pbr_lighting(pbr_input);
    out = main_pass_post_lighting_processing(pbr_input, out);
    return out;
",
    );
    s
}

const HEADER: &str = "\
// GENERATED BY shadergraph — DO NOT EDIT.
//
// Edit the graph and recompile. Any change made here is lost the next time the
// graph is emitted.

#import bevy_pbr::forward_io::VertexOutput
#import bevy_pbr::pbr_types::{PbrInput, pbr_input_new}
#import bevy_pbr::pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing}
#import bevy_pbr::mesh_view_bindings::view

// Always emitted, and named apart from the optional `sg_reject` snippet so the
// two never collide: the output node uses this to flatten the anisotropy
// tangent into the surface regardless of what the graph contains.
fn sg_reject_root(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    let denom = dot(b, b);
    if (denom < 1e-12) {
        return a;
    }
    return a - b * (dot(a, b) / denom);
}
";

const FRAGMENT_OPEN: &str = "\
@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    var pbr_input = pbr_input_new();
";

/// Fragment context that the renderer needs regardless of the graph's contents.
const CONTEXT: &str = "    \
pbr_input.frag_coord = in.position;
    pbr_input.world_position = in.world_position;
    pbr_input.world_normal = sg_normal;
    pbr_input.N = sg_normal;
    pbr_input.V = sg_view;
    // `clip_from_view[3][3] == 1.0` is the documented orthographic test.
    pbr_input.is_orthographic = view.clip_from_view[3][3] == 1.0;";

const FRAGMENT_CLOSE: &str = "}\n";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{BlendMode, MathOp};
    use crate::ramp::{ColorRamp, ColorStop, RampInterp};

    /// A minimal graph whose base color comes from `ramp`.
    fn graph_with(ramp: ColorRamp) -> Graph {
        let mut g = Graph::new();
        let factor = g.param_float(0.5);
        let node = g.add(NodeKind::ColorRamp(ramp));
        let (surface, _) = g.surface_output();
        g.link(factor, "Value", node, "Factor").unwrap();
        g.link(node, "Color", surface, "Base Color").unwrap();
        g
    }

    fn three_stops() -> ColorRamp {
        ColorRamp {
            stops: vec![
                ColorStop::new(0.0, [0.9, 0.1, 0.1, 1.0]),
                ColorStop::new(0.4, [0.1, 0.8, 0.2, 1.0]),
                ColorStop::new(1.0, [0.2, 0.2, 0.9, 1.0]),
            ],
            interp: RampInterp::QuinticSpline,
        }
    }

    /// The claim `ramp-live-params` exists to make, asserted end to end.
    ///
    /// `ramp::tests` checks the generated *function* is stable; this checks the
    /// whole compile is — that nothing between the ramp and the finished shader
    /// smuggles a stop position back into the source.
    #[test]
    fn moving_a_ramp_stop_leaves_the_shader_identical() {
        let (before, before_layout) = emit_with_layout(&graph_with(three_stops())).unwrap();

        let mut moved = three_stops();
        moved.stops[1].position = 0.8;
        moved.stops[1].color = [1.0, 1.0, 0.0, 1.0];
        let (after, after_layout) = emit_with_layout(&graph_with(moved)).unwrap();

        assert_eq!(after, before, "a stop edit reached the source");
        assert!(
            after_layout.same_shape(&before_layout),
            "same shader, different slot shape — the emitter and layout disagree"
        );
        assert_ne!(
            after_layout, before_layout,
            "the edit has to reach the uniform, or nothing happens at all"
        );
    }

    #[test]
    fn adding_a_ramp_stop_recompiles() {
        let (before, before_layout) = emit_with_layout(&graph_with(three_stops())).unwrap();

        let mut grown = three_stops();
        grown.insert(ColorStop::new(0.7, [0.0, 0.0, 0.0, 1.0]));
        let (after, after_layout) = emit_with_layout(&graph_with(grown)).unwrap();

        assert_ne!(after, before, "a new segment must change the function");
        assert!(!after_layout.same_shape(&before_layout));
    }

    /// Naming a parameter is metadata, not code.
    ///
    /// The name never reaches the shader — the slot index does — so exposing one
    /// must leave the compiled source untouched. If it ever does not, then
    /// renaming an input in the editor would rebuild a pipeline, and worse, two
    /// materials differing only in what their inputs are called would stop
    /// sharing a program.
    #[test]
    fn exposing_a_parameter_changes_the_layout_and_not_the_shader() {
        let anonymous = crate::materials::hull_metal().unwrap();
        let (before, before_layout) = emit_with_layout(&anonymous).unwrap();

        let mut named = crate::materials::hull_metal().unwrap();
        // The first Param in the graph, whichever it is — this test is about
        // naming, not about which socket got named.
        let first_param = named
            .nodes()
            .find(|(_, kind)| matches!(kind, NodeKind::Param { .. }))
            .map(|(id, _)| id)
            .expect("hull_metal has parameters");
        named.expose(first_param, "tuning").unwrap();

        let (after, after_layout) = emit_with_layout(&named).unwrap();
        assert_eq!(after, before, "a parameter's name reached the source");
        assert!(after_layout.slot_of("tuning").is_some());
        assert!(before_layout.slot_of("tuning").is_none());
    }

    #[test]
    fn two_parameters_cannot_share_a_name() {
        let mut g = Graph::new();
        let a = g.param_float(0.0);
        let b = g.param_float(1.0);
        let sum = g.add(NodeKind::Math(MathOp::Add));
        let (surface, _) = g.surface_output();
        g.link(a, "Value", sum, "Value").unwrap();
        g.link(b, "Value", sum, "Value_001").unwrap();
        g.link(sum, "Value", surface, "Metallic").unwrap();

        g.expose(a, "wear").unwrap();
        g.expose(b, "wear").unwrap();

        match g.validate() {
            Err(GraphError::DuplicateParamName { name }) => assert_eq!(name, "wear"),
            other => panic!("expected DuplicateParamName, got {other:?}"),
        }
    }

    #[test]
    fn only_a_param_can_be_exposed() {
        let mut g = Graph::new();
        let node = g.add(NodeKind::Math(MathOp::Add));
        assert!(matches!(
            g.expose(node, "nope"),
            Err(GraphError::NotAParam { node: "Math" })
        ));
    }

    /// A base-color chain of `paths.len()` texture reads, all sharing one
    /// coordinate. Mixed pairwise so every node is reachable from the output.
    fn graph_with_textures(paths: &[&str]) -> Graph {
        let mut g = Graph::new();
        let geometry = g.add(NodeKind::Geometry);
        let (surface, _) = g.surface_output();
        let mut acc: Option<NodeId> = None;
        for path in paths {
            let tex = g.add(NodeKind::tex_image(*path));
            g.link(geometry, "Position", tex, "Vector").unwrap();
            acc = Some(match acc {
                None => tex,
                Some(prev) => {
                    let mix = g.add(NodeKind::MixColor(BlendMode::Mix));
                    // By index because the color output is socket 0 on both a
                    // TexImage ("Color") and a MixColor ("Result").
                    g.link_by_index(prev, 0, mix, 1).unwrap();
                    g.link_by_index(tex, 0, mix, 2).unwrap();
                    mix
                }
            });
        }
        let last = acc.expect("at least one texture");
        g.link_by_index(last, 0, surface, 0).unwrap();
        g
    }

    /// The claim the whole texture design rests on: a path is a *reference*,
    /// and editing one is a rebind, never a recompile.
    #[test]
    fn editing_a_texture_path_leaves_the_shader_identical() {
        let (before, before_layout) =
            emit_with_layout(&graph_with_textures(&["hull.png"])).unwrap();
        let (after, after_layout) =
            emit_with_layout(&graph_with_textures(&["repainted.png"])).unwrap();

        assert_eq!(after, before, "a path edit reached the source");
        assert!(
            after_layout.same_shape(&before_layout),
            "same shader, different slot shape — the emitter and layout disagree"
        );
        assert_ne!(
            after_layout, before_layout,
            "the edit has to reach the layout, or nothing happens at all"
        );
    }

    /// Slot order follows evaluation order, and the generated source uses
    /// exactly the slots the layout reports.
    #[test]
    fn texture_slots_follow_evaluation_order() {
        let (wgsl, layout) = emit_with_layout(&graph_with_textures(&["a.png", "b.png"])).unwrap();

        let paths: Vec<&str> = layout.textures().iter().map(|t| t.path.as_str()).collect();
        assert_eq!(paths, ["a.png", "b.png"]);

        for slot in 0..2 {
            assert!(
                wgsl.contains(&format!("var sg_texture_{slot}: texture_2d<f32>;")),
                "slot {slot} was assigned but never declared"
            );
            assert!(
                wgsl.contains(&format!(
                    "textureSample(sg_texture_{slot}, sg_sampler_{slot},"
                )),
                "slot {slot} was declared but never sampled"
            );
        }
        assert!(
            !wgsl.contains("sg_texture_2"),
            "a slot was declared for a texture that does not exist"
        );
    }

    /// Dead-code tolerance extends to textures: an unreachable `TexImage`
    /// consumes no slot and emits no binding — otherwise an experiment left
    /// hanging off to the side would occupy one of only four slots, and its
    /// unwired coordinate would fail validation for a node that draws nothing.
    #[test]
    fn an_unreachable_tex_image_consumes_no_slot() {
        let mut g = graph_with_textures(&["used.png"]);
        g.add(NodeKind::tex_image("orphan.png"));

        let (wgsl, layout) = emit_with_layout(&g).unwrap();
        assert_eq!(layout.textures().len(), 1);
        assert_eq!(layout.textures()[0].path, "used.png");
        assert!(!wgsl.contains("sg_texture_1"));
    }

    #[test]
    fn running_out_of_texture_slots_reports_how_many_were_wanted() {
        let over = ["a.png", "b.png", "c.png", "d.png", "e.png"];
        assert!(over.len() > TEXTURE_SLOTS, "the test premise");

        match emit_with_layout(&graph_with_textures(&over)) {
            Err(GraphError::TooManyTextures { needed, limit }) => {
                assert_eq!(needed, over.len());
                assert_eq!(limit, TEXTURE_SLOTS);
            }
            other => panic!("expected TooManyTextures, got {other:?}"),
        }
    }

    #[test]
    fn two_textures_cannot_share_a_name() {
        // Built directly rather than through the helper: the graph is
        // append-only, so the duplicate names have to go in at construction.
        let mut g = Graph::new();
        let geometry = g.add(NodeKind::Geometry);
        let (surface, _) = g.surface_output();
        let named = |path: &str| NodeKind::TexImage {
            path: path.to_string(),
            name: Some("skin".to_string()),
        };
        let a = g.add(named("a.png"));
        let b = g.add(named("b.png"));
        let mix = g.add(NodeKind::MixColor(BlendMode::Mix));
        g.link(geometry, "Position", a, "Vector").unwrap();
        g.link(geometry, "Position", b, "Vector").unwrap();
        g.link(a, "Color", mix, "A").unwrap();
        g.link(b, "Color", mix, "B").unwrap();
        g.link(mix, "Result", surface, "Base Color").unwrap();

        match g.validate() {
            Err(GraphError::DuplicateTextureName { name }) => assert_eq!(name, "skin"),
            other => panic!("expected DuplicateTextureName, got {other:?}"),
        }

        // Two *anonymous* reads of different images are ordinary — only
        // exposed names collide.
        assert!(graph_with_textures(&["a.png", "b.png"]).validate().is_ok());
    }

    /// Params and textures are separate namespaces, on purpose: one is driven
    /// by writing a uniform slot, the other by rebinding an image, so the
    /// names can never race each other.
    #[test]
    fn a_texture_and_a_param_may_share_a_name() {
        let mut g = Graph::new();
        let geometry = g.add(NodeKind::Geometry);
        let (surface, _) = g.surface_output();
        let tex = g.add(NodeKind::TexImage {
            path: "a.png".to_string(),
            name: Some("skin".to_string()),
        });
        let wear = g.param_float(0.5);
        g.expose(wear, "skin").unwrap();
        g.link(geometry, "Position", tex, "Vector").unwrap();
        g.link(tex, "Color", surface, "Base Color").unwrap();
        g.link(wear, "Value", surface, "Metallic").unwrap();

        assert!(g.validate().is_ok());
        let (_, layout) = emit_with_layout(&g).unwrap();
        assert!(layout.slot_of("skin").is_some());
        assert_eq!(layout.texture_slot_of("skin"), Some(0));
    }

    #[test]
    fn running_out_of_slots_reports_how_many_were_wanted() {
        // A chain of adds, one Param each, so every node is reachable from the
        // output and therefore actually assigned a slot.
        let mut g = Graph::new();
        let mut acc = g.param_float(0.0);
        for i in 0..=PARAM_SLOTS {
            let p = g.param_float(i as f32);
            let m = g.add(NodeKind::Math(MathOp::Add));
            g.link(acc, "Value", m, "Value").unwrap();
            g.link(p, "Value", m, "Value_001").unwrap();
            acc = m;
        }
        let (surface, _) = g.surface_output();
        g.link(acc, "Value", surface, "Metallic").unwrap();

        match emit_with_layout(&g) {
            Err(GraphError::TooManyParams { needed, limit }) => {
                assert_eq!(limit, PARAM_SLOTS);
                assert_eq!(needed, PARAM_SLOTS + 2, "one seed param plus the loop");
            }
            other => panic!("expected TooManyParams, got {other:?}"),
        }
    }
}
