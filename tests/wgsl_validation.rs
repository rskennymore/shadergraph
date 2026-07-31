//! Parse and type-check every emitted shader, without a GPU.
//!
//! A code generator's characteristic failure is emitting something that is not
//! valid in the target language — a missing `.field`, a `vec3` where a `vec4`
//! belongs, a name that does not resolve. Left to runtime, those surface as a
//! wall of driver diagnostics pointing into generated source the author never
//! wrote, usually while wearing a headset.
//!
//! # What the stub is, and what it is not
//!
//! See [`shadergraph::standalone`], which owns the stub and the import
//! stripping. It lives in the library rather than here because the editor needs
//! to validate a graph before swapping a shader in, and a second copy of a
//! transcribed Bevy interface is a second copy to keep in step.

use shadergraph::standalone::{numbered, to_standalone_wgsl, BEVY_STUB, MESH_ATTRIBUTE_DEFS};
use shadergraph::{emit, materials, Graph};

/// Check a graph compiles on a mesh that has every optional vertex attribute
/// *and* on one that has none.
///
/// Both, always, not just the branch the preview happens to exercise. A graph
/// reading UVs produces different source on a mesh without a UV channel, and
/// checking one branch is how that difference stays invisible until a hull that
/// was never unwrapped renders as a black screen and a driver log.
fn validate(name: &str, graph: &Graph) {
    let generated = emit(graph).unwrap_or_else(|e| panic!("{name} failed to emit: {e}"));
    for (mesh, defs) in [
        ("a fully-attributed mesh", MESH_ATTRIBUTE_DEFS),
        ("a mesh with no UVs or vertex colors", &[][..]),
    ] {
        validate_one(name, mesh, &generated, defs);
    }
}

fn validate_one(name: &str, mesh: &str, generated: &str, defs: &[&str]) {
    let source = to_standalone_wgsl(generated, defs)
        .unwrap_or_else(|e| panic!("`{name}` has unbalanced preprocessor directives: {e}"));

    let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|e| {
        panic!(
            "generated WGSL for `{name}` on {mesh} does not parse:\n{}\n\n--- source ---\n{}",
            e.emit_to_string(&source),
            numbered(&source)
        )
    });

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );

    validator.validate(&module).unwrap_or_else(|e| {
        panic!(
            "generated WGSL for `{name}` on {mesh} does not type-check:\n{}\n\n--- source ---\n{}",
            e.emit_to_string(&source),
            numbered(&source)
        )
    });
}

/// Prove the harness can fail.
///
/// A validation test that passes unconditionally is worse than none, because it
/// reads as coverage. This feeds the same pipeline a type error — returning a
/// `vec3` from a function declared to return `f32` — and asserts it is caught.
#[test]
fn the_validator_actually_rejects_invalid_wgsl() {
    // The stub carries `#ifdef`s of its own now, so it has to be resolved before
    // naga sees it — otherwise this would "pass" by rejecting the directives
    // rather than the type error it is meant to be probing.
    let stub = shadergraph::standalone::resolve_ifdefs(BEVY_STUB, MESH_ATTRIBUTE_DEFS).unwrap();
    let bad = format!("{stub}\nfn broken() -> f32 {{ return vec3<f32>(1.0); }}\n");

    let rejected = match naga::front::wgsl::parse_str(&bad) {
        Err(_) => true,
        Ok(module) => naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .is_err(),
    };

    assert!(
        rejected,
        "the WGSL validation harness accepted a known type error, \
         so it would not catch a real codegen bug either"
    );
}

#[test]
fn hull_metal_is_valid_wgsl() {
    validate("hull_metal", &materials::hull_metal().unwrap());
}

#[test]
fn hull_dark_is_valid_wgsl() {
    validate("hull_dark", &materials::hull_dark().unwrap());
}

#[test]
fn copper_accent_is_valid_wgsl() {
    validate("copper_accent", &materials::copper_accent().unwrap());
}

#[test]
fn scorched_plating_is_valid_wgsl() {
    validate("scorched_plating", &materials::scorched_plating().unwrap());
}

#[test]
fn glass_is_valid_wgsl() {
    validate(
        "glass",
        &materials::glass([0.1, 0.12, 0.2, 1.0], 0.3).unwrap(),
    );
}

#[test]
fn worn_paint_is_valid_wgsl() {
    validate("worn_paint", &materials::worn_paint().unwrap());
}

/// Every gradient form, each in its own shader.
///
/// One graph per form rather than one graph using all seven, because the point
/// is that *each* emits something valid — a single graph would pull the whole
/// snippet in once and prove only that the file parses.
#[test]
fn every_gradient_form_emits_valid_wgsl() {
    use shadergraph::{GradientType, NodeKind};

    for &form in GradientType::ALL {
        let mut graph = Graph::new();
        let gradient = graph.add(NodeKind::Gradient(form));
        let (surface, _out) = graph.surface_output();
        graph.link(gradient, "Value", surface, "Metallic").unwrap();

        let generated = emit(&graph).expect("a gradient graph emits");
        assert!(
            generated.contains("fn sg_gradient_"),
            "{} pulled in no gradient function:\n{generated}",
            form.label()
        );

        validate(&format!("gradient {}", form.label()), &graph);
    }
}

/// A ramp driving a color and a scalar at once, from one evaluation.
///
/// The Alpha output turns the node into an arbitrary 1D curve alongside its
/// gradient. Both sockets read off the *same* generated call, so this also pins
/// that the swizzle-based multi-output path emits something valid rather than,
/// say, calling the function twice.
#[test]
fn a_color_ramp_drives_color_and_a_scalar_together() {
    use shadergraph::{ColorRamp, ColorStop, NodeKind, RampInterp};

    let ramp = ColorRamp {
        stops: vec![
            ColorStop::new(0.0, [0.1, 0.1, 0.12, 0.2]),
            ColorStop::new(1.0, [0.8, 0.75, 0.7, 0.9]),
        ],
        interp: RampInterp::Linear,
    };

    let mut graph = Graph::new();
    let gradient = graph.add(NodeKind::Gradient(shadergraph::GradientType::Linear));
    let node = graph.add(NodeKind::ColorRamp(ramp));
    let (surface, _out) = graph.surface_output();

    graph.link(gradient, "Value", node, "Factor").unwrap();
    graph.link(node, "Color", surface, "Base Color").unwrap();
    graph
        .link(node, "Alpha", surface, "Perceptual Roughness")
        .unwrap();

    let generated = emit(&graph).expect("a two-output ramp emits");
    assert_eq!(
        generated.matches("sg_n1_ramp(").count(),
        2,
        "the ramp should be declared once and called once:\n{generated}"
    );
    // Both sockets read off the one binding, by swizzle.
    assert!(generated.contains("n1_ramp.rgba"), "{generated}");
    assert!(generated.contains("n1_ramp.a"), "{generated}");

    validate("ramp color and alpha", &graph);
}

/// A bare gradient works with nothing wired into it.
///
/// Its Vector input falls back to world position, which is what makes the node
/// worth dropping on a canvas to see what it does. If that default ever became
/// `Required`, the node would arrive broken.
#[test]
fn an_unwired_gradient_still_compiles() {
    use shadergraph::{GradientType, NodeKind};

    let mut graph = Graph::new();
    let gradient = graph.add(NodeKind::Gradient(GradientType::Spherical));
    let (surface, _out) = graph.surface_output();
    graph.link(gradient, "Value", surface, "Metallic").unwrap();

    let generated = emit(&graph).expect("an unwired gradient emits");
    assert!(generated.contains("sg_position"), "{generated}");
    validate("unwired gradient", &graph);
}

/// Two complete materials, one mask — the thing `layered-surface` is for.
///
/// Worth validating rather than assuming, because this is the only generated
/// code that passes a **struct** across a function boundary. Everything else the
/// compiler emits is scalars and vectors, which naga would accept even if the
/// declaration and the constructor disagreed about field order; a struct
/// argument is the one construction where a mismatch is a hard type error rather
/// than a silently wrong material.
#[test]
fn mixed_surfaces_emit_valid_wgsl() {
    use shadergraph::{MathOp, NodeKind, NoiseOctaves};

    let mut graph = Graph::new();
    let geometry = graph.add(NodeKind::Geometry);
    let mask = graph.add(NodeKind::Noise(NoiseOctaves::Live));
    let sharpen = graph.add(NodeKind::Math(MathOp::SmoothStep));

    // Two layers that differ in more than color, so the mix has to carry
    // several properties rather than degenerating into a MixColor.
    let paint = graph.add(NodeKind::Surface);
    let metal = graph.add(NodeKind::Surface);
    let paint_color = graph.param_color([0.16, 0.17, 0.19, 1.0]);
    let metal_color = graph.param_color([0.55, 0.56, 0.58, 1.0]);
    let metal_metallic = graph.param_float(1.0);

    let mix = graph.add(NodeKind::MixSurface);
    // Not `surface_output()` here: that helper builds the ordinary one-surface
    // root, and the whole point of this graph is that the root takes a mix.
    let out = graph.add(NodeKind::PbrOutput);

    graph.link(geometry, "Position", mask, "Vector").unwrap();
    graph.link(mask, "Value", sharpen, "Value").unwrap();
    graph
        .link(paint_color, "Result", paint, "Base Color")
        .unwrap();
    graph
        .link(metal_color, "Result", metal, "Base Color")
        .unwrap();
    graph
        .link(metal_metallic, "Value", metal, "Metallic")
        .unwrap();

    graph.link(sharpen, "Value", mix, "Factor").unwrap();
    graph.link(paint, "Surface", mix, "A").unwrap();
    graph.link(metal, "Surface", mix, "B").unwrap();

    graph.link(mix, "Surface", out, "Surface").unwrap();

    let generated = emit(&graph).unwrap();
    assert!(
        generated.contains("fn sg_mix_surface"),
        "a graph that mixes must pull in the blend function:\n{generated}"
    );

    validate("mixed surfaces", &graph);
}

/// The case the two-pass check exists for.
///
/// None of the presets read a mesh attribute, so without this the second pass
/// would be checking that unconditional shaders are still unconditional. This
/// graph reads both optional attributes, so it genuinely compiles to two
/// different shaders and both have to hold up.
#[test]
fn a_graph_reading_mesh_attributes_is_valid_with_and_without_them() {
    use shadergraph::{NodeKind, VectorOp};

    let mut graph = Graph::new();
    let geometry = graph.add(NodeKind::Geometry);
    let scaled = graph.add(NodeKind::VectorMath(VectorOp::Scale));
    let voronoi = graph.add(NodeKind::Voronoi);
    let (surface, _out) = graph.surface_output();

    graph.link(geometry, "UV", scaled, "Vector").unwrap();
    graph.link(scaled, "Vector", voronoi, "Vector").unwrap();
    graph
        .link(voronoi, "F1 Distance", surface, "Metallic")
        .unwrap();
    graph
        .link(geometry, "Vertex Color", surface, "Base Color")
        .unwrap();

    let generated = emit(&graph).unwrap();
    assert!(
        generated.contains("#ifdef VERTEX_UVS_A") && generated.contains("#ifdef VERTEX_COLORS"),
        "a graph reading both attributes must guard both:\n{generated}"
    );

    validate("mesh attributes", &graph);
}

/// Every conversion node, chained, so each one's output feeds the next.
///
/// The `Separate*` nodes are the reason this exists. They have no code generator
/// behind them — their outputs are read as `binding.{ident}`, and the idents are
/// chosen so that read *is* a WGSL swizzle. Nothing in the Rust type system
/// checks that `r` is a legal accessor on a `vec4<f32>`; only a WGSL front end
/// does. So this asserts the swizzles appear and then hands the result to naga.
#[test]
fn the_conversion_nodes_emit_valid_wgsl() {
    use shadergraph::NodeKind;

    let mut graph = Graph::new();
    let geometry = graph.add(NodeKind::Geometry);
    let separate_xyz = graph.add(NodeKind::SeparateXyz);
    let combine_xyz = graph.add(NodeKind::CombineXyz);
    let to_color = graph.add(NodeKind::VectorToColor);
    let separate_color = graph.add(NodeKind::SeparateColor);
    let combine_color = graph.add(NodeKind::CombineColor);
    let to_vector = graph.add(NodeKind::ColorToVector);
    let (surface, _out) = graph.surface_output();

    graph
        .link(geometry, "Normal", separate_xyz, "Vector")
        .unwrap();
    for axis in ["X", "Y", "Z"] {
        graph.link(separate_xyz, axis, combine_xyz, axis).unwrap();
    }
    graph
        .link(combine_xyz, "Vector", to_color, "Vector")
        .unwrap();
    graph
        .link(to_color, "Color", separate_color, "Color")
        .unwrap();
    for channel in ["Red", "Green", "Blue", "Alpha"] {
        graph
            .link(separate_color, channel, combine_color, channel)
            .unwrap();
    }
    graph
        .link(combine_color, "Color", to_vector, "Color")
        .unwrap();

    graph
        .link(combine_color, "Color", surface, "Base Color")
        .unwrap();
    graph.link(to_vector, "Vector", surface, "Normal").unwrap();
    graph.link(separate_xyz, "X", surface, "Metallic").unwrap();

    let generated = emit(&graph).unwrap();
    for swizzle in [".x", ".y", ".z", ".r", ".g", ".b", ".a"] {
        assert!(
            generated.contains(swizzle),
            "expected a `{swizzle}` component read in:\n{generated}"
        );
    }

    validate("conversions", &graph);
}

/// Both noise variants, in one graph, so the shared and generated paths are
/// type-checked against each other.
///
/// This is the only thing that proves `sg_perlin_3d` and the unrolled generator
/// are well-formed. Everything else about noise is asserted on generated *text*;
/// naga is what says the text is a program.
#[test]
fn both_noise_variants_emit_valid_wgsl() {
    use shadergraph::{NodeKind, NoiseOctaves};

    let mut graph = Graph::new();
    let geometry = graph.add(NodeKind::Geometry);
    let live = graph.add(NodeKind::Noise(NoiseOctaves::Live));
    let unrolled = graph.add(NodeKind::Noise(NoiseOctaves::Unrolled(4)));
    let (surface, _out) = graph.surface_output();

    graph.link(geometry, "Position", live, "Vector").unwrap();
    graph
        .link(geometry, "Position", unrolled, "Vector")
        .unwrap();
    graph.link(live, "Value", surface, "Metallic").unwrap();
    graph
        .link(unrolled, "Value", surface, "Perceptual Roughness")
        .unwrap();

    let generated = emit(&graph).unwrap();
    assert!(
        generated.contains("fn sg_fbm_3d"),
        "the live variant must bring the shared loop"
    );
    assert!(
        generated.contains("fn sg_perlin_3d"),
        "both variants share one gradient noise implementation"
    );
    assert_eq!(
        generated.matches("fn sg_perlin_3d").count(),
        1,
        "gradient noise must be emitted once, not once per noise node"
    );

    validate("noise", &graph);
}

/// An unrolled node on its own must not drag in the loop it exists to replace.
#[test]
fn unrolled_noise_alone_emits_no_dynamic_loop() {
    use shadergraph::{NodeKind, NoiseOctaves};

    let mut graph = Graph::new();
    let geometry = graph.add(NodeKind::Geometry);
    let noise = graph.add(NodeKind::Noise(NoiseOctaves::Unrolled(3)));
    let (surface, _out) = graph.surface_output();
    graph.link(geometry, "Position", noise, "Vector").unwrap();
    graph.link(noise, "Value", surface, "Metallic").unwrap();

    let generated = emit(&graph).unwrap();
    assert!(
        !generated.contains("fn sg_fbm_3d"),
        "an unrolled-only graph carried the dynamic loop:\n{generated}"
    );
    assert!(generated.contains("fn sg_perlin_3d"));

    validate("unrolled noise", &graph);
}

/// Every blend mode, chained, plus the two new scalar mixers.
///
/// Blend modes are emitted as inline arithmetic rather than through a shared
/// function, so a malformed one is a malformed *expression* — the kind of thing
/// that reads fine and does not compile. Chaining them means each mode's output
/// is also the next one's input, which catches a mode that produced the wrong
/// type as well as one that produced nonsense.
#[test]
fn every_blend_mode_emits_valid_wgsl() {
    use shadergraph::{BlendMode, NodeKind};

    let mut graph = Graph::new();
    let geometry = graph.add(NodeKind::Geometry);
    let (surface, _out) = graph.surface_output();

    // A scalar to drive every factor, so the modes are the only variable.
    let factor = graph.param_float(0.5);

    // Start from a color that is not a literal, so nothing constant-folds away.
    // The output socket name changes after the first hop — `VectorToColor` calls
    // its output `Color`, `MixColor` calls its `Result`, following Blender.
    let mut color = graph.add(NodeKind::VectorToColor);
    let mut color_socket = "Color";
    graph.link(geometry, "Normal", color, "Vector").unwrap();

    for &mode in BlendMode::ALL {
        let mixed = graph.add(NodeKind::MixColor(mode));
        graph.link(factor, "Value", mixed, "Factor").unwrap();
        graph.link(color, color_socket, mixed, "A").unwrap();
        color = mixed;
        color_socket = "Result";
    }
    graph
        .link(color, color_socket, surface, "Base Color")
        .unwrap();

    // The two new mixers, wired to sockets of their own types.
    let mix_float = graph.add(NodeKind::MixFloat);
    graph.link(factor, "Value", mix_float, "Factor").unwrap();
    graph.link(mix_float, "Value", surface, "Metallic").unwrap();

    let mix_vector = graph.add(NodeKind::MixVector);
    graph.link(factor, "Value", mix_vector, "Factor").unwrap();
    graph.link(geometry, "Normal", mix_vector, "A").unwrap();
    graph.link(mix_vector, "Vector", surface, "Normal").unwrap();

    let generated = emit(&graph).unwrap();
    assert!(
        generated.contains("fn sg_overlay"),
        "Overlay is the one mode needing a helper, and it was not emitted"
    );

    validate("blend modes", &graph);
}

/// A mapped coordinate feeding a texture-ish node — the node's whole purpose.
#[test]
fn the_mapping_node_emits_valid_wgsl() {
    use shadergraph::{NodeKind, NoiseOctaves};

    let mut graph = Graph::new();
    let geometry = graph.add(NodeKind::Geometry);
    let mapping = graph.add(NodeKind::Mapping);
    let rotation = graph.param_vec3([0.0, 0.0, std::f32::consts::FRAC_PI_4]);
    let noise = graph.add(NodeKind::Noise(NoiseOctaves::Live));
    let (surface, _out) = graph.surface_output();

    graph.link(geometry, "Position", mapping, "Vector").unwrap();
    graph.link(rotation, "Vector", mapping, "Rotation").unwrap();
    graph.link(mapping, "Vector", noise, "Vector").unwrap();
    graph.link(noise, "Value", surface, "Metallic").unwrap();

    let generated = emit(&graph).unwrap();
    assert!(generated.contains("fn sg_euler_xyz"));
    assert!(
        generated.contains("mat3x3<f32>"),
        "the rotation must build a matrix rather than three separate rotations"
    );

    validate("mapping", &graph);
}

/// Noise driving a bump, which is the whole reason the node exists.
///
/// Worth validating rather than assuming: `dpdx`/`dpdy` are fragment-stage-only
/// builtins with a uniform-control-flow requirement, so a bump emitted into the
/// wrong place is a shader that a front end rejects outright.
#[test]
fn bump_from_a_height_field_emits_valid_wgsl() {
    use shadergraph::{NodeKind, NoiseOctaves};

    let mut graph = Graph::new();
    let geometry = graph.add(NodeKind::Geometry);
    let noise = graph.add(NodeKind::Noise(NoiseOctaves::Live));
    let bump = graph.add(NodeKind::Bump);
    let (surface, _out) = graph.surface_output();

    graph.link(geometry, "Position", noise, "Vector").unwrap();
    graph.link(noise, "Value", bump, "Height").unwrap();
    graph.link(bump, "Normal", surface, "Normal").unwrap();

    let generated = emit(&graph).unwrap();
    assert!(generated.contains("fn sg_bump"));
    for derivative in [
        "dpdx(position)",
        "dpdy(position)",
        "dpdx(height)",
        "dpdy(height)",
    ] {
        assert!(generated.contains(derivative), "missing {derivative}");
    }
    // The basis must come from the real surface position, never from whatever
    // coordinate the height was sampled at.
    assert!(
        generated.contains(", sg_position)"),
        "bump did not reconstruct its basis from the surface position:\n{generated}"
    );

    validate("bump", &graph);
}

/// An unwired Bump is inert rather than wrong.
///
/// Height defaults to a constant, so its derivative is zero and the normal comes
/// back untouched — a node you can drop in and wire up afterwards without the
/// surface going strange in between.
#[test]
fn an_unwired_bump_still_compiles() {
    use shadergraph::NodeKind;

    let mut graph = Graph::new();
    let bump = graph.add(NodeKind::Bump);
    let (surface, _out) = graph.surface_output();
    graph.link(bump, "Normal", surface, "Normal").unwrap();

    validate("unwired bump", &graph);
}

/// A graph using no exotic blend mode must not carry the helper.
#[test]
fn a_plain_mix_does_not_drag_in_the_blend_helper() {
    let generated = emit(&materials::hull_metal().unwrap()).unwrap();
    assert!(
        !generated.contains("sg_overlay"),
        "a plain Mix pulled in the blend snippet:\n{generated}"
    );
}

/// ...and the converse: a graph that does not read them carries no branch.
#[test]
fn a_graph_ignoring_mesh_attributes_emits_no_conditionals() {
    let generated = emit(&materials::hull_metal().unwrap()).unwrap();
    assert!(
        !generated.contains("#ifdef"),
        "an unconditional graph grew a conditional:\n{generated}"
    );
}

/// Two textures sampled at the mesh UV, one masking the other by its alpha.
///
/// The graph a decal actually is, and it exercises every texture-specific
/// emission at once: two slots, both declaration pairs, the `.rgba` and `.a`
/// swizzle reads, and the interaction with the UV `#ifdef` prelude — which is
/// why it must validate on both mesh branches like everything else.
#[test]
fn tex_image_emits_valid_wgsl() {
    use shadergraph::{BlendMode, NodeKind};

    let mut graph = Graph::new();
    let geometry = graph.add(NodeKind::Geometry);
    let base = graph.add(NodeKind::tex_image("hull_base.png"));
    let decal = graph.add(NodeKind::tex_image("insignia.png"));
    let mix = graph.add(NodeKind::MixColor(BlendMode::Mix));
    let (surface, _out) = graph.surface_output();
    graph.link(geometry, "UV", base, "Vector").unwrap();
    graph.link(geometry, "UV", decal, "Vector").unwrap();
    graph.link(base, "Color", mix, "A").unwrap();
    graph.link(decal, "Color", mix, "B").unwrap();
    graph.link(decal, "Alpha", mix, "Factor").unwrap();
    graph.link(mix, "Result", surface, "Base Color").unwrap();

    let generated = emit(&graph).unwrap();
    for binding in [
        "var sg_texture_0: texture_2d<f32>;",
        "var sg_sampler_0: sampler;",
        "var sg_texture_1: texture_2d<f32>;",
        "var sg_sampler_1: sampler;",
    ] {
        assert!(generated.contains(binding), "missing {binding}");
    }

    validate("tex_image", &graph);
}

/// ...and the converse: a graph sampling nothing declares no texture bindings.
///
/// This is what keeps every pre-texture golden byte-identical, so it is pinned
/// as its own claim rather than left implicit in the golden files.
#[test]
fn a_texture_free_graph_carries_no_texture_bindings() {
    let generated = emit(&materials::hull_metal().unwrap()).unwrap();
    for fragment in ["sg_texture_", "sg_sampler_", "textureSample"] {
        assert!(
            !generated.contains(fragment),
            "a graph with no TexImage emitted `{fragment}`:\n{generated}"
        );
    }
}
