//! Golden-file tests over emitted WGSL.
//!
//! These do not check that a shader is *correct* — nothing without a GPU can —
//! but they check that it is *stable*. That is the property that matters during
//! development of a compiler: when a refactor is meant to change nothing, these
//! prove it changed nothing, and when it is meant to change one line, the diff
//! shows exactly one line.
//!
//! To update after an intentional change:
//!
//! ```sh
//! SHADERGRAPH_BLESS=1 cargo test
//! ```
//!
//! Then read the resulting diff before committing it. A blessed golden file
//! that nobody looked at is worse than no golden file at all, because it
//! converts a loud failure into a silent endorsement.

use std::path::PathBuf;

use shadergraph::{emit, materials, Graph};

fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(format!("{name}.wgsl"))
}

fn check(name: &str, actual: &str) {
    let path = golden_path(name);

    if std::env::var_os("SHADERGRAPH_BLESS").is_some() {
        let dir = path.parent().expect("golden path always has a parent");
        std::fs::create_dir_all(dir)
            .unwrap_or_else(|e| panic!("cannot create golden directory {}: {e}", dir.display()));
        std::fs::write(&path, actual)
            .unwrap_or_else(|e| panic!("cannot write golden file {}: {e}", path.display()));
        return;
    }

    let expected = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "cannot read golden file {}: {e}\n\
             If this is a new material, create it with: SHADERGRAPH_BLESS=1 cargo test",
            path.display()
        )
    });

    if expected == actual {
        return;
    }

    // Report the first divergence rather than dumping both files. A 90-line
    // shader printed twice buries the one line that actually moved.
    let mut detail = String::new();
    for (i, (want, got)) in expected.lines().zip(actual.lines()).enumerate() {
        if want != got {
            detail = format!(
                "first difference at line {}:\n  expected: {want}\n  actual:   {got}",
                i + 1
            );
            break;
        }
    }
    if detail.is_empty() {
        detail = format!(
            "line counts differ: expected {}, got {}",
            expected.lines().count(),
            actual.lines().count()
        );
    }

    panic!(
        "emitted WGSL no longer matches {}\n{detail}\n\n\
         If the change is intended: SHADERGRAPH_BLESS=1 cargo test",
        path.display()
    );
}

#[test]
fn hull_metal_is_stable() {
    let graph = materials::hull_metal().expect("hull_metal builds");
    check("hull_metal", &emit(&graph).expect("hull_metal emits"));
}

#[test]
fn glass_is_stable() {
    let graph = materials::glass([0.1, 0.12, 0.2, 1.0], 0.3).expect("glass builds");
    check("glass", &emit(&graph).expect("glass emits"));
}

/// The only preset carrying a generated per-node function.
///
/// Worth a golden of its own because a color ramp's WGSL is computed from stop
/// data rather than copied from a fixed snippet, so it is the one place where a
/// change to the maths shows up as changed *source* — which is exactly what a
/// golden file is for.
#[test]
fn scorched_plating_is_stable() {
    let graph = materials::scorched_plating().expect("scorched_plating builds");
    check(
        "scorched_plating",
        &emit(&graph).expect("scorched_plating emits"),
    );
}

/// The only preset that reads the mesh's optional vertex attributes.
///
/// Its golden is where the `#ifdef` guards live, and they are the one part of
/// generated source that has a branch the preview never renders. A change to how
/// an attribute is bound — or to its fallback value, which is load-bearing and
/// easy to flip by accident — shows up here and nowhere else.
#[test]
fn panel_wear_is_stable() {
    let graph = materials::panel_wear().expect("panel_wear builds");
    check("panel_wear", &emit(&graph).expect("panel_wear emits"));
}

/// The only preset that layers two whole surfaces.
///
/// Its golden is where the `SgSurface` constructor appears twice and the blend
/// function appears at all. Field order in that constructor is positional and
/// generated from the same table that declares the struct, so this file is the
/// place a drift between the two would become visible — and a drift between two
/// same-typed neighbours would otherwise still compile.
#[test]
fn worn_paint_is_stable() {
    let graph = materials::worn_paint().expect("worn_paint builds");
    check("worn_paint", &emit(&graph).expect("worn_paint emits"));
}

/// The unrolled noise generator, pinned.
///
/// Generated per-node code, like the color ramp's function — its text depends
/// on data the node carries, so a change to the maths shows up as changed source
/// rather than as a slightly different surface nobody notices.
#[test]
fn unrolled_noise_is_stable() {
    use shadergraph::{NodeKind, NoiseOctaves};

    let mut graph = Graph::new();
    let geometry = graph.add(NodeKind::Geometry);
    let noise = graph.add(NodeKind::Noise(NoiseOctaves::Unrolled(3)));
    let (surface, _out) = graph.surface_output();
    graph.link(geometry, "Position", noise, "Vector").unwrap();
    graph.link(noise, "Value", surface, "Metallic").unwrap();

    check(
        "unrolled_noise",
        &emit(&graph).expect("unrolled noise emits"),
    );
}

/// Emission must not depend on anything that varies between runs.
///
/// A `HashMap` anywhere on the emission path would make this flap
/// intermittently rather than fail outright, which is the sort of bug that
/// costs an afternoon. Cheap insurance.
#[test]
fn emission_is_deterministic() {
    let graph = materials::hull_metal().unwrap();
    let first = emit(&graph).unwrap();
    for _ in 0..16 {
        assert_eq!(first, emit(&graph).unwrap());
    }
}

/// The generated shader must reference only Bevy items that actually exist.
///
/// Every name here was checked against `bevy_pbr` 0.16 by hand. They are
/// pinned in a test because a Bevy upgrade that renames one produces a shader
/// that fails to compile at runtime, inside generated code, with a message
/// pointing at a line the author never wrote.
#[test]
fn generated_shader_targets_known_bevy_symbols() {
    let wgsl = emit(&materials::hull_metal().unwrap()).unwrap();
    for symbol in [
        "bevy_pbr::forward_io::VertexOutput",
        "bevy_pbr::pbr_types::{PbrInput, pbr_input_new}",
        "bevy_pbr::mesh_view_bindings::view",
        "apply_pbr_lighting",
        "main_pass_post_lighting_processing",
        "pbr_input.material.perceptual_roughness",
        "pbr_input.anisotropy_T",
        "pbr_input.anisotropy_B",
        "pbr_input.anisotropy_strength",
        "view.clip_from_view[3][3]",
    ] {
        assert!(
            wgsl.contains(symbol),
            "generated shader is missing `{symbol}`"
        );
    }
}

/// The only golden carrying texture bindings.
///
/// Its file is where the declaration pairs, the binding arithmetic and the
/// `textureSample` lines live, so a change to any of them shows up here as a
/// readable diff — and, just as important, in *no other* golden: texture
/// declarations are conditional on the graph sampling something, and every
/// other golden proves the condition stays off.
#[test]
fn tex_image_is_stable() {
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

    check("tex_image", &emit(&graph).expect("tex_image emits"));
}
