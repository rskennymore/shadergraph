//! Material graphs built by hand.
//!
//! These exist before any editor does, and they are the reason the build order
//! puts the compiler first: a graph written as a Rust literal exercises the
//! whole pipeline without a single line of UI, and it can be tested in
//! milliseconds with no GPU present.

use crate::graph::{Graph, GraphError};
use crate::node::{BlendMode, ColorToFloatOp, MathOp, NodeKind, NoiseOctaves, VectorOp};
use crate::ramp::{ColorRamp, ColorStop, RampInterp};

/// Tunables for the brushed-metal graph.
///
/// Every metal preset in this module is this one graph with different numbers,
/// which is the whole argument for the parameter split: three materials, one
/// piece of shading logic to get right.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrushedMetal {
    /// Surface color away from the grain boundaries.
    pub base_color: [f32; 4],
    /// Color in the grooves between Voronoi cells. Usually a darker shade of
    /// `base_color` rather than a different hue.
    pub groove_color: [f32; 4],
    /// Grain density, in cells per world unit. Higher is finer.
    pub brush_scale: f32,
    /// How wide the darkened boundary between cells reads, as a fraction of
    /// cell size. Small values give a crisp scratch, large a soft mottle.
    pub groove_width: f32,
    /// How far feature points stray from their lattice positions. At 0 the
    /// grain is a regular grid, which looks manufactured in the wrong way.
    pub randomness: f32,
    /// Passed through to the surface unchanged.
    pub metallic: f32,
    /// Passed through to the surface unchanged.
    pub perceptual_roughness: f32,
    /// How strongly the highlight stretches along the grain. 0 is an ordinary
    /// isotropic metal; this is the parameter that makes it read as *brushed*.
    pub anisotropy: f32,
}

impl Default for BrushedMetal {
    fn default() -> Self {
        // Color, metallic and roughness deliberately match values tuned in a
        // StandardMaterial workflow, so that switching a mesh between backends
        // changes only the *shading*. Left at generic mid-grey defaults the
        // toggle would show an albedo jump on top of the lighting change, and
        // the comparison it exists for would be worthless.
        Self {
            base_color: [0.25, 0.25, 0.28, 1.0],
            groove_color: [0.18, 0.18, 0.21, 1.0],
            brush_scale: 4.0,
            groove_width: 0.15,
            randomness: 1.0,
            metallic: 0.95,
            perceptual_roughness: 0.3,
            anisotropy: 0.8,
        }
    }
}

/// Build the brushed-metal graph.
///
/// The shape of it:
///
/// ```text
///   Geometry.Position ──▶ Scale ──┬──▶ Voronoi ──┬──▶ F1 Position ──▶ Subtract ──▶ Anisotropy Tangent
///                                 └──────────────┼────────────────────▲
///                                                ├──▶ F2 Distance ──▶ Subtract ──▶ SmoothStep ──▶ Factor
///                                                └──▶ F1 Distance ──▶ ▲                            │
///                                                                     MixColor ◀───────────────────┘
///                                                                        └──▶ Base Color
/// ```
///
/// Two things come out of one Voronoi evaluation. The **brush direction** is
/// the vector from the shaded point toward its nearest cell center; feeding it
/// to the anisotropy tangent is what stretches the specular highlight along the
/// grain, and it varies per fragment, which is why this needs no UV channel and
/// no mesh tangents. The **groove mask** is the gap between the nearest and
/// second-nearest distances, which approaches zero exactly on a cell boundary
/// and so darkens the seams between grains.
///
/// Notably absent: a Fresnel rim term. The shader this replaces computed one by
/// hand, because it did its own lighting and had to. Terminating in
/// `PbrOutput` hands the surface to a real BRDF that already accounts for
/// Fresnel, so adding another would double-count it — the rim would read as
/// too hot at grazing angles and no amount of parameter tuning would fix it.
pub fn brushed_metal(params: BrushedMetal) -> Result<Graph, GraphError> {
    let mut g = Graph::new();

    let geo = g.add(NodeKind::Geometry);

    // --- grain coordinate ---------------------------------------------------
    // World position scaled into grain space. The Voronoi node has its own
    // Scale input, but scaling here instead means the scaled position is
    // available as a node in its own right — which it must be, because the
    // brush direction is measured against it below.
    let scale = g.param_float(params.brush_scale);
    let grain_pos = g.add(NodeKind::VectorMath(VectorOp::Scale));
    g.link(geo, "Position", grain_pos, "Vector")?;
    g.link(scale, "Value", grain_pos, "Scale")?;

    let unit_scale = g.param_float(1.0);
    let randomness = g.param_float(params.randomness);
    let voronoi = g.add(NodeKind::Voronoi);
    g.link(grain_pos, "Vector", voronoi, "Vector")?;
    g.link(unit_scale, "Value", voronoi, "Scale")?;
    g.link(randomness, "Value", voronoi, "Randomness")?;

    // --- brush direction ----------------------------------------------------
    // Toward the nearest cell centre. Left un-normalised and not flattened into
    // the surface: the output node rejects it against the shading normal and
    // normalises it, so doing either here would be redundant work that also
    // hides where the real requirement lives.
    let brush = g.add(NodeKind::VectorMath(VectorOp::Subtract));
    g.link(voronoi, "F1 Position", brush, "Vector")?;
    g.link(grain_pos, "Vector", brush, "Vector_001")?;

    // --- groove mask --------------------------------------------------------
    // F2 - F1 is zero on a cell boundary and grows toward cell interiors, so
    // smoothstepping it from zero gives 0 in the seams and 1 on the flats.
    let edge_delta = g.add(NodeKind::Math(MathOp::Subtract));
    g.link(voronoi, "F2 Distance", edge_delta, "Value")?;
    g.link(voronoi, "F1 Distance", edge_delta, "Value_001")?;

    let zero = g.param_float(0.0);
    let groove_width = g.param_float(params.groove_width);
    let groove = g.add(NodeKind::Math(MathOp::SmoothStep));
    g.link(zero, "Value", groove, "Value")?;
    g.link(groove_width, "Value", groove, "Value_001")?;
    g.link(edge_delta, "Value", groove, "Value_002")?;

    let groove_color = g.param_color(params.groove_color);
    let base_color = g.param_color(params.base_color);
    let albedo = g.add(NodeKind::MixColor(BlendMode::Mix));
    g.link(groove, "Value", albedo, "Factor")?;
    g.link(groove_color, "Result", albedo, "A")?;
    g.link(base_color, "Result", albedo, "B")?;

    // --- surface ------------------------------------------------------------
    let metallic = g.param_float(params.metallic);
    let roughness = g.param_float(params.perceptual_roughness);
    let anisotropy = g.param_float(params.anisotropy);

    let (surface, _out) = g.surface_output();
    g.link(albedo, "Result", surface, "Base Color")?;
    g.link(metallic, "Value", surface, "Metallic")?;
    g.link(roughness, "Value", surface, "Perceptual Roughness")?;
    g.link(anisotropy, "Value", surface, "Anisotropy Strength")?;
    g.link(brush, "Vector", surface, "Anisotropy Tangent")?;

    Ok(g)
}

/// The default hull plating.
pub fn hull_metal() -> Result<Graph, GraphError> {
    brushed_metal(BrushedMetal::default())
}

/// Darker hull plating. Same graph, different numbers.
pub fn hull_dark() -> Result<Graph, GraphError> {
    brushed_metal(BrushedMetal {
        base_color: [0.08, 0.08, 0.1, 1.0],
        groove_color: [0.05, 0.05, 0.07, 1.0],
        perceptual_roughness: 0.45,
        ..BrushedMetal::default()
    })
}

/// Copper trim. Finer grain and a smoother finish, so the anisotropy reads
/// more strongly.
pub fn copper_accent() -> Result<Graph, GraphError> {
    brushed_metal(BrushedMetal {
        base_color: [0.72, 0.45, 0.2, 1.0],
        groove_color: [0.55, 0.33, 0.14, 1.0],
        brush_scale: 6.0,
        metallic: 1.0,
        perceptual_roughness: 0.25,
        ..BrushedMetal::default()
    })
}

/// Scorched hull plating: the grain drives a color ramp instead of a two-way
/// mix.
///
/// Exists to exercise [`NodeKind::ColorRamp`] end to end, and it earns its place
/// as a material too: a mix node can only ever interpolate between two colors,
/// so heat discoloration — which runs pale through straw through blue-black —
/// is exactly the thing it cannot express. Three stops and a quintic spline do
/// it in one node.
///
/// The spline rather than the ease deliberately: heat staining is a continuous
/// gradient, and the ease's flat spot at each stop would read as banding across
/// a large panel.
pub fn scorched_plating() -> Result<Graph, GraphError> {
    let mut g = Graph::new();

    let geo = g.add(NodeKind::Geometry);

    let scale = g.param_float(3.0);
    let grain_pos = g.add(NodeKind::VectorMath(VectorOp::Scale));
    g.link(geo, "Position", grain_pos, "Vector")?;
    g.link(scale, "Value", grain_pos, "Scale")?;

    let unit_scale = g.param_float(1.0);
    let randomness = g.param_float(1.0);
    let voronoi = g.add(NodeKind::Voronoi);
    g.link(grain_pos, "Vector", voronoi, "Vector")?;
    g.link(unit_scale, "Value", voronoi, "Scale")?;
    g.link(randomness, "Value", voronoi, "Randomness")?;

    // Distance to the nearest cell centre, which varies smoothly across each
    // cell and so makes a serviceable "how hot did this patch get" field.
    let ramp = g.add(NodeKind::ColorRamp(ColorRamp {
        stops: vec![
            ColorStop::new(0.0, [0.62, 0.60, 0.55, 1.0]),
            ColorStop::new(0.35, [0.45, 0.34, 0.13, 1.0]),
            ColorStop::new(0.70, [0.16, 0.13, 0.22, 1.0]),
            ColorStop::new(1.0, [0.05, 0.05, 0.06, 1.0]),
        ],
        interp: RampInterp::QuinticSpline,
    }));
    g.link(voronoi, "F1 Distance", ramp, "Factor")?;

    let metallic = g.param_float(0.7);
    let roughness = g.param_float(0.65);

    let (surface, _out) = g.surface_output();
    g.link(ramp, "Color", surface, "Base Color")?;
    g.link(metallic, "Value", surface, "Metallic")?;
    g.link(roughness, "Value", surface, "Perceptual Roughness")?;

    Ok(g)
}

/// Fresnel-driven glass.
///
/// The second graph exists to prove the compiler is not overfitted to one
/// material: it shares no nodes with the metal beyond the output, and it is the
/// only current user of the Fresnel node — legitimately so, because here the
/// term drives *transparency* rather than a lighting response the BRDF already
/// models.
pub fn glass(base_color: [f32; 4], opacity: f32) -> Result<Graph, GraphError> {
    let mut g = Graph::new();

    let geo = g.add(NodeKind::Geometry);
    let ior = g.param_float(1.45);
    let fresnel = g.add(NodeKind::Fresnel);
    g.link(ior, "Value", fresnel, "IOR")?;
    g.link(geo, "Normal", fresnel, "Normal")?;
    g.link(geo, "Incoming", fresnel, "View")?;

    // alpha = opacity + (1 - opacity) * fresnel
    //
    // Glass seen face-on is mostly transmissive and at a grazing angle mostly
    // reflective, so alpha rises to fully opaque at the silhouette. Spelled with
    // Math nodes rather than a MixFloat because MixFloat does not exist yet —
    // and this is exactly the evidence for adding it.
    let one = g.param_float(1.0);
    let opacity_param = g.param_float(opacity);
    let inv_opacity = g.add(NodeKind::Math(MathOp::Subtract));
    g.link(one, "Value", inv_opacity, "Value")?;
    g.link(opacity_param, "Value", inv_opacity, "Value_001")?;

    let rim = g.add(NodeKind::Math(MathOp::Multiply));
    g.link(inv_opacity, "Value", rim, "Value")?;
    g.link(fresnel, "Factor", rim, "Value_001")?;

    let alpha = g.add(NodeKind::Math(MathOp::Add));
    g.link(opacity_param, "Value", alpha, "Value")?;
    g.link(rim, "Value", alpha, "Value_001")?;

    let color = g.param_color(base_color);
    let roughness = g.param_float(0.05);
    let metallic = g.param_float(0.0);

    let (surface, _out) = g.surface_output();
    g.link(color, "Result", surface, "Base Color")?;
    g.link(alpha, "Value", surface, "Alpha")?;
    g.link(roughness, "Value", surface, "Perceptual Roughness")?;
    g.link(metallic, "Value", surface, "Metallic")?;

    Ok(g)
}

/// A painted panel weathered by two mesh-authored channels.
///
/// The only preset that reads the optional vertex attributes, and it exists to
/// make them testable in one click rather than by hand-wiring a Geometry node
/// every time. It is also a real material and not a test card: this is the shape
/// the "wear level" idea takes once it is driven from Blender.
///
/// - **UV** scales a Voronoi field, so the grime pattern follows the unwrap
///   rather than world space — which is what makes it stay put on a moving hull.
/// - **Vertex Color** is the wear mask, painted per vertex in Blender. It needs
///   no UV unwrap of its own, which is exactly why it is the cheap channel for
///   baked occlusion and hand-painted damage.
///
/// NOTE: On a mesh with neither, this renders as the clean color everywhere: the
/// UV reads zero and the vertex color reads white. That is the intended
/// fallback, and it is deliberately quiet, which is why the editor calls it out.
pub fn panel_wear() -> Result<Graph, GraphError> {
    let mut g = Graph::new();

    let geo = g.add(NodeKind::Geometry);

    // Grime pattern in UV space.
    let uv_scale = g.param_float(6.0);
    g.expose(uv_scale, "grime_scale")?;
    let scaled_uv = g.add(NodeKind::VectorMath(VectorOp::Scale));
    g.link(geo, "UV", scaled_uv, "Vector")?;
    g.link(uv_scale, "Value", scaled_uv, "Scale")?;

    let cells = g.add(NodeKind::Voronoi);
    g.link(scaled_uv, "Vector", cells, "Vector")?;

    // The pattern's color, from clean paint through to bare scorched metal.
    let grime = g.add(NodeKind::ColorRamp(ColorRamp {
        stops: vec![
            ColorStop::new(0.0, [0.32, 0.30, 0.28, 1.0]),
            ColorStop::new(0.45, [0.18, 0.16, 0.15, 1.0]),
            ColorStop::new(1.0, [0.06, 0.05, 0.05, 1.0]),
        ],
        interp: RampInterp::QuinticSpline,
    }));
    g.link(cells, "F1 Distance", grime, "Factor")?;

    // Blend clean paint toward grime by the painted wear mask. The mask arrives
    // as a color and the blend factor wants a scalar, so its luminance stands
    // in — a greyscale mask has the same value in every channel anyway, and a
    // mask that was accidentally painted in color still behaves predictably.
    let painted = g.add(NodeKind::ColorToFloat(ColorToFloatOp::Luminance));
    g.link(geo, "Vertex Color", painted, "Color")?;

    // A whole-surface wear dial on top of the painted mask, exposed so a mesh
    // can drive it from a glTF custom property. Applied as a *ceiling* rather
    // than a subtraction: `min` caps how clean the surface is allowed to be,
    // which leaves the painted variation intact underneath, where subtracting
    // would slide the whole mask down and flatten it.
    //
    // At 0 the surface is exactly as painted; at 1 it is fully worn everywhere,
    // painted mask or not — which is what makes this useful on a module that
    // has no vertex colors at all.
    let wear_level = g.param_float(0.0);
    g.expose(wear_level, "wear_level")?;
    let one = g.param_float(1.0);
    let ceiling = g.add(NodeKind::Math(MathOp::Subtract));
    g.link(one, "Value", ceiling, "Value")?;
    g.link(wear_level, "Value", ceiling, "Value_001")?;

    let wear = g.add(NodeKind::Math(MathOp::Minimum));
    g.link(painted, "Value", wear, "Value")?;
    g.link(ceiling, "Value", wear, "Value_001")?;

    let clean = g.param_color([0.42, 0.44, 0.47, 1.0]);
    let albedo = g.add(NodeKind::MixColor(BlendMode::Mix));
    g.link(wear, "Value", albedo, "Factor")?;
    g.link(grime, "Color", albedo, "A")?;
    g.link(clean, "Result", albedo, "B")?;

    let metallic = g.param_float(0.15);
    let roughness = g.param_float(0.75);

    let (surface, _out) = g.surface_output();
    g.link(albedo, "Result", surface, "Base Color")?;
    g.link(metallic, "Value", surface, "Metallic")?;
    g.link(roughness, "Value", surface, "Perceptual Roughness")?;

    Ok(g)
}

/// Worn paint over bare metal — two whole materials and one mask.
///
/// The preset `layered-surface` exists for, and the honest counterpart to
/// [`panel_wear`]: that one mixes a *color* and then gives the result a single
/// metallic and a single roughness, because before the `Surface` type that was
/// all it could do. Paint and bare metal do not differ only in color. Paint is
/// a dielectric — metallic 0, fairly rough, no anisotropy. Exposed hull is
/// metallic 1, smoother, and carries the brushed grain that makes the highlight
/// stretch. Mixing those as two surfaces gets all of it from one mask; mixing
/// them as a color gets none of it.
///
/// NOTE: Both layers are evaluated at every fragment — see `wgsl/surface.wgsl`.
/// This material costs the paint plus the metal plus the mask, everywhere, not
/// whichever one the mask selects.
pub fn worn_paint() -> Result<Graph, GraphError> {
    let mut g = Graph::new();

    let geo = g.add(NodeKind::Geometry);

    // --- the mask ------------------------------------------------------------
    // Noise in world space, so wear does not swim when a module is instanced and
    // needs no UV channel to exist at all.
    let mask_scale = g.param_float(2.5);
    g.expose(mask_scale, "wear_scale")?;
    let scaled = g.add(NodeKind::VectorMath(VectorOp::Scale));
    g.link(geo, "Position", scaled, "Vector")?;
    g.link(mask_scale, "Value", scaled, "Scale")?;

    let noise = g.add(NodeKind::Noise(NoiseOctaves::Live));
    g.link(scaled, "Vector", noise, "Vector")?;

    // Pushed toward its extremes so the transition is a band rather than a
    // gradient across the whole panel. Worn paint has edges; a soft ramp between
    // paint and metal everywhere reads as dirt, not wear.
    let sharpen = g.add(NodeKind::Math(MathOp::SmoothStep));
    g.link(noise, "Value", sharpen, "Value")?;

    // The same whole-surface dial `panel_wear` uses, and for the same reason: a
    // glTF custom property can drive it per instance, so two copies of one
    // module can be weathered differently without a second material.
    let wear_level = g.param_float(0.35);
    g.expose(wear_level, "wear_level")?;
    let wear = g.add(NodeKind::Math(MathOp::Multiply));
    g.link(sharpen, "Value", wear, "Value")?;
    g.link(wear_level, "Value", wear, "Value_001")?;

    // --- layer one: paint ----------------------------------------------------
    let paint_color = g.param_color([0.15, 0.16, 0.18, 1.0]);
    g.expose(paint_color, "paint_color")?;
    let paint_roughness = g.param_float(0.62);
    let paint = g.add(NodeKind::Surface);
    g.link(paint_color, "Result", paint, "Base Color")?;
    g.link(paint_roughness, "Value", paint, "Perceptual Roughness")?;

    // --- layer two: the hull underneath --------------------------------------
    // A brush direction that lies in the surface: take a fixed world axis and
    // remove whatever part of it points along the normal, which is exactly what
    // `Reject` is for. A constant direction rather than `brushed_metal`'s
    // Voronoi-derived one — rolled hull plate has a single grain, and this keeps
    // the second layer cheap given both layers are always evaluated.
    //
    // The output block rejects the tangent against the shading normal again
    // before use, so this is not load-bearing for correctness. It is here so the
    // graph says what it means rather than relying on the root to fix it up.
    let up = g.param_vec3([0.0, 0.0, 1.0]);
    let brush = g.add(NodeKind::VectorMath(VectorOp::Reject));
    g.link(up, "Vector", brush, "Vector")?;
    g.link(geo, "Normal", brush, "Vector_001")?;

    let metal_color = g.param_color([0.52, 0.54, 0.57, 1.0]);
    let metal_metallic = g.param_float(1.0);
    let metal_roughness = g.param_float(0.34);
    let metal_aniso = g.param_float(0.5);
    let metal = g.add(NodeKind::Surface);
    g.link(metal_color, "Result", metal, "Base Color")?;
    g.link(metal_metallic, "Value", metal, "Metallic")?;
    g.link(metal_roughness, "Value", metal, "Perceptual Roughness")?;
    g.link(metal_aniso, "Value", metal, "Anisotropy Strength")?;
    g.link(brush, "Vector", metal, "Anisotropy Tangent")?;

    // --- the layering --------------------------------------------------------
    // Factor 0 is all paint, 1 is all metal, matching every other mix node.
    let mix = g.add(NodeKind::MixSurface);
    g.link(wear, "Value", mix, "Factor")?;
    g.link(paint, "Surface", mix, "A")?;
    g.link(metal, "Surface", mix, "B")?;

    let out = g.add(NodeKind::PbrOutput);
    g.link(mix, "Surface", out, "Surface")?;

    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::emit;

    #[test]
    fn every_preset_validates_and_compiles() {
        for (name, graph) in [
            ("hull_metal", hull_metal()),
            ("hull_dark", hull_dark()),
            ("copper_accent", copper_accent()),
            ("scorched_plating", scorched_plating()),
            ("panel_wear", panel_wear()),
            ("worn_paint", worn_paint()),
            ("glass", glass([0.1, 0.12, 0.2, 1.0], 0.3)),
        ] {
            let graph = graph.unwrap_or_else(|e| panic!("{name} failed to build: {e}"));
            graph
                .validate()
                .unwrap_or_else(|e| panic!("{name} failed to validate: {e}"));
            emit(&graph).unwrap_or_else(|e| panic!("{name} failed to emit: {e}"));
        }
    }

    /// The claim `worn_paint` is making, checked rather than asserted in prose.
    ///
    /// A layered material has to differ from a color blend in the properties a
    /// color blend cannot reach. If this ever collapses to one `Surface`, the
    /// preset has quietly become `panel_wear` with extra steps.
    #[test]
    fn worn_paint_layers_two_whole_surfaces() {
        let graph = worn_paint().unwrap();
        let surfaces = graph
            .nodes()
            .filter(|(_, kind)| matches!(kind, NodeKind::Surface))
            .count();
        assert_eq!(surfaces, 2, "worn paint is two materials, not one");

        // The metal layer carries a brush direction; the paint layer does not.
        // That difference is unreachable through a MixColor, and it is the whole
        // reason this preset exists.
        assert!(graph.uses_anisotropy());

        let wgsl = emit(&graph).unwrap();
        assert!(wgsl.contains("sg_mix_surface("), "{wgsl}");
    }

    #[test]
    fn metal_reaches_for_voronoi_and_glass_does_not() {
        // Confirms dead-code elimination is doing real work rather than every
        // shader carrying every support function.
        let metal = emit(&hull_metal().unwrap()).unwrap();
        assert!(metal.contains("sg_voronoi_f1f2_3d"));
        assert!(!metal.contains("fn sg_fresnel"));

        let glass = emit(&glass([0.0; 4], 0.5).unwrap()).unwrap();
        assert!(glass.contains("fn sg_fresnel"));
        assert!(!glass.contains("sg_voronoi_f1f2_3d"));
    }

    #[test]
    fn presets_that_differ_only_in_parameters_share_a_shader() {
        // The payoff for moving parameters into a uniform. Every brushed metal
        // is the same graph with different numbers, so all three compile to
        // byte-identical WGSL and therefore to ONE pipeline rather than three.
        //
        // This asserted the opposite while parameters were baked as literals,
        // and the inversion is the feature — if it ever fails again, a value has
        // leaked back into the source.
        let metal = emit(&hull_metal().unwrap()).unwrap();
        let dark = emit(&hull_dark().unwrap()).unwrap();
        let copper = emit(&copper_accent().unwrap()).unwrap();

        assert_eq!(metal, dark);
        assert_eq!(metal, copper);

        // Glass is a genuinely different graph, so it must not collapse in too
        // — otherwise this test would pass just as well on an emitter that
        // returned a constant.
        assert_ne!(
            metal,
            emit(&glass([0.1, 0.12, 0.2, 1.0], 0.3).unwrap()).unwrap()
        );
    }

    #[test]
    fn parameters_reach_the_layout_rather_than_the_source() {
        use crate::emit::emit_with_layout;
        use crate::params::SlotValue;
        use crate::value::Value;

        let params = BrushedMetal {
            brush_scale: 4.0,
            ..BrushedMetal::default()
        };
        let (wgsl, layout) = emit_with_layout(&brushed_metal(params).unwrap()).unwrap();

        // The distinctive grain scale must appear in the layout...
        assert!(
            layout
                .slots()
                .iter()
                .any(|s| s.value == SlotValue::Param(Value::Float(4.0))),
            "brush_scale never reached the layout"
        );

        // ...and nowhere in the shader. `4.0` as a bare literal would mean a
        // slider edit still recompiles, which is the whole thing this prevents.
        assert!(
            !wgsl.contains("= 4.0;"),
            "a parameter was baked into the source:\n{wgsl}"
        );
    }
}
