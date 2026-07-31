// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Blending two evaluated surfaces.
//
// WHY THIS IS POSSIBLE HERE AND NOT IN BLENDER: a Blender `Shader` socket holds
// a closure — a BSDF that has not been evaluated — so mixing two of them means
// mixing two *functions*, which is why Cycles needs a whole closure system to do
// it and why it cannot be lowered to a fixed structure. `SgSurface` is a struct
// of numbers that has already been evaluated, so mixing two surfaces is ten
// lerps and nothing else.
//
// NOTE: COST: there is no branch here, and there cannot be one. Both surfaces are
// fully evaluated at every fragment before either is used, so a two-layer
// material costs both layers everywhere — not the cheaper one where the mask
// says so. Nesting these for a third layer adds a third. Blender behaves the
// same way; it is worth knowing before layering six of them.

// Lerp two surfaces.
//
// `factor` is clamped rather than trusted. WGSL's `mix` EXTRAPOLATES outside
// [0,1] — a factor of 1.4 does not saturate at surface B, it overshoots past it,
// producing out-of-gamut colors and negative roughness. Every other mix node in
// this compiler clamps for the same reason.
fn sg_mix_surface(factor: f32, a: SgSurface, b: SgSurface) -> SgSurface {
    let t = clamp(factor, 0.0, 1.0);

    var out: SgSurface;
    out.base_color = mix(a.base_color, b.base_color, t);
    out.metallic = mix(a.metallic, b.metallic, t);
    out.perceptual_roughness = mix(a.perceptual_roughness, b.perceptual_roughness, t);
    out.reflectance = mix(a.reflectance, b.reflectance, t);
    out.emissive = mix(a.emissive, b.emissive, t);
    out.occlusion = mix(a.occlusion, b.occlusion, t);

    // NOTE: A lerp of two directions is not a rotation between them: it cuts the
    // chord rather than following the arc, and shortens as the two diverge.
    // Left un-normalised HERE on purpose — the output block normalises the
    // normal it is given, unconditionally, so normalising again would be dead
    // work at every fragment. The direction is right; only the length is not,
    // and nothing between here and there reads the length.
    out.normal = mix(a.normal, b.normal, t);

    out.anisotropy_strength = mix(a.anisotropy_strength, b.anisotropy_strength, t);
    // Same story as the normal, and the output block gives this one the same
    // treatment: it is rejected against the shading normal and re-normalised,
    // with the strength forced to zero if the result is degenerate. So a lerp
    // that passes exactly through zero midway between two opposed tangents
    // fails safe rather than producing NaN.
    out.anisotropy_tangent = mix(a.anisotropy_tangent, b.anisotropy_tangent, t);

    out.alpha = mix(a.alpha, b.alpha, t);
    return out;
}
