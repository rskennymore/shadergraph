// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Blend modes that need more than one arithmetic expression.
//
// Everything else in `BlendMode` is a single operator and is emitted inline;
// only what genuinely needs a branch lives here.

// Overlay: Multiply where the base is dark, Screen where it is light.
//
// The two halves meet continuously at 0.5, which is what stops the transition
// from showing as a visible edge on a smooth gradient.
//
// Written with `step` and `mix` rather than an `if`, because this runs per
// fragment per channel and a branch on a per-channel comparison would have to be
// scalarised. `step(0.5, a)` is 0 below the midpoint and 1 at or above it, so the
// `mix` selects per component with no divergence at all.
fn sg_overlay(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    let dark = 2.0 * a * b;
    let light = 1.0 - 2.0 * (1.0 - a) * (1.0 - b);
    return mix(dark, light, step(vec3<f32>(0.5), a));
}
