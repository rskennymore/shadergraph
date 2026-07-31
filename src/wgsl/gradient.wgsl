// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Gradient fields: closed-form functions of position, with no noise and no
// sampling.
//
// PROVENANCE: these are the standard gradient forms — a linear ramp, its square,
// a smoothstep, a diagonal, a polar angle and a radial falloff. Written from the
// mathematical descriptions, which are textbook. The socket set and the names
// follow Blender's Gradient Texture so that muscle memory transfers; no code was
// copied from it.
//
// WHY THESE ARE ONE SNIPPET AND NOT SIX NODES: unlike the Math or Vector
// operations, every form here takes the same input and returns the same thing.
// The choice is a shape, not a different operation with different sockets, so it
// belongs on one node as an enum — the same reasoning that gives `MixColor` its
// blend modes rather than nine nodes.
//
// The coordinate is deliberately NOT normalised or wrapped. A gradient is
// meaningful outside [0,1] — that is what lets it be scaled and offset by a
// Mapping node upstream, and clamped downstream by a Color Ramp if that is
// wanted. Clamping here would silently remove the ability to do either.

// A straight ramp along X. The identity of the family.
fn sg_gradient_linear(p: vec3<f32>) -> f32 {
    return p.x;
}

// The ramp squared, so it starts slowly and accelerates.
//
// Negative X is clamped away first: squaring it would mirror the ramp back up
// on the wrong side of the origin, turning a one-sided gradient into a valley.
fn sg_gradient_quadratic(p: vec3<f32>) -> f32 {
    let r = max(p.x, 0.0);
    return r * r;
}

// A smoothstep across [0,1] — flat at both ends, steepest in the middle.
//
// `3t² − 2t³`, the classic cubic Hermite ease. Note this is the *cubic* one, not
// the quintic `6t⁵−15t⁴+10t³` used by the color ramp and the noise fade: those
// need a continuous second derivative because they are differentiated
// downstream, and this is not.
fn sg_gradient_easing(p: vec3<f32>) -> f32 {
    let r = clamp(p.x, 0.0, 1.0);
    return r * r * (3.0 - 2.0 * r);
}

// A ramp along the X+Y diagonal, halved so the unit diagonal still spans 0..1.
fn sg_gradient_diagonal(p: vec3<f32>) -> f32 {
    return (p.x + p.y) * 0.5;
}

// Angle about the Z axis, remapped from [-pi, pi] to [0, 1].
//
// NOTE: This is the one form with a discontinuity: the value wraps from 1 back to 0
// across the -X axis, and anything derived from its slope — a Bump, most
// obviously — will show a seam along that line. That is inherent to a polar
// angle, not a defect here.
fn sg_gradient_radial(p: vec3<f32>) -> f32 {
    const INV_TAU: f32 = 0.15915494;
    return atan2(p.y, p.x) * INV_TAU + 0.5;
}

// Distance from the origin, inverted so the centre is 1 and it reaches 0 at
// unit radius. Clamped, so it stays 0 further out rather than going negative.
fn sg_gradient_spherical(p: vec3<f32>) -> f32 {
    return max(1.0 - length(p), 0.0);
}

// The spherical falloff squared — a rounder, softer blob with a flatter edge.
fn sg_gradient_quadratic_sphere(p: vec3<f32>) -> f32 {
    let r = max(1.0 - length(p), 0.0);
    return r * r;
}
