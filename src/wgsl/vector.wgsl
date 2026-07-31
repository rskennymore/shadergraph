// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Vector decomposition helpers. Textbook linear algebra; no external source.

// The component of `a` that lies along `b`.
//
// Guards against a degenerate `b`: `dot(b, b)` is zero for a zero-length
// vector, and dividing by it yields NaN. A NaN here does not stay local — it
// propagates through the brush direction into the anisotropy tangent and
// finally into the lit color, where it renders as a black or garbage fragment
// with nothing pointing back at the division that caused it.
fn sg_project(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    let denom = dot(b, b);
    if (denom < 1e-12) {
        return vec3<f32>(0.0);
    }
    return b * (dot(a, b) / denom);
}

// The component of `a` perpendicular to `b`.
//
// This is the operation that flattens an arbitrary direction into a surface:
// pass a surface normal as `b` and whatever remains is tangent to the surface.
// Blender has no such node and expects Project followed by Subtract.
fn sg_reject(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return a - sg_project(a, b);
}
