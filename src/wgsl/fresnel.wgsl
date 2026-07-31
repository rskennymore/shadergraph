// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Fresnel reflectance.
//
// PROVENANCE: Schlick's approximation, from the published description in
// Christophe Schlick, "An Inexpensive BRDF Model for Physically-based
// Rendering" (Eurographics 1994). Textbook material; no code copied.

// Approximate the fraction of light reflected rather than transmitted at a
// dielectric interface, given the index of refraction and the viewing angle.
//
// `normal` and `view` are normalised defensively rather than trusted. They
// usually arrive already normalised, but a graph is free to wire anything into
// these sockets, and an un-normalised input silently biases `cos_theta` — which
// does not error, it just makes the rim term wrong in a way that reads as a
// material tuning problem rather than a graph bug.
//
// `cos_theta` is clamped to [0, 1] because back-facing fragments produce a
// negative dot product, and `pow(negative, 5.0)` is undefined in WGSL.
fn sg_fresnel(ior: f32, normal: vec3<f32>, view: vec3<f32>) -> f32 {
    // Reflectance at normal incidence, from the IOR ratio against air.
    let f0_root = (1.0 - ior) / (1.0 + ior);
    let f0 = f0_root * f0_root;

    let cos_theta = clamp(dot(normalize(normal), normalize(view)), 0.0, 1.0);

    // Schlick: F0 + (1 - F0)(1 - cos O)^5. The fifth power is what makes the
    // effect stay near F0 across most of the surface and climb steeply only at
    // grazing angles.
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}
