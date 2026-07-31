// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Bump mapping from a height field, with no UVs and no mesh tangents.
//
// PROVENANCE: the surface-gradient formulation is Morten Mikkelsen's, from his
// published work on bump mapping unparametrised surfaces (2010). Written from
// the published description of the method; no shader code was copied.
//
// WHY THIS SHAPE: the usual way to perturb a normal needs a tangent basis, which
// needs UVs. This pipeline's premise is that meshes are never unwrapped, so that
// route is closed. Screen-space derivatives give an alternative basis: the
// fragment's own rate of change across a 2x2 pixel quad is enough to reconstruct
// the surface gradient of the height field, and it needs nothing authored.

// Perturb a normal by the gradient of a height field.
//
// `position` must be the true world-space surface position, NOT whatever
// coordinate the height field was sampled at. The derivative of a *mapped* or
// rotated coordinate would describe the mapping rather than the surface, and the
// reconstructed basis would be skewed by it. The height may come from any
// coordinate at all; the basis it is projected onto must be the real one.
//
// NOTE: `dpdx`/`dpdy` require uniform control flow. Every generated shader computes
// its graph as a straight-line chain of `let` bindings before the output block,
// so this is always called uniformly — but a future node that introduces a
// branch above a Bump would break that, silently, on some drivers only.
//
// NOTE: Derivatives are evaluated per 2x2 pixel quad, so the effect is resolution
// and distance dependent: a bump softens as a surface recedes and gets noisier
// at grazing angles. A height field with detail finer than a pixel will alias
// into shimmer no matter what is done here — the fix is always to scale the
// noise down, never to scale the bump up.
fn sg_bump(
    height: f32,
    strength: f32,
    distance: f32,
    normal: vec3<f32>,
    position: vec3<f32>,
) -> vec3<f32> {
    // Screen-space tangents of the actual surface. These two vectors span the
    // tangent plane at this fragment without anyone having authored one.
    let sigma_x = dpdx(position);
    let sigma_y = dpdy(position);

    // `r1` and `r2` are the tangent-plane basis rotated a quarter turn, which is
    // what converts a gradient measured in screen space into a displacement
    // along the surface. `det` carries the area scale of the quad — dividing it
    // out is what makes the result independent of how big the triangle is on
    // screen.
    let r1 = cross(sigma_y, normal);
    let r2 = cross(normal, sigma_x);
    let det = dot(sigma_x, r1);

    let dhdx = dpdx(height);
    let dhdy = dpdy(height);

    // `sign(det)` rather than a division: back-facing and mirrored geometry flip
    // the winding, and without this the bump would invert on exactly those
    // surfaces — which reads as "the normals are wrong on that one panel".
    let surface_gradient = sign(det) * (dhdx * r1 + dhdy * r2);

    // Negative `distance` inverts the effect, which is what Blender's separate
    // Invert checkbox does. One control instead of two, and the sign is visible
    // in the value rather than hidden in a flag.
    let perturbed = abs(det) * normal - distance * strength * surface_gradient;

    // A degenerate quad — a triangle exactly edge-on, or a fragment where the
    // height field is perfectly flat and the geometry is too — leaves nothing to
    // normalise. Returning the untouched normal is the correct answer there, and
    // normalising a zero vector would put NaN speckle along silhouettes.
    let len2 = dot(perturbed, perturbed);
    if (len2 < 1e-12) {
        return normal;
    }
    return perturbed * inverseSqrt(len2);
}
