// SPDX-License-Identifier: MIT OR Apache-2.0
//
// 3D Voronoi / cellular noise, returning the two nearest feature points.
//
// PROVENANCE: written from the published description of Steven Worley's
// cellular texture basis function ("A Cellular Texture Basis Function",
// SIGGRAPH 1996) — scatter one feature point per integer lattice cell, then
// search the neighbouring cells for the nearest few. No shader code was copied
// from any renderer.

struct SgVoronoi {
    f1_distance: f32,
    f1_position: vec3<f32>,
    f2_distance: f32,
    f2_position: vec3<f32>,
}

// Euclidean 3D Voronoi over the unit lattice.
//
// Returns both the nearest (F1) and second-nearest (F2) feature points, in the
// same space as `coord`. Two are returned rather than one because the
// interesting surface properties live in the *relationship* between them:
// `f2_distance - f1_distance` approaches zero exactly on a cell boundary, which
// is what makes it a groove mask, and the direction toward a cell centre is
// what gives an anisotropic surface its brush direction.
//
// `randomness` scales how far each feature point strays from its cell centre.
// At 0 the points sit on a perfect lattice; at 1 they fill their cells. It is
// clamped rather than trusted, because the 3x3x3 search below is only correct
// while every feature point stays inside its own cell — let a point wander
// further and the true nearest neighbour can fall outside the searched
// neighbourhood, producing discontinuities that look like cracks in the
// material and are miserable to diagnose after the fact.
fn sg_voronoi_f1f2_3d(coord: vec3<f32>, randomness: f32) -> SgVoronoi {
    let base = floor(coord);
    let jitter_scale = clamp(randomness, 0.0, 1.0);

    var result: SgVoronoi;
    // Larger than any distance reachable within the searched neighbourhood
    // (which is bounded by the diagonal of a 3x3x3 block), so the first real
    // candidate always wins. Deliberately not f32::MAX: subsequent arithmetic
    // on an unreplaced sentinel would overflow to infinity rather than merely
    // look wrong.
    result.f1_distance = 1e9;
    result.f2_distance = 1e9;
    result.f1_position = coord;
    result.f2_position = coord;

    for (var k: i32 = -1; k <= 1; k++) {
        for (var j: i32 = -1; j <= 1; j++) {
            for (var i: i32 = -1; i <= 1; i++) {
                let cell = base + vec3<f32>(f32(i), f32(j), f32(k));

                // Centre of the cell, displaced by a per-cell random offset in
                // [-0.5, 0.5) scaled by `randomness`. Offsetting from the centre
                // rather than the corner is what makes randomness = 0 produce an
                // even lattice instead of points piled on cell corners.
                let jitter = (sg_hash_cell(cell) - vec3<f32>(0.5)) * jitter_scale;
                let feature = cell + vec3<f32>(0.5) + jitter;

                let d = distance(coord, feature);

                // Insertion into a two-element sorted list. The `else if`
                // matters: without it, a new nearest point would overwrite F1
                // and then immediately be compared against F2 and inserted
                // twice, collapsing F1 and F2 onto the same point and making
                // every boundary-derived value read zero.
                if (d < result.f1_distance) {
                    result.f2_distance = result.f1_distance;
                    result.f2_position = result.f1_position;
                    result.f1_distance = d;
                    result.f1_position = feature;
                } else if (d < result.f2_distance) {
                    result.f2_distance = d;
                    result.f2_position = feature;
                }
            }
        }
    }

    return result;
}
