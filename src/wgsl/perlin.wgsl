// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Perlin gradient noise, and the per-octave weighting both fBm variants share.
//
// PROVENANCE: written from Ken Perlin's published description of gradient noise
// (1985, and the 2002 "Improving Noise" revision that introduced the quintic
// fade). No shader code was copied from any renderer: Cycles serves only as a
// correctness reference (compare rendered output), never as a source, because
// copying its Apache-2.0 code would sink this crate's MIT half.

// Largest number of octaves either fBm path will evaluate.
//
// NOTE: Mirrored by `node::MAX_NOISE_OCTAVES` on the Rust side, which bounds the
// unrolled variant's octave count. `the_octave_cap_agrees_across_the_language_boundary`
// pins the two together; a silent disagreement would mean the unrolled node
// generating more octaves than the dynamic one can reach, so the two would stop
// being comparable — which is the entire reason both exist.
const SG_MAX_OCTAVES: i32 = 8;

// A pseudo-random unit vector for one lattice cell.
//
// The hash gives three values in [0,1); those map to a point in the [-1,1] cube
// and are then normalised onto the sphere.
//
// NOTE: Normalising a uniform cube sample is very slightly biased toward the cube's
// corners — the eight diagonal directions are marginally more likely than the
// axial ones. Perlin's 2002 revision avoids this by picking from twelve fixed
// gradients on the cube's edges. That is measurably better for noise examined on
// its own and invisible under the kind of layered grunge this is for, so the
// cheaper form is used deliberately rather than by oversight. If a directional
// artifact ever does show up on a flat panel, the twelve-gradient set is the fix.
fn sg_gradient(cell: vec3<f32>) -> vec3<f32> {
    let h = sg_hash_cell(cell) * 2.0 - 1.0;
    let len2 = dot(h, h);
    // A cell whose hash lands almost exactly at the cube centre would normalise
    // to NaN and put a black speck on the surface. Vanishingly unlikely and
    // trivially cheap to exclude.
    if (len2 < 1e-8) {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    return h * inverseSqrt(len2);
}

// Gradient noise at a point, in roughly [-1, 1].
fn sg_perlin_3d(p: vec3<f32>) -> f32 {
    let cell = floor(p);
    let f = p - cell;

    // KEY: Perlin's quintic fade: 6t^5 - 15t^4 + 10t^3. This is the SAME polynomial
    // as `RampInterp::QuinticEase` in the color ramp, and for the same reason —
    // it is the quintic Hermite segment with zero velocity AND zero acceleration
    // at both ends. Cubic smoothstep would do here too, but its second derivative
    // is discontinuous at the lattice, which shows up as faint grid lines once
    // the noise drives a normal.
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    // Dot each corner's gradient with the offset from that corner to the point.
    let g000 = dot(sg_gradient(cell + vec3<f32>(0.0, 0.0, 0.0)), f - vec3<f32>(0.0, 0.0, 0.0));
    let g100 = dot(sg_gradient(cell + vec3<f32>(1.0, 0.0, 0.0)), f - vec3<f32>(1.0, 0.0, 0.0));
    let g010 = dot(sg_gradient(cell + vec3<f32>(0.0, 1.0, 0.0)), f - vec3<f32>(0.0, 1.0, 0.0));
    let g110 = dot(sg_gradient(cell + vec3<f32>(1.0, 1.0, 0.0)), f - vec3<f32>(1.0, 1.0, 0.0));
    let g001 = dot(sg_gradient(cell + vec3<f32>(0.0, 0.0, 1.0)), f - vec3<f32>(0.0, 0.0, 1.0));
    let g101 = dot(sg_gradient(cell + vec3<f32>(1.0, 0.0, 1.0)), f - vec3<f32>(1.0, 0.0, 1.0));
    let g011 = dot(sg_gradient(cell + vec3<f32>(0.0, 1.0, 1.0)), f - vec3<f32>(0.0, 1.0, 1.0));
    let g111 = dot(sg_gradient(cell + vec3<f32>(1.0, 1.0, 1.0)), f - vec3<f32>(1.0, 1.0, 1.0));

    // Trilinear blend along the faded axes.
    let x00 = mix(g000, g100, u.x);
    let x10 = mix(g010, g110, u.x);
    let x01 = mix(g001, g101, u.x);
    let x11 = mix(g011, g111, u.x);
    let y0 = mix(x00, x10, u.y);
    let y1 = mix(x01, x11, u.y);

    // 3D gradient noise with unit gradients peaks near sqrt(3)/2 ~= 0.866, not
    // 1.0. Scaling by its reciprocal makes the output span [-1, 1] so that
    // `roughness` behaves the same here as it would for any other basis.
    return mix(y0, y1, u.z) * 1.1547005;
}

// How much octave `octave` contributes at a given `detail`.
//
// KEY: Shared by BOTH fBm variants — the dynamic loop calls it per iteration and
// the unrolled generator calls it per unrolled step. That sharing is not tidiness:
// it is what makes the two provably the same function, so switching between them
// compares a dynamic loop against straight-line code and nothing else.
//
// Octave 0 is always full. Each subsequent octave fades in over one unit of
// detail, so `detail` is continuous rather than a step count — dragging it does
// not pop an octave into existence.
fn sg_octave_weight(detail: f32, octave: i32) -> f32 {
    return clamp(detail - f32(octave) + 1.0, 0.0, 1.0);
}
