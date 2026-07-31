// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Fractal Brownian motion over Perlin noise, with a runtime octave count.
//
// Only the LIVE variant of the noise node pulls this in. The unrolled variant
// generates its own straight-line equivalent per node — see `NoiseOctaves`.

// Sum octaves of gradient noise, output remapped to [0, 1].
//
// The loop bound is `detail`, which arrives from a uniform slot, so dragging the
// detail slider is a buffer write rather than a recompile. That is the whole
// point of this variant, and it is not free: a loop whose trip count the shader
// compiler cannot see is harder to optimise than an unrolled one.
//
// The `break` matters more than it looks. Weights fall to zero once the octave
// index passes `detail`, so a graph running at detail 2 does three iterations,
// not eight — which means this is often FASTER than an unrolled node capped at
// eight, despite the dynamic bound. Worth measuring rather than assuming.
fn sg_fbm_3d(p: vec3<f32>, detail: f32, roughness: f32, lacunarity: f32) -> f32 {
    var sum = 0.0;
    var norm = 0.0;
    var amp = 1.0;
    var freq = 1.0;

    for (var i = 0; i < SG_MAX_OCTAVES; i = i + 1) {
        let w = amp * sg_octave_weight(detail, i);
        if (w <= 0.0) {
            break;
        }
        sum = sum + w * sg_perlin_3d(p * freq);
        norm = norm + w;
        amp = amp * roughness;
        freq = freq * lacunarity;
    }

    // Normalising by the accumulated weight rather than by a closed form keeps
    // the output range stable as `roughness` changes. Without it, a low
    // roughness would make the result hug 0.5 and a high one would blow past
    // the range, so the two controls would not be independent.
    if (norm <= 0.0) {
        return 0.5;
    }
    return sum / norm * 0.5 + 0.5;
}
