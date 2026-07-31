// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Integer avalanche hashing for procedural noise.
//
// PROVENANCE: written from published descriptions of multiply-xorshift bit
// mixing. The xorshift/multiply/xorshift construction is the standard shape of
// an integer finaliser (the same form as SplitMix's finaliser and Wang's
// integer hash); the specific constants are the `lowbias32` pair from Chris
// Wellons' "Prospecting for Hash Functions", which the author released into the
// public domain. No shader code was copied from any renderer.

// Avalanche a single 32-bit integer.
//
// Three rounds, each an xor-with-right-shift followed by an odd-constant
// multiply. The xorshift moves entropy from the high bits (where multiplication
// concentrates it) back down into the low bits; the multiply then spreads it up
// again. Two full rounds plus a final xorshift is the minimum that passes
// avalanche testing — one round leaves the lowest bits visibly correlated,
// which in a Voronoi shows up as feature points aligning on faint grid lines.
fn sg_hash_u32(x_in: u32) -> u32 {
    var x = x_in;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

// Hash an integer lattice cell to three independent values in [0, 1).
//
// The cell coordinate arrives as a float because it came from `floor()`, and is
// bitcast through a signed integer rather than converted directly to `u32`:
// negative cells are ordinary (any surface at negative world coordinates lives
// in them), and `u32(-1.0)` is undefined in WGSL whereas `i32(-1.0)` is not.
//
// The three channels are decorrelated by re-hashing with distinct odd constants
// rather than by hashing three different tuples, which costs two extra rounds
// instead of two extra full hashes. The constants are the golden-ratio and
// SplitMix increments — any odd values with well-spread bits would do; these
// are conventional and easy to recognise.
fn sg_hash_cell(cell: vec3<f32>) -> vec3<f32> {
    let c = bitcast<vec3<u32>>(vec3<i32>(cell));

    // Fold the three axes into one value first, so that swapping two
    // coordinates gives an unrelated result. Hashing each axis separately and
    // summing would make the field symmetric about the diagonal — a subtle
    // artifact that is very visible on flat panels.
    let h0 = sg_hash_u32(c.x ^ sg_hash_u32(c.y ^ sg_hash_u32(c.z)));
    let h1 = sg_hash_u32(h0 ^ 0x9e3779b9u);
    let h2 = sg_hash_u32(h1 ^ 0x85ebca6bu);

    // Divide by 2^32 rather than 2^32 - 1 so the result is a half-open [0, 1)
    // range. A closed range would let a feature point land exactly on a cell
    // boundary, where it belongs to two cells at once.
    return vec3<f32>(vec3<u32>(h0, h1, h2)) * (1.0 / 4294967296.0);
}
