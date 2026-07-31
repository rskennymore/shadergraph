// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Euler rotation for the Mapping node.

// Build a rotation matrix from XYZ Euler angles in radians.
//
// Convention matches Blender's Mapping node: rotate about X first, then Y, then
// Z, i.e. R = Rz * Ry * Rx. Getting the order wrong produces a rotation that is
// correct for any single axis and subtly wrong the moment two are combined —
// which is the hardest version of this bug to notice, because the first thing
// anyone tries is one axis at a time.
//
// Derivation, so the constants below can be checked rather than trusted:
//
//   Rx = [1   0    0 ]   Ry = [ cy  0  sy]   Rz = [cz  -sz  0]
//        [0   cx  -sx]        [ 0   1  0 ]        [sz   cz  0]
//        [0   sx   cx]        [-sy  0  cy]        [0    0   1]
//
//   Ry*Rx = [ cy   sy*sx   sy*cx ]
//           [ 0    cx     -sx    ]
//           [-sy   cy*sx   cy*cx ]
//
//   R = Rz*(Ry*Rx) = [ cz*cy   cz*sy*sx - sz*cx   cz*sy*cx + sz*sx ]
//                    [ sz*cy   sz*sy*sx + cz*cx   sz*sy*cx - cz*sx ]
//                    [ -sy     cy*sx              cy*cx            ]
//
// NOTE: WGSL's `mat3x3` constructor takes COLUMNS, not rows, so the matrix above is
// transposed on the way in. Passing rows would give the inverse rotation for any
// asymmetric angle — right-looking, backwards.
fn sg_euler_xyz(radians: vec3<f32>) -> mat3x3<f32> {
    let c = cos(radians);
    let s = sin(radians);

    return mat3x3<f32>(
        // column 0
        vec3<f32>(c.z * c.y, s.z * c.y, -s.y),
        // column 1
        vec3<f32>(
            c.z * s.y * s.x - s.z * c.x,
            s.z * s.y * s.x + c.z * c.x,
            c.y * s.x,
        ),
        // column 2
        vec3<f32>(
            c.z * s.y * c.x + s.z * s.x,
            s.z * s.y * c.x - c.z * s.x,
            c.y * c.x,
        ),
    );
}
