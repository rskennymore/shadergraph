//! A color ramp: colors placed along a 0..1 axis, and the curve between them.
//!
//! # One evaluator, four interpolations
//!
//! Every mode here is the same degree-5 polynomial per segment, evaluated by the
//! same Horner chain. Only the coefficients differ, and three of the four modes
//! are degenerate cases of the fourth:
//!
//! | mode | coefficients |
//! |---|---|
//! | `Constant` | `[p₀, 0, 0, 0, 0, 0]` |
//! | `Linear` | `[p₀, Δ, 0, 0, 0, 0]` |
//! | `QuinticEase` | `[p₀, 0, 0, 10Δ, −15Δ, 6Δ]` |
//! | `QuinticSpline` | the full Hermite basis with estimated derivatives |
//!
//! That the ease row *is* Perlin's `6t⁵ − 15t⁴ + 10t³` fade is not a coincidence
//! and not a special case: it is exactly the quintic Hermite segment with zero
//! velocity and zero acceleration at both ends. Writing them as one path is
//! therefore honest rather than clever — there is genuinely only one curve here,
//! asked three different questions about its derivatives.
//!
//! The practical benefit is that emitted WGSL has the same shape whatever the
//! mode, so switching modes in an editor changes six numbers rather than
//! selecting a different code generator.
//!
//! # Ease versus spline
//!
//! `QuinticEase` has zero derivative *at* every stop, so each stop is a small
//! plateau and the ramp reads as a sequence of soft steps. Crucially it is
//! monotone on `[0,1]`, so **it cannot leave the gamut**: a channel between two
//! in-range stops stays in range.
//!
//! `QuinticSpline` estimates derivatives from neighboring stops instead, so the
//! curve flows through each stop without flattening. It is C² and interpolating,
//! which is a combination Blender's color band does not offer — its `CARDINAL`
//! is cubic and only C¹, and `B_SPLINE` does not pass through its stops at all.
//! The price is real: **an interpolating C² curve overshoots**, so a channel can
//! exit `[0,1]` between stops. [`ColorRamp::sample`] clamps, and so does the
//! generated shader.
//!
//! # Where the numbers live
//!
//! The coefficients are computed here, on the CPU, and *uploaded* — they are not
//! written into the generated WGSL. Only the segment count reaches the source,
//! because only the segment count changes the shape of the emitted function.
//!
//! So moving a stop, recoloring one, or switching interpolation mode all leave
//! the shader byte-identical and cost a buffer write; adding or removing a stop
//! recompiles. The mode being free is a direct consequence of the one-polynomial
//! design above: all four modes are the same Horner chain, so switching between
//! them really is just six different numbers.
//!
//! Computing them on the CPU rather than in the shader is not laziness.
//! `ColorRamp::derivatives_at` looks at a stop's *neighbors*, which is a
//! global operation over the stop list — doing it per fragment would mean
//! re-deriving the whole ramp millions of times a frame to get a value that
//! changes only when someone drags something.
//!
//! # color space
//!
//! Stops are **linear**, not sRGB, because that is what the renderer consumes and
//! what Blender interpolates in. Mixing sRGB-encoded values is the reason a
//! saturated red-to-green ramp turns muddy brown in the middle instead of passing
//! through a bright yellow — a mistake that is invisible in the graph and obvious
//! on the surface. Converting for a color picker is the editor's job, at the
//! edge, once.

use std::fmt::Write as _;

use crate::params::SlotValue;

/// One color placed on the ramp.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ColorStop {
    /// Position along the ramp. Conventionally within `0..=1`.
    pub position: f32,
    /// Linear RGBA. See the module docs on color space.
    pub color: [f32; 4],
}

impl ColorStop {
    /// A stop at `position` with the given linear RGBA color.
    pub fn new(position: f32, color: [f32; 4]) -> Self {
        Self { position, color }
    }
}

/// How the ramp gets from one stop to the next.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RampInterp {
    /// Hold each stop's color until the next. Hard edges, for masks and bands.
    Constant,
    /// Straight line between stops. The predictable default, and what Blender
    /// starts a color band with.
    #[default]
    Linear,
    /// Perlin's quintic fade between stops: smooth to the second derivative, and
    /// guaranteed not to overshoot.
    QuinticEase,
    /// A C² curve *through* the stops. No flat spots; can overshoot.
    QuinticSpline,
}

impl RampInterp {
    /// Every mode, in menu order, so an editor's dropdown is generated from the
    /// vocabulary rather than restating it.
    pub const ALL: &'static [RampInterp] = &[
        RampInterp::Constant,
        RampInterp::Linear,
        RampInterp::QuinticEase,
        RampInterp::QuinticSpline,
    ];

    /// Menu name, for an editor's dropdown.
    pub fn label(self) -> &'static str {
        match self {
            RampInterp::Constant => "Constant",
            RampInterp::Linear => "Linear",
            RampInterp::QuinticEase => "Quintic Ease",
            RampInterp::QuinticSpline => "Quintic Spline",
        }
    }
}

/// A color ramp.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ColorRamp {
    /// Sorted by position. [`ColorRamp::insert`] maintains that; the field is
    /// public because an editor legitimately rearranges stops wholesale, and
    /// [`ColorRamp::normalize`] restores order.
    pub stops: Vec<ColorStop>,
    /// How color travels between adjacent stops.
    pub interp: RampInterp,
}

impl Default for ColorRamp {
    fn default() -> Self {
        Self {
            stops: vec![
                ColorStop::new(0.0, [0.0, 0.0, 0.0, 1.0]),
                ColorStop::new(1.0, [1.0, 1.0, 1.0, 1.0]),
            ],
            interp: RampInterp::default(),
        }
    }
}

impl ColorRamp {
    /// Sort stops and pull apart any that coincide.
    ///
    /// Two stops at the same position would give a segment of zero width, and
    /// the derivative estimates divide by that width. Rather than guard every
    /// division, positions are separated by a hair — which is also what dragging
    /// one stop onto another should visually do.
    pub fn normalize(&mut self) {
        self.stops.sort_by(|a, b| {
            a.position
                .partial_cmp(&b.position)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for i in 1..self.stops.len() {
            let previous = self.stops[i - 1].position;
            if self.stops[i].position <= previous + MIN_GAP {
                self.stops[i].position = previous + MIN_GAP;
            }
        }
    }

    /// Add a stop, keeping the list ordered.
    pub fn insert(&mut self, stop: ColorStop) {
        self.stops.push(stop);
        self.normalize();
    }

    /// Remove a stop, unless it is one of the last two.
    ///
    /// A ramp needs two stops to be a ramp. Returning `false` rather than
    /// silently doing nothing lets an editor grey the control out.
    pub fn remove(&mut self, index: usize) -> bool {
        if self.stops.len() <= 2 || index >= self.stops.len() {
            return false;
        }
        self.stops.remove(index);
        true
    }

    /// Evaluate the ramp on the CPU.
    ///
    /// Matches what the generated shader computes, and exists so a preview
    /// widget draws the curve that will actually be rendered rather than an
    /// approximation of it.
    pub fn sample(&self, t: f32) -> [f32; 4] {
        match self.stops.len() {
            0 => return [0.0, 0.0, 0.0, 1.0],
            1 => return self.stops[0].color,
            _ => {}
        }

        let first = self.stops[0];
        let last = self.stops[self.stops.len() - 1];
        if t <= first.position {
            return first.color;
        }
        if t >= last.position {
            return last.color;
        }

        let segment = self
            .stops
            .windows(2)
            .position(|w| t < w[1].position)
            .unwrap_or(self.stops.len() - 2);

        let (start, end) = (self.stops[segment], self.stops[segment + 1]);
        let width = (end.position - start.position).max(MIN_GAP);
        let u = ((t - start.position) / width).clamp(0.0, 1.0);

        let coefficients = self.segment_coefficients(segment);
        let mut out = [0.0f32; 4];
        for (channel, slot) in out.iter_mut().enumerate() {
            let c = &coefficients[channel];
            // Horner, matching the generated WGSL exactly so CPU preview and GPU
            // result cannot disagree about evaluation order.
            let value = c[0] + u * (c[1] + u * (c[2] + u * (c[3] + u * (c[4] + u * c[5]))));
            // The spline mode can overshoot; a negative channel reaching the
            // renderer is worse than a clipped one.
            *slot = value.clamp(0.0, 1.0);
        }
        out
    }

    /// Polynomial coefficients for one segment, per channel.
    ///
    /// Indexed `[channel][power]`, in the local parameter `u ∈ [0,1]` across the
    /// segment rather than in `t`, which is what makes every segment's evaluation
    /// identical regardless of how wide it is.
    pub fn segment_coefficients(&self, segment: usize) -> [[f32; 6]; 4] {
        let (start, end) = (self.stops[segment], self.stops[segment + 1]);
        let width = (end.position - start.position).max(MIN_GAP);

        let mut out = [[0.0f32; 6]; 4];
        for (channel, row) in out.iter_mut().enumerate() {
            let p0 = start.color[channel];
            let p1 = end.color[channel];
            let delta = p1 - p0;

            *row = match self.interp {
                RampInterp::Constant => [p0, 0.0, 0.0, 0.0, 0.0, 0.0],
                RampInterp::Linear => [p0, delta, 0.0, 0.0, 0.0, 0.0],
                // Zero velocity and zero acceleration at both ends. Expanding
                // the Hermite basis under those conditions collapses it to
                // Perlin's fade, which is why this row is written out rather
                // than routed through the general case: the general case would
                // compute the same six numbers from twelve zeros.
                RampInterp::QuinticEase => [p0, 0.0, 0.0, 10.0 * delta, -15.0 * delta, 6.0 * delta],
                RampInterp::QuinticSpline => {
                    // Derivatives are estimated per stop in `t`, then scaled into
                    // the segment's local `u`. Skipping that scale is the classic
                    // way to get a spline that looks right only when every
                    // segment happens to be the same width.
                    let (v0, a0) = self.derivatives_at(segment, channel);
                    let (v1, a1) = self.derivatives_at(segment + 1, channel);
                    let (v0, v1) = (v0 * width, v1 * width);
                    let (a0, a1) = (a0 * width * width, a1 * width * width);

                    [
                        p0,
                        v0,
                        0.5 * a0,
                        10.0 * delta - 6.0 * v0 - 4.0 * v1 - 1.5 * a0 + 0.5 * a1,
                        -15.0 * delta + 8.0 * v0 + 7.0 * v1 + 1.5 * a0 - a1,
                        6.0 * delta - 3.0 * v0 - 3.0 * v1 - 0.5 * a0 + 0.5 * a1,
                    ]
                }
            };
        }
        out
    }

    /// Velocity and acceleration of one channel at one stop, in `t`.
    ///
    /// Central differences inside, one-sided at the ends, and zero acceleration
    /// at the ends — the "natural" boundary condition, chosen because the
    /// alternative (extrapolating curvature off the end) is where an
    /// interpolating spline overshoots hardest, and the ends are exactly where an
    /// overshoot is most visible.
    fn derivatives_at(&self, index: usize, channel: usize) -> (f32, f32) {
        let n = self.stops.len();
        let value = |i: usize| self.stops[i].color[channel];
        let position = |i: usize| self.stops[i].position;

        if n < 2 {
            return (0.0, 0.0);
        }

        if index == 0 {
            let h = (position(1) - position(0)).max(MIN_GAP);
            return ((value(1) - value(0)) / h, 0.0);
        }
        if index == n - 1 {
            let h = (position(n - 1) - position(n - 2)).max(MIN_GAP);
            return ((value(n - 1) - value(n - 2)) / h, 0.0);
        }

        let back = (position(index) - position(index - 1)).max(MIN_GAP);
        let forward = (position(index + 1) - position(index)).max(MIN_GAP);
        let slope_back = (value(index) - value(index - 1)) / back;
        let slope_forward = (value(index + 1) - value(index)) / forward;

        let velocity = (slope_forward * back + slope_back * forward) / (back + forward);
        let acceleration = 2.0 * (slope_forward - slope_back) / (back + forward);
        (velocity, acceleration)
    }

    /// How many segments this ramp has. Zero for a degenerate one-stop ramp.
    ///
    /// This is the *only* property of a ramp that reaches the generated source,
    /// and therefore the only edit that costs a recompile.
    pub fn segment_count(&self) -> usize {
        self.stops.len().saturating_sub(1)
    }

    /// Total uniform slots this ramp occupies: `7 · segments + 1`.
    pub fn slot_count(&self) -> usize {
        self.segment_count() * SLOTS_PER_SEGMENT + 1
    }

    /// The color returned past the last stop.
    fn tail_color(&self) -> [f32; 4] {
        self.stops
            .last()
            .map(|stop| stop.color)
            .unwrap_or([0.0, 0.0, 0.0, 1.0])
    }

    /// Everything the generated function reads, in slot order.
    ///
    /// Per segment: six coefficient rows — row *k* holds power *k* across the
    /// four channels — then one row of `(start, 1/width, end, 0)`. The inverse
    /// width is stored rather than the width so the shader multiplies instead of
    /// dividing, which is both faster and unable to divide by zero.
    ///
    /// One trailing row holds the color past the last stop. There is no leading
    /// row for the color *before* the first: every mode's `c0` is `p₀`, so the
    /// first segment's first coefficient row already is it. The tail cannot be
    /// derived the same way — `Constant` evaluates to `p₀` at `u = 1`, not `p₁`.
    pub fn slot_values(&self) -> Vec<SlotValue> {
        let mut rows = Vec::with_capacity(self.slot_count());

        for segment in 0..self.segment_count() {
            let c = self.segment_coefficients(segment);
            // An index loop on purpose: this is a transpose. `c` is
            // `[channel][power]` and each row emitted here is one power across
            // all four channels, so neither array can be iterated directly.
            #[allow(clippy::needless_range_loop)]
            for power in 0..6 {
                rows.push(SlotValue::Raw([
                    c[0][power],
                    c[1][power],
                    c[2][power],
                    c[3][power],
                ]));
            }

            let (start, end) = (self.stops[segment], self.stops[segment + 1]);
            let width = (end.position - start.position).max(MIN_GAP);
            rows.push(SlotValue::Raw([
                start.position,
                1.0 / width,
                end.position,
                0.0,
            ]));
        }

        rows.push(SlotValue::Raw(self.tail_color()));
        rows
    }

    /// Generate the WGSL function that evaluates this ramp.
    ///
    /// `base` is the first slot [`ColorRamp::slot_values`] was written to. It is
    /// the only ramp-specific number in the output: every coefficient is read
    /// from the uniform, so two ramps with the same segment count and the same
    /// base emit identical source however different they look.
    pub fn emit_wgsl(&self, name: &str, base: usize) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "fn {name}(t: f32) -> vec4<f32> {{");

        let segments = self.segment_count();
        let tail = base + segments * SLOTS_PER_SEGMENT;

        // A ramp with fewer than two stops is a constant. Still a slot read, so
        // that recoloring it stays as free as recoloring a real ramp.
        if segments == 0 {
            let _ = writeln!(s, "    return params.slots[{tail}];");
            let _ = writeln!(s, "}}");
            return s;
        }

        // Bounds are hoisted rather than read inline because each is used three
        // times (start, inverse width, end) and hoisting makes the branch chain
        // below readable as a chain.
        for segment in 0..segments {
            let _ = writeln!(
                s,
                "    let bounds{segment} = params.slots[{}];",
                base + segment * SLOTS_PER_SEGMENT + 6
            );
        }

        // Below the first stop, hold its color — which is segment 0's `c0`.
        let _ = writeln!(
            s,
            "    if (t <= bounds0.x) {{ return params.slots[{base}]; }}"
        );

        for segment in 0..segments {
            let c = base + segment * SLOTS_PER_SEGMENT;
            let _ = writeln!(s, "    if (t < bounds{segment}.z) {{");
            // Clamped for parity with `sample`, which clamps for the same reason:
            // a float edge case at a segment boundary must not evaluate the
            // polynomial outside the range it was fitted on.
            let _ = writeln!(
                s,
                "        let u = clamp((t - bounds{segment}.x) * bounds{segment}.y, 0.0, 1.0);"
            );
            // Horner, matching `sample` term for term.
            let _ = writeln!(
                s,
                "        return clamp(params.slots[{}] + u * (params.slots[{}] + u * \
                 (params.slots[{}] + u * (params.slots[{}] + u * (params.slots[{}] + u * \
                 params.slots[{}])))), vec4<f32>(0.0), vec4<f32>(1.0));",
                c,
                c + 1,
                c + 2,
                c + 3,
                c + 4,
                c + 5
            );
            let _ = writeln!(s, "    }}");
        }

        let _ = writeln!(s, "    return params.slots[{tail}];");
        let _ = writeln!(s, "}}");
        s
    }
}

/// Slots one segment occupies: six coefficient rows plus one bounds row.
const SLOTS_PER_SEGMENT: usize = 7;

/// Smallest gap allowed between two stops.
///
/// Big enough that dividing by it cannot produce an absurd slope, small enough
/// to be invisible on any ramp a person would draw.
const MIN_GAP: f32 = 1e-4;

#[cfg(test)]
mod tests {
    use super::*;

    fn two_stop(interp: RampInterp) -> ColorRamp {
        ColorRamp {
            stops: vec![
                ColorStop::new(0.0, [0.0, 0.0, 0.0, 1.0]),
                ColorStop::new(1.0, [1.0, 1.0, 1.0, 1.0]),
            ],
            interp,
        }
    }

    #[test]
    fn every_mode_hits_its_stops_exactly() {
        // Interpolation is about what happens *between* stops. A mode that
        // moves the stops themselves is broken however good the curve looks.
        for &interp in RampInterp::ALL {
            let ramp = ColorRamp {
                stops: vec![
                    ColorStop::new(0.0, [1.0, 0.0, 0.0, 1.0]),
                    ColorStop::new(0.3, [0.0, 1.0, 0.0, 1.0]),
                    ColorStop::new(1.0, [0.0, 0.0, 1.0, 1.0]),
                ],
                interp,
            };
            for stop in &ramp.stops {
                let got = ramp.sample(stop.position);
                for channel in 0..4 {
                    assert!(
                        (got[channel] - stop.color[channel]).abs() < 1e-5,
                        "{} missed the stop at {}: {got:?} != {:?}",
                        interp.label(),
                        stop.position,
                        stop.color
                    );
                }
            }
        }
    }

    #[test]
    fn quintic_ease_is_perlins_fade() {
        // The load-bearing identity: a quintic Hermite segment with zero
        // velocity and zero acceleration at both ends *is* 6t^5 - 15t^4 + 10t^3.
        // If this drifts, the two "quintic" modes have stopped being the same
        // curve asked different questions.
        let ramp = two_stop(RampInterp::QuinticEase);
        for step in 0..=20 {
            let t = step as f32 / 20.0;
            let expected = 6.0 * t.powi(5) - 15.0 * t.powi(4) + 10.0 * t.powi(3);
            assert!(
                (ramp.sample(t)[0] - expected).abs() < 1e-5,
                "at t={t}: {} != {expected}",
                ramp.sample(t)[0]
            );
        }
    }

    #[test]
    fn quintic_ease_is_flat_at_the_stops_and_the_spline_is_not() {
        // The visible difference between the two modes, stated as a measurement
        // rather than left to the eye.
        let probe = |interp: RampInterp| {
            let ramp = ColorRamp {
                stops: vec![
                    ColorStop::new(0.0, [0.0; 4]),
                    ColorStop::new(0.5, [0.5, 0.5, 0.5, 0.5]),
                    ColorStop::new(1.0, [1.0; 4]),
                ],
                interp,
            };
            // Slope across the middle stop.
            (ramp.sample(0.51)[0] - ramp.sample(0.49)[0]) / 0.02
        };

        // Ease flattens at every stop; the spline flows through this one at the
        // average slope of its neighbours, which here is 1.0.
        assert!(probe(RampInterp::QuinticEase) < 0.2);
        assert!((probe(RampInterp::QuinticSpline) - 1.0).abs() < 0.1);
    }

    #[test]
    fn ease_never_leaves_the_gamut_even_where_the_spline_would() {
        // A step in the middle is the case that makes an interpolating C² curve
        // overshoot. Ease is monotone and cannot; the spline is clamped, which
        // is the guarantee that actually reaches the renderer.
        let stops = vec![
            ColorStop::new(0.0, [0.0, 0.0, 0.0, 1.0]),
            ColorStop::new(0.45, [0.0, 0.0, 0.0, 1.0]),
            ColorStop::new(0.55, [1.0, 1.0, 1.0, 1.0]),
            ColorStop::new(1.0, [1.0, 1.0, 1.0, 1.0]),
        ];

        for &interp in &[RampInterp::QuinticEase, RampInterp::QuinticSpline] {
            let ramp = ColorRamp {
                stops: stops.clone(),
                interp,
            };
            for step in 0..=200 {
                let t = step as f32 / 200.0;
                let c = ramp.sample(t);
                for channel in 0..4 {
                    assert!(
                        (0.0..=1.0).contains(&c[channel]),
                        "{} left the gamut at t={t}: {c:?}",
                        interp.label()
                    );
                }
            }
        }
    }

    #[test]
    fn normalize_orders_stops_and_separates_coincident_ones() {
        let mut ramp = ColorRamp {
            stops: vec![
                ColorStop::new(0.8, [1.0; 4]),
                ColorStop::new(0.2, [0.0; 4]),
                ColorStop::new(0.2, [0.5; 4]),
            ],
            interp: RampInterp::Linear,
        };
        ramp.normalize();

        assert!(ramp.stops[0].position < ramp.stops[1].position);
        assert!(ramp.stops[1].position < ramp.stops[2].position);
    }

    #[test]
    fn a_ramp_never_drops_below_two_stops() {
        let mut ramp = ColorRamp::default();
        assert!(
            !ramp.remove(0),
            "removing from a two-stop ramp must be refused"
        );

        ramp.insert(ColorStop::new(0.5, [1.0, 0.0, 0.0, 1.0]));
        assert!(ramp.remove(1));
        assert_eq!(ramp.stops.len(), 2);
    }

    /// The segment polynomial really is the quintic Hermite basis.
    ///
    /// `segment_coefficients` returns monomial coefficients, which is what the
    /// shader wants but which hides the algebra that produced them. This
    /// evaluates the published basis functions directly and checks the two agree
    /// — so a transposed sign in the expansion fails here rather than becoming a
    /// material that looks *almost* right.
    ///
    /// Basis on `s ∈ [0,1]`:
    /// `H₀ = 1 − 10s³ + 15s⁴ − 6s⁵`, `H₁ = s − 6s³ + 8s⁴ − 3s⁵`,
    /// `H₂ = s²/2 − 3s³/2 + 3s⁴/2 − s⁵/2`, `H₃ = 10s³ − 15s⁴ + 6s⁵`,
    /// `H₄ = −4s³ + 7s⁴ − 3s⁵`, `H₅ = s³/2 − s⁴ + s⁵/2`.
    #[test]
    fn the_segment_polynomial_is_the_hermite_basis() {
        let ramp = ColorRamp {
            stops: vec![
                ColorStop::new(0.0, [0.2, 0.0, 0.0, 1.0]),
                ColorStop::new(0.35, [0.9, 0.0, 0.0, 1.0]),
                ColorStop::new(1.0, [0.4, 0.0, 0.0, 1.0]),
            ],
            interp: RampInterp::QuinticSpline,
        };

        for segment in 0..ramp.stops.len() - 1 {
            let (start, end) = (ramp.stops[segment], ramp.stops[segment + 1]);
            let h = end.position - start.position;

            // Same derivatives the coefficients were built from, scaled into the
            // segment's local parameter the same way.
            let (v0, a0) = ramp.derivatives_at(segment, 0);
            let (v1, a1) = ramp.derivatives_at(segment + 1, 0);
            let (v0, v1) = (v0 * h, v1 * h);
            let (a0, a1) = (a0 * h * h, a1 * h * h);
            let (p0, p1) = (start.color[0], end.color[0]);

            let c = ramp.segment_coefficients(segment)[0];

            for step in 0..=20 {
                let s = step as f32 / 20.0;
                let (s2, s3, s4, s5) = (s * s, s * s * s, s.powi(4), s.powi(5));

                let basis = p0 * (1.0 - 10.0 * s3 + 15.0 * s4 - 6.0 * s5)
                    + v0 * (s - 6.0 * s3 + 8.0 * s4 - 3.0 * s5)
                    + a0 * (0.5 * s2 - 1.5 * s3 + 1.5 * s4 - 0.5 * s5)
                    + p1 * (10.0 * s3 - 15.0 * s4 + 6.0 * s5)
                    + v1 * (-4.0 * s3 + 7.0 * s4 - 3.0 * s5)
                    + a1 * (0.5 * s3 - s4 + 0.5 * s5);

                let monomial = c[0] + s * (c[1] + s * (c[2] + s * (c[3] + s * (c[4] + s * c[5]))));

                assert!(
                    (basis - monomial).abs() < 1e-4,
                    "segment {segment} at s={s}: basis {basis} != monomial {monomial}"
                );
            }
        }
    }

    fn three_stop_spline() -> ColorRamp {
        ColorRamp {
            stops: vec![
                ColorStop::new(0.0, [0.9, 0.1, 0.1, 1.0]),
                ColorStop::new(0.4, [0.1, 0.8, 0.2, 1.0]),
                ColorStop::new(1.0, [0.2, 0.2, 0.9, 1.0]),
            ],
            interp: RampInterp::QuinticSpline,
        }
    }

    #[test]
    fn the_uploaded_slots_carry_what_the_cpu_sampler_uses() {
        // The successor to the old "every coefficient appears in the source"
        // check, which cannot be written any more because the coefficients are
        // no longer in the source. Same class of bug caught — the two evaluators
        // written from the same idea and differing in a coefficient, which no
        // amount of looking at a surface reliably reveals — but asserted against
        // the buffer the GPU actually reads.
        let ramp = three_stop_spline();
        let rows = ramp.slot_values();
        assert_eq!(rows.len(), ramp.slot_count());
        assert_eq!(rows.len(), 2 * SLOTS_PER_SEGMENT + 1);

        for segment in 0..ramp.segment_count() {
            let c = ramp.segment_coefficients(segment);
            for power in 0..6 {
                let row = rows[segment * SLOTS_PER_SEGMENT + power].packed();
                for channel in 0..4 {
                    assert!(
                        (row[channel] - c[channel][power]).abs() < 1e-6,
                        "segment {segment} power {power} channel {channel}: \
                         slot has {}, `sample` uses {}",
                        row[channel],
                        c[channel][power]
                    );
                }
            }

            // The bounds row the shader branches and rescales on.
            let bounds = rows[segment * SLOTS_PER_SEGMENT + 6].packed();
            let (start, end) = (ramp.stops[segment], ramp.stops[segment + 1]);
            assert_eq!(bounds[0], start.position);
            assert_eq!(bounds[2], end.position);
            assert!((bounds[1] * (end.position - start.position) - 1.0).abs() < 1e-5);
        }

        assert_eq!(rows[2 * SLOTS_PER_SEGMENT].packed(), [0.2, 0.2, 0.9, 1.0]);
    }

    #[test]
    fn only_the_segment_count_reaches_the_source() {
        // The claim the whole design makes: dragging a stop, recoloring it, or
        // switching interpolation mode is a buffer write, and only adding or
        // removing a stop is a recompile.
        let original = three_stop_spline().emit_wgsl("sg_ramp", 4);

        let mut moved = three_stop_spline();
        moved.stops[1].position = 0.75;
        assert_eq!(moved.emit_wgsl("sg_ramp", 4), original, "moving a stop");

        let mut recolored = three_stop_spline();
        recolored.stops[1].color = [1.0, 1.0, 0.0, 1.0];
        assert_eq!(recolored.emit_wgsl("sg_ramp", 4), original, "recoloring");

        for &interp in RampInterp::ALL {
            let mut switched = three_stop_spline();
            switched.interp = interp;
            assert_eq!(
                switched.emit_wgsl("sg_ramp", 4),
                original,
                "switching to {}",
                interp.label()
            );
        }

        let mut grown = three_stop_spline();
        grown.insert(ColorStop::new(0.7, [0.0; 4]));
        assert_ne!(grown.emit_wgsl("sg_ramp", 4), original, "adding a stop");
    }

    #[test]
    fn a_degenerate_ramp_still_reads_a_slot() {
        // One stop is zero segments. It must still be a slot read rather than a
        // baked literal, or recoloring a collapsed ramp would recompile where
        // recoloring a real one does not.
        let ramp = ColorRamp {
            stops: vec![ColorStop::new(0.0, [0.3, 0.4, 0.5, 1.0])],
            interp: RampInterp::Linear,
        };
        assert_eq!(ramp.slot_count(), 1);
        assert_eq!(ramp.slot_values()[0].packed(), [0.3, 0.4, 0.5, 1.0]);
        assert!(ramp.emit_wgsl("sg_ramp", 9).contains("params.slots[9]"));
    }
}
