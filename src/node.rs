//! Node kinds, their sockets, and how each one turns into WGSL.
//!
//! # Why the node set grows slowly
//!
//! Every node here earned its place by being needed for a real material. Every
//! additional node is surface area that nothing exercises, and an unexercised
//! node in a compiler is worse than a missing one: it looks available, and the
//! failure shows up as a shader that compiles and renders wrongly.
//!
//! # Deliberate deviations from Blender's node set
//!
//! The socket *names* mirror Blender's so that muscle memory transfers, but
//! three things differ on purpose:
//!
//! 1. **Polymorphic nodes are split by type.** Blender's `Mix` carries four A/B
//!    socket pairs and hides the irrelevant three behind a `data_type` dropdown.
//!    Here that is `MixColor`, with `MixFloat` and `MixVector` alongside it.
//!    An editor can auto-insert a conversion node when a user wires mismatched
//!    sockets; the compiler never needs to know about it.
//! 2. **`VectorMath` is split by output type.** Blender's has both a `Vector`
//!    and a `Value` output, only one of which is meaningful for a given
//!    operation. Splitting into [`NodeKind::VectorMath`] (vector out) and
//!    [`NodeKind::VectorToFloat`] (float out) keeps every node single-output,
//!    which in turn keeps the emitter's SSA bindings trivial.
//! 3. **Nothing is implicit.** Blender's texture nodes fall back to generated
//!    coordinates when their `Vector` input is unconnected, and `Fresnel` reads
//!    the incoming ray without exposing it. Here those are ordinary sockets that
//!    must be wired. It is more typing in a handwritten graph and less magic in
//!    the compiler; implicit-default sugar belongs in the editor, if anywhere.

use crate::params::SlotValue;
use crate::ramp::ColorRamp;
use crate::value::{Value, ValueType};

/// A WGSL support function that a node depends on.
///
/// Snippets are emitted once each, in the fixed order of [`Snippet::ALL`],
/// which is also their dependency order — WGSL requires a function to be
/// declared before use, and `Voronoi` calls into `Hash`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Snippet {
    /// The integer hash underlying every stochastic node.
    Hash,
    /// Cellular noise: nearest and second-nearest feature points.
    Voronoi,
    /// Gradient noise and the shared per-octave weighting. Needed by both noise
    /// variants, because the unrolled one still calls `sg_perlin_3d`.
    Perlin,
    /// The dynamic-octave fBm loop. Only the live noise variant needs it.
    Fbm,
    /// Blend modes too involved to emit inline. Only `Overlay` today.
    Blend,
    /// Euler rotation, for the Mapping node.
    Mapping,
    /// Closed-form gradient fields. One snippet for all seven forms.
    Gradient,
    /// Screen-space-derivative bump mapping.
    Bump,
    /// Vector helpers shared by the vector-math nodes.
    Vector,
    /// View-angle falloff.
    Fresnel,
    /// Lerping two evaluated surfaces. Only a graph containing a `MixSurface`
    /// needs it — the `SgSurface` struct itself lives in the emitted header,
    /// because every graph ends in one.
    MixSurface,
}

impl Snippet {
    /// Every snippet, in emission order. Dependencies must precede dependents.
    pub const ALL: &'static [Snippet] = &[
        Snippet::Hash,
        Snippet::Voronoi,
        Snippet::Perlin,
        Snippet::Fbm,
        Snippet::Blend,
        Snippet::Mapping,
        Snippet::Gradient,
        Snippet::Bump,
        Snippet::Vector,
        Snippet::Fresnel,
        Snippet::MixSurface,
    ];

    /// The WGSL text of this snippet, compiled into the binary.
    pub fn source(self) -> &'static str {
        match self {
            Snippet::Hash => include_str!("wgsl/hash.wgsl"),
            Snippet::Voronoi => include_str!("wgsl/voronoi.wgsl"),
            Snippet::Perlin => include_str!("wgsl/perlin.wgsl"),
            Snippet::Fbm => include_str!("wgsl/fbm.wgsl"),
            Snippet::Blend => include_str!("wgsl/blend.wgsl"),
            Snippet::Mapping => include_str!("wgsl/mapping.wgsl"),
            Snippet::Gradient => include_str!("wgsl/gradient.wgsl"),
            Snippet::Bump => include_str!("wgsl/bump.wgsl"),
            Snippet::Vector => include_str!("wgsl/vector.wgsl"),
            Snippet::Fresnel => include_str!("wgsl/fresnel.wgsl"),
            Snippet::MixSurface => include_str!("wgsl/surface.wgsl"),
        }
    }
}

/// Largest octave count the noise node will evaluate.
///
/// NOTE: Mirrored by `SG_MAX_OCTAVES` in `wgsl/perlin.wgsl`; see the comment there
/// for why they must agree.
pub const MAX_NOISE_OCTAVES: u8 = 8;

/// How a noise node's octaves get evaluated.
///
/// The two produce **identical output** — they call the same `sg_perlin_3d` and
/// the same `sg_octave_weight`. The only difference is whether the summation is
/// a loop the shader compiler cannot bound or straight-line code it can. That is
/// exactly what makes them worth comparing: any difference you see is the loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NoiseOctaves {
    /// Octave count comes from the `Detail` socket at runtime.
    ///
    /// Dragging Detail is a buffer write, never a recompile — the same property
    /// every other value in this compiler has, which is why it is the default.
    /// The loop short-circuits once the weights reach zero, so low detail costs
    /// few iterations despite the bound being dynamic.
    Live,
    /// Octaves unrolled at compile time, up to this many.
    ///
    /// `Detail` still works and still fades the last octave in — it simply
    /// cannot exceed the baked ceiling. Changing the ceiling recompiles.
    /// Straight-line, no loop, and no early exit, so this pays for every octave
    /// whether the detail setting reaches it or not.
    Unrolled(u8),
}

impl NoiseOctaves {
    /// Short human-readable name, for an editor's node header.
    pub fn label(self) -> String {
        match self {
            NoiseOctaves::Live => "Live".to_string(),
            NoiseOctaves::Unrolled(n) => format!("Unrolled ×{n}"),
        }
    }

    /// Octave count, clamped to what the shared cap allows.
    ///
    /// At least one: an unrolled noise with zero octaves is a constant, which is
    /// never what someone reaching for a noise node meant.
    pub fn unrolled_count(self) -> u8 {
        match self {
            NoiseOctaves::Live => 0,
            NoiseOctaves::Unrolled(n) => n.clamp(1, MAX_NOISE_OCTAVES),
        }
    }
}

/// What an input socket evaluates to when nothing is wired into it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SocketDefault {
    /// A literal constant.
    Value(Value),
    /// Nothing sensible to fall back to; leaving it unwired is an error caught
    /// by [`crate::Graph::validate`].
    Required,
    /// A WGSL expression drawn from the fragment context.
    ///
    /// This exists for the handful of sockets where the *renderer* has a
    /// meaningful default that no literal can express — `PbrOutput`'s normal
    /// falling back to the interpolated geometric normal being the motivating
    /// case. Without it, every graph would have to wire a Geometry node into
    /// the output just to say "nothing unusual here", and forgetting to would
    /// be an error rather than the common case.
    ///
    /// It is deliberately *not* used to give texture nodes implicit coordinates
    /// the way Blender does. That fallback is convenience rather than renderer
    /// semantics, and hiding it costs more in confusion than it saves in wiring.
    Context(&'static str),
}

/// One input or output socket.
#[derive(Clone, Copy, Debug)]
pub struct SocketDef {
    /// Display name, mirroring Blender's where an equivalent exists.
    pub name: &'static str,
    /// WGSL-safe identifier. Doubles as the struct field name for nodes whose
    /// outputs arrive as a struct (see [`Emission::Struct`]).
    pub ident: &'static str,
    /// The type this socket carries; wires only connect matching types.
    pub ty: ValueType,
    /// Value used when no wire arrives. Meaningless on output sockets, where it
    /// is always [`SocketDefault::Required`].
    pub default: SocketDefault,
}

/// How a node turns into WGSL.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Emission {
    /// Reads straight from fragment context; one fixed expression per output,
    /// no binding emitted. Used by [`NodeKind::Geometry`].
    Inline(&'static [&'static str]),
    /// A parameter. One binding that reads the node's slot in the uniform.
    ///
    /// Was a literal until parameters moved into a uniform buffer; see
    /// [`crate::params`] for why. The variant kept a name of its own because
    /// "which slot" is emitter state that no other emission needs.
    Param,
    /// A texture read. One binding that samples this node's texture slot.
    ///
    /// Not [`Emission::Expr`], because the expression depends on emitter state
    /// the node cannot see: which of the [`crate::params::TEXTURE_SLOTS`]
    /// bindings this node was assigned. The emitter writes the `textureSample`
    /// call itself, the same way it writes a `Param`'s slot read — and for the
    /// same reason the slot index is the emitter's business, not the node's.
    Texture,
    /// One binding whose right-hand side is built from the input expressions.
    Expr,
    /// One binding holding a struct; outputs are read as `.field` using each
    /// output socket's `ident`.
    Struct,
    /// The node generates a WGSL function of its own, and the binding calls it.
    ///
    /// Every other emission turns a node into an expression built from fixed
    /// text. This one exists for nodes whose code depends on *data they carry* —
    /// a color ramp's stop list, later a curve's control points — where the
    /// generated function differs from instance to instance and so cannot come
    /// from the fixed [`Snippet`] library.
    ///
    /// The function is emitted once per node, named after it, ahead of the
    /// fragment entry point. See [`NodeKind::function`].
    Function,
    /// The graph root. The emitter writes the terminating block itself.
    Root,
}

/// Scalar operation performed by a [`NodeKind::Math`] node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MathOp {
    /// `a + b`.
    Add,
    /// `a - b`.
    Subtract,
    /// `a * b`.
    Multiply,
    /// `a / b`.
    Divide,
    /// `pow(a, b)`.
    Power,
    /// `min(a, b)`.
    Minimum,
    /// `max(a, b)`.
    Maximum,
    /// Hermite interpolation between two edges. Blender expresses this through
    /// its Map Range node rather than Math; it is on Math here because it is
    /// the natural spelling of "fade this factor in over a range" and the
    /// alternative is a Map Range node that does nothing else.
    SmoothStep,
}

impl MathOp {
    /// Every operation, in menu order.
    ///
    /// Exists so an editor's dropdown is generated from the vocabulary rather
    /// than restating it. A hand- ritten list in the UI is a list that silently
    /// stops matching the compiler the first time an operation is added here.
    pub const ALL: &'static [MathOp] = &[
        MathOp::Add,
        MathOp::Subtract,
        MathOp::Multiply,
        MathOp::Divide,
        MathOp::Power,
        MathOp::Minimum,
        MathOp::Maximum,
        MathOp::SmoothStep,
    ];

    /// Menu name, for an editor's dropdown.
    pub fn label(self) -> &'static str {
        match self {
            MathOp::Add => "Add",
            MathOp::Subtract => "Subtract",
            MathOp::Multiply => "Multiply",
            MathOp::Divide => "Divide",
            MathOp::Power => "Power",
            MathOp::Minimum => "Minimum",
            MathOp::Maximum => "Maximum",
            MathOp::SmoothStep => "Smooth Step",
        }
    }
}

/// Vector-in, vector-out operation performed by a [`NodeKind::VectorMath`] node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VectorOp {
    /// `a + b`, componentwise.
    Add,
    /// `a - b`, componentwise.
    Subtract,
    /// `a * b`, componentwise.
    Multiply,
    /// `v * scale`, one scalar over all components.
    Scale,
    /// `v / length(v)`.
    Normalize,
    /// Component of A along B.
    Project,
    /// Component of A perpendicular to B — i.e. A with its B-parallel part
    /// removed. Blender has no such operation and expects Project followed by
    /// Subtract. It is first-class here because flattening a vector into a
    /// surface is the single most common thing a shader does with a normal,
    /// and spending two nodes on it every time obscures the graph.
    Reject,
    /// Mirror A about the plane whose normal is B.
    Reflect,
}

impl VectorOp {
    /// Every operation, in menu order. See [`MathOp::ALL`].
    pub const ALL: &'static [VectorOp] = &[
        VectorOp::Add,
        VectorOp::Subtract,
        VectorOp::Multiply,
        VectorOp::Scale,
        VectorOp::Normalize,
        VectorOp::Project,
        VectorOp::Reject,
        VectorOp::Reflect,
    ];

    /// Menu name, for an editor's dropdown.
    pub fn label(self) -> &'static str {
        match self {
            VectorOp::Add => "Add",
            VectorOp::Subtract => "Subtract",
            VectorOp::Multiply => "Multiply",
            VectorOp::Scale => "Scale",
            VectorOp::Normalize => "Normalize",
            VectorOp::Project => "Project",
            VectorOp::Reject => "Reject",
            VectorOp::Reflect => "Reflect",
        }
    }
}

/// Vector-in, float-out operation performed by a [`NodeKind::VectorToFloat`] node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VectorToFloatOp {
    /// `dot(a, b)`.
    Dot,
    /// `length(a)`.
    Length,
    /// `distance(a, b)`.
    Distance,
}

impl VectorToFloatOp {
    /// Every operation, in menu order. See [`MathOp::ALL`].
    pub const ALL: &'static [VectorToFloatOp] = &[
        VectorToFloatOp::Dot,
        VectorToFloatOp::Length,
        VectorToFloatOp::Distance,
    ];

    /// Menu name, for an editor's dropdown.
    pub fn label(self) -> &'static str {
        match self {
            VectorToFloatOp::Dot => "Dot",
            VectorToFloatOp::Length => "Length",
            VectorToFloatOp::Distance => "Distance",
        }
    }
}

/// Which closed-form gradient a [`NodeKind::Gradient`] evaluates.
///
/// One node with a shape rather than seven nodes, for the same reason
/// [`BlendMode`] lives on `MixColor`: every form takes the same socket and
/// returns the same thing, so nothing is hidden by the choice. That is exactly
/// the test this vocabulary applies to config enums — a dropdown is wrong when
/// it changes what the sockets *mean*, and fine when it changes only the
/// arithmetic between them.
///
/// The maths is in `wgsl/gradient.wgsl`, one function per variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GradientType {
    /// A straight ramp along X.
    #[default]
    Linear,
    /// The ramp squared: slow start, accelerating.
    Quadratic,
    /// Smoothstepped across `[0,1]` — flat at both ends.
    Easing,
    /// A ramp along the X+Y diagonal.
    Diagonal,
    /// Angle about Z, wrapped into `[0,1]`.
    ///
    /// NOTE: The only form with a discontinuity — it jumps from 1 to 0 across the
    /// -X axis, and a Bump downstream will show a seam there.
    Radial,
    /// Distance falloff: 1 at the origin, 0 at unit radius.
    Spherical,
    /// The spherical falloff squared — softer, with a flatter edge.
    QuadraticSphere,
}

impl GradientType {
    /// Every form, in menu order. See [`MathOp::ALL`].
    pub const ALL: &'static [GradientType] = &[
        GradientType::Linear,
        GradientType::Quadratic,
        GradientType::Easing,
        GradientType::Diagonal,
        GradientType::Radial,
        GradientType::Spherical,
        GradientType::QuadraticSphere,
    ];

    /// Menu name, for an editor's dropdown.
    pub fn label(self) -> &'static str {
        match self {
            GradientType::Linear => "Linear",
            GradientType::Quadratic => "Quadratic",
            GradientType::Easing => "Easing",
            GradientType::Diagonal => "Diagonal",
            GradientType::Radial => "Radial",
            GradientType::Spherical => "Spherical",
            GradientType::QuadraticSphere => "Quadratic Sphere",
        }
    }

    /// The WGSL function this form calls.
    fn function(self) -> &'static str {
        match self {
            GradientType::Linear => "sg_gradient_linear",
            GradientType::Quadratic => "sg_gradient_quadratic",
            GradientType::Easing => "sg_gradient_easing",
            GradientType::Diagonal => "sg_gradient_diagonal",
            GradientType::Radial => "sg_gradient_radial",
            GradientType::Spherical => "sg_gradient_spherical",
            GradientType::QuadraticSphere => "sg_gradient_quadratic_sphere",
        }
    }
}

/// How two colors combine before the blend factor is applied.
///
/// Every mode has the same shape — `mix(a, blend(a, b), factor)` — so the factor
/// always means "how much of the effect", never "which one wins". That is
/// Blender's convention and it is what makes stacking layers predictable.
///
/// A deliberate subset. Blender has around eighteen, most of which are
/// photo-compositing modes nobody reaches for on a material graph. Mirroring
/// the full list would be combinatorial sprawl; these nine are the ones that
/// actually do work in layered grunge, and adding a tenth is one match arm.
///
/// Modes apply to RGB. Alpha is always a plain interpolation, because a
/// `Multiply` that also multiplied alpha would quietly make surfaces
/// transparent — a bug that reads as a lighting problem.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BlendMode {
    /// Plain interpolation between the two colors.
    #[default]
    Mix,
    /// Darkens. The workhorse for layering dirt and shadow.
    Multiply,
    /// Lightens. Multiply's mirror — good for dust, frost and bloom-ish grime.
    Screen,
    /// Multiply in the dark half, Screen in the light half. Adds contrast
    /// without flattening either end, which is what makes it the usual choice
    /// for overlaying a noise detail pass.
    Overlay,
    /// `a + b`, clamped downstream rather than here.
    Add,
    /// `a - b`.
    Subtract,
    /// Per-channel absolute difference. Mostly a masking tool.
    Difference,
    /// Per-channel minimum.
    Darken,
    /// Per-channel maximum.
    Lighten,
}

impl BlendMode {
    /// Every mode, in menu order. See [`MathOp::ALL`].
    pub const ALL: &'static [BlendMode] = &[
        BlendMode::Mix,
        BlendMode::Multiply,
        BlendMode::Screen,
        BlendMode::Overlay,
        BlendMode::Add,
        BlendMode::Subtract,
        BlendMode::Difference,
        BlendMode::Darken,
        BlendMode::Lighten,
    ];

    /// Menu name, for an editor's dropdown.
    pub fn label(self) -> &'static str {
        match self {
            BlendMode::Mix => "Mix",
            BlendMode::Multiply => "Multiply",
            BlendMode::Screen => "Screen",
            BlendMode::Overlay => "Overlay",
            BlendMode::Add => "Add",
            BlendMode::Subtract => "Subtract",
            BlendMode::Difference => "Difference",
            BlendMode::Darken => "Darken",
            BlendMode::Lighten => "Lighten",
        }
    }

    /// The blended RGB, before the factor is applied. `a` and `b` are `vec4`
    /// expressions; the result is a `vec3`.
    fn blended_rgb(self, a: &str, b: &str) -> String {
        match self {
            BlendMode::Mix => format!("({b}).rgb"),
            BlendMode::Multiply => format!("({a}).rgb * ({b}).rgb"),
            // 1 - (1-a)(1-b), the standard screen identity.
            BlendMode::Screen => {
                format!("1.0 - (1.0 - ({a}).rgb) * (1.0 - ({b}).rgb)")
            }
            BlendMode::Overlay => format!("sg_overlay(({a}).rgb, ({b}).rgb)"),
            BlendMode::Add => format!("({a}).rgb + ({b}).rgb"),
            BlendMode::Subtract => format!("({a}).rgb - ({b}).rgb"),
            BlendMode::Difference => format!("abs(({a}).rgb - ({b}).rgb)"),
            BlendMode::Darken => format!("min(({a}).rgb, ({b}).rgb)"),
            BlendMode::Lighten => format!("max(({a}).rgb, ({b}).rgb)"),
        }
    }

    /// Whether this mode needs the shared blend helper.
    fn needs_snippet(self) -> bool {
        matches!(self, BlendMode::Overlay)
    }
}

/// How a color is *reduced* to a single number.
///
/// Deliberately only the operations that mix channels together.
/// [`NodeKind::SeparateColor`] owns pulling one channel out, and carrying a
/// second way to do that would mean two nodes with the same answer — exactly the
/// surface area this node set exists to avoid. What is left here is what a
/// per-channel split genuinely cannot express.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ColorToFloatOp {
    /// Rec. 709 luma. The right default for a mask painted as greyscale, and the
    /// only one that stays sensible if the mask is accidentally colored.
    Luminance,
    /// Unweighted mean of the three color channels.
    Average,
}

impl ColorToFloatOp {
    /// Every operation, in menu order. See [`MathOp::ALL`].
    pub const ALL: &'static [ColorToFloatOp] =
        &[ColorToFloatOp::Luminance, ColorToFloatOp::Average];

    /// Menu name, for an editor's dropdown.
    pub fn label(self) -> &'static str {
        match self {
            ColorToFloatOp::Luminance => "Luminance",
            ColorToFloatOp::Average => "Average",
        }
    }
}

/// The node vocabulary.
///
/// `Clone` rather than `Copy`, deliberately. Every variant here is small enough
/// to copy today, but the nodes that are obviously coming are not: a color ramp
/// carries a list of stops, an RGB curve carries control points, and a named
/// parameter carries a name. Each of those can be squeezed into a fixed-capacity
/// array to preserve `Copy` — and doing that three times would mean three
/// arbitrary caps and three sets of bookkeeping, to protect a trait that buys
/// nothing here. A node is cloned when a graph is built, and never in a loop
/// that matters.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NodeKind {
    /// Surface geometry in world space, read from the fragment input.
    Geometry,
    /// A value the author set.
    ///
    /// Occupies one uniform slot and emits `params.slots[n]` rather than a
    /// literal, which is why two materials differing only in a color share one
    /// compiled shader and why dragging a slider does not rebuild a pipeline.
    /// See [`crate::params`].
    Param {
        /// The constant this node holds, and the default when it is exposed.
        value: Value,
        /// If set, this parameter is part of the material's *interface*: a value
        /// something outside the graph may drive by name, with `value` as the
        /// default when nothing does.
        ///
        /// The name never reaches the generated WGSL — the shader reads a slot
        /// index either way — so naming a parameter, or renaming one, costs a
        /// buffer write at most. It exists so a mesh authored in Blender with a
        /// `wear_level` custom property can find the slot to write.
        ///
        /// `None` is an ordinary internal constant, and is the common case. The
        /// distinction is deliberate rather than "every param is drivable":
        /// a named parameter is a promise to whatever drives it, and a graph
        /// where every anonymous tweak is part of the public interface is a
        /// graph that cannot be rewired without breaking something.
        name: Option<String>,
    },
    /// An image sampled at a coordinate the graph supplies.
    ///
    /// Carries an image *reference* — a path — rather than image data. The
    /// compiler assigns each reachable `TexImage` one of the
    /// [`crate::params::TEXTURE_SLOTS`] texture bindings and emits a
    /// `textureSample` against it; what the path means (a file, an asset id)
    /// is the host application's business, and the layout hands it back
    /// alongside the slot index so the host can load and bind the right
    /// image. Editing the path is therefore a rebind, never a recompile —
    /// the same property every `Param` edit has.
    ///
    /// The `Vector` input is required rather than defaulted, per the module
    /// docs' "nothing is implicit" rule: Blender's generated-coordinate
    /// fallback is editor sugar, not compiler semantics. Only `.xy` of the
    /// vector is sampled.
    ///
    /// No sRGB handling in the shader, deliberately: sampling an sRGB-format
    /// texture already returns linear values, which is what PBR lighting
    /// consumes. color space is the *image's* property, declared where the
    /// image is loaded.
    TexImage {
        /// Where the image comes from. Opaque to the compiler — carried
        /// through to [`crate::params::TextureSlot`] for the host to resolve.
        /// Empty is valid while a graph is being authored; what an unresolved
        /// slot samples (a fallback white, say) is the host's call.
        path: String,
        /// If set, this texture is part of the material's *interface*: a slot
        /// something outside the graph may rebind by name, exactly as a named
        /// `Param` is a drivable value — and like a param's name, it never
        /// reaches the generated WGSL. A separate namespace from param names,
        /// because the two are driven through different calls.
        name: Option<String>,
    },
    /// Position, orient and scale a vector — usually a coordinate about to be
    /// fed to a texture node.
    Mapping,
    /// Perturb a normal by the gradient of a height field.
    ///
    /// The node that turns everything else here from paint into relief: noise
    /// can already drive color and roughness, but until a normal moves, a
    /// surface does not catch light differently.
    Bump,
    /// 3D Voronoi returning the two nearest feature points.
    ///
    /// Fixed to 3D / Euclidean / F1+F2 for now. Blender's equivalent spans
    /// 4 dimensions × 4 distance metrics × 5 features = 80 variants, and
    /// generating all of them before one is needed is how this kind of project
    /// drowns. The config enum goes on this variant when a second variant is
    /// actually called for.
    Voronoi,
    /// Layered gradient noise — fBm. See [`NoiseOctaves`] for the two ways it
    /// sums its octaves and why both exist.
    Noise(NoiseOctaves),
    /// Scalar arithmetic. See [`MathOp`].
    Math(MathOp),
    /// Vector-in, vector-out arithmetic. See [`VectorOp`].
    VectorMath(VectorOp),
    /// Vector-in, float-out arithmetic. See [`VectorToFloatOp`].
    VectorToFloat(VectorToFloatOp),
    /// Interpolate between two numbers.
    MixFloat,
    /// Interpolate between two vectors.
    MixVector,
    /// Combine two colors, with a choice of how. See [`BlendMode`].
    MixColor(BlendMode),
    /// A color reduced to one number by mixing its channels.
    ColorToFloat(ColorToFloatOp),
    /// A vector split into its three components.
    SeparateXyz,
    /// Three numbers assembled into a vector.
    CombineXyz,
    /// A color split into its four channels.
    SeparateColor,
    /// Four numbers assembled into a color.
    CombineColor,
    /// A vector reinterpreted as a color, with an alpha supplied separately.
    VectorToColor,
    /// A color reinterpreted as a vector, dropping alpha.
    ColorToVector,
    /// View-angle falloff — bright at grazing incidence, dark head-on.
    Fresnel,
    /// colors placed along a 0..1 axis, with a choice of curve between them.
    ///
    /// The first node whose emitted WGSL depends on its own *data* rather than
    /// only on its kind — a three-stop ramp is a different function from a
    /// two-stop one. That is what [`Emission::Function`] exists for, and why
    /// this node is the one that forced it.
    ColorRamp(ColorRamp),
    /// A closed-form gradient of position — a ramp, a sphere, an angle.
    ///
    /// The counterpart to noise, and the other half of most procedural
    /// materials: noise supplies the irregularity, a gradient supplies the
    /// structure. Wired into a `ColorRamp`'s Factor it is how you get a painted
    /// stripe, a scorch fading up a hull, or a glow falling off from a point —
    /// none of which noise can express, because they are not random.
    ///
    /// Costs nothing beyond a few arithmetic ops: unlike Voronoi or Perlin there
    /// is no hashing, no lattice and no sampling. It is the cheapest source of
    /// variation in the node set.
    Gradient(GradientType),
    /// A wire elbow: one input, one output, same type, same value.
    ///
    /// Purely organisational. It computes nothing and exists so a long wire can
    /// be routed around a graph instead of across it — Blender's Reroute, and
    /// the reason large node trees stay readable.
    ///
    /// The only **polymorphic** node here, which is why it carries its type
    /// rather than having four variants. That is not a hole in this crate's
    /// "one node per data type" rule: that rule exists because a `Mix` with a type
    /// dropdown would hide three quarters of its sockets, and the sockets would
    /// mean different things per type. A reroute has one socket that means
    /// exactly the same thing for every type — "the value, unchanged" — so there
    /// is nothing to hide and nothing to disambiguate. `Param` already sets the
    /// precedent by choosing its output table from the value it holds.
    Reroute(ValueType),
    /// Gathers the ten surface properties into one value.
    ///
    /// Blender's Principled BSDF, in the role it actually plays: everything a
    /// physically-based renderer needs about a point on a surface, bundled so it
    /// can be handled as a unit. Unlike Blender's, this bundle is *evaluated* —
    /// ten numbers, not a closure — which is exactly why [`NodeKind::MixSurface`]
    /// can exist.
    Surface,
    /// Blends two surfaces, property by property.
    ///
    /// The node this whole type exists for. Worn paint over bare metal is two
    /// complete materials and one mask, rather than a MixColor for the albedo, a
    /// MixFloat for the metallic, another for the roughness, and a standing
    /// invitation for them to disagree when the mask changes.
    MixSurface,
    /// The graph root: the surface handed to the renderer.
    PbrOutput,
}

// ---------------------------------------------------------------------------
// Socket tables
// ---------------------------------------------------------------------------

const fn f(name: &'static str, ident: &'static str, default: f32) -> SocketDef {
    let default = SocketDefault::Value(Value::Float(default));
    SocketDef {
        name,
        ident,
        ty: ValueType::Float,
        default,
    }
}

const fn v_linked(name: &'static str, ident: &'static str) -> SocketDef {
    SocketDef {
        name,
        ident,
        ty: ValueType::Vec3,
        default: SocketDefault::Required,
    }
}

/// A vector input that falls back to a fragment-context expression.
const fn v_ctx(name: &'static str, ident: &'static str, expr: &'static str) -> SocketDef {
    SocketDef {
        name,
        ident,
        ty: ValueType::Vec3,
        default: SocketDefault::Context(expr),
    }
}

const fn v(name: &'static str, ident: &'static str, default: [f32; 3]) -> SocketDef {
    let default = SocketDefault::Value(Value::Vec3(default));
    SocketDef {
        name,
        ident,
        ty: ValueType::Vec3,
        default,
    }
}

const fn c(name: &'static str, ident: &'static str, default: [f32; 4]) -> SocketDef {
    let default = SocketDefault::Value(Value::Color(default));
    SocketDef {
        name,
        ident,
        ty: ValueType::Color,
        default,
    }
}

const fn out_f(name: &'static str, ident: &'static str) -> SocketDef {
    SocketDef {
        name,
        ident,
        ty: ValueType::Float,
        default: SocketDefault::Required,
    }
}

const fn out_v(name: &'static str, ident: &'static str) -> SocketDef {
    SocketDef {
        name,
        ident,
        ty: ValueType::Vec3,
        default: SocketDefault::Required,
    }
}

const fn out_c(name: &'static str, ident: &'static str) -> SocketDef {
    SocketDef {
        name,
        ident,
        ty: ValueType::Color,
        default: SocketDefault::Required,
    }
}

/// A surface socket, input or output.
///
/// Only one constructor, because there is no defaulted variant and there never
/// will be: a surface has no literal, so an unwired one has nothing to evaluate
/// to. That is why `PbrOutput` with nothing plugged in is a validation error
/// naming the socket rather than a default grey — the alternative would be a
/// second place where the ten property defaults are written down, free to drift
/// from the `Surface` node's own.
const fn s(name: &'static str, ident: &'static str) -> SocketDef {
    SocketDef {
        name,
        ident,
        ty: ValueType::Surface,
        default: SocketDefault::Required,
    }
}

const GEOMETRY_OUT: &[SocketDef] = &[
    out_v("Position", "position"),
    out_v("Normal", "normal"),
    // Blender's name for the direction from the surface toward the viewer.
    out_v("Incoming", "incoming"),
    // A Vector rather than a two-component type, matching Blender's UV Map
    // output. The third component is always zero. That costs nothing and buys
    // everything: every VectorMath node and Voronoi accept a UV directly,
    // with no new socket type and no conversion node to remember.
    out_v("UV", "uv"),
    out_c("Vertex Color", "vertex_color"),
];

/// Index of `Geometry`'s UV output.
///
/// Named because the emitter has to ask "does anything read this socket?" to
/// decide whether to bind it, and a bare `3` in that question is a silent
/// mis-binding the first time a socket is inserted above it.
pub const GEOMETRY_UV_OUTPUT: usize = 3;
/// Index of `Geometry`'s vertex color output. See [`GEOMETRY_UV_OUTPUT`].
pub const GEOMETRY_VERTEX_COLOR_OUTPUT: usize = 4;

const VORONOI_IN: &[SocketDef] = &[
    v_linked("Vector", "vector"),
    f("Scale", "scale", 5.0),
    f("Randomness", "randomness", 1.0),
];

/// Blender's Gradient Texture sockets, minus the color output.
///
/// Blender emits both a `Color` and a `Fac`, where the color is only the factor
/// broadcast to RGB. That is redundant here: the useful destination is a Color
/// Ramp's Factor, which wants the scalar, and anyone needing a grey needs one
/// `CombineColor`. Emitting both would be a second way to say one thing.
///
/// `Vector` falls back to world position rather than being required, so a bare
/// Gradient dropped on the canvas already does something — which is how you find
/// out what the seven forms look like.
const GRADIENT_IN: &[SocketDef] = &[v_ctx("Vector", "vector", "sg_position")];

const VORONOI_OUT: &[SocketDef] = &[
    out_f("F1 Distance", "f1_distance"),
    out_v("F1 Position", "f1_position"),
    out_f("F2 Distance", "f2_distance"),
    out_v("F2 Position", "f2_position"),
];

/// Blender's Mapping node, Point semantics: scale, then rotate, then translate.
///
/// Only that one mode. Blender also has Texture (the inverse transform, for
/// texture coordinates), Vector (rotation and scale, no translation) and Normal
/// (rotation with inverse scale, renormalised). Vector mode is just this one
/// with Location left at zero — which is the default — and the other two are a
/// config enum for when something actually needs them, the same call as Voronoi
/// being fixed to 3D/Euclidean.
///
/// NOTE: Rotation is in **radians**. Blender's UI shows degrees and stores radians;
/// this stores what the shader uses. An editor showing degrees is welcome to
/// convert at the widget, which is the only place the distinction should exist.
const MAPPING_IN: &[SocketDef] = &[
    v_linked("Vector", "vector"),
    v("Location", "location", [0.0, 0.0, 0.0]),
    v("Rotation", "rotation", [0.0, 0.0, 0.0]),
    v("Scale", "scale", [1.0, 1.0, 1.0]),
];

/// Blender's Bump socket set, minus the Invert checkbox — a negative `Distance`
/// does the same thing, and puts the sign somewhere you can see it.
///
/// `Normal` falls back to the interpolated surface normal, so a Bump dropped in
/// and wired to a height field alone already does the useful thing.
const BUMP_IN: &[SocketDef] = &[
    f("Height", "height", 0.0),
    f("Strength", "strength", 1.0),
    f("Distance", "distance", 0.1),
    v_ctx("Normal", "normal", "sg_normal"),
];

const BUMP_OUT: &[SocketDef] = &[out_v("Normal", "normal")];

/// Blender's Noise Texture socket set, minus the parts that need a second
/// output or a distortion pass. `Detail` is a float rather than an integer for
/// the same reason Blender's is: an octave that pops into existence as you drag
/// is far more distracting than one that fades.
const NOISE_IN: &[SocketDef] = &[
    v_linked("Vector", "vector"),
    f("Scale", "scale", 5.0),
    f("Detail", "detail", 2.0),
    f("Roughness", "roughness", 0.5),
    f("Lacunarity", "lacunarity", 2.0),
];

const NOISE_OUT: &[SocketDef] = &[out_f("Value", "value")];

const MATH_IN: &[SocketDef] = &[
    f("Value", "a", 0.5),
    f("Value_001", "b", 0.5),
    f("Value_002", "c", 0.5),
];

const MATH_OUT: &[SocketDef] = &[out_f("Value", "value")];

const VECTOR_IN: &[SocketDef] = &[
    v("Vector", "a", [0.0, 0.0, 0.0]),
    v("Vector_001", "b", [0.0, 0.0, 0.0]),
    f("Scale", "scale", 1.0),
];

const VECTOR_OUT: &[SocketDef] = &[out_v("Vector", "vector")];
const VECTOR_TO_FLOAT_OUT: &[SocketDef] = &[out_f("Value", "value")];

const MIX_COLOR_IN: &[SocketDef] = &[
    f("Factor", "factor", 0.5),
    c("A", "a", [0.0, 0.0, 0.0, 1.0]),
    c("B", "b", [1.0, 1.0, 1.0, 1.0]),
];

/// One mix node per data type, by deliberate decision — rather than Blender's
/// single `Mix` with a `data_type` dropdown hiding three quarters of its sockets.
const MIX_FLOAT_IN: &[SocketDef] = &[
    f("Factor", "factor", 0.5),
    f("A", "a", 0.0),
    f("B", "b", 1.0),
];

const MIX_VECTOR_IN: &[SocketDef] = &[
    f("Factor", "factor", 0.5),
    v("A", "a", [0.0, 0.0, 0.0]),
    v("B", "b", [1.0, 1.0, 1.0]),
];

const MIX_COLOR_OUT: &[SocketDef] = &[out_c("Result", "result")];

/// White rather than black, matching the vertex-color fallback: an unwired mask
/// should read as "fully on", which is what a missing mask means everywhere else
/// in this pipeline.
const COLOR_TO_FLOAT_IN: &[SocketDef] = &[c("Color", "color", [1.0, 1.0, 1.0, 1.0])];
const COLOR_TO_FLOAT_OUT: &[SocketDef] = &[out_f("Value", "value")];

// --- conversions between the three socket types ---
//
// These exist because nothing here is implicit. Blender promotes a Float link
// into a Color socket, and a Vector into a Color, silently — convenient in an
// interactive editor and a source of invisible behaviour in a compiler. Making
// the conversion a node means the graph says what it does.
//
// NOTE: The `Separate*` nodes are why output socket idents are constrained to be
// WGSL-safe. They emit as [`Emission::Struct`], which reads each output as
// `binding.{ident}` — and with idents `x`/`y`/`z` and `r`/`g`/`b`/`a` that
// expression *is* a WGSL swizzle. The whole node is a socket table; there is no
// code generator behind it.

const SEPARATE_XYZ_IN: &[SocketDef] = &[v("Vector", "vector", [0.0, 0.0, 0.0])];
const SEPARATE_XYZ_OUT: &[SocketDef] = &[out_f("X", "x"), out_f("Y", "y"), out_f("Z", "z")];

const COMBINE_XYZ_IN: &[SocketDef] = &[f("X", "x", 0.0), f("Y", "y", 0.0), f("Z", "z", 0.0)];

const SEPARATE_COLOR_IN: &[SocketDef] = &[c("Color", "color", [0.0, 0.0, 0.0, 1.0])];
const SEPARATE_COLOR_OUT: &[SocketDef] = &[
    out_f("Red", "r"),
    out_f("Green", "g"),
    out_f("Blue", "b"),
    out_f("Alpha", "a"),
];

/// Alpha defaults to 1.0 rather than 0.0 — a color assembled from three
/// channels and left alone should be opaque, not invisible.
const COMBINE_COLOR_IN: &[SocketDef] = &[
    f("Red", "red", 0.0),
    f("Green", "green", 0.0),
    f("Blue", "blue", 0.0),
    f("Alpha", "alpha", 1.0),
];

const COLOR_OUT: &[SocketDef] = &[out_c("Color", "color")];

const VECTOR_TO_COLOR_IN: &[SocketDef] = &[
    v("Vector", "vector", [0.0, 0.0, 0.0]),
    f("Alpha", "alpha", 1.0),
];

const COLOR_TO_VECTOR_IN: &[SocketDef] = &[c("Color", "color", [0.0, 0.0, 0.0, 1.0])];

const FRESNEL_IN: &[SocketDef] = &[
    f("IOR", "ior", 1.45),
    v_linked("Normal", "normal"),
    v_linked("View", "view"),
];

const FRESNEL_OUT: &[SocketDef] = &[out_f("Factor", "factor")];

/// A color ramp takes a single scalar and returns a color, exactly like
/// Blender's. The default matches the ramp's own clamping: below the first stop
/// is the first stop's color, so an unwired factor reads as the ramp's left
/// end rather than as an error.
const COLOR_RAMP_IN: &[SocketDef] = &[f("Factor", "factor", 0.0)];
/// Blender's Color Ramp outputs, both of them.
///
/// `Alpha` is not an afterthought — it turns the node into an arbitrary **1D
/// curve**. The stops already carry a fourth channel that the color picker
/// edits, and it is independent of the other three, so a ramp can shape a scalar
/// response (roughness rising through a wear band, say) while its RGB does
/// something entirely unrelated, from one factor and one node.
///
/// NOTE: The idents are **vec4 swizzles**, not field names. The generated function
/// returns a `vec4`, and a node with more than one output reads each of them off
/// that value by swizzle — see `emit::output_expr`. `rgba` is the whole vector;
/// `a` is the fourth channel.
const COLOR_RAMP_OUT: &[SocketDef] = &[out_c("Color", "rgba"), out_f("Alpha", "a")];

/// A texture read takes a coordinate and returns the sampled texel.
///
/// `Vector` is required rather than defaulted — see [`SocketDefault::Context`]:
/// implicit texture coordinates are exactly the Blender convenience this crate
/// declines to copy. A vector rather than a 2D type for the same reason
/// `Geometry`'s UV output is one: every node that produces or transforms a
/// coordinate already speaks Vec3, and only `.xy` reaches the sampler.
const TEX_IMAGE_IN: &[SocketDef] = &[v_linked("Vector", "vector")];

/// The idents are **vec4 swizzles**, the same contract as [`COLOR_RAMP_OUT`]:
/// the binding is the sampled `vec4`, `rgba` reads all of it and `a` the fourth
/// channel — which makes an image's alpha a first-class mask output rather than
/// a `SeparateColor` away.
const TEX_IMAGE_OUT: &[SocketDef] = &[out_c("Color", "rgba"), out_f("Alpha", "a")];

/// The renderer-facing surface properties.
///
/// `Anisotropy Strength` and `Anisotropy Tangent` are the reason a brushed metal
/// can terminate here at all. The tangent is a world-space direction evaluated
/// per fragment, so a varying brush direction needs no UV channel and no mesh
/// tangents — which matters, because this pipeline's premise is that meshes are
/// never UV unwrapped.
///
/// NOTE: This table is the single source of truth for the `SgSurface` struct the
/// emitter declares: field order, WGSL type and field name all come from here, in
/// this order. `emit::surface_struct` generates the declaration from it, so
/// adding a property is an edit to this list alone. Nothing may be inserted in
/// the middle without understanding that `MixSurface` lerps every field — a new
/// property that must not be interpolated does not belong in this struct.
pub(crate) const SURFACE_IN: &[SocketDef] = &[
    c("Base Color", "base_color", [0.8, 0.8, 0.8, 1.0]),
    f("Metallic", "metallic", 0.0),
    f("Perceptual Roughness", "perceptual_roughness", 0.5),
    f("Reflectance", "reflectance", 0.5),
    c("Emissive", "emissive", [0.0, 0.0, 0.0, 1.0]),
    f("Occlusion", "occlusion", 1.0),
    // Unwired means "use the surface as it is", i.e. the interpolated geometric
    // normal that the emitter binds as `sg_normal`.
    v_ctx("Normal", "normal", "sg_normal"),
    f("Anisotropy Strength", "anisotropy_strength", 0.0),
    // Zero is inert: the renderer only consults the tangent when strength is
    // non-zero, and a zero vector matches what `pbr_input_new` already sets.
    v_ctx("Anisotropy Tangent", "anisotropy_tangent", "vec3<f32>(0.0)"),
    f("Alpha", "alpha", 1.0),
];

const SURFACE_OUT: &[SocketDef] = &[s("Surface", "surface")];

// Reroute sockets, one table per type — and each serves as **both** the input
// and the output list, because a reroute's two ends are the same socket by
// definition. Four tables rather than one generated on demand because
// [`NodeKind::inputs`] hands back `&'static [SocketDef]`; `Param` picks from
// existing tables the same way for the same reason.
//
// Every one is `Required`. A reroute is placed *while* a graph is being
// rearranged, so an unwired one is a common transient state — but giving it a
// default would mean a reroute you forgot to connect silently emits zero, which
// is a black hole in the middle of a graph and hard to see. An error naming the
// node is the better failure, and the editor keeps the last good shader on the
// mesh while you finish wiring.
const REROUTE_FLOAT: &[SocketDef] = &[out_f("Value", "value")];
const REROUTE_VEC3: &[SocketDef] = &[out_v("Vector", "vector")];
const REROUTE_COLOR: &[SocketDef] = &[out_c("Color", "color")];
const REROUTE_SURFACE: &[SocketDef] = &[s("Surface", "surface")];

/// The socket a reroute of this type carries, as both its input and its output.
const fn reroute_sockets(ty: ValueType) -> &'static [SocketDef] {
    match ty {
        ValueType::Float => REROUTE_FLOAT,
        ValueType::Vec3 => REROUTE_VEC3,
        ValueType::Color => REROUTE_COLOR,
        ValueType::Surface => REROUTE_SURFACE,
    }
}

/// Blender's Mix Shader, with sockets that mean what they say.
///
/// `A` and `B` rather than Blender's two identically-named `Shader` sockets:
/// theirs are told apart only by index, which is unusable in a text-addressed
/// API like [`crate::Graph::link`]. Factor 0 is all `A`, factor 1 is all `B`,
/// matching every other mix node here.
const MIX_SURFACE_IN: &[SocketDef] = &[f("Factor", "factor", 0.0), s("A", "a"), s("B", "b")];

/// The graph's terminus: one surface, handed to the renderer.
///
/// A single socket, because everything that used to be here now lives on
/// `Surface`. That split is the whole of `layered-surface`: the properties
/// became a value that can be produced, mixed and reused, and this node went
/// back to meaning only "and this is the one that gets drawn".
const PBR_OUTPUT_IN: &[SocketDef] = &[s("Surface", "surface")];

impl NodeKind {
    /// One entry per thing an author can place, in menu order.
    ///
    /// Not simply "one per variant": `Param` appears three times because its
    /// output socket type follows the value it holds, so "add a Param" is not a
    /// single choice. The op-carrying variants appear once each with a starting
    /// operation, since the op is changed in place afterwards rather than by
    /// picking a different node.
    ///
    /// `PbrOutput` is deliberately absent. A graph must contain exactly one, so
    /// an editor seeds it rather than offering it — and offering it would make
    /// [`crate::GraphError::MultipleOutputNodes`] a thing you could do by
    /// accident from a menu.
    ///
    /// A function rather than a `const` slice because [`NodeKind`] is no longer
    /// `Copy` and its coming variants own heap data, which a `const` cannot
    /// hold. Called once when a menu opens.
    pub fn palette() -> Vec<NodeKind> {
        vec![
            NodeKind::Geometry,
            NodeKind::param(Value::Float(0.0)),
            NodeKind::param(Value::Vec3([0.0, 0.0, 0.0])),
            NodeKind::param(Value::Color([0.8, 0.8, 0.8, 1.0])),
            NodeKind::Mapping,
            NodeKind::Bump,
            NodeKind::Voronoi,
            NodeKind::Noise(NoiseOctaves::Live),
            // Placed with an empty path, like a Param is placed with a zero
            // value: the reference is data edited on the node afterwards.
            NodeKind::tex_image(""),
            NodeKind::Math(MathOp::Add),
            NodeKind::VectorMath(VectorOp::Add),
            NodeKind::VectorToFloat(VectorToFloatOp::Dot),
            NodeKind::MixFloat,
            NodeKind::MixVector,
            NodeKind::MixColor(BlendMode::Mix),
            NodeKind::ColorToFloat(ColorToFloatOp::Luminance),
            NodeKind::SeparateXyz,
            NodeKind::CombineXyz,
            NodeKind::SeparateColor,
            NodeKind::CombineColor,
            NodeKind::VectorToColor,
            NodeKind::ColorToVector,
            NodeKind::Fresnel,
            NodeKind::ColorRamp(ColorRamp::default()),
            // One entry with a starting form, changed in place afterwards — the
            // same treatment `Math` and `MixColor` get.
            NodeKind::Gradient(GradientType::Linear),
            NodeKind::Surface,
            NodeKind::MixSurface,
            // One entry, not four, even though the node has four typed forms.
            // Unlike `Param` — whose three entries are three different things to
            // place — a reroute adopts the type of the first wire it is given,
            // so choosing up front would be asking a question the next click
            // answers. Float is simply where it waits.
            NodeKind::Reroute(ValueType::Float),
        ]
    }

    /// Menu and title text, distinguishing what [`NodeKind::name`] cannot.
    ///
    /// `name()` returns the *kind* ("Math"), which is what an error message
    /// wants. A menu needs to tell `Param(Float)` from `Param(Color)`, since
    /// those are separate entries in [`NodeKind::palette`].
    pub fn label(&self) -> String {
        match self {
            // A named param shows its name, because that is the thing an author
            // is actually looking for in a graph full of otherwise identical
            // "Param (Float)" boxes.
            NodeKind::Param {
                name: Some(name), ..
            } => name.clone(),
            NodeKind::Param { value, name: None } => format!("Param ({})", value.ty()),
            // A named texture shows its name, like a named param. An unnamed
            // one shows the file it samples — that is what tells two texture
            // nodes apart on a canvas — and only an empty path falls back to
            // the kind.
            NodeKind::TexImage {
                name: Some(name), ..
            } => name.clone(),
            NodeKind::TexImage { path, name: None } => match path.rsplit(['/', '\\']).next() {
                Some(file) if !file.is_empty() => file.to_string(),
                _ => "Tex Image".to_string(),
            },
            NodeKind::ColorRamp(_) => "Color Ramp".to_string(),
            NodeKind::Gradient(kind) => format!("Gradient ({})", kind.label()),
            NodeKind::Noise(octaves) => format!("Noise ({})", octaves.label()),
            NodeKind::MixColor(mode) => format!("Mix Color ({})", mode.label()),
            other => other.name().to_string(),
        }
    }

    /// An unnamed constant — the common case.
    ///
    /// A shorthand rather than a wrapper: `NodeKind::Param { value, name: None }`
    /// is what almost every call site wants, and spelling the `None` out at each
    /// of them buries the interesting cases in noise.
    pub fn param(value: Value) -> Self {
        NodeKind::Param { value, name: None }
    }

    /// An unnamed texture read. The [`NodeKind::param`] shorthand's twin.
    pub fn tex_image(path: impl Into<String>) -> Self {
        NodeKind::TexImage {
            path: path.into(),
            name: None,
        }
    }

    /// The name this parameter is exposed under, if it is exposed at all.
    pub fn param_name(&self) -> Option<&str> {
        match self {
            NodeKind::Param { name, .. } => name.as_deref(),
            _ => None,
        }
    }

    /// The name this texture is exposed under, if it is exposed at all.
    ///
    /// A separate accessor from [`NodeKind::param_name`] because the two are
    /// separate namespaces: a param and a texture may share a name, since one
    /// is driven by writing a uniform slot and the other by rebinding an image.
    pub fn texture_name(&self) -> Option<&str> {
        match self {
            NodeKind::TexImage { name, .. } => name.as_deref(),
            _ => None,
        }
    }

    /// The uniform slots this node occupies, in order.
    ///
    /// A `Param` takes one. A color ramp takes a run of them holding its
    /// polynomial coefficients. Everything else takes none — its behaviour is
    /// fully described by the code the emitter writes for it.
    ///
    /// This is the extension point for every future node that carries tunable
    /// data: return rows here and read them back by offset from the base the
    /// emitter reports, and editing that data becomes a buffer write for free.
    pub fn slots(&self) -> Vec<SlotValue> {
        match self {
            NodeKind::Param { value, .. } => vec![SlotValue::Param(*value)],
            NodeKind::ColorRamp(ramp) => ramp.slot_values(),
            _ => Vec::new(),
        }
    }

    /// The WGSL function this node contributes, if any.
    ///
    /// Only [`Emission::Function`] nodes return one. `name` is the function's
    /// identifier, chosen by the emitter so it is unique within the shader, and
    /// `base_slot` is where this node's [`NodeKind::slots`] were written — the
    /// generated body addresses its data relative to it.
    pub fn function(&self, name: &str, base_slot: usize) -> Option<String> {
        match self {
            NodeKind::ColorRamp(ramp) => Some(ramp.emit_wgsl(name, base_slot)),
            NodeKind::Noise(octaves @ NoiseOctaves::Unrolled(_)) => {
                Some(unrolled_fbm(name, octaves.unrolled_count()))
            }
            _ => None,
        }
    }

    /// Display name, used in error messages.
    pub fn name(&self) -> &'static str {
        match self {
            NodeKind::Geometry => "Geometry",
            NodeKind::Param { .. } => "Param",
            NodeKind::TexImage { .. } => "TexImage",
            NodeKind::Mapping => "Mapping",
            NodeKind::Bump => "Bump",
            NodeKind::Voronoi => "Voronoi",
            NodeKind::Noise(_) => "Noise",
            NodeKind::Math(_) => "Math",
            NodeKind::VectorMath(_) => "VectorMath",
            NodeKind::VectorToFloat(_) => "VectorToFloat",
            NodeKind::MixFloat => "MixFloat",
            NodeKind::MixVector => "MixVector",
            NodeKind::MixColor(_) => "MixColor",
            NodeKind::ColorToFloat(_) => "ColorToFloat",
            NodeKind::SeparateXyz => "SeparateXyz",
            NodeKind::CombineXyz => "CombineXyz",
            NodeKind::SeparateColor => "SeparateColor",
            NodeKind::CombineColor => "CombineColor",
            NodeKind::VectorToColor => "VectorToColor",
            NodeKind::ColorToVector => "ColorToVector",
            NodeKind::Fresnel => "Fresnel",
            NodeKind::ColorRamp(_) => "ColorRamp",
            NodeKind::Gradient(_) => "Gradient",
            NodeKind::Reroute(_) => "Reroute",
            NodeKind::Surface => "Surface",
            NodeKind::MixSurface => "MixSurface",
            NodeKind::PbrOutput => "PbrOutput",
        }
    }

    /// Short lowercase tag used to build readable SSA binding names.
    pub fn tag(&self) -> &'static str {
        match self {
            NodeKind::Geometry => "geo",
            NodeKind::Param { .. } => "param",
            NodeKind::TexImage { .. } => "tex",
            NodeKind::Mapping => "map",
            NodeKind::Bump => "bump",
            NodeKind::Voronoi => "voronoi",
            NodeKind::Noise(_) => "noise",
            NodeKind::Math(_) => "math",
            NodeKind::VectorMath(_) => "vec",
            NodeKind::VectorToFloat(_) => "vecf",
            NodeKind::MixFloat => "mixf",
            NodeKind::MixVector => "mixv",
            NodeKind::MixColor(_) => "mix",
            NodeKind::ColorToFloat(_) => "colf",
            NodeKind::SeparateXyz => "sepxyz",
            NodeKind::CombineXyz => "combxyz",
            NodeKind::SeparateColor => "sepcol",
            NodeKind::CombineColor => "combcol",
            NodeKind::VectorToColor => "vectocol",
            NodeKind::ColorToVector => "coltovec",
            NodeKind::Fresnel => "fresnel",
            NodeKind::ColorRamp(_) => "ramp",
            NodeKind::Gradient(_) => "grad",
            NodeKind::Reroute(_) => "via",
            NodeKind::Surface => "surface",
            NodeKind::MixSurface => "mixsurf",
            NodeKind::PbrOutput => "out",
        }
    }

    /// This kind's input sockets, in wire-index order.
    pub fn inputs(&self) -> &'static [SocketDef] {
        match self {
            NodeKind::Geometry | NodeKind::Param { .. } => &[],
            NodeKind::TexImage { .. } => TEX_IMAGE_IN,
            NodeKind::Mapping => MAPPING_IN,
            NodeKind::Bump => BUMP_IN,
            NodeKind::Voronoi => VORONOI_IN,
            NodeKind::Noise(_) => NOISE_IN,
            NodeKind::Math(_) => MATH_IN,
            NodeKind::VectorMath(_) | NodeKind::VectorToFloat(_) => VECTOR_IN,
            NodeKind::MixFloat => MIX_FLOAT_IN,
            NodeKind::MixVector => MIX_VECTOR_IN,
            NodeKind::MixColor(_) => MIX_COLOR_IN,
            NodeKind::ColorToFloat(_) => COLOR_TO_FLOAT_IN,
            NodeKind::SeparateXyz => SEPARATE_XYZ_IN,
            NodeKind::CombineXyz => COMBINE_XYZ_IN,
            NodeKind::SeparateColor => SEPARATE_COLOR_IN,
            NodeKind::CombineColor => COMBINE_COLOR_IN,
            NodeKind::VectorToColor => VECTOR_TO_COLOR_IN,
            NodeKind::ColorToVector => COLOR_TO_VECTOR_IN,
            NodeKind::Fresnel => FRESNEL_IN,
            NodeKind::ColorRamp(_) => COLOR_RAMP_IN,
            NodeKind::Gradient(_) => GRADIENT_IN,
            NodeKind::Reroute(ty) => reroute_sockets(*ty),
            NodeKind::Surface => SURFACE_IN,
            NodeKind::MixSurface => MIX_SURFACE_IN,
            NodeKind::PbrOutput => PBR_OUTPUT_IN,
        }
    }

    /// This kind's output sockets, in wire-index order.
    pub fn outputs(&self) -> &'static [SocketDef] {
        match self {
            NodeKind::Geometry => GEOMETRY_OUT,
            NodeKind::Param { value, .. } => match value.ty() {
                ValueType::Float => MATH_OUT,
                ValueType::Vec3 => VECTOR_OUT,
                ValueType::Color => MIX_COLOR_OUT,
                // `Value` has no Surface variant, so `value.ty()` cannot return
                // this. Spelled out rather than left to a catch-all so that
                // adding one later is a compile error here — which is where the
                // question "what would a surface literal even be?" needs asking.
                ValueType::Surface => unreachable!("a Param cannot hold a surface"),
            },
            NodeKind::TexImage { .. } => TEX_IMAGE_OUT,
            NodeKind::Bump => BUMP_OUT,
            NodeKind::Voronoi => VORONOI_OUT,
            NodeKind::Noise(_) => NOISE_OUT,
            NodeKind::Math(_) => MATH_OUT,
            NodeKind::VectorMath(_) => VECTOR_OUT,
            NodeKind::VectorToFloat(_) => VECTOR_TO_FLOAT_OUT,
            NodeKind::MixFloat => MATH_OUT,
            NodeKind::MixColor(_) => MIX_COLOR_OUT,
            NodeKind::ColorToFloat(_) => COLOR_TO_FLOAT_OUT,
            NodeKind::SeparateXyz => SEPARATE_XYZ_OUT,
            NodeKind::CombineXyz
            | NodeKind::ColorToVector
            | NodeKind::MixVector
            | NodeKind::Mapping => VECTOR_OUT,
            NodeKind::SeparateColor => SEPARATE_COLOR_OUT,
            NodeKind::CombineColor | NodeKind::VectorToColor => COLOR_OUT,
            NodeKind::Fresnel => FRESNEL_OUT,
            NodeKind::ColorRamp(_) => COLOR_RAMP_OUT,
            // Reuses Math's single float output: the socket is called "Value"
            // and that is exactly what a gradient produces.
            NodeKind::Gradient(_) => MATH_OUT,
            // The same table as its input, deliberately — see `reroute_sockets`.
            NodeKind::Reroute(ty) => reroute_sockets(*ty),
            NodeKind::Surface | NodeKind::MixSurface => SURFACE_OUT,
            NodeKind::PbrOutput => &[],
        }
    }

    /// How this kind turns into WGSL. See [`Emission`].
    pub fn emission(&self) -> Emission {
        match self {
            // These read the fragment context directly, from bindings the
            // generated fragment function establishes before the graph body; see
            // `emit.rs`. The last two are conditional on the mesh carrying the
            // attribute, and are only bound when something reads them.
            NodeKind::Geometry => Emission::Inline(&[
                "sg_position",
                "sg_normal",
                "sg_view",
                "sg_uv",
                "sg_vertex_color",
            ]),
            NodeKind::Param { .. } => Emission::Param,
            NodeKind::TexImage { .. } => Emission::Texture,
            NodeKind::Voronoi => Emission::Struct,
            // The binding is the input value itself; each output reads a
            // component off it as a swizzle. See the socket tables above.
            NodeKind::SeparateXyz | NodeKind::SeparateColor => Emission::Struct,
            NodeKind::ColorRamp(_) => Emission::Function,
            // Live calls a shared snippet; unrolled generates its own function,
            // because how many octaves it sums IS its code.
            NodeKind::Noise(NoiseOctaves::Live) => Emission::Expr,
            NodeKind::Noise(NoiseOctaves::Unrolled(_)) => Emission::Function,
            NodeKind::PbrOutput => Emission::Root,
            _ => Emission::Expr,
        }
    }

    /// WGSL support functions this node needs.
    pub fn snippets(&self) -> &'static [Snippet] {
        match self {
            NodeKind::Voronoi => &[Snippet::Hash, Snippet::Voronoi],
            // The unrolled variant deliberately does NOT pull in `Fbm` — it
            // replaces that loop, and emitting it anyway would put a dynamic
            // loop in a shader that exists to avoid one.
            NodeKind::Noise(NoiseOctaves::Live) => &[Snippet::Hash, Snippet::Perlin, Snippet::Fbm],
            NodeKind::Noise(NoiseOctaves::Unrolled(_)) => &[Snippet::Hash, Snippet::Perlin],
            NodeKind::MixColor(mode) if mode.needs_snippet() => &[Snippet::Blend],
            NodeKind::Mapping => &[Snippet::Mapping],
            // All seven forms come in together. Splitting `Snippet` per variant
            // would trim a few dead functions the driver already discards, in
            // exchange for seven enum members and seven files.
            NodeKind::Gradient(_) => &[Snippet::Gradient],
            NodeKind::Bump => &[Snippet::Bump],
            NodeKind::Fresnel => &[Snippet::Fresnel],
            NodeKind::VectorMath(VectorOp::Project) | NodeKind::VectorMath(VectorOp::Reject) => {
                &[Snippet::Vector]
            }
            // `Surface` needs nothing: constructing the struct is a constructor
            // call, and the struct itself is declared in the header because every
            // graph ends in one. Only the blend function is conditional.
            NodeKind::MixSurface => &[Snippet::MixSurface],
            _ => &[],
        }
    }

    /// Build this node's right-hand-side expression from its resolved inputs.
    ///
    /// `args` holds one expression per input socket, in socket order. Only
    /// called for [`Emission::Expr`] and [`Emission::Struct`] nodes.
    pub fn expr(&self, args: &[String]) -> String {
        match self {
            NodeKind::Voronoi => {
                let (vector, scale, randomness) = (&args[0], &args[1], &args[2]);
                format!("sg_voronoi_f1f2_3d({vector} * {scale}, {randomness})")
            }
            // `sg_position` rather than a socket, deliberately. The basis has to
            // be reconstructed from the *true* surface position; the derivative
            // of a mapped or rotated coordinate would describe the mapping
            // instead. Where the height came from is the graph's business, and
            // has no bearing on which plane it is projected onto.
            NodeKind::Bump => {
                let (height, strength, distance, normal) = (&args[0], &args[1], &args[2], &args[3]);
                format!("sg_bump({height}, {strength}, {distance}, {normal}, sg_position)")
            }
            // A positional struct constructor, so the argument order IS
            // `SURFACE_IN`'s order — which is also the order `surface_struct`
            // declares the fields in. One table drives both; they cannot drift.
            NodeKind::Gradient(kind) => format!("{}({})", kind.function(), args[0]),
            // The whole node. It binds its input to a fresh name and that is all
            // it does — one SSA `let` the driver folds away to nothing, which is
            // the correct cost for something that exists to move a wire on a
            // canvas. Resolving reroutes away in the emitter instead would make
            // the generated source marginally shorter and `output_expr` a
            // special case; not worth it.
            NodeKind::Reroute(_) => args[0].clone(),
            NodeKind::Surface => format!("SgSurface({})", args.join(", ")),
            NodeKind::MixSurface => {
                let (factor, a, b) = (&args[0], &args[1], &args[2]);
                // Clamped inside the function rather than here, unlike the other
                // mix nodes. They fold a literal factor at compile time to keep
                // goldens readable; there is nothing to fold into here, because
                // the work is a call either way.
                format!("sg_mix_surface({factor}, {a}, {b})")
            }
            // Scale, then rotate, then translate — Blender's Point order. The
            // order is not arbitrary: scaling after rotation would shear the
            // result on any non-uniform scale, and translating before rotation
            // would swing the offset around with the angle.
            NodeKind::Mapping => {
                let (vector, location, rotation, scale) = (&args[0], &args[1], &args[2], &args[3]);
                format!("(sg_euler_xyz({rotation}) * ({vector} * {scale}) + {location})")
            }
            NodeKind::Noise(_) => {
                let (vector, scale, detail, roughness, lacunarity) =
                    (&args[0], &args[1], &args[2], &args[3], &args[4]);
                format!("sg_fbm_3d({vector} * {scale}, {detail}, {roughness}, {lacunarity})")
            }
            NodeKind::Math(op) => {
                let (a, b, c) = (&args[0], &args[1], &args[2]);
                match op {
                    MathOp::Add => format!("({a} + {b})"),
                    MathOp::Subtract => format!("({a} - {b})"),
                    MathOp::Multiply => format!("({a} * {b})"),
                    MathOp::Divide => format!("({a} / {b})"),
                    MathOp::Power => format!("pow({a}, {b})"),
                    MathOp::Minimum => format!("min({a}, {b})"),
                    MathOp::Maximum => format!("max({a}, {b})"),
                    MathOp::SmoothStep => format!("smoothstep({a}, {b}, {c})"),
                }
            }
            NodeKind::VectorMath(op) => {
                let (a, b, scale) = (&args[0], &args[1], &args[2]);
                match op {
                    VectorOp::Add => format!("({a} + {b})"),
                    VectorOp::Subtract => format!("({a} - {b})"),
                    VectorOp::Multiply => format!("({a} * {b})"),
                    VectorOp::Scale => format!("({a} * {scale})"),
                    VectorOp::Normalize => format!("normalize({a})"),
                    VectorOp::Project => format!("sg_project({a}, {b})"),
                    VectorOp::Reject => format!("sg_reject({a}, {b})"),
                    VectorOp::Reflect => format!("reflect({a}, {b})"),
                }
            }
            NodeKind::VectorToFloat(op) => {
                let (a, b) = (&args[0], &args[1]);
                match op {
                    VectorToFloatOp::Dot => format!("dot({a}, {b})"),
                    VectorToFloatOp::Length => format!("length({a})"),
                    VectorToFloatOp::Distance => format!("distance({a}, {b})"),
                }
            }
            // WGSL's `mix` takes the blend factor last; Blender's socket order
            // puts it first. Socket order is what an author sees, so it wins.
            NodeKind::MixFloat | NodeKind::MixVector => {
                let (factor, a, b) = (&args[0], &args[1], &args[2]);
                format!("mix({a}, {b}, {})", clamped(factor))
            }
            NodeKind::MixColor(mode) => {
                let (factor, a, b) = (&args[0], &args[1], &args[2]);
                let factor = clamped(factor);
                match mode {
                    // Plain interpolation of all four channels is exactly what a
                    // `vec4` mix already is, so the general form below would only
                    // rebuild `b` component by component. Kept simple because
                    // this is by far the most common node in any graph.
                    BlendMode::Mix => format!("mix({a}, {b}, {factor})"),
                    // The blended RGB is packed against B's alpha and then
                    // interpolated as one `vec4`, so alpha lerps plainly whatever
                    // the mode does to the color — and the factor is evaluated
                    // once rather than once per channel group.
                    _ => format!(
                        "mix({a}, vec4<f32>({}, ({b}).a), {factor})",
                        mode.blended_rgb(a, b)
                    ),
                }
            }
            NodeKind::ColorToFloat(op) => {
                let color = &args[0];
                match op {
                    // Rec. 709, the same weights Bevy's tonemapping uses.
                    ColorToFloatOp::Luminance => {
                        format!("dot(({color}).rgb, vec3<f32>(0.2126, 0.7152, 0.0722))")
                    }
                    ColorToFloatOp::Average => {
                        format!("dot(({color}).rgb, vec3<f32>(0.33333334))")
                    }
                }
            }
            // A `Separate*` node's right-hand side is simply its input: the
            // binding exists so each output can swizzle off one name rather than
            // re-evaluating the upstream expression four times.
            NodeKind::SeparateXyz | NodeKind::SeparateColor => args[0].clone(),
            NodeKind::CombineXyz => {
                let (x, y, z) = (&args[0], &args[1], &args[2]);
                format!("vec3<f32>({x}, {y}, {z})")
            }
            NodeKind::CombineColor => {
                let (r, g, b, a) = (&args[0], &args[1], &args[2], &args[3]);
                format!("vec4<f32>({r}, {g}, {b}, {a})")
            }
            // WGSL builds a vec4 from a vec3 and a scalar directly, so this is a
            // construction rather than four component reads.
            NodeKind::VectorToColor => {
                let (vector, alpha) = (&args[0], &args[1]);
                format!("vec4<f32>({vector}, {alpha})")
            }
            NodeKind::ColorToVector => format!("({}).rgb", args[0]),
            NodeKind::Fresnel => {
                let (ior, normal, view) = (&args[0], &args[1], &args[2]);
                format!("sg_fresnel({ior}, {normal}, {view})")
            }
            // A ramp's code is a generated function, not an inline expression;
            // the emitter calls it by name. See [`Emission::Function`]. A
            // texture's call is written by the emitter too, because only the
            // emitter knows which slot the node was assigned.
            NodeKind::Geometry
            | NodeKind::Param { .. }
            | NodeKind::TexImage { .. }
            | NodeKind::ColorRamp(_)
            | NodeKind::PbrOutput => {
                unreachable!("{} does not emit an expression from inputs", self.name())
            }
        }
    }
}

/// Clamp a blend factor into `[0, 1]`.
///
/// WGSL's `mix` **extrapolates** outside that range rather than clamping, so a
/// factor of 1.4 does not mean "all the way to B" — it means somewhere past B,
/// which for colors is out of gamut and reads as a lighting bug rather than a
/// wiring one. Blender clamps its Mix factor by default for the same reason.
///
/// A literal is folded here rather than wrapped, so an unwired factor stays a
/// plain number in the generated source instead of `clamp(0.5, 0.0, 1.0)`.
fn clamped(factor: &str) -> String {
    if let Ok(value) = factor.parse::<f32>() {
        return crate::value::wgsl_f32(value.clamp(0.0, 1.0));
    }
    format!("clamp({factor}, 0.0, 1.0)")
}

/// Generate the straight-line equivalent of `sg_fbm_3d` for a fixed octave count.
///
/// Every line here mirrors one iteration of the loop in `wgsl/fbm.wgsl`, calling
/// the *same* `sg_perlin_3d` and `sg_octave_weight`. Keeping them term for term
/// is what lets the two variants be compared honestly: if the outputs ever
/// diverge, one of them is wrong rather than "different".
///
/// Parameters are in socket order, because the emitter builds the call from
/// [`NodeKind::inputs`] — see [`Emission::Function`].
fn unrolled_fbm(name: &str, octaves: u8) -> String {
    use std::fmt::Write as _;

    let mut s = String::new();
    let _ = writeln!(
        s,
        "fn {name}(vector: vec3<f32>, scale: f32, detail: f32, roughness: f32, \
         lacunarity: f32) -> f32 {{"
    );
    let _ = writeln!(s, "    let p = vector * scale;");
    let _ = writeln!(s, "    var sum = 0.0;");
    let _ = writeln!(s, "    var norm = 0.0;");
    let _ = writeln!(s, "    var amp = 1.0;");
    let _ = writeln!(s, "    var freq = 1.0;");

    for octave in 0..octaves {
        // No early exit, unlike the loop. That is the trade this variant makes:
        // straight-line code the compiler can see through, at the cost of
        // evaluating octaves whose weight is zero.
        let _ = writeln!(
            s,
            "    let w{octave} = amp * sg_octave_weight(detail, {octave});"
        );
        let _ = writeln!(s, "    sum = sum + w{octave} * sg_perlin_3d(p * freq);");
        let _ = writeln!(s, "    norm = norm + w{octave};");
        // Skipped on the last octave; nothing reads them afterwards, and WGSL
        // warns about an assignment whose result is never used.
        if octave + 1 < octaves {
            let _ = writeln!(s, "    amp = amp * roughness;");
            let _ = writeln!(s, "    freq = freq * lacunarity;");
        }
    }

    let _ = writeln!(s, "    if (norm <= 0.0) {{ return 0.5; }}");
    let _ = writeln!(s, "    return sum / norm * 0.5 + 0.5;");
    let _ = writeln!(s, "}}");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_socket_ident_is_unique_within_its_node() {
        // Derived from the palette rather than listed, so a node added to the
        // vocabulary is checked without anyone remembering to add it here. A
        // hand-written list is a list that silently stops covering the node set.
        let mut kinds = NodeKind::palette();
        kinds.push(NodeKind::PbrOutput);

        for kind in kinds {
            for sockets in [kind.inputs(), kind.outputs()] {
                for (i, a) in sockets.iter().enumerate() {
                    for b in &sockets[i + 1..] {
                        assert_ne!(a.ident, b.ident, "duplicate ident in {}", kind.name());
                        assert_ne!(a.name, b.name, "duplicate name in {}", kind.name());
                    }
                }
            }
        }
    }

    #[test]
    fn inline_nodes_declare_one_expression_per_output() {
        // A mismatch here would index out of bounds during emission, in a
        // generated shader, at runtime. Catch it at the table instead.
        for kind in [NodeKind::Geometry] {
            if let Emission::Inline(exprs) = kind.emission() {
                assert_eq!(exprs.len(), kind.outputs().len(), "{}", kind.name());
            } else {
                panic!("{} was expected to be Inline", kind.name());
            }
        }
    }

    /// A `Separate*` node's output idents have to be legal WGSL swizzles,
    /// because the emitter reads them as `binding.{ident}` and nothing else
    /// checks that. Get one wrong and the failure is a shader that does not
    /// compile, citing generated source nobody wrote.
    #[test]
    fn separate_nodes_name_their_outputs_as_swizzles() {
        assert_eq!(
            NodeKind::SeparateXyz
                .outputs()
                .iter()
                .map(|s| s.ident)
                .collect::<Vec<_>>(),
            vec!["x", "y", "z"]
        );
        assert_eq!(
            NodeKind::SeparateColor
                .outputs()
                .iter()
                .map(|s| s.ident)
                .collect::<Vec<_>>(),
            vec!["r", "g", "b", "a"]
        );

        // Both must be Struct, which is what makes the swizzle the read path.
        for kind in [NodeKind::SeparateXyz, NodeKind::SeparateColor] {
            assert_eq!(kind.emission(), Emission::Struct, "{}", kind.name());
        }
    }

    /// A texture's outputs are swizzles off the sampled `vec4`, and its
    /// emission is the one the emitter writes a `textureSample` for.
    ///
    /// The same contract [`COLOR_RAMP_OUT`] pins: get an ident wrong and the
    /// failure is a generated shader citing a field nobody declared.
    #[test]
    fn tex_image_outputs_read_as_swizzles() {
        let kind = NodeKind::tex_image("hull.png");
        assert_eq!(
            kind.outputs().iter().map(|s| s.ident).collect::<Vec<_>>(),
            vec!["rgba", "a"]
        );
        assert_eq!(kind.emission(), Emission::Texture);
    }

    /// An unnamed texture labels itself by file, a named one by name.
    ///
    /// The label is how two texture nodes are told apart on a canvas, so the
    /// path has to win over the generic kind whenever there is one.
    #[test]
    fn tex_image_labels_follow_name_then_file() {
        assert_eq!(NodeKind::tex_image("").label(), "Tex Image");
        assert_eq!(
            NodeKind::tex_image("decals/insignia.png").label(),
            "insignia.png"
        );
        let named = NodeKind::TexImage {
            path: "decals/insignia.png".to_string(),
            name: Some("insignia".to_string()),
        };
        assert_eq!(named.label(), "insignia");
    }

    /// Every socket type can reach every other.
    ///
    /// The point of the conversion set: a graph should never be stuck because a
    /// value is the wrong shape. This asserts the vocabulary actually spans it
    /// rather than leaving one direction quietly missing.
    #[test]
    fn the_conversion_set_spans_every_pair_of_types() {
        use ValueType::*;

        let palette = NodeKind::palette();
        for (from, to) in [
            (Float, Vec3),
            (Float, Color),
            (Vec3, Float),
            (Vec3, Color),
            (Color, Float),
            (Color, Vec3),
        ] {
            let reachable = palette.iter().any(|kind| {
                kind.inputs().iter().any(|i| i.ty == from)
                    && kind.outputs().iter().any(|o| o.ty == to)
            });
            assert!(reachable, "nothing converts {from} to {to}");
        }
    }

    #[test]
    fn the_named_geometry_output_indices_point_where_they_claim() {
        // These constants are how the emitter decides whether to bind a
        // conditional attribute. If a socket is ever inserted above them, this
        // fails here rather than as a shader that reads the normal as a UV.
        assert_eq!(GEOMETRY_OUT[GEOMETRY_UV_OUTPUT].ident, "uv");
        assert_eq!(
            GEOMETRY_OUT[GEOMETRY_VERTEX_COLOR_OUTPUT].ident,
            "vertex_color"
        );
    }

    #[test]
    fn param_output_type_follows_its_value() {
        assert_eq!(
            NodeKind::param(Value::Float(1.0)).outputs()[0].ty,
            ValueType::Float
        );
        assert_eq!(
            NodeKind::param(Value::Color([0.0; 4])).outputs()[0].ty,
            ValueType::Color
        );
    }

    #[test]
    fn snippet_order_places_dependencies_first() {
        let at = |want: Snippet| Snippet::ALL.iter().position(|s| *s == want).unwrap();
        assert!(
            at(Snippet::Hash) < at(Snippet::Voronoi),
            "voronoi calls into hash"
        );
        assert!(
            at(Snippet::Hash) < at(Snippet::Perlin),
            "perlin calls into hash"
        );
        assert!(
            at(Snippet::Perlin) < at(Snippet::Fbm),
            "fbm calls into perlin"
        );
    }

    /// The octave cap is written in two languages and must agree.
    ///
    /// If the WGSL cap were lower, the live variant would silently stop short of
    /// what an unrolled node generates and the two would stop matching — which
    /// destroys the only reason both exist.
    #[test]
    fn the_octave_cap_agrees_across_the_language_boundary() {
        let declared = format!("const SG_MAX_OCTAVES: i32 = {MAX_NOISE_OCTAVES};");
        assert!(
            Snippet::Perlin.source().contains(&declared),
            "wgsl/perlin.wgsl does not declare `{declared}`"
        );
    }

    /// Both noise variants must sum octaves the same way.
    ///
    /// Asserted on the generated text rather than by running it, because the two
    /// live in different languages' control flow — a loop and a straight line —
    /// and the thing that has to match is which terms they build, not how they
    /// iterate. Both must reach for the same two shared functions.
    #[test]
    fn the_unrolled_noise_mirrors_the_dynamic_loop() {
        let unrolled = unrolled_fbm("sg_test", 3);
        let dynamic = Snippet::Fbm.source();

        for shared in ["sg_octave_weight(detail,", "sg_perlin_3d(p"] {
            assert!(unrolled.contains(shared), "unrolled is missing {shared}");
        }
        for shared in ["sg_octave_weight(detail,", "sg_perlin_3d(p"] {
            assert!(dynamic.contains(shared), "the loop is missing {shared}");
        }

        // Three octaves, so three weight terms and two frequency steps between
        // them — the last octave must not advance state nothing reads.
        assert_eq!(unrolled.matches("sg_perlin_3d(p * freq)").count(), 3);
        assert_eq!(unrolled.matches("freq = freq * lacunarity;").count(), 2);

        // Both end at the same normalisation, or their outputs are on different
        // scales and no comparison between them means anything.
        for source in [unrolled.as_str(), dynamic] {
            assert!(source.contains("return sum / norm * 0.5 + 0.5;"));
        }
    }

    /// Mapping applies scale, then rotation, then translation.
    ///
    /// Order is the whole content of this node. Scaling after rotation shears on
    /// any non-uniform scale; translating before rotation swings the offset
    /// around with the angle. Both look plausible until you use two controls at
    /// once, which is exactly when nobody suspects the node.
    #[test]
    fn mapping_transforms_in_blenders_order() {
        let args = [
            "v".to_string(),
            "loc".to_string(),
            "rot".to_string(),
            "scl".to_string(),
        ];
        assert_eq!(
            NodeKind::Mapping.expr(&args),
            "(sg_euler_xyz(rot) * (v * scl) + loc)"
        );
    }

    /// The Euler matrix is built from columns, not rows.
    ///
    /// NOTE: This is the one bug in `sg_euler_xyz` that produces a *plausible*
    /// result: transposing a rotation matrix inverts the rotation, so every
    /// single-axis test still looks fine and only the direction is backwards.
    /// Asserted against the derivation written out in `wgsl/mapping.wgsl` — the
    /// `-s.y` term belongs to column 0, and appears nowhere else in the matrix,
    /// so finding it there pins the orientation.
    ///
    /// What this cannot check is whether Blender's convention is the one we
    /// want on screen. Only looking at a rotated grain does that.
    #[test]
    fn the_euler_matrix_is_column_major() {
        let source = Snippet::Mapping.source();
        assert!(
            source.contains("vec3<f32>(c.z * c.y, s.z * c.y, -s.y)"),
            "column 0 of the Euler matrix does not match the derivation"
        );
    }

    /// A blend factor must never extrapolate.
    ///
    /// WGSL's `mix` does not clamp, so a factor above 1 sends a color past its
    /// own endpoint and out of gamut. That reads as a lighting bug rather than a
    /// wiring one, which is a genuinely expensive kind of confusion.
    #[test]
    fn every_mix_node_clamps_its_factor() {
        for kind in [
            NodeKind::MixFloat,
            NodeKind::MixVector,
            NodeKind::MixColor(BlendMode::Mix),
            NodeKind::MixColor(BlendMode::Overlay),
        ] {
            let args = ["n1_wired".to_string(), "a".to_string(), "b".to_string()];
            let expr = kind.expr(&args);
            assert!(
                expr.contains("clamp(n1_wired, 0.0, 1.0)"),
                "{} did not clamp its factor: {expr}",
                kind.name()
            );
        }
    }

    /// A literal factor is folded rather than wrapped.
    ///
    /// `clamp(0.5, 0.0, 1.0)` in generated source is noise that makes every
    /// golden file harder to read for no behavioural gain.
    #[test]
    fn a_literal_factor_is_clamped_at_compile_time() {
        let args = ["1.4".to_string(), "a".to_string(), "b".to_string()];
        assert_eq!(NodeKind::MixFloat.expr(&args), "mix(a, b, 1.0)");

        let args = ["0.25".to_string(), "a".to_string(), "b".to_string()];
        assert_eq!(NodeKind::MixFloat.expr(&args), "mix(a, b, 0.25)");
    }

    /// Alpha interpolates plainly however the color channels combine.
    #[test]
    fn a_blend_mode_never_touches_alpha() {
        let args = ["f".to_string(), "a".to_string(), "b".to_string()];

        // Multiply on alpha would quietly make a surface transparent — a bug
        // that presents as a lighting problem.
        let expr = NodeKind::MixColor(BlendMode::Multiply).expr(&args);
        assert!(
            expr.contains("(b).a"),
            "alpha must come from B, not the blend: {expr}"
        );
        assert!(
            !expr.contains("(a).a * (b).a"),
            "alpha was blended rather than interpolated: {expr}"
        );

        // Plain Mix stays the simple whole-vector form.
        assert_eq!(
            NodeKind::MixColor(BlendMode::Mix).expr(&args),
            "mix(a, b, clamp(f, 0.0, 1.0))"
        );
    }

    /// Only the modes that need the helper pull the snippet in.
    #[test]
    fn only_overlay_reaches_for_the_blend_snippet() {
        assert!(NodeKind::MixColor(BlendMode::Overlay)
            .snippets()
            .contains(&Snippet::Blend));
        for mode in [BlendMode::Mix, BlendMode::Multiply, BlendMode::Screen] {
            assert!(
                NodeKind::MixColor(mode).snippets().is_empty(),
                "{} emitted a snippet it does not use",
                mode.label()
            );
        }
    }

    #[test]
    fn an_unrolled_octave_count_is_clamped_to_something_useful() {
        assert_eq!(NoiseOctaves::Unrolled(0).unrolled_count(), 1);
        assert_eq!(
            NoiseOctaves::Unrolled(99).unrolled_count(),
            MAX_NOISE_OCTAVES
        );
        assert_eq!(NoiseOctaves::Unrolled(4).unrolled_count(), 4);
    }

    /// The live variant must not drag the unrolled one's dead loop along, and
    /// vice versa.
    #[test]
    fn each_noise_variant_pulls_in_only_the_snippets_it_uses() {
        let live = NodeKind::Noise(NoiseOctaves::Live).snippets();
        assert!(live.contains(&Snippet::Fbm), "live needs the loop");

        let unrolled = NodeKind::Noise(NoiseOctaves::Unrolled(4)).snippets();
        assert!(
            !unrolled.contains(&Snippet::Fbm),
            "an unrolled node emitted the dynamic loop it exists to avoid"
        );
        assert!(
            unrolled.contains(&Snippet::Perlin),
            "unrolled still calls perlin"
        );
    }
}
