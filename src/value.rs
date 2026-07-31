//! Socket data types and the literals that fill unconnected inputs.
//!
//! The type vocabulary here is deliberately tiny — three types, not a type
//! *system*. That is a direct consequence of the design decision to split
//! polymorphic nodes into one node type per data type (`MixColor` rather than a
//! `Mix` with a `data_type` dropdown hiding three quarters of its sockets).
//! With that split, type checking degenerates into "do these two socket types
//! match", which needs no inference, no unification, and no coercion rules.

/// The data type carried by a socket.
///
/// `Color` is `vec4<f32>` rather than `vec3<f32>` because alpha has to survive
/// the trip to `PbrOutput`, and a color that silently drops its fourth channel
/// halfway through a graph is a bug that only shows up on transparent materials.
/// Serialised because [`crate::node::NodeKind::Reroute`] carries one — it is the
/// only node whose *type* is part of its saved state rather than implied by its
/// kind. Nothing else needs this, and nothing else should start: a socket type
/// belonging to a node is what makes reroutes polymorphic, and that is meant to
/// stay the exception.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ValueType {
    /// A scalar; `f32` in WGSL.
    Float,
    /// A direction or position; `vec3<f32>` in WGSL.
    Vec3,
    /// RGBA, linear, unpremultiplied; `vec4<f32>` in WGSL.
    Color,
    /// A complete set of evaluated surface properties — everything `PbrOutput`
    /// needs, travelling down one wire.
    ///
    /// The odd one out, and deliberately so. The other three are *data*: they
    /// have literals, they can fill an unwired socket, and a `Param` can hold
    /// one. A surface has none of those properties — there is no such thing as a
    /// literal surface, so every `Surface` input is
    /// [`crate::node::SocketDefault::Required`] and no `Value` variant
    /// corresponds to it.
    ///
    /// It exists so that mixing two whole materials is one wire rather than ten.
    /// This is the piece of Blender's design that does *not* transfer as-is:
    /// Blender's `Shader` socket carries an unevaluated closure, which is why a
    /// Mix Shader cannot be compiled to a fixed structure. Ours carries a struct
    /// of numbers that has already been evaluated, so mixing is a lerp. See
    /// `Surface` and `MixSurface` in [`crate::node`].
    Surface,
}

impl ValueType {
    /// The WGSL spelling of this type.
    pub fn wgsl(self) -> &'static str {
        match self {
            ValueType::Float => "f32",
            ValueType::Vec3 => "vec3<f32>",
            ValueType::Color => "vec4<f32>",
            // Declared in the emitted header rather than by a snippet: every
            // graph ends in a surface, so gating it would be a condition that is
            // always true.
            ValueType::Surface => "SgSurface",
        }
    }
}

impl core::fmt::Display for ValueType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            ValueType::Float => "Float",
            ValueType::Vec3 => "Vector",
            ValueType::Color => "Color",
            ValueType::Surface => "Surface",
        })
    }
}

/// A literal value: either a node's constant, or the fallback for an input
/// socket with no incoming wire.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Value {
    /// A scalar literal.
    Float(f32),
    /// A vector literal.
    Vec3([f32; 3]),
    /// An RGBA color literal, linear and unpremultiplied.
    Color([f32; 4]),
}

impl Value {
    /// The [`ValueType`] this literal fills a socket of.
    pub fn ty(&self) -> ValueType {
        match self {
            Value::Float(_) => ValueType::Float,
            Value::Vec3(_) => ValueType::Vec3,
            Value::Color(_) => ValueType::Color,
        }
    }

    /// Render as a WGSL literal expression.
    pub fn wgsl_literal(&self) -> String {
        match self {
            Value::Float(v) => wgsl_f32(*v),
            Value::Vec3(v) => format!(
                "vec3<f32>({}, {}, {})",
                wgsl_f32(v[0]),
                wgsl_f32(v[1]),
                wgsl_f32(v[2])
            ),
            Value::Color(v) => format!(
                "vec4<f32>({}, {}, {}, {})",
                wgsl_f32(v[0]),
                wgsl_f32(v[1]),
                wgsl_f32(v[2]),
                wgsl_f32(v[3])
            ),
        }
    }
}

/// Format an `f32` as a WGSL float literal.
///
/// WGSL infers `1` as an integer, so `vec3<f32>(1, 0, 0)` is a type error where
/// `vec3<f32>(1.0, 0.0, 0.0)` is fine. Every float this crate emits therefore
/// has to carry a decimal point or an exponent. Rust's default `{}` formatting
/// prints `1f32` as `"1"`, so the trailing `.0` is added by hand.
///
/// This is also why formatting lives in one function rather than at each call
/// site: golden-file tests compare emitted WGSL byte for byte, so a second
/// float-formatting rule appearing elsewhere would show up as spurious test
/// churn rather than as the bug it is.
pub fn wgsl_f32(v: f32) -> String {
    if !v.is_finite() {
        // WGSL has no inf/nan literals. Emitting one would produce a shader that
        // fails to compile at runtime with a message pointing at generated code
        // the author never wrote, so fail loudly here instead.
        panic!("cannot emit non-finite float {v} as a WGSL literal");
    }
    let s = format!("{v}");
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        format!("{s}.0")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whole_numbers_get_a_decimal_point() {
        assert_eq!(wgsl_f32(1.0), "1.0");
        assert_eq!(wgsl_f32(0.0), "0.0");
        assert_eq!(wgsl_f32(-3.0), "-3.0");
    }

    #[test]
    fn fractions_are_left_alone() {
        assert_eq!(wgsl_f32(0.5), "0.5");
        assert_eq!(wgsl_f32(1.45), "1.45");
    }

    #[test]
    fn literals_render_with_explicit_constructors() {
        assert_eq!(Value::Float(2.0).wgsl_literal(), "2.0");
        assert_eq!(
            Value::Vec3([1.0, 0.0, -1.0]).wgsl_literal(),
            "vec3<f32>(1.0, 0.0, -1.0)"
        );
        assert_eq!(
            Value::Color([0.5, 0.5, 0.5, 1.0]).wgsl_literal(),
            "vec4<f32>(0.5, 0.5, 0.5, 1.0)"
        );
    }

    #[test]
    #[should_panic(expected = "non-finite")]
    fn non_finite_floats_are_rejected() {
        wgsl_f32(f32::INFINITY);
    }
}
