//! A node graph that compiles to WGSL.
//!
//! Procedural materials are authored as a graph of small nodes and emitted as
//! shader source. This crate is the compiler half only: it has no editor, no
//! renderer, and no dependencies.
//!
//! ```
//! use shadergraph::{emit, materials};
//!
//! let graph = materials::hull_metal().unwrap();
//! let wgsl = emit(&graph).unwrap();
//! assert!(wgsl.contains("apply_pbr_lighting"));
//! ```
//!
//! # Why a compiler rather than an uber-shader
//!
//! The alternative — one large shader with a runtime variant switch and a fixed
//! uniform block — puts every material's code on the GPU whether or not a given
//! draw call uses it, and makes a new look a source edit plus an enum arm.
//! Emitting per graph means a shader contains only the nodes its material
//! actually reaches.
//!
//! # Why the graph terminates in `PbrOutput` rather than a BSDF
//!
//! Blender's shader nodes split cleanly in two. Nodes whose sockets carry
//! floats, vectors and colors are pure functions and port to WGSL almost
//! directly. Nodes touching a `Shader` socket are not functions at all — a
//! `Shader` socket is a closure that the renderer's integrator evaluates, and
//! Cycles and EEVEE implement it entirely differently. There is no WGSL
//! translation of a Principled BSDF, and pursuing one is how a project like
//! this fails.
//!
//! So the boundary is drawn at exactly "everything upstream of the first
//! `Shader` socket", and the graph terminates in [`node::NodeKind::PbrOutput`],
//! whose sockets are the surface properties a physically-based renderer
//! consumes. The renderer keeps its own lighting model; the graph decides what
//! the surface *is*, not how light behaves when it arrives.
//!
//! # Licensing
//!
//! The WGSL in `src/wgsl/` is written from published algorithm descriptions,
//! each carrying a provenance header, so that the crate's `MIT OR Apache-2.0`
//! licence is honest. See `Cargo.toml` before pasting in third-party shader
//! code.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

/// Turning a validated graph into WGSL source.
pub mod emit;
/// The graph itself: nodes, wires, validation, and evaluation order.
pub mod graph;
/// Material graphs built by hand, usable as presets and exercised as tests.
pub mod materials;
/// The node vocabulary: kinds, sockets, and how each one emits.
pub mod node;
/// The uniform slot table that lets parameters change without recompiling.
pub mod params;
/// color ramps: ordered stops and the interpolation between them.
pub mod ramp;
/// Headless validation support: a stub of the engine surface the shaders import.
pub mod standalone;
/// The value types a socket can carry, and literal values for defaults.
pub mod value;

pub use emit::{emit, emit_with_layout};
pub use graph::{Graph, GraphError, NodeId};
pub use node::{
    BlendMode, ColorToFloatOp, Emission, GradientType, MathOp, NodeKind, NoiseOctaves, Snippet,
    SocketDef, SocketDefault, VectorOp, VectorToFloatOp, GEOMETRY_UV_OUTPUT,
    GEOMETRY_VERTEX_COLOR_OUTPUT, MAX_NOISE_OCTAVES,
};
pub use params::{ParamLayout, ParamSlot, SlotValue, TextureSlot, PARAM_SLOTS, TEXTURE_SLOTS};
pub use ramp::{ColorRamp, ColorStop, RampInterp};
pub use value::{Value, ValueType};
