//! Renders a `shadergraph` graph in Bevy.
//!
//! `shadergraph` itself has no idea a renderer exists — it turns a graph into
//! WGSL text and stops. This crate is the other half: the [`Material`] that
//! carries a compiled shader, and the two calls that get a graph onto a mesh.
//!
//! It is a separate crate from any application that uses it because the
//! authoring tool needs the *same* binding. A preview that renders through a second copy of
//! this code is a preview of something other than what ships, and the moment
//! the copies drift the tool starts lying.
//!
//! ```no_run
//! # use bevy::prelude::*;
//! # use shadergraph_bevy::{compile_graph, GraphMaterialPlugin};
//! # fn setup(
//! #     mut shaders: ResMut<Assets<Shader>>,
//! #     mut mats: ResMut<Assets<shadergraph_bevy::GraphMaterial>>,
//! # ) -> Result<(), shadergraph::GraphError> {
//! let graph = shadergraph::materials::hull_metal()?;
//! let mut material = compile_graph(&graph, "hull_metal", &mut shaders)?;
//! // Named inputs can be driven straight away; the material carries its layout.
//! let _ = material.set_input("wear_level", shadergraph::Value::Float(0.7));
//! let handle = mats.add(material);
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod material;

use bevy::prelude::*;
use shadergraph::{emit_with_layout, Graph, GraphError, ParamLayout};

pub use material::{GraphMaterial, GraphMaterialKey, GraphMaterialParams, SetInputError};
pub use shadergraph::{PARAM_SLOTS, TEXTURE_SLOTS};

/// Registers [`GraphMaterial`] with the renderer.
///
/// Adding `MaterialPlugin::<GraphMaterial>::default()` by hand does the same
/// thing; this exists so a consumer does not have to know that is what it is.
pub struct GraphMaterialPlugin;

impl Plugin for GraphMaterialPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<GraphMaterial>::default());
    }
}

/// Compile a graph and build an opaque material from it.
///
/// `label` names the shader in renderer diagnostics. It is not a file path and
/// nothing reads it from disk, but wgpu will quote it back in a validation
/// error, and an error citing `shadergraph://hull_metal.wgsl` is a great deal
/// more useful than one citing nothing.
///
/// The anisotropic lighting path is enabled from [`Graph::uses_anisotropy`]
/// rather than by a flag the caller has to remember, because forgetting it
/// produces a wrong-looking surface with no diagnostic at all.
///
/// Use [`GraphMaterial::with_alpha_mode`] on the result for anything
/// transparent — that is a property of how the surface is *drawn*, which the
/// graph does not describe.
pub fn compile_graph(
    graph: &Graph,
    label: &str,
    shaders: &mut Assets<Shader>,
) -> Result<GraphMaterial, GraphError> {
    let (wgsl, layout) = emit_with_layout(graph)?;
    let shader = shaders.add(Shader::from_wgsl(
        wgsl,
        format!("shadergraph://{label}.wgsl"),
    ));

    let mut material = GraphMaterial::opaque(shader);
    material.params.write(&layout);
    material.layout = layout;
    if graph.uses_anisotropy() {
        material = material.with_anisotropy();
    }
    Ok(material)
}

/// Replace the source of an already-compiled shader, in place.
///
/// This is what makes live editing possible. `AssetEvent::Modified` on a
/// `Shader` reaches `PipelineCache`, which re-queues every pipeline built from
/// it — so overwriting the asset behind an existing `AssetId` recompiles the
/// material without touching the material, its handle, or any entity holding
/// it. Adding a *new* shader instead would mean chasing down every mesh.
///
/// On failure the previous shader is left untouched, which is the behaviour an
/// editor wants: a half-finished graph should leave the last good surface on
/// screen rather than blanking it.
pub fn recompile_graph_in_place(
    graph: &Graph,
    label: &str,
    shader: &Handle<Shader>,
    shaders: &mut Assets<Shader>,
) -> Result<ParamLayout, GraphError> {
    let (wgsl, layout) = emit_with_layout(graph)?;
    shaders
        .insert(
            shader.id(),
            Shader::from_wgsl(wgsl, format!("shadergraph://{label}.wgsl")),
        )
        .expect("the caller holds a live strong handle, so its generation cannot be stale");
    Ok(layout)
}
