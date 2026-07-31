//! The loop the editor exists for: edit a graph, see the lit result.
//!
//! # Why this does not diff
//!
//! Every frame the document is compiled and the result compared, as text, to
//! what is already on the mesh. Identical means there was no edit and nothing
//! happens. This sounds wasteful and is not: compiling a few dozen nodes is
//! string manipulation over a small IR, and comparing a few kilobytes costs
//! less than the bookkeeping a dirty flag would need.
//!
//! It is also the only version that cannot be wrong. A dirty flag has to be set
//! by every code path that can change the document — dragging a wire, editing a
//! value, deleting a node, loading a preset — and the one that forgets is a bug
//! that presents as "the preview stopped updating", which is indistinguishable
//! from the compiler being broken.
//!
//! # Why it validates first
//!
//! Bevy does not crash on a bad shader: `PipelineCache` logs the naga
//! diagnostic and leaves the pipeline in an error state, and because writing the
//! shader asset again re-queues its pipelines, fixing the graph recovers on its
//! own. So validation here is not crash protection. It is so the diagnostic
//! arrives in a panel next to the graph that caused it, instead of in a terminal
//! the author is not looking at — and so the *last good* shader stays on the
//! mesh while a half-finished graph is in progress, rather than the surface
//! going blank the instant a wire is unplugged.

use bevy::asset::LoadState;
use bevy::image::{
    ImageAddressMode, ImageFilterMode, ImageLoaderSettings, ImageSampler, ImageSamplerDescriptor,
};
use bevy::prelude::*;
use shadergraph::standalone::{numbered, to_standalone_wgsl, MESH_ATTRIBUTE_DEFS};
use shadergraph::{ParamLayout, TextureSlot, TEXTURE_SLOTS};
use shadergraph_bevy::{compile_graph, recompile_graph_in_place, GraphMaterial};

use crate::canvas::GraphDocument;
use crate::preview::PreviewSubject;

/// The result of the most recent compile attempt.
#[derive(Default)]
pub enum CompileStatus {
    #[default]
    Pending,
    /// Compiled, validated and applied.
    Ok,
    /// Rejected. The mesh still shows whatever last succeeded.
    Failed(String),
}

#[derive(Resource, Default)]
pub struct CompileState {
    pub status: CompileStatus,
    /// The WGSL currently on the mesh, and the comparison key for the next
    /// frame's compile.
    pub wgsl: String,
    /// Slot assignment for the shader currently on the mesh, and the values last
    /// uploaded. Changing a `Param` leaves the WGSL identical, so this — not the
    /// source — is what says an upload is due.
    layout: ParamLayout,
    /// Set when a value edit was applied without recompiling. Shown in the UI,
    /// because "the picture changed and the shader did not" is the claim this
    /// whole design makes and it should be visible rather than asserted.
    pub live_param_writes: u64,
    /// Incremented once per *structural* recompile — a new shader, not a new
    /// value.
    ///
    /// A counter rather than a flag because the consumer
    /// ([`crate::keep_rendering_while_it_matters`]) needs to notice the change
    /// from a system that does not run every frame. Bevy's own change detection
    /// cannot serve here: `recompile` takes this resource as `ResMut` and writes
    /// to it unconditionally, so `is_changed()` is true on every frame and says
    /// nothing.
    pub compiles: u64,
    /// Optional mesh attributes the current graph reads. Paired with
    /// [`crate::preview::PreviewMeshAttributes`] to warn when the graph wants
    /// something the mesh on the stand has not got.
    pub uses_uv: bool,
    pub uses_vertex_color: bool,
    material: Option<Handle<GraphMaterial>>,
    /// Image handles by the path that produced them, so a path is loaded once
    /// rather than once per frame. Grows while paths are being typed — every
    /// keystroke is a distinct path — so it is bounded in [`bind_textures`]
    /// rather than trusted to stay small.
    image_cache: std::collections::HashMap<String, Handle<Image>>,
    /// Texture slots that are not showing the image the graph names — a
    /// missing path, a file that failed to load. The slot renders fallback
    /// white meanwhile, which is exactly why these need to be *said*: the
    /// mesh keeps drawing, so nothing else looks wrong.
    pub texture_warnings: Vec<String>,
}

impl CompileState {
    pub fn error(&self) -> Option<&str> {
        match &self.status {
            CompileStatus::Failed(message) => Some(message),
            _ => None,
        }
    }
}

pub struct CompilePlugin;

impl Plugin for CompilePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CompileState>()
            // Deliberately in `Update` rather than inside the egui pass. With
            // multipass enabled the UI closure runs more than once per frame, so
            // compiling there would do the same work two or three times over.
            //
            // `bind_textures` is chained after `recompile` so a texture edit
            // lands the same frame it compiled, rather than one behind.
            .add_systems(Update, (recompile, bind_textures).chain());
    }
}

fn recompile(
    document: Res<GraphDocument>,
    mut state: ResMut<CompileState>,
    mut shaders: ResMut<Assets<Shader>>,
    mut materials: ResMut<Assets<GraphMaterial>>,
    subject: Query<Entity, With<PreviewSubject>>,
    mut commands: Commands,
    preview: Res<crate::preview::PreviewSettings>,
) {
    let graph = match document.to_graph() {
        Ok(graph) => graph,
        Err(e) => {
            state.status = CompileStatus::Failed(e.to_string());
            return;
        }
    };

    // Caught here rather than left to `emit` because these two carry the
    // messages that actually name the offending node and socket.
    if let Err(e) = graph.validate() {
        state.status = CompileStatus::Failed(e.to_string());
        return;
    }

    // Recorded before the fast path below returns, so the attribute chips stay
    // correct on a frame where only a value changed.
    state.uses_uv = graph.uses_uv();
    state.uses_vertex_color = graph.uses_vertex_color();

    let (wgsl, layout) = match shadergraph::emit_with_layout(&graph) {
        Ok(pair) => pair,
        Err(e) => {
            state.status = CompileStatus::Failed(e.to_string());
            return;
        }
    };

    // EXPERIMENT: A/B harness for the bump quad-derivative artifact — see
    // `PreviewSettings::fine_derivatives`. A text rewrite rather than a
    // compiler option deliberately: the two spellings differ only inside
    // `sg_bump`, and the winner gets baked into wgsl/bump.wgsl, not kept
    // switchable. Applied before the fast-path compare, so flipping the
    // toggle is a structural recompile like any other source change.
    let wgsl = if preview.fine_derivatives {
        wgsl.replace("dpdx(", "dpdxFine(").replace("dpdy(", "dpdyFine(")
    } else {
        wgsl
    };

    // The fast path, and the reason parameters moved into a uniform. A value
    // edit leaves the source byte-identical, so there is nothing to validate and
    // no pipeline to rebuild — the new numbers go straight into the buffer.
    if wgsl == state.wgsl && state.material.is_some() {
        // NOTE: Cleared here as well as on the structural path below, and it has to
        // be. Recovering from a broken graph does not always change the source:
        // unplug a wire and plug the same one back, or fix a branch that was
        // never reached, and the emitted WGSL is byte-identical to the last good
        // one — so this returns early and the stale error would sit in the panel
        // forever, describing a graph that no longer exists.
        //
        // Reaching this line at all means `to_graph`, `validate` and `emit` have
        // just succeeded, which is the entire definition of Ok.
        state.status = CompileStatus::Ok;
        if layout != state.layout {
            debug_assert!(
                layout.same_shape(&state.layout),
                "identical WGSL with a different slot shape means the emitter \
                 and the layout have drifted apart"
            );
            if let Some(mut material) = state
                .material
                .clone()
                .and_then(|handle| materials.get_mut(&handle))
            {
                material.params.write(&layout);
                material.layout = layout.clone();
                state.layout = layout;
                state.live_param_writes += 1;
            }
        }
        return;
    }

    if let Err(message) = validate_wgsl(&wgsl) {
        state.status = CompileStatus::Failed(message);
        return;
    }

    match state.material.clone() {
        // Rewrite the existing shader in place. `AssetEvent::Modified` reaches
        // `PipelineCache`, which re-queues every pipeline built from it — so the
        // material, its handle and the entity holding it all stay exactly as
        // they are, and only the compiled program changes. The re-emit inside
        // the helper is redundant with the compile above but only runs on the
        // rare structural path, and using the library's own live-edit entry
        // point keeps this tool honest about exercising it.
        Some(handle) => {
            let Some(mut material) = materials.get_mut(&handle) else {
                return;
            };
            if let Err(e) =
                recompile_graph_in_place(&graph, "preview", &material.shader, &mut shaders)
            {
                state.status = CompileStatus::Failed(e.to_string());
                return;
            }
            material.params.write(&layout);
            material.layout = layout.clone();
            // Anisotropy is part of the pipeline key, so this is not cosmetic:
            // wiring up a brush direction has to switch the lighting path on,
            // and unwiring it has to switch it back off.
            material.anisotropic = graph.uses_anisotropy();
        }

        // First successful compile: build the material and put it on the mesh.
        None => {
            let material = match compile_graph(&graph, "preview", &mut shaders) {
                Ok(material) => material,
                Err(e) => {
                    state.status = CompileStatus::Failed(e.to_string());
                    return;
                }
            };
            let handle = materials.add(material);
            for entity in &subject {
                commands
                    .entity(entity)
                    .insert(MeshMaterial3d(handle.clone()));
            }
            state.material = Some(handle);
        }
    }

    // EXPERIMENT: both branches above re-emit from the graph inside the
    // library helpers, so what they wrote to the shader asset is the plain
    // spelling. When the toggle is on, write the rewritten source over it —
    // the second Modified event lands the same frame and coalesces into the
    // same pipeline re-queue.
    if preview.fine_derivatives {
        if let Some(shader) = state
            .material
            .as_ref()
            .and_then(|handle| materials.get(handle))
            .map(|material| material.shader.clone())
        {
            let _ = shaders.insert(
                shader.id(),
                Shader::from_wgsl(wgsl.clone(), "shadergraph://preview.wgsl".to_string()),
            );
        }
    }

    state.wgsl = wgsl;
    state.layout = layout;
    state.status = CompileStatus::Ok;
    // Reached only on the structural path; the value-only fast path returns
    // above. This is what tells the update-mode system that a pipeline is about
    // to be built and the loop must keep turning until it is.
    state.compiles += 1;
}

/// Keep the preview material's texture slots bound to what the layout names.
///
/// Runs every frame, like `recompile`, and for the same reason: the paths on
/// the graph, the load state of their images, and the material's fields can
/// each change independently, and polling all three is cheaper than getting
/// the event bookkeeping for their product right. The material asset is only
/// written when a slot actually changes, because writing one re-prepares its
/// bind group.
///
/// A slot resolves to its image only once the asset has **loaded**. While it
/// is loading — and if it fails, or the path is empty — the slot binds `None`,
/// which the renderer draws as opaque white. The mesh never vanishes over a
/// texture; the warning list is how the author finds out white meant
/// "missing" rather than "white".
fn bind_textures(
    mut state: ResMut<CompileState>,
    assets: Res<AssetServer>,
    mut materials: ResMut<Assets<GraphMaterial>>,
) {
    let Some(handle) = state.material.clone() else {
        return;
    };

    // Typing a path is one distinct string per keystroke, each of which lands
    // here as a cache entry. Dropping the whole cache on overflow is crude and
    // correct: live handles are re-requested next frame, and `AssetServer`
    // dedupes loads of paths it already knows.
    if state.image_cache.len() > 256 {
        state.image_cache.clear();
    }

    let textures: Vec<TextureSlot> = state.layout.textures().to_vec();
    let mut desired: Vec<Option<Handle<Image>>> = vec![None; TEXTURE_SLOTS];
    let mut warnings = Vec::new();

    for (slot, texture) in textures.iter().enumerate().take(TEXTURE_SLOTS) {
        let label = texture
            .name
            .clone()
            .unwrap_or_else(|| format!("slot {slot}"));
        if texture.path.is_empty() {
            warnings.push(format!(
                "texture {label} has no image path; it samples white"
            ));
            continue;
        }

        let image = match state.image_cache.get(&texture.path) {
            Some(image) => image.clone(),
            None => {
                let image = load_image(&assets, &texture.path);
                state
                    .image_cache
                    .insert(texture.path.clone(), image.clone());
                image
            }
        };

        match assets.get_load_state(image.id()) {
            Some(LoadState::Loaded) => desired[slot] = Some(image),
            Some(LoadState::Failed(error)) => warnings.push(format!(
                "texture {label} failed to load {}: {error}; it samples white",
                texture.path
            )),
            // Still loading (or not yet tracked): fallback white for now, no
            // warning — this state resolves itself within a few frames.
            _ => {}
        }
    }

    if state.texture_warnings != warnings {
        state.texture_warnings = warnings;
    }

    // Compare against the material before touching it mutably; `get_mut` is
    // what marks the asset modified.
    let Some(material) = materials.get(&handle) else {
        return;
    };
    let unchanged =
        (0..TEXTURE_SLOTS).all(|slot| material.texture_slot(slot) == desired[slot].as_ref());
    if unchanged {
        return;
    }

    let Some(mut material) = materials.get_mut(&handle) else {
        return;
    };
    for (slot, image) in desired.into_iter().enumerate() {
        let _ = material.set_texture_slot(slot, image);
    }
}

/// Load an image for a texture slot, through the same `abs` asset source
/// dropped meshes use — a graph names files anywhere on disk, not under an
/// asset root this tool does not have.
///
/// Wrap is Repeat and filtering trilinear, chosen here because the sampler
/// rides with the image: tiling detail is what material textures overwhelmingly
/// are, and the engine's default clamp-to-edge turns a tiled coordinate into
/// smeared edge pixels that read as a broken texture rather than a sampler
/// setting.
fn load_image(assets: &AssetServer, path: &str) -> Handle<Image> {
    let absolute = std::path::absolute(path).unwrap_or_else(|_| path.into());
    let label = format!(
        "abs://{}",
        absolute.to_string_lossy().trim_start_matches('/')
    );
    assets
        .load_builder()
        .with_settings(|settings: &mut ImageLoaderSettings| {
            settings.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
                address_mode_u: ImageAddressMode::Repeat,
                address_mode_v: ImageAddressMode::Repeat,
                mag_filter: ImageFilterMode::Linear,
                min_filter: ImageFilterMode::Linear,
                mipmap_filter: ImageFilterMode::Linear,
                ..Default::default()
            });
        })
        .load(&label)
}

/// Parse and type-check generated WGSL, standalone, for both kinds of mesh.
///
/// The engine's imports are stripped and replaced with a transcribed stub — see
/// [`shadergraph::standalone`] — so this proves the generated body is
/// well-formed WGSL using the expected surface. It cannot prove the shader is
/// *right*; only looking at the mesh does that.
///
/// Both branches are checked even though the preview only ever renders one of
/// them. A graph reading UVs compiles differently on a mesh without a UV
/// channel, and the author should find that out here rather than the next time
/// the material lands on an un-unwrapped mesh in a real application.
fn validate_wgsl(wgsl: &str) -> Result<(), String> {
    for (mesh, defs) in [
        ("a mesh with UVs and vertex colors", MESH_ATTRIBUTE_DEFS),
        ("a mesh with neither", &[][..]),
    ] {
        validate_one(wgsl, mesh, defs)?;
    }
    Ok(())
}

fn validate_one(wgsl: &str, mesh: &str, defs: &[&str]) -> Result<(), String> {
    let source = to_standalone_wgsl(wgsl, defs)
        .map_err(|e| format!("generated WGSL has unbalanced preprocessor directives: {e}"))?;

    let module = naga::front::wgsl::parse_str(&source).map_err(|e| {
        format!(
            "generated WGSL does not parse on {mesh}:\n{}\n\n{}",
            e.emit_to_string(&source),
            numbered(&source)
        )
    })?;

    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| {
        format!(
            "generated WGSL does not type-check on {mesh}:\n{}\n\n{}",
            e.emit_to_string(&source),
            numbered(&source)
        )
    })?;

    Ok(())
}
