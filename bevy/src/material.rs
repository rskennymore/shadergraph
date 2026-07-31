//! The Bevy material that renders a compiled shader graph.
//!
//! # One Rust type, many shaders
//!
//! `Material::fragment_shader()` is an associated function with no `&self`, so
//! Bevy resolves it once per *material type*. Taken at face value that would
//! mean a Rust type per graph, and a build script to generate them.
//!
//! It is avoidable. `MaterialPipeline::specialize` assigns the default fragment
//! shader and *then* calls `Material::specialize`, so an override here wins.
//! `#[bind_group_data]` is what carries the per-instance shader identity into
//! the pipeline key that `specialize` receives, and because the key is part of
//! the pipeline cache's lookup, two materials with different graphs correctly
//! specialize to two different pipelines instead of sharing one.
//!
//! So: one [`GraphMaterial`] type, arbitrarily many graphs, no build step, and
//! graphs can be compiled at runtime.
//!
//! # The parameter uniform
//!
//! `Param` nodes do not appear in the generated WGSL as numbers; they read slots
//! of the array bound here. See `shadergraph::params` for why — briefly, it means
//! changing a value neither recompiles a shader nor forks a pipeline, so every
//! material sharing a graph shares one compiled program.
//!
//! The array's size is [`shadergraph::PARAM_SLOTS`] rather than a number written
//! out here, because the emitter writes that same constant into the shader's
//! `struct SgParams`. Two independently-maintained lengths for the same buffer
//! is a mismatch that surfaces as garbage in the last slot, not as an error.
//!
//! The bind group lives in `shadergraph::params::uniform_declaration` as the
//! `#{MATERIAL_BIND_GROUP}` placeholder — the same spelling Bevy's own
//! `pbr_bindings.wgsl` uses — so the emitted declaration follows the engine if
//! material bindings are ever renumbered again.

use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::render::mesh::MeshVertexBufferLayoutRef;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;
use shadergraph::{ParamLayout, SlotValue, Value, ValueType};

pub use params::GraphMaterialParams;

// Scoped purely to contain one lint. `ShaderType`'s derive emits a `check`
// function that only a manual `AsBindGroup` implementation would call, and
// ours is derived — so it is dead in a way no attribute on the struct or the
// field can reach. Everything the derive generates that *is* worth exercising
// is exercised: see `the_parameter_buffer_is_uniform_compatible` below.
mod params {
    #![allow(dead_code)]

    use bevy::prelude::*;
    use bevy::render::render_resource::ShaderType;
    use shadergraph::{ParamLayout, PARAM_SLOTS};

    use super::SetInputError;

    /// The uniform block every graph material binds: one `vec4` per slot.
    ///
    /// The Rust mirror of the `SgParams` struct the emitter declares; the two
    /// must agree on layout, which `assert_uniform_compat` pins in a test.
    #[derive(Clone, Copy, Debug, ShaderType, Reflect)]
    pub struct GraphMaterialParams {
        /// Slot values, indexed exactly as `ParamLayout` assigns them.
        pub slots: [Vec4; PARAM_SLOTS],
    }

    impl Default for GraphMaterialParams {
        fn default() -> Self {
            Self {
                slots: [Vec4::ZERO; PARAM_SLOTS],
            }
        }
    }

    impl GraphMaterialParams {
        /// Fill the slots from a compiled graph's layout.
        ///
        /// Slots past the end of the layout are zeroed rather than left alone. A
        /// graph that shrinks — a Param deleted, a branch unwired — would
        /// otherwise leave a stale value in a slot that a *later* edit could
        /// start reading again, producing a value from a node that no longer
        /// exists.
        pub fn write(&mut self, layout: &ParamLayout) {
            self.slots = [Vec4::ZERO; PARAM_SLOTS];
            for (slot, value) in layout.packed().into_iter().enumerate().take(PARAM_SLOTS) {
                self.slots[slot] = Vec4::from_array(value);
            }
        }

        /// Write one slot, padding the value the way the shader expects to read
        /// it. Refuses an index past the end of the buffer rather than
        /// panicking — see [`super::GraphMaterial::set_input`].
        pub fn set(&mut self, slot: usize, value: shadergraph::Value) -> Result<(), SetInputError> {
            match self.slots.get_mut(slot) {
                Some(entry) => {
                    *entry = Vec4::from_array(shadergraph::SlotValue::Param(value).packed());
                    Ok(())
                }
                // Unreachable via `set_input`, whose slot came from a layout the
                // emitter already bounded by PARAM_SLOTS. Reported rather than
                // asserted because a panic in a material update takes the frame
                // with it.
                None => Err(SetInputError::Unknown),
            }
        }
    }
}

// The texture fields below and the emitter's `texture_declaration` describe
// the same bindings from two sides, and only the compiler side can be derived
// from the constant — attribute arguments must be literals. This pins the
// other side: if the slot count moves, this file has to gain or lose a field
// pair, and the assert is what says so at compile time instead of letting the
// bind group layout silently disagree with the shader.
const _: () = assert!(
    shadergraph::TEXTURE_SLOTS == 4,
    "GraphMaterial declares one texture/sampler field pair per TEXTURE_SLOTS \
     slot; update the texture_N fields to match the new count"
);

/// A material whose fragment shader was compiled from a `shadergraph` graph.
#[derive(Asset, AsBindGroup, Clone, Debug, Reflect)]
#[bind_group_data(GraphMaterialKey)]
pub struct GraphMaterial {
    /// The parameter uniform, kept in sync with `layout` by
    /// [`GraphMaterial::set_input`] and [`GraphMaterialParams::write`].
    #[uniform(0)]
    pub params: GraphMaterialParams,

    /// The image bound to texture slot 0, if the graph samples it.
    ///
    /// One field per [`shadergraph::TEXTURE_SLOTS`] slot, at the bindings
    /// `shadergraph::params::texture_declaration` numbers: slot `k` is texture
    /// `1 + 2k`, sampler `2 + 2k`. `None` binds the renderer's fallback image
    /// — opaque white — so an unresolved slot renders as "no tint" rather
    /// than as a missing bind group and a vanished mesh. The sampler rides
    /// with the image asset, which is why each slot binds its own.
    ///
    /// Fill these from [`shadergraph::ParamLayout::textures`], by index, or
    /// through [`GraphMaterial::set_texture`] by exposed name.
    #[texture(1)]
    #[sampler(2)]
    pub texture_0: Option<Handle<Image>>,
    /// The image bound to texture slot 1. See [`GraphMaterial::texture_0`].
    #[texture(3)]
    #[sampler(4)]
    pub texture_1: Option<Handle<Image>>,
    /// The image bound to texture slot 2. See [`GraphMaterial::texture_0`].
    #[texture(5)]
    #[sampler(6)]
    pub texture_2: Option<Handle<Image>>,
    /// The image bound to texture slot 3. See [`GraphMaterial::texture_0`].
    #[texture(7)]
    #[sampler(8)]
    pub texture_3: Option<Handle<Image>>,

    /// The compiled shader.
    ///
    /// The pipeline key clones this handle, so the shader asset stays alive at
    /// least as long as the material and any specialized pipeline built from it.
    pub shader: Handle<Shader>,

    /// How the surface is drawn. See [`GraphMaterial::with_alpha_mode`].
    pub alpha_mode: AlphaMode,

    /// Whether to compile the anisotropic lighting path.
    ///
    /// Bevy gates anisotropy behind the `STANDARD_MATERIAL_ANISOTROPY` shader
    /// def, so setting `anisotropy_strength` in a generated shader does nothing
    /// unless the pipeline also defines it. Left off, a brushed metal renders
    /// as an ordinary isotropic one — no error, no warning, just a wrong-looking
    /// surface, which is a genuinely nasty thing to debug.
    ///
    /// Prefer [`crate::compile_graph`], which sets this from
    /// `Graph::uses_anisotropy()` rather than leaving it to be remembered.
    pub anisotropic: bool,

    /// Where this material's parameters live in the uniform.
    ///
    /// Carried on the material rather than handed back separately so the
    /// material is self-describing: anything holding one can drive it by name
    /// without also having kept the layout that came out of the same compile.
    /// A layout that has drifted from the shader it was produced with is a
    /// silent wrong-slot write, and the surest way to prevent that is to make
    /// the two impossible to separate.
    ///
    /// Ignored by reflection: it is compiler output, not something an inspector
    /// should offer to edit.
    #[reflect(ignore)]
    pub layout: ParamLayout,
}

/// Why driving a material input failed.
///
/// Distinguished rather than collapsed to `false` because the two call for
/// completely different fixes — one is a typo in a mesh's custom properties,
/// the other is a property authored as the wrong kind of thing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SetInputError {
    /// No parameter is exposed under that name.
    Unknown,
    /// The parameter exists but expects a different type.
    TypeMismatch {
        /// The type the exposed parameter actually has.
        expected: ValueType,
    },
}

impl std::fmt::Display for SetInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SetInputError::Unknown => f.write_str("no such material input"),
            SetInputError::TypeMismatch { expected } => {
                write!(f, "material input expects a {expected}")
            }
        }
    }
}

impl GraphMaterial {
    /// An opaque material rendering the given compiled graph.
    ///
    /// Parameters start zeroed; use [`GraphMaterialParams::write`] — or
    /// [`crate::compile_graph`], which does it — to fill them.
    pub fn opaque(shader: Handle<Shader>) -> Self {
        Self {
            params: GraphMaterialParams::default(),
            texture_0: None,
            texture_1: None,
            texture_2: None,
            texture_3: None,
            shader,
            alpha_mode: AlphaMode::Opaque,
            anisotropic: false,
            layout: ParamLayout::default(),
        }
    }

    /// Drive a named material input.
    ///
    /// This is the whole runtime interface for per-instance variation: a mesh
    /// authored with `wear_level = 0.7` finds the slot by name and writes it.
    /// Nothing recompiles and nothing forks a pipeline — two materials at
    /// different wear levels are the same program reading different bytes.
    ///
    /// Note what this does *not* do: it writes the material's own buffer, so
    /// callers wanting per-instance values need a material asset per instance.
    /// That is a clone of a struct and a uniform buffer, not of a shader.
    pub fn set_input(&mut self, name: &str, value: Value) -> Result<(), SetInputError> {
        let Some(expected) = self.layout.slot_type(name) else {
            return Err(SetInputError::Unknown);
        };
        if expected != value.ty() {
            return Err(SetInputError::TypeMismatch { expected });
        }
        let slot = self
            .layout
            .slot_of(name)
            .expect("slot_type only answers for names that resolve");

        // Written to the layout as well as the buffer, so the material's record
        // of its own values stays true. A later `params.write(&layout)` — which
        // is how a recompile refills the buffer — would otherwise quietly revert
        // every driven input to the graph's default.
        self.layout.set(slot, SlotValue::Param(value));
        self.params.set(slot, value)
    }

    /// Every input this material exposes, in sorted order. For diagnostics.
    pub fn inputs(&self) -> impl Iterator<Item = &str> {
        self.layout.exposed_names()
    }

    /// Rebind a named texture input.
    ///
    /// The texture half of [`GraphMaterial::set_input`]: a graph that exposed
    /// a `TexImage` as `"insignia"` gets its image swapped by name, with no
    /// recompile and no pipeline fork — the shader reads a slot index either
    /// way. `None` reverts the slot to the fallback white.
    ///
    /// Rebinding is a material-asset edit, so like every other field change it
    /// takes effect when the material is written back through `Assets`.
    pub fn set_texture(
        &mut self,
        name: &str,
        image: Option<Handle<Image>>,
    ) -> Result<(), SetInputError> {
        let Some(slot) = self.layout.texture_slot_of(name) else {
            return Err(SetInputError::Unknown);
        };
        self.set_texture_slot(slot, image)
    }

    /// The image currently bound to a texture slot, if the slot exists and an
    /// image is bound. The read half of [`GraphMaterial::set_texture_slot`],
    /// so a caller polling asset state can compare before writing — mutating
    /// a material asset is what re-prepares its bind group, and writing an
    /// unchanged value would pay that cost every frame.
    pub fn texture_slot(&self, slot: usize) -> Option<&Handle<Image>> {
        match slot {
            0 => self.texture_0.as_ref(),
            1 => self.texture_1.as_ref(),
            2 => self.texture_2.as_ref(),
            3 => self.texture_3.as_ref(),
            _ => None,
        }
    }

    /// Rebind a texture slot by index.
    ///
    /// The indexed form for callers walking [`shadergraph::ParamLayout::textures`]
    /// — which is every caller that resolves paths, since the layout hands
    /// slots back in index order. Refuses an out-of-range slot rather than
    /// panicking, for the same reason `set_input` does: the index came from
    /// somewhere, and a panic in a material update takes the frame with it.
    pub fn set_texture_slot(
        &mut self,
        slot: usize,
        image: Option<Handle<Image>>,
    ) -> Result<(), SetInputError> {
        let field = match slot {
            0 => &mut self.texture_0,
            1 => &mut self.texture_1,
            2 => &mut self.texture_2,
            3 => &mut self.texture_3,
            _ => return Err(SetInputError::Unknown),
        };
        *field = image;
        Ok(())
    }

    /// Enable the anisotropic lighting path. Required for any graph that drives
    /// `PbrOutput`'s anisotropy sockets.
    pub fn with_anisotropy(mut self) -> Self {
        self.anisotropic = true;
        self
    }

    /// Set how the surface is drawn — the one surface property that is not in
    /// the graph, because it configures the pipeline rather than the shading.
    pub fn with_alpha_mode(mut self, mode: AlphaMode) -> Self {
        self.alpha_mode = mode;
        self
    }
}

/// Per-instance data lifted into the pipeline key.
///
/// Both fields change which pipeline a material needs, so both must be here:
/// the shader because it *is* the fragment program, and `anisotropic` because
/// it changes the shader defs the program is compiled with.
///
/// The handle is strong, and `Handle`'s `Hash`/`Eq` resolve by asset id, so two
/// materials compiled to the same shader still share a pipeline. Not `Copy`,
/// because a strong handle is a reference count.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct GraphMaterialKey {
    shader: Handle<Shader>,
    anisotropic: bool,
}

impl From<&GraphMaterial> for GraphMaterialKey {
    fn from(material: &GraphMaterial) -> Self {
        Self {
            shader: material.shader.clone(),
            anisotropic: material.anisotropic,
        }
    }
}

impl Material for GraphMaterial {
    fn fragment_shader() -> ShaderRef {
        // Deliberately Default. There is no single shader for this type; the
        // real one is chosen per instance in `specialize` below.
        ShaderRef::Default
    }

    fn alpha_mode(&self) -> AlphaMode {
        self.alpha_mode
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let Some(fragment) = descriptor.fragment.as_mut() else {
            // Reached only in a pipeline configured with no fragment stage at
            // all (a depth-only pass, say). Nothing to override, and erroring
            // would break an otherwise valid pipeline.
            return Ok(());
        };

        // The key carries the strong handle (see `GraphMaterialKey`), so this
        // clone both selects the fragment program and keeps it alive for as
        // long as the specialized pipeline is cached.
        fragment.shader = key.bind_group_data.shader.clone();

        if key.bind_group_data.anisotropic {
            fragment
                .shader_defs
                .push("STANDARD_MATERIAL_ANISOTROPY".into());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::render::render_resource::ShaderType;

    /// The uniform layout is legal as a uniform binding.
    ///
    /// This is the check `ShaderType`'s derive generates and that nothing would
    /// otherwise call. It earns its place: std140 requires a 16-byte-aligned
    /// stride, and a parameter buffer whose alignment is subtly wrong does not
    /// fail — it reads neighbouring slots, so a color comes back holding part
    /// of a roughness. That is the exact failure the `vec4` array exists to
    /// prevent, and this asserts the prevention actually holds.
    #[test]
    fn the_parameter_buffer_is_uniform_compatible() {
        GraphMaterialParams::assert_uniform_compat();
    }

    /// A named texture reaches the field its slot index says it should.
    ///
    /// Exercised through a real compile so the name→slot lookup runs against a
    /// layout the emitter produced, not one assembled by hand to agree.
    #[test]
    fn textures_rebind_by_name_through_the_layout() {
        use shadergraph::{Graph, NodeKind};

        let mut g = Graph::new();
        let geometry = g.add(NodeKind::Geometry);
        let (surface, _out) = g.surface_output();
        let tex = g.add(NodeKind::TexImage {
            path: "insignia.png".to_string(),
            name: Some("insignia".to_string()),
        });
        g.link(geometry, "UV", tex, "Vector").unwrap();
        g.link(tex, "Color", surface, "Base Color").unwrap();

        let mut shaders = Assets::<Shader>::default();
        let mut material = crate::compile_graph(&g, "test", &mut shaders).unwrap();
        assert!(material.texture_0.is_none(), "slots start on the fallback");
        assert_eq!(material.layout.textures()[0].path, "insignia.png");

        let image = Handle::<Image>::default();
        material
            .set_texture("insignia", Some(image.clone()))
            .unwrap();
        assert_eq!(material.texture_0, Some(image));

        assert_eq!(
            material.set_texture("nonexistent", None),
            Err(SetInputError::Unknown)
        );
        assert_eq!(
            material.set_texture_slot(shadergraph::TEXTURE_SLOTS, None),
            Err(SetInputError::Unknown)
        );
    }

    #[test]
    fn writing_a_shorter_layout_clears_the_slots_it_no_longer_covers() {
        use shadergraph::{emit_with_layout, materials, Graph};

        let mut params = GraphMaterialParams::default();
        params.write(
            &emit_with_layout(&materials::hull_metal().unwrap())
                .unwrap()
                .1,
        );
        assert_ne!(params.slots[9], Vec4::ZERO, "hull_metal fills ten slots");

        // A graph with a single parameter. Every slot past the first must come
        // back zeroed, or a later edit could start reading a value left behind
        // by a node that no longer exists.
        let mut small = Graph::new();
        let color = small.param_color([1.0, 0.0, 0.0, 1.0]);
        let (surface, _out) = small.surface_output();
        small.link(color, "Result", surface, "Base Color").unwrap();

        params.write(&emit_with_layout(&small).unwrap().1);
        assert_eq!(params.slots[0], Vec4::new(1.0, 0.0, 0.0, 1.0));
        assert!(
            params.slots[1..].iter().all(|s| *s == Vec4::ZERO),
            "a stale value survived a graph that shrank"
        );
    }
}
