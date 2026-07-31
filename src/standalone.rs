//! Turning an emitted shader into something a plain WGSL parser will accept.
//!
//! Emitted shaders reference the renderer's PBR types through `#import`, and
//! guard vertex attributes with `#ifdef` — both `naga_oil` preprocessor syntax
//! rather than WGSL. Anything that wants to parse or type-check generated source
//! without booting the engine — the test suite, an editor validating a graph
//! before it swaps a shader in — has to stand those in for itself.
//!
//! This module is that stand-in, and it lives here rather than in either caller
//! because there is exactly one right answer and two places that need it. Kept
//! in `tests/`, the editor would have grown a second copy, and the copy that
//! drifts is always the one you are not looking at.
//!
//! # Validate twice
//!
//! A graph that reads UVs compiles to *two* different shaders depending on the
//! mesh it lands on, because `VERTEX_UVS_A` comes from the mesh's vertex layout.
//! Checking only the branch that happens to match the preview mesh is how a
//! material passes every test and then fails to resolve on the one hull that was
//! never unwrapped. So callers resolve with [`MESH_ATTRIBUTE_DEFS`](crate::standalone::MESH_ATTRIBUTE_DEFS)
//! and again with nothing defined, and both must pass.
//!
//! # What the stub is, and what it is not
//!
//! The declarations below are transcribed from `bevy_pbr` 0.16, and their field
//! names are pinned by the golden tests. They are still a stub: matching them
//! proves generated code is *well-formed WGSL using the expected surface*, not
//! that it is correct against the real engine. Structural drift after an engine
//! upgrade is caught here; semantic drift is not, and only rendering through
//! the real engine catches that.

/// Shader defs a mesh carrying both optional vertex attributes would push.
///
/// Transcribed from `bevy_pbr::render::mesh`, which pushes these from the mesh's
/// vertex buffer layout. `VERTEX_UVS` rides along with `VERTEX_UVS_A` there and
/// is included so the set matches what the engine really defines, even though
/// nothing generated reads it.
pub const MESH_ATTRIBUTE_DEFS: &[&str] = &["VERTEX_UVS", "VERTEX_UVS_A", "VERTEX_COLORS"];

/// Minimal stand-ins for the Bevy items a generated shader imports.
///
/// Transcribed from `bevy_pbr-0.19.0/src/render/{pbr_types,forward_io}.wgsl`
/// and `bevy_render-0.19.0/src/view/view.wgsl`. Only the fields the emitter
/// actually writes are declared — a field missing here that the emitter starts
/// using shows up as a resolution failure, which is exactly the alarm wanted.
///
/// The conditional fields mirror the engine's own `VertexOutput`, guards
/// included, so that resolving this stub with the same defs the engine would
/// push reproduces the same struct the engine would compile against.
pub const BEVY_STUB: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
#ifdef VERTEX_UVS_A
    @location(2) uv: vec2<f32>,
#endif
#ifdef VERTEX_COLORS
    @location(5) color: vec4<f32>,
#endif
}

struct View {
    clip_from_view: mat4x4<f32>,
    world_position: vec3<f32>,
}

@group(0) @binding(0) var<uniform> view: View;

struct StandardMaterial {
    base_color: vec4<f32>,
    emissive: vec4<f32>,
    reflectance: vec3<f32>,
    perceptual_roughness: f32,
    metallic: f32,
    flags: u32,
}

struct PbrInput {
    material: StandardMaterial,
    diffuse_occlusion: vec3<f32>,
    specular_occlusion: f32,
    frag_coord: vec4<f32>,
    world_position: vec4<f32>,
    world_normal: vec3<f32>,
    N: vec3<f32>,
    V: vec3<f32>,
    anisotropy_strength: f32,
    anisotropy_T: vec3<f32>,
    anisotropy_B: vec3<f32>,
    is_orthographic: bool,
    flags: u32,
}

fn pbr_input_new() -> PbrInput {
    var pbr_input: PbrInput;
    return pbr_input;
}

fn apply_pbr_lighting(in: PbrInput) -> vec4<f32> {
    return in.material.base_color;
}

fn main_pass_post_lighting_processing(pbr_input: PbrInput, color: vec4<f32>) -> vec4<f32> {
    return color;
}
"#;

/// Strip `naga_oil` preprocessor directives and splice in [`BEVY_STUB`].
///
/// `defined` is the set of shader defs to resolve `#ifdef` against — pass
/// [`MESH_ATTRIBUTE_DEFS`] for a fully-attributed mesh and `&[]` for a bare one,
/// and check both. See the module docs on validating twice.
///
/// The result is *not* what the engine compiles — it is the same generated body
/// against a stand-in prelude. Use it to check the body, never to ship.
///
/// Besides the `#ifdef` family, exactly one naga_oil substitution is understood:
/// `#{MATERIAL_BIND_GROUP}` becomes a literal group index, because the material
/// binding [`crate::params::uniform_declaration`] emits uses the placeholder the
/// engine's own shaders use. The number only has to be *a* valid group index for
/// naga to validate against; it is not required to match the engine's.
pub fn to_standalone_wgsl(generated: &str, defined: &[&str]) -> Result<String, String> {
    let body: String = generated
        .lines()
        .filter(|line| !line.trim_start().starts_with("#import"))
        .collect::<Vec<_>>()
        .join("\n")
        .replace("#{MATERIAL_BIND_GROUP}", "3");
    resolve_ifdefs(&format!("{BEVY_STUB}\n{body}"), defined)
}

/// Evaluate `#ifdef` / `#ifndef` / `#else` / `#endif`, dropping inactive lines.
///
/// A deliberately small subset of what `naga_oil` does — this is not a general
/// preprocessor and is not trying to be one. It handles exactly the directives
/// this crate emits, and everything else passes through untouched.
///
/// Nesting is supported because supporting it is a stack and not supporting it
/// is a wrong answer that looks like a right one. Unbalanced directives are an
/// error naming the line, since in generated source they can only mean a bug in
/// the emitter.
pub fn resolve_ifdefs(source: &str, defined: &[&str]) -> Result<String, String> {
    /// One `#ifdef` level: whether its current branch emits, and whether any
    /// branch has already been taken (so `#else` knows what to do).
    struct Level {
        active: bool,
        taken: bool,
    }

    let mut stack: Vec<Level> = Vec::new();
    let mut out: Vec<&str> = Vec::new();

    for (index, line) in source.lines().enumerate() {
        let number = index + 1;
        let trimmed = line.trim_start();

        // A nested block inside an inactive parent must still be tracked, or its
        // `#endif` would close the parent instead of itself.
        let emitting = stack.iter().all(|level| level.active);

        if let Some(name) = trimmed.strip_prefix("#ifdef ") {
            let held = defined.contains(&name.trim());
            stack.push(Level {
                active: held,
                taken: held,
            });
        } else if let Some(name) = trimmed.strip_prefix("#ifndef ") {
            let held = !defined.contains(&name.trim());
            stack.push(Level {
                active: held,
                taken: held,
            });
        } else if trimmed.starts_with("#else") {
            let level = stack
                .last_mut()
                .ok_or_else(|| format!("`#else` at line {number} with no open `#ifdef`"))?;
            level.active = !level.taken;
            level.taken = true;
        } else if trimmed.starts_with("#endif") {
            stack
                .pop()
                .ok_or_else(|| format!("`#endif` at line {number} with no open `#ifdef`"))?;
        } else if emitting {
            out.push(line);
        }
    }

    if !stack.is_empty() {
        return Err(format!(
            "{} `#ifdef` block(s) left unclosed at end of source",
            stack.len()
        ));
    }

    Ok(out.join("\n"))
}

/// Source with line numbers, for reporting an error against generated code.
///
/// A validation error cites a line number, and the source is generated — so
/// there is no file to open and count in. Printing the numbered source next to
/// the diagnostic is the difference between a usable error and a shrug.
pub fn numbered(source: &str) -> String {
    source
        .lines()
        .enumerate()
        .map(|(i, l)| format!("{:4} | {l}", i + 1))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_branches_are_reachable() {
        let source = "a\n#ifdef X\nb\n#else\nc\n#endif\nd";
        assert_eq!(resolve_ifdefs(source, &["X"]).unwrap(), "a\nb\nd");
        assert_eq!(resolve_ifdefs(source, &[]).unwrap(), "a\nc\nd");
    }

    #[test]
    fn ifndef_is_the_mirror_image() {
        let source = "#ifndef X\nabsent\n#else\npresent\n#endif";
        assert_eq!(resolve_ifdefs(source, &["X"]).unwrap(), "present");
        assert_eq!(resolve_ifdefs(source, &[]).unwrap(), "absent");
    }

    #[test]
    fn a_nested_block_does_not_escape_an_inactive_parent() {
        // The case a depth counter gets wrong: the inner `#endif` must close the
        // inner block, not the outer one, or everything after it leaks through.
        let source = "\
#ifdef OUTER
outer
#ifdef INNER
inner
#endif
still_outer
#endif
after";
        assert_eq!(resolve_ifdefs(source, &[]).unwrap(), "after");
        assert_eq!(
            resolve_ifdefs(source, &["OUTER"]).unwrap(),
            "outer\nstill_outer\nafter"
        );
        assert_eq!(
            resolve_ifdefs(source, &["OUTER", "INNER"]).unwrap(),
            "outer\ninner\nstill_outer\nafter"
        );
    }

    #[test]
    fn unbalanced_directives_are_reported_not_guessed() {
        assert!(resolve_ifdefs("#ifdef X\nbody", &[]).is_err());
        assert!(resolve_ifdefs("body\n#endif", &[]).is_err());
        assert!(resolve_ifdefs("#else", &[]).is_err());
    }

    #[test]
    fn the_stub_declares_the_optional_attributes_only_when_defined() {
        // Matched on the full declaration including its location, because
        // `color: vec4<f32>` alone also matches `StandardMaterial.base_color`.
        let with = to_standalone_wgsl("", MESH_ATTRIBUTE_DEFS).unwrap();
        assert!(with.contains("@location(2) uv: vec2<f32>"));
        assert!(with.contains("@location(5) color: vec4<f32>"));

        let without = to_standalone_wgsl("", &[]).unwrap();
        assert!(!without.contains("@location(2) uv: vec2<f32>"));
        assert!(!without.contains("@location(5) color: vec4<f32>"));
    }

    #[test]
    fn imports_are_stripped_before_conditionals_are_resolved() {
        let generated = "#import bevy_pbr::forward_io::VertexOutput\nfn f() {}";
        let source = to_standalone_wgsl(generated, &[]).unwrap();
        assert!(!source.contains("#import"));
        assert!(source.contains("fn f() {}"));
    }
}
