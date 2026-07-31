//! The lit 3D preview: a mesh, a light you can move, and a camera you can orbit.
//!
//! This is a *rig*, not a replica of any particular scene. A space game lights
//! a hull with a single distant star; that is a poor way to judge a material,
//! because half the surface is in shadow and the specular only ever arrives
//! from one direction. Being able to swing the light around and watch the
//! highlight move is the whole point — it is how you tell an anisotropic streak
//! from a smear in the albedo, and it is not something a fixed scene can show.

use bevy::asset::AssetId;
use bevy::camera::primitives::MeshAabb as _;
use bevy::camera::Viewport;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;

/// Which primitive the material is shown on.
///
/// A cube is the default because flat faces read a procedural pattern most
/// clearly, but it is a bad judge of anything view-dependent: with only six
/// normals in the whole mesh, a Fresnel term or an anisotropic highlight has
/// almost nothing to vary across. The curved options exist for that.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum PreviewMesh {
    #[default]
    Cube,
    Sphere,
    Torus,
    /// A mesh dragged in from a glTF file. Held separately from the primitives
    /// because it only exists once something has been loaded.
    Imported,
}

impl PreviewMesh {
    /// The primitives offered in the picker. `Imported` is absent because you
    /// cannot pick it — it appears by dropping a file.
    pub const ALL: &'static [PreviewMesh] =
        &[PreviewMesh::Cube, PreviewMesh::Sphere, PreviewMesh::Torus];

    pub fn label(self) -> &'static str {
        match self {
            PreviewMesh::Cube => "Cube",
            PreviewMesh::Sphere => "Sphere",
            PreviewMesh::Torus => "Torus",
            PreviewMesh::Imported => "Imported",
        }
    }

    fn build(self) -> Option<Mesh> {
        Some(match self {
            PreviewMesh::Cube => Cuboid::new(1.4, 1.4, 1.4).into(),
            PreviewMesh::Sphere => Sphere::new(0.9).mesh().uv(48, 24),
            PreviewMesh::Torus => Torus::new(0.35, 0.9).into(),
            // Not built from a shape; the handle arrives from the asset server.
            PreviewMesh::Imported => return None,
        })
    }
}

/// Everything the preview panel exposes as a control.
#[derive(Resource)]
pub struct PreviewSettings {
    pub mesh: PreviewMesh,
    /// Key light direction, as angles rather than a vector so a slider can
    /// drive it without ever producing a degenerate direction.
    pub light_yaw: f32,
    pub light_pitch: f32,
    /// Sweep the light continuously. The fastest way to see whether a highlight
    /// is anchored to the surface or painted on.
    pub orbit_light: bool,
    pub illuminance: f32,
    /// Stand-in for image-based lighting, which the editor has no cubemap for.
    /// A rough metal with no ambient term is very nearly black everywhere the
    /// key light does not directly hit, which looks like a broken shader.
    pub ambient: f32,
    /// EXPERIMENT: rewrite `dpdx`/`dpdy` to their `Fine` variants in the
    /// compiled shader, to A/B the quad-derivative bump artifact and its cost
    /// live on one graph. Delete once a winner is picked and baked into
    /// `wgsl/bump.wgsl`.
    pub fine_derivatives: bool,
}

impl Default for PreviewSettings {
    fn default() -> Self {
        Self {
            mesh: PreviewMesh::default(),
            light_yaw: 0.9,
            light_pitch: -0.6,
            orbit_light: false,
            illuminance: 6_000.0,
            ambient: 300.0,
            fine_derivatives: false,
        }
    }
}

/// Which optional vertex attributes the mesh currently on the stand carries.
///
/// Exists because the fallback for a missing attribute is *silent*: a graph
/// reading UVs on a mesh without them gets zero everywhere and renders a flat,
/// plausible-looking surface with nothing to say why. The engine cannot warn —
/// as far as it is concerned the shader compiled and ran. So the editor reports
/// it, which turns "why is this flat" into "oh, this glb was never unwrapped".
#[derive(Resource, Default, PartialEq, Eq)]
pub struct PreviewMeshAttributes {
    /// `None` while the mesh asset has not loaded. An imported glTF arrives a
    /// frame or more after the drop, and answering "no UVs" in the meantime
    /// would be a lie that flickers red on every import.
    pub uv: Option<bool>,
    pub vertex_color: Option<bool>,
}

/// Where the 3D view is allowed to draw, in physical pixels.
///
/// Written by the UI (which is the only thing that knows how much room the node
/// canvas took) and read by the camera. Kept as a resource rather than set
/// directly from the egui system because with multipass enabled that system
/// runs more than once per frame, and a camera should be assigned its viewport
/// exactly once.
#[derive(Resource, Default)]
pub struct PreviewViewport(pub Option<Viewport>);

/// Orbit state, in spherical coordinates about the origin.
#[derive(Resource)]
pub struct OrbitCamera {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    /// Zoom limits, sized to whatever is currently on the stand.
    ///
    /// Not constants, because the stand holds both a 1.4-unit cube and whatever
    /// glTF gets dropped on it — which can be tens of metres across. A fixed
    /// ceiling of 20 would mean a large model could not be backed away from far
    /// enough to see, which looks exactly like the mesh having failed to load.
    pub min_distance: f32,
    pub max_distance: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            yaw: 0.6,
            pitch: 0.35,
            distance: 4.0,
            min_distance: 1.2,
            max_distance: 20.0,
        }
    }
}

impl OrbitCamera {
    /// Back off far enough to see a subject of this radius, and re-scale the
    /// zoom limits around it.
    ///
    /// The distance is `radius / sin(fov/2)`, the standard fit for a bounding
    /// sphere, with a little margin so the silhouette is not flush against the
    /// edge of the viewport.
    fn fit(&mut self, radius: f32) {
        // Bevy's default perspective vertical FOV. Read rather than assumed
        // would be better; it is a constant here because the preview camera is
        // spawned with the default and nothing changes it.
        const HALF_FOV: f32 = std::f32::consts::FRAC_PI_8;
        // A degenerate mesh — a single point, or one whose positions are all
        // equal — would otherwise put the camera exactly on top of it.
        let radius = radius.max(1e-3);

        self.distance = radius / HALF_FOV.sin() * 1.15;
        self.min_distance = radius * 0.15;
        self.max_distance = radius * 12.0;
    }
}

#[derive(Component)]
pub struct PreviewCamera;

/// The mesh the material under construction is applied to.
#[derive(Component)]
pub struct PreviewSubject;

#[derive(Component)]
struct KeyLight;

pub struct PreviewPlugin;

impl Plugin for PreviewPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PreviewSettings>()
            .init_resource::<PreviewViewport>()
            .init_resource::<PreviewMeshAttributes>()
            .init_resource::<OrbitCamera>()
            .add_systems(Startup, spawn_preview)
            .add_systems(
                Update,
                (
                    rebuild_mesh_on_change,
                    receive_dropped_files,
                    read_mesh_attributes,
                    frame_subject_on_load,
                    drive_light,
                    orbit_camera_input,
                    apply_viewport,
                ),
            );
    }
}

fn spawn_preview(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    settings: Res<PreviewSettings>,
    mut egui_settings: ResMut<bevy_egui::EguiGlobalSettings>,
) {
    // bevy_egui attaches its primary context to a camera, and by default it
    // creates one on whatever camera renders the window — which here is the
    // preview camera. That is a feedback loop: the UI fits the preview camera's
    // viewport to the hole the panels leave, and egui would then lay the panels
    // out *inside that hole*, shrinking it every frame until the layout wraps.
    // So the automatic context is disabled and egui gets its own full-window
    // overlay camera that renders nothing but the UI.
    egui_settings.auto_create_primary_context = false;
    commands.spawn((
        bevy_egui::PrimaryEguiContext,
        Camera2d,
        // Nothing from the world: this camera exists only to give egui a
        // full-window surface to draw on, above the preview.
        bevy::camera::visibility::RenderLayers::none(),
        Camera {
            order: 1,
            output_mode: bevy::camera::CameraOutputMode::Write {
                blend_state: Some(bevy::render::render_resource::BlendState::ALPHA_BLENDING),
                clear_color: ClearColorConfig::None,
            },
            clear_color: ClearColorConfig::Custom(Color::NONE),
            ..default()
        },
    ));

    commands.spawn((
        PreviewCamera,
        Camera3d::default(),
        Camera {
            // Left unset for the first frame. The UI has not run yet, so there
            // is no rect to fit to; `apply_viewport` fills it in as soon as
            // there is one.
            viewport: None,
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.spawn((
        KeyLight,
        DirectionalLight {
            illuminance: settings.illuminance,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::default(),
    ));

    commands.spawn((
        PreviewSubject,
        Mesh3d(
            meshes.add(
                settings
                    .mesh
                    .build()
                    .unwrap_or_else(|| Cuboid::default().into()),
            ),
        ),
        Transform::default(),
    ));
}

/// Swap the mesh when the picker changes.
///
/// Rebuilds rather than caching all three: a primitive mesh is a few hundred
/// vertices, and holding three handles alive to avoid a build that costs
/// microseconds would be optimising the wrong thing.
///
/// # Why this tracks the value instead of asking `is_changed`
///
/// NOTE: `PreviewSettings.is_changed()` is true on **every frame**, and not because
/// anything changed: the toolbar binds `&mut settings.light_yaw` and friends
/// straight into `DragValue`s, and taking that mutable borrow sets the flag
/// whether or not the widget was touched. Bevy's change detection is per
/// *resource*, so six unrelated fields sharing one struct means the flag can
/// only ever say "somebody looked at this", which is not the question being
/// asked here.
///
/// Relying on it would rebuild the mesh every frame, minting a fresh `AssetId`
/// each time — and because [`frame_subject_on_load`] re-frames on every new id,
/// that presents as the preview snapping back the instant the camera is moved.
/// Tracking the value means a rebuild happens only when the mesh choice does.
fn rebuild_mesh_on_change(
    settings: Res<PreviewSettings>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut subject: Query<&mut Mesh3d, With<PreviewSubject>>,
    mut current: Local<Option<PreviewMesh>>,
) {
    if *current == Some(settings.mesh) {
        return;
    }
    let Some(built) = settings.mesh.build() else {
        // An imported mesh's handle is owned by the asset server and installed
        // by `receive_dropped_files`; there is nothing to rebuild. Recorded all
        // the same, so this does not retry every frame.
        *current = Some(settings.mesh);
        return;
    };
    *current = Some(settings.mesh);
    for mut mesh in &mut subject {
        mesh.0 = meshes.add(built.clone());
    }
}

/// Load a glTF mesh dropped onto the window.
///
/// Drag and drop rather than a file dialog or a path field: it needs no
/// dependency, and "show me this material on that part" is a gesture rather than
/// a path someone wants to type.
///
/// Only the first primitive of the first mesh is taken. A whole glTF scene would
/// be a different feature — this is a preview stand, not a scene viewer, and a
/// material is judged on one surface at a time.
fn receive_dropped_files(
    mut dropped: MessageReader<FileDragAndDrop>,
    assets: Res<AssetServer>,
    mut settings: ResMut<PreviewSettings>,
    mut subject: Query<&mut Mesh3d, With<PreviewSubject>>,
    mut document: ResMut<crate::canvas::GraphDocument>,
) {
    for event in dropped.read() {
        let FileDragAndDrop::DroppedFile { path_buf, .. } = event else {
            continue;
        };

        let extension = path_buf
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase);

        // An image drop is a *path edit*, nothing more: the path lands on a
        // Tex Image node and the compile loop's texture binding does what it
        // would do had the path been typed. Preferring a node with an empty
        // path makes the obvious gesture work — place the node, drop the file
        // — without stealing the image from a node that already has one.
        if matches!(
            extension.as_deref(),
            Some("png" | "jpg" | "jpeg" | "ktx2" | "dds" | "tga" | "exr")
        ) {
            let Some(dropped_path) = path_buf.to_str() else {
                warn!("ignoring dropped image with a non-UTF-8 path");
                continue;
            };

            let mut first = None;
            let mut empty = None;
            for (id, kind) in document.snarl.node_ids() {
                if let shadergraph::NodeKind::TexImage { path, .. } = kind {
                    first.get_or_insert(id);
                    if path.is_empty() {
                        empty.get_or_insert(id);
                    }
                }
            }
            match empty.or(first) {
                Some(id) => {
                    if let Some(shadergraph::NodeKind::TexImage { path, .. }) =
                        document.snarl.get_node_mut(id)
                    {
                        info!(
                            "assigning dropped image {} to a Tex Image node",
                            dropped_path
                        );
                        *path = dropped_path.to_string();
                    }
                }
                None => warn!(
                    "dropped image {} but the graph has no Tex Image node to give it to",
                    path_buf.display()
                ),
            }
            continue;
        }

        if !matches!(extension.as_deref(), Some("gltf" | "glb")) {
            warn!(
                "ignoring dropped file {}: meshes are .gltf/.glb, images are \
                 .png/.jpg/.ktx2/.dds/.tga/.exr",
                path_buf.display()
            );
            continue;
        }

        // Routed through the `abs` asset source registered in `main`, because
        // the asset server resolves paths against its own root and a dropped
        // file is an absolute path from anywhere on disk.
        let Some(path) = path_buf.to_str() else {
            warn!("ignoring dropped file with a non-UTF-8 path");
            continue;
        };
        let label = format!("abs://{}#Mesh0/Primitive0", path.trim_start_matches('/'));
        info!("loading preview mesh from {}", path_buf.display());

        let handle: Handle<Mesh> = assets.load(&label);
        for mut mesh in &mut subject {
            mesh.0 = handle.clone();
        }
        // Assigned through the bypass so the change does not retrigger
        // `rebuild_mesh_on_change`, which would immediately replace the imported
        // mesh with a primitive.
        settings.bypass_change_detection().mesh = PreviewMesh::Imported;
    }
}

/// Report what the mesh on the stand actually carries.
///
/// Polled rather than event-driven because there are three ways the answer
/// changes — the picker, a dropped file, and that file finishing loading — and
/// only the first two are events. Two hash lookups a frame is not worth the
/// bookkeeping of getting the third one right.
fn read_mesh_attributes(
    meshes: Res<Assets<Mesh>>,
    subject: Query<&Mesh3d, With<PreviewSubject>>,
    mut attributes: ResMut<PreviewMeshAttributes>,
) {
    let loaded = subject.single().ok().and_then(|mesh| meshes.get(&mesh.0));
    let found = match loaded {
        Some(mesh) => PreviewMeshAttributes {
            uv: Some(mesh.attribute(Mesh::ATTRIBUTE_UV_0).is_some()),
            vertex_color: Some(mesh.attribute(Mesh::ATTRIBUTE_COLOR).is_some()),
        },
        None => PreviewMeshAttributes::default(),
    };

    // Assigned only on a real change, so the resource's change flag stays
    // meaningful for anything that later wants to watch it.
    if *attributes != found {
        *attributes = found;
    }
}

/// Centre and frame whatever mesh is on the stand, once it has loaded.
///
/// # Why this is not optional
///
/// The primitives are built at the origin and are about a unit across, so the
/// default camera framed them by construction and nothing needed doing. A
/// dropped glTF has neither property, and both failures look identical from the
/// outside — **an empty viewport**:
///
/// - **Off-centre.** The orbit camera looks at the world origin. A part authored
///   forty metres from the origin in Blender keeps that offset, so it sits far
///   outside the view no matter how the camera is turned. Fixed by moving the
///   subject rather than the camera, which keeps the look-at-origin assumption
///   the orbit maths is built on.
/// - **Off-scale.** A dropped model can be tens of metres across while the
///   camera sat at four with a hard ceiling of twenty. Even backed all the way
///   off you would be inside it.
///
/// Polled rather than event-driven for the same reason `read_mesh_attributes`
/// is: the mesh becoming *available* is not an event. The drop is, but the asset
/// arrives some frames later, and the AABB cannot be computed until it does.
/// Keyed on `AssetId` so it fires once per mesh and does not fight the user's
/// scrolling afterwards.
fn frame_subject_on_load(
    meshes: Res<Assets<Mesh>>,
    mut subject: Query<(&Mesh3d, &mut Transform), With<PreviewSubject>>,
    mut orbit: ResMut<OrbitCamera>,
    mut framed: Local<Option<AssetId<Mesh>>>,
) {
    let Ok((mesh3d, mut transform)) = subject.single_mut() else {
        return;
    };

    let id = mesh3d.0.id();
    if *framed == Some(id) {
        return;
    }
    // Still loading. Deliberately does not record `id`, so this runs again on
    // the frame the asset lands.
    let Some(mesh) = meshes.get(&mesh3d.0) else {
        return;
    };
    let Some(aabb) = mesh.compute_aabb() else {
        // No positions to measure. Nothing sensible to frame, and retrying every
        // frame would be pointless, so treat it as handled.
        *framed = Some(id);
        return;
    };

    *framed = Some(id);
    transform.translation = -Vec3::from(aabb.center);
    orbit.fit(Vec3::from(aabb.half_extents).length());
}

fn drive_light(
    time: Res<Time>,
    mut settings: ResMut<PreviewSettings>,
    mut light: Query<(&mut Transform, &mut DirectionalLight), With<KeyLight>>,
    mut ambient: ResMut<GlobalAmbientLight>,
) {
    if settings.orbit_light {
        // `bypass_change_detection` so a sweeping light does not mark the
        // settings changed every frame — `rebuild_mesh_on_change` watches the
        // same resource and would otherwise rebuild the mesh 60 times a second.
        let dt = time.delta_secs();
        let yaw = &mut settings.bypass_change_detection().light_yaw;
        // Wrapped rather than left to accumulate. An angle is periodic, so an
        // unbounded one is only ever a liability: left running it reaches
        // thousands of radians, where the number in the UI is meaningless, the
        // field it sits in keeps growing, and `sin_cos` is being handed a value
        // whose useful precision has been eaten by its own magnitude.
        *yaw = (*yaw + dt * 0.6).rem_euclid(std::f32::consts::TAU);
    }

    let (yaw, pitch, illuminance) = (
        settings.light_yaw,
        settings.light_pitch,
        settings.illuminance,
    );

    for (mut transform, mut directional) in &mut light {
        // A directional light shines down its local -Z, so composing yaw then
        // pitch aims it without ever needing a look-at target.
        transform.rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);
        directional.illuminance = illuminance;
    }

    ambient.brightness = settings.ambient;
}

/// Drag to orbit, wheel to zoom — but only while the pointer is over the 3D
/// view.
///
/// Gating on the viewport rect rather than on egui's "wants pointer input" is
/// deliberate: dragging a wire in the canvas and dragging the camera are both
/// left-button drags, and the rect is an unambiguous answer to which one the
/// author meant.
fn orbit_camera_input(
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion: MessageReader<MouseMotion>,
    mut wheel: MessageReader<MouseWheel>,
    windows: Query<&Window>,
    viewport: Res<PreviewViewport>,
    mut orbit: ResMut<OrbitCamera>,
    mut camera: Query<&mut Transform, With<PreviewCamera>>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(rect) = viewport.0.as_ref() else {
        motion.clear();
        wheel.clear();
        return;
    };

    let over_preview = window
        .cursor_position()
        // `cursor_position` is in logical points; the viewport is in physical
        // pixels. Mixing the two silently misplaces the boundary on any display
        // that is not at 100% scaling.
        .map(|p| p * window.scale_factor())
        .is_some_and(|p| {
            let min = rect.physical_position.as_vec2();
            let max = min + rect.physical_size.as_vec2();
            p.x >= min.x && p.x < max.x && p.y >= min.y && p.y < max.y
        });

    if !over_preview {
        motion.clear();
        wheel.clear();
        return;
    }

    if buttons.pressed(MouseButton::Left) {
        let delta: Vec2 = motion.read().map(|m| m.delta).sum();
        orbit.yaw -= delta.x * 0.005;
        orbit.pitch = (orbit.pitch + delta.y * 0.005)
            // Stop just short of the poles: at exactly ±90° the up vector and
            // the view direction are parallel and `looking_at` has no answer.
            .clamp(-1.54, 1.54);
    } else {
        motion.clear();
    }

    let scroll: f32 = wheel.read().map(|w| w.y).sum();
    if scroll != 0.0 {
        let (near, far) = (orbit.min_distance, orbit.max_distance);
        orbit.distance = (orbit.distance * (1.0 - scroll * 0.1)).clamp(near, far);
    }

    let (sin_yaw, cos_yaw) = orbit.yaw.sin_cos();
    let (sin_pitch, cos_pitch) = orbit.pitch.sin_cos();
    let eye = Vec3::new(
        orbit.distance * cos_pitch * sin_yaw,
        orbit.distance * sin_pitch,
        orbit.distance * cos_pitch * cos_yaw,
    );

    for mut transform in &mut camera {
        *transform = Transform::from_translation(eye).looking_at(Vec3::ZERO, Vec3::Y);
    }
}

fn apply_viewport(
    viewport: Res<PreviewViewport>,
    mut camera: Query<&mut Camera, With<PreviewCamera>>,
) {
    if !viewport.is_changed() {
        return;
    }
    for mut cam in &mut camera {
        cam.viewport = viewport.0.clone();
    }
}

/// Fit a viewport to a rectangle given in physical pixels, or `None` if the
/// rectangle is degenerate.
///
/// Clamping is not defensive tidiness: a viewport that runs past the surface,
/// or has a zero dimension, is a wgpu validation error rather than a visual
/// glitch. Dragging the splitter to the very top or bottom of the window is an
/// entirely ordinary thing to do, so this has to be handled and not asserted.
pub fn viewport_from_physical(min: Vec2, max: Vec2, surface: UVec2) -> Option<Viewport> {
    let min = min.max(Vec2::ZERO).as_uvec2().min(surface);
    let max = max.max(Vec2::ZERO).as_uvec2().min(surface);

    if max.x <= min.x || max.y <= min.y {
        return None;
    }

    Some(Viewport {
        physical_position: min,
        physical_size: max - min,
        depth: 0.0..1.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SURFACE: UVec2 = UVec2::new(1600, 1000);

    #[test]
    fn an_ordinary_split_keeps_its_rectangle() {
        let v = viewport_from_physical(Vec2::new(0.0, 40.0), Vec2::new(1600.0, 500.0), SURFACE)
            .expect("a non-degenerate rect");
        assert_eq!(v.physical_position, UVec2::new(0, 40));
        assert_eq!(v.physical_size, UVec2::new(1600, 460));
    }

    /// Dragging the splitter to the very top is ordinary, and a zero-height
    /// viewport is a wgpu validation error rather than a blank rectangle. It has
    /// to come back as "no viewport", not as a viewport of nothing.
    #[test]
    fn a_collapsed_split_yields_no_viewport() {
        assert!(
            viewport_from_physical(Vec2::new(0.0, 500.0), Vec2::new(1600.0, 500.0), SURFACE)
                .is_none()
        );
        assert!(
            viewport_from_physical(Vec2::new(0.0, 600.0), Vec2::new(1600.0, 400.0), SURFACE)
                .is_none()
        );
    }

    /// Framing has to put the camera outside the subject, at any scale.
    ///
    /// The failure this guards against is not subtle arithmetic — it is the
    /// camera ending up *inside* a large mesh, which renders as either nothing
    /// or an inside-out shell, and reads as "the import did not work".
    #[test]
    fn fitting_backs_off_far_enough_to_see_the_whole_subject() {
        for radius in [0.5_f32, 1.0, 12.0, 40.0, 250.0] {
            let mut orbit = OrbitCamera::default();
            orbit.fit(radius);

            assert!(
                orbit.distance > radius,
                "camera ended up inside a subject of radius {radius}"
            );
            // The zoom range has to contain where we just put the camera, or the
            // first scroll would snap the subject back out of frame.
            assert!(
                orbit.distance >= orbit.min_distance && orbit.distance <= orbit.max_distance,
                "framed distance {} is outside its own zoom limits {}..={}",
                orbit.distance,
                orbit.min_distance,
                orbit.max_distance
            );
        }
    }

    /// A mesh with no extent must not put the camera on top of it.
    ///
    /// `compute_aabb` succeeds for a mesh whose vertices are all at one point,
    /// so this is reachable from a real file rather than only in theory.
    #[test]
    fn fitting_a_degenerate_subject_still_leaves_a_camera_somewhere_usable() {
        let mut orbit = OrbitCamera::default();
        orbit.fit(0.0);
        assert!(orbit.distance > 0.0 && orbit.distance.is_finite());
        assert!(orbit.max_distance > orbit.min_distance);
    }

    /// egui reports its rect in points and can round a fraction past the window
    /// edge; the result must still fit inside the surface.
    #[test]
    fn an_oversized_rect_is_clipped_to_the_surface() {
        let v = viewport_from_physical(Vec2::new(-8.0, -8.0), Vec2::new(2000.0, 1400.0), SURFACE)
            .expect("a non-degenerate rect");
        assert_eq!(v.physical_position, UVec2::ZERO);
        assert_eq!(v.physical_size, SURFACE);
    }
}
