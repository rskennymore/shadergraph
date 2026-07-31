# shadergraph

A node graph that compiles to WGSL. Procedural materials are authored as a graph
of small nodes and emitted as shader source targeting a physically-based
renderer.

This project is a dev tool, there are rough edges, ux/ui is a work in progress.

The root crate is the compiler half only — no editor, no renderer, and **no
dependencies**.

```rust
use shadergraph::{emit, materials, GraphError};

fn main() -> Result<(), GraphError> {
    let graph = materials::hull_metal()?;
    let wgsl = emit(&graph)?;
    println!("{wgsl}");
    Ok(())
}
```

Or from the command line:

```sh
cargo run --example dump -- hull_metal
```

## The three crates

| Crate | What it is | Dependencies |
|---|---|---|
| `shadergraph` | graph → WGSL. The whole compiler | **none** (`naga` dev-only) |
| `shadergraph-bevy` | the `Material` that renders the output | `bevy` |
| `shadergraph-editor` | node canvas with a live lit preview | `bevy`, `bevy_egui`, `egui-snarl` |

```sh
cargo run -p shadergraph-editor
```

The split is not organisational tidiness. The compiler is the part worth
vendoring anywhere, and it has no business knowing a renderer exists; keeping its
dependency list empty is what makes it testable headlessly in milliseconds. The
engine binding is a separate crate because a consuming application and the
editor must render through the *same* material — a preview drawing through its
own copy is a preview of the preview.

`cargo test` at the workspace root builds only the compiler. The members are
opt-in with `-p`, so a game engine in the workspace does not slow the test loop.

### Features

- `serde` (off by default) — `Serialize`/`Deserialize` on the node vocabulary,
  so an editor can save a graph. A consumer that only emits WGSL should not pay
  for a serialisation format, so it is opt-in.

## Why it exists

glTF carries PBR scalars and texture references and nothing else, so a shader
graph authored in Blender evaporates at the export boundary. Blender can preview
a procedural material but can never ship one. Everywhere else in an asset
pipeline the answer is "make Blender the editor"; for materials it structurally
cannot be, and that is the whole justification for this crate.

## The port boundary

Blender's shader nodes split cleanly in two:

| Population | Identified by | Portable? |
|---|---|---|
| **Value nodes** | Float / Vector / Color in → out | **Yes.** Pure functions |
| **Shader nodes** | anything touching a `Shader` socket | **No.** Not functions at all |

A `Shader` socket is a *closure the renderer's integrator evaluates*, and two
renderers implement it completely differently. There is no WGSL translation of a
Principled BSDF, and pursuing one is how a project like this fails.

So the boundary is exactly "everything upstream of the first `Shader` socket",
and a graph terminates in this crate's own `PbrOutput` node, whose sockets are
the surface properties a PBR renderer consumes. The renderer keeps its lighting
model; the graph decides what the surface *is*.

## Node set

Sources and coordinates: `Geometry`, `Param`, `TexImage`, `Mapping`.
Patterns: `Voronoi`, `Noise` (fBm, live or unrolled), `Gradient`, `Fresnel`.
Maths and mixing: `Math`, `VectorMath`, `VectorToFloat`, `MixFloat`,
`MixVector`, `MixColor` (nine blend modes), `ColorRamp`, `Bump`.
Conversions: `SeparateXyz`/`CombineXyz`, `SeparateColor`/`CombineColor`,
`VectorToColor`/`ColorToVector`, `ColorToFloat`.
Structure and output: `Reroute`, `Surface`, `MixSurface`, `PbrOutput`.

Socket *names* mirror Blender's so muscle memory transfers, with three
deliberate deviations, all documented in `src/node.rs`: polymorphic nodes are
split per data type rather than hiding sockets behind a dropdown; `VectorMath`
is split by output type so every node is single-output; and nothing is implicit.

## Testing

```sh
cargo test
```

Three layers, none of which need a GPU:

- **Unit tests** over the IR — socket tables, type checking, cycle detection,
  dead-code elimination.
- **Golden files** (`tests/golden/`) over emitted WGSL. These check *stability*,
  not correctness: when a refactor should change nothing they prove it, and when
  it should change one line the diff shows one line. Update with
  `SHADERGRAPH_BLESS=1 cargo test`, then **read the diff before committing it**.
- **WGSL validation** (`tests/wgsl_validation.rs`) parses and type-checks every
  emitted shader with `naga`, against a stub of the Bevy items it imports. This
  catches the characteristic codegen failure — emitting something that is not
  valid in the target language — without launching an application.

The validation harness carries its own negative control, because a validation
test that cannot fail reads as coverage while providing none.

What none of this proves is that a shader *looks right*. Only a GPU and a mark 1 
eyeball can do that.

## What it does not do yet

See [`WISHLIST.md`](WISHLIST.md), which is ranked and honest about difficulty. The short version:
there is no normal-map node yet (image sampling exists; tangent-space normals are the hard half),
and the editor's save format has no version field.

## Licensing

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

The WGSL in `src/wgsl/` is written from published algorithm descriptions — each
file carries a provenance header naming its reference — specifically so the dual
licence stays honest. Copying from a single-licence source, **even a permissive
one**, would force the crate to that licence alone plus a `NOTICE` file, and
would mislead anyone who picked the MIT half.

Read `Cargo.toml` before pasting third-party shader code in here.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
