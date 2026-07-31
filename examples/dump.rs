//! Print a material's compiled WGSL to stdout.
//!
//! ```sh
//! cargo run --example dump -- hull_metal
//! ```
//!
//! Useful for eyeballing the output of a change without going near a GPU, and
//! for pasting into a WGSL validator when a shader misbehaves at runtime.

use shadergraph::{emit, materials, Graph, GraphError};

fn build(name: &str) -> Option<Result<Graph, GraphError>> {
    Some(match name {
        "hull_metal" => materials::hull_metal(),
        "hull_dark" => materials::hull_dark(),
        "copper_accent" => materials::copper_accent(),
        "glass" => materials::glass([0.1, 0.12, 0.2, 1.0], 0.3),
        _ => return None,
    })
}

const KNOWN: &[&str] = &["hull_metal", "hull_dark", "copper_accent", "glass"];

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!(
            "usage: dump <material>\nknown materials: {}",
            KNOWN.join(", ")
        );
        std::process::exit(2);
    });

    let Some(graph) = build(&name) else {
        eprintln!(
            "unknown material `{name}`\nknown materials: {}",
            KNOWN.join(", ")
        );
        std::process::exit(2);
    };

    let graph = graph.unwrap_or_else(|e| {
        eprintln!("material `{name}` failed to build: {e}");
        std::process::exit(1);
    });

    match emit(&graph) {
        Ok(wgsl) => print!("{wgsl}"),
        Err(e) => {
            eprintln!("material `{name}` failed to compile: {e}");
            std::process::exit(1);
        }
    }
}
