//! The graph itself: nodes, wires, validation, and evaluation order.

use std::collections::BTreeMap;
use std::fmt;

use crate::node::{
    NodeKind, SocketDef, SocketDefault, GEOMETRY_UV_OUTPUT, GEOMETRY_VERTEX_COLOR_OUTPUT,
};
use crate::value::{Value, ValueType};

/// Handle to a node. Indexes into [`Graph`]'s node list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub(crate) usize);

impl NodeId {
    /// Position in the graph's node list. Used to build stable binding names.
    pub fn index(self) -> usize {
        self.0
    }
}

/// Which end of a node a socket lookup refers to. Only used to make error
/// messages say "input"/"output" rather than leaving the reader to guess.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Direction {
    Input,
    Output,
}

impl fmt::Display for Direction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Direction::Input => "input",
            Direction::Output => "output",
        })
    }
}

/// Everything that can be wrong with a graph.
///
/// Every variant names the node kind and socket involved. A graph error that
/// says only "type mismatch" leaves the author diffing a node list by hand.
#[derive(Clone, Debug, PartialEq)]
pub enum GraphError {
    /// A wire named a socket the node does not have.
    UnknownSocket {
        /// Kind of the node whose socket was looked up.
        node: &'static str,
        /// The socket name that failed to resolve.
        socket: String,
        /// `"input"` or `"output"` — which table was searched.
        direction: &'static str,
        /// The socket names that *do* exist, pre-joined for the message.
        available: String,
    },
    /// A wire connects two sockets of different [`ValueType`]s.
    TypeMismatch {
        /// Kind of the node the wire leaves.
        from_node: &'static str,
        /// Output socket the wire leaves from.
        from_socket: &'static str,
        /// Type that output carries.
        from_ty: ValueType,
        /// Kind of the node the wire enters.
        to_node: &'static str,
        /// Input socket the wire lands on.
        to_socket: &'static str,
        /// Type that input expects.
        to_ty: ValueType,
    },
    /// A second wire was aimed at an input that already has one.
    SocketOccupied {
        /// Kind of the node holding the input.
        node: &'static str,
        /// The input socket that is already wired.
        socket: &'static str,
    },
    /// An input with no default was left unwired.
    RequiredLinkMissing {
        /// Kind of the node holding the input.
        node: &'static str,
        /// The socket that must be wired.
        socket: &'static str,
    },
    /// The graph has no terminal node, so there is nothing to compile toward.
    NoOutputNode,
    /// The graph has more than one terminal node.
    MultipleOutputNodes {
        /// How many were found; exactly one is required.
        count: usize,
    },
    /// The graph is not acyclic.
    Cycle {
        /// Kind of a node on the cycle, as a starting point for untangling it.
        node: &'static str,
    },
    /// The graph wants more uniform slots than [`PARAM_SLOTS`](crate::PARAM_SLOTS) provides.
    TooManyParams {
        /// Slots the graph asked for. Reported alongside the limit because
        /// "over by one" and "over by ninety" call for different fixes.
        needed: usize,
        /// The fixed slot budget the request exceeded.
        limit: usize,
    },
    /// The graph samples more textures than [`TEXTURE_SLOTS`](crate::TEXTURE_SLOTS) allows.
    TooManyTextures {
        /// Texture slots the graph asked for — one per reachable `TexImage`.
        needed: usize,
        /// The fixed slot budget the request exceeded.
        limit: usize,
    },
    /// Two reachable `Param` nodes exposed under the same name.
    DuplicateParamName {
        /// The name both parameters claimed.
        name: String,
    },
    /// Two reachable `TexImage` nodes exposed under the same name.
    ///
    /// A namespace of its own: a texture sharing a name with a *param* is
    /// fine, because the two are driven through different calls.
    DuplicateTextureName {
        /// The name both textures claimed.
        name: String,
    },
    /// [`Graph::expose`] was pointed at something that is not a `Param`.
    NotAParam {
        /// Kind of the node that was wrongly nominated.
        node: &'static str,
    },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::UnknownSocket {
                node,
                socket,
                direction,
                available,
            } => write!(
                f,
                "node `{node}` has no {direction} socket `{socket}` (available: {available})"
            ),
            GraphError::TypeMismatch {
                from_node,
                from_socket,
                from_ty,
                to_node,
                to_socket,
                to_ty,
            } => write!(
                f,
                "cannot wire `{from_node}.{from_socket}` ({from_ty}) into \
                 `{to_node}.{to_socket}` ({to_ty}): types differ"
            ),
            GraphError::SocketOccupied { node, socket } => write!(
                f,
                "input socket `{node}.{socket}` already has a wire; \
                 an input accepts at most one"
            ),
            GraphError::RequiredLinkMissing { node, socket } => write!(
                f,
                "input socket `{node}.{socket}` has no default and must be wired"
            ),
            GraphError::NoOutputNode => {
                write!(
                    f,
                    "graph has no PbrOutput node, so there is nothing to compile toward"
                )
            }
            GraphError::MultipleOutputNodes { count } => write!(
                f,
                "graph has {count} PbrOutput nodes; exactly one is required"
            ),
            GraphError::Cycle { node } => write!(
                f,
                "graph contains a cycle reaching node `{node}`; \
                 a shader graph must be acyclic"
            ),
            GraphError::TooManyParams { needed, limit } => write!(
                f,
                "graph needs {needed} uniform slots but only {limit} exist \
                 (`shadergraph::params::PARAM_SLOTS`); a Param takes one slot and \
                 a color ramp takes seven per segment, so remove a ramp stop, \
                 reuse a Param across sockets, or raise PARAM_SLOTS"
            ),
            GraphError::TooManyTextures { needed, limit } => write!(
                f,
                "graph samples {needed} textures but only {limit} slots exist \
                 (`shadergraph::params::TEXTURE_SLOTS`); each TexImage takes one \
                 slot, so feed several reads from one node where the image is \
                 shared, or raise TEXTURE_SLOTS together with the renderer's \
                 material bindings"
            ),
            GraphError::DuplicateParamName { name } => write!(
                f,
                "two Param nodes are both exposed as `{name}`; a material input \
                 has to name exactly one slot, so rename one of them"
            ),
            GraphError::DuplicateTextureName { name } => write!(
                f,
                "two TexImage nodes are both exposed as `{name}`; a texture \
                 input has to name exactly one slot, so rename one of them"
            ),
            GraphError::NotAParam { node } => write!(
                f,
                "only a Param can be exposed as a material input, and this is a `{node}`"
            ),
        }
    }
}

impl std::error::Error for GraphError {}

/// A directed acyclic graph of shading nodes terminating in one `PbrOutput`.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    nodes: Vec<NodeKind>,
    /// Destination `(node, input socket index)` to source `(node, output socket
    /// index)`.
    ///
    /// Keyed by destination because an input accepts at most one wire while an
    /// output may feed many — so this direction makes the "already occupied"
    /// check a single lookup, and resolving a node's inputs during emission a
    /// direct one too.
    ///
    /// A `BTreeMap` rather than a `HashMap`: emission order must be identical
    /// across runs for golden-file tests to mean anything, and a hash map's
    /// iteration order is not.
    links: BTreeMap<(usize, usize), (usize, usize)>,
}

impl Graph {
    /// An empty graph. Equivalent to `Graph::default()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a node and return the id that names it from now on.
    pub fn add(&mut self, kind: NodeKind) -> NodeId {
        self.nodes.push(kind);
        NodeId(self.nodes.len() - 1)
    }

    /// Add a constant node. Convenience for the common case.
    pub fn param_float(&mut self, v: f32) -> NodeId {
        self.add(NodeKind::param(Value::Float(v)))
    }

    /// Add a constant vector node. See [`Graph::param_float`].
    pub fn param_vec3(&mut self, v: [f32; 3]) -> NodeId {
        self.add(NodeKind::param(Value::Vec3(v)))
    }

    /// Add a constant color node. See [`Graph::param_float`].
    pub fn param_color(&mut self, v: [f32; 4]) -> NodeId {
        self.add(NodeKind::param(Value::Color(v)))
    }

    /// Expose a `Param` under a name, making it part of the material's
    /// interface — a value something outside the graph can drive, with whatever
    /// the node already holds as the default.
    ///
    /// The one mutation this otherwise append-only IR allows, and it is allowed
    /// because it changes no wiring, no ids and nothing the shader reads. An
    /// editor still rebuilds rather than calling this; it exists so a graph
    /// written in Rust can declare its interface at the point the parameter is
    /// created rather than through a parallel table.
    ///
    /// Fails if `param` is not a `Param` node, because silently doing nothing
    /// would leave a material whose input never arrives and no way to find out
    /// why.
    pub fn expose(&mut self, param: NodeId, name: impl Into<String>) -> Result<(), GraphError> {
        match &mut self.nodes[param.0] {
            NodeKind::Param { name: slot, .. } => {
                *slot = Some(name.into());
                Ok(())
            }
            other => Err(GraphError::NotAParam { node: other.name() }),
        }
    }

    /// Add a `Surface` feeding a `PbrOutput`, and return both.
    ///
    /// Every graph needs this pair and no graph needs anything else at its root,
    /// so spelling out three lines at every call site would be ceremony. Wire the
    /// surface properties into the first; the second exists only to say which
    /// surface gets drawn.
    ///
    /// The link cannot fail — both sockets are named in this crate's own tables
    /// and are the only Surface-typed pair that exists — so a failure here is a
    /// bug in those tables rather than anything a caller did.
    pub fn surface_output(&mut self) -> (NodeId, NodeId) {
        let surface = self.add(NodeKind::Surface);
        let out = self.add(NodeKind::PbrOutput);
        self.link(surface, "Surface", out, "Surface")
            .expect("Surface.Surface -> PbrOutput.Surface is a fixed pair defined in node.rs");
        (surface, out)
    }

    /// The node an id refers to.
    pub fn kind(&self, id: NodeId) -> &NodeKind {
        &self.nodes[id.0]
    }

    /// Number of nodes, reachable or not.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph has no nodes at all.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Every node, in insertion order.
    ///
    /// Read-only on purpose. This graph is a *compile-time* IR: it is built once
    /// by an author (a Rust literal, or an editor walking its own document) and
    /// then consumed. There is deliberately no `remove_node`, because [`NodeId`]
    /// is a position in this list and removal would either invalidate live ids
    /// or force a tombstone that every consumer would then have to reason about.
    /// An editor keeps its own editable document and rebuilds a `Graph` from
    /// scratch on each change, which costs microseconds and keeps this type
    /// simple enough to be obviously correct.
    pub fn nodes(&self) -> impl Iterator<Item = (NodeId, &NodeKind)> + '_ {
        self.nodes
            .iter()
            .enumerate()
            .map(|(i, kind)| (NodeId(i), kind))
    }

    /// Every wire, as `(source node, source output index)` to
    /// `(destination node, destination input index)`.
    ///
    /// Ordering is deterministic — see the `links` field.
    pub fn links(&self) -> impl Iterator<Item = ((NodeId, usize), (NodeId, usize))> + '_ {
        self.links.iter().map(|(&(to, to_idx), &(from, from_idx))| {
            ((NodeId(from), from_idx), (NodeId(to), to_idx))
        })
    }

    /// Wire an output socket to an input socket.
    ///
    /// Sockets are named, and either the display name (`"Base Color"`) or the
    /// identifier (`"base_color"`) is accepted — handwritten graphs read better
    /// with one, generated code with the other, and both are unique within a
    /// node.
    pub fn link(
        &mut self,
        from: NodeId,
        from_socket: &str,
        to: NodeId,
        to_socket: &str,
    ) -> Result<(), GraphError> {
        let from_idx = find_socket(self.nodes[from.0].outputs(), from_socket)
            .ok_or_else(|| unknown_socket(&self.nodes[from.0], from_socket, Direction::Output))?;
        let to_idx = find_socket(self.nodes[to.0].inputs(), to_socket)
            .ok_or_else(|| unknown_socket(&self.nodes[to.0], to_socket, Direction::Input))?;

        self.link_by_index(from, from_idx, to, to_idx)
    }

    /// Wire an output socket to an input socket, by socket position.
    ///
    /// The same operation as [`Graph::link`] without the name lookup. An editor
    /// already holds socket indices — it drew the sockets from
    /// [`NodeKind::inputs`] / [`NodeKind::outputs`] in order — so making it
    /// stringify a name for this to parse it straight back would be a round trip
    /// that can only lose information.
    pub fn link_by_index(
        &mut self,
        from: NodeId,
        from_idx: usize,
        to: NodeId,
        to_idx: usize,
    ) -> Result<(), GraphError> {
        // Every one of these is `&'static`, so each borrow of `self.nodes` ends
        // with its own statement. That is what lets `self.links` be mutated at
        // the end of this function without the node borrow still being alive.
        let (from_name, from_outputs) = (self.nodes[from.0].name(), self.nodes[from.0].outputs());
        let (to_name, to_inputs) = (self.nodes[to.0].name(), self.nodes[to.0].inputs());

        if from_idx >= from_outputs.len() {
            return Err(unknown_socket(
                &self.nodes[from.0],
                &format!("#{from_idx}"),
                Direction::Output,
            ));
        }
        if to_idx >= to_inputs.len() {
            return Err(unknown_socket(
                &self.nodes[to.0],
                &format!("#{to_idx}"),
                Direction::Input,
            ));
        }

        let from_def = &from_outputs[from_idx];
        let to_def = &to_inputs[to_idx];

        if from_def.ty != to_def.ty {
            return Err(GraphError::TypeMismatch {
                from_node: from_name,
                from_socket: from_def.name,
                from_ty: from_def.ty,
                to_node: to_name,
                to_socket: to_def.name,
                to_ty: to_def.ty,
            });
        }

        if self.links.contains_key(&(to.0, to_idx)) {
            return Err(GraphError::SocketOccupied {
                node: to_name,
                socket: to_def.name,
            });
        }

        self.links.insert((to.0, to_idx), (from.0, from_idx));
        Ok(())
    }

    /// The source feeding an input socket, if any.
    pub(crate) fn source_of(&self, node: NodeId, input_idx: usize) -> Option<(NodeId, usize)> {
        self.links
            .get(&(node.0, input_idx))
            .map(|&(n, s)| (NodeId(n), s))
    }

    /// Locate the single `PbrOutput` node.
    pub fn root(&self) -> Result<NodeId, GraphError> {
        let mut found = None;
        let mut count = 0;
        for (i, kind) in self.nodes.iter().enumerate() {
            if matches!(kind, NodeKind::PbrOutput) {
                count += 1;
                found.get_or_insert(NodeId(i));
            }
        }
        match count {
            0 => Err(GraphError::NoOutputNode),
            1 => Ok(found.expect("count == 1 implies a node was recorded")),
            _ => Err(GraphError::MultipleOutputNodes { count }),
        }
    }

    /// Nodes in dependency order, ending with the root.
    ///
    /// Reached by depth-first search *backwards from the root*, so nodes that
    /// nothing downstream consumes never appear. That is the dead-code pass:
    /// an author can leave a half-built experiment hanging off to the side of a
    /// graph, and it costs nothing in the compiled shader.
    pub fn evaluation_order(&self) -> Result<Vec<NodeId>, GraphError> {
        #[derive(Clone, Copy, PartialEq)]
        enum Mark {
            Unvisited,
            InProgress,
            Done,
        }

        let root = self.root()?;
        let mut marks = vec![Mark::Unvisited; self.nodes.len()];
        let mut order = Vec::new();

        // Explicit stack rather than recursion: graph depth is author-controlled
        // and a deep chain should produce an error, not blow the native stack.
        // Each frame tracks which input it is up to, so the node can be emitted
        // once all of its inputs have been.
        let mut stack: Vec<(NodeId, usize)> = vec![(root, 0)];
        marks[root.0] = Mark::InProgress;

        while let Some((node, cursor)) = stack.pop() {
            let input_count = self.nodes[node.0].inputs().len();

            if cursor == input_count {
                marks[node.0] = Mark::Done;
                order.push(node);
                continue;
            }

            // Re-push with the cursor advanced before descending, so this node
            // is resumed after its child completes.
            stack.push((node, cursor + 1));

            if let Some((src, _)) = self.source_of(node, cursor) {
                match marks[src.0] {
                    Mark::Done => {}
                    Mark::InProgress => {
                        return Err(GraphError::Cycle {
                            node: self.nodes[src.0].name(),
                        })
                    }
                    Mark::Unvisited => {
                        marks[src.0] = Mark::InProgress;
                        stack.push((src, 0));
                    }
                }
            }
        }

        Ok(order)
    }

    /// Check that every link-only input on every reachable node is wired.
    ///
    /// Restricted to reachable nodes deliberately: an unreachable node
    /// contributes nothing to the output, so holding its dangling inputs
    /// against the author would make dead-code tolerance useless.
    pub fn validate(&self) -> Result<(), GraphError> {
        // Names of exposed parameters seen so far. Checked over the reachable
        // set rather than every node, matching the required-link check below:
        // an unreachable node contributes nothing to the compiled shader, and
        // erroring on one would mean a graph that fails to compile because of a
        // node it does not use.
        let mut exposed: BTreeMap<&str, ()> = BTreeMap::new();
        // Texture names live in their own map because they are their own
        // namespace: a param and a texture may share a name, since one is
        // driven by writing a slot and the other by rebinding an image.
        let mut exposed_textures: BTreeMap<&str, ()> = BTreeMap::new();

        for node in self.evaluation_order()? {
            let kind = &self.nodes[node.0];

            if let Some(name) = kind.param_name() {
                // A duplicate is unresolvable rather than merely untidy: whatever
                // drives this material by name would write one of the two slots
                // and silently leave the other at its default.
                if exposed.insert(name, ()).is_some() {
                    return Err(GraphError::DuplicateParamName {
                        name: name.to_string(),
                    });
                }
            }

            if let Some(name) = kind.texture_name() {
                if exposed_textures.insert(name, ()).is_some() {
                    return Err(GraphError::DuplicateTextureName {
                        name: name.to_string(),
                    });
                }
            }

            for (idx, socket) in kind.inputs().iter().enumerate() {
                if socket.default == SocketDefault::Required && self.source_of(node, idx).is_none()
                {
                    return Err(GraphError::RequiredLinkMissing {
                        node: kind.name(),
                        socket: socket.name,
                    });
                }
            }
        }
        Ok(())
    }

    /// Whether this graph drives anisotropy on any surface it actually draws.
    ///
    /// A renderer typically compiles its anisotropic lighting path only on
    /// request — Bevy gates it behind the `STANDARD_MATERIAL_ANISOTROPY` shader
    /// def. Forget to ask for it and a graph that carefully computes a brush
    /// direction renders as an ordinary isotropic surface, with no error and no
    /// warning to say why. That is a genuinely awful thing to debug, and it is
    /// avoidable: the graph already knows, because the answer is just "is
    /// anything wired into those two sockets".
    ///
    /// Asked of every reachable `Surface`, not of the root. Since
    /// `layered-surface` the properties live on `Surface` nodes and a graph may
    /// hold several — brushed metal showing through worn paint is exactly the
    /// case where one layer is anisotropic and the other is not, and the lighting
    /// path has to be switched on for the mix.
    ///
    /// Reachability matters as much as it does for mesh attributes: a `Surface`
    /// left disconnected on the canvas while its replacement is wired up must not
    /// keep a pipeline feature alive. `evaluation_order` already drops those.
    ///
    /// Returns `false` for a graph that does not compile rather than erroring —
    /// the caller is about to fail on that anyway, with a better message.
    pub fn uses_anisotropy(&self) -> bool {
        let Ok(order) = self.evaluation_order() else {
            return false;
        };
        order.iter().any(|&node| {
            self.kind(node)
                .inputs()
                .iter()
                .enumerate()
                .filter(|(_, socket)| socket.ident.starts_with("anisotropy_"))
                .any(|(idx, _)| self.source_of(node, idx).is_some())
        })
    }

    /// Whether this graph reads `Geometry`'s UV output.
    ///
    /// A mesh without a UV channel makes this read zero, silently, so a tool
    /// showing a graph next to a mesh wants to know — see
    /// [`Graph::uses_mesh_attribute`].
    pub fn uses_uv(&self) -> bool {
        self.uses_mesh_attribute(GEOMETRY_UV_OUTPUT)
    }

    /// Whether this graph reads `Geometry`'s vertex color output.
    pub fn uses_vertex_color(&self) -> bool {
        self.uses_mesh_attribute(GEOMETRY_VERTEX_COLOR_OUTPUT)
    }

    /// Whether any node reachable from the output reads a given `Geometry`
    /// output socket.
    ///
    /// Reachability matters: a Geometry node left wired to a branch that goes
    /// nowhere is not a dependency on the mesh having that attribute, and
    /// treating it as one would emit a binding for a value nothing consumes.
    ///
    /// Returns `false` for a graph that has no valid evaluation order at all.
    /// The caller is about to fail on that with a better message.
    pub fn uses_mesh_attribute(&self, output: usize) -> bool {
        let Ok(order) = self.evaluation_order() else {
            return false;
        };
        self.consumes_geometry_output(&order, output)
    }

    /// The reachability-aware core of [`Graph::uses_mesh_attribute`], for the
    /// emitter, which already has the evaluation order in hand and should not
    /// pay for a second topological sort per attribute.
    pub(crate) fn consumes_geometry_output(&self, order: &[NodeId], socket: usize) -> bool {
        order.iter().any(|&node| {
            (0..self.kind(node).inputs().len()).any(|input| {
                matches!(
                    self.source_of(node, input),
                    Some((src, src_socket))
                        if src_socket == socket && matches!(self.kind(src), NodeKind::Geometry)
                )
            })
        })
    }
}

fn find_socket(sockets: &[SocketDef], name: &str) -> Option<usize> {
    sockets
        .iter()
        .position(|s| s.name == name || s.ident == name)
}

fn unknown_socket(kind: &NodeKind, socket: &str, direction: Direction) -> GraphError {
    let sockets = match direction {
        Direction::Input => kind.inputs(),
        Direction::Output => kind.outputs(),
    };
    let available = sockets
        .iter()
        .map(|s| s.name)
        .collect::<Vec<_>>()
        .join(", ");
    GraphError::UnknownSocket {
        node: kind.name(),
        socket: socket.to_string(),
        direction: match direction {
            Direction::Input => "input",
            Direction::Output => "output",
        },
        available: if available.is_empty() {
            "none".to_string()
        } else {
            available
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{MathOp, VectorOp};

    #[test]
    fn links_by_display_name_or_ident() {
        let mut g = Graph::new();
        let p = g.param_color([1.0, 0.0, 0.0, 1.0]);
        let (surface, _) = g.surface_output();
        assert!(g.link(p, "Result", surface, "Base Color").is_ok());

        let mut g2 = Graph::new();
        let p2 = g2.param_color([1.0, 0.0, 0.0, 1.0]);
        let (surface2, _) = g2.surface_output();
        assert!(g2.link(p2, "result", surface2, "base_color").is_ok());
    }

    #[test]
    fn indexed_and_named_linking_agree() {
        // The editor uses the indexed form exclusively, so the two spellings
        // diverging would mean hand-written and authored graphs quietly
        // compiling to different shaders.
        let build = |indexed: bool| {
            let mut g = Graph::new();
            let p = g.param_color([1.0, 0.0, 0.0, 1.0]);
            let (surface, _) = g.surface_output();
            if indexed {
                g.link_by_index(p, 0, surface, 0).unwrap();
            } else {
                g.link(p, "Result", surface, "Base Color").unwrap();
            }
            g
        };
        assert_eq!(
            crate::emit(&build(true)).unwrap(),
            crate::emit(&build(false)).unwrap()
        );
    }

    #[test]
    fn out_of_range_socket_index_names_the_node() {
        let mut g = Graph::new();
        let p = g.param_float(1.0);
        let out = g.add(NodeKind::PbrOutput);
        let text = g.link_by_index(p, 0, out, 99).unwrap_err().to_string();
        assert!(text.contains("PbrOutput"), "{text}");
        assert!(text.contains("#99"), "{text}");
    }

    #[test]
    fn read_accessors_round_trip_a_graph() {
        // Reloading a preset into an editor means walking these two iterators
        // and rebuilding; if either drops something the canvas comes back wrong.
        let original = crate::materials::hull_metal().unwrap();

        let mut rebuilt = Graph::new();
        for (_, kind) in original.nodes() {
            rebuilt.add(kind.clone());
        }
        for ((from, from_idx), (to, to_idx)) in original.links() {
            rebuilt.link_by_index(from, from_idx, to, to_idx).unwrap();
        }

        assert_eq!(
            crate::emit(&rebuilt).unwrap(),
            crate::emit(&original).unwrap()
        );
    }

    #[test]
    fn mesh_attribute_use_is_detected_and_ignores_dead_branches() {
        // Reachability is the whole point: a Geometry node whose UV output feeds
        // a branch that goes nowhere is not a dependency on the mesh having UVs.
        let mut graph = Graph::new();
        let geometry = graph.add(NodeKind::Geometry);
        let dead = graph.add(NodeKind::VectorMath(crate::node::VectorOp::Normalize));
        let (surface, _) = graph.surface_output();
        graph.link(geometry, "UV", dead, "Vector").unwrap();
        graph.link(geometry, "Normal", surface, "Normal").unwrap();

        assert!(!graph.uses_uv(), "the UV branch terminates nowhere");
        assert!(!graph.uses_vertex_color());

        // Wire the dead branch to the surface and it becomes a real dependency.
        graph
            .link(dead, "Vector", surface, "Anisotropy Tangent")
            .unwrap();
        assert!(graph.uses_uv());
    }

    #[test]
    fn anisotropy_is_detected_from_the_wiring() {
        // hull_metal wires a brush direction into the tangent; glass does not.
        assert!(crate::materials::hull_metal().unwrap().uses_anisotropy());
        assert!(!crate::materials::glass([0.0; 4], 0.5)
            .unwrap()
            .uses_anisotropy());

        // An empty graph has no output node at all. Reporting `false` rather
        // than panicking matters: the editor asks this on every keystroke,
        // including while a graph is half-built.
        assert!(!Graph::new().uses_anisotropy());
    }

    #[test]
    fn rejects_mismatched_types_and_names_both_ends() {
        let mut g = Graph::new();
        let p = g.param_float(1.0);
        let (surface, _) = g.surface_output();
        let err = g.link(p, "Value", surface, "Base Color").unwrap_err();
        match err {
            GraphError::TypeMismatch {
                from_node,
                to_socket,
                ..
            } => {
                assert_eq!(from_node, "Param");
                assert_eq!(to_socket, "Base Color");
            }
            other => panic!("expected a type mismatch, got {other}"),
        }
    }

    /// A surface is not a color, in both directions.
    ///
    /// The new type has to be as closed as the other three. Wiring a surface
    /// into a color socket would be a category error that no amount of
    /// downstream code could recover from, and the reverse — dropping a color
    /// straight into `PbrOutput` the way graphs did before `layered-surface` —
    /// is the exact mistake someone rebuilding an old graph from memory will
    /// make, so the message has to name both types.
    #[test]
    fn a_surface_is_not_a_color() {
        let mut g = Graph::new();
        let color = g.param_color([1.0, 0.0, 0.0, 1.0]);
        let (surface, out) = g.surface_output();

        let err = g.link(color, "Result", out, "Surface").unwrap_err();
        match err {
            GraphError::TypeMismatch { from_ty, to_ty, .. } => {
                assert_eq!(from_ty.to_string(), "Color");
                assert_eq!(to_ty.to_string(), "Surface");
            }
            other => panic!("expected a type mismatch, got {other}"),
        }

        let mix = g.add(NodeKind::MixColor(crate::node::BlendMode::Mix));
        let err = g.link(surface, "Surface", mix, "A").unwrap_err();
        match err {
            GraphError::TypeMismatch { from_ty, to_ty, .. } => {
                assert_eq!(from_ty.to_string(), "Surface");
                assert_eq!(to_ty.to_string(), "Color");
            }
            other => panic!("expected a type mismatch, got {other}"),
        }
    }

    #[test]
    fn unknown_socket_lists_what_was_available() {
        let mut g = Graph::new();
        let p = g.param_float(1.0);
        let (surface, _) = g.surface_output();
        let err = g.link(p, "Value", surface, "Nonexistent").unwrap_err();
        let text = err.to_string();
        assert!(text.contains("Nonexistent"), "{text}");
        assert!(text.contains("Base Color"), "{text}");
    }

    #[test]
    fn an_input_accepts_only_one_wire() {
        let mut g = Graph::new();
        let a = g.param_color([1.0, 0.0, 0.0, 1.0]);
        let b = g.param_color([0.0, 1.0, 0.0, 1.0]);
        let (surface, _) = g.surface_output();
        g.link(a, "Result", surface, "Base Color").unwrap();
        assert!(matches!(
            g.link(b, "Result", surface, "Base Color"),
            Err(GraphError::SocketOccupied { .. })
        ));
    }

    #[test]
    fn evaluation_order_puts_dependencies_first_and_root_last() {
        let mut g = Graph::new();
        let a = g.param_float(1.0);
        let b = g.param_float(2.0);
        let m = g.add(NodeKind::Math(MathOp::Add));
        let col = g.param_color([1.0; 4]);
        let (surface, out) = g.surface_output();
        g.link(a, "Value", m, "Value").unwrap();
        g.link(b, "Value", m, "Value_001").unwrap();
        g.link(m, "Value", surface, "Metallic").unwrap();
        g.link(col, "Result", surface, "Base Color").unwrap();

        let order = g.evaluation_order().unwrap();
        let pos = |n: NodeId| order.iter().position(|x| *x == n).unwrap();
        assert!(pos(a) < pos(m));
        assert!(pos(b) < pos(m));
        assert!(pos(m) < pos(surface));
        // The surface has to be built before the node that draws it, and the
        // root stays last however many layers precede it.
        assert!(pos(surface) < pos(out));
        assert_eq!(*order.last().unwrap(), out);
    }

    #[test]
    fn unreachable_nodes_are_dropped() {
        let mut g = Graph::new();
        let used = g.param_color([1.0; 4]);
        let orphan = g.param_float(9.0);
        let (surface, _) = g.surface_output();
        g.link(used, "Result", surface, "Base Color").unwrap();

        let order = g.evaluation_order().unwrap();
        assert!(!order.contains(&orphan));
        // ...and its dangling state does not make the graph invalid.
        assert!(g.validate().is_ok());
    }

    #[test]
    fn missing_output_node_is_an_error() {
        let g = Graph::new();
        assert_eq!(g.evaluation_order().unwrap_err(), GraphError::NoOutputNode);
    }

    #[test]
    fn two_output_nodes_is_an_error() {
        let mut g = Graph::new();
        g.add(NodeKind::PbrOutput);
        g.add(NodeKind::PbrOutput);
        assert_eq!(
            g.root().unwrap_err(),
            GraphError::MultipleOutputNodes { count: 2 }
        );
    }

    #[test]
    fn cycles_are_detected() {
        let mut g = Graph::new();
        let a = g.add(NodeKind::VectorMath(VectorOp::Normalize));
        let b = g.add(NodeKind::VectorMath(VectorOp::Normalize));
        let (surface, _) = g.surface_output();
        g.link(a, "Vector", b, "Vector").unwrap();
        g.link(b, "Vector", a, "Vector").unwrap();
        g.link(b, "Vector", surface, "Normal").unwrap();
        assert!(matches!(
            g.evaluation_order(),
            Err(GraphError::Cycle { .. })
        ));
    }

    #[test]
    fn link_only_inputs_must_be_wired() {
        let mut g = Graph::new();
        // Voronoi's Vector input has no default.
        let vor = g.add(NodeKind::Voronoi);
        let (surface, _) = g.surface_output();
        g.link(vor, "F1 Distance", surface, "Metallic").unwrap();
        assert_eq!(
            g.validate().unwrap_err(),
            GraphError::RequiredLinkMissing {
                node: "Voronoi",
                socket: "Vector"
            }
        );
    }

    /// A `PbrOutput` with nothing plugged in names the socket.
    ///
    /// The deliberate consequence of giving `Surface` no default: there is no
    /// literal surface for an unwired root to fall back to, and inventing a
    /// default grey here would duplicate the ten property defaults that already
    /// live on the `Surface` node's sockets. One source of truth, one clear
    /// error — and the editor keeps showing the last good shader meanwhile.
    #[test]
    fn an_output_with_no_surface_says_so() {
        let mut g = Graph::new();
        g.add(NodeKind::PbrOutput);
        assert_eq!(
            g.validate().unwrap_err(),
            GraphError::RequiredLinkMissing {
                node: "PbrOutput",
                socket: "Surface"
            }
        );
    }

    /// A reroute changes where a wire goes and nothing else.
    ///
    /// The property that makes it safe to scatter them through a graph purely
    /// for tidiness: the *values* the shader computes must be identical with and
    /// without. Only the binding names may differ, since the reroute occupies a
    /// node index — so this compares the root block, where the actual surface
    /// assignments live, rather than the whole source.
    #[test]
    fn a_reroute_passes_its_value_through_unchanged() {
        let build = |detour: bool| {
            let mut g = Graph::new();
            let value = g.param_float(0.6);
            let (surface, _) = g.surface_output();
            if detour {
                let via = g.add(NodeKind::Reroute(crate::ValueType::Float));
                g.link(value, "Value", via, "Value").unwrap();
                g.link(via, "Value", surface, "Metallic").unwrap();
            } else {
                g.link(value, "Value", surface, "Metallic").unwrap();
            }
            crate::emit(&g).unwrap()
        };

        let root_of = |wgsl: String| {
            wgsl.split("// --- surface properties ---")
                .nth(1)
                .expect("every shader has a root block")
                .to_string()
        };

        assert_eq!(
            root_of(build(true)),
            root_of(build(false)),
            "routing a wire through a reroute changed what the shader computes"
        );
    }

    /// A reroute is the one node whose type is not implied by its kind, so the
    /// type checker has to treat it like any other typed socket.
    #[test]
    fn a_reroute_only_accepts_its_own_type() {
        let mut g = Graph::new();
        let color = g.param_color([1.0, 0.0, 0.0, 1.0]);
        let via = g.add(NodeKind::Reroute(crate::ValueType::Float));
        assert!(matches!(
            g.link(color, "Result", via, "Value"),
            Err(GraphError::TypeMismatch { .. })
        ));

        // ...and a surface reroute carries surfaces, which is the case that
        // matters for keeping a layered graph readable.
        let mut g = Graph::new();
        let paint = g.add(NodeKind::Surface);
        let via = g.add(NodeKind::Reroute(crate::ValueType::Surface));
        let out = g.add(NodeKind::PbrOutput);
        g.link(paint, "Surface", via, "Surface").unwrap();
        g.link(via, "Surface", out, "Surface").unwrap();
        assert!(g.validate().is_ok());
    }

    /// Surfaces nest, which is what makes a third layer possible at all.
    #[test]
    fn surfaces_mix_and_nest() {
        let mut g = Graph::new();
        let a = g.add(NodeKind::Surface);
        let b = g.add(NodeKind::Surface);
        let c = g.add(NodeKind::Surface);
        let inner = g.add(NodeKind::MixSurface);
        let outer = g.add(NodeKind::MixSurface);
        let out = g.add(NodeKind::PbrOutput);

        g.link(a, "Surface", inner, "A").unwrap();
        g.link(b, "Surface", inner, "B").unwrap();
        g.link(inner, "Surface", outer, "A").unwrap();
        g.link(c, "Surface", outer, "B").unwrap();
        g.link(outer, "Surface", out, "Surface").unwrap();

        assert!(g.validate().is_ok());
        let wgsl = crate::emit(&g).unwrap();
        assert_eq!(
            wgsl.matches("sg_mix_surface(").count(),
            3,
            "two call sites plus the one declaration"
        );
    }
}
