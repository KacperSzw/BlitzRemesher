# Algorithm decisions and open questions

The optimization objective is the smallest **validated** chain found within a
fixed work budget. No available simplification heuristic establishes global
optimality for this objective. Proposal costs and acceptance metrics are
deliberately separate.

| Approach | Strength | Limitation / role here |
|---|---|---|
| Garland–Heckbert QEM | Compact additive surface approximation, inexpensive collapse ranking | Geometry cost is not a screen-space bound; native starting point |
| Endpoint-only QEM | Reuses the original vertex buffer | Less freedom of placement; separate leaderboard |
| Attribute quadrics | Joint geometry/attribute approximation | Attribute units require calibration; not implemented as a full high-dimensional quadric yet |
| Probabilistic quadrics | Models uncertain positions/normals; regularizes noisy scans | Tested through CGAL's actual implementation; native positional regularization is a different method |
| Lindstrom–Turk | Memoryless local volume/boundary constraints | CGAL reference; oriented manifold input requirements |
| Fast Quadric threshold passes | Simple, fast threshold-based proposal generation | Placement and topology behavior need the same independent audit |
| meshoptimizer | Production-oriented simplification, attribute and vertex-update APIs | Strong external reference; ordinary simplify APIs reuse vertices |
| Progressive meshes | Nested collapse histories, geomorph/progressive transmission | Our progressive mode regenerates from the preceding mesh; it is not a persistent progressive-mesh encoding |
| Vertex clustering / voxel remeshing | Very fast aggressive topology changes | Can erase thin structures and UV islands; future proposal providers |
| Image-driven optimization | Direct relationship to appearance at intended size | Expensive and susceptible to camera overfitting; independent rotated audit required |
| Component pruning | Removes visually irrelevant disconnected pieces | Only a proposal; every delivered result still passes both visual gates |
| Meshlets / cluster-local LOD | Streaming and GPU culling integration | Orthogonal to this library's object LOD chain; deferred |

Primary references:

- [QEM and attribute extensions](https://www.mgarland.org/research/quadrics.html).
- [Hoppe: progressive meshes](https://hhoppe.com/proj/pm/).
- [Lindstrom and Turk: image-driven simplification](https://faculty.cc.gatech.edu/~turk/my_papers/image_simp_tog2000.pdf).
- [Trettner and Kobbelt: probabilistic quadrics](https://graphics.rwth-aachen.de/publication/03308/).
- [CGAL 6.1.1 simplification policies](https://doc.cgal.org/6.1.1/Surface_mesh_simplification/index.html).
- [meshoptimizer at the pinned revision](https://github.com/zeux/meshoptimizer/tree/9e1f07b159d3cb777f1c67ed31fc11fd117986f4).
- [Fast Quadric at the pinned revision](https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification/tree/65df07dc54766e3ee480482f1c881a62767831cc).

The native implementation uses normalized float geometry, double plane
quadrics, CSR incidence and sorted packed edges. Independent collapse batches
avoid per-edge heap allocations. Duplicate-position wedges, material
boundaries, nonmanifold edges and UV winding are protected conservatively.
This currently sacrifices reduction opportunities at seams. It does not
silently weld or repair the source.

The native regularized option adds positional anchors to plane quadrics.
The visual option adds a normal-curvature penalty to the proposal ranking;
its name does **not** mean it computes image gradients or implements the
Lindstrom–Turk image-driven algorithm. Actual image guidance currently occurs
through candidate rejection and chain search. These distinctions must remain
explicit when recording experiments.

Hybrid search is a bounded beam through audited candidates. It minimizes
total triangles over the retained paths, not over all possible meshes or
all possible paths. A logarithmic count ladder includes aggressive and
near-incumbent proposals; errors need not be monotone in triangle count.

## Evaluator contract

Coverage compares filled foreground sets by symmetric Hausdorff distance.
Conservative triangle/cell intersection and the additive 2*sqrt(2)/s
allowance bound the cell-grid approximation for the audited cameras.
The appearance part measures visible center samples in a product metric.
It is **sampled**, including at silhouettes and subpixel features; it is not
a continuous shading guarantee. Conservative coverage is still checked even
when a thin triangle has no visible center sample.

Both empty images agree; one empty image fails against nonempty. Candidate
clipping fails. Source bounds fix all cameras. RGB streams are interpreted
as linear. Material IDs are categorical. Texture images, opacity maps and
normal maps do not participate. UV diagnostics report density, conformal
anisotropy and negative/degenerate winding; negative winding alone is not a
defect because mirrored charts are valid.

An exact source fallback uses the mathematical identity shortcut. Other
proposals pay the raster uncertainty allowance. Reports identify the camera
and sampling settings; changing them creates a different scenario.

## Next research opportunities

1. Attribute transfer and placement for the implemented coupled wedge proposals.
2. Attribute quadrics and genuine render-derived proposal priorities.
3. Visibility caches / tiled structure-of-arrays raster storage.
4. Persistent collapse histories and geomorphable transitions.
5. More aggressive topology proposals, admitted only by unchanged gates.
6. Wider default-camera benchmarks after the small fixed pilot scenarios.
