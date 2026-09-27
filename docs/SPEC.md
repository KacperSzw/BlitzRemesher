# Accepted implementation specification

Build a portable C++20 static opaque render-mesh simplification library, C ABI,
CLI, independent evaluator and reproducible research harness. Original code is
MIT OR Apache-2.0. Skinning, morphs, alpha textures, texture-image scoring and
engine-specific plugins are deferred.

Output modes: Rebuild (default, owned new positions/attributes) and Reuse
(only indices/material metadata; immutable original vertex IDs and streams).
Topology changes are permitted only when the chosen visual contract passes.
Chain modes: Direct from source, Progressive from previous, Hybrid (default)
combining proposals and selecting minimum-total-triangle valid chains.

Reference D is the source bounding-sphere diameter in metres. Defaults:
rho=512 px/m, S_last=16 px, N=8 total levels including unchanged LOD0.
S_i=S_0*(S_last/S_0)^(i/(N-1)); explicit base screen size is also supported.
Transition curve defaults 2 to 3 px. Sample evenly across N-1 transitions;
one transition uses the start. B_i=sum(j=1..i,delta_j*S_i/S_j).
An optional max_lod0_delta_px caps B_i. Both source and adjacent errors are
measured directly at S_i. B is a source-fidelity POLICY, not a proof that
appearance, rasterization or perspective errors compose under rescaling.

Coverage is symmetric Hausdorff distance of filled foreground, not contours.
Appearance uses Hausdorff in the product metric:
sqrt(pixel_distance^2+(normal_weight*angle)^2+
color_weight^2*linear_rgb_distance^2+material_weight^2*(id_mismatch)).
Profiles are coverage, coverage+normals, and coverage+normals+attributes.
Defaults: 1 pixel per 10 degrees, color weight 4, material mismatch weight 4.
Weights are configurable curves, including zero. UV charts are preserved and
distortion is reported; opaque texture images and normal maps are not scored.

Cameras are derived only from source bounds. Search: 42 orthographic and 12
perspective; intermediate: 162+32; audit: separately rotated 642+64. FOV=60 deg.
Search uses 4x supersampling, audit uses conservative 8x coverage refined to
16x/32x when uncertain. Raster allowance 2*sqrt(2)/s plus numerical safeguards.
One empty render versus nonempty fails; two empty renders agree.
Normals are sampled visible interpolated normals, or face normals when absent.
These are audited-camera/sample guarantees, never universal appearance bounds.

Quality preset allows 64 candidate evaluations/level, fast eight. Hybrid
retains eight reduced candidates plus incumbent and source fallback. Counts
must not increase with level. Cancellation returns a validated incumbent with
an explicit completion status. Benchmarks use deterministic work budgets.

Public C ABI: strided borrowed streams, versioned descriptors, explicit status,
opaque result ownership, read-only views and destruction inside the library.
No C++ exception, STL or cross-module allocator ownership crosses the ABI.
Initial file formats: glTF/GLB, OBJ, PLY, STL. glTF output defaults to LOD0
and includes a JSON LOD manifest; reuse output shares source accessors.

Core storage: float streams, double quadric accumulation/solving, checked u32
IDs, smaller bounded fields and packed flags. Scalar reference before AVX2;
runtime feature dispatch and scalar fallback. Stream raster data and bound
caches. Parallelize assets/views before topology mutation.

Acquire 120 distinct CC0 assets: 30 rocks, 30 opaque organics (>=10 woody),
40 manufactured, 20 complex scans/stress. Sources: Poly Haven, ambientCG,
Smithsonian. Preserve licensing/provenance/hash metadata; count related
formats/LODs/scans once. Source-data budget 50 GiB, two binary downloads.
Freeze grouped stratified 80/20/20 development/validation/held-out splits.
Procedural fixtures do not count toward the 120 downloads.

Research rounds: objectives/placement, visual guidance/topology/search, then
memory/scheduling/SIMD. Pin meshoptimizer 1.3 commit
9e1f07b159d3cb777f1c67ed31fc11fd117986f4; Fast Quadric and CGAL are separate
research executables and never dependencies of the shipped core.

Complete when all interfaces/modes, corpus, reproducible reports and three
recorded experiment rounds work end-to-end. Report limitations honestly.
