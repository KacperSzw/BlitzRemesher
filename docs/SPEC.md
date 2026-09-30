# Accepted implementation specification

Build a portable C++20 static opaque render-mesh simplification library, C ABI,
CLI, independent evaluator and reproducible research harness. Original code is
MIT OR Apache-2.0. Skinning, morphs, alpha textures, texture-image scoring and
engine-specific plugins are deferred.

Production uses automatic hybrid generation. Both source and predecessor inputs,
and both endpoint and repositioning reductions, propose audited candidates.
The selected complete chain minimizes packed resident vertex/index bytes while
each scheduled LOD uses at most floor(reference_triangles*(1+overhead)) triangles.
The reference is one complete minimum-total-triangle path found by this run;
it is not a global optimum. Default overhead is 500 basis points (5%), with
0..10000 supported. The proposal pool is independent of overhead. Source vertex
streams remain immutable; owned levels are compact. Source buffers count once,
exact consecutive runtime duplicates once, and all present attributes/u32 indices
count. Forced output/origin modes remain research controls. See research/hybrid/PLAN.md.
Borrowed reducer/proposer outputs address the input passed to that call. In
progressive rebuild, compact IDs from the preceding LOD must not be interpreted
against LOD0; preserve the referenced input for as long as the output needs it.

Reference D is the source bounding-sphere diameter in metres. Defaults:
rho=512 px/m, S_last=16 px, N=8 total levels including unchanged LOD0.
S_i=S_0*(S_last/S_0)^(i/(N-1)); explicit base screen size is also supported.
Transition curve defaults 2 to 3 px. Sample evenly across N-1 transitions;
one transition uses the start. B_i=sum(j=1..i,delta_j*S_i/S_j).
An optional max_lod0_delta_px caps B_i. Both source and adjacent errors are
measured directly at S_i. B is a source-fidelity POLICY, not a proof that
appearance, rasterization or perspective errors compose under rescaling.
An inactive source cap leaves the cumulative policy unchanged.

Coverage is symmetric Hausdorff distance of filled foreground, not contours.
An optional independent coverage-area audit limits `1 - mask IoU` for
conservative supersampled opaque geometry in each configured full audit view.
It applies to both source and preceding-LOD comparisons. The default maximum
is 1.0 for compatibility; the supplied experimental preset uses 0.5. Pixel error
remains separately bounded. Search views filter by pixel error but do not
reject candidates for area, because their coarse samples cannot establish
full-audit area validity.
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

The optional neural Vulkan backend defines separate `vulkan-v1` raster semantics:
hardware conservative coverage, center-sampled D32 visibility with first-face
depth ties, and FP32 interpolated attributes. It uses the same metric, limits,
camera schedule and refinement allowance, including final GPU confirmation.
Hardware edges/interpolation can differ from the CPU/CUDA reference; record the
backend when comparing results. Packed draw storage is the Vulkan default:
the original source remains FP32 and candidate quantization consumes the existing
budget. It does not alter master/source streams. See
research/neural/GPU-REFACTOR.md for formats, measured limits and evidence.

Quality preset allows 64 candidate evaluations/level, fast eight. Automatic search
reserves half the bounded beam for triangles and fills the rest by resident bytes,
with an exact source fallback. Counts
must not increase with level. Cancellation returns a validated incumbent with
an explicit completion status. Benchmarks use deterministic work budgets.
The opt-in research setting `topology_fallback` applies only to the quadric
objective. If a quadric proposal stops more than four times above its requested
triangle count with link-condition rejections, it tries at most one additional
topology-relaxed proposal per LOD. This work is in addition to `candidate_budget`
and is counted in `candidate_evaluations` and `topology_fallback_proposals`.
The candidate must pass the same source and adjacent pixel and area audits.
The reducer still checks face orientation, UV foldovers and attribute/material
locks, but it does not guarantee manifold topology or prevent new intersections.

Public C ABI: strided borrowed streams, versioned descriptors, explicit status,
opaque result ownership, read-only views and destruction inside the library.
No C++ exception, STL or cross-module allocator ownership crosses the ABI.
Initial file formats: glTF/GLB, OBJ, PLY, STL. glTF output defaults to LOD0
and includes a JSON LOD manifest; reuse output shares source accessors.

Scheduled and runtime LOD counts are separate. Exact consecutive duplicates
share a runtime mesh, exposed through C++ runtime_levels and additive C ABI
queries. Scheduled nodes/manifest rows retain their original thresholds,
source checks and adjacent checks. No score denominator changes. No merge
based only on triangle count or approximate visual similarity.

Optional render-cost diagnostics use fixed pixel centers, a top-left fill
rule, material culling and aligned 2x2 quads. For each primitive, count
covered samples P and touched quads Q before depth testing. P/(4Q) is geometric
lane utilization, not measured shader occupancy. P/unique covered pixels is
pre-depth geometric overlap. Projected area below 1 or 4 pixel squared and
zero-sample triangles describe setup pressure without assigning hardware time.
Compare source and reduced foreground coverage at the same size. These values
do not affect the visual gates or SCORE.

Vertex colors use linear RGBA8 (0..255 per channel, 256 levels), including
the strided C++ and C API streams. The C ABI version and shared-library
compatibility version are 4; older descriptors are unsupported. The C settings
descriptor includes `max_changed_area`; each C LOD record includes source and
adjacent area errors and their worst-view indices.
Import rounds normalized float/unsigned-16 colors to nearest with ties upward,
rejects nonfinite/out-of-range channels, and defaults missing alpha to 255.
Reuse preserves the imported/supplied bytes. Rebuild rounds interpolated RGB
on storage and preserves the retained endpoint's alpha. Raster interpolation
and the linear RGB metric continue to use floating-point arithmetic.
glTF exports normalized UNSIGNED_BYTE VEC4 colors; PLY exports uchar RGBA.

Core storage: float geometry streams, RGBA8 colors, double quadric accumulation/solving, checked u32
IDs, smaller bounded fields and packed flags. Scalar reference before AVX2;
runtime feature dispatch and scalar fallback. Stream raster data and bound
caches. Parallelize assets/views before topology mutation.
Coverage-only evaluation uses an exact packed mask and skips attribute/depth
work. The full public rasterizer remains available, and the per-view sample
cap and all acceptance/refinement rules are unchanged. Coverage generation
caches those masks and the existing float32 squared distance fields under a
per-bake allowance. C++/CLI `research.coverage_cache_mib` accepts 0..256 MiB,
default 256; zero disables caching. Half belongs to references for one scheduled level, half to
the current candidate. Full stores bypass new entries without evicting retained
ones. Allocation failure drops optional storage and retries the current view
without an extra cancellation poll. Charged bytes include retained vector and
entry capacities, including overlapping entry arrays during growth; ordinary
evaluator scratch and allocator bookkeeping are outside the cache allowance.
The cache preserves view order, refinement, float distances and acceptance.
Public standalone evaluation is uncached. Production has one
precision path: float32 geometry, double quadric storage/arithmetic/solving,
and double candidate costs. Float quadric/cost variants are archived research.
UNORM16 positions are excluded following the feasibility probe; supplied
float positions and all reuse-mode streams retain their existing contracts.

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
