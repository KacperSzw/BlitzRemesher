# Accepted implementation specification

Build a portable C++20 static opaque render-mesh simplification library, C ABI,
CLI, independent evaluator and reproducible research harness. Original code is
MIT OR Apache-2.0. Skinning, morphs, alpha textures, texture-image scoring and
engine-specific plugins are deferred.

The primary product direction is a standalone neural generator that minimizes
triangles at each scheduled LOD within the configured source and adjacent visual
limits. Level count, minimum screen size and both error policies are configurable.
Coverage is the first training objective; normal and attribute auditing are
selectable. Neural quality configurations use zero triangle overhead and disable
the classical added-vertex cap (`max_added_vertex_bytes_bps: null`). Time and
resident bytes remain diagnostics and secondary choices after the visual and
triangle objectives. Minutes of offline work are acceptable when they improve
audited output. Classical reducers can supply teachers and independent controls;
the neural runtime does not require a classical simplifier to produce proposals.
These objectives describe bounded audited search, not a global optimum.

Classical compatibility path: automatic hybrid generation remains supported.
Both source and predecessor inputs,
and both endpoint and repositioning reductions, propose audited candidates.
The selected complete chain obeys a cap on added packed vertex bytes. The default
cap is 2000 basis points (20%) of the packed source vertex streams; zero forbids
added vertex storage and null disables the cap. The cap counts each distinct
consecutive runtime mesh once. With a cap, the reference is the audited path
found by this run with the fewest triangles at the final LOD, breaking ties at
each preceding LOD in reverse order, then by resident bytes. With the cap
disabled, the reference minimizes total chain triangles. Neither is a global
optimum. The default triangle overhead is zero, so the selected chain is this
reference. With nonzero overhead, selection minimizes
resident bytes subject to each scheduled LOD using at most
floor(reference_triangles*(1+overhead)) triangles. Overhead supports 0..10000
basis points. The proposal pool is independent of overhead. Source vertex
streams remain immutable; ordinary owned levels are compact. Source buffers count once,
exact consecutive runtime duplicates once, and all present attributes/u32 indices
count. Forced output/origin modes remain research controls. See research/hybrid/PLAN.md.
Forced output research controls bypass the production vertex cap and report it
as disabled so archived placement experiments keep their original meaning.
The experimental C++/CLI `research.graph_passes` control (0..3, default zero)
adds bounded whole-chain improvement passes after this reference search.
With it enabled, selection uses the equally weighted LOD1–N triangle total
under the configured vertex cap. Source admission and transition edges are
audited independently. Each layer retains at most `2*beam_width` additional
geometries, four incoming predecessor geometries per candidate, and
`4*beam_width` active paths, plus pinned source/incumbent paths. Dominance
applies only at identical terminal geometry and preserves triangle counts,
added bytes and resident bytes; nonzero overhead also preserves earlier
per-slot counts. The complete incumbent is retained through interruption and
allocation failure. This is a bounded search, with no optimality guarantee.
Each graph pass uses at most `candidate_budget` reduction calls per transition;
its optional topology-relaxed calls replace calls within this budget.
Source/transition audit counts and complete-incumbent progress are exported.
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
Weights are configurable curves, including zero. The classical path and neural
v1-v3 models preserve UV charts and report distortion. Neural v4 policies encode
the separately configurable `preserve_uv` condition. It defaults to on; turning
it off removes UV-specific seam/foldover restrictions while retaining available
UVs and tangents on a best-effort basis. It does not remove geometric legality or
enabled visual gates, and it does not discard supplied source attributes. Reuse
still preserves the referenced source bytes. Opaque texture images and normal
maps are not scored in either mode.

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
budget. Supplied streams and the immutable source are never mutated. Packed
working geometry is decoded once into FP32 GPU master streams. An optional
bitmap keeps exact FP32 positions for at most 5% of referenced surviving vertex
IDs, counting separate wedges. Promotions are bounded, witness-directed and
fully re-audited; exceeding the cap rejects the trial. Old replay files retain
strict packing semantics. See
research/neural/GPU-REFACTOR.md for historical measurements and
research/neural/COMPACT-PIPELINE.md for the current storage contracts and local
evidence. The research learning cycle uses compact feature shards decoded to
FP32 on GPU, fixed source quantization bounds and bounded packing repair.
The coverage pretraining profile uses a single conservative R8 Vulkan pass,
owned one-bit reference masks and direct candidate surfaces. It retains source,
predecessor, pixel-distance and area gates, with no appearance supervision.
Normal edits are suppressed for this profile. Full-shading diagnostics remain
nonblocking for pretraining and do not establish release quality. The unchanged
attributes profile still supervises appearance. Predecessors must pass at both
preparation and destination sizes; audited exhausted/no-reduction outcomes are
recorded explicitly without inventing a predecessor.
Representation failures yield no training labels; an unconfirmed final chain
remains an explicit failed diagnostic. Packing does not relax visual limits.

The classical quality preset allows 64 candidate evaluations/level, fast eight.
Its automatic search reserves half the bounded beam for triangles and fills the
rest by least added vertex bytes when a cap is active, or resident bytes when disabled,
with an exact source fallback. Counts
must not increase with level. Cancellation returns a validated incumbent with
an explicit completion status. Benchmarks use deterministic work budgets.
With a positive vertex cap and the default reducer, up to eight deterministic
compact tail probes run before the scheduled search. A probe must fit the cap
and pass source and source-path transition audits at the final scheduled size.
The smallest accepted tail reserves its actual packed vertex bytes for the
last LOD; earlier levels can use only the remainder. At the last level, the
reserved mesh is offered to every retained prefix and receives the usual
source and adjacent audits. Probe work is additional to `candidate_budget`
and included in `candidate_evaluations`; diagnostics record its count and
reserved bytes. If no probe passes, no bytes are reserved. The source fallback
also remains eligible for direct tail proposals. Rebuild requests are bounded
by the remaining budget divided by three times the packed source vertex
stride, so even a single proposal can probe a compact tail. Actual emitted
bytes decide admission; target counts are only search hints.
If the selected final scheduled LOD is an exact duplicate of its predecessor
and at most half of a positive added-vertex budget was spent, automatic hybrid
generation runs one additional search with adaptive triangle targets. This
retry is skipped for a custom proposer or an already adaptive research run.
Both passes use identical source, schedule, cameras, cap and visual gates.
Their audited finalist pools are selected together, so the retry cannot replace
a better chain from the first pass. Candidate and trace diagnostics count both
passes; traces identify the pass, and the selected chain alone determines
runtime meshes and storage. An interrupted or resource-limited retry keeps
the first chain and reports an incomplete status.
The opt-in research setting `topology_fallback` applies only to the quadric
objective. If a quadric proposal stops more than four times above its requested
triangle count with link-condition rejections, it tries at most one additional
topology-relaxed proposal per LOD. This work is in addition to `candidate_budget`
and is counted in `candidate_evaluations` and `topology_fallback_proposals`.
The candidate must pass the same source and adjacent pixel and area audits.
The classical reducer still checks face orientation, UV foldovers and
attribute/material locks, but it does not guarantee manifold topology or prevent
new intersections.

Neural action policies can propose from source, preceding emitted LOD, or both.
Origins share the same total candidate budget in a matched comparison. Each
proposal's action trials and batch size are explicit execution settings, recorded
separately from visual limits. Predecessor inputs retain their own compact index
domain and ownership; feature normalization and quantization stay in the original
source domain. Every accepted proposal still passes source and adjacent audits.
Record target completion, exhausted legal/acceptable actions, infeasible seeds,
trial caps, cancellation and resource failures separately. A work cap or failed
seed is not evidence of visual exhaustion. Keep all scheduled LODs and unreduced
fallbacks in quality comparisons; show per-level regressions as well as the
category-balanced chain aggregate. Incomplete comparisons have no score.
Cancelled packed generation retains a fully confirmed chain if available;
otherwise it returns only unchanged LOD0 with Cancelled status. An unaudited
scheduled raw-source chain is not a valid packed fallback.
If an explicit added-vertex cap excludes every valid owned packed fallback and
no audited reduced candidate fits, generation returns only unchanged LOD0 with
`BudgetLimited` status. It does not fill scheduled levels with unaudited source
duplicates. An unrelated inability to construct a valid representation remains
an error.

Public C ABI: strided borrowed streams, versioned descriptors, explicit status,
opaque result ownership, read-only views and destruction inside the library.
No C++ exception, STL or cross-module allocator ownership crosses the ABI.
Initial file formats: glTF/GLB, OBJ, PLY, STL. glTF output defaults to LOD0
and includes a JSON LOD manifest; reuse output shares source accessors.

Scheduled and runtime LOD counts are separate. Exact consecutive duplicates
share a runtime mesh, exposed through C++ runtime_levels and C ABI queries.
Per-runtime-level added vertex and index bytes, and cumulative added vertex
bytes, are exposed through C++ runtime_storage, JSON and the C ABI.
Scheduled nodes/manifest rows retain their original thresholds,
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
compatibility version are 5; older descriptors are unsupported. The C settings
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
Public standalone evaluation is uncached. The classical path has one
precision path: float32 geometry, double quadric storage/arithmetic/solving,
and double candidate costs. Float quadric/cost variants are archived research.
UNORM16 positions are excluded from the production CPU reducer following the
feasibility probe (the optional neural Vulkan draw path is described above); supplied
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

Experimental appearance proposals use C++/CLI research.appearance_stage 0..3:
off, collapse ordering, wedge attribute fitting, and joint position fitting.
All default to off. Affine original-face normal/RGB fields accumulate
area-weighted double coefficients in contiguous per-wedge storage; only present,
nonzero-weight channels allocate coefficients (11+4m doubles per wedge, m<=6).
Geometry cost uses the scheduled pixels per input diameter and effective normal
and color weights. This is a proposal surrogate, not a max-pixel error bound.
Materials remain discrete. The coupled positional path retains original
attribute wedges separately; it does not merge wedges across discontinuities.
Ordinary vertex contractions accumulate their endpoint field coefficients.
Ordering evaluates the existing emitted attributes. Fitting normalizes normals,
rounds clamped linear RGB to RGBA8, retains alpha and tangent handedness, and
orthogonalizes tangents to fitted normals. Position fitting eliminates the
independent attribute variables before a 3x3 solve, retaining bounded geometric
and boundary fallbacks. Reuse never writes source streams. Existing topology,
orientation, UV and full source/transition acceptance gates continue to apply.

research.conservative_screen defaults false. When enabled, only a coverage lower
bound above the limit rejects a candidate at search cameras; clipped/uncertain
views defer to the full configured audit. The screen does not certify appearance
or area. Final audits and their sampling/refinement policy remain unchanged.
An optional borrowed EvaluationWitness captures the first failing appearance
sample and the best visible correspondence inside the spatial search radius;
its squared metric components are diagnostics, not continuous error bounds.

Experimental C++/CLI rebuilt-storage controls default to false:

- `research.density_targets` estimates triangle requests from referenced input
  vertices, then actual emitted triangles and packed vertex bytes. Each input,
  placement and search slot owns its feedback. Over-budget predictions leave
  1/16 headroom; stalled requests shrink. Failed appearance does not establish
  monotonicity. Existing candidate/tail-probe budgets still apply, and actual
  emitted storage decides admission.
- `research.merge_wedges` merges contracted continuous interior corner fans.
  Original seams, boundaries, nonmanifold edges, material changes, alpha and
  tangent handedness remain separate. Area-weighted original-face fields fit
  normal/RGB/UV values; boundary UVs stay fixed. A surviving UV reversal restores
  its original fans and attributes locally, with at most four full checks before
  reverting the entire postpass. The ordinary source/adjacent gates still apply.
- In automatic generation, `research.shared_rebuild` matches complete emitted vertex tuples byte for byte
  against the source. Only unmatched tuples consume the added-vertex cap. It
  preserves untouched/endpoint world coordinates through normalization and uses
  the ordinary triangle ladder without the rebuilt triangle-soup clamp. Source
  positions, normals, UVs, RGBA and tangents retain their original bytes and IDs.
  A mixed LOD addresses an immutable owned source prefix plus changed vertices;
  all selected mixed LODs share one combined contiguous pool. Ordinary fully
  rebuilt LODs remain compact. Source-only levels keep prefix accessors; glTF
  writes the combined attribute buffer once. Runtime accounting charges its
  added suffix once at the first mixed level. Result copies and LOD copies retain
  shared pool ownership. Graph dominance also preserves allocation history:
  a path cannot dominate another that owns a different reusable pool. Input
  index/material spans still require the source
  lifetime. If final packing runs out of memory, the valid uncombined incumbent
  is returned with incomplete status. These controls do not change C ABI 5.

These proposal/storage heuristics do not qualify a chain independently or
establish optimality. Frozen trials and audits are in research/density.
