# Vegetation reduction: measured improvement round

The new proposal policy substantially reduces distant geometry on the fern,
broadleaf tree and the broader development collection. It is available through
opt-in C++/CLI settings; the default reducer and C descriptor remain unchanged.
The cost is longer baking, and individual meshes still regress. There is no
universal winner: the pinned meshoptimizer adapter is substantially better on
the shared-vertex fern.

Open the [interactive chain board](board/index.html), read the
[complete measurements](RESULTS.md), or reproduce the run with
[these commands](REPRODUCE.md). The board contains actual baked geometry,
before/after fern and tree pairs in both vertex modes, per-chain seconds,
coverage change and independent dense-audit results. Its first four opaque
examples are archived context, not newly measured algorithm comparisons.

## What changed

Ordinary face-plane quadrics barely penalize in-plane movement on flat cards.
Boundary quadrics add the missing direction. The implementation uses planes
perpendicular to each boundary face, weighted by normalized edge length squared.
The search compares both legal boundary endpoints and, in rebuilt mode,
minimizes along the boundary edge. Existing orientation and material checks
remain in force.

Coincident positions previously locked separate indexed charts wholesale.
The experimental independent-chart path allows contractions within each chart
without welding its streams to another chart. Shared mode keeps all source
vertices and attributes immutable; it also rejects UV orientation inversions.
Rebuilt mode can produce owned streams, while a legal endpoint proposal may
still reference source storage. The board labels actual storage per LOD.

The adaptive target scheduler refines the interval between accepted and rejected
requests while retaining broad coarse probes: visual acceptance is not monotonic
in triangle count. A bounded duplicate cache avoids repeated evaluation of the
same geometry under the same parent. Duplicate reductions still consume the
proposal budget. No extra candidate is obtained by changing the denominator.

An optional component proposer ranks material-scoped components by marginal
multi-view coverage contribution per triangle. Counts update as components are
removed, so overlapping cards do not all appear independently redundant.
It uses packed masks, compact component IDs and bounded scratch memory. This
experiment did not improve the pilot enough to join the selected preset.

All proposals use the same four gates: source search, previous-LOD search,
source acceptance audit and previous-LOD acceptance audit. The source fallback,
monotone triangle counts and exact runtime duplicate compaction are unchanged.

## The measured preset

Use [shared vertices](preset-reuse.json) or
[rebuilt vertices](preset-rebuild.json) with the CLI. The selected research
fields are:

```json
{
  "boundary_weight": 10,
  "boundary_placement": true,
  "independent_seams": true,
  "adaptive_targets": true,
  "component_candidates": false,
  "trace": false
}
```

These fields belong inside the `research` object. The full presets fix
512→16 px, eight scheduled slots, 2→3 px transitions, an 8 px source cap,
hybrid search, beam width two and eight proposals per level. Their small search
and acceptance camera sets are research settings; independent dense tail
checks are reported separately. They do not change the library quality default.
The final effective source limit is 6.8905 px; raising the inactive 8 px cap
would not improve this particular experiment.

For the six-example matrix at eight proposals:

| Asset | Vertex mode | Previous final tris | Candidate final tris |
| --- | --- | ---: | ---: |
| Fern | Shared original buffer | 482 | 260 |
| Fern | Rebuilt allowed | 48 | 16 |
| Broadleaf tree | Shared original buffer | 523 | 110 |
| Broadleaf tree | Rebuilt allowed | 704 | 110 |

The rebuilt fern's 16 triangles must be compared with the previous rebuilt
result of 48, not the shared result of 482. The candidate improves
category-balanced tail retention by 55.8% shared and 49.4% rebuilt over these
six examples. Across all 28 development models, improvements are 39.2% shared
and 41.5% rebuilt. The frozen 12-model validation improves tail retention by
47.0% shared and 48.1% rebuilt, with improved aggregate whole-chain retention
in both modes. Every individual regression is retained in the generated
measurements.

All 20 independent dense tail checks passed: the twelve candidate board
chains, four primary controls, and the four 64-proposal fern/tree candidates.
That is 120 source/adjacent tail gates. These results support the experimental
presets for this geometry contract; they do not certify opacity or shading.

## What did not win

- Increasing the proposal budget improves some chains but is expensive.
  Native shared fern ends at 260/236/233 triangles for budgets 8/32/64;
  rebuilt fern ends at 16/9/5. These are different work budgets, not a
  same-budget algorithm comparison. Whole-chain selection can also worsen a
  particular final LOD while improving the total chain.
- Boundary weight one produces a very small shared tree in one screening
  configuration, but that setting does not win the broader tail comparison.
  We do not choose a different weight for each displayed asset.
- Component-only proposals and the combined component portfolio did not
  justify their cost in the primary pilot. They remain available for research,
  disabled in the preset.
- meshoptimizer pruning and permissive options did not materially improve
  these two primary assets. Its default shared fern result is already much
  smaller than the native result. The adapter and all 36 comparisons remain
  in the repository, including unsuccessful variants.
- Shrub and flowering ground cover can retain more final triangles under the
  eight-proposal candidate despite better aggregate chain/tail retention.
  Their regressions are visible in the board table and per-asset results.
- The validation pine saplings expose a stronger limitation: all three regress
  in shared-mode tail retention. Clump 3 ends at 1,264 triangles versus the
  control's 149, despite a better whole-chain result. This is a reason to keep
  the control available and investigate a broader proposal mix.

## Interpretation and next research

The 40-model collection is split by family into 28 development and 12 validation
models. The validation settings were frozen before running those families;
these are not claimed to be unseen holdouts. No foliage SCORE is mixed with
the frozen opaque corpus or its score. Raw measurements include input hashes,
configurations, build identities, completion status and failures.

The next useful work is to close the shared-fern gap with stronger legal
topology proposals and reduce repeated coverage-audit work. For the candidate
rebuilt fern, the measured 31.26 s splits into 0.13 s reduction, 4.93 s
rasterization and 26.20 s distance evaluation. Shared mode has the same pattern:
0.12 s reduction, 4.94 s rasterization and 26.56 s distance evaluation. Audits
dominate these cases; optimizing collapse generation alone would barely change
their bake time.

A bounded cache of source/parent masks and exact distance transforms for fixed
cameras, sizes and sampling levels is the next performance experiment. It must
preserve the current conservative acceptance results and be measured against
uncached evaluation. The traces also show more proposals reaching later gates
under the candidate policy. Component removal needs better candidate allocation
before further optimization is justified.

Triangle count alone does not establish vegetation appearance or rendering
cost. This round measures solid card coverage, without opacity textures,
normal shading or wind deformation. Area loss, tiny triangles, per-primitive
2×2 quads and overlap are separate diagnostics. In particular, passing a
distance gate does not imply a small changed area, and CPU quad counts are not
GPU time. Original material payloads remain engine-owned.

For example, the eight-proposal rebuilt fern changes 59.1% of the source's
filled coverage in its worst small audit view, versus 42.5% for the previous
48-triangle result. Its final per-primitive quad count falls from 499 to 271;
the rebuilt tree falls from 2,649 to 1,469. Those counts sum the same 16 views.
This is fewer triangles and less geometry work under the stated distance
contract, with a visible coverage tradeoff. Across the independent dense views,
the rebuilt fern's worst changed area reaches 67.5% at eight proposals and
82.7% for the five-triangle, 64-proposal result. Both pass the distance limits;
neither should be described as preserving the source's filled area.

The implementation uses the existing double-precision quadrics and scalar/AVX2
evaluator. No additional SIMD path or reduced-precision arithmetic was added
without a measured need. The new algorithm options require a C++ rebuild;
the stable C ABI is unchanged.

On the separate frozen eight-asset opaque pilot, the opt-in candidate raises
the shared-mode SCORE from 62.1479 to 94.8371 and rebuilt-mode SCORE from
94.6897 to 97.2338. This uses the existing 32→16 px direct pilot contract,
not the vegetation matrix. All 16 default-control chains reproduce the old
executable's geometry, attributes and LOD measurements exactly. See
[validation](VALIDATION.md) for the recorded checks.

## Sources and provenance

- [Garland–Heckbert quadric error metrics](https://mgarland.org/research/quadrics.html)
  provide the boundary-plane basis.
- [Lindstrom–Turk image-driven simplification](https://faculty.cc.gatech.edu/~turk/my_papers/image_simp_tog2000.pdf)
  motivates evaluating rendered impact. The component heuristic here is an
  engineering experiment, not a reproduction of that algorithm.
- [Pinned meshoptimizer 1.3](https://github.com/zeux/meshoptimizer/tree/9e1f07b159d3cb777f1c67ed31fc11fd117986f4)
  supplies the separate research baseline; it is not a shipped core dependency.
- [Frozen cohort](cohort.json), [plan](PLAN.md),
  [raw measurements](measurements.json), [validation](VALIDATION.md).

The measured v3 core overlay is archived as `archive/source-v3.tar.gz`, SHA-256
`e07ff05ce338b64d3ecd4af8cdf3c75f5cdad09a19b277c437a9c6da09af44dc`.
Historical v1/v2 screening runs retain their original hashes and timings and
are not treated as final v3 results.
