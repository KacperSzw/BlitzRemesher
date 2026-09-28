# Round 4 — larger screens and useful final LODs

The accepted change combines a collapse-neighbor guard with better use of the existing candidate budget. Exact duplicate runtime compaction and render-cost diagnostics are separate from the quality score.

At fixed 512→16 px settings, the development SCORE changes from **75.86 to 80.18**. The category-balanced final retained ratio falls 48.9%; the last-three-level ratio falls 50.9%. On all 20 frozen validation meshes, SCORE changes from **80.30 to 86.48**. No held-out assets were used this round.

[Interactive board](board/index.html) · [Board image](board/board.png) · [Shareable PDF](board/board.pdf)

## Fixed-contract algorithm experiments

Eight frozen development assets; hybrid coupled rebuild, coverage, eight scheduled slots, eight proposals per slot, beam width two plus source fallback. 512→16 px, source cap 8 px, 2→3 px transition curve. Search 6+2 cameras at 2×; audit 12+4 at 4× with refinement to 8×. These are finite-camera audits.

| Variant | SCORE ↑ | Final retained ↓ | Last 3 retained ↓ | Runtime / scheduled | Seconds |
|---|---:|---:|---:|---:|---:|
| [Previous implementation](base-s512-cap8-analysis.json) | 75.86 | 3.14% | 6.61% | 54 / 64 | 397.7 |
| [Relax topology locks](relaxed-s512-cap8-analysis.json) | 75.68 | 3.03% | 6.69% | 56 / 64 | 398.6 |
| [Prevent new topology locks](link-s512-cap8-analysis.json) | 75.75 | 3.02% | 6.52% | 55 / 64 | 395.4 |
| [Guard + improved search prototype](search-s512-cap8-analysis.json) | 80.18 | 1.61% | 3.25% | 53 / 64 | 369.1 |
| [Final: correct progressive ownership](final-s512-cap8-analysis.json) | 80.18 | 1.61% | 3.25% | 53 / 64 | 404.9 |

The reducer probe found zero initially locked edges in the stool, tree and moon rock, but many at the end of the old reduction. Removing UVs/normals did not lower those floors. Some collapses created edges with more than two incident faces, then permanently locked their endpoints. The new common-neighbor check avoids these contractions and also rejects interior contractions joining two boundary vertices. It is not a complete topology-preservation proof. See the three reducer-floor JSON files for unaudited minima and rejection counters.

Relaxing locks alone and adding the guard alone both slightly lower the main SCORE. They improve some tails but can worsen intermediate choices. These negative results are retained. The combined search change avoids redundant hybrid inputs and spends the partial final proposal round on a deeper target. Candidate and camera limits are unchanged; the old scheduler sometimes left budget unused. This remains a bounded heuristic search, without a monotonic-error or optimality assumption.

Code review then found a progressive-input ownership bug: a borrowed result from a rebuilt predecessor could be interpreted against the original source buffer. The final version discards exact unchanged proposals already represented by the incumbent and owns the predecessor streams when borrowed indices change. Both paths have a regression test. The final implementation was rerun across the scenario sweep and all 20 validation assets; prototype measurements remain archived.

Every final replay preserved all prototype position/index and attribute hashes across seven eight-asset scenarios and the 20-asset validation cohort. See [per-asset replay identity](replay-identity.json).

## Final meshes and redundant slots

| Asset | Previous final tris | New final tris | Previous runtime levels | New runtime levels |
|---|---:|---:|---:|---:|
| ph_dead_quiver_trunk | 108 | 2 | 6 | 8 |
| ph_grass_bermuda_01 | 95 | 30 | 6 | 6 |
| ph_metal_stool_02 | 122 | 124 | 6 | 6 |
| ph_moon_rock_02 | 84 | 18 | 8 | 8 |
| ph_painted_wooden_shelves | 26 | 26 | 7 | 7 |
| ph_rock_face_02 | 748 | 119 | 8 | 6 |
| si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652 | 1443 | 1273 | 5 | 5 |
| si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7 | 1642 | 844 | 8 | 7 |

Runtime selection removes only consecutive meshes with identical position, index, normal, UV, color, tangent and material streams. It retains the first slot and its threshold for each group. The scheduled source/adjacent audit records remain intact. glTF nodes share a single mesh payload; C and C++ callers can query the retained slot indices. Compaction changes neither SCORE nor the geometry. Equal triangle counts do not justify merging different meshes. Runtime counts apply the same exact-byte rule to every archived chain; the older exporter still wrote duplicate payloads.

The tree chain changes from 17978 → 1278 → 640 → 320 → 160 → 160 → 108 → 108 to **17978 → 1066 → 534 → 268 → 134 → 68 → 6 → 2**. The new chain has fewer triangles but 8 distinct levels; choosing a six-slot schedule is a separate experiment below.

## Starting size, source cap and six levels

Each row is a different requested visual contract. Compare algorithm versions only within identical settings. The cap clamps the cumulative policy; increasing an inactive cap cannot loosen that policy. All rows use a 16 px final size.

| Algorithm | Start px | Requested cap px | Slots | Actual final source limit px | SCORE | Final retained | Tree final tris | Tree runtime levels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [Previous](base-s128-cap8-analysis.json) | 128 | 8 | 8 | 8.000 | 88.19 | 1.65% | 108 | 4 |
| [New](final-s128-cap8-analysis.json) | 128 | 8 | 8 | 8.000 | 92.19 | 1.54% | 2 | 7 |
| [New](final-s128-cap12-analysis.json) | 128 | 12 | 8 | 9.141 | 92.19 | 1.54% | 2 | 7 |
| [Previous](base-s512-cap2-analysis.json) | 512 | 2 | 8 | 2.000 | 75.13 | 3.76% | 108 | 6 |
| [New](final-s512-cap2-analysis.json) | 512 | 2 | 8 | 2.000 | 76.51 | 2.12% | 34 | 7 |
| [New](final-s512-cap4-analysis.json) | 512 | 4 | 8 | 4.000 | 80.18 | 1.61% | 2 | 8 |
| [Previous](base-s512-cap8-analysis.json) | 512 | 8 | 8 | 6.891 | 75.86 | 3.14% | 108 | 6 |
| [New](final-s512-cap8-analysis.json) | 512 | 8 | 8 | 6.891 | 80.18 | 1.61% | 2 | 8 |
| [New](final-s512-cap8-n6-analysis.json) | 512 | 8 | 6 | 5.406 | 79.52 | 1.93% | 14 | 6 |
| [New](final-s1024-cap8-analysis.json) | 1024 | 8 | 8 | 6.182 | 70.78 | 1.52% | 6 | 8 |

The eight-slot 512 px policy ends at 6.8905 px, so cap 8 already stops constraining the final source error. Cap 4 and cap 2 are active. A larger starting size changes both the sampling schedule and the cumulative budget; six slots also change the transition schedule. Cross-scenario SCORE differences are not algorithm wins.

Caps 4 and 8 produced identical position/index and attribute hashes on all eight development assets. The six-slot tree chain is **17978 → 1066 → 534 → 268 → 134 → 14**. It trades fewer scheduled transitions for a denser final mesh than the eight-slot aggressive chain.

## Pixel and quad costs — useful diagnostics, with a caveat

Fixed sample centers at (x+0.5,y+0.5), top-left fill, source material culling, no MSAA or depth rejection. P counts covered pixel centers per primitive and Q counts touched aligned 2×2 quads per primitive. P/(4Q) is geometric lane utilization; P/unique covered pixels is pre-depth overlap. Zero-sample primitives still have setup cost. These are CPU geometry proxies, not GPU timings or measured helper invocations.

| Tree at 16 px, summed over 16 views | Previous | New |
|---|---:|---:|
| Triangles | 108 | 2 |
| Projected triangles below 1 px² | 94.27% | 18.75% |
| Projected triangles covering zero centers | 86.98% | 56.25% |
| Per-primitive quads | 314 | 22 |
| Quad lane utilization | 32.17% | 40.91% |
| Union covered pixels | 189 | 18 |
| Original source union covered pixels at 16 px | 274 | 274 |

**The 2-triangle result is an aggressive coverage-only choice.** Its lower quad count also reflects substantial loss of filled area. The pilot reports a worst-view changed-area fraction of 87.16%. Pixel-distance error alone does not preserve silhouette density, volume or shading on thin objects. At cap 2 the new tree keeps 34 triangles. Do not interpret the cost reduction as free GPU speed at identical appearance.

The last three tree levels were additionally checked against source and predecessor with 642+64 cameras, 8× sampling and refinement up to 32×. Dense check: **all six gates passed**. This check does not rewrite the pilot score or establish an all-view bound. [Raw dense audit](dense-tree-final.json).

## Frozen validation split

| Variant | Assets | SCORE | Final retained | Last 3 retained | Fallbacks | Failed | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| [Previous](validation-base-s512-cap8-analysis.json) | 20/20 | 80.30 | 3.30% | 6.04% | 0 | 0 | 2124.0 |
| [Search prototype](validation-search-s512-cap8-analysis.json) | 20/20 | 86.48 | 1.36% | 2.69% | 0 | 0 | 2212.4 |
| [Final](validation-final-s512-cap8-analysis.json) | 20/20 | 86.48 | 1.36% | 2.69% | 0 | 0 | 2094.4 |

All 20 frozen validation assets remain in the denominator. Per-asset regressions are retained, not discarded. Timings include evaluation/export on a shared workstation and do not establish an isolated speedup. The main score still averages scheduled slots with equal category weight. No quad-cost term, area constraint or new normal weight was introduced.

## Acceptance and remaining work

Accept the combined guard/search change, exact runtime compaction and diagnostics. Keep the lock-relaxed objective explicitly experimental; its pilot SCORE did not improve. The stool and some intermediate chains still regress, and aircraft tails remain expensive. Coverage-area preservation and strict normal/attribute quality need their own explicit contracts and experiments. Do not silently fold these into SCORE v1.

Release, ASan/UBSan, GCC scalar, shared C ABI and installed C/C++ consumer checks are recorded in [VALIDATION.md](../../research/VALIDATION.md). New contracts cover overshared edges on a manifold handle, strided exact comparison, runtime slot/mesh mapping, proposal budget use, shared-edge rasterization, tiny triangles, culling and overlap.

## Reproduction

See [PLAN.md](PLAN.md), [ablation instructions](ablations/README.md), raw [runs](../runs/) and source/binary/configuration hashes in every run. Mesh exports are ignored by Git; regenerate them from the frozen corpus before running diagnostics.

    build/release/blitz bench research/pilot.json research/round4/configs/s512-cap8.json NEW_RUN_DIRECTORY
    build/release/blitz-lod-report NEW_RUN_DIRECTORY OUTPUT-analysis.json
    build/release/blitz-tail-audit NEW_RUN_DIRECTORY ph_dead_quiver_trunk OUTPUT-dense.json
    node research/round4/make-report.mjs

Primary references: [quad-fragment merging](https://graphics.stanford.edu/papers/fragmerging/), [raster fill conventions](https://learn.microsoft.com/en-us/windows/win32/direct3d9/rasterization-rules), [meshoptimizer simplification options](https://github.com/zeux/meshoptimizer#simplification). Our proxy explicitly uses half-integer centers and is not a bit-exact emulation of Direct3D 9 or any GPU.
