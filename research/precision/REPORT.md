# RGBA8 and precision measurements

Recorded 2026-09-28, protocol v2, on the pinned Nix shell with Clang 21.1.8
and an Intel Core i7-9700K. See the [frozen plan](PLAN.md),
[machine-readable summary](summary.json), [raw runs](../runs/precision-v2/),
and [per-executable build stamps](builds/).

These measurements describe the archived experimental implementation.
Production now retains only packed coverage and double reducer state, with
float32 positions. See the [archive/restoration instructions](ARCHIVE.md),
[UNORM16 follow-up](unorm16/REPORT.md), and [production replay](production/REPORT.md).

All 46 planned batches completed: 392 asset runs, including three explicit
CGAL failures. The summary checked 6,224 delivered source/adjacent gate
records, including LOD0 and repeated runs. All 74 full/packed paired asset-run
comparisons have identical geometry, attributes and result records.

## Implemented defaults

- Owned and borrowed vertex colors are linear RGBA8: four bytes, including
  alpha, with 256 values per channel. This reduces each stored color from
  16 to four bytes. Float and normalized unsigned-16 file inputs round once
  at import; nonfinite and out-of-range values fail import. Rebuild rounds
  interpolated RGB when storing a vertex and preserves the retained alpha;
  reuse preserves the supplied bytes. Raster interpolation remains floating
  point. Both PLY and glTF export byte colors.
- C ABI and shared-library compatibility version are 2. C++ callers use
  `ColorRGBA8`; C callers use `blitz_color_rgba8`. ABI 1 descriptors are rejected.
- Coverage evaluation uses one bit per sample and skips unused depth and
  attribute work. It shares projection, clipping, culling, conservative edge
  tests and distance-transform arithmetic with the full raster reference.
  Normal and attribute profiles continue using the full raster.
- The measured source provided independent float-quadric and float-cost build
  experiments, both **off by default**. Their switches are now archived.
  Quadric solves and cost evaluation
  retain double intermediates. No determinant tolerance, placement guard,
  search budget or visual threshold was relaxed.

The color contract changes canonical input attributes, so this work starts
protocol v2 and reruns all five external baselines. Historical v1 SCOREs are
not comparison baselines. No speedup is attributed to the color change alone:
there is no matched pre-RGBA8 timing experiment here.

## Representation sizes

| Stored data | Reference | Compact representation |
|---|---:|---:|
| Vertex RGBA | 16 bytes | 4 bytes |
| Coverage sample | 36-byte full pixel | 1 bit |
| Projected vertex for coverage | 56 bytes | 24 bytes |
| Persistent quadric | 80 bytes | 40 bytes, experimental |
| Collapse candidate | 32 bytes | 24 bytes, experimental |

These are record sizes, not whole-process memory reductions. At 512 pixels
and 4× supersampling, the two padded coverage grids contain 8,652,800 samples:
297.07 MiB as full pixels versus 1.03 MiB as masks. Projection storage,
distance fields, mesh copies, reducer arrays and allocator overhead remain.

## Workloads and timing method

Five frozen executables use the same source tree:
`e125d42d7d199d1cb90a8cda13ae098f69500ea8729ee97e5995cb5a2d57f9f4`.
Every benchmark checks its executable hash against its own stamp. The variants
are full raster/double state, packed/double state, packed/float quadrics,
packed/float costs, and packed/both float options.

The pilot contains eight development assets, two per category, using the
existing 32→16, eight-level direct coverage configuration. The validation set
contains all 20 validation assets under that same configuration. Search uses
6 orthographic + 2 perspective cameras; audit uses 12 + 4. The normal and
attribute profiles are separate eight-asset scenarios.

The larger-screen workload contains only the first two pilot assets, using
the existing 512→16, eight-level hybrid configuration with an 8-pixel source
cap. It also enables pruning, coupled wedges and the available SIMD backend.
It is a separate scenario; its scores do not establish a gain over the pilot.

Repeat 1 timings are exploratory and can overlap compilation. Timing tables
use all three complete repeats 2–4. Within each pilot repetition, variants
are interleaved full, packed, quadrics, costs, both; repeat 3 reverses this
order. The larger-screen pairs likewise reverse order for repeat 3.
Several single-thread batches share the workstation, with at most four CPU
workers and less than 24 GiB combined memory. These are elapsed times under
shared load, without CPU pinning; small differences should not be interpreted
as isolated CPU throughput gains. Raw values and ranges are retained.
Unrelated compilation and interactive workloads were observed during the
repetitions; their processes were left untouched.

Generation time includes reduction, search and audit. Stage timers partition
reduction/proposer, rasterization, and distance/attribute measurement work;
other chain work accounts for the remainder. Whole-run time also includes
import and export. RSS is the process high-water mark, not a sum of per-asset
peaks; external baseline child-process memory is not included in the runner's
RSS. No GPU timing is measured.

## Results

### Complete-chain pilot timings

Three complete repetitions per row, eight development assets each. Stage
medians are calculated separately and need not sum to the generation median.

| Variant | Generation median (range), seconds | Reduction | Raster | Distance | Peak RSS median |
|---|---:|---:|---:|---:|---:|
| Full raster, double state | 144.67 (136.07–221.23) | 40.25 s | 95.49 s | 8.10 s | 296.46 MiB |
| Packed, double state | 89.99 (85.98–110.50) | 41.60 s | 42.72 s | 4.79 s | 296.48 MiB |
| Packed, float quadrics | 94.16 (81.40–100.06) | 43.37 s | 44.87 s | 5.09 s | 285.14 MiB |
| Packed, float costs | 87.86 (85.43–131.13) | 43.88 s | 39.57 s | 5.07 s | 296.91 MiB |
| Packed, both float options | 85.04 (76.11–131.73) | 40.02 s | 39.31 s | 5.04 s | 285.12 MiB |

Packed/double generation is **1.61× faster by median** than full raster,
with identical results. The narrow reducer variants have median speed ratios
of 0.96×–1.06× relative to packed/double, with overlapping ranges and changed
work for float quadrics. These timings do not establish a clear reducer
throughput gain. Pilot peak RSS is dominated by mesh/reducer storage; packing
small coverage grids barely changes the process high-water mark.

### Larger-screen timings

Three complete repetitions per row, two development assets each, under the
separate 512→16 hybrid configuration.

| Variant | Generation median (range), seconds | Reduction | Raster | Distance | Peak RSS median |
|---|---:|---:|---:|---:|---:|
| Full raster, double state | 76.04 (64.63–77.39) | 0.43 s | 33.86 s | 40.62 s | 525.86 MiB |
| Packed, double state | 36.25 (35.17–36.97) | 0.42 s | 10.45 s | 25.37 s | 64.41 MiB |

Packed coverage gives **2.10× faster median generation and 87.8% lower median
peak RSS** in this scenario. Results are identical in every pair. Both paths
score 76.40993, with final retained ratio 3.43009%, last-three retained ratio
7.08362%, maximum changed coverage 35.28817%, and no failures or unreduced
fallbacks. These ratios describe this two-asset scenario only.

### Validation timing and memory observations

These are single complete runs of all 20 validation assets, including a
1.26-million-triangle source. Their timings are exploratory observations,
not repeated throughput estimates.

| Variant | Generation | Whole run | Peak RSS |
|---|---:|---:|---:|
| Full raster, double state | 1,730.96 s | 1,733.15 s | 685.78 MiB |
| Packed, double state | 1,073.55 s | 1,075.59 s | 679.72 MiB |
| Packed, float quadrics | 1,335.67 s | 1,337.89 s | 658.07 MiB |
| Packed, float costs | 1,314.57 s | 1,316.64 s | 676.14 MiB |
| Packed, both float options | 1,342.63 s | 1,344.63 s | 658.45 MiB |

Float quadrics reduce observed validation peak RSS by about 3.2% relative
to packed/double. Their 50% record-size reduction is a much smaller reduction
of the full working set. Float costs save about 0.5% in this observation.

### CPU kernel fixture

The procedural fixture has 9,409 vertices and 18,432 triangles, with colors,
normals and UVs. It renders a source/translated pair at 512 pixels and 4×
supersampling. One warmup precedes seven measured samples, alternating full
and packed order. Both paths return exactly the same 1-pixel distance.

| Kernel, packed/double executable | Full median | Packed median | Ratio |
|---|---:|---:|---:|
| Rasterize the pair | 115.83 ms | 11.05 ms | 10.48× |
| Coverage distance | 285.53 ms | 219.19 ms | 1.30× |

Across all five executables, paired raster speed ratios range from 8.57× to
10.48×. These are CPU kernel measurements, not complete-chain speedups or
GPU timings. Raw samples are in [micro-packed.json](micro-packed.json) and
the other `micro-*.json` files, with corresponding executable stamps.

The independent reduction fixture reaches 2,048 triangles in every variant.
Median times are 35.23 ms for packed/double, 35.84 ms for float quadrics,
33.41 ms for float costs and 35.09 ms for both. Samples overlap; this small
fixture does not demonstrate a large speedup from narrower reducer records.
Float quadric output hashes differ, while float costs preserve the reference
hash. This fixture has no visual SCORE.

### Retention and coverage

Ratios below are percentages of source triangles, averaged per category in
the same way as SCORE. Higher SCORE means fewer retained triangles. Changed
coverage is the maximum delivered-view XOR/union fraction; it is a diagnostic,
not the acceptance threshold or a perceptual similarity score.

| Scenario / state | SCORE | Final retained | Last three retained | Max changed coverage |
|---|---:|---:|---:|---:|
| Pilot, packed/double or float costs | 62.41592 | 37.10587% | 37.24313% | 59.53488% |
| Pilot, float quadrics or both | 62.39239 | 37.12939% | 37.26666% | 59.53488% |
| Validation, packed/double or float costs | 78.38101 | 21.48119% | 21.48119% | 53.25521% |
| Validation, float quadrics or both | 78.36833 | 21.48331% | 21.48331% | 50.98157% |
| Normals, every state | 3.87804 | 92.17408% | 93.68333% | 1.57068% |
| Attributes, every state | 3.87804 | 92.17408% | 93.68333% | 1.57068% |

The coverage-only contract can accept substantial area differences. The
float experiments passing its gates is evidence about those frozen sampled
checks, not evidence of pixel-identical images. RGBA8 versus the previous
float-color source has not received a separate visual comparison.
Its import quantization error is bounded by 1/510 per channel in linear
units; that representation bound is not a perceptual validation result.

### Fresh external baselines

All baselines use the v2 pilot's canonical inputs, coverage gates and fixed
work budgets. External adapters expose geometry-only coverage capabilities;
they do not establish normal/color/UV preservation results.

| Method | SCORE | Final retained | Last three retained | Max changed coverage | Failed / unreduced |
|---|---:|---:|---:|---:|---:|
| Native packed/double | 62.41592 | 37.10587% | 37.24313% | 59.53488% | 0 / 0 |
| meshoptimizer | 81.10080 | 16.30308% | 16.47328% | 44.69274% | 0 / 1 |
| Fast Quadric | 82.76084 | 11.97191% | 13.10046% | 68.72964% | 0 / 0 |
| CGAL Lindstrom–Turk | 81.06412 | 16.74342% | 17.02694% | 33.33333% | 1 / 1 |
| CGAL QEM | 84.24260 | 14.79056% | 14.84715% | 77.35680% | 1 / 1 |
| CGAL probabilistic | 84.27547 | 14.78869% | 14.99572% | 88.60963% | 1 / 1 |

An unreduced count includes failed assets. Each CGAL policy rejects Bell X-1
because the adapter requires oriented manifold connectivity; it performs no
silent repair. Each failure contributes a retained ratio of one, including
to both tail ratios. Logs are retained in [logs/](logs/) and failed rows in
the raw run directories. The precision work does not close the native
reducer's retention gap under this configuration.

Whole-run times are 73.63 s for meshoptimizer, 80.99 s for Fast Quadric,
180.46 s for CGAL LT, 135.57 s for CGAL QEM and 263.52 s for CGAL probabilistic.
These are single shared-load observations including adapter process launches
and PLY interchange, not standalone simplifier timings. The summary retains
their parent-process RSS; it does not measure child-process peak memory.

## Correctness and numerical behavior

Every delivered LOD is checked against both its predecessor and source.
The summary independently checks delivered gate records and matches full
and packed output hashes, attribute hashes and complete result JSON. It also
checks canonical input attribute hashes across precision variants.

The float-cost pilot and validation outputs match the packed/double reference.
Float quadrics change seven of eight pilot outputs and all 20 validation
outputs. Their pilot SCORE falls from 62.41592 to 62.39239, and validation
SCORE from 78.38101 to 78.36833. Both-float results have the same aggregate
scores as float quadrics. Passing the sampled gates permits these different
collapse choices; it does not make them an improvement.

All native pilot and validation runs have zero failed assets and zero
unreduced fallbacks. Normal and attribute profiles each retain two unchanged
chains and SCORE 3.87804 across the five variants. Those fallbacks remain in
the denominator. Float quadrics change some accepted outputs in these
profiles even though the aggregate triangle ratios match.

No nonfinite solve or candidate-cost event occurred in the completed native
pilot or validation runs. Position fallbacks still occur normally when a
solve is singular or its candidate fails the placement/cost guard. In the
validation run, packed/double records 20,824,914 such fallbacks; float quadrics
records 41,472,430. Singular solve counts are 18,112,267 and 18,891,006,
respectively. Smaller persistent coefficients measurably affect the solver's
behavior despite double intermediates.

## Verification

- All five Clang release configurations: 5/5 CTest targets each.
- Combined float options with ASan+UBSan: 5/5 targets; 402,990 bounded fuzz
  executions in 61 seconds, seed 2971082790, maximum input 4,096 bytes.
- GCC scalar and shared-library configurations: 4/4 targets each.
- Separately installed C and C++ consumers: built and ran against ABI 2.
- Contracts cover all 256 color values, rounding boundaries, invalid imports,
  normalized 16-bit and float colors, alpha, byte-aligned strided reuse,
  PLY/glTF round trips and ABI migration. Raster tests compare every coverage
  sample and use an independent brute-force distance reference. Reducer tests
  cover planar and nearly planar surfaces and scales from 1e-12 to 1e12.

See [the validation record](../VALIDATION.md) for scope and earlier checks.
The full 642 + 64 camera defaults and held-out corpus were not run here.
These frozen small-camera experiments do not establish all-view guarantees.

## Decision and follow-up scope

Keep RGBA8, packed coverage and double quadrics/costs as the single production
precision path. Remove the float research switches from production and retain
their measurements and restoration patch. Float quadrics affect
collapse choices and slightly worsen retention, while the small timing
differences are sensitive to shared-machine load.

Positions, normals, UVs and tangents retain their existing float streams.
The subsequent [UNORM16 probe](unorm16/REPORT.md) found a 20–26% reducer CPU
penalty and appearance/degeneracy failures, so positions stay float32.
Packed normal/UV encodings remain outside this change. Reuse retains its
byte-exact borrowed-data contract. The evidence supports reducing unused
raster state and does not establish a general speed multiplier from halving
vertex precision.

## Reproduction

To reproduce the historical variants, restore the archived source in a
separate checkout using [ARCHIVE.md](ARCHIVE.md). Its source hash matches the
measured implementation. Current production commands are shown below and
write separate production stamps and run directories.

Run commands inside `nix develop`. Each benchmark call checkpoints at 50
minutes and resumes only with matching input, config, protocol and binary
hashes. Keep the aggregate four-worker limit when using concurrent shells.

```sh
bash tools/precision-research.sh build production
bash tools/precision-research.sh bench production pilot 1
bash tools/precision-research.sh bench production large 1
bash research/precision/verify-production.sh
bash tools/precision-research.sh micro production
c++ -std=c++20 -O2 research/precision/summarize.cpp -o build/precision-summary
build/precision-summary research/runs/precision-v2 > research/precision/summary.json
```

Production rejects the archived `full`, `quadrics`, `costs` and `both` variants.
Use `normals` and `attributes` for the separate profiles. External baselines use
`baseline-meshopt`, `baseline-fastquadric`, `baseline-cgal-lt`,
`baseline-cgal-qem` and `baseline-cgal-probabilistic`, with the frozen adapters
in `build/precision-baselines`. Preserve existing stamps when overlapping
jobs; rebuilding an executable requires a new run identity.
