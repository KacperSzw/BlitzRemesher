# Exact coverage audit caching

The frozen production automatic coverage pilot completed five paired repetitions. Every output stream, audit record, rejection count, candidate selection, and numerical counter agrees with uncached generation. Cache-disabled output also agrees with the pre-change executable on all eight assets.

Category-balanced geometric mean generation time decreased **40.35%**. Median total generation time was **202.33 → 121.83 seconds**. The largest per-asset time ratio was 0.68. The 25% improvement / 5% maximum slowdown pilot gate **passed**. Peak charged cache storage was 255.31 MiB, within the 256 MiB allowance.

| Asset | Uncached median s | Cached median s | Time reduction | Distance fields before → after | Cache peak MiB |
|---|---:|---:|---:|---:|---:|
| ph_dead_quiver_trunk | 32.81 | 19.45 | 40.72% | 4154 → 2099 | 252.95 |
| ph_grass_bermuda_01 | 6.58 | 4.21 | 36.06% | 3848 → 1595 | 207.54 |
| ph_metal_stool_02 | 28.82 | 15.37 | 46.66% | 3562 → 1698 | 252.36 |
| ph_moon_rock_02 | 26.62 | 15.51 | 41.75% | 4442 → 2006 | 249.01 |
| ph_painted_wooden_shelves | 13.61 | 7.26 | 46.66% | 3030 → 1407 | 239.38 |
| ph_rock_face_02 | 26.10 | 14.54 | 44.28% | 4572 → 1889 | 249.01 |
| si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652 | 26.06 | 17.80 | 31.68% | 2336 → 1227 | 255.31 |
| si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7 | 41.31 | 27.70 | 32.95% | 4040 → 1845 | 255.01 |

Per pilot run, raster builds decreased 30880 → 13479 and distance-field builds decreased 29984 → 13766. Outputs and cache-work counters repeat exactly across all five pairs.

Measured with Clang 21.1.8, AVX2, 1 benchmark worker, on Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz. The pinned Nix environment and complete settings are recorded in each run's metadata.

## Contract and measurement

The scenario uses eight scheduled levels from 512 to 16 pixels, source cap 3 px, adjacent cap 2 px, coverage-area cap 0.5, eight proposals, beam width two, automatic storage and a 5% triangle allowance. Search cameras are 6+2 at 2×; audit cameras are 12+4 at 4×, refining to 8×. It is a small-camera development scenario, not the default quality audit. SCORE remains 75.0434 in both variants; the improvement is bake time.

Five complete paired repetitions, one benchmark process at a time. Order reverses each repetition. Shared workstation; no CPU pinning. No repeat excluded. Raw per-asset times and ranges are in [pilot-analysis.json](pilot-analysis.json). The existing benchmark records process RSS as a process high-water mark, not per-asset live memory. Cache accounting includes requested entry-array bytes and retained vector capacities, including overlapping arrays during growth; allocator bookkeeping is reflected only in process RSS. The 256 MiB allowance is additional cache storage, not a whole-process memory limit.

## Validation and dense views

All 20 frozen validation assets have exact cached/uncached agreement, with zero failures and zero unchanged-chain fallbacks. Cached validation peaked at 255.90 MiB of charged cache and 867.41 MiB process RSS. [Comparison](validation-agreement.json). Validation used two concurrent workers; its timings do not enter the pilot speed claim.

Independent tail checks use 642+64 cameras, rotation seed 3665436710, 8× sampling and refinement to 32× on LOD5–7. Cached and uncached measurements agree exactly. These are finite camera checks, not an all-view bound. Dense timings below are single pairs, separate from repeated generation timings.

| Asset | Audit quality | Uncached s | Cached s | Cache peak MiB |
|---|---|---:|---:|---:|
| ph_painted_wooden_shelves | pass | 17.87 | 15.64 | 255.98 |
| ph_dead_quiver_trunk | pass | 14.74 | 13.51 | 255.98 |

## Regression checks and default

The corrected implementation passed all 9 release and 9 ASan/UBSan CTest targets before the timing experiment. Tests exercise bounded admission, cache lifetimes, disabled/tiny/full allowances, refinement, cancellation, allocation failure and exact measurements. A mutable-callback regression rejected the first implementation; its raw runs remain in [v1](v1/README.md) and are excluded from this result.

After the gates passed, coverage caching was enabled by default with a 256 MiB allowance. All 8 pilot assets were replayed using a configuration that omits the cache key: results and cache counters exactly match the explicitly enabled measured binary. [Default replay](production-agreement.json). The final default-enabled source passed all 9 [release](checks/release-final-ctest.log) and 9 [ASan/UBSan](checks/sanitize-final-ctest.log) targets.

Set `research.coverage_cache_mib` to an integer from 0 to 256 in C++/CLI settings; zero disables caching. Only coverage-profile generation uses the cache. C ABI 4 is preserved; C++ clients must rebuild for the changed settings and statistics layouts. This experiment establishes speed for the stated coverage scenario; normal/attribute and full default-camera speed are unmeasured.

## Reproduction and provenance

Run `node research/audit-cache/run.mjs smoke`, then `pilot`, `validation`, `dense`, `production`, and `report`. The runner requires the frozen executables named in its source and validates exact agreement before reporting performance. Every batch has a 50-minute limit; incomplete batches receive no aggregate comparison. No held-out assets were used.

The baseline is Git revision 5c0c62386b212f573d2cae07bf8e33f73dfa9052. [Build stamps](builds/) identify the executables. `builds/candidate-source.tar.gz` contains the measured source snapshot; `builds/production-source.tar.gz` contains the final default-enabled source. Overlay each archive onto a separate checkout of the baseline revision, which supplies the unchanged vendored dependencies, then build with the pinned Nix shell and CMake Release. The two experiment configurations differ only in `research.coverage_cache_mib`. [Raw runs](../runs/audit-cache/) retain all rows, summaries and metadata.
