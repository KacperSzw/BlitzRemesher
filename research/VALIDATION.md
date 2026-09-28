# Validation record

## Round 4 additions

The larger-screen/tail work passed 4/4 release CTest targets, 4/4
ASan+UBSan targets, 3/3 GCC scalar targets and 3/3 shared-library targets.
The separately installed C and C++ consumers built and ran successfully,
including the additive runtime-LOD queries and render-cost header/symbols.
The sanitizer reducer fuzz campaign completed 411,473 executions in 61
seconds (seed 3399045530, maximum input 4096 bytes), with all four objective
modes exercised and no sanitizer failure. Build logs and corpus are under
the ignored build directory.

New contracts check common-neighbor topology guards on a manifold handle,
exact strided-stream equality, immutable reuse, runtime slot mapping and
glTF mesh sharing, distinct root proposals within test-owned work budgets,
quad occupancy, shared-edge fill, zero-sample triangles, culling and overlap.
Diagnostics remain separate from acceptance and SCORE.
The final suites also exercise both unchanged and changed borrowed proposals
from a rebuilt predecessor, whose compact vertex IDs differ from LOD0.

The last three levels of the new Quiver tree chain also passed all six
source/adjacent checks using 642 orthographic and 64 perspective cameras,
8× supersampling with refinement up to 32×. This additional check covers
one example's tail, not the full corpus. The aggressive result still loses
substantial filled area under a coverage-distance-only contract.
See [the final-build raw check](round4/dense-tree-final.json) and
[Round 4 results](round4/REPORT.md) for the complete experiment matrix and
frozen 20-asset validation comparison.

All 22 Round 4 batches completed: 19 eight-asset development runs and three
20-asset validation runs, including archived prototypes and final replays.
Every batch had zero failed assets and zero unchanged-chain fallbacks.
The final implementation preserved all position/index and attribute hashes
from the prototype in seven scenarios and the validation cohort.
The final validation SCORE is 86.47595 against the matched baseline 80.30050;
these runs use the small-camera contract described in the Round 4 report.

The final offline board passed Chromium checks for all 32 geometry tiles,
runtime compaction, display modes, target scale, source identity, selection,
rotation and mobile overflow. It made zero network requests and had no
browser errors. See [browser-checks.json](round4/board/browser-checks.json).

## Initial implementation

Recorded 2026-09-28 on NixOS, Intel Core i7-9700K, using the pinned flake.
These checks validate the first research implementation, not production
readiness or an all-view visual guarantee.

| Configuration | Result |
|---|---|
| Clang 21.1.8 release, tools and AVX2 enabled | 3/3 CTest targets passed |
| Clang 21.1.8 AddressSanitizer + UndefinedBehaviorSanitizer | 3/3 CTest targets passed |
| Clang shared library, tools disabled | 2/2 CTest targets passed |
| Installed shared package, separate C and C++ CMake consumers | Both configured, linked and executed successfully |
| GCC 15.2.0, tools and AVX2 disabled | 2/2 CTest targets passed |

The contract suites cover borrowed strided streams, immutable vertex reuse,
source and adjacent LOD gates, screen-size budgets, zero attribute weights,
normal/color/material metrics, empty raster distances, an independent
brute-force distance reference, scalar/SIMD agreement, degenerate surfaces,
coupled attribute wedges, cancellation, C ABI ownership/version errors,
import transforms and glTF accessor/LOD manifest export.

The instrumented reducer passed bounded libFuzzer campaigns of 681,605,
596,246 and 515,929 executions. The final campaign lasted 61 seconds, used
seed 2971082790 and a 1,024-byte maximum input. A discovered degenerate-face
budget regression has a permanent seed and contract test. Generated fuzz
corpora and build logs remain under the ignored build directory.

The frozen corpus contains 120 downloaded, importable, checksummed meshes:
30 rocks, 30 organics (12 woody), 40 manufactured objects and 20 complex
scans. Selected source files total 636,827,730 bytes. Grouped splits contain
80 development, 20 validation and 20 held-out assets. Provenance, per-file
hashes and rights records are in [corpus.json](corpus.json) and
[scan-rights.json](scan-rights.json).

All 120 assets completed the progressive coverage smoke benchmark with
zero failures or unchanged-chain fallbacks. SCORE was 90.7764. This is the
32→16 pixel, 12+4 audit-camera scenario, not the default 642+64-camera quality
preset. The full default-preset corpus audit remains future work.

Across the original QEM pilot and three performance repeats, all eight
assets retained identical output hashes and SCORE. Scalar/AVX2 checksums
also matched in the fixed distance-transform microbenchmark. Raw rows,
hash scope, source/build/configuration/camera hashes, timings and process
memory measurements are retained in [runs](runs/).

Strict normal/attribute profiles currently reduce poorly: the coupled
eight-asset scenario scored 5.76 with three unchanged-chain fallbacks.
Texture images, opacity, skinning and morph targets are outside this
version's contract. Finite camera/sample audits and a bounded candidate
search cannot establish universal visibility bounds or global optimality.

Reproduce the main build and sanitizer checks using the commands in
[README.md](../README.md). The independent installed-package consumer is
in [tests/consumer](../tests/consumer/); configure it with
CMAKE_PREFIX_PATH pointing at the installation prefix. Benchmark replay
requires the exact recorded configuration and a fresh output directory
when source or executable hashes differ.
