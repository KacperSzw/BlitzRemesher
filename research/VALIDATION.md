# Validation record

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
