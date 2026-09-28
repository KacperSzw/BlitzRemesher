# Validation record

Core source: archived v3 overlay, starting from `26ad36e`.
The frozen opaque corpus, scoring protocol, foliage source manifest and
AGENTS.md retain their pre-run SHA-256 values. No original mesh or texture was
rewritten; experiments use reproducible opaque geometry derivatives.

## Contracts and sanitizers

- Release CTest: **7/7 passed**. [Exact output](checks/release-ctest.txt).
- Debug AddressSanitizer + UndefinedBehaviorSanitizer CTest: **7/7 passed**.
  [Exact output](checks/sanitizer-ctest.txt).
- No compiler warnings in the final affected-target or sanitizer builds.
  Nix reports the expected dirty-working-tree notice before commit.
- `git diff --check` passed.

The new fixtures cover boundary sheets at several weights and vertex orders,
aggressive and curved-sheet reductions, immutable shared streams, UV chart
contractions, dynamic overlapping-component contribution, material partitions,
bounded-memory fallback, cancellation, all chain modes, proposal-budget bounds,
and deliberately nonmonotonic candidate acceptance. Settings tests exercise
partial research objects, round trips, invalid weights and unknown fields.
Existing C ABI, appearance, import, precision and render-cost tests also pass.

Commands:

```sh
nix develop . --command ctest --test-dir build/vegetation-final --output-on-failure -j2
nix develop . --command cmake -S . -B build/vegetation-sanitize -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBLITZ_SANITIZE=ON
nix develop . --command cmake --build build/vegetation-sanitize --target blitz-vegetation-tests blitz-tests blitz-precision-tests blitz-render-cost-tests blitz-c-smoke blitz-io-tests blitz-foliage-tests -j2
nix develop . --command ctest --test-dir build/vegetation-sanitize --output-on-failure -j2
```

## Mesh exports and visual gates

The [export checks](exports/) cover all four six-example matrix runs and all
four development and validation runs: **184 chains, 2,944 recorded source and
adjacent gates**, including unchanged LOD0 records. Checks verify file hashes,
completion, measured timing scope, decreasing triangle counts, valid exported
indices and the exact LOD0 accessor bindings for every shared output.

All **20 independent dense tail audits passed**: 12 candidate matrix chains,
four primary legacy controls, and four 64-proposal primary candidates.
That is **120 tail gates** against LOD0 and the previous LOD. The configured
camera set is 642 orthographic + 64 perspective, rotation seed `0xB1172031`,
8× supersampling refined to 32×. Identical render data can short-circuit an
evaluation. These tests do not assert an all-view guarantee.

Dense records in [dense/](dense/) include executable, input row, chain and run
hashes. They were generated after preset selection; settings and thresholds
were not changed after seeing them. The [measurement tables](RESULTS.md)
retain all per-chain results and area changes.

## Compatibility and research scope

The preserved pre-change executable and the new executable with unchanged
settings agree exactly on all eight frozen opaque pilot meshes in both modes:
owned geometry hashes, all output attribute hashes and every LOD measurement.
See [default replay](checks/default-replay.json). The C descriptor and C ABI
version remain unchanged. C++ clients must rebuild against the updated header.

The candidate improves aggregate chain and tail retention at the same budget
on the six-model pilot, 28-model development set and frozen 12-model validation
set. It also improves both opaque pilot scores. Individual regressions and
longer bake times are kept visible. The default policy remains unchanged;
the measured geometry-only preset is explicitly opt-in.

Three one-worker timing repeats verify identical output geometry. The table
reports every sample, median and range. Other activity on the shared workstation
was not controlled. All owned bake/audit/build batches finished before timing
repeats began; the first repeat also includes one-worker quality observations
at budgets 32/64. These extra observations are not three-repeat estimates.
There are 94 complete vegetation run groups containing 380 chains, plus 48
opaque compatibility/preset chains. No final group is incomplete. Maximum
reported vegetation process peak RSS is 155,040 KiB; all owned batches stayed
within four workers and the 24 GiB combined memory budget.

meshoptimizer source was verified at
`9e1f07b159d3cb777f1c67ed31fc11fd117986f4`. Its 36 primary-case runs are a separate
research baseline. No dependency or unsupported attribute dropping was added
to the shipped library.

## Board

The [browser check](board/browser-checks.json) passes for 20 chains and 160 tiles.
Checks cover actual mesh triangle counts, measured seconds, shared/new/mixed
buffer labels, comparison and dense-audit text, all three rendering modes,
target scale, runtime duplicate compaction, inspector selection/rotation,
desktop and 390 px mobile layout. There were zero external requests or browser
errors. The fern/tree screenshots were visually inspected.

The main board contains nature LOD chains, with no separate nature source
gallery. Its first four opaque rows are marked as archived examples. Coverage
change is displayed separately from distance acceptance. Opacity and shading
are not audited in this round, and quad/tiny-triangle statistics are CPU
geometry proxies rather than measured GPU time.
