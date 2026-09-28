# Reproduce the vegetation experiments

Use the pinned Nix development shell. The production library remains C++20;
Node scripts only schedule, verify and report C++ bakes. The timing driver uses
GNU time at the shared NixOS workstation path and records process peak RSS.

```sh
nix develop . --command cmake -S . -B build/vegetation-final -G Ninja -DCMAKE_BUILD_TYPE=Release
nix develop . --command cmake --build build/vegetation-final -j2
nix develop . --command ctest --test-dir build/vegetation-final --output-on-failure
build/vegetation-final/blitz-corpus foliage-replay research/foliage/manifest.json data/foliage
build/vegetation-final/blitz-corpus foliage-geometry research/foliage/manifest.json data/foliage build/vegetation-geometry
```

Copy a plan JSON to a fresh filename, change its `name`, then run:

```sh
node tools/vegetation-run.mjs NEW_PLAN.json build/vegetation-final/blitz build/vegetation-geometry
node tools/vegetation-report.mjs
```

Run `development.json` before `validation.json`; do not retune the preset on
validation results. The six-example matrix is already exposed development
data. The final matched runs use the v3 source overlay archived in this folder.

Recorded plan files are immutable. `matrix.json` is the six-example comparison;
`quality.json` separates 32/64-proposal runs; `development.json` and
`validation.json` apply the frozen setting to grouped families. Every run keeps
its exact configurations, source hashes, binary/compiler identity, timing scope,
per-asset measurements and completion status. Changed inputs require a new run
name. The driver stops at 50 minutes and resumes matching completed rows only.
Keep total concurrent research workers at four or fewer and combined RAM below
24 GiB. Incomplete runs receive no aggregate retention result or SCORE.

`pilot.json` and `combined.json` are historical screening stages. Their timings
describe those prototypes. Final v3 comparisons add the independent-chart UV
orientation guard and use packed masks for the experimental component proposer.
The archived v3 source overlay preserves the measured core; defaults still use
the original proposal policy. Never compare different budgets as algorithm wins.

The external adapter is built with `BLITZ_RESEARCH=ON` and the pinned
meshoptimizer source. Run `meshopt.json` with `blitz-vegetation-meshopt` as the
binary argument. It preserves geometry/UV/material semantics and explicitly
rejects unsupported normal/color/tangent streams rather than dropping them.

For any complete run:

```sh
node tools/check-foliage-chains.mjs RUN_DIRECTORY CHECK.json
build/vegetation-final/blitz-lod-report RUN_DIRECTORY COST_REPORT.json
build/vegetation-final/blitz-tail-audit RUN_DIRECTORY ASSET_ID DENSE.json 0xB1172031
```

The dense audit is a subsequent test, never a replacement for failed records.
Its `worst_view` identifies the largest distance error; `changed_area` is the
separate maximum symmetric-difference/union area across evaluated views.
Render costs count opaque geometry samples and aligned 2×2 quads, not actual
GPU time or opacity-texture overdraw.

The bounded two-worker post-check driver covers all 12 candidate board chains,
four primary controls and four 64-proposal primary candidates. It verifies
hashes before reusing a completed dense check:

```sh
node tools/vegetation-audit.mjs dense
node tools/vegetation-audit.mjs exports
node tools/vegetation-audit.mjs cost
node tools/vegetation-opaque.mjs
```

The opaque regression driver additionally needs the preserved pre-change
`build/vegetation-baseline/blitz` executable. Rebuild revision `26ad36e` in an
isolated checkout to reproduce that control with a fresh binary identity.
Do not overwrite recorded run directories with different binaries; select a
new run name for the new build. The eight-asset manifest and old configuration
files remain frozen.

For one-worker timing repeats, run `timing-r1.json`, `timing-r2.json`, and
`timing-r3.json` sequentially with the main experiment driver, after other
owned bake/audit/build jobs finish. The first pass also records single-worker
32/64-proposal observations; only the eight-proposal cases are repeated three
times and used in the median/range table. Then regenerate the tables:

```sh
node tools/vegetation-report.mjs
node tools/vegetation-summary.mjs
```

Regenerate the board after the comparisons and dense checks are complete:

```sh
node tools/vegetation-report.mjs
node tools/vegetation-board.mjs
build/vegetation-final/blitz-board . research/vegetation/board/examples.json research/vegetation/board
```

`tools/capture-board.mjs` checks offline rendering, actual triangle counts,
vertex-mode and time labels, the comparison table, dense-audit status, runtime
compaction, inspector controls and desktop/mobile layout. Its Playwright and
Chromium locations can be provided through `BLITZ_PLAYWRIGHT_MODULE` and
`BLITZ_CHROMIUM`.

The resulting HTML embeds all presentation meshes and works offline. The raw
mesh exports remain ignored and can be regenerated; source files and the frozen
opaque benchmark are never rewritten.
