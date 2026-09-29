# Whole-chain reduction

[Open the offline board](../../examples/reduction-board/index.html),
[measurements](REPORT.md), [decision](DECISION.md), and [frozen plan](PLAN.md).
The old thirteen-model board remains a separate gallery.

## Try the search

```sh
build/release/blitz simplify model.glb \
  --config research/chain-search/configs/coverage-graph-2.json --out output/model
```

`graph-2` adds one bounded graph pass. `graph-4` adds three. The names describe
nominal proposal tiers; seed retries, transition audits and elapsed time are
reported separately. These development presets use **12+4 audit cameras**,
not the default 642+64 quality audit. Appearance and strict presets are separate
visual contracts. Production defaults remain unchanged pending qualification.

The output contains geometry, shared or owned vertex streams, primitive
material IDs, scheduled thresholds, exact consecutive runtime deduplication,
per-mesh/cumulative bytes, configured source/step audits and the audit contract.
The engine keeps its materials and textures. Texture images and normal maps
are unscored. Export validity and independent visual qualification are separate.

## Reproduce the experiment

The baseline is revision `2669dba9db01fc0af69f66c287e33ff41f529515`.
Build it in a separate checkout. Copy its executable to
`build/chain-search/baseline/blitz` and stamp that source checkout and executable
with `tools/stamp-build.sh`. Build this implementation, freeze its executable
at `build/chain-search/v3/blitz`, and stamp it in
`research/chain-search/builds/candidate-v3.json`. The recorded stamps identify
the exact local frozen binaries; rebuilt executables require fresh run IDs.
The exact v3 source is archived in `builds/source-v3.tar.gz`
(SHA-256 `8daf499016faa76e0cd7e2f528b1b02ddff07dc11d7fdd01d3c52d1c96305850`).
The baseline source is available at its Git revision. Frozen binaries are local
build artifacts, excluded from Git. Later corrections to uncapped objective
labels and audit-tool completion handling are documented in PLAN.md.

```sh
build/release/blitz-corpus data research/pilot.json
node tools/chain-search-run.mjs v3 coverage baseline
node tools/chain-search-run.mjs v3 coverage legacy-2
node tools/chain-search-run.mjs v3 coverage legacy-4
node tools/chain-search-run.mjs v3 coverage graph-2
node tools/chain-search-run.mjs v3 coverage graph-4
node tools/chain-search-run.mjs v3 coverage topology-4
```

Repeat baseline/graph-2 with `appearance` and `strict`. Restore validation using
`research/chain-search/validation.json` (the original twenty entries), then add
`validation` to the run command. Held-out assets are reserved for release audits.
At most four workers and 24 GiB combined memory. Every batch checkpoints at
50 minutes; resume requires matching input, configuration, protocol and build.
Small smoke outputs may contain the runner's single-asset summary; those are
not pilot scores and are excluded from the comparative report.

```sh
node tools/chain-search-audit.mjs chain-search-v3-coverage-graph-2
node tools/chain-search-report.mjs v3
node tools/chain-search-decision.mjs
gnuplot --version
node tools/chain-search-plot.mjs
node tools/chain-search-board.mjs
BLITZ_PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs \
  BLITZ_CHROMIUM=/path/to/chromium node tools/chain-search-board-check.mjs
```

Independent audits use all seven reduced levels, 642+64 cameras rotated with
seed `0xA1172026`, and 8× coverage sampling refined to 32×. They preserve failed
views and per-level checkpoints. An optional start index and stride partition
the audit runner across workers. Each record hashes the source run, row, glTF,
buffer and auditing executable. The board refuses mismatched audit exports.
Completed comparisons can resume from a partial audit after provenance checks;
old checkpoints and their per-level auditing executable hashes are retained.
Run `node tools/chain-search-timing.mjs v3` for three sequential, alternating
pairs of the complete frozen pilot. It verifies every output attribute hash.

The board verifies exported buffer sizes, runtime mappings and triangle counts
with `blitz-chain-inspect`; previews decode the exact exported accessor bytes.
Configured worst-view buttons use the evaluator's actual camera transforms.
All geometry is embedded for offline loading. Raw exports and research links
resolve within the repository. `preview.png` and `progress.svg` are shareable
standalone artifacts. `progress.csv` and the gnuplot source preserve plot inputs.

## What the implementation changes

The graph retains several source-admissible geometries at each scheduled size,
then independently checks transitions from alternative predecessors. It keeps
nondominated cost histories only when they end at identical geometry. Both
source and incumbent remain available through the search. No triangle target
is treated as a monotone oracle for pixel error.

This improves how existing quadric/endpoint proposals are combined. It does
not introduce image-error-guided collapse placement. Appearance failures and
dense-camera failures remain evidence about that limitation, not reasons to
relax the frozen limits.
