# Reproduce the storage experiment

Run from the repository root, inside `nix develop`. The corpus cache and frozen
development manifest are required. One benchmark worker per invocation; use at
most four workers and 24 GiB combined. Runs checkpoint within 50 minutes.

Build and verify the current source:

```sh
cmake --preset release
cmake --build --preset release
ctest --preset release
cmake --preset sanitize
cmake --build --preset sanitize
ctest --preset sanitize
node tools/chain-search-audit-check.mjs research/density/audit-check.json
```

`builds/source-v1.tar.gz` through `source-v5.tar.gz` retain exact experiment
sources; `source-final.tar.gz` also includes the final regression. Extract a
chosen archive into an isolated checkout of checkpoint `636d139`, configure
and build there, then copy its CLI to `build/density/VERSION/blitz`. Check the
binary against the corresponding stamp before replaying a run. A different
compiler/build must use a new stamp and run directory; never overwrite a
recorded cohort or bypass its hash checks.

The recorded v5 commands were:

```sh
node tools/density-run.mjs v5 shared stool
node tools/density-run.mjs v5 shared trunk
node tools/density-run.mjs v5 shared
```

Stages `screen`, `ordering`, `attributes`, `merged`, `merged-attributes`, and
`shared` use separate saved configurations. Smokes enable proposal traces and
remain unscored observations. v1's uncoupled attributes smoke used
`configs/attributes-uncoupled-smoke.json` directly with the benchmark CLI.

The dense v4 audit used two workers. v5 shares its proofs only after verifying
byte-identical exports and complete contract/provenance equality:

```sh
node tools/chain-search-audit.mjs density-v4-shared 0 2 research/density 0xA1172028
node tools/chain-search-audit.mjs density-v4-shared 1 2 research/density 0xA1172028
node tools/appearance-audit-share.mjs --experiment=research/density --seed=0xA1172028 density-v4-shared density-v5-shared
```

The two audit commands may run concurrently. The proof-sharing command follows
their completion. The full audit resumes only matching hashes and retains raw
interrupted checkpoints; rejection witnesses count as resolved failures.

Regenerate the measurements and offline board from local exports:

```sh
node tools/density-smoke.mjs
node tools/appearance-report.mjs research/density appearance-screen:screen appearance-ordering:ordering density-v3-merged:merged density-v5-shared:shared
node tools/chain-search-board.mjs research/density examples/density-board
nix shell nixpkgs#gnuplot -c node tools/chain-search-plot.mjs research/density appearance
node tools/chain-search-board-check.mjs examples/density-board research/density
```

The browser check accepts `BLITZ_PLAYWRIGHT_MODULE` and `BLITZ_CHROMIUM` for local
installations. The board embeds compressed geometry and opens without a server.
Its raw glTF/buffer links address generated local exports, which are excluded
from Git. Raw measurements, source/build stamps, proof records, progress data,
and the self-contained board are versioned.
