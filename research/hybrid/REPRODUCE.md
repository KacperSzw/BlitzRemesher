# Reproduce automatic hybrid research

Build the committed C++20 source with the pinned Nix shell:

```sh
nix develop . --command cmake -S . -B build/hybrid -G Ninja -DCMAKE_BUILD_TYPE=Release
nix develop . --command cmake --build build/hybrid -j2
nix develop . --command ctest --test-dir build/hybrid --output-on-failure
```

The original sources and derived geometry are replayed with the procedures in
[the previous experiment](../vegetation/REPRODUCE.md). Do not replace frozen input
or output records. To rerun, copy the relevant v5 plan to a **new name**, adjust
its `name` field, then use the runner with that new plan:

```sh
node tools/vegetation-run.mjs NEW_PLAN.json build/hybrid/blitz build/vegetation-geometry
node tools/hybrid-report.mjs NEW_PLAN.json
```

The CLI consumes schema-3 configurations with the optional forced modes under
`research`. Historical top-level output/chain settings require `--legacy-config`
instead of `--config` for `simplify`; research C++ readers can explicitly call
`settings_json(json, true)`. Archived executable revisions are still needed to
reproduce historical proposal policies exactly. Do not interpret compatibility
parsing as algorithm equivalence.

Run development before validation; do not tune on validation. The v5 quality
matrix covers the primary assets at 32/64 proposals. The origin comparison keeps
the same proposal budget and audits. Memory selection sweeps use the **same
final candidate pool**, without additional bakes. The fixed 10% prefix search
envelope does not change with the requested final allowance. Extra audited
reconnections are recorded separately from reduction proposals.

For the exact pre-commit measurement overlay, start from the base revision in
`source-v5.json` and extract `source-v5.tar.gz`; verify its recorded hash. The
committed current sources contain the same measured core. Prototype v2–v4 raw
results retain their identities, but only the final v5 overlay is archived.

Current-run checks and board generation:

```sh
node tools/hybrid-audit.mjs
node tools/hybrid-opaque.mjs
node tools/hybrid-report.mjs research/hybrid/pilot-v5.json research/hybrid/quality-v5.json research/hybrid/development-v5.json research/hybrid/origins-v5.json research/hybrid/validation-v5.json
node tools/hybrid-board.mjs
build/hybrid/blitz-board . research/hybrid/board/examples.json research/hybrid/board
```

Dense checks use seed `0xB1172032`, 642 orthographic + 64 perspective views, and
8× sampling refined to 32×. They test the last three LODs against source and
predecessor, without replacing recorded pilot failures. The board embeds geometry
and shaders and can open offline. Generated mesh binaries are ignored by Git;
the board, raw measurements, manifests, configuration and hashes are committed.

The runner rejects changed binaries, source trees, inputs and configuration when
resuming a run. Use a new name for a new build. Batches checkpoint at 50 minutes;
keep combined concurrent jobs at four or fewer and RAM below 24 GiB. Timings on
the shared workstation are observations, not GPU speed predictions. Source and
buffer hashes, normalized settings, compiler and executable identities live in
each run's metadata. Failed prototypes remain recorded and receive no SCORE.

For repeat timings, run `timing-v5.json` alone after other owned bake, audit and
build jobs finish; use new run names for a new executable. To verify a shared
C ABI build, configure with `-DBUILD_SHARED_LIBS=ON -DBLITZ_TOOLS=OFF`, build
`blitz-c-smoke`, then run CTest with `-R c-abi`.
