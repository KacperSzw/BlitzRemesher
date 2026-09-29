# Appearance-preserving reduction

[Plan](PLAN.md) · [Measurements](REPORT.md) · [Decision](DECISION.md) ·
[Offline comparison board](../../examples/appearance-board/index.html).

The previous graph experiment is frozen at commit ab6a8bd. This experiment adds
conservative screening and three separately selectable appearance-QEM stages.
The C ABI stays at version 5 and production defaults remain unchanged.

```sh
nix develop . -c cmake --build --preset release
node tools/appearance-run.mjs ordering stool
node tools/appearance-run.mjs ordering trunk
node tools/appearance-run.mjs ordering
node tools/appearance-report.mjs
node tools/chain-search-audit.mjs appearance-ordering 0 1 research/appearance 0xA1172027
node tools/appearance-audit-share.mjs
node tools/appearance-decision.mjs
node tools/chain-search-board.mjs research/appearance examples/appearance-board
```

Replace ordering with screen, attributes or position. Freeze the corresponding
binary and stamp first; current measurements use builds/screen.json and
builds/qem-v1.json. Frozen executables reside in build/appearance, outside Git.
The exact QEM source archive is builds/source-qem-v1.tar.gz. Rebuilt executables
need new run IDs. Each runner checks the frozen executable hash and the bench
checks input/configuration/protocol provenance on resume. All individual timings
include generation audits and are shared-workstation observations.

The audit-sharing step reuses completed fresh proofs only for byte-identical
exports with matching visual contracts and provenance. Shared records link the
original proof and do not claim another timed audit. Distinct exports need their
own audit; see [Verification](VERIFICATION.md) for the final checks.

The replay tool compares the same in-memory candidate at five sampling densities
and two separately identified camera sets, recording failure samples. Three
triangle targets (.99, .9, .5 of source), two screen sizes (16, 32) and two assets
produce 120 comparisons. Replay results are diagnostics, never chain SCORE.
Candidate PLY files are visualization aids; the evaluator compares the imported
in-memory streams without reimporting those files. Replay manifests hash inputs,
outputs and the frozen executable. Mesh textures and normal maps are unscored.

Appearance field storage is 23 doubles per wedge for one three-component field,
or 35 for normals plus RGB. It is additional to geometric quadrics and uses the
smallest active channel count. Reports include its allocated byte count and
process peak RSS. Coupled original wedges remain separate even when positions
collapse together. This conservative representation avoids inventing continuity
across discontinuities, at the cost of retaining their separate field histories.

Stages fit only supplied continuous normal/RGB streams. Absent normals continue
to use geometric normals in the audit; they do not allocate appearance fields.
No texture baking, material interpolation, or per-vertex heap objects are added.
