# Decision: retain the storage work as experimental

Source sharing was the strongest path in this experiment. It fixes a real budget
limitation and admits rebuilt geometry, but **it does not meet the requested
small-LOD improvement**. The new shared run has **zero independently qualified
reduced chains**. All three controls remain off by default.

## Measured outcome

- Indexed targets and interior wedge merging produced exactly the old
  appearance-screen exports on all eight assets. Their better storage density
  did not reach a new accepted chain.
- Source sharing (v5) completed 8/8 assets: SCORE **1.721**,
  category-balanced chain retention improved **1.09%**
  against the matched v3 appearance reference. The target was at least 10%.
  No asset regressed in whole-chain retention; no asset met the 25% tail target.
- The three unchanged source fallbacks remain in the denominator. The five
  reduced chains all fail the fresh independent full-chain audit.
- v4 and corrected v5 export identical glTF and buffer bytes. Their measured
  total bake times were 260.79 and 251.76 seconds on a shared workstation; these
  are observations, not a paired speed claim.

[Complete measurements](REPORT.md) · [smokes and failed probes](SMOKE.md) ·
[byte equivalence](export-equivalence.json) · [offline board](../../examples/density-board/index.html).

## Independent qualification

642+64 fresh cameras, rotation 0xA1172028, 8× through 32×, every LOD1–7 against
source and predecessor. Audits remain finite-view/sample checks. v5 shares v4
proofs only after exact export and contract checks; the original measurements
remain intact. Archived screen/ordering rows retain their earlier 0xA1172027
records and are labeled separately in the board.

| Asset | Outcome | Failing comparisons |
|---|---|---|
| ph_painted_wooden_shelves | Fail | LOD7 source (view 127); LOD7 adjacent (view 53) |
| ph_metal_stool_02 | Pass · unchanged source | — |
| ph_grass_bermuda_01 | Pass · unchanged source | — |
| ph_dead_quiver_trunk | Fail | LOD1 source (view 56); LOD1 adjacent (view 56); LOD2 source (view 254); LOD3 source (view 509); LOD4 source (view 555); LOD5 source (view 581); LOD6 source (view 702) |
| ph_moon_rock_02 | Fail | LOD1 source (view 16); LOD1 adjacent (view 16); LOD2 source (view 88); LOD3 source (view 88); LOD4 source (view 116); LOD4 adjacent (view 12); LOD5 source (view 19); LOD7 source (view 36) |
| ph_rock_face_02 | Pass · unchanged source | — |
| si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652 | Fail | LOD7 source (view 11); LOD7 adjacent (view 11) |
| si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7 | Fail | LOD4 source (view 67); LOD4 adjacent (view 22); LOD5 source (view 24); LOD6 source (view 79); LOD7 source (view 109) |

[Raw qualification](qualification.json) records every failing witness. No new
chain from this run is offered as game-ready. The existing appearance-screen
and ordering Bell chain remains an archived qualified result; the more reduced
Bell candidate here fails at LOD7.

## What changed in the library

1. Indexed targets use measured referenced/output vertex density within existing
   work budgets, with actual emitted bytes deciding admission.
2. Interior wedge merging fits continuous attributes while protecting seams,
   boundaries, discrete values and UV orientation. Local UV repair has a bounded
   fallback.
3. Source sharing matches every tuple byte, keeps exact source IDs, and exports
   one shared source-prefix pool plus changed vertices. The budget ledger counts
   the actual exported layout. Allocation history participates in graph pruning.

Controlled regressions, exported byte counts, release tests and sanitizer tests
pass; see [verification](VERIFICATION.md) and [reproduction](REPRODUCE.md). C ABI 5
is unchanged; C++ consumers must rebuild.

## Best next algorithm direction

Prioritize **local proposal repair driven by appearance failures**. Global QEM
proposals still lose small normal/visibility features, so one bad region rejects
an otherwise useful reduction. Changing the triangle request or buffer layout
does not repair that region. The earlier [appearance replay](../appearance/DECISION.md)
and the new dense failures are the evidence for this direction.

Start with a deterministic two-region fixture: one freely reducible region and
one small appearance feature that makes the global proposal fail. Map a failing
audited sample to original faces, preserve that local region, and retry the rest
within the existing proposal allowance. Retain source/adjacent checks and all
current caps. The first requirement is fewer triangles than both source fallback
and a uniform milder reduction while passing an untouched camera set. Only then
run stool/trunk and the frozen pilot. This direction is not implemented or
claimed proven in this change.

The development usefulness gate failed, so validation20 and paired timing
repetitions were not launched. Held-out assets remain unused.
