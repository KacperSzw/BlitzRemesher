# Tail-first vertex budget development pilot

Hypothesis: reserving the 20% added-vertex budget for an audited 16 px mesh
before middle LODs spend it will improve the final runtime mesh without
violating source or adjacent view limits. The old forward-budget run found a
44-triangle Moon rock LOD at 43.1 px and reused it at 16 px. A separate
32-proposal Moon rock pilot found a 12-triangle 16 px mesh under the same
20% cap, showing that a smaller audited tail was available with more search.
That pilot had higher middle-level counts, so it was not used as a matched
8-proposal comparison.

The accepted implementation probes up to eight deterministic direct rebuilt
meshes at the 16 px target, reserves the vertex bytes of its smallest passing
candidate, and spends only the remainder before the final slot. All connected
meshes still pass both scheduled audits. Capped selection ranks the last LOD
first, then preceding LODs in reverse order. The custom proposer research
hook skips the dedicated probes. Both the forward search and tail probes are
included in `candidate_evaluations`; the latter and reserved bytes appear in
`proposal_diagnostics`.

Every row below uses the same source, cameras, 20% cap, 0% triangle overhead,
eight transition proposals and asset-specific level schedule in the linked
configs. These are single development examples with shared-workstation timing
noise, not corpus scores. Raw [forward-budget results](forward-budget-baseline/)
and [first tail-probe results](tail-probe-first-pass/) remain available.

| Asset | Forward final | First probe final | Current final | Current added / cap bytes | Current bake seconds |
|---|---:|---:|---:|---:|---:|
| Moon rock | 44 | 16 | 16 | 11,360 / 11,820 | 9.70 |
| Bermuda grass | 21 | 13 | 13 | 3,712 / 5,382 | 4.08 |
| Metal stool | 299 | 90 | 90 | 23,968 / 28,736 | 8.06 |
| Dead quiver trunk | 2 | 30 | 2 | 60,480 / 60,704 | 14.64 |
| Painted shelves | 36 | 36 | 36 | 2,752 / 3,507 | 0.13 |
| Rock face | 135 | 63 | 63 | 101,920 / 104,934 | 11.11 |

The first probe ladder started at 0.25% of source triangles. It missed the
trunk's viable 2-triangle tail and reserved 4,160 bytes for a 30-triangle
candidate. Adding targets down to one triangle restored the 2-triangle result;
its actual tail buffer occupies 192 bytes. This negative result is preserved,
not hidden from the comparison.

Tail priority changes middle LODs. Metal stool's 70.7 px LOD went from 299
to 1,253 triangles while its 16 px tail improved from 299 to 90. The final
mesh alone does not measure the whole chain's rendering cost. The selected
chains and complete per-slot counts are in [BOARD.md](BOARD.md). No aggregate
score is assigned to these examples. Held-out assets were not used for tuning.
