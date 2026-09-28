# Automatic hybrid: memory within a triangle allowance

Production generation now chooses vertex storage automatically. Every scheduled
LOD remains within **5% of a complete triangle-minimizing reference chain found
in the same search**, then the selector minimizes resident vertex/index bytes.
Source and predecessor pixel gates are unchanged. No universal LOD-number or
screen-size switch is imposed. The selected chain can use source vertices,
compact new vertices, or both. Direct and Progressive remain internal proposal
origins, with explicit research overrides.

[Open the offline board](board/index.html). It contains fourteen actual chains,
measured bake times, source/new vertex labels, resident payloads, and independent
check status. [All measurements](RESULTS.md), [raw summary](measurements.json),
[implementation/research contract](PLAN.md), [reproduction](REPRODUCE.md), and
[validation](VALIDATION.md) provide the supporting records.

## What changed in the measurements

Matched primary examples against the previous v3 rebuilt candidate:

| Asset / proposals per level | Previous → automatic resident bytes | Previous → automatic final triangles |
|---|---:|---:|
| Fern / 8 | 201,484 → 152,272 (−24.4%) | 16 → 22 |
| Broadleaf / 8 | 77,100 → 57,676 (−25.2%) | 110 → 109 |
| Fern / 32 | 225,748 → 243,668 (+7.9%) | 9 → 8 |
| Broadleaf / 32 | 84,156 → 114,108 (+35.6%) | 103 → 86 |
| Fern / 64 | 221,124 → 168,332 (−23.9%) | 5 → 6 |
| Broadleaf / 64 | 197,488 → 73,924 (−62.6%) | 90 → 46 |

At 64 proposals, fern shares LOD0–2 and owns compact buffers for LOD3–7.
Broadleaf's winning automatic chain uses source vertices throughout. Keeping
endpoint reduction internally is useful even after removing the public
shared-only policy. There is no guarantee that increasing the proposal budget
improves every independent run: this is a bounded beam with different explored
paths, not nested exhaustive searches.

| Eight-proposal cohort | Previous rebuilt bytes → automatic bytes | Category-balanced triangle retention, previous → automatic |
|---|---:|---:|
| 28 development assets | 13,731,872 → 7,982,628 (−41.9%) | 53.779% → 57.229% |
| 12 validation assets | 30,400,920 → 23,863,700 (−21.5%) | 39.057% → 42.745% |

These small-budget cohorts retain **6.4% and 9.4% more triangles**, respectively,
relative to the older search. Category-balanced last-three retention also rises:
28.612% → 30.004% on development and 13.221% → 14.646% on validation. Those
regressions are not hidden by the memory metric. The 5% guarantee compares with
this run's reference, not a historical chain or a global optimum. Vegetation
uses opaque card geometry only and receives no SCORE; opacity and shading are
outside these coverage checks.

## Effect of the allowance, using identical candidate pools

| Allowance | Development resident bytes | Validation resident bytes |
|---|---:|---:|
| 0% | 8,104,744 | 23,878,108 |
| 2% | 8,044,072 | 23,878,108 |
| 5% | 7,982,628 | 23,863,700 |
| 10% | 7,940,128 | 23,861,324 |

The selector is monotonic in memory on a fixed pool. Most savings here come from
allowing mixed storage during search; the extra allowance buys a smaller
additional reduction. Fern illustrates a stronger individual effect: its
64-proposal chain falls from 233,540 bytes at 0% to 168,332 bytes at 5%, with the
same six-triangle tail. The 32-proposal pool only finds substantial memory savings
at 10%; these results remain visible rather than raising the default after
validation. The initial 5% default is retained, not claimed universally optimal.

## Direct versus Progressive

The six development examples at eight proposals use identical gates and reducer
proposal budgets. Extra transition reconnections are separately counted; bake
cost is reported, so equal proposal counts do not imply equal audit work.

| Internal origin policy | Mean triangle retention | Total resident bytes |
|---|---:|---:|
| Mixed origins | 51.663% | 1,851,712 |
| Direct only | 48.623% | 2,110,992 |
| Progressive only | 44.948% | 3,196,224 |

Mixed search does not dominate at this small budget. Progressive wins the
aggregate triangle measure, while mixed uses the least memory. Direct finds a
20-triangle fern tail versus progressive's 54; progressive finds a 160-triangle
grass tail versus direct's 476. Keep both origins internal. A future iteration
should improve budget allocation between them, rather than assume one origin
always gives the best result.

## Search changes and negative experiments

- v1 rejected automatic configuration parsing: JSON merge-patch removes null
  fields. The decoder now handles absent research output correctly; round-trip
  tests cover the production default. Both failures remain recorded.
- v2 split the target search too thinly across origins and placements; fern's
  tail regressed to 65 triangles. v3 shared target evidence but repeatedly
  matched accepted counts and retained cheap, over-detailed histories; tails
  regressed to 96 for fern and 256 for broadleaf.
- v4 restricts the memory beam to a fixed 10% prefix envelope, independent of the
  requested final allowance, and reduces redundant placement probes. This is a
  search heuristic, not an additional quality guarantee.
- v5 reconnects the triangle leader to alternative histories with actual audits,
  and shares proposal/cache work for byte-identical predecessor geometry. Fern
  can preserve its cheaper shared prefix while reaching the rebuilt tail.
  Reconnection audits are additional work and are included in bake time.

All previous records, scoring protocol, corpus and AGENTS.md are unchanged.
ABI 3 removes production mode selectors and exposes the allowance, reference
triangles and storage statistics. Canonical float/RGBA8 vertex streams and u32
indices define the memory model; CPU peak RSS is recorded separately. Engine
compression, alternate packing and streaming residency may change the tradeoff.

## Timing repeats

Three sequential single-worker eight-proposal bakes produced identical geometry
and export hashes. Fern median: **33.43 s**, range 33.10–35.05 s.
Broadleaf median: **10.83 s**, range 10.83–10.86 s. These measure generation plus
audits, not import/export or GPU rendering. Shared-workstation noise remains.
