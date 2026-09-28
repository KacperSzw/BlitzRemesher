# Validation of automatic hybrid v5

The final reducer/search/IO source hash matches the frozen v5 pilot identity.
[source-v5.json](source-v5.json) identifies the executable and source overlay;
all plan files and raw run identities are committed. Validation was run after
freezing the v5 algorithm. No validation results were used to retune it.

## Contracts and integration

- Release CTest: **8/8 passed** (hybrid, vegetation, general contracts, precision,
  render cost, C ABI, foliage import and IO).
- AddressSanitizer + UndefinedBehaviorSanitizer: **8/8 passed**.
- The hybrid fixture's configuration was subsequently made explicit rather than
  relying on a tunable default; its release and sanitizer tests both passed again.
- Shared-library C ABI smoke check: **passed**, with `.so.3` compatibility names.
- ABI 2 descriptors are rejected. Source streams remain unchanged; borrowed
  predecessor indices are resolved against the correct storage. Integer allowance
  boundaries, per-slot rather than aggregate constraints, duplicate runtime buffer
  accounting, JSON round trips and explicit legacy parsing are covered.
- The frozen AGENTS.md, scoring protocol, corpus and original foliage manifest
  have no diff. Production C++ remains independent of external simplifier tools.

## Recorded chains and byte checks

**90 completed chain exports** were checked: 74 vegetation records (including
prototype comparisons and six timing repeats), plus 16 opaque regression records.
The two failed initial configuration parses remain recorded separately.
All completed chains satisfy their configured source and predecessor gates,
nonincreasing triangle counts, and the integer per-slot reference allowance.
Measured binary sizes match the selected costs and reported storage totals;
vegetation exports additionally verify recorded file hashes and accessor payload
sizes. Fixed-pool 0/2/5/10% memory selections are monotonic.

Three serial timing repeats per primary asset reproduce identical geometry and
glTF hashes. [Timing records](timing.json): fern median 33.43 s (33.10–35.05),
broadleaf median 10.83 s (10.83–10.86). These are generation/audit times at eight
proposals per slot, on a shared workstation.

The 40-asset geometry collection and all 12 origin-control chains complete.
The twelve validation assets include the three large pine saplings; none are
omitted. Vegetation is unscored card geometry, not opacity-aware validation.

## Independent dense audit: 9/10 chains pass

Each tested chain receives six source/predecessor checks across the last three
scheduled levels, using 642 orthographic + 64 perspective cameras, a separate
rotation seed `0xB1172032`, and 8× supersampling refined through 32×.
**59/60 checks pass.** Both primary models pass at all three tested budgets.

The eight-proposal leafy branch (`loaf_leafstick3_3`) fails its LOD5 predecessor
check: conservative upper error **2.6738429466 px**, limit **2.6666666667 px**,
at view 96 after refinement to 32×. Source checks and its later two transitions
pass. The failed check stops at the first rejection; its changed-area diagnostic
is not an exhaustive maximum over all dense views.

The chain passed its configured 12+4 bake audit, but this independent check
finds a view not covered by that smaller camera contract. The failure is shown
on the board and retained in [the raw audit](dense/mixed-b8-loaf_leafstick3_3.json).
It is not relabeled as passing, discarded, or replaced after validation. Finite
view audits are not an all-view appearance guarantee. The default library audit
is denser than these small research bakes; no full-corpus default-quality claim
is made here.

## Opaque regression and board

The frozen eight-asset opaque pilot completes for both automatic core and
experimental preset configurations, with no failed or unchanged assets. SCORE
uses the existing protocol formula: **95.5778 core**, **97.3361 preset**.
The old v3 rebuilt controls score 94.6897 and 97.2338 respectively. Origin and
storage policies differ intentionally; these are fixed-proposal-budget research
comparisons, not a claim of equal wall-clock work or corpus-wide default quality.

The offline board was checked in Chromium at 1680 px and 390 px widths: fourteen
chains, 112 scheduled tiles, bake/storage metadata on every row, one visible dense
failure, working inspection, no JavaScript errors, no external requests, and no
horizontal page overflow. [Recorded browser checks](board/checks.json).
