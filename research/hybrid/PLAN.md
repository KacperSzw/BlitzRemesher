# Automatic hybrid research contract

Production generation uses automatic storage and origin selection. The reference
is the complete audited path with the lowest sum of scheduled LOD1..N triangle
counts found within this run's work budget; memory breaks ties. This is not a
global optimum or a per-level minimum assembled from incompatible paths.

The selected path minimizes resident packed vertex/index bytes subject to
`10000 * selected[i] <= (10000 + overhead_bps) * reference[i]` at every reduced
scheduled slot. Default overhead is 500 basis points (5%). LOD0 is unchanged.
The pixel contract, source and adjacent audits, and historical SCORE stay intact.
Exact runtime duplicates cost no additional buffers; scheduled SCORE slots remain.
All vertex streams and u32 indices count; source padding, CPU containers, material
metadata and texture payloads do not. These are canonical exported bytes, not a
prediction of a particular engine's allocation or compression.

The beam reserves ceil(width/2) paths by triangles and fills remaining capacity
by resident bytes within a fixed 10% per-slot prefix envelope; unused slots
fall back to triangle ranking. This search heuristic is independent of selection
overhead, including settings above 10%. It keeps different parent histories and an exact source
fallback. The final accepted pool is recorded before beam truncation. Selection
at 0%, 2%, 5%, 10% uses that same pool; overhead never influences proposals.
Both source and previous-LOD origins and both vertex placements share one proposal
budget. Equivalent origins are removed. Automatic adaptive target probes share
parent-local evidence, occasionally probe coarse counts, and try accepted counts
with a source-sharing endpoint placement. The current triangle leader is also
audited against alternative histories, allowing a shared prefix to reconnect to
a rebuilt tail. Those extra transition checks are counted separately from reducer
proposals. Equal predecessor geometry shares proposal work and cached gate results.
Evidence guides proposals, never acceptance or a claim
that error is monotonic. Copied results borrowing an owned predecessor are compacted.

## Frozen comparison design

Start with fern and broadleaf, eight proposals per slot, beam two and the previous
vegetation camera/scale/appearance settings. Preserve failed pilot runs. Expand a
promising candidate to 28 development assets, then freeze before the 12 validation
assets; validation is previously exposed data, not a fresh holdout. Compare the
existing v3 candidate shared/rebuilt controls and reference triangle quality.
Mixed, direct-only and progressive-only automatic policies use equal budgets.
Run 32/64-proposal primary quality checks, eight opaque pilot regressions, and
independent 642+64-view last-three source/adjacent checks (seed 0xB1172032).
No incomplete batch gets an aggregate or SCORE. Vegetation remains unscored
opaque card geometry; opacity textures and shading are not audited.

Keep no more than four CPU jobs and 24 GiB combined process memory; checkpoint
batches at 50 minutes. Record inputs, configuration, executable/source hashes,
compiler, hardware, timing scope, RSS, every candidate cost, failures and exports.
Tests protect exact allowance boundaries, ownership, both visual gates, byte
accounting, ABI rejection and deterministic selection. Release and ASan/UBSan
checks are required before publishing the run.

## API migration

C ABI and shared-library compatibility version are 3. C/C++ production settings
replace the two mode selectors with `triangle_overhead_bps` (0..10000).
C++ `research.output` optionally forces a historical chain-wide placement;
`research.chain` forces an origin for comparisons. The low-level reducer still
has a placement setting. Legacy JSON requires `--legacy-config` for simplify,
or an explicit `settings_json(json, true)` call in research readers. Existing
records are never silently relabeled automatic. Current JSON uses nested research
controls. C results expose reference triangles and resident storage statistics.

Rebuilt-buffer pooling, compressed vertex formats, u16 indices and streaming
selection are deferred. An automatic all-shared chain is valid if it wins the
rule; there is no forced storage boundary. AGENTS.md is unchanged.
