# BlitzRemesher

Read docs/SPEC.md and research/PROTOCOL.md before changing algorithms or scores.
The implementation language is C++20. Do not introduce Python.


## Correctness and memory

Use contiguous indexed storage, compact IDs, packed flags, explicit ownership,
and borrowed views. Establish range, sentinel, alignment, lifetime and resize
invalidation before choosing field types. Do not allocate objects per vertex.
SIMD and wider storage need measurements, not assumptions.
Never mutate supplied source streams. Reuse mode preserves original vertex IDs
and attributes byte for byte. The C ABI exposes no STL types or exceptions.

## Research

QEM cost is a proposal heuristic, never a pixel-error certificate.
Every delivered LOD needs adjacent and direct-to-source validation.
Freeze corpus, camera sets, settings and score protocol across comparisons.
Never improve scores by changing thresholds, excluding failures, silently
repairing hard assets, changing splits, or tuning against held-out samples.
Protocol changes require a new version and rerunning every baseline.
Save hypotheses, raw measurements, failures and negative results.
Incomplete batches have no aggregate score. Report unreduced fallbacks.
Do not claim global optimality or all-view guarantees.
Report final-level and last-three-level retained ratios alongside SCORE when
working on tails. A changed source cap, starting size or level count is a
different scenario, not evidence of an algorithmic gain. The source cap only
clamps the cumulative screen-space policy; an already inactive cap adds nothing.
Tiny-triangle and quad-occupancy measurements are geometry proxies before
depth testing, not GPU timings. Report coverage changes alongside cost changes.
Runtime compaction may merge only exact consecutive render-data duplicates;
retain scheduled audit records and the original SCORE denominator. Equal
triangle counts are insufficient. Preserve each retained slot's threshold.
Reducer/proposer borrowed outputs address their actual input. In progressive
rebuild, those compact vertex IDs must never be interpreted against LOD0.
Use explicit per-binary build stamps for overlapping benchmark jobs; never
pair a frozen executable with a mutable, unrelated global source stamp.

## Tests and changes

Tests own their settings and protect contracts, boundaries and lifecycle.
Do not freeze tunable defaults or arbitrary output topology in golden tests.
Run focused tests, then required CTest/sanitizer checks. Record validation.
Commit descriptions explain the concrete reason and measured consequences.
