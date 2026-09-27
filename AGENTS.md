# BlitzRemesher

Read docs/SPEC.md and research/PROTOCOL.md before changing algorithms or scores.
The implementation language is C++20. Do not introduce Python.

## Workstation

Use the pinned Nix shell. Never reboot automatically. Other agents may be active.
Do not close terminals without verifying they contain no live Codex session.
Preserve desktop workspace and focus; GUI launches use workstation-desktop.
Research batches are resumable and must stay below one hour, four CPU workers,
and 24 GiB combined memory. Do not change /etc/nixos for this project.

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

## Tests and changes

Tests own their settings and protect contracts, boundaries and lifecycle.
Do not freeze tunable defaults or arbitrary output topology in golden tests.
Run focused tests, then required CTest/sanitizer checks. Record validation.
Commit descriptions explain the concrete reason and measured consequences.
