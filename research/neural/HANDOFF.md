# Action learning handoff — 2026-09-29

Worktree: `/home/kacper/Projects/BlitzRemesher-neural`.
Branch: `research/neural-lod-gpu`. The primary worktree is separate.

## Current milestone

Follow [AUDIT.md](AUDIT.md): GPU final confirmation, resident mesh data,
recycled audit buffers, exact completed-audit caching, failure replay and an
evaluation-only comparison of three saved models. No new optimizer steps are
part of this milestone, locally or in the cloud.

The previous rental `runs/neural/runpod-action-pilot-03` ended at 12:30:36 UTC.
Results were checksummed and collected; compute and its volume were deleted.
Its first full-settings asset reached the five-minute limit: zero complete
assets, no SCORE. CPU final confirmation took 136.878 seconds and GPU audits
162.975 seconds. See [raw evidence](evidence/action-audit-v2/manifest.json).
This was a runtime failure, not evidence that learned ranking improved LODs.

The next run directory is `runs/neural/runpod-action-evaluation-01`. Check for
`rental.json` before assuming it has launched:

```sh
node research/neural/runpod.mjs status runs/neural/runpod-action-evaluation-01
```

The diagnostic uses shelves and moon rock 02 across all nine models/controls
with the full visual settings. Its score is always null. Only a complete,
healthy diagnostic with enough remaining time permits a frozen eight-asset
pilot. Training is never started. Remote output is under
`/workspace/results/action-v2-evaluate`; `phase.json`, scenario
`progress.json`, `report.json` and `result.json` record live progress and
completion. A resource, numeric or disagreement failure stops the comparison.

The cumulative cloud cap remains $10. Before this evaluation, the conservative
ledger plus prior estimate was $4.899. A 90-minute rental at the $2.50/hour
cap, storage allowance and $1 reserve fits under $9.664 cumulative. Actual
billing/resources are reconciled again by launch. The controller and independent
watchdog collect/checksum evidence and terminate compute; a failed collection
preserves its volume. Durable state, not an old timestamp here, owns the cutoff.

## Established evidence

Three seeds passed the one-mesh proof at two checkpoints; see
[raw endpoint evidence](evidence/action-v2/README.md). All ranking controls also
met that proof's quota. A reduced-settings two-mesh smoke showed 6.94% mean
triangle reduction versus 1.43% for constant ranking. Full-pilot superiority
remains unproven.

All three curriculum seeds completed 8,192 updates with finite parameters and
gradients, exact model/AdamW restoration, native/FP64 error below 2e-4, and
98.78–99.39% queried preferred membership. Loss fell from about 1.35 to 0.020.
The US-MO-2 repeat took 20.14–25.04 seconds per seed with 16.41% mean GPU
utilization over 73 one-second samples. The earlier 164-state curriculum
averaged 38.74%. Neither run saturated the GPU. Saved models, labels and health
reports are under [curriculum evidence](evidence/action-curriculum-v2/README.md).

## Remaining learning gates

Follow [V2.md](V2.md). Require three learned seeds to beat the strongest
constant/shuffled control by at least one SCORE point, with reduction in at
least two categories, persisting at another checkpoint under the same contract.
Also report shortest-edge and current-plane controls. The two-category
diagnostic alone cannot meet the full-pilot/persistence gates.

Profile useful throughput and convergence before changing precision, model
size or batch size. BF16, CUDA graphs, segment placement and a GNN remain
conditional. Training labels come only from the frozen training selection.
Pilot families stay out of training; held-out assets remain release-only.
