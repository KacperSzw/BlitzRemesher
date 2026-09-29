# Action learning handoff — 2026-09-29

Worktree: `/home/kacper/Projects/BlitzRemesher-neural`.
Branch: `research/neural-lod-gpu`. The primary worktree is separate.

## Current milestone

The user has authorized **$8 more** for staged GPU optimization and training.
The active plan is now [ACCELERATION.md](ACCELERATION.md). Exact GPU distance
and appearance work is faster on the bounded smoke; label preparation reuses
its GPU audit workspace. A new `prepare-action-staged` mode checks full-settings
diagnostics before refreshing training labels and training/auditing two short
checkpoints. The hard rental reservation is at most $7.275 of this new grant.
The first staged rental has finished without training: shelves was 43.3% faster
with exact full-settings output parity, but moon rock exceeded ten minutes.
Its evidence was collected and both resources deleted. The measured GPU
FP32/FP64 ratio is 64:1; the new A100 80GB PCIe profile tests a better fit for
the strict-FP64 audits. H100 quotes lacked persistent storage and were not
rented. See the follow-up section in ACCELERATION.md. The A100 reservation plus
the completed attempt is at most $5.91458 of the new $8 grant. Local and new
remote optimizer steps remain zero at this commit.

A100 inventory disappeared before creation, including six bounded availability
retries. The prepared A100 directory has **no rental**. H200 with persistent
storage was quoted in AP-JP-1 at $4.59/hour; its new 80-minute maximum profile
keeps the completed RTX attempt, retry reservation and reserve below $7.912 of
the additional $8. See ACCELERATION.md for the exact launch command.

The following records describe the preceding evaluation milestone and its old
budget, for context:

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

The evaluation run is `runs/neural/runpod-action-evaluation-01`; it has also
ended, with results collected/checksummed and all cloud resources deleted:

```sh
node research/neural/runpod.mjs status runs/neural/runpod-action-evaluation-01
```

Revision `2a45fc4` passed all 17 remote evaluation-build CTests. The first
model completed shelves in 144.485 seconds, including 15.339 seconds of GPU
confirmation and zero CPU confirmation. Its mean retained ratio was
0.999454744. Moon rock 02 reached the shared five-minute method cutoff after
155.511 seconds and is cancelled. The run stopped before the remaining eight
methods or full pilot. No new training occurred and no comparison SCORE exists.

GPU utilization averaged 97.54% across 301 samples. GPU audits consumed almost
all generation time; topology/feature work was 0.120/0.244 seconds for
shelves/rock. The CPU confirmation bottleneck is removed in the selected GPU
mode. The next step is GPU stage profiling (raster, distance, appearance), then
a measured optimization before repeating the full-settings comparison.
See [remote outcome](evidence/action-audit-v2/cloud/outcome.json) and [AUDIT.md](AUDIT.md).

The cumulative cloud cap remains $10. The conservative ledger is now $5.513;
this evaluation used 14.69 rental minutes (about $0.512 quoted compute, before
storage). The fixed 90-minute reservation plus $1 reserve no longer fits the
remaining cap. Do not repeat it blindly: a future bounded run needs a smaller
reservation or a user-authorized budget change. The current run's cleanup was
verified at 13:42:30 UTC with zero pods and volumes remaining.

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
