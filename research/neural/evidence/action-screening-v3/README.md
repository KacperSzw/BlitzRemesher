# Expanded curriculum: healthy first checkpoints

Snapshot of the active H200 run at revision `46ba983`, 2026-09-29. This is
training-health evidence, not a completed LOD comparison or a SCORE.
See [milestone.json](milestone.json) and the checksummed [manifest](manifest.json).

All 19 remote CTests passed. Both full-settings baseline assets completed in
449.554 seconds. Their output hashes and complete per-LOD measurements exactly
match the preceding H200 baseline. Four fresh, CPU-reference-confirmed label
shards added 72 states in three categories; the complete curriculum has 236
states across 11 shards. The original seven shards remain frozen.

| Seed | Updates | First loss | Last loss | Preferred membership on training states | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 101 | 8,192 | 1.35907 | 0.01526 | 99.15% | 42.889 |
| 202 | 8,192 | 1.34561 | 0.02125 | 97.46% | 44.851 |
| 303 | 8,192 | 1.36585 | 0.01852 | 98.73% | 39.975 |

All three checkpoints have finite values, changed parameters, exact model and
AdamW restoration, native inference error at most 4.30e-6 and FP64 comparison
error at most 1.98e-5 (limit 2e-4). Their exported models and optimizer checkpoints
were downloaded and independently hash-checked. There were 24,576 remote
optimizer updates and zero local optimizer updates.

GPU telemetry over the captured interval averaged 95.25% busy time in audits,
53.18% in preparation and **10.77% in training**. Training does not saturate H200.
Busy time is not a measurement of achieved arithmetic throughput. The small
network's CPU launch/synchronization overhead needs profiling before deciding
between CUDA graphs, fused updates or batching changes. Increasing repeated
work solely to raise the utilization number would not establish better learning.

The first fixed comparison (constant ranking, then seed 101) is running on the
two diagnostic meshes, with the full visual configuration and eight action
trials per proposal. The 12-minute per-method limit and hard cutoff apply.
Full-pilot superiority and generalization remain unproven. The controller may
attempt a second checkpoint only after a complete comparison and sufficient
remaining measured time.

## Recovery and next step

Local rental directory: `runs/neural/runpod-action-screening-h200-01`.
Hard cutoff: **2026-09-29 16:25:55 UTC / 18:25:55 Europe/Warsaw**.
Maximum additional reservation, including earlier attempts and $1 reserve:
**$7.827003 of the authorized $8**. No further rental is authorized by this
snapshot. The remaining grant must be recalculated from the durable ledger.

```sh
node research/neural/runpod.mjs status runs/neural/runpod-action-screening-h200-01
```

The existing controller collects the final archive with checksum verification
and deletes compute and storage on success. If collection fails, it terminates
compute and retains storage for recovery. This snapshot does **not** claim the
rental has ended. After it ends, inspect `collection.json`, the final comparison
report and provider resource state; preserve incomplete or negative comparisons.
Do not replace those outcomes with the favorable training-loss measurements.

Setup took roughly 12 minutes. Reusing a verified build is another cost
optimization to investigate before the next paid experiment.
