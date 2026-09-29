# Expanded curriculum: completed screening

The H200 run **completed both checkpoints and comparisons** and was terminated
at **18:11:51 Warsaw / 16:11:51 UTC on 2026-09-29**. The final archive was
collected and its SHA-256 independently rechecked. The provider reports zero
Pods and zero network volumes. See [final outcome](final/outcome.json),
[raw reports and models](final/) and [file hashes](final/manifest.json).

All three seeds reached **16,384 updates**, for 49,152 total remote optimizer
updates. Their final losses are 0.009886, 0.012313 and 0.008636. Every checkpoint
passed finite-value, native/FP64 export and exact model/AdamW restore checks.
No local optimizer updates ran.

The fixed screened seed (101) improved mean triangle reduction across generated
LODs on both assets, under identical visual settings and action budgets:

| Asset | Constant control | Learned at 8,192 | Learned at 16,384 |
| --- | ---: | ---: | ---: |
| Painted shelves | 0.164% | 0.436% | 0.491% |
| Moon rock 02 | 0.190% | 1.280% | 2.369% |

All methods completed with clean resource/numerical/confirmation health. The
constant control's meshes and complete per-LOD measurements repeat exactly
between stages. This is an early positive signal for **one seed on two meshes**;
the full three-seed, nine-method pilot and its superiority gate remain unproven.
Neither screening has an aggregate SCORE.

The learned comparison also cost more time: **584.066 seconds versus 104.756
seconds** for constant at the second checkpoint (5.58x). Final GPU telemetry
averaged **95.38% busy time in audits and 10.17% in training**. Trainer saturation
has not been achieved. Further work should profile trainer launch/synchronization
and audit cost before another paid comparison; broader quality evidence is still
needed before production use.

The rental lasted 45.938 minutes, about $3.514 quoted compute before storage.
The conservative ledger, including both preceding attempts in this grant, is
**$5.747 of the additional $8**. This estimate uses rate caps and storage
allowances; it is not the provider's final invoice. No cloud resources remain.

The original first-checkpoint snapshot below is retained unchanged for provenance.

## Earlier first-checkpoint snapshot

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

### Recovery information at snapshot time

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
