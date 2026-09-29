# GPU confirmation and mesh residency

Historical milestone. Current GPU residency, captured updates, free placement and
confirmation defaults are documented in [GPU_REFACTOR.md](GPU_REFACTOR.md).


This milestone evaluates the three saved 8,192-update action models. It runs
no additional training and makes no quality claim from an incomplete comparison.

## Measured reason for the change

The previous cloud pilot stopped during its first asset, painted wooden
shelves, after 300.225 seconds. GPU proposal audits consumed 162.975 seconds;
CPU final confirmation consumed 136.878 seconds. Inference consumed 0.097
seconds and topology/feature work 0.108 seconds. No asset completed and SCORE
is null. The confirmation rejection was cancellation, not evidence of a
CPU/GPU disagreement. Raw evidence and archive provenance are in
[evidence/action-audit-v2](evidence/action-audit-v2/manifest.json).

## Implementation steps

1. Add `--neural-confirmation cpu|gpu|compare`. GPU confirmation uses the full
   configured cameras, sampling/refinement, source and preceding emitted LOD.
   CPU remains the public API default; the evaluation-only job selects GPU.
   Compare runs both and rejects decision or metric disagreement.
2. Keep immutable source attributes on the device for one generation. Keep
   exact candidate/reference index and material copies in two device slots;
   unchanged topology is reused across views and successive comparisons.
   Other vertex streams get their own uploads. Mesh ownership stays explicit.
3. Recycle CUDA scratch within the configured memory allowance, charging
   retained capacity. Evict unused buffers before reporting a resource limit.
   Cache complete passing measurements only, with exact topology, bounds and
   numeric settings, under a 4 MiB / 64-entry host allowance. Cancellation is
   checked before cache reuse. Neither resource failures nor incomplete audits
   become successful cache entries.
4. Reduce triangle bin counts on the device. Download one small bin summary
   containing the exact 64-bit total and clipping flag, then validate the
   31-bit sorting bound. Raster images stay on the GPU during normal audits.
5. Record upload/download bytes, allocations/reuse, measurement cache hits and
   separate GPU/CPU confirmation times. Distinguish visual, resource, numeric,
   cancellation and comparison failures. CLI failure bundles contain exact
   meshes, checksums, bounds and settings. Replay with
   `blitz audit-replay FILE --gpu-memory-mib N`; it runs no optimizer.
6. Run the two-category diagnostic across three learned models, constant,
   three shuffled controls, shortest edge and current plane. Use the full
   visual settings in `action-pilot.json`, not the reduced smoke settings.
   The diagnostic manifest is explicitly ineligible for SCORE. Stop for health
   failures or incomplete methods. Consider the frozen eight-asset pilot only
   after a complete diagnostic and a conservative remaining-time estimate.

## Numerical boundary repair

A deterministic normals fixture exposed a one-ULP difference between CPU
and CUDA `acos`: CPU error 1.6699247832167754 versus GPU
1.6699247832167756. At the exact CPU limit, decisions disagreed even though
the rasterized normals were byte-identical. Both evaluators now use the same
double-precision fdlibm-derived angular primitive. Its original Sun permission
notice is retained in `src/metric_angle.hpp`; the independent host libm oracle
and below/at/above-threshold decision tests protect the change.

This changes the last bits of some old angular measurements. It does not
change the error formula, tolerances or quality thresholds. Fresh comparisons
record their binary hash; old results are context, not a matched baseline.
Primary implementation reference:
[OpenLibm e_acos.c](https://github.com/JuliaMath/openlibm/blob/master/src/e_acos.c).

## CPU/GPU boundary and remaining priorities

The active action trainer uploads its feature/label dataset once and samples
those tensors on the device. Each training step still uploads sample IDs and
reads scalar loss/gradient health. Checkpoints intentionally export models and
verification tensors. This milestone does not alter optimizer behavior.

During action inference, CPU code enumerates legal collapses, forms features,
uploads features and receives small action scores. It then edits its existing
CPU topology; it does not fetch a complete mesh back from the GPU. The older
vertex model also downloads embeddings and uploads them to its conditioned
head; that is a separate, inactive architecture in this experiment.

Normal audits retain mesh and raster data on the device. Remaining readbacks
are bin summaries, coverage/appearance summaries and final acceptance data.
These synchronize execution even when their byte count is small. Batching
independent views and moving more rejection logic to the device are the next
profiling candidates. Preserve deterministic view order, cancellation,
resource reporting and exact acceptance when investigating them. A full GPU
topology rewrite has not been justified by the measured full-settings run.

`gpu_upload_bytes` / `gpu_download_bytes` describe evaluator transfers only,
not action inference, checkpoint export or total process traffic. Debug raster
parity tests explicitly download images. Source raster-image caching and
asynchronous multi-view execution are not implemented.

## Verification and execution

The small shelves/rock smoke compares GPU-only and CPU+GPU confirmation with
identical settings and model. Both meshes have identical output hashes and
zero confirmation disagreements. This validates the path, not learning quality
or default-quality performance. Local timing varies on the shared RTX 2080.
Raw metadata, rows and checks are retained with the evidence for this milestone.
All 19 neural-build CTests and five relevant ASan/UBSan CTests passed; the
standalone cloud/action Node suite passed 21 tests. No optimizer steps ran.
The GPU-only smoke used 60/55 allocations with 20,283/25,377 buffer reuses
(shelves/rock), and reused 71/104 completed measurements. Final CPU audit time
was zero. These are reduced-settings operational checks, not throughput or
learning benchmarks; see [verification.json](evidence/action-audit-v2/verification.json).

After committing and pushing, prepare with:

```sh
node research/neural/runpod.mjs prepare-action-evaluation runs/neural/runpod-action-evaluation-01
BLITZ_RUNPOD_DATA_CENTER=US-MO-2 node research/neural/runpod.mjs launch runs/neural/runpod-action-evaluation-01
```

Preparation verifies the pushed revision, frozen asset checksums and saved
model health/hashes. The cloud build excludes LibTorch/training executables.
The experiment lasts at most 50 minutes after setup inside a 90-minute rental
reservation. An independent watchdog enforces the cutoff; collection verifies
the archive checksum before volume deletion. The launcher reconciles live
resources and cumulative spend against the existing $10 authorization,
including a $1 reserve. Read durable rental state for actual deadlines/costs.

## Remote outcome

Revision `2a45fc4` ran on the approved RTX PRO 6000 Blackwell. All 17
evaluation-build CTests passed. The first method completed shelves in 144.485
seconds: 128.634 seconds of proposal auditing, 15.339 seconds of final GPU
confirmation and zero CPU confirmation. Topology/feature work was 0.120 seconds;
inference 0.117 seconds. There were no resource or numeric failures. The mean
retained ratio was 0.999454744 (only 0.0545% reduction across scheduled LODs).

The second mesh reached the shared five-minute method deadline after 155.511
seconds, with 155.224 seconds spent auditing on the GPU. It is cancelled and
unscored. The runner stopped after this incomplete method; the remaining eight
methods and full pilot were not run. The complete comparison and learning
advantage therefore remain unproven. GPU utilization averaged 97.54% over 301
one-second samples, with peak device usage of 1,611 MiB. No training occurred.

The archive was collected and its SHA-256 verified; both pod and volume were
deleted at 13:42:30 UTC. Provider reads confirmed zero remaining resources.
The 14.69-minute rental represents about $0.512 compute at the quoted rate,
before storage. The conservative cumulative experiment ledger is $5.513.
See [cloud/outcome.json](evidence/action-audit-v2/cloud/outcome.json) and the
adjacent raw rows, telemetry and reports.

The remaining measured cost is GPU auditing. Profile GPU rasterization,
distance transforms and appearance matching separately before implementing
another optimization. Consider bounded reference-image/field reuse or work
shared across candidate audits if those stages dominate. Scalar synchronization
can still be measured, but CPU topology work is below 0.3 seconds per attempted
asset here. Do not start more training or weaken visual gates to compensate
for this incomplete evaluation. A future rental must fit the newly updated
ledger; the current fixed 90-minute reservation no longer fits the $10 cap.
