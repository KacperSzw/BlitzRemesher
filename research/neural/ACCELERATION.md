# GPU audit acceleration and staged learning — 2026-09-29

## Plan and gates

1. Profile the unchanged evaluator with Nsight Systems on two bounded development
   meshes. Save kernel costs and transfers; this workload has no SCORE.
2. Optimize the dominant distance transform, then appearance work. Preserve
   arithmetic, float rounding, camera order, pixel/area thresholds and all
   protocol-v3 semantics. Require exact outputs, CPU/GPU boundary tests, CUDA
   memory checks, CTest and relevant ASan/UBSan contracts.
3. Retain source uploads and audit scratch in label preparation. Compare labels
   and action trajectories with the original preparation executable.
4. Commit and push before provisioning the RTX PRO 6000. Run remote CTests and
   a full-settings shelves/moon-rock diagnostic before any optimizer steps.
5. Add four frozen training-only label shards: intermediate pixel sizes and a
   rock category. Require verified labels and audited preceding LODs where used.
   Train three independent seeds to 8,192 updates, screen seed 101 against
   constant ranking on both full-settings assets, then resume all three to
   16,384 updates and repeat if the first comparison completed and
   the remaining time covers it. All optimizer steps execute remotely.
6. Every training checkpoint verifies finite values, native/FP64 export parity
   and exact model/AdamW restoration. Health failures or incomplete comparisons
   stop continuation. Collect checksummed results and terminate the rental.

The diagnostic keeps the full visual configuration and deterministic work
budgets from `action-pilot.json`; only its runtime allowance increases to ten
minutes per method. Each comparison is capped at 45 minutes. Two diagnostic
assets cannot prove the eight-asset pilot gate or release quality. Healthy
optimization and lower training loss also cannot establish LOD generalization.
The initial nine-method schedule was replaced after the measured H200 baseline
showed it could not fit the remaining authorization. Screening is explicitly
unscored; the nine-method, three-seed full-pilot gate remains separate.

## Local findings

Baseline revision: `58d2002`. Hardware: shared RTX 2080, CUDA 12.9. See
[raw measurements](evidence/audit-acceleration-v3/).

* Original traced GPU kernel time: distance transform 39.9%, raster 32.2%,
  appearance 24.2%. Device-to-host transfers themselves took 1.016 ms.
  Blocking `cudaMemcpy` API time includes waiting for preceding kernels.
* Transpose both separable passes and store stacks by depth/column. Binary-mask
  first-pass nearest-site scans replace the general lower-envelope algorithm.
  Traced distance work fell from 820.0 ms to 166.4 ms, excluding the small
  transpose kernels. The layout-only trial was only a modest improvement.
* Appearance reduces maxima within each block, skips a center already sampled,
  and skips searches when their minimum possible spatial cost cannot improve
  the center. It retains that center's original floating-point cost.
* Final unprofiled A/B/B/A totals: baseline 2.3506/2.3039 s; optimized
  1.6639/1.6886 s. About 28% less time (1.39x throughput) on this smoke workload.
  Mesh hashes, attributes and complete per-LOD audit results match exactly.
  Shared-workstation timings do not predict Blackwell/full-audit speedups.
* Persistent label workspace: four states, 64 queried actions; identical binary
  labels and complete trajectories. Before/after 1.7754/1.1997 s; one bounded
  measurement, not a general throughput claim.

No visual threshold, precision, camera count, LOD schedule or score changed.
Geometry and raster images stay on the GPU during normal audit calls. Small
verdict/bin summaries still reach the CPU. Topology, feature assembly and
label-preparation reference confirmations remain CPU work.

## Spending and recovery

User authorized **$8 additional** after the earlier rentals were deleted.
Conservative baseline: $5.513044952777777. The new cumulative guard is that
baseline plus $8, not the old $10 ceiling plus $8. Prior rentals, retries,
storage allowance and a $1 reserve all count.

`prepare-action-staged` reserves 150 rental minutes at at most $2.50/hour plus
$0.01/hour storage allowance: $6.275 rental + $1 reserve = **$7.275** maximum
additional reservation. Live quote before launch: $2.09/hour in US-MO-2.
Setup is capped at 30 minutes, staged experiments at 110, collection at ten.
Individual training/preparation processes are short; comparison batches stop
before 50 minutes. The independent watchdog enforces the absolute cutoff.
Early completion terminates immediately. A failed collection preserves its
volume for recovery rather than falsely claiming successful cleanup.

```sh
node research/neural/runpod.mjs prepare-action-staged runs/neural/runpod-action-staged-01
BLITZ_RUNPOD_DATA_CENTER=US-MO-2 node research/neural/runpod.mjs launch runs/neural/runpod-action-staged-01
node research/neural/runpod.mjs status runs/neural/runpod-action-staged-01
```

## Research used

Prophet research was checked against primary sources:

* [NVIDIA event API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html): event lifetime, stream ordering and timing.
* [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/#cuda-trace): kernel/API/transfer tracing; no permanent timing instrumentation needed here.
* [PBA+ author notes](https://www.comp.nus.edu.sg/~tants/pba.html): column-oriented coalescing and transposes. A full PBA replacement was deferred; memory-layout changes plus a binary first pass were sufficient for a measured pilot improvement.
* [Felzenszwalb/Huttenlocher EDT](https://cs.brown.edu/people/pfelzens/dt/index.html): separable lower-envelope distance transform. The general second pass retains its existing arithmetic and float-distance output.

## Blackwell outcome and FP64 follow-up

`runpod-action-staged-01` used revision `9159228`. All 19 remote CTests passed.
Shelves finished in **81.926 s** versus **144.485 s** previously (43.3% less
time), with identical output/attribute hashes and complete per-LOD measurements.
Moon rock still exceeded the shared ten-minute diagnostic cutoff: 518.043 s
on that asset, including 516.535 s in GPU audits, 299 action trials and 206
legal collapses. It remains cancelled, with no SCORE. Training did not start.
The archive was checksummed and collected; compute and storage were deleted.
See [raw cloud evidence](evidence/audit-acceleration-v3/cloud-blackwell/).

A CUDA device query measured a **64:1 FP32/FP64 throughput ratio** on the rented
RTX PRO 6000, with 188 multiprocessors. Its high busy-time utilization does not
establish high utilization of the whole processor. The profiler and arithmetic
mix suggest an FP64-capable device is worth testing; this is a hardware-fit
hypothesis, not a measured speedup guarantee.

[NVIDIA specifies](https://www.nvidia.com/en-us/data-center/h100/) native FP64
at 30 TFLOPS for H100 NVL and 34 for SXM. The live NVL quote was $3.19/hour,
but no quoted H100 location had the required standard persistent storage.
No H100 was rented. A100 80GB PCIe was available with that storage in CA-MTL-3
at **$1.59/hour**. [NVIDIA's A100 specification](https://www.nvidia.com/en-us/data-center/a100/)
lists **9.7 TFLOPS native FP64**, distinct from its FP64 Tensor Core figure.

The `a100` profile uses CUDA architecture 80 and verifies the full device,
driver, memory and hourly rate. Profile selection propagates to the durable
controller, watchdog and remote build. The ledger charges each rental using
its own rate cap. The new 150-minute reservation is $4.15 plus $1 reserve;
including the completed Blackwell attempt's conservative $0.76458 estimate,
the maximum additional reservation is **$5.91458 of the authorized $8**.
All algorithm, training and audit gates remain the same.

```sh
BLITZ_RUNPOD_PROFILE=a100 node research/neural/runpod.mjs prepare-action-staged runs/neural/runpod-action-staged-a100-01
BLITZ_RUNPOD_PROFILE=a100 BLITZ_RUNPOD_DATA_CENTER=CA-MTL-3 node research/neural/runpod.mjs launch runs/neural/runpod-action-staged-a100-01
```

A100 availability disappeared at launch. Six bounded quote retries created no
resources and incurred no rental charge. The next available FP64 device with
standard storage was **H200 in AP-JP-1 at $4.59/hour**. NVIDIA lists native FP64
at 34 TFLOPS for [H200 SXM](https://www.nvidia.com/en-us/data-center/h200/).
The `h200` profile caps the GPU rate at $4.60/hour and the rental at **80 minutes**
(30 setup, 40 experiment, ten collection). Its reservation, the completed RTX
attempt, storage allowance and $1 reserve total at most **$7.91125 of the new
$8 grant**. No A100/H100 compute was created. This availability-driven fallback
keeps all audit and optimizer health gates.

```sh
BLITZ_RUNPOD_PROFILE=h200 node research/neural/runpod.mjs prepare-action-staged runs/neural/runpod-action-staged-h200-01
BLITZ_RUNPOD_PROFILE=h200 BLITZ_RUNPOD_DATA_CENTER=AP-JP-1 node research/neural/runpod.mjs launch runs/neural/runpod-action-staged-h200-01
```

## H200 outcome and corrected scheduling

Revision `d7a00fd` passed all 19 remote CTests. Both full-settings assets
completed: shelves **34.020 s**, moon rock **413.364 s**, total **447.383 s**.
Shelves output hashes and all per-LOD measurements exactly match the optimized
Blackwell result; elapsed time is 2.41x faster. Mean retained triangle ratios
were 0.999454744 and 0.986509858 respectively. No resource, nonfinite or
confirmation disagreement occurred. These are one saved model's measurements,
not a learned-versus-control quality comparison.

The old scheduling guard reserved nine times that baseline plus 20 minutes
before allowing any optimizer work. It therefore refused training within the
40-minute experiment. This was a scheduling error for the requested quick
training milestone. The run ended after 18.904 rental minutes, with collected,
checksummed evidence and provider-verified deletion of compute and storage.
See [H200 evidence](evidence/audit-acceleration-v3/cloud-h200/).

The corrected staged workflow trains after a healthy baseline and verified
fresh labels. Its first comparison is fixed constant ranking followed by seed
101 at identical full visual settings and eight action trials per proposal.
Each screening method can use at most 12 minutes, allowing margin beyond the
measured 7.46-minute baseline; the absolute experiment deadline takes precedence.
All three seeds still train and must satisfy numerical/export/optimizer restore
checks plus the exact requested update count. A second checkpoint is conditional
on completion and measured remaining time. No incomplete screening or training
metric can satisfy the full pilot gate. Regression tests reproduce the old
447-second-baseline scheduling failure and cover missing/duplicate/cancelled
audit rows. Thirty Node contracts and both affected CTests pass; C++ is unchanged
since the earlier sanitizer/memory checks and full local/remote CTest passes.

Conservative additional spending through these two rentals is **$2.217003**.
The next H200 rental is capped at **60 minutes**, including setup and collection.
The experiment is at most 40 minutes, shortened by the absolute rental cutoff
when setup uses more than ten minutes. At the $4.60 hourly compute cap plus
storage allowance, prior spending + $4.61 rental + $1 reserve = **$7.827003**,
within the user's additional $8. No resources remain active while implementing
and testing this correction. Commit and push before creating the next rental.

```sh
BLITZ_RUNPOD_PROFILE=h200 node research/neural/runpod.mjs prepare-action-staged runs/neural/runpod-action-screening-h200-01
BLITZ_RUNPOD_PROFILE=h200 node research/neural/runpod.mjs launch runs/neural/runpod-action-screening-h200-01
```

## Healthy training milestone

The corrected run at `46ba983` passed all 19 remote tests and repeated both
baseline outputs exactly. Its fresh labels are reference-confirmed; 72 added
states bring the curriculum to 236. Each of three seeds completed 8,192 updates
with finite values, verified exports and exact model/AdamW restore. Final losses
were 0.01526, 0.02125 and 0.01852. All checkpoints and labels have been captured
locally with hash verification; see [raw training evidence](evidence/action-screening-v3/README.md).

The matched screening comparison is still running at this milestone. No SCORE
or generalization claim is available. GPU busy time averaged 95.25% in audits
and only 10.77% in the small trainer. This leaves trainer launch/synchronization
overhead as a profiling target; it does not justify claiming full saturation or
enlarging the network without quality evidence. The hard rental cutoff remains
18:25:55 Warsaw time, with automatic collection and cleanup.

## Completed screening and cleanup

Both checkpoints and matched screenings finished successfully. All three seeds
reached 16,384 updates; seed 101's mean LOD-chain triangle reduction improved
from 0.436% to 0.491% on shelves and from 1.280% to 2.369% on moon rock 02.
Constant ranking repeated at 0.164% and 0.190% respectively, with identical
control outputs at both stages. The second learned comparison took 584.066
seconds against the control's 104.756 seconds. This single-seed, two-mesh
screening is an early signal; the full pilot gate remains unproven and unscored.

The archive and all saved model/checkpoint hashes were verified. The rental
ended early at 18:11:51 Warsaw time on 2026-09-29; provider verification confirms
zero Pods and volumes. Conservative spending across this additional grant is
$5.746592 of $8. Final training GPU busy time was 10.17%, versus 95.38% for
audits. Preserve these limitations when reporting the positive reduction signal.
See [complete raw evidence](evidence/action-screening-v3/README.md).
