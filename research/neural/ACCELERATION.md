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
   Train three independent seeds to 8,192 updates, audit all nine methods, then
   resume to 16,384 updates and repeat if the first comparison completed and
   the remaining time covers it. All optimizer steps execute remotely.
6. Every training checkpoint verifies finite values, native/FP64 export parity
   and exact model/AdamW restoration. Health failures or incomplete comparisons
   stop continuation. Collect checksummed results and terminate the rental.

The diagnostic keeps the full visual configuration and deterministic work
budgets from `action-pilot.json`; only its runtime allowance increases to ten
minutes per method. Each comparison is capped at 45 minutes. Two diagnostic
assets cannot prove the eight-asset pilot gate or release quality. Healthy
optimization and lower training loss also cannot establish LOD generalization.

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
