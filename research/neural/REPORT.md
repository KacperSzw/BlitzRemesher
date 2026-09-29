# Implementation and first experiment — updated 2026-09-29

Implemented on `research/neural-lod-gpu` in the sibling worktree
`/home/kacper/Projects/BlitzRemesher-neural`. The primary checkout remains separate.
Training is stopped at the user's request. The original sustained run failed
during its 25,000-update audit. The repaired audit completes; a briefly resumed
continuation saved update 26,459 when stopped at 05:44 UTC on September 29.
No validation or release-quality model has been completed. The earlier
`blitz-neural-first-pass.service` was stopped after
preparation, bootstrap/refinement training and the first full pilot. All its
checkpoints, rows and partial audits remain in `runs/neural/first-pass`.
Historical training measurements below remain valid for their recorded runs.

## Audit failure and repair — 2026-09-29

The original run's tensor training reached update 25,000 with finite gradients
and successful model/AdamW restoration. At 20:59:37 UTC on September 28 it exited
after retrying an audit that made no progress. Three of eight pilot assets had
completed. `ph_dead_quiver_trunk` requested a raster at 312.0674954763457 pixels
and 32x supersampling: `ceil(312.0674954763457 + 8) * 32 = 10272` per side,
or **105,513,984 samples**, exceeding the fixed **64,000,000** per-view cap.
An exception trace against the frozen original executable and model confirmed
`raster exceeds per-view sample cap`. Training VRAM exhaustion was not the cause.

The runner's health gate verified training without exercising a complete audit
cycle. The benchmark stopped at the first resource-limited asset, while its
generic BudgetLimited result concealed the underlying sample limit. Retrying
the unchanged request could never succeed. The earlier handoff consequently
overstated readiness of the complete experiment.

The repair bounds candidate refinement to the last fitting member of the
original sampling sequence. Required initial sampling, pixel limits, views,
metric and final source/adjacent CPU confirmation are preserved. Uncertain
upper bounds reject a proposal, so this may sacrifice reductions that a larger
budget could verify. It does not make an uncertain candidate acceptable.
Standalone CUDA evaluation retains literal resource refusal. Diagnostics record
the first resource kind, request, limit, view, screen size and supersampling.

The benchmark now visits subsequent assets after fixed resource failures and
returns exit 3 with named blocked assets and null SCORE. Both runners stop on
resource limits or bake exceptions, write an incomplete report, and require a
complete pilot before declaring readiness. Readiness caches are keyed by model
hash. Protocol v3 SCORE is unchanged; benchmark metadata records the new
candidate-refinement policy.

Quick verification, without starting another training run:

- CUDA build: **12/12 CTests**; portable release: **10/10**; CPU ASan/UBSan:
  **8/8**. Node contracts cover immediate resource refusal, time-segment
  progress/stagnation and bake exceptions. Runner syntax was checked after the
  model-hash readiness adjustment.
- Saved 25,000-update model: **8/8 pilot assets complete in 47.27 seconds**,
  no blocked assets or bake exceptions. The formerly blocked trunk reports
  zero resource failures and peak owned audit scratch of 2,532,376,776 bytes.
- Quality remains **eight unreduced fallbacks, SCORE 0**. The successful audit
  establishes recovery of the pipeline, not an improvement in mesh quality.
- An intentionally oversized initial raster refused both selected smoke assets,
  reported 270,536,704 requested samples versus 64,000,000, returned exit 3 and
  retained null SCORE. Initial sampling was not silently reduced.

Before the user's stop request, the repaired service completed the saved-model
readiness audit and briefly resumed from update 25,000. SIGTERM saved model and
AdamW state at **26,459**, with finite gradients, verified optimizer restoration
and native export error `9.54e-6`. No training was restarted after that request.
The last checkpoint itself has not received a new full mesh audit. The original
failed run and its incomplete measurements remain unchanged.

Raw evidence and checksums are in [evidence/audit-recovery](evidence/audit-recovery/manifest.json).
The saved-model audit binary is identified separately from the frozen trainer
binary in `recovery.json`. Later readiness path naming changes do not alter that
audit's measurements. See [CLOUD_GPU.md](CLOUD_GPU.md) for external training
options, cost calculations and the remaining checkpoint migration requirement.

## What works

- Optional CUDA evaluator with conservative coverage, deterministic face-order
  depth resolution, exact squared distance transforms, visible interpolated
  normals, linear RGB/material comparisons and refinement. CUDA failures are
  explicit. CPU confirmation validates selected chains and triangle references.
- Three-layer width-64 graph network with a pixel/limit/weight-conditioned head.
  Native CUDA inference uses approximately 100 KiB of exported float weights.
  Source embeddings are computed once in bounded patches with three-hop halos.
  LibTorch is linked only by the C++ training executable.
- Endpoint representative decoder with exact attribute deduplication, seam and
  boundary locks, material junction locks, link/duplicate-face checks, orientation
  checks and UV foldover rejection. Inputs remain immutable. Both shared source
  vertices and compact owned output use unchanged endpoint attribute values.
- Explicit C++/CLI/benchmark neural mode, additive ABI-4 model/generation/info
  functions and checksummed versioned weight files. No automatic CPU generator
  substitution occurs when CUDA or a model is unavailable.
- Geometric overlap breaks equal-triangle/equal-storage ties. Protocol v3 SCORE
  and source/adjacent quality gates retain their existing definitions.
- C++ CUDA training, QEM initialization labels, audited neural-label refinement,
  bounded background segments, immutable executable snapshots, atomic model and
  optimizer checkpoints, health checks and automatic pilot/validation reporting.

## Verification

The CUDA build passed **11/11 CTests**. The portable release build passed
**10/10**, and the CPU AddressSanitizer/UBSan build passed **8/8**. Installed C and
C++ consumers also built and ran successfully. Logs are in `evidence/`.

Neural contracts cover deterministic reduction, source immutability, endpoint
storage, boundaries, UV seams/orientation, three-hop halos, tetrahedron duplicate
faces, nonfinite predictions, model corruption, CUDA raster/metric decisions,
coplanar normals/material ownership, overlap agreement, memory refusal and
cancellation during final reference confirmation.

The initial two-development-mesh training smoke completed 200 updates. Loss
dropped from 0.725 to 0.165; native export differed from LibTorch by at most
`4.8e-7`. A separate resumed run restored model/optimizer state and continued
from update 100 to 200. Refinement preparation and training from exported weights
also ran successfully. First versus last individual minibatch loss is not a
validation metric; the refinement smoke's individual losses increased while
sampling different examples, and no quality improvement is claimed from it.

The initial four-level, 32-to-16 px mesh smoke used two frozen pilot meshes.
Painted wooden shelves retained 93.89% of triangles on average; metal stool
remained unreduced. Both returned audited chains. This bounded smoke establishes
plumbing and visible fallback behavior; it is not a production-quality result or
a comparison with the CPU generator. Its raw scenario summary is preserved.

A new overlap parity assertion initially used an edge-on camera with zero
samples. It was replaced with a populated view, and the GPU viewport origin and
top-left convention now match the CPU diagnostic. Final-confirmation cancellation
also received a focused regression. The run's benchmark binary was updated before
any mesh benchmark; `provenance.json` records both executable hashes and separate
source archives for training/preparation and auditing. That earlier pilot also
inherited a stale default build-stamp field; its executable/source archive hashes
are the actual provenance. The latest benchmark skips that unrelated stamp for
neural runs, and explicitly reports direct source proposals and neural strategies.
The sustained runner freezes its audit configuration and full source archive.

## Measured training budget

Hardware: RTX 2080, 8 GiB, sm_75; driver 595.71.05. The initial GPU preparation of
all **66 assets / 8,499,332 triangles** took **171.89 seconds**. Excluding pilot
source families removed six more assets than the original asset-only split.
Validation and held-out assets were not used for training.

The full-data 100-update health checkpoint recorded:

| Measurement | Observed |
| --- | ---: |
| Mean loss, first 25 updates | 0.71779 |
| Mean loss, last 25 updates | 0.65696 |
| Time per update, including checkpoint verification | 0.01022 s |
| Estimated time for 5,120 updates at this rate | 52.32 s |
| Peak PyTorch allocated / reserved memory | 135.61 / 410 MiB |
| Native inference maximum absolute difference | 1.79e-7 |
| Parameter changes / checkpoint restore | verified |

PyTorch memory figures exclude the CUDA context, native evaluator and desktop.
These measurements support fitting the training work comfortably inside ten
hours. They do not predict full camera audit duration or establish model quality.
The runner enforces a ten-hour total budget and 50-minute segments; incomplete
audit batches retain null SCORE. The implementing agent need not wait for that
budget to expire.

## GPU throughput iteration

The first training configuration was launch/data limited. Increasing the batch
and overlapping bounded host packing with CUDA changed measured throughput:

| Configuration | GPU active time | Supervised vertices/s | Total device memory peak |
| --- | ---: | ---: | ---: |
| 4 patches, synchronous preparation | 39.2% | 1.93 million | 2346 MiB |
| 32 patches, one preparation worker | 67.7% | 3.93 million | 5523 MiB |
| 32 patches, two preparation workers | 94.2% | 5.43 million | 5523 MiB |
| 64 patches, two preparation workers | 97.8% | 5.63 million | 6883 MiB |

Each patch has up to 4096 core vertices and an exact three-hop halo. The network
remains 24,580 parameters and FP32. Two workers fill at most four pinned batches;
the consumer preserves update order, and restores reproduce the sampling seed
from the update number. Throughput includes checkpoint/export verification.
These are shared-workstation measurements, with five warmup and two tail samples
excluded; 81 one-second samples support the final 97.8% result. GPU active time
does not measure theoretical FLOPs or SM occupancy. Desktop/video processes were
present, with approximately zero GPU utilization before training. Power averaged
184 W in the final calibration. Raw telemetry, contracts and metrics remain in
`runs/neural/utilization`; `evidence/utilization.json` records the summaries.

Checkpoint verification now compares model parameters, AdamW update counters,
first moments and second moments exactly after restoring. A separate 12-to-24
update continuation passed. A SIGTERM test saved update 13 and resumed to update
23, with matching optimizer state. All 2000 batch shapes/core counts agreed
between the one-worker and two-worker runs. Each training checkpoint also
verifies exported native inference within 2e-4. CUDA accumulation can differ in
the last bits. The final CPU path retains move-based ownership transfer; neural
confirmation alone materializes independent candidate chains.

The sustained run passed its actual 4096-update health gate: mean training loss
over the first/last 256 updates decreased from **0.25061 to 0.24204**. Its last
60 telemetry samples averaged **97.95% GPU utilization** (minimum 97%, tenth
percentile 98%). Native export error was **1.15e-5**, parameters changed and all
model/optimizer restoration checks passed. The service then resumed from update
4096 and passed another checkpoint at update 6144 while continuing training.
`evidence/sustained-health.json` and `sustained-resumed.json` retain those records.

## Historical handoff and remaining evidence

The bootstrap stage completed all 5,120 updates in approximately 48 seconds
including both process launches and checkpoint verification. Its final sampled
loss was 0.1498, native export error 3.82e-6, and checkpoint restore passed. The
refinement preparation completed in 370.66 seconds and its second 5,120-update
stage completed in 42.26 seconds. The initial full eight-asset pilot produced
**eight unreduced fallbacks, SCORE 0**. The refined pilot was interrupted for the
GPU throughput iteration and has no complete score. No validation or held-out
result is claimed. The independent
audit scenario is eight levels, 512 to 16 pixels, 3 px source cap, 0.5 maximum
changed area, attributes, eight candidates and the full 642+64 audit cameras.

The sustained run warm-starts from the refined model, using the same 66-asset
refined curriculum and the measured 64-patch configuration. The runner verifies
4096 updates, then continues to 25,000-update stage boundaries. Every stage gets
a complete frozen development pilot audit; two consecutive stages without a
better pilot score stop further training. The cap is 100,000 updates and the
original ten-hour deadline (2026-09-29 06:19:33 UTC). At the measured 0.043 seconds
per update, a 25,000-update stage needs about 18 minutes and 100,000 updates about
72 minutes, excluding dataset load and mesh audits. This leaves time for audits
within ten hours; completion and quality remain measured outcomes.

Use `research/neural/README.md` for commands and API behavior. Inspect:

- `runs/neural/sustained/status.json`, `gpu.jsonl` and stage logs for the failed run;
- `runs/neural/sustained/health.json` for the sustained utilization/health gate;
- `runs/neural/sustained/training/latest.json` and `metrics.jsonl` for
  model/checkpoint paths, hashes, loss, timing and memory;
- `runs/neural/sustained/progress.json` for complete stage pilot results;
- `runs/neural/recovered/report.json` for the interrupted continuation; no
  selected or release-approved model has been produced by this experiment.

The first curriculum covers 16/32/64 pixels and fixed 3 px limits. Other pixel
sizes/weights rely on generalization plus independent audits. Strict seam and
boundary locks can limit reduction, especially on heavily split assets. The
decoder does not predict free vertex positions or texture content. Dense
attribute raster buffers can reach the configured GPU memory cap; these outcomes
remain resource-limited failures/fallbacks. The finite fixtures do not prove
universal bitwise CPU/GPU equivalence; CPU confirmation remains part of production
acceptance. Release evidence still requires the held-out audit after model and
configuration are frozen.
