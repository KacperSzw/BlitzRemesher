# Implementation and first experiment — updated 2026-09-29

## Current action-model evidence

The subsequent 164-state curriculum also trained all three seeds successfully:
98.78–99.39% preferred membership, loss about 1.35 to 0.02, native/FP64 error
below 2.244e-5, and exact restoration. Its pilot stopped at a missing CLI memory
option, now repaired and covered by a real executable test. Results were
collected and compute/storage deleted. See [curriculum evidence](evidence/action-curriculum-v2/README.md).
Its two-mesh smoke scored 4.096% mean reduction, below the original 6.940%
smoke; broader learning has not yet demonstrated better mesh quality. The saved
shards are reused for the bounded retry described in [HANDOFF.md](HANDOFF.md).

The v2 endpoint experiment completed on an RTX PRO 6000 with all three seeds
passing the one-mesh gate at 4,096 and 8,192 updates. The six portable models,
raw proof histories, queried labels and telemetry are committed under
[evidence/action-v2](evidence/action-v2/README.md). Final preferred memberships
were 100%, 98.4375% and 100%; every control also met the same 128-triangle quota.
This establishes learning and a working executor, not superior full-pilot LODs.

The later batch executor passed topology, rollback, source-ownership and target
floor tests. A matched two-mesh smoke produced 6.9404% mean triangle reduction
versus 1.4261% for constant ranking with the same eight-trial/32-action limits.
The five-minute full pilot completed no assets and retains null SCORE. GPU
auditing accounted for 299.685 seconds; inference used 0.045 seconds. A bounded
exact-rejection cache addresses repeated source queries, but its short local
probe ended during the first proposal and established no speedup.

Multi-level label preparation now freezes an audited preceding LOD and varies
pixels, limits and target ratio. A regression fix distinguishes known failed
views from cancellation/resource failures, preserving negative training labels.
The tight-limit smoke retained 16 negatives without a false preparation error.
Numerical checkpoint probes now cover the full dataset collection.

Verification: eight neural CTests passed, including CUDA/LibTorch inference and
loss-gradient checks without optimizer steps. ASan/UBSan neural and action
contracts passed. All 20 cloud/action Node tests passed. No local training was
run. Full cloud setup repeats all CTests before training.

Continuation follows [V2.md](V2.md): two allowed training assets, three seeds,
two checkpoints and nine matched full-pilot methods, under fixed visual limits.
It reserves at most 90 rental minutes and 50 experiment minutes. The preflight
budget upper estimate was $8.3185 cumulative including earlier action rentals
and a $1 reserve, below the authorized $10 cap. Generalization, full GPU
utilization, BF16 and segment placement remain unproven or gated.

## Archived v1 execution history

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
# RTX 5090 deployment implementation — 2026-09-29

Added the staged Runpod controller/runbook in `RUNPOD.md`: committed-source and
checksummed original-data packaging, explicit from-scratch training, 2/4-worker
and 64/128-batch calibration, pinned CUDA 12.9 image and LibTorch, independent
local systemd termination watchdog, two-hour absolute rental deadline, and
checksum-verified result collection before deleting persistent storage. REST v2
catalog, Pod, and network-volume shapes were checked against the live official
OpenAPI schema. The management credential remains local.

Validation (no local training):

- CUDA build and CTest: **13/13 passed**.
- Portable ASan/UBSan CTest: **8/8 passed**.
- Node runner/cloud contracts: **16/16 passed**; cover scratch arguments,
  calibration correctness filtering, sustained telemetry, price/hardware caps,
  ambiguous create recovery without duplicate allocation, setup/hard deadlines,
  failed termination, volume retention, rejected-provisioning storage cleanup,
  and paginated REST responses.
- Nontraining ordered-prefetch comparison: 12 batches starting at step 7 with
  each of 1, 2 and 4 workers; all tensors and parity patches match serial
  preparation across multiple ring-buffer wraparounds.
- Invalid worker/memory/batch/step bounds rejected before dataset access.
- Node/Bash syntax checks and `git diff --check` passed.

Authenticated read-only preflight returned RTX 5090 Secure Cloud list price
**$0.99/GPU-hour**, with **availability NONE**, both with and without the CUDA
12.9 host filter. No paid resource was created. The remote image build,
calibration, sustained GPU utilization and model quality remain unverified until
an actual allocation is available. The earlier SCORE 0 result still stands.

## RTX PRO 6000 deployment profile — 2026-09-29

Switched the authorized rental to one full **RTX PRO 6000 Blackwell Server
Edition, 96 GB**, because the RTX 5090 was unavailable. The shared deployment
profile is archived with the source and copied with the local controller. It
checks the catalog identity/VRAM, allocated host RAM and price separately, then
requires the actual device name, compute capability 12.0, at least 90,000 MiB
device memory and a CUDA 12.9 compatible driver before building. Old bundles
with another profile cannot launch. The two-hour deadline remains unchanged;
the GPU price ceiling is $2.50/hour ($5 for two hours before storage and tax).

Local validation, with no local training:

- Node runner/cloud contracts: **17/17 passed**, including full-device identity,
  driver boundaries, distinct host/device memory requirements, rate rejection,
  provisioning reconciliation and storage retention/cleanup.
- CUDA build and CTest: **13/13 passed**.
- Portable ASan/UBSan CTest: **8/8 passed**.
- Bash/Node syntax checks and `git diff --check` passed.

Remote setup, training health, utilization and quality still require the actual
run. This hardware change does not alter the model, curriculum or audit limits.

### Network-volume extraction correction

The first allocation was terminated after extraction failed, with its persistent
volume retained. On the replacement Pod, the original archive matched its
SHA-256, and replaying extraction reproduced `tar` exit 2: the network volume
rejects restoring uid 1000/gid 100 (`Operation not permitted`). Extraction now
uses `--no-same-owner --no-same-permissions`; audit assets are copied without
preserving ownership. The controller also records setup phases and retains
bounded SSH stderr in a private local diagnostic log, so cleanup does not hide
the original failure. All 17 Node contracts and Node syntax checks passed.
The replacement reuses the existing volume only after confirming the first Pod
was deleted, and retains the original setup/training/rental deadlines.

The corrected extraction, all input checksums and Git restoration passed on the
network volume. The next setup failure was CMake's inability to discover CUDA
through the SSH session. Setup now sets the explicit CUDA compiler/toolkit path
and checks the compiler before installing/downloading dependencies. Diagnostics
were collected with a verified checksum and the replacement Pod and volume were
deleted. No training occurred during either attempt. A separate status-command
branching bug was fixed with a regression covering absent optional state files.
Shortened replacement rentals can retain the original final cutoff; setup stays
bounded to 30 minutes per allocation. All **19 Node tests** pass.

### LibTorch backend registration correction

The next remote build and all 13 existing CTests passed, but the nontraining
prefetch check reported `LibTorch CUDA unavailable`. Re-linking the local
trainer with Ubuntu's `--as-needed` behavior reproduced the failure without
training: `libtorch_cuda.so` disappeared from ELF dependencies even though the
native CUDA runtime and `libc10_cuda.so` remained linked. The Linux link now
retains the CUDA registration library with a scoped `--no-as-needed` setting.
This failure mode is documented in PyTorch's
[CUDA hooks interface](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/detail/CUDAHooksInterface.h).

A new `--check-cuda` CTest compares native device availability with LibTorch,
then computes a known sum of squares on the GPU. It does not create a model,
load data or train. A native GPU with missing Torch hooks fails rather than
being skipped. Local CUDA CTest is now **14/14 passed** and portable ASan/UBSan
remains **8/8 passed**. Remote diagnostics now include trainer shared-library
dependencies. The failed allocation and storage were deleted after verified
collection; no remote training has yet been claimed.

### User-authorized training window extension

The user subsequently requested **two hours for training**, allowing the earlier
cutoff to move. Provisioning/build/calibration now have a separate 30-minute
allowance and collection has ten minutes. Training's two-hour window starts
after calibration, subject to the allocation's reserved absolute limit. The
controller records that measured start and shortens its independent watchdog
deadline accordingly. The maximum new allocation is 160 minutes; at the observed
$2.09/hour that is $5.57 before storage/tax, plus the earlier failed allocations.
All **20 Node contracts** pass, including the delayed training start and shortened
absolute limit. Numerical health and final LOD quality remain separate gates.

The requested full training window continues through unchanged pilot scores and
the previous step cap, while retaining numerical-health failures and the hard
deadline. Deadline-interrupted stages stay unscored; final validation/selection
can follow collection. The Netherlands allocation produced no runtime/SSH
readiness for over 12 minutes, before any source upload. It was terminated and
its known-empty volume deleted. A catalog-validated location override permits
returning to the already exercised Iceland location. **21 Node tests** pass.

Iceland subsequently had no compatible inventory; a North Carolina allocation
became ready and restored the bundle. During setup one SSH connection timed out;
the next connection succeeded less than two seconds later, but the controller
had already entered cleanup. Added up to four bounded retries for repeatable
monitoring, checksums, downloads and the marker-protected job launch. Remote
command failures still fail immediately, and cloud create calls are never
retried by this helper. **22 Node tests** pass, including recovery from transport
failures, preservation of command failures and exhaustion of the retry limit.
The terminated Pod's persistent volume retains the uploaded source and data.

### Measured Blackwell utilization correction

The replacement completed all **14 remote CTests**, ordered prefetch, calibration,
and the full eight-asset readiness audit. At update 8,192, loss/gradients were
finite, model and AdamW restoration were exact, native export error was
**3.82e-6**, and training processed **17.95 million core vertices/second**. The
last 61-second GPU window averaged **83.24%**, with **p10 74%**: the sustained
utilization gate correctly stopped the run. This was not a healthy-run success.
The readiness pilot retained eight unreduced fallbacks and SCORE 0. The results
archive was collected and verified (`def4a6ccca65f4109eb31a19811d77dd6edb5c689730f3d1740c8b4a4833da94`),
then the Pod and volume were deleted.

CPU preparation averaged 43.5 ms/batch and the consumer waited 1.30 ms/update.
The bounded ordered queue now supports up to eight workers/slots, and cloud
allocation requires at least 16 vCPUs. Calibration compares 64/4, 64/8 and 128/8
batch/worker pairs for 75 seconds each, using the production checkpoint cadence.
It records GPU telemetry and selects throughput only among candidates meeting
the unchanged sustained-utilization gate. The initial health segment is longer
to avoid restarting the trainer halfway through the measured minute. Failed
health evidence is now saved explicitly as `health-attempt.json`.

Local validation, without training: **14/14 CUDA CTests**, **8/8 portable
ASan/UBSan CTests**, **22/22 Node contracts**, and an eight-worker ordered-prefetch
comparison over 28 batches (multiple ring wraparounds) passed. New remote
utilization and useful mesh reduction remain to be demonstrated.

### Healthy two-hour Blackwell run — 2026-09-29 08:57 UTC

The next Pod completed all **14 remote CTests** and the 28-batch ordered-prefetch
check. The full GPU has 96 GB VRAM; this allocation provides 16 vCPUs and 188 GB
host RAM at **$2.09/hour**. Training source is committed revision `2aca84e`.
All three 75-second calibration trials passed numerical, restore, native parity,
and sustained GPU gates:

| Batch / workers | Core vertices/s after warmup | Mean GPU activity | GPU p10 |
| --- | ---: | ---: | ---: |
| 64 / 4 | 20.01 million | 91.90% | 89% |
| 64 / 8 | 20.64 million | 92.84% | 90% |
| 128 / 8 | 20.46 million | 95.95% | 95% |

Selected **64 / 8** by highest measured useful throughput among passing trials.
These are single calibration trials. The earlier 83.24% result came from a
different allocation and cannot isolate the effect of worker count.
Calibration models were discarded; the experiment started from the fixed random
seed with the original 66-asset dataset. The complete eight-asset readiness audit
took 43.95 seconds and returned **eight unreduced fallbacks, SCORE 0**, without
resource limits or bake errors.

The fresh run passed health at **update 8,192**:

- Mean first/last 256-update training loss: **0.537584 -> 0.235760**.
- Finite loss and gradients; positive parameter updates; exact model and AdamW
  restoration; native export maximum absolute error **7.63e-6** (limit 2e-4).
- **62 GPU samples over 61.013 seconds**: mean **93.27%**, p10 **91%**.
- Health-segment throughput **19.92 million core vertices/s**; mean data wait
  **0.055 ms/update**, peak Torch allocation **1,677 MiB**, reserved **7,520 MiB**,
  total reported device memory peak **8,223 MiB**.
- Training continued to at least **update 10,240** after passing the gate.

Local evidence: `runs/neural/runpod-pro6000-saturated/live-health.json`, captured
at **08:57:26 UTC**, SHA-256
`8e79c2aea028fbeb65b9d68cd0780a61bfc590fd0faf4c49b94ab6dd09482554`.
It includes raw calibration, health, readiness, source provenance, remote CTest
output, and a running-state snapshot. The independent controller and watchdog
remain responsible for collection and termination. The two-hour experiment
window is **08:54:27–10:54:27 UTC**; the hard rental cutoff is **11:04:27 UTC**.

This establishes a healthy training pipeline and measured GPU activity. Useful
LOD improvement remains unproven; complete later development audits must provide
that evidence. Flat audit scores do not end this requested window early.

### First trained-checkpoint audit — 2026-09-29 09:02 UTC

The **25,000-update** checkpoint completed the frozen development pilot: all
eight assets processed, no bake/resource errors, **eight unreduced fallbacks and
SCORE 0**. Thus there is no demonstrated mesh-quality improvement over the
100-update readiness checkpoint. Lower teacher-training loss and healthy GPU
execution are insufficient evidence of useful LOD reduction.

Model SHA-256: `aa3be5f2d848264d26f48cc3987afa1c276beef31c29a101f86a6c59a27df1c5`.
Pilot run hash: `f9f4b5501ae0f36ad1a78c70e084eda7a61b42e28c3e00b6997581a812db1bb6`.
The complete audit took 61.50 seconds. Raw local snapshot:
`runs/neural/runpod-pro6000-saturated/first-pilot.json`, SHA-256
`b6c1d894f2044440d241d4e4c773a20e5d605e191beb9dd2c7f9b015e51f450a`.
The run continued to at least update 28,672; the next audit is at 50,000 updates.
Further improvement is possible, but has not yet been observed. The authorized
two-hour window and unchanged numerical-health gates remain in force.
