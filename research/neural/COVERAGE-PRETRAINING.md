# Coverage pretraining implementation record

The coverage profile now uses one conservative Vulkan R8 pass, no depth or
shading attachments, one-bit owned reference masks and direct candidate surfaces.
It skips appearance search and uses ten position-only teacher alternatives plus
an optional policy proposal. Normal targets and normal edits are disabled in this
profile. Immutable audit sources remain FP32; working meshes and training shards
remain compact on disk and in transfers, decoded into FP32 GPU working state.

Packed positions allow sparse exact FP32 exceptions capped at 5% of referenced
surviving vertex IDs, including separate wedges. The bitmap and sparse values
survive packing, uploads, GPU trials, commits, compaction, reset, cache and replay.
Both free placement and vertex reuse enforce the surviving-vertex cap. The old
strict replay remains strict; repair is a new, fully audited representation.

Predecessors pass both preparation and destination pixel sizes. Audited exhausted
or unavailable predecessors remain explicit outcomes. Coverage quality gates,
geometry, resources and finite arithmetic remain blocking. Scheduled full shading
audits are diagnostic for shape pretraining. This checkpoint is not release-quality
proof and needs later appearance-aware fine-tuning and unseen-model evaluation.

The cycle owns one resident dataset, model and optimizer. Completed phase history
is append-only; the recovery journal contains the active dataset window and durable
history offset. Recovery excludes downtime from completed learning time. Learning
and finalization have separate budgets: 120 minutes plus ten minutes, respectively.
Changed objective means original weights with fresh Adam and fresh examples.

## Local evidence (RTX 2080, shared workstation)

[Architecture and timing hierarchy](evidence/coverage-local/ARCHITECTURE.md).
All raw measurements and checks are in [coverage-local](evidence/coverage-local/).
These are development smoke measurements, not an aggregate corpus score.

| Five alternating matched runs; 2 teacher states, 1024 updates | Median cycle |
|---|---:|
| Coverage with full attachments, reference optimizer | 1.123499 s |
| Mask-only coverage, reference optimizer | 1.072565 s |
| Mask-only coverage, fused optimizer | 1.024252 s |

The mask implementation reduces whole-cycle time by 4.53% (1.047×), and measured
teacher audit peak allocation by 41.65% (211,875,468 → 123,623,796 bytes).
Coverage variants produce identical training data; both reference-optimizer
variants produce identical weight payloads. Fused gradients and restart pass the
numerical contracts. Its final local median win over mask/reference is 4.50%, below
the 5% selection threshold; the remote benchmark will select its own backend.
Earlier local repeats are retained, including an 8.3% fused win, showing contention.

The four-view raster profile reports roughly 49–52 µs for full attachments and
28–30 µs for mask-only GPU drawing. Renderer allocation falls from 19,329,678 to
6,702,862 bytes (65.3%). Coverage pixels match exactly. Packing/interop wall times
are noisy on this shared GPU, including slower mask samples; raw rows are retained.
Do not multiply isolated GPU drawing speedups into full-cycle claims.

A separate five-repeat objective comparison measured median teacher time 0.822173 s
for appearance-aware teaching versus 0.366975 s for shape-only teaching (2.24×).
That larger saving changes the learning objective; it is not renderer acceleration.

The packed curriculum passed all 48 conditions in 13.812 s: 40 complete, six audited
search-exhausted and two predecessor-unavailable, with data in all four categories.
The historical moon-rock packed failure still reproduces. Promoting three of 1847
referenced vertices (0.1624%) repairs it under the unchanged 2 px limit: 1.171382 px
in the full 16-view audit. GPU state and compact snapshot pass; search audit also
passes at 1.914215 px. No visual threshold was loosened.

The duration smoke completed 60,000 ms of learning, 104,448 updates and 102 teacher
iterations, then final audits/publication; total 74.077 s. Coverage and shading
audits passed for constant and learned ranking. Journal size was 2822 bytes versus
379,582 bytes of append-only history. This is a lifecycle proof on one condition,
not evidence that those optimizer updates improve unseen models.

Validation: 28/28 CTest; 13/13 ASan/UBSan; 22 Node tests; CUDA memcheck zero errors
for action, Vulkan and resident training contracts. Vulkan validation found an
unused-input mismatch in the new coverage shader; fixed and rerun with zero
validation errors. Existing unused full-shader attachment output warnings remain
and are preserved in the compressed log. Sparse shader lookup, cap boundaries,
failed-then-valid reuse, exact optimizer restart and history recovery are covered.

## Remote launch contract

Latest future-spending authorization: $8 total, replacing the previous remaining
$2. Grant baseline is pinned to the reconciled ledger; retries consume this same
grant. At most 180 rental minutes, independent deadline watchdog, verified result
collection before volume deletion. Prefer a graphics-capable 24 GB or larger GPU.

Remote gates repeat CTest during setup, action/Vulkan memcheck, Vulkan validation,
resident optimizer/restart checks, historical strict replay, sparse repair,
cross-GPU export replay, five matched timing repeats, packed 48-condition pilot
and hardware raster parity. The remote architecture/timing artifact is written
before starting the 120-minute full learning cycle. A failed gate stops the job.

Original warm-start SHA256:
`9152bb42cb807a2e91fe3217ab6dc3bbcf11be12618bcd706acf71a8e3fff185`.
Remote preflight has passed on L40S; the two-hour training run is active.
Completion and resulting model quality are not yet established. See the record below.

## Next bottlenecks

Candidate audits remain the largest teacher stage. The next substantial change
should batch candidate × view work and GPU decision reduction to remove repeated
host waits; benchmark the full cycle with unchanged candidates and coverage gates.
The small MLP update and frequent checkpoint publication also matter. Keep setup
and graph capture amortized in the resident run; use its telemetry to decide
whether larger update batches or a bounded checkpoint cadence justify further work.
Packing precision and visual thresholds should remain unchanged for those controls.

## First remote attempt

The first allocated rental (setup-02, Blackwell Server) stopped during compilation:
`blitz-neural-placement-prepare` included CUDA implementation headers without an
explicit CMake CUDA dependency. Nix's ambient include path had hidden this existing
portability bug. The target now declares CUDA runtime and cuBLAS dependencies;
its generated compile command contains the toolkit include directory explicitly.
The target rebuilt and its bounded coverage preparation smoke passed locally.

No training started. Failure evidence was checksum-collected, and both compute and
volume deletion were verified. Conservative billed-time upper estimate: $0.4311,
charged to the same $8 grant. [Failure evidence](evidence/coverage-remote/setup-02/).
A retry remains subject to the original pinned cumulative cap.


## Validated remote launch (L40S, retry-04)

[Remote architecture and timing hierarchy](evidence/coverage-remote/validated-04/ARCHITECTURE.md)
and [raw validation](evidence/coverage-remote/validated-04/validation.json).
Source revision: f336a53. NVIDIA L40S, 46,068 MiB reported VRAM, driver 580.159.04.
All 28 CTest cases pass, CUDA memory checks have zero errors, Vulkan validation
has no errors, both backends pass exact restart, and cross-GPU export replay passes.
The original strict moon failure still reproduces and the sparse repair passes.
The packed coverage pilot passes all 48 conditions in 19.565 s.

Five-repeat medians, identical coverage data and 1024-update work budgets:

| L40S cold cycle | Seconds |
|---|---:|
| Full attachments + reference optimizer | 1.479830 |
| Mask-only + reference optimizer | 1.406557 |
| Mask-only + fused optimizer | 1.285885 |

Mask-only saves 4.95%; fused updates save another 8.58% against mask/reference.
The combined whole-cycle reduction is 13.11%, so fused is selected. GPU raster
median is 35.6 → 20.2 µs/view, with identical mask pixels. These cold cycle times
include startup, filesystem and final publication; do not compare GPUs solely
from those totals. Persistent operation amortizes setup and capture.

The full learning cycle launched at 2026-09-30 16:46:05 UTC (18:46:05 Warsaw),
with 120 learning minutes and ten additional finalization minutes. Expected end
of learning is approximately 18:46 UTC / 20:46 Warsaw. Fresh Adam, original weights,
compact data and packed geometry, shape-only teaching, scheduled nonblocking
shading diagnostics. The controller verifies collection and then deletes compute
and storage. Budget is cumulative with the failed setup, below the $8 grant.

[First 112 seconds](evidence/coverage-remote/validated-04/early-training.json):
196,608 updates, about 1754 updates/s, 824 generated teacher states, 30,247 teacher
queries, zero failed conditions and 16 audited empty conditions. About 43% GPU
utilization and 1022 MiB peak GPU allocation reported by telemetry. At that early
pace, 120 minutes projects to 12.6 million updates; this is not a promise of model
quality or sustained throughput. Detailed history and telemetry snapshots are
preserved alongside the summary. Independent file reads can differ by one phase.

Initial persistent time split is approximately 62% teacher generation, 20%
optimizer computation and 18% checkpoint handling. The most useful next work is
reducing teacher host waits and checkpoint stalls, then testing overlap of teaching
and optimizer work with explicit weight snapshots and stream ownership. Memory
capacity is not the current bottleneck. Preserve quality gates and measure the
full cycle when testing those changes.

## Interpreting update counts and GPU activity

The [11-minute snapshot](evidence/coverage-remote/validated-04/eleven-minute-training.json)
records 1,054,208 updates in 670.044 learning seconds: 1573 updates/s, projecting
11.33 million updates in two hours if that pace persists. This supersedes the
first 112-second throughput estimate for this observation window; neither is a
completed run. There are 4376 generated teacher states and zero failed conditions.
The scheduled 30-minute quality audit has not yet occurred.

| Completed phase wall time | Seconds | Share |
|---|---:|---:|
| Teacher generation, including candidate audits | 395.037 | 59.19% |
| Optimizer updates | 119.948 | 17.97% |
| Checkpoint handling | 150.001 | 22.47% |

Other training work accounts for the remainder. Teacher candidate audits alone
take 207.934 s, 31.15% of completed phase time. These are wall-time categories,
not an attribution of GPU idle time. The teacher also contains GPU work.
GPU activity averages 40.86% during learning and 37.97% over the last five minutes,
with individual samples spanning 0–95%. NVIDIA defines this counter as the fraction
of a sampling interval with GPU kernel execution; it is not a percentage of peak
arithmetic throughput. See [NVIDIA's utilization documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html#utilization).

The loop generates one shard, updates the policy, and then generates the next.
It publishes a verified checkpoint every 512 updates, roughly three publications
per second at this measured pace. CheckpointWriter has one snapshot in flight;
the next submission joins the previous publication, and snapshot/parity checks
also cause host/device transfers. Candidate audits batch only two alternatives.
The 13,196-parameter 128→64→64→12 network has little arithmetic per update.
Together, this code structure and these timings point to insufficient overlapping
work and excessive checkpoint cadence. A GPU timeline is still needed to attribute
each idle gap precisely. The active experiment remains unchanged.

Next controlled experiments should prioritize time-based checkpoint publication,
independent mesh/candidate batching, and overlapping teacher generation with
updates against immutable policy snapshots. Preserve recovery and visual gates.
Even halving optimizer time alone would save only about 9% of the measured full
cycle; teacher scheduling offers a larger opportunity.

The current curriculum contains 12 training assets, with 48 conditions. An update
samples 512 local states, each containing up to four action rows; it does not see
512 independent meshes. The loop currently performs 2048 updates per new shard,
which produced 8 or 10 states in this snapshot (104 conditions were audited empty).
Thus millions of updates imply heavy replay. They do not establish generalization
to unseen assets, and greater optimizer throughput alone may not improve quality.
Compare smaller update budgets per fresh shard at equal total GPU time on the
development protocol before changing the next training schedule. Keep release
held-out assets reserved for release audits.

Related primary sources checked on 2026-09-30:

- [Neural Mesh Simplification, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Potamias_Neural_Mesh_Simplification_CVPR_2022_paper.pdf):
  150 epochs on a train/test split of the 80-mesh TOSCA dataset; learned sampling
  and triangulation. No directly comparable updates/s training measurement found.
- [SFSP-QEM, 2025](https://www.techscience.com/cmc/v83n2/60527/html):
  150-epoch schedule, TOSCA split 80%/20% (64 training and 16 test meshes), learned
  feature-preserving sampling guiding QEM. Their geometry metrics differ from our
  audited screen-space limits.
- [GNN-guided QEM, 2026 author preprint](https://www.preprints.org/manuscript/202604.1809):
  50 epochs, batch one mesh, 64 training/16 test meshes per cross-validation fold,
  RTX 2060 SUPER. The [official training loop](https://github.com/Geo3D-AI-CSU/GNN-QEM/blob/main/train_edge_importance.py)
  takes one optimizer step per mesh batch, implying about 3200 whole-mesh updates
  per fold under the reported settings. This is a calculation, not a published
  measured step count. Each graph update sees mesh-wide data, unlike our local
  action minibatches; comparing their step count directly to ours is invalid.

Prophet research was attempted at high effort but failed with a browser automation
error; the sources above were checked directly. No cross-paper speedup or quality
ranking is claimed.
