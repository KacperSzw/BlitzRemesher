# Sequential endpoint learning handoff — 2026-10-02

Worktree: `/home/kacper/Projects/BlitzRemesher-neural`. Branch: `main`;
`origin/main` is the default. The user requested finishing this part and pushing
so another agent can continue. No new rental was created in this part.

## Result and next decision

The full endpoint scorer learns the current labels (97.5–97.8% preferred
membership across three seeds), but its ordinary UV-on rock rollout regressed
about 18%. A concrete support diagnostic now recovers that regression:

| UV-on moon rock, seed 101 | LOD1 | LOD2 | LOD3 |
| --- | ---: | ---: | ---: |
| Initializer, ordinary runtime | 1,638 | 820 | 410 |
| Trained scorer, ordinary runtime | 1,936 | 968 | 484 |
| Initializer, frozen collector pool | 1,638 | 820 | 410 |
| Trained scorer, identical collector pool | 1,256 | 628 | 314 |

Both supported runs completed with exact source and adjacent visual gates.
They share source, binary, visual and execution hashes. The trained global top
escaped the teacher's 16-action pool in 3,705/3,709 rescoring events (99.892%);
the initializer had zero escapes. Within that support, the trained scorer uses
about 23% fewer triangles at every rock LOD. This is a conditional local learning
signal on one development rock, one seed and UV-on. Support construction also
runs collector inference and geometric rankings; the result does not establish
an independent neural deployment improvement or pass the remote gate.

The next focused step is to teach the trained scorer's globally preferred
unsupported actions, rather than repeat more optimizer steps on the same narrow
pools. First inspect `training/runtime_teacher.hpp` and the `PolicyMixed`
selector in `src/neural/action_gpu.cu`; preserve the actual trajectory actor,
exact endpoint outcomes, all four gates, target eligibility and unknown outcomes.
A distinct query policy would need its own model hash in the teacher contract,
without silently relabeling the original collector or admitting inconsistent
placement labels. This proposal is not implemented yet. A further support
comparison on shelves and UV-off is also needed before extrapolating the signal.
No architectural feature or objective change was made for the support diagnostic.

Then train fresh local seeds, compare ordinary neural output against matched
initializer/constant controls on the two smoke assets in both UV modes, and only
escalate a promising result to the frozen 12-family small-source pilot. Keep the
predeclared thresholds: every seed/mode must improve category-balanced retained
triangles by at least 5%, with at most 2% increase at any LOD and complete visual
and control checks. See [the gate](sequential-pilot-gate.json),
[the runner](../../scripts/neural/sequential-pilot.mjs) and
[the investigation](POLICY-ALIGNMENT.md). Do not revise the gate after seeing
results. Shading and full-size source quality remain later checks.

## Code and local artifacts

- `--objective endpoint-ranking` trains the two hidden layers and score row
  (12,481 parameters for width 64). It preserves the other output weights, but
  their predictions change, so BLZNET03 requires explicit Reuse. Automatic and
  Rebuild use reject the scorer. Checkpoints, asynchronous export, model-only
  restoration and resident weight refresh preserve the restriction. Legacy
  joint/ranking-row optimizer v2 and BLZNET01/02 meanings remain intact.
- Dataset admission is extracted into `training/action_dataset.hpp` and binds
  version-10 runtime teaching, original collector payload, shard-index categories,
  requests, trajectory, geometry mask and exact source reference.
- Runtime per-collapse gates use conservative sparse certificates. Final emitted
  errors and teacher margins remain exact. Ordered traces, final FP32 meshes and
  errors matched the old exact path on both smoke assets in both UV modes.
- An interrupted source-only result now exports without attempting a selector
  that requires at least two levels. The failing regression and repair are saved.
- Support restriction is internal diagnostic plumbing, enabled only by
  `ranking_support` in a native model-audit settings file; no public runtime flag.
  The fixture proves that rejection of all supported actions cannot fall through
  to unsupported global actions. The original observer parity test caught and
  verified the repair of a stale host action count introduced during this work.

The exact local artifacts remain in the shared ignored `runs/neural` directory:

- `v4-initialization/model.blzn`: original v4 initializer; SHA-256
  `306ec18cfeef36ef8af16ef2a9d008021e142d5bf114b4ca8ea243debf9d2fae`.
- `sequential-dataset-01`: 16 verified copied shards, 2,173 observed/1,720
  informative states, four parents, two source scales and both UV modes.
- `endpoint-ranking-01/training/{101,211,307}/step-512.blzn` and matching
  `step-512.pt`: fresh trained scorers and optimizer checkpoints. The first
  complete UV-on comparison is negative; further audits were explicitly stopped.
- `endpoint-support-01` and `endpoint-support-initial-01`: finished support
  diagnostics, exact inputs, raw traces, hypotheses, execution logs and results.
  `endpoint-support-01/control-comparison.json` contains the matched control.
- `endpoint-plane-probe-01`: both UV-mode classical comparator diagnostics;
  rock improves, some shelves levels regress. Not neural success.
- `endpoint-diversity-01`: one new training-rock teaching shard and descriptive
  label probe. Keep the invalid bitmap-parser result and explicit correction;
  the corrected shard has no fully-passing same-removal margin pairs.
- `next-step-reviews-01`: concrete Prophet reviews, including endpoint scope
  repairs, support-mismatch diagnosis and a conditional remote runner proposal.

Machine-readable reports, diagnostic scripts and logs are committed compressed
under [the evidence manifest](evidence/policy-alignment/manifest.json), with
uncompressed SHA-256 and byte counts. Model/checkpoint/data blobs remain local;
a fresh clone needs those exact artifacts and corpus files. Do not treat missing
blobs as permission to substitute a different model or teacher dataset.

## Training and spending

No long training or new remote rental was started in this part. The user permits
at most $8–10 for the next remote run, conditional on successful local output
validation. The current frozen pilot is not passed. Older rentals were collected,
verified and terminated. Do not reuse an old run's remaining-budget calculation
as the new grant ledger.

The existing `policy-ranking.mjs` wrapper has a three-minute per-seed process
allowance; its current experiment is not a remote hour-training job. Native
`action_train` also caps one run at 1,000,000 updates and 50 minutes, and currently
forbids `--warmstart` for ranking objectives. Longer orchestration therefore needs
explicit collector identity and checkpoint-continuation contracts. Teacher
generation still dominates useful experiment work; the short local full-scorer
optimizer fits took roughly 1.2 seconds per seed, not an hour of useful learning.
The earlier
Prophet remote proposal is a design only: it still needs canonical
`scripts/neural/runpod.mjs` integration, a fresh grant ledger, bounded setup and
experiment phases, immutable artifact staging, hash-verified collection and the
independent cutoff watchdog. Setup completing must not be reported as training
starting. Verify actual learner updates and a healthy checkpoint on the rental.

## Validation

Final CPU CTests: 4/4 model scope, dataset, JS ranking/pilot admission and I/O;
3/3 neural core/action/chain. ASan/UBSan: 3/3 model scope, dataset and I/O.
Native CUDA CTests: 7/7 action executor, device update, resident cycle, schema,
checkpoint, model scope and dataset; observer parity/support fixture 1/1.
CUDA Compute Sanitizer memcheck passed the observer/support fixture with zero
errors. The final evidence manifest retains all these logs.
Raw failures, corrected checks, negative runs and stopped audits are retained.
These are implementation checks; the single-rock support result is not the
full-pilot quality gate.

## Earlier historical handoff

### Action learning handoff — 2026-09-29

Worktree: `/home/kacper/Projects/BlitzRemesher-neural`.
Branch: `research/neural-lod-gpu`. The primary worktree is separate.

## Current milestone

**The staged run is complete and cloud resources are deleted.** All three seeds
reached 16,384 updates (49,152 total); both full-settings screening comparisons
completed with clean health. The final archive and checkpoint hashes were
verified locally. Provider verification found zero Pods and network volumes.
See [final evidence](evidence/action-screening-v3/README.md).
An [offline interactive viewer](viewer/index.html) compares the actual source,
constant and network meshes with synchronized cameras, all LODs, saved normals,
UV checker and changed-face highlights. See [viewer provenance](viewer/README.md).
The [learning-loop explanation](viewer/learning-loop.html) shows CPU/GPU work,
audit internals and evidence-derived timing charts, including the explicitly
unclassified setup/process/collection remainder.

Seed 101 beat constant ranking on both diagnostic meshes at both checkpoints.
Mean triangle reduction across generated LODs at the second checkpoint was
0.491% versus 0.164% for shelves, and 2.369% versus 0.190% for moon rock 02.
The learned comparison took 584.066 seconds versus 104.756 seconds (5.58x).
These two meshes and one evaluated seed establish an early signal, not the
full pilot superiority gate or release quality; there is no aggregate SCORE.

The rental ended at **18:11:51 Warsaw / 16:11:51 UTC on 2026-09-29**. Conservative
spending across the new $8 grant is **$5.747**, including prior attempts and
storage/rate allowances. Recalculate the ledger before any further paid work.
Training averaged only 10.17% GPU busy time, while audits averaged 95.38%.
Next priorities are profiling trainer launch/synchronization and audit cost,
then a budgeted full comparison if the measured improvements justify it.
No new rental or training was started while collecting this final result.

### Earlier milestone while the comparison was running

**Healthy remote training reached.** On H200, all three seeds completed 8,192
updates on the expanded 236-state curriculum (24,576 updates total). Loss fell
from about 1.35 to 0.0153–0.0212. All numerical, export and optimizer restore
checks passed. All 19 remote tests passed. Checkpoints, models, labels and logs
are saved in [action-screening-v3](evidence/action-screening-v3/README.md).

The constant-versus-seed-101 comparison is currently running. Quality superiority
is unproven. Audit GPU busy time averaged 95.25%, but training averaged only
10.77%; do not claim full GPU saturation. The small trainer needs a targeted
launch/synchronization profile before further performance changes.

Active directory: `runs/neural/runpod-action-screening-h200-01`, source `46ba983`.
Hard cutoff: **18:25:55 Warsaw / 16:25:55 UTC on 2026-09-29**. The independent
watchdog and automatic collection/cleanup are active. Maximum additional
reservation including previous attempts and reserve: **$7.827003 of $8**.
Check its final report and verified cleanup before any further paid work.
No local optimizer steps ran. The following entries preserve earlier failures.

### Scheduling correction before healthy training

The H200 full-settings baseline is now complete and healthy on both diagnostic
meshes, with all 19 remote tests passing. Shelves took 34.020 seconds (2.41x faster
than optimized Blackwell, exact output and audit parity); moon rock took 413.364
seconds. The run then stopped without training because its schedule reserved
over an hour for nine comparisons. Its archive is verified and all resources
were deleted. Conservative additional spend is $2.217003 of the new $8.

The corrected plan in [ACCELERATION.md](ACCELERATION.md) trains all three seeds
after verified labels, then screens seed 101 against constant ranking on the
same two assets with full visual settings. All nine methods and all three seeds
are still required for the separate full-pilot quality gate. The new H200 rental
cap is 60 minutes; including previous spending, storage and a $1 reserve its
maximum additional reservation is $7.827003. Thirty Node contracts and both
affected CTests pass. No optimizer steps have run locally or in either new
rental at this commit. Next directory: `runs/neural/runpod-action-screening-h200-01`.

### Earlier attempts in this milestone

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
