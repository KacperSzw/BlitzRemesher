# Bounded RTX PRO 6000 experiment

This implements the approved from-scratch experiment on one Runpod Secure Cloud
RTX PRO 6000 Blackwell Server Edition (full 96 GB GPU). Local training is not part of deployment validation. The
existing model's audited pilot still has SCORE 0; this migration does not change
the curriculum, decoder, quality limits, or release status.

## Closed run — 2026-09-29

**Current status:** this run ended early at 09:07:44 UTC, update 52,224,
on native export disagreement. Latest validated checkpoint: 51,200.
Collection was checksum verified; compute termination and volume deletion
are recorded in `collection.json` and `rental.json`. There is no active run.
The 25k and 50k complete pilots both returned SCORE 0. See V2.md before any
new rental; its cumulative $10 budget supersedes the older per-run schedule.
The timing and commands below describe the closed experiment.

The run in `runs/neural/runpod-pro6000-saturated` passed its initial health gate
and continued training at 08:57 UTC. Source revision: `2aca84e`; batch 64, eight
workers; mean GPU activity 93.27%, p10 91% over 61 seconds. Raw evidence and source
provenance are saved locally in that directory's `live-health.json`; REPORT.md
records its checksum. Initial readiness completed all eight meshes but returned
eight unreduced fallbacks, so this is not evidence of useful LOD improvement.
At 09:02 UTC the first trained-checkpoint audit (25,000 updates) also reported
eight unreduced fallbacks and SCORE 0. Its complete raw summary is saved in
`first-pilot.json`; training continued toward the next audit at 50,000 updates.

The two-hour experiment ends at **10:54:27 UTC / 12:54:27 Europe/Warsaw**. The
independent watchdog's final rental cutoff is **11:04:27 UTC / 13:04:27 Warsaw**.
Health failures may stop earlier. The controller collects a checksummed archive,
deletes compute, then deletes persistent storage after verified collection.

```sh
node research/neural/runpod.mjs status runs/neural/runpod-pro6000-saturated
systemctl --user is-active blitz-a7a0caed-3cd9-449b-a79a-5c50a30b42ce-control.service blitz-a7a0caed-3cd9-449b-a79a-5c50a30b42ce-watchdog.service
```

Keep the workstation awake and online. Check `collection.json`, `rental.json`
and `watchdog.json` before considering recovery. When the run ends, inspect the
archive's report, complete pilot rows and latest checkpoint; a normal full-window
exit can have `training_window_exhausted: true` and `complete: false`, because
final validation and model selection have not run. An interrupted audit has no
score. Retain the results even if every complete pilot returns SCORE 0.
Do not create another rental merely to inspect this run.

## Steps and gates

1. **Prepare locally.** Commit and push this branch. `prepare` refuses unpushed
   or uncommitted source. Package its Git bundle, the original 66-asset teacher
   dataset, and the frozen development pilot plus validation assets. Check each
   shard/file against its manifest. No refined labels, old models, AdamW state,
   held-out assets, or credentials enter the upload.
2. **Provision within the budget.** Query REST v2 catalog availability and the
   current Secure Cloud GPU list price. Refuse anything above $2.50/GPU-hour or a
   different GPU. Require 16 vCPU and 32 GB host RAM, a 50 GB container disk and a
   20 GB STANDARD network volume in the same data center. Recheck the allocated
   hardware and hourly rate. No spot instance or GPU substitution.
3. **Set up and calibrate within 30 minutes.** Use the pinned official base image
   and LibTorch archive. Build native CUDA for architecture 120, run CTest and a
   nontraining ordered-prefetch check, then compare batch/worker pairs 64/4,
   64/8 and 128/8 for 75 seconds each. Calibration gets at most five minutes and a 12 GiB
   LibTorch allocator cap. Select useful core vertices/second only among trials
   passing finite-gradient, parameter-update, exact checkpoint/AdamW restore,
   native-export parity checks and at least 60 seconds of sustained GPU use
   (mean >=90%, p10 >=85%). Discard their weights before the real run.
4. **Train and audit.** Start random weights with the fixed seed, checkpoint after
   100 updates, complete all eight readiness assets, then require decreasing
   moving-average loss and at least 60 seconds of training telemetry (mean GPU
   utilization >=90%, p10 >=85%). Each 25,000-update stage gets the frozen pilot.
   The requested full window continues past stalled pilot scores and the former
   100,000-update cap; numerical/health failures still stop it. Checkpoints and
   complete pilot rows are retained. A deadline-interrupted stage stays unscored;
   final checkpoint selection/validation can follow after collection. Never tune
   against held-out assets.
5. **Collect and terminate.** Allow two hours after setup/calibration, download the immutable
   results archive and verify SHA-256. Delete the Pod, confirm termination, then
   delete the volume only after verified collection. Failure to collect retains
   the volume and its ongoing storage charge for recovery.
   A definitive rejected Pod request also deletes its newly created, unattached
   empty volume; an ambiguous create response requires reconciliation.

The user extended the budget to a two-hour training window after setup on
2026-09-29. Setup/calibration have up to 30 minutes; collection gets ten more.
The initial maximum rental is 160 minutes. Once setup finishes, the controller
shortens the final cutoff to the actual training deadline plus collection time.
The training window includes bootstrap, readiness and periodic quality audits;
health failures can stop early. Both the local controller and an
independent user systemd watchdog request termination. Durable create intents,
unique resource names and reconciliation prevent blind duplicate provisioning
after a timeout. Unknown create outcomes stay unresolved until reconciled.
Repeatable SSH monitoring and downloads tolerate up to four transport failures;
command errors still stop the run. Cloud create requests are not blindly retried.

**Keep this workstation awake and online until termination is confirmed.** The
watchdog is local; a power loss or a provider/network outage can delay API
termination beyond the target deadline. No unsupported provider timer is assumed.
Services restart after failure and at the next user session with the original
absolute deadline. They never extend the budget. Do not disable the watchdog to
stop training: use the stop command below. The API key stays local and is never
included in Pod environment variables, shell command arguments, or result logs.

The 12 GiB cap covers Torch allocations, not CUDA contexts, native evaluator
scratch, or driver memory. Four to eight pinned host slots remain bounded. The remote
training supervisor samples process-group RSS and stops above 24 GiB; this is a
sampled guard, not a kernel memory reservation. Compilation uses two workers.
CPU packing, mesh decoding, and final audits still run on CPU.

The shared `runpod-profile.mjs` fixes the GPU identity, 96 GB catalog VRAM,
90,000 MiB minimum reported device memory, compute capability 12.0, and price
cap. The API Pod `gpu.memory` field describes host RAM, which remains a separate
32 GB minimum. The profile is copied with the local controller and archived
source; a bundle prepared for another profile cannot be launched. The observed
$2.09/hour rate gives $4.18 for two GPU hours before storage and applicable tax.

## Commands

From this worktree with Node, Git, SSH, tar and user systemd available:

```sh
node research/neural/runpod.mjs prepare runs/neural/runpod-pro6000-first
node research/neural/runpod.mjs launch runs/neural/runpod-pro6000-first
node research/neural/runpod.mjs status runs/neural/runpod-pro6000-first
node research/neural/runpod.mjs stop runs/neural/runpod-pro6000-first
```

`launch` accepts an optional absolute deadline in milliseconds after the run
directory to shorten a replacement rental to the original budget cutoff. It
cannot extend a rental beyond 160 minutes and keeps ten minutes for collection.
Confirm the previous Pod is deleted before launching a replacement. Each new
allocation has at most 30 minutes for setup, within that shared final cutoff.

`prepare` is offline except for checking the pushed Git revision. `launch` rents
the approved resources; don't invoke it just to validate the scripts. Before
launch, save the account API key in `~/.config/blitz/runpod-api-key` with mode 600.
The account needs sufficient credit and permission to read catalog resources and
create/read/delete Pods and network volumes. A dedicated SSH identity is created
locally; only its public key reaches the Pod. No manual template deployment or
SSH setup is needed after onboarding.

Set `BLITZ_RUNPOD_DATA_CENTER` for a specific catalog location when recovering
from a provider startup failure. It must still offer the approved GPU, price
and STANDARD storage; the launcher does not substitute another location.

`rental.json` records IDs, quote, absolute deadlines and termination confirmation;
`watchdog.json` records independent cleanup attempts; `collection.json` confirms
the downloaded archive checksum. `transport-errors.jsonl` retains private SSH
failure diagnostics, while `rental.json` identifies the setup phase. Extraction
does not preserve workstation ownership on the provider's network volume.
The control service name is `<rental-name>-control`
and the independent watchdog is `<rental-name>-watchdog`. Inspect them with
`systemctl --user status` / `journalctl --user -u`. An interrupted controller can
be restarted with `systemctl --user restart <rental-name>-control`; do not launch
another experiment directory as a recovery action.

`results.tar.gz` contains bootstrap/build and CTest logs, package/compiler/GPU
versions, calibration measurements, the experiment's source provenance, per-step
loss/gradient/timing data, GPU telemetry, model and optimizer checkpoints, audit
rows and the final report. A finite-camera audit and a healthy training loop are
separate from demonstrating a useful reduction. Incomplete audits stay unscored.

## Pinned dependencies and provider references

- Image: `runpod/base:1.0.7-cuda1290-ubuntu2404` at
  `sha256:c776d549e38023c51a28c25267e029ec2aa53a525c87a934b77694d76572c629`.
- LibTorch 2.10.0 cu128 CXX11 ABI, SHA-256
  `429aa9fead3cf3d557e7c310442a1fae3879cdc14a469ff452043b39b61666a9`;
  the native compiler is CUDA 12.9. Driver >=575.51.03 is checked.
- Ubuntu apt dependency versions are recorded after installation; its rolling
  repository packages are not digest-pinned. Remote compatibility is established
  by the actual build, CTest and native/Torch comparison, not by the image alone.
- [REST v2 overview](https://docs.runpod.io/api-reference-v2/overview),
  [live OpenAPI schema](https://api.runpod.io/v2/openapi.json),
  [GPU catalog](https://docs.runpod.io/api-reference-v2/catalog/list-gpu-types),
  [Pod creation](https://docs.runpod.io/api-reference-v2/pods/create-a-pod),
  [network volume creation](https://docs.runpod.io/api-reference-v2/network-volumes/create-a-network-volume),
  [credentials](https://docs.runpod.io/get-started/credentials).

At the GPU price cap, 160 minutes cost at most $6.67 for GPU time, plus container disk,
network storage and applicable taxes. Catalog price and actual Pod rate are both
checked; the API does not offer an atomic client price ceiling, so a quote change
can result in a short rejected allocation before deletion. Preserved network
storage continues billing until explicitly recovered and deleted.
