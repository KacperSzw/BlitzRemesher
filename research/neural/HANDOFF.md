# Action learning handoff — 2026-09-29

Worktree: `/home/kacper/Projects/BlitzRemesher-neural`.
Branch: `research/neural-lod-gpu`. The primary worktree is separate.
The branch includes the endpoint/curriculum models, audited labels, CLI repair
and tests. The rental state records its exact source revision.

## Current experiment

Local durable state: `runs/neural/runpod-action-pilot-03/rental.json`.
The continuation uses one RTX PRO 6000 Blackwell Server Edition (96 GB),
last quoted at $2.09/hour. The initial rental limit is 90 minutes; the controller
shortens it after setup to the actual 50-minute experiment deadline plus
collection time. Read the state file for the effective cutoff and resource status.

```sh
node research/neural/runpod.mjs status runs/neural/runpod-action-pilot-03
```

The controller and independent watchdog run as user systemd services. Their
names derive from `rental.json`'s `name` field. They collect and verify
`results.tar.gz`, terminate compute, and delete the volume after verified
collection. A collection failure preserves the volume and its evidence.
`stop RUN` requests early termination; do not delete uncollected evidence.

Remote result root: `/workspace/results/action-v2-pilot`.

- `curriculum/progress.json`: completed data shards.
- `progress.json`: checkpoint health and matched comparison progress.
- `seed-{101,202,303}/latest.json`: model, optimizer, numeric and membership checks.
- `pilot-{1,2}/report.json`: all matched methods and persistence gate.
- `result.json`: final completion, stop reason and next phase.
- `gpu.jsonl`: utilization split by preparation, training and audit phase.

The continuation reuses seven checksummed shards from two allowed training
assets at several pixel sizes and fixed preceding LODs, then trains three fresh
seeds at 8,192 and 16,384 updates. All
methods use eight trials, batches up to 32, 16 GiB evaluator scratch and the
unchanged full visual gates in `action-pilot.json`. No third stage is automatic.
The experiment is bounded; incomplete pilots remain unscored.

The first curriculum rental, `runpod-action-pilot-01`, has ended. All three
seeds reached 8,192 updates with healthy numeric/restore checks and 98.78–99.39%
queried preferred membership. The pilot failed at a missing CLI option; that
interface is repaired and tested. Evidence was collected and both resources
deleted. Saved curriculum/model evidence is under `evidence/action-curriculum-v2`.

The second allocation (`runpod-action-pilot-02`) exposed no public IP/direct
TCP mapping despite a running container and requested port 22. It was stopped
before any upload; the newly created unused volume was verified and removed.
The third allocation uses another data center and the same tested source.

Verified snapshot at **12:27 UTC (14:27 Warsaw)**: all 18 cloud CTests passed,
all seven saved shards passed reuse checks, and all three seeds completed 8,192
updates with finite parameters/gradients, exact model/AdamW restoration and
native/FP64 error below 2e-4. Loss fell from approximately 1.35 to 0.020;
queried preferred membership was 98.78–99.39%. The repaired benchmark passed
argument parsing and entered the GPU audit phase. No full-pilot score exists
yet. Raw training checks and telemetry are in
[repeat-8192.json](evidence/action-curriculum-v2/repeat-8192.json).

The effective experiment deadline is **13:14 UTC (15:14 Warsaw)** and the
collection/compute cutoff is **13:24 UTC (15:24 Warsaw)** on September 29.
Early completion or a failed evidence gate stops sooner. Check durable rental
state for subsequent changes. The launch budget ceiling was $8.93 cumulative,
including the $1 reserve. `progress.json` records completed operations; the last
`gpu.jsonl` row gives the current live phase while a comparison is in progress.

## Established evidence

Three seeds passed the one-mesh proof at two checkpoints; see the six portable
models and [raw evidence](evidence/action-v2/README.md). All ranking controls also
met that proof's reduction quota. A two-mesh smoke showed 6.94% mean triangle
reduction versus 1.43% for constant ranking. Full-pilot superiority is unproven.

Local verification passed eight neural CTests, two relevant ASan/UBSan targets,
20 Node cloud/action contracts and bounded data preparation checks. No local
optimizer steps were run. The local high-pixel check was incomplete because
Unity/desktop use left insufficient device memory; cloud capacity is separate.

## Remaining gates

Follow [V2.md](V2.md). Require three learned seeds to beat the strongest
constant/shuffled control by at least one SCORE point, with reduction in at
least two categories, persisting at another checkpoint under the same contract.
Also report shortest-edge and current-plane controls; do not imply superiority
over them merely from passing the constant/shuffled gate.

The tiny proof averaged 38.2% GPU utilization during training; the broader
164-state curriculum averaged 38.74% on its first rental. The current repeat in
US-MO-2 averaged 16.41% over 73 one-second training samples and took 20.14–25.04
seconds per seed. It is not saturating the GPU. Auditing dominated the full
local attempt. Profile
useful throughput and convergence before changing precision, model size or batch
size. BF16, CUDA graphs, segment placement and a GNN remain conditional.

The cumulative cloud cap is $10, with no local training. The launcher includes
conservative previous spend and a $1 reserve in every preflight budget.
Reconcile actual billing and all earlier action rentals before any new launch.
Generate training labels only from the frozen training selection. Pilot families
stay out of training, validation is for later checkpoint selection, and held-out
assets remain release-only.
