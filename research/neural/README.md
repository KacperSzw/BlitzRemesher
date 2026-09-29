# Neural LOD path

Status, 2026-09-29: local training is stopped. The audit failure
is fixed and the saved 25,000-update model completes the eight-asset pilot, with
eight unreduced fallbacks (SCORE 0). No release-quality model is claimed. See
[the incident report](REPORT.md#audit-failure-and-repair--2026-09-29) and
[remote GPU options and costs](CLOUD_GPU.md). The requested two-hour RTX PRO 6000
deployment is described in [the Runpod runbook](RUNPOD.md). Its remote training
health and LOD quality must be measured on the actual allocation.

This is an experimental, separately selected generator. A compact three-layer
graph network predicts retention logits and representative displacements. A
deterministic decoder contracts legal edges toward existing source vertices.
It preserves vertex attributes, locks chart/boundary/material junctions, checks
the link condition and rejects face/UV flips. The topology restriction can leave
substantial unreduced geometry. No network output bypasses visual acceptance.

CUDA performs sparse inference, conservative rasterization, exact distance
transforms, visible-normal/color/material comparisons and geometric overlap.
Final selected chains and the minimum-triangle reference receive CPU audits.
Import, graph assembly, decoding and teacher generation run on CPU. Finite
camera audits do not establish an all-view or globally optimal guarantee.

Candidate CUDA audits follow the configured refinement sequence only while the
next raster fits the 64,000,000-sample per-view cap. Initial sampling is never
lowered; uncertain upper bounds reject candidates. Pixel limits and final CPU
confirmation remain unchanged. This can reject candidates that a larger audit
budget might accept. Standalone `evaluate_cuda` honors the literal settings.
`NeuralStats` reports bounded audits, peak owned scratch bytes and the first
resource failure's kind, request, limit, view and sampling resolution.

Overdraw breaks equal per-level triangle-count/equal resident-byte ties only.
The diagnostic counts geometry before depth testing, not shader execution.
UV coordinates/charts are preserved at retained endpoints; distortion is
reported by the existing manifest. Texture images and normal maps are unscored.

## Build

The default portable build retains explicit unavailable neural entry points.
CUDA 12+, cuBLAS and OpenSSL enable inference. LibTorch is a training-only
dependency. On this NixOS workstation:

```sh
mkdir -p .cache
curl --fail --location 'https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.10.0%2Bcu128.zip' -o .cache/libtorch-2.10.0-cu128.zip
sha256sum -c research/neural/libtorch.sha256
unzip -q .cache/libtorch-2.10.0-cu128.zip -d .cache
nix develop .#neural
cmake -S . -B build/neural -G Ninja -DCMAKE_BUILD_TYPE=Release -DBLITZ_CUDA=ON -DBLITZ_NEURAL_TRAIN=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCUDAToolkit_ROOT="$CUDA_PATH"
cmake --build build/neural -j 3
ctest --test-dir build/neural --output-on-failure
```

The official cu128 distribution retains sm_75 support. The pinned Nix CUDA 12.9
toolkit builds native kernels; LibTorch carries its CUDA 12.8 runtime libraries.
Runtime binaries do not link LibTorch. The exported `.blzn` model contains fixed
architecture/schema, provenance, finite float32 tensors and a SHA-256 checksum.
It is approximately 100 KiB. Loading a model never executes serialized code.

```sh
build/neural/blitz simplify input.glb --config research/neural/first-pass.json --neural-model runs/neural/first-pass/selected.blzn --out runs/my-lods
```

`--device` selects a CUDA device; `--gpu-memory-mib` caps owned inference scratch
(default 6144 MiB). Driver/cuBLAS allocations are additional. Missing device or
model causes an explicit error. The C API adds model load/destroy/generate/info
functions without changing ABI-4 descriptor layouts. Returned LOD views borrow
source vertex streams; model lifetime need not extend past generation. Concurrent
calls own separate workspaces; each call's memory cap is independent.

## First experiment

`training-manifest.json` records 66 development assets. Six additional assets
were excluded because their source families overlap the eight development pilot
assets. Validation is used for checkpoint selection; held-out is release-only.

Teacher preparation stores four endpoint QEM examples and three coarse camera
audited examples per asset. The first curriculum samples 16, 32 and 64 pixels,
3 px limits and the attributes profile. These are training labels, not release
audits. Refinement evaluates the trained generator's proposals and replaces a
label only when an audited neural candidate uses fewer triangles. Final model
selection uses the independent 512-to-16 px scenario in `first-pass.json`.

Training uses C++/LibTorch CUDA, width 64, four connected patches per update,
4096 core vertices per patch and exact three-hop halos. Each stage uses 5120
updates; checkpoints include model and AdamW state. Sampling is a deterministic
function of update number. CUDA accumulation may vary in last bits. FP32 is the
verified initial precision; no unmeasured mixed-precision accuracy claim is made.
This small initial run checks the pipeline; it does not saturate this GPU.

```sh
systemd-run --user --unit=blitz-neural-first-pass --collect \
  --working-directory="$PWD" --property=MemoryMax=24G --property=TasksMax=128 \
  --property=RuntimeMaxSec=10h \
  nix develop .#neural -c node research/neural/run.mjs runs/neural/first-pass
```

The runner snapshots binaries, records source/input/model/hardware hashes, uses
segments no longer than 50 minutes, checks finite loss/gradients, parameter
updates, decreasing pilot loss, restored checkpoints and native inference
agreement and a complete eight-asset readiness audit before continuing. It then
refines, audits the development pilot and
selects between checkpoints on validation. Every unsuccessful/incomplete run is
retained. No automatic release promotion or held-out tuning occurs.

Watch `status.json`, `health.json`, stage logs and `*/latest.json` inside the run
directory. `report.json` records completion or failure, including resource
diagnostics; check its `complete` field. Stop with
`systemctl --user stop blitz-neural-first-pass`; resume by rerunning the same
command before the recorded deadline. Use a new directory for changed inputs,
executable, schema or optimizer settings. Completed audit rows resume only with
matching hashes. The ten-hour limit includes preparation/training/audits after
implementation and compilation; it is a budget, not a claimed required duration.

## Sustained GPU training

The measured RTX 2080 configuration in `sustained.json` uses 64 patches/update,
4096 core vertices/patch, two preparation workers and four bounded pinned-host
slots. Workers assemble batches in parallel and the consumer reads them in exact
update order. CUDA performs the forward pass, loss, backward pass and AdamW
update. Host packing overlaps those operations. The 24,580-parameter architecture
and approximately 100 KiB exported model remain the same.

Measured GPU utilization increased from 39.2% to 97.8%, and throughput from
1.93 to 5.63 million supervised vertices/second. Total device memory, including
the desktop, peaked at 6883 MiB out of 8192 MiB. LibTorch's allocator is capped at
the smaller of 5 GiB or initially free memory minus 512 MiB. CUDA contexts and
other processes use additional memory. Allocation failure is explicit.

Continue from prepared data and a warm checkpoint:

```sh
systemd-run --user --unit=blitz-neural-training --collect \
  --working-directory="$PWD" --property=MemoryMax=24G --property=TasksMax=128 \
  --property=RuntimeMaxSec=10h --property=TimeoutStopSec=90 --property=Nice=5 \
  nix develop .#neural -c node research/neural/train.mjs \
  runs/neural/sustained runs/neural/first-pass/refined-data \
  runs/neural/first-pass/refined/step-5120.blzn
```

An optional final argument supplies an earlier absolute deadline in milliseconds.
The runner freezes executables, training/audit settings and a source archive.
It first audits the current saved model on the full pilot, then verifies 4096
updates and continues in 25,000-update stages,
up to 100,000 updates or the deadline. Each stage receives the full development
pilot audit; two stages without score improvement stop further training.
Validation selects among at most two viable stage checkpoints. Held-out assets
remain unused. No checkpoint is automatically approved for release.

Those limits describe the local sustained configuration. The active two-hour
cloud run sets `train_until_deadline: true`, so it continues past flat pilots
and the step cap while retaining health checks and the absolute cutoff. Its
initial health segment is 8,192 updates. See [RUNPOD.md](RUNPOD.md) for current
run evidence and collection/termination instructions.

`health.json` includes a sustained telemetry window, moving-window training loss,
parameter changes, exact model/optimizer-moment restoration and native export
agreement. `gpu.jsonl` records GPU utilization, memory, power and temperature each
second. `training/latest.json` identifies the latest atomic model/AdamW checkpoint;
`training/metrics.jsonl` records loss, gradients, preparation/wait time and vertex
counts. `status.json` identifies the live phase/process, and `report.json` appears
when the experiment has ended. A healthy training status does not establish LOD
quality: the initial full eight-asset pilot returned unreduced fallbacks.

Stop with `systemctl --user stop blitz-neural-training`. Resume before the stored
deadline using the same run/data/model arguments and matching runner source.
The original failed run is preserved in `runs/neural/sustained`; the repaired
audit and interrupted continuation are in `runs/neural/recovered`. Both are
inactive. The latter saved model and AdamW state at update 26,459. Its historical
deadline remains 2026-09-29 06:19:33 UTC. Use a new run directory for a new budget.
When snapshotting runners, include `audit.mjs` and `training.mjs` beside `train.mjs`.
Inputs and executable hashes must still match on resume.

Use `--from-scratch` in place of the initial model argument to start with random
weights and fresh AdamW state. The C++ trainer accepts `--workers 1..8` and
`--gpu-memory-mib 512..131072` (defaults 2 and 5120); the latter still reserves
512 MiB of initially free device memory. `--check-prefetch 1..32` compares ordered
host batches with serial preparation and exits without constructing or training
a network. It still needs the CUDA-enabled LibTorch pinned-memory allocator.

Benchmark exit 2 denotes an incomplete time segment. Exit 3 reports fixed
resource limits, preserving all blocked asset IDs and null SCORE. The runner
stops immediately on those limits or bake exceptions and writes a failure report.
Readiness directories include the model hash so resuming with a later checkpoint
cannot inherit another model's audit. Retain incomplete measurements unscored.
