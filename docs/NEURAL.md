# Neural LOD architecture

The product generates a configurable chain of static, opaque meshes. Every
accepted level passes independent audits against immutable LOD0 and the preceding
emitted LOD, at that level's screen size. Triangle count is the primary objective;
normal/attribute preservation is optional. The neural preset uses zero triangle
overhead. The existing camera/sample contract remains a finite audit, not an
all-view or global-optimum guarantee.

## Runtime

The current placement policy is an independent-action MLP with 128 inputs, two
64-wide hidden layers and 12 outputs: ranking, two teacher-feasibility auxiliaries,
position displacement and two wedge-normal displacements. The auxiliaries are
not calibrated acceptance probabilities. Actual generated geometry must pass the
auditor regardless of those outputs.

The executor owns contiguous topology and GPU workspaces. It ranks legal actions,
selects disjoint contractions, audits trial geometry, and preserves a validated
incumbent. Source, predecessor and combined proposal origins share the existing
candidate budget. Their work and termination causes are reported explicitly;
reaching a work cap does not mean no further geometric reduction is possible.

Source bounds, quantization domain and feature normalization remain fixed for the
whole chain. A compact predecessor owns its local indices and streams; those IDs
must never be interpreted against LOD0. Quantization consumes the existing error
budget. Uncertain predicates, resource failures and cancelled audits cannot admit
a candidate. Cancellation in packed mode returns only original LOD0 when no
complete chain has passed confirmation, or retains an already confirmed chain.
It never presents unaudited packed duplicates as a valid scheduled chain.

The v4 placement format conditions the policy on UV preservation. Existing v1-v3
models remain readable with their original strict behavior. Disabling UV
preservation requires v4, retains best-effort UV data, and removes UV-specific
locks rather than weakening geometric legality or enabled visual constraints.
UVs and chart safety do not constitute a texture-image audit.

## Training and evaluation

The teacher queries plane-ranked, policy-ranked and exploratory actions and audits
placement alternatives. Confirmed equivalent preferred actions may share a target;
unqueried or pruned outcomes remain unknown. A deployment-policy rollout changes
state using the actual policy decision and audits, rather than presenting a
teacher-selected trajectory as a deployment rollout.

The next curriculum covers multiple screen sizes, real emitted predecessors,
independent source/transition limits and preservation settings. The completed or
stopped coverage-pretraining run is an initialization/control artifact, not proof
of full LOD or shading quality. No final training is started by test scripts.

`blitz-neural-cycle` runs learning orchestration. `blitz-neural-diagnostics` owns
contract checks, model audits, replay and explicit policy migration. For a v3
checkpoint, migration uses its exact saved `input.pt` probe, folds the redundant
feature weights and verifies both FP64 and native outputs against the unchanged
2e-4 numerical gate before writing a fresh v4 model. Conversion may fail that
gate; never rename or reinterpret an old file as v4.

Use `scripts/neural/budget-quality.mjs` for bounded audit-only budget diagnosis.
The historical three-level smoke remains an integrity check. Its 256-triangle
per-level ceiling makes it unsuitable for selecting model capacity. Meaningful
comparisons use all reduced LODs and category-balanced retained ratios, disclose
per-level regressions and failures, and freeze work/visual settings before
validation. Constant/shuffled ranking controls still use learned placements;
classical reducers are separate whole-system controls.

## Preparing the next v4 pilot

[The next experiment](../research/neural/next-experiment.json) is a bounded manual
pilot recipe, with `launch: false` and `final_training: false`. These metadata
fields do not prevent someone from invoking the executable. No training launcher
currently consumes this recipe. In particular, `prepare-gpu-refactor` still uses
the historical v3 [next-training.json](../research/neural/next-training.json);
do not use that route to launch the v4 experiment.

The recipe uses `--initialize runs/neural/v4-initialization/model.blzn`, not
`--model`. Omitting `--initialize` creates random weights. The initializer's file
SHA-256 is `306ec18cfeef36ef8af16ef2a9d008021e142d5bf114b4ca8ea243debf9d2fae`.
Its [migration proof](../research/neural/evidence/stopped-pretraining-05/migration.json)
records a native maximum absolute difference of 3.44e-6 against the unchanged
2e-4 gate. This establishes conversion parity, not v4 learning quality. The model
is a local run artifact, so a checkout alone does not contain it. Copy that exact
file with the source/manifests and required corpus files when preparing another
machine. Start with fresh Adam state; coverage initialization cannot use
`--warmstart` from the old run.

Build and test the integrated source on the intended NVIDIA device. The CUDA
architecture must match that device: the tested local RTX 2080 uses 75; the
repository's A40 profile uses 86, Ada profiles use 89 and Blackwell profiles use 120. Blackwell
profiles require the repository's CUDA 12.9 and driver checks. Use the pinned
[LibTorch archive checksum](../research/neural/libtorch.sha256), expose the NVIDIA
Vulkan graphics driver, and verify both CUDA and Vulkan tests. For example, in
the local `nix develop .#neural` shell:

```sh
nvidia-smi --query-gpu=name,compute_cap,memory.free,driver_version --format=csv
cmake -S . -B build/neural -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBLITZ_CUDA=ON -DBLITZ_VULKAN=ON -DBLITZ_NEURAL_TRAIN=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DBLITZ_LIBTORCH_ROOT="$PWD/.cache/libtorch"
cmake --build build/neural -j2
ctest --test-dir build/neural --output-on-failure
```

On an already provisioned remote session, transfer the same files under the same
relative paths, configure with its actual CUDA architecture and LibTorch path
(the existing cloud setup installs `/opt/blitz/libtorch`), and repeat those checks
there. The recipe's 16,384 MiB library allocation cap needs additional driver and
runtime headroom; it is not suitable for the 8 GiB workstation. Lowering the cap
or changing worker/batch counts creates a different measured experiment and must
be recorded. Do not interpret a 512 MiB fixture pass as validation of that cap or
of the large assets.

First run bounded teacher preparation without optimizer updates:

```sh
node scripts/neural/teacher-readiness.mjs \
  runs/neural/v4-initialization/model.blzn runs/neural/v4-teacher-readiness-01
```

For remote contracts and the same teacher fixtures, `prepare-core-validation`
creates an immutable source/model/asset bundle. Set
`BLITZ_CORE_READINESS_MODEL=runs/neural/v4-initialization/model.blzn` when preparing
it. An explicit `BLITZ_CORE_STORAGE=container` uses only disposable container
storage for this test route. Completed results are checksum-collected before
termination; uncollected container results are lost on termination and recorded
as incomplete. The cumulative validation cap is $1.25 and each rental is bounded
to 20 minutes of setup, 10 minutes of testing and 5 minutes of collection.
Container storage is rejected for learning experiments.

This checks source and emitted-predecessor fixtures in batch order 1/4/4/1 with
a 250-second local deadline. It records parity and timing evidence, not a quality
score. For the existing test-only remote bundle, set
`BLITZ_CORE_READINESS_MODEL` to the absolute initializer path when invoking
`node scripts/neural/runpod.mjs prepare-core-validation runs/neural/NEW` with a
compatible core-validation profile. Preparation copies an immutable model/request
and the two fixture assets; it does not rent a GPU or start learning. The optional
teacher checks share the remote contract tests' ten-minute deadline.

Prepare a reviewable pilot command from the recipe without executing it:

```sh
node --input-type=module <<'JS'
import fs from 'node:fs';
import { createHash } from 'node:crypto';
const recipe = JSON.parse(fs.readFileSync('research/neural/next-experiment.json'));
const hash = p => createHash('sha256').update(fs.readFileSync(p)).digest('hex');
for (const [file, expected] of [
  [recipe.initialization_model, recipe.initialization_sha256],
  [recipe.curriculum, recipe.curriculum_sha256],
]) if (hash(file) !== expected) throw new Error('Checksum mismatch: ' + file);
for (const manifest of [recipe.training_selection, recipe.validation_selection,
                        'research/neural/action-diagnostic.json'])
  for (const asset of JSON.parse(fs.readFileSync(manifest)).assets)
    for (const file of asset.files)
      if (hash(file.path) !== file.sha256) throw new Error('Corpus mismatch: ' + file.path);
const run = 'runs/neural/v4-pilot-01';
if (fs.existsSync(run)) throw new Error('Choose a fresh run directory');
const argv = ['timeout', '--signal=TERM', '--kill-after=30s',
              String(recipe.process_deadline_seconds) + 's',
              'build/neural/blitz-neural-cycle', run, ...recipe.cycle_arguments];
fs.mkdirSync('runs/neural/v4-pilot-prepared', { recursive: true });
fs.writeFileSync('runs/neural/v4-pilot-prepared/command.json', JSON.stringify({
  argv, training_started: false,
  binary_sha256: hash('build/neural/blitz-neural-cycle'),
  recipe_sha256: hash('research/neural/next-experiment.json'),
}, null, 2) + '\n');
JS
```

After the prerequisites are satisfied and a bounded learning pilot is approved,
execute that saved argument vector from the repository root in the prepared
session. The recipe requests five minutes of learning and two minutes for
checkpoint/final-audit finalization. The outer ten-minute limit also covers
initialization; forced termination after its additional 30-second grace period
is a failed run, not a successful checkpoint. Preserve the run contract, reports,
unavailable/failed conditions and numerical-gate artifacts. A `duration_complete`
report does not establish `curriculum_coverage_complete`: the 600 conditions are
ordered by screen band, and a short run can see only early source conditions.

Final training stays disabled until matched development comparisons demonstrate
complete-chain quality under frozen work/visual budgets. Before choosing throughput
settings, measure representative larger assets, both UV modes and the actual
64-trial rollout path; the tiny readiness fixtures do not cover these. Keep held-out
assets for the release audit.

## Boundaries and dependencies

- `src/neural/`: model format, actions, inference, CUDA/Vulkan audits.
- `training/`: teacher, data/shards, model/loss/update, replay and checkpoint logic.
- `tools/neural/`: command-line composition and diagnostics.
- `scripts/neural/`: local artifact/process utilities, experiment orchestration and
  explicit cloud provider/controller operations.
- `research/neural/`: frozen manifests/configurations, measurements and history.

The runtime does not link LibTorch. Corpus HTTP/archive support is independently
selected. Portable CI runs the pure Node contracts without requiring a GPU.
Training CUDA updates compile into one private training library. Diagnostic and
training failures preserve their exact reproduction data without publishing an
invalid recovery checkpoint.

## Evidence and limits

The latest prior completed twenty-minute coverage soak produced 30,006 fresh
states and 240,512 updates. Learner waiting occupied 93.36% of process time;
optimizer execution occupied 2.57%. Worker times overlap and must not be added to
wall time. These measurements motivate audit/preparation work, not claims of GPU
idleness or output quality.

Teacher timing version 2 keeps `seconds` as post-load preparation time and adds
`total_wall_seconds` from load/session setup through final index assembly. The
final index write, object teardown and caller-side model loading are excluded;
the readiness process wall timer includes them. Numeric `timings` are disjoint
host wall intervals, not GPU service times. They now separate rollout (including
its seed artifact), exact winner confirmations, predecessor snapshot/audits, final audits and final
encoding/writes/hashes. `timing_breakdown` is nested detail already included in
its parent phase, so do not add it again. Worker totals still overlap across
concurrent jobs. Zero generic inference/audit counters in a teacher report do
not imply that no GPU work occurred; use these teacher phase timings.

The implementation refactor is verified separately from learning quality.
Behavior changes are tested with deterministic fixtures, versioned interfaces,
model export checks, CPU sanitizers and bounded GPU validation. Larger models,
lower precision, speculative scheduler changes and incremental topology require
separate measured evidence. See [the protocol](../research/PROTOCOL.md) and
[historical evidence](../research/neural/README.md).
