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

The implementation refactor is verified separately from learning quality.
Behavior changes are tested with deterministic fixtures, versioned interfaces,
model export checks, CPU sanitizers and bounded GPU validation. Larger models,
lower precision, speculative scheduler changes and incremental topology require
separate measured evidence. See [the protocol](../research/PROTOCOL.md) and
[historical evidence](../research/neural/README.md).
