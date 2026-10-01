# Neural/main integration

This normal merge preserves neural commit `111f010` and classical main commit
`6ec4da2`, including their histories and research evidence. ABI 5, classical graph
and tail search, shared vertex storage, and appearance/wedge controls remain.
The neural preset explicitly disables the new classical vertex-memory cap so its
triangle-first comparison contract does not change by inheriting a new default.

Focused fixtures cover cap rejection, audited packed fallbacks, one-shot
cancellation, and exact-position metadata across shared pools and wedge edits.
They complement the existing classical and neural contract suites. Commands,
counts, integration contracts, and outstanding checks are in `validation.json`.
The earlier `core-refactor` evidence describes the pre-merge revision and is
unchanged. These checks are correctness and readiness evidence, not a quality
score or a final training run.

Local validation passed 28 CPU tests, 27 ASan/UBSan tests, 41 CUDA/Vulkan/training
tests, 73 Node contracts and two installed-package consumer tests. GPU memory
checks reported zero errors. Local Vulkan validation reported zero errors and
1,109 existing unused-output warnings, retained in its log.

The frozen revision `4729907` passed 40 tests on a rented RTX 2000 Ada, both GPU
memory checks and eight teacher runs with identical payloads across batch sizes.
Acquisition was disabled remotely, accounting for the one-test difference.
The [remote record](remote/record.json) includes archive verification and provider
cleanup; no Pods or network volumes remained. The 12.2-minute rental cost about
$0.049 at the quoted GPU rate, excluding storage and billing rounding.
No training session was launched; synthetic optimizer contract tests ran.

The [teacher measurements](remote/teacher-analysis.json) identify candidate audits
(49.59%) and exact confirmations (23.17%) as the largest batch-4 post-load costs.
Batch 4 measured 1.48x post-load speedup but only 1.04x process speedup on these two
small fixtures. Loading, session setup and work outside the teacher timer matter;
these observations do not establish production throughput. Profile representative
larger meshes, both UV modes and rollout64 before a long training run. Keep exact
audits and full-chain quality controls as acceptance requirements.
