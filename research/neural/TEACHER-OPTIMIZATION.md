# Teacher generation experiment — 2026-10-01

The prior resident soak spent 93.36% of learner wall time waiting for fresh data.
This work targets teacher audits and proposal search. Increasing replay updates
or timing repeated process startup would not establish better teacher throughput.
See the [prior measurements](evidence/main-integration/README.md).

## Changes and invariants

The teacher-only `blitz-neural-placement-prepare --jobs REQUEST OUTPUT` mode uses
the existing persistent worker pool, immutable policy per wave, two queued jobs
per worker and ordered consumption. It records submission, start, completion and
consumption timestamps. Warmup is separate from measured work, and every output
is freshly generated. It does not link LibTorch or perform optimizer updates.

The exhaustive teacher reuses owned coverage masks within one group of audits
over unchanged candidate geometry. The group must end before any trial, failed
trial, batch, commit or reset can overwrite borrowed geometry. Cached raster data
does not replace a verdict or exact confirmation. The source cache retains two
raster domains under its existing shared byte allowance, with one geometry
snapshot. Original-source FP32 reference identity remains explicit.

`--teacher-strategy coverage-core-first` is experimental and limited to v4
coverage training. It queries endpoints, midpoint, plane fit and the learned
proposal first. Offset proposals expand the search if the core has no safe winner
or exact confirmation rejects it. Expansion invalidates incumbent-derived
pruning and recomputes the winner over valid queried candidates. Resource,
cancellation and unknown results never become negative labels. Exhaustive search
remains the default; the model format and runtime acceptance rules are unchanged.

Policy seeds record `policy.outcome`: the stop reason, integer triangle target,
and whether the actual final triangle count reached that target. The existing
`policy.complete` field means execution remained known and noncancelled; it does
not mean the requested retained fraction was attained. This diagnostic object
does not change labels, episode payloads or the teacher contract.

## Frozen comparison

The resident baseline is commit `81fd92618c194b6b2838047fbddcd4c0bc4e7c8c`, before
raster reuse and strategy changes. [The seven conditions](teacher-profile.json)
cover source/predecessor audits, UV modes, policy rollouts, simplifier seeds and
several mesh sizes. The runner checks actual payload checksums, semantic outcomes,
input hashes and additive stage timing before reporting exhaustive ABBA speedups.
Only the documented exhaustive teacher contract migration is normalized.

Strategy comparisons use the same optimized binary and separate exhaustive and
core-first requests. Label/trajectory changes remain visible. Fresh states per
second, not just lower elapsed time for potentially fewer states, determines
teacher throughput. Shared-workstation measurements are smoke evidence; the
isolated rented GPU supplies the performance comparison.

[The paired experiment](teacher-optimization.json) uses seeds 101, 211 and 307,
the same v4 width-64 initializer, fixed updates per shard, and five minutes of
learning plus at most two minutes of finalization per run. Pair order alternates.
The [curriculum](teacher-optimization-curriculum.json) exercises source, rollout,
simplifier, predecessor and UV conditions. The runner verifies native cycle
contracts, fresh optimizer work and checkpoint hashes before accepting a pilot.

The current request additionally runs one continuous 60-minute learning capability
check after engineering contracts and before the teacher comparisons. It uses
exhaustive teaching, seed 101, the same width-64 v4 initializer and curriculum, and
at most two minutes of finalization. The completed hour and its checkpoints remain
separate artifacts even if a later comparison fails. This does not replace the
paired experiment or authorize strategy adoption or final training.

The initializer and hour checkpoint then use the same bounded
[capability diagnostic](teacher-capability-quality.json) on the two development
assets in [action-diagnostic.json](action-diagnostic.json): painted wooden shelves
(524 triangles) and moon rock 02 (3,304 triangles). The four LODs span 128 to 32
pixels, with 8 search views, 16 audit views, sampling 2/4 with maximum 8, coverage
limits 3 pixels and 0.5 changed area, 64 action trials and batches of 16. This
reuses the existing integration smoke's visual settings and is explicitly
separate from the eight-LOD quality protocol. Each model gets at most ten minutes,
including termination grace, within a combined twenty-minute deadline.

Both models run learned and constant rankings. Constant ranking uses equal edge
scores with stable action order while retaining model-predicted placements; it is
not an independent classical QEM baseline. The report keeps every LOD's source
and adjacent measurements, triangles, action stop reasons and partial failures.
It compares the final learned chain against the initializer and records pointwise
best observed comparator counts without calling those counts a combined valid
chain. Trial-limited results remain censored. No aggregate SCORE, full-quality
claim or strategy adoption follows from this diagnostic.

Promotion also requires complete comparisons on all twelve frozen development
assets, all eight LODs, identical [visual and work settings](teacher-optimization-quality.json),
all three seeds and **no triangle increase at any matched LOD**. Every source and
adjacent audit must pass. Incomplete, resource-limited or numerically invalid runs
receive no aggregate quality score and cannot promote the strategy. Development
evidence is separate from release-only held-out audits.

Adoption also requires known, uncensored action termination. An audited fallback
after the action budget is exhausted remains valid matched-budget evidence, but
cannot establish that the strategy preserves reduction quality. Gate version 2
records those termination reasons explicitly. The first remote bundle was frozen
at `5244fb3` before this stricter review rule; its raw comparisons must be evaluated
again with the current gate before any adoption decision.

## Bounded remote execution

Local contracts and a small warm comparison precede a rental. The separate user
authorization is capped at $3.50 cumulatively, including failed attempts and storage
allowance. The user authorized extending the original $3 cap for the requested
hour. After four terminated attempts, the conservative prior cost is $0.751511;
another 232-minute A40 reservation plus $0.30 reserve totals $3.294178 rounded up.
The original route reserves 140 minutes: 20 setup, 20 contracts and
teacher comparisons, 42 paired learning, 50 quality and 8 collection. The explicit
`teacher-optimization-a40-hour` route adds 60 minutes of capability learning,
2 minutes of finalization and 20 minutes of initial/final diagnostics, and allows
30 minutes for setup: 232 minutes total. The grant ledger must still reserve that
attempt together with prior cost before launch. The controller
and independent watchdog enforce deadlines; artifact checksums are verified before
normal cleanup. The experiment cannot start final training.

```sh
BLITZ_RUNPOD_PROFILE=teacher-optimization-a40-hour node scripts/neural/runpod.mjs prepare-teacher-optimization runs/neural/teacher-optimization-NEW
BLITZ_RUNPOD_PROFILE=teacher-optimization-a40-hour node scripts/neural/runpod.mjs launch runs/neural/teacher-optimization-NEW
```

Use a fresh directory and review the current grant ledger before launch. These
commands describe the experiment; they do not imply that it has passed. A local
full eight-LOD calibration on two small assets reached a 240-second deadline
without producing a complete audit. The full remote quality comparison may also
exceed its budget; preserve that outcome rather than lowering audit settings or
calling a partial comparison a quality result.

## Evidence status

Earlier integrated local validation passed 44/44 GPU/training CTests, 28/28 CPU tests and
28/28 ASan/UBSan tests. The updated cloud/profile/optimization CTests passed 3/3.
Grouped Vulkan audits and the core-first resident teacher passed Compute Sanitizer
with zero errors. [Checksummed local evidence](evidence/teacher-optimization/local/manifest.json)
records source revisions, raw reports, tests and the incomplete quality calibration.

Two local ABBA smokes preserve actual label/episode bytes and all common
intermediate query, pruning, preferred-action and selection outcomes across 32
jobs. The medium predecessor/rollout/simplifier comparison measured 2.356 seconds
of warm work before reuse and 2.009 after (1.173x); process time was 5.904 versus
5.115 seconds. These shared RTX 2080 timings are diagnostic only. It removed 2,112
rasterizations without changing 1,174 evaluations. Owned workspace increased by
698,880 bytes and allocation count rose from 322 to 502.

Candidate audits still consume 44.71% of optimized post-load worker time and exact
confirmation 21.36% on that medium smoke. Reuse does not remove the teacher
bottleneck. The limited coffee rollout retained 99.7903% of triangles despite a
75% target; it exercised the rollout path but did not demonstrate deep progress.
The branch simplifier seed retained 49.9970%. These distinctions must remain
visible when evaluating learning throughput and state coverage.

After the cancellation fix, local focused CTests passed 3/3 and the isolated
cancellation/strategy contracts passed 2/2 in both CPU and ASan/UBSan builds.
The teacher and resident cycle rebuilt. A final one-worker, 512 MiB ABBA smoke
completed another 16 jobs with matching payloads and common intermediate
trajectories. These supplementary records are in the same local manifest; they
do not validate two-worker teardown on the A40.

The first launched A40 attempt used immutable source `5244fb3`, quoted at
$0.49/hour. Remote CTests passed 43/43, both Compute Sanitizer runs reported zero
errors, and Vulkan validation passed. All 11 jobs in the first optimized process
completed and were consumed. Independent checks of their actual action and
episode bytes, normalized contracts, common trajectories and semantic counters
match the baseline. The optimized process then stalled after the last persisted
wave and was killed at its deadline. The existing report does not identify which
shutdown or enclosing cleanup call blocked.

[The failed-attempt evidence](evidence/teacher-optimization/remote-attempt-01/README.md)
preserves raw reports, logs, the parity proof and the verified archive checksum.
The complete ABBA comparison was not obtained; no remote speedup, strategy
adoption or quality result is accepted. No paired learning pilot or final training
started. The supplemental CPU cancellation contract also passed on that host
without changing the frozen experiment source.

Collection was verified before compute termination on 2026-10-01 at 11:23:03 UTC.
An independent provider readback at 11:35:50 UTC showed zero Pods and zero network
volumes. The rental ledger covers 1,101.427 seconds; charging the conservative
$0.55/hour GPU cap plus $0.03/hour storage allowance gives $0.1775 rounded up.
This is an estimate for the grant ledger, not a provider billing receipt.

The [second A40 attempt](evidence/teacher-optimization/remote-attempt-02/README.md)
used frozen source `ec9660f`. Both native builds completed. The debugger preflight
passed and ordinary GDB captured 10 threads with 86 frames when the teardown
fixture stalled in its third round after two successful rounds. One worker had
finished logged C++ resource destruction and was in NVIDIA EGL/GLSI thread-exit
cleanup; the other remained inside `vkDestroyDevice`. This identifies an observed
teardown boundary, without proving the proprietary lock cycle or a source ownership
defect. The attempt stopped before contracts, teacher comparisons or learning.

Cloud setup had replaced the injected vendor GLX ICD with EGL. Both captured
dependency reports resolve all listed libraries. NVIDIA's
[580.159.04 documentation](https://download.nvidia.com/XFree86/Linux-x86_64/580.159.04/README/installedcomponents.html)
supports both ICDs, defaults to GLX, and recommends EGL when X11 client libraries
are unavailable. At that point forced EGL was a candidate environment cause;
the subsequent local control below shows that selecting GLX alone is insufficient.

The second archive was verified before compute termination at 12:08:16 UTC.
Independent provider readback at 12:11:56 UTC again showed zero Pods and volumes.
Its 872.058-second rental adds $0.1404982333 to the conservative grant ledger,
bringing both attempts to $0.3179503611 before any further reservation. These are
elapsed-time estimates at the GPU cap plus storage allowance, not invoices.

The unchanged native fixture also stalled locally on the RTX 2080 with driver
595.71.05, no display server variables, and the explicitly selected vendor GLX ICD.
Ordinary GDB captured 12 threads and 123 frames: one worker was inside
`vkDestroyDevice` while another was in NVIDIA thread-exit cleanup. The GLX-selected
driver still used internal EGL components. A traced run and a destructor-lock
control each passed 12 rounds, but those successes do not establish safe thread
retirement. The exact proprietary lock cycle remains unproved.

The owned teacher pool now starts workers sequentially through readiness, joins a
failed new worker before retiring existing workers, and waits for every active
worker to quiesce before permitting one retirement at a time through OS thread
join. This covers driver thread-local cleanup after C++ destructors. Four fresh
headless GLX processes, alternating tracing off/on/off/on, passed 48 fixture rounds;
12 more rounds passed Compute Sanitizer with zero errors. These results concern
the owned teacher pool and join fixture. They do not verify arbitrary caller-owned
threads using public neural sessions or establish a general driver fix.

The integrated native suite passed 48/48 tests at `58381e5`; the focused retirement
and rollout-outcome contracts passed 2/2 in both CPU and ASan/UBSan builds. Separate
two-worker strategy and lifecycle-overlay parity smokes completed normal shutdown
in all four processes each. They preserve the raw reports and distinguish strict
reuse parity from strategy-dependent query/pruning changes. Shared-workstation
timings are diagnostic only, and the tiny rollout remains short of its requested
triangle target. Subsequent normal concurrent-fixture lifetime hardening at
`6b64051` passed its focused CTest, Vulkan validation and Compute Sanitizer with
zero errors. [Local lifecycle evidence](evidence/teacher-optimization/local/lifecycle-validation.json)
preserves the failure stacks, controls, exact revisions and validation scope,
including an attributed external-review summary whose exact response was not saved.

The remaining grant permits bounded remote validation of the owned-worker fix.
Complete representative teacher comparison, paired learning and uncensored full
development quality are still required. The next recipe explicitly keeps
`exhaustive`; long training remains disabled while those gates are unresolved.

The [third remote attempt](evidence/teacher-optimization/remote-attempt-03/README.md)
retargeted the same A40 rental to frozen source `8bc9c14` for the requested hour
and continuation of validation. All 48 owned-worker join fixture rounds completed,
including two untraced processes. CTest passed 47/49: all native cases passed, but
the native-stack JavaScript test and mocked optimization-continuation test failed.
The engineering gate stopped the job before the capability hour, teacher
comparisons or quality evaluation; later standalone sanitizer stages did not run.
This provides remote fixture evidence, without a learning, performance or quality
result. The verified archive and retarget history are retained, and provider
readback at 13:20:59 UTC confirmed zero Pods and volumes. Full-rental conservative
cost adds $0.2459791278, bringing the three attempts to $0.5639294889. Correcting
the failed tests and retrying the requested work remain pending.

The native-stack denial failure was reproduced locally with the exact Ubuntu GDB
15.1 package as UID0 without CAP_SYS_PTRACE. Its explicit Yama permission advice
was followed by a misleading errno; the parser missed that advice. Fix `959f248`
recognizes the specific warning and preserves all positive stack-capture gates.
The [checksummed CPU reproduction](evidence/teacher-optimization/native-stack-denial/README.md)
records the old 7/8 failure and corrected 9/9 result, package and source hashes,
raw debugger output and commands. This does not replace the failed remote result
or establish a training or quality result.

Before the retry frozen at `959f248`, the [local regression runs](evidence/teacher-optimization/local-retry-validation/README.md)
passed 49 bulk CTests excluding native-stack, then that contract separately after
its fix. Their test-name union covers all 50 configured local CTests; this was
not one simultaneous suite. The capability orchestration tests also passed 25/25
from source, build and unrelated working directories. Raw logs and source scopes
are preserved, without changing the prior remote failure or claiming a completed
learning hour.

The [fourth remote attempt](evidence/teacher-optimization/remote-attempt-04/README.md)
passed 49/49 CTests, both memchecks, Vulkan validation and 48 join fixture rounds
on an A40 with driver 595.91.07. The requested hour then started but stopped after
about 110 seconds when CoffeeCart teacher shard 90 failed its final adjacent audit.
History records 11,520 completed updates; the latest verified checkpoint remains
step 6,272. These are separate facts, and neither means the hour completed.
The failed shard, history, checkpoint verification and binary hashes are preserved.
No subsequent LOD comparison or paired validation ran. Collection preceded
termination, provider readback confirmed zero Pods and volumes, and conservative
cumulative cost reached $0.75151035. Investigating the seeded audit failure and
retrying the requested work remain necessary.

The seed-admission correction `ecfc478` rejects the reproduced CoffeeCart seed
under its unchanged adjacent limit, then completes all 16 requested teacher states
from the confirmed source baseline. The corresponding five-minute local cycle
completed 24,832 updates and 3,104 requested-condition states, with passing final
checkpoint verification. The earlier 768 MiB local workspace-cap failure remains
visible alongside the successful 1,024 MiB run.

The corrected `worker-join-seed-v2` baseline and current teacher then completed a
two-condition local ABBA check: four normal process exits and 24 jobs, including
16 warmups and eight measured jobs. Actual label and episode bytes, normalized
contracts and common trajectories match across all four runs. The observed warm
wall ratio was 1.057 on the shared workstation; it is not an accepted performance
gain. A preceding test deliberately requested the rejected CoffeeCart simplifier
seed. Its fallback produced a complete, audited teacher shard, but benchmark
qualification correctly failed because that requested seed condition was not
exercised. That incomplete attempt has no speedup result. Both plans, raw outputs,
baseline build provenance and checksums are preserved in the
[local admission evidence](evidence/teacher-optimization/seed-admission-local/manifest.json).

The subsequent [local two-asset quality diagnostic](evidence/teacher-optimization/local-quality-24832/README.md)
compared the initializer with the verified checkpoint after 24,832 updates in a
five-minute local soak. All three learned shelves LODs improved their triangle
counts (518/259/247 to 500/250/236), while all three moon-rock LODs regressed
(1862/932/930 to 3252/1626/1626). Every emitted source and adjacent audit passed.
The trained constant-ranking control also beats trained learned ranking on
shelves, and many proposals exhaust the 64-trial budget. This shows changed
behavior, not consistent progress or a known minimum update count for improvement.
Raw initial/final outputs, fixed settings, checkpoint hashes and the preceding
JSON-key-order reporting failure are preserved. No aggregate SCORE or full-quality
claim follows from these two development assets.
