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
authorization is capped at $3 cumulatively, including failed attempts and storage
allowance. Each attempt reserves at most 140 minutes: 20 setup, 20 contracts and
teacher comparisons, 42 paired learning, 50 quality and 8 collection. The controller
and independent watchdog enforce deadlines; artifact checksums are verified before
normal cleanup. The experiment cannot start final training.

```sh
node scripts/neural/runpod.mjs prepare-teacher-optimization runs/neural/teacher-optimization-01
BLITZ_RUNPOD_PROFILE=teacher-optimization-a40 node scripts/neural/runpod.mjs launch runs/neural/teacher-optimization-01
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
are unavailable. Forced EGL is a candidate environment cause; the GLX control with unchanged native
implementation is pending review and execution, with no fix yet established.

The second archive was verified before compute termination at 12:08:16 UTC.
Independent provider readback at 12:11:56 UTC again showed zero Pods and volumes.
Its 872.058-second rental adds $0.1404982333 to the conservative grant ledger,
bringing both attempts to $0.3179503611 before any further reservation. These are
elapsed-time estimates at the GPU cap plus storage allowance, not invoices.

The remaining grant permits a bounded diagnostic control. Reliable normal shutdown
comes first, followed by complete representative teacher comparison and uncensored
full development quality. The next recipe explicitly keeps `exhaustive`; long
training remains disabled while those gates are unresolved.
