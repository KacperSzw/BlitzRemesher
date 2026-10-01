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

Implementation and integrated validation are in progress. No new rental or
learning pilot has started at this documentation checkpoint. Final measured
results, remaining bottlenecks and resource cleanup must be recorded here before
claiming readiness for a longer training session.
