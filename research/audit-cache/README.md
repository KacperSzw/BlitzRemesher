# Coverage audit cache experiment

Hypothesis: reusing exact coverage masks and squared distance fields can reduce
generation time by at least 25% on the frozen automatic coverage pilot while
preserving every output and audit decision. No asset's median generation time
may increase by more than 5%. The cache allowance is 256 MiB per bake.

The motivation is the existing strict eight-asset development run: 65.83 seconds
of rasterization and 139.93 seconds of distance work out of 239.68 seconds of
generation. This is evidence of a bottleneck, not a predicted speedup.

## Implementation and lifetime

`research.coverage_cache_mib` accepts integer values from 0 to 256, default 256.
Zero uses uncached evaluation. The control is available through C++ and CLI research
settings. The C descriptor remains ABI 4. C++ consumers rebuild for the new
research setting and optional performance counters.

The cache accelerates the coverage profile. It stores one-bit masks and the
existing float32 squared Euclidean distance fields. No distance truncation,
resampling, field quantization, approximate matching or camera reduction is
used. The underlying transform follows the separable algorithm described by
[Felzenszwalb and Huttenlocher](https://theoryofcomputing.org/articles/v008a019/).
The field for B is sampled at foreground pixels of A, and vice versa; their
maximum gives the same symmetric foreground Hausdorff distance as recomputation.

Half the allowance belongs to immutable source/parent references for one
scheduled level. Half belongs to the current candidate across its four gates.
Candidate storage is cleared at offer boundaries; reference storage is destroyed
at the level boundary. The cache owns image data and borrows no mesh storage
beyond evaluation. Stable reference IDs, camera family/index, and sampling
identify entries inside a context with fixed bounds, screen size and rendering
settings. View indices have 17 bits, sufficient for both uint16 camera counts.

Entries are kept in a sorted contiguous array. Admission never evicts an entry:
once full, a store bypasses subsequent entries that do not fit. This preserves
reuse when a deterministic camera sweep exceeds the allowance. Distance fields
are built only after identical/empty mask handling requires them. Cache storage
accounts for retained vector capacities and entry-array capacity, including
the temporary overlap while growing that array. Ordinary evaluator scratch
memory and allocator bookkeeping are outside this additional cache allowance.
Allocation failure drops both stores and retries the current view without an
extra cancellation poll.

## Frozen comparison

The primary scenario uses `research/pilot.json` and production automatic storage.
It derives from the existing strict 3 px source / 2 px adjacent coverage scenario
with eight levels, base 512 px, last 16 px, 0.5 area cap, eight proposals, beam
width two, 6+2 search views and 12+4 audit views. The triangle allowance is 500
basis points. These parameters are frozen before measuring the first candidate.

The two comparison configs differ only in cache allowance. The baseline config
omits the new key so the pre-change executable can read it. Comparisons first
verify source/protocol/camera/compiler identities, then compare complete result
JSON, source/output attribute hashes, output geometry hashes, retained ratios,
failures and numerical counters. Timings and cache counters are intentionally
outside this equality check. Candidate counts and audit records remain inside.

1. Build focused targets and pass contract tests.
2. Run shelves/tree smoke checks against the old executable and both cache modes.
3. Pass release and ASan/UBSan suites before scored timing.
4. Run the complete old-executable pilot, then five complete A/B repetitions,
   reversing order each repetition. Use one benchmark process at a time.
5. Compute each asset's median cached/uncached time ratio, average log ratios
   within categories, then average categories and exponentiate. Require a ratio
   at most 0.75 and every asset's ratio at most 1.05. Preserve every repeat.
6. Only after a pilot signal, compare all 20 frozen validation assets and run
   the two independent dense tail checks. Held-out assets are not used.

Validation uses two independent single-thread workers to check equivalence;
their timings are recorded but do not enter the pilot speed claim. Dense checks
use at most one additional worker, keeping this experiment below four workers
and 24 GiB of combined memory.

The runner's `smoke`, `pilot`, `validation`, `dense`, `production`, and `report`
modes perform these stages. Each benchmark batch checkpoints with a 50-minute limit. A run
that reaches the limit can be resumed with the same command and frozen binary.
Incomplete runs remain visible and are never accepted as a complete comparison.

Build the baseline from Git revision
`5c0c62386b212f573d2cae07bf8e33f73dfa9052` into
`build/audit-cache-baseline/blitz`. In a separate checkout of that revision,
overlay `builds/candidate-source.tar.gz` and build with the pinned Nix shell
and CMake Release. The checkout supplies the unchanged vendored `third_party`
dependencies; archives contain the measured first-party sources and build files.
Then place its `blitz` and `blitz-tail-audit` in `build/audit-cache-frozen/`. Rebuilt
binaries need fresh stamps and run directories; archived stamps cannot identify
a newly compiled executable. `tools/stamp-build.sh` produces those stamps.

After all gates pass, `production` replays the complete pilot with the final
default-enabled executable in `build/audit-cache-production/blitz`, its own
`builds/production.json` stamp, and a config that omits the cache key. It requires
exact output and cache-counter agreement with the explicit 256 MiB candidate.
The final source is retained separately as `builds/production-source.tar.gz`.

`run.mjs` checks exact measurements before reporting speed. Raw checkpointed
rows, configuration/camera/input/build hashes and summaries live in
`research/runs/audit-cache/`. Mesh exports follow the repository's ignored
benchmark-output policy and are reproduced by rerunning the corresponding bake.
Dense check files contain export hashes and every audited tail slot.

## Validation notes

The tests cover tiny/disabled/full budgets, oversized sweeps, immutable reference
IDs, candidate replacement, high camera indices, area rejection, clipping,
culling, resource limits, refinement, cancellation and allocation failure.
Generation fixtures exercise automatic, forced shared and forced owned output;
normal and attribute profiles also undergo cached-setting/uncached-setting
agreement checks. Existing brute-force mask tests verify the distance kernel.

An initial implementation was rejected after a regression demonstrated that
copying evaluation settings reset mutable cancellation callbacks between calls.
The corrected cache borrows each gate's actual evaluation settings. The initial
timings and the test-injector sanitizer correction remain in [v1](v1/README.md).
The corrected experiment uses a fresh executable and separate run directories;
historical timings are not pooled into its result.

`checks/production-stamp-race.log` records a refused default replay started
before its build stamp finished writing. No asset ran in that attempt; the
replay was restarted after the stamp completed.
