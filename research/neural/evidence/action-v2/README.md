# Endpoint learning evidence — 2026-09-29

These are experimental architecture-2 models, trained on one CC0 development
asset (`ph_sweet_potato`). They are an overfit proof, not a release model.
`manifest.json` pins model hashes, the original source revision and collected
archive checksum. `proof-data/contract.json` retains source licensing and hashes.
The complete cloud archive remains in the ignored local rental directory.

All three seeds passed the preferred-set membership gate (>=95%) and achieved
the teacher's 128-triangle reduction at both 4,096 and 8,192 updates. The final
memberships were 100%, 98.4375% and 100%. Every ranking control also reached that
same reduction quota, so this proof alone establishes no LOD quality advantage.
Exact model/AdamW restoration passed. All numerical comparisons passed the
unchanged 2e-4 limit; the largest native/FP64 difference was about 5.154e-5.

The RTX PRO 6000 proof experiment lasted approximately 195 seconds. Training
telemetry averaged 38.2% GPU utilization (30 one-second samples). Audit telemetry
averaged 72.23%. The tiny network did not saturate the GPU. The Pod and volume
were deleted after checksum collection. No local optimizer steps were run.

## Subsequent batching smoke

`smoke-learned/` and `smoke-constant/` use the same two development meshes,
four-level 32-to-16-pixel scenario, zero triangle overhead, eight combined trials
per proposal and at most 32 independent collapses per audit. Learned ranking uses
seed 101 at update 8,192. Average triangle reduction was 6.9404% versus 1.4261%
for constant ranking. Times were 14.67 versus 7.00 seconds on the local RTX 2080;
work caps match, actual accepted work differs. Both runs completed without
fallbacks. These are smoke measurements, not the full eight-asset quality gate.
Other seeds and shuffled controls must still be compared on the frozen pilot.

`full-pilot-incomplete/` records the five-minute full-quality attempt: zero
completed assets, null SCORE. The first asset spent 299.685 seconds in GPU audits
and 0.045 seconds in inference, with no accepted contractions before cancellation.
This identifies auditing as the measured bottleneck. A later one-minute cache
probe ended during the first proposal with zero cache hits; no cache speedup is
claimed from that incomplete probe.

`multilevel-check/` is a no-training data preparation check. It reduced the
2,042-triangle source to a fixed intermediate 2,038-triangle reference, then to
2,030 triangles. Both final CPU gates passed; source limit was 3 px, adjacent
limit 2 px, and target size 16 px after a 32 px warmup. Its 96 queried actions
retain the fixed reference's triangle count in the trajectory.
`multilevel-final/` repeats that check with the final binary, cycling all four
target fractions and using a test-owned 128 MiB scratch cap. It also passed both
CPU confirmations and emitted all six states.

`rejection-labels/` uses a test-owned 0.01 px limit to exercise negative labels.
All 16 queried actions failed their pixel gates and were retained as known
negatives; the original mesh stayed unchanged and preparation completed. No
optimizer update was run. This test setting is not a quality scenario.

`high-pixel-incomplete/` records the local 512-to-256 px preparation probe. Eight
queries passed, then the ninth hit a resource limit with the local 6 GiB scratch
allowance. The diagnostic repeat in `high-pixel-diagnostic/` identified exhausted
device memory; Unity and the desktop were using most of the 8 GB card. No
state/shard was promoted. Cloud continuation explicitly allocates
up to 16 GiB of evaluator scratch on the 96 GB device; camera and quality limits
stay fixed.

## Limits

Models rank legal directed endpoint collapses. They do not interpolate new
positions or UV charts and do not score texture images/normal maps. Finite
camera audits do not imply an all-view guarantee. The batched executor preserves
the conservative topology restrictions, then audits each joint candidate and
falls back to smaller batches on rejection. Full generalization, higher GPU
utilization, BF16 and segment placement remain evidence-gated work.
