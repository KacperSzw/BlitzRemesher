# Completed one-hour training run

The requested run completed its **3,600,000 ms learning window**, exited normally
and published a verified checkpoint at **413,184 updates**. Source
`2246b9195edd04a5f4b31cd898bb20f4a03e077f` remained frozen. The model used schema 4,
width 64, exhaustive coverage teacher, seed 101, two persistent teacher workers,
candidate batch 4, 16 states per request, 128 updates per shard and update batch
512 on an A40. The frozen curriculum contains 16 conditions over four training
assets; this was not a full-corpus training or release-quality evaluation.

The run recorded 53,260 fresh states including predecessor states and 51,648
requested-condition states. Reported failed/empty conditions were zero and
curriculum coverage was complete. The duration is elapsed learning-window time,
including teacher waiting and orchestration; it is not 60 minutes of optimizer
GPU computation. Training reports and full history are retained as compressed
raw evidence, so cumulative, per-block and last-minibatch metrics can be inspected
without treating a lower training loss as proof of better LOD quality.

The final checkpoint SHA-256 is
`2d19118f645aaa059ce7ff7b11d35108944bc46cdc54472712e4f44b3de042ed`;
the model SHA-256 is
`08fe5fabd82e8352116161c7c2a9496bada9a3c429fa7df9a95a8f8f15a24ace`.
The six indexed checkpoint artifacts were checked locally and against collection.
Numerical verification passed: native/Torch maximum absolute difference was zero;
FP64 maximum absolute difference was about 3.03e-6. Tensor/model payloads are
excluded from Git; checkpoint index and verification records preserve identities.

## Local quality result

The matched initializer/final diagnostic ran locally on two development assets
with four LODs, coverage audits, both neural origins, UV preservation, 64 action
trials and action batch 16. Both processes exited normally. All 48 LOD1–3 source
and adjacent confirmations passed; there were no resource or confirmation
failures. Eight of the final learned run's 12 proposals exhausted their trial
budgets. This is a censored diagnostic with `score: null`, not a release result.

| Asset | Ranking | Initial LOD1 / LOD2 / LOD3 | Final LOD1 / LOD2 / LOD3 |
| --- | --- | --- | --- |
| Shelves | Constant | 524 / 262 / 244 | 524 / 262 / 262 |
| Shelves | Learned | 518 / 259 / 247 | 514 / 257 / 251 |
| Moon rock | Constant | 3222 / 1910 / 1902 | 3274 / 1638 / 1636 |
| Moon rock | Learned | 1862 / 932 / 930 | 2274 / 1138 / 1134 |

The completed hour did **not** establish consistent improvement: learned shelves
improved by four/two triangles at the first two LODs and worsened by four at the
third; learned moon rock worsened by 412/206/204 triangles. Constant ranking still
uses the model's learned placements and is not a classical baseline. Local
shared-workstation timings are not isolated performance evidence. See the
[intermediate diagnostic](../hour-quality-diagnostics/README.md) for the
183,168-update checkpoint, batch-1 ablation and source-review qualifications.

## Completion and deferred validation

The user requested local model/teacher quality checks before further remote
learning. An operator watcher waited for normal hour completion and verified
checkpoint publication, then cancelled the experiment supervisor. The subsequent
nonzero experiment exit records that deliberate post-hour handoff; it does not
mean the training hour failed. Further paired learning and remote full quality
validation were deferred. The archive retains the operator policy/handoff,
supervisor report and native child exit status separately. The raw post-hour
initial audit reports SIGTERM cancellation; the attempted warm profile has zero
rows and a cancellation error. Neither is counted as a completed comparison.

Collection verified the 3,468,751,232-byte archive against SHA-256
`b5880b0b90a6a04f047ccc4158ea4a6eb5e9f9c9d9b3b66538d5e8bafad8c041`.
Compute terminated at 2026-10-01 15:36:54.523 UTC; the 15:37:54.691 UTC
provider check reported **zero Pods and zero network volumes**. Rental duration
was 5,243.377 seconds. The conservative estimate is $0.844766 for this rental
and $1.596277 cumulatively, below the $3.50 authorization; this is not an invoice.

The remote engineering gate passed **50/50 CTests** in 76.46 seconds with no
skips, clean Vulkan validation, zero errors in both Compute Sanitizer checks
and 48 completed join-retirement stress rounds. The archive contains metadata
for all 432 saved checkpoints, without tensor payloads.

The evidence includes remote engineering contracts, Vulkan validation,
Compute Sanitizer and four join-retirement stress logs, original frozen
curriculum/input identities, complete hour history, checkpoint metadata and the
local quality raw outputs. `manifest.json` records compressed and raw hashes.
Sanitized rental metadata retains timing and receipts; Vulkan machine UUIDs
are omitted from an explicitly marked derived copy. SSH endpoints, credential
and identity files, data shards, mesh caches, wave models and tensor payloads are
excluded. No additional paid run or optimizer work was performed for this archive.
