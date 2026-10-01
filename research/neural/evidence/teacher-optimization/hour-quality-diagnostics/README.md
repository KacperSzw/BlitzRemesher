# Intermediate hour diagnostics

This archive compares the initializer with the **183,168-update intermediate
checkpoint**, captured after 1,592,177 ms of the learning window. Training was
still running. The remote source was `2246b9195edd04a5f4b31cd898bb20f4a03e077f`,
and its teacher strategy was **exhaustive**, not coverage-core-first. The generic
runner's raw `final` variant and filenames mean the intermediate model here;
they do not establish completion of the requested hour.

The local diagnostic used two development assets, four LODs, coverage audits,
both neural origins, UV preservation and 64 action trials. Initial/intermediate
processes exited normally. All emitted LOD1–3 source/adjacent confirmations
passed. The results remain trial-censored, with `score: null` and no release
quality claim. These are shared RTX 2080 workstation runs; timing is not an
isolated performance measurement.

| Batch | Asset | Ranking | Initial LOD1 / LOD2 / LOD3 | Intermediate LOD1 / LOD2 / LOD3 |
| --- | --- | --- | --- | --- |
| 16 | Shelves | Constant | 524 / 262 / 244 | 500 / 250 / 240 |
| 16 | Shelves | Learned | 518 / 259 / 247 | 514 / 257 / 247 |
| 16 | Moon rock | Constant | 3222 / 1910 / 1902 | 3294 / 2036 / 2034 |
| 16 | Moon rock | Learned | 1862 / 932 / 930 | 2902 / 1452 / 1448 |
| 1 | Shelves | Constant | 484 / 467 / 442 | 484 / 467 / 448 |
| 1 | Shelves | Learned | 398 / 270 / 248 | 500 / 373 / 351 |
| 1 | Moon rock | Constant | 3200 / 3072 / 3038 | 3200 / 3072 / 3042 |
| 1 | Moon rock | Learned | 3176 / 3048 / 3020 | 3176 / 3048 / 3040 |

Constant means equal ranking scores with stable action order; it still executes
the model's predicted placements. It is not an independent classical baseline.
The batch-1 ablation changes only action batch size, with the same binary,
models and settings otherwise. It did not restore the initial learned-ranking
advantage on these observations. All 48 batch-1 proposals stopped at their trial
budget. Equal trial counts allow fewer edits per accepted attempt at batch 1;
cross-batch final triangle counts are therefore not equal-work comparisons.

## Source review and next questions

The compressed Prophet response is preserved verbatim, including its original
reference tokens. The following conclusions were independently checked against
the repository source, which is unchanged from the remote revision for these
functions:

- `prepare_placements` in [placement_teacher.hpp](../../../../../training/placement_teacher.hpp)
  labels and commits the independently audited teacher-winner placement.
  `preferred_actions` in [teacher_labels.hpp](../../../../../training/teacher_labels.hpp)
  prefers fewer faces, then smaller exact audit margin, retaining exact ties.
- `rank_actions` and `decode_placements` in
  [action_gpu.cu](../../../../../src/neural/action_gpu.cu) use output 0 for learned
  ordering but execute placements decoded from outputs 3–11. Pass logits 1/2
  do not affect ranking. This is a real supervision/execution distinction;
  it is not yet proof of the cause of the intermediate regression.
- `select_batch`, `verdict` and `GpuActionState::execute` audit joint independent
  batches and halve failures, unlike the teacher's singleton commits. The
  batch-1 result does not support batching alone as an explanation here.
- `placement_denominators` and `placement_gradient` in
  [update_cuda.cu](../../../../../training/update_cuda.cu) give all-preferred
  states no pairwise ordering gradient. Shared-trunk placement/pass gradients
  can still move their scores. The static review found no obvious hinge-sign,
  mask or target-index error; it is not a proof that all implementation is correct.

The separate earlier-local label analysis concerns the completed **24,832-update
five-minute run**: 1,983 of 3,200 stored states (61.96875%) had four preferred
actions out of four queried. All 388 referenced index/trajectory hashes were
checked and the count independently reproduced. This is neither live-hour
label coverage nor a count of replay sampling frequency.

A useful next diagnostic is exact auditing of each current actor placement on
fixed states, then comparing teacher-preferred and actor-preferred masks and
top-ranked choices. Separate actor-placement rank labels and a frozen actor are
experimental training proposals, not validated fixes. Their extra exact audits,
distinct validity masks and replay compatibility need explicit treatment.

## Contents and verification

`manifest.json` records compressed and raw SHA-256 values for 23 artifacts:
both diagnostics' raw audits/logs/reports/settings, checkpoint index/verification/
capture, the source review and earlier-local label analysis. All six checkpoint
files were checked against their index; model and tensor payloads are omitted.
`record.json` preserves model/binary identities, outcomes and limitations.
Additional Prophet audits and the final-hour checkpoint are outside this archive.
