# Vegetation improvement round

Started from `26ad36e`, preserving the frozen opaque corpus, protocol, source
collection and AGENTS.md. Geometry-only coverage and both vertex modes remain
the contract. Alpha, textures and shading are unscored. Broadleaf tree and Fern
02 are the initial development cases; the family split is in `cohort.json`.

## Fixed comparisons

512→16 px, eight scheduled levels, 2→3 px transition curve, source cap 8 px,
hybrid, beam two. Search 6+2 cameras at 2×; audit 12+4 at 4× refined to 8×.
Budgets 8, 32 and 64 are separate comparisons. The actual final source limit
is 6.8905 px; raising the inactive 8 px cap would not change this contract.

Test single changes before combining them: boundary-plane QEM at weights
0/1/10/100; legal endpoint and edge-constrained placement; adaptive target
selection; dynamic multi-view component subset proposals. The second ablation
tests independent collapses within original index/UV charts, without welding
or modifying shared vertex streams. Material locks remain. The final path also
guards UV orientation during these experimental shared-vertex contractions.

The native defaults remain available as a control. A changed proposal cannot
bypass source/adjacent search or acceptance cameras. Keep exact source fallback
and runtime duplicate compaction. Frozen meshoptimizer 1.3 is an external
research adapter, with UVs and material partitions retained. Compare default,
pruning and permissive-plus-pruning options, and keep unsupported inputs visible.

## Validation and promotion

Run contract fixtures before pilots; expand promising settings to six board
examples and then 28 development models. Reserve the 12 models from the other
four families for validation; these are not unseen holdouts. Promote only after
matched-budget tail retention improves without aggregate scheduled-chain or
opaque-pilot regression. Individual regressions remain in the report.

Recheck finalists' last three levels with 642+64 cameras, rotation seed
`0xB1172031`, sampling 8× refined to 32×. Dense failures block promotion, rather
than changing thresholds or silently replacing failed results. Report coverage
change, tiny triangles and quad proxies separately from geometric error.
Repeat final timings three times with one worker. Other batches use two workers
(at most four across concurrent tasks), 24 GiB combined memory and 50-minute
checkpointing. Every resume must match input/configuration/build identities.

Update the existing chain board with before/after fern and tree examples in
both modes, timings and coverage diagnostics. Preserve raw failed experiments,
commit and push the completed work, and open the resulting board.

## References

- [Garland–Heckbert QEM](https://mgarland.org/research/quadrics.html): boundary
  planes perpendicular to faces address face-plane error's planar nullspace.
- [Image-driven simplification](https://faculty.cc.gatech.edu/~turk/my_papers/image_simp_tog2000.pdf):
  rendered impact informs proposals. Whole-component marginal coverage removal
  here is an engineering experiment, not a claimed reproduction of that paper.
- [meshoptimizer](https://github.com/zeux/meshoptimizer/tree/9e1f07b159d3cb777f1c67ed31fc11fd117986f4):
  pruning, permissive and movable-vertex research references.
