# BlitzRemesher

## User requirements

Build a reusable C++20 library for game-engine import pipelines that generates
configurable LOD chains for static, opaque meshes.
Minimize triangles within audited screen-space error limits against the previous LOD and source, using selectable coverage, normal, and attribute quality profiles.
Automatically choose vertex storage and reduction strategy within the configured
triangle/memory tradeoff;
measure bake time and improve results through reproducible comparisons.

Audited views are not an all-view or optimality guarantee. Read docs/SPEC.md
for behavior and research/PROTOCOL.md before changing scores or comparisons.

## Implementation

Use contiguous indexed storage, compact IDs and flags, explicit ownership, and
borrowed views. Establish ranges, sentinels, lifetimes, alignment, and resize
invalidation; avoid per-vertex objects. Measure before adding SIMD or wider fields.
Never mutate supplied streams. Reuse mode preserves source vertex IDs and
attributes byte for byte. No STL type or exception crosses the C ABI.

## Iteration

Start algorithm work with the smallest deterministic fixture or regression.
Build the affected target and run focused tests before any chain benchmark.
Try one or two development meshes with bounded pilot settings to catch failures;
this is a smoke check, not a default-quality score.
Compare promising changes on the frozen development pilot with identical inputs,
settings, cameras, work budgets, and protocol. Escalate to validation and the
full corpus only after a pilot signal or for release evidence. Use held-out
assets only for release audits, never tuning.
Keep failures and unreduced fallbacks visible. Save hypotheses, raw measurements,
negative results, and incomplete runs without assigning them an aggregate score.

## Completion

Tests own settings and protect contracts, boundaries, and lifetimes rather than
tunable defaults or arbitrary topology. Finish code changes with relevant CTest
and sanitizer checks; record what ran. Commit descriptions explain why the
change was made and its measured consequences.
