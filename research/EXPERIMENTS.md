# Experiment log

The initial scenarios are small **pilot** audits: 32→16 pixels, 8 total
LODs, the unchanged 2→3 pixel transition curve, 6+2 search cameras and
12+4 audit cameras, 2×/4× supersampling with refinement to 8×.
They do not stand in for the default 642+64-camera quality setting.
The eight pilot assets were selected before reduction measurements:
the smallest and median development asset under 150k triangles in each
of the four categories. The frozen full corpus is separate.

## Round 1 — proposal objectives and external references

Hypothesis: positional regularization or extra normal-curvature weighting
might improve the valid triangle count over plain QEM.

The same conservative native reducer, direct chain policy and work budget
were compared. QEM scored 55.6927, positional regularization 55.7578 and
curvature-weighted QEM 55.6709. The small differences provide no persuasive
reason to replace plain QEM. Positional regularization is **not**
probabilistic quadrics.

Separate executables compare meshoptimizer 1.3, Fast Quadric and CGAL 6.1.1
(Lindstrom–Turk, plane QEM and actual probabilistic plane policies).
Their candidates use the same scheduler and visual gates. Adapters expose
geometry-only, single-material capabilities; unsupported cases remain
recorded failures in the denominator. CGAL's manifold rejection is preserved.
No silent welding or repair is performed in an external adapter.

## Round 2 — attribute wedges, topology and chain search

Observation: Bell X-1 has 298,542 render vertices for 99,586 triangles.
Conservatively locking every duplicated position prevented almost all
reduction, even though a large part of that duplication encodes separate
attribute wedges.

Hypothesis: simplify shared positional topology, trace surviving source
faces, and move all wedges at a collapsed position together while keeping
their separate UVs, normals, colors and material IDs.

The coupled rebuild proposal increased the direct coverage pilot SCORE
from 55.6927 to 87.6755 under unchanged visual gates. Bell X-1's mean
retained ratio changed from 0.9993 to 0.01613. A hybrid chain with component
pruning scored 95.9834; this is explicitly a different chain-mode scenario.
The original conservative proposal remains available, and vertex reuse
still uses original endpoints and streams.

Normals and attributes are evaluated as separate scenarios. Their much
lower initial scores are a substantive limitation: moving positions while
retaining wedge attributes can violate the strict sampled normal metric.
Do not hide that result or reduce the normal weight to claim an improvement.
Attribute-aware placement/transfer is a priority for subsequent research.

## Round 3 — allocation layout and SIMD

Hypothesis: replace one hash node per position with a sorted 16-byte
position/index entry; preserve duplicate detection while reducing temporary
allocation overhead and memory. Source positions, collapse ordering and
acceptance gates remain unchanged. Four bounded topology flags share one
byte; temporary material IDs use 16 bits plus a seen flag, and their buffer
is released before collapse rounds.

The first explicit AVX2 distance-transform implementation was slower:
0.965× scalar speed on the fixed filled-disc fixture. Calling the vector
kernel for single-sample spans erased its benefit. Restricting dispatch to
spans of at least eight samples measured 1.081× on the same fixture, with
identical checksums. These are shared-workstation medians, not a general
8% end-to-end speed claim. Scalar remains selectable and is tested against
an independent brute-force reference.

Raw measurements: microbench-initial.json and microbench-optimized.json.
End-to-end repeats and current run status are in RESULTS.md.

## Validation findings

- Empty optional vectors can retain a non-null allocation. Validation now
  accepts empty streams regardless of retained capacity.
- Fuzzing found a degenerate-face budget bug that could remove the last real
  surface. The reducer now separates zero-area faces from the collapse
  budget; a permanent regression seed and contract test protect this case.
- Baseline bridge testing found a scratch-path lifetime bug in the
  research-only callback. The callback now owns its scratch path.
- 681,605 sanitized fuzz executions passed after the first fix; an expanded
  run including coupled wedges and component pruning passed 596,246 inputs.
  These are bounded fuzz campaigns, not proof of memory safety.

Preliminary runs were recorded before the first Git commit. They retain
binary/source hashes and complete per-asset rows; subsequent runs should
start from committed source. The baseline comparison is an adapter and
end-to-end study, not an isolated reducer speed ranking.
