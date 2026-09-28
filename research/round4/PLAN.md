# Round 4 — large starting sizes and useful final LODs

The user requested larger LOD0 screen sizes, larger source-error allowances,
fewer redundant levels and investigation of quad/tiny-triangle costs.
These are separate hypotheses, not permission to relabel a relaxed visual
contract as an algorithmic improvement.

The frozen eight-asset development pilot remains the main comparison set.
Start with 512→16 px, eight scheduled levels, the 2→3 px transition curve,
hybrid coupled rebuild and eight proposals per level. Compare source caps
of 2 and 8 px; use 128→16 px as a separate scale scenario. Camera sets and
sampling match the earlier pilot. Reserve validation assets for confirming
the accepted change; do not tune against held-out assets.

The source cap only clamps the cumulative policy. Raising a cap beyond that
policy does nothing. Report the actual source limits, the gates rejecting
proposals and the reducer's unaudited minimum independently.

Hypotheses to test:

1. A requested low triangle count may stall because every chosen placement
   is invalid, topology locks accumulate, or the beam receives too few
   useful target counts. Diagnose before changing constraints.
2. Exact consecutive duplicates need only one runtime mesh/LOD. Preserve all
   scheduled audit slots and the existing SCORE denominator. Never merge
   different meshes merely because their triangle counts match. Skipping a
   genuinely different mesh would require auditing its replacement transition.
3. Larger source caps can help only when the source gate is limiting.
   Compare algorithm changes within identical settings and work budgets.
4. Final-LOD quality deserves separate reporting: last-level triangles,
   category-balanced final and last-three-level retained ratios, exact
   duplicate slots and runtime LOD count, alongside the unchanged main SCORE.

Render-cost diagnostics are not acceptance criteria or a new combined SCORE.
At fixed views, measure projected triangles below 1 and 4 pixel², triangles
covering zero pixel centers, per-primitive covered samples P, touched aligned
2×2 quads Q, and geometric lane utilization P/(4Q). Count each primitive
separately. Keep pre-depth overlap and final visible-pixel coverage separate.
Use a deterministic sample-center/top-left fill convention, no MSAA, and
record screen resolution, culling, camera set and clipped-view status.
These are CPU geometry proxies, not hardware helper invocations or GPU time.

Primary references:

- [Quad-fragment merging, Fatahalian et al.](https://graphics.stanford.edu/papers/fragmerging/)
- [Microsoft rasterization conventions](https://learn.microsoft.com/en-us/windows/win32/direct3d9/rasterization-rules)
- [meshoptimizer simplification and aggressive fallbacks](https://github.com/zeux/meshoptimizer#simplification)

Run batches under 50 minutes, at most four CPU workers and 24 GiB combined.
Preserve baseline executables and source/build/configuration hashes. Record
negative results and changed scenarios. Accept changes only with correctness
checks and measured benefit; do not make plateau removal alter the audit metric.
