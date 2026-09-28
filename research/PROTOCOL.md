# Scoring protocol v3

Version 3 adds an optional per-view opaque-coverage area limit to the visual
contract. For reference mask A and candidate mask B, changed area is
`|A xor B| / |A or B|` (`1 - IoU`), with two empty masks defined as zero.
The maximum over audited views must be at most `max_changed_area` for both
source-to-LOD and adjacent-LOD comparisons. One empty mask gives changed area
one. The distance gate still applies independently. The default limit is 1.0
for compatibility; the first quality preset uses 0.5. The area fraction is
measured on conservative supersampled opaque-geometry masks. It does not
measure texture opacity, shading or all-view error. Report both the maximum
fraction and its camera index for each comparison.

Use separate scenario labels for the 1.0 and 0.5 limits. A lower SCORE under
the 0.5 contract is a measured cost of tighter quality, not an algorithm
regression. Compare cap-only and cap-plus-adaptive search only at the same
0.5 limit, input, cameras, sampling, candidate budget and build conditions.
Fresh v3 uncapped runs establish the reference; archived v2 rows are context,
not their baseline. Dense rotated-camera tail checks are independent audits,
not replacements for the scheduled-chain SCORE. Any failing or incomplete
view remains visible, including the distance and area limits and worst views.

Version 2 fixes canonical vertex colors to linear RGBA8. Normalized float and
unsigned-16 file colors are rounded to nearest (ties upward) at import;
nonfinite/out-of-range colors fail import. Source streams supplied to the
library are already RGBA8 and remain immutable. All variants and external
baselines consume the same canonical inputs; record their attribute hashes.
The SCORE formula is unchanged from v2. Historical v1 results are archived
observations, not comparable baselines for v2 or v3. Rerun every baseline
under the current protocol before a new comparison; unsupported capabilities
remain failures.

Precision comparisons record quadric/candidate record sizes, packed/full
coverage mode, generation and stage timings, numeric rejection counters,
process peak RSS, output hashes and final/last-three retained ratios. Repeat
timings with identical workloads and disclose shared-workstation noise.

For each complete asset chain:
r_m=sum(T_i,i=1..N-1)/((N-1)*T_0).
SCORE=100*(1-mean_categories(mean_assets(r_m))).

Every candidate must pass both adjacent and source visual gates. Separate
leaderboards by output mode, profile, chain mode and configuration. Main
benchmark N=8; N=6 and N=12 are separate scenarios. Time and memory never
compensate for failing the visual gates. An unchanged chain scores zero.
Unsuccessful/unreduced assets remain in the denominator. Incomplete batches
have no SCORE. Unsupported baseline capabilities are explicit.

Freeze manifests and protocols before comparing methods. Record Git revision,
dirty-state hash, source SHA-256, configuration and camera hashes, baseline
revision, compiler, hardware, ISA, threads, seeds, wall time, peak RSS, errors,
output hashes and completion/failure reason. Development and validation guide
choices; held-out results are only release audits.

Seed: 0xB1172026. Group related source identities before splitting. Stratify
80/20/20 across 30 rocks, 30 organic, 40 manufactured and 20 stress cases.
Normalize viewing scale virtually; never rewrite reuse-mode source streams.

Run batches below one hour (checkpoint at 50 minutes), at most four workers
and 24 GiB combined memory. Resume only matching input/protocol/build hashes.
Use fixed work budgets for scored comparisons. Publish all per-asset rows,
worst views, failures and negative findings alongside aggregates.

## Primary research

- QEM: https://www.mgarland.org/research/quadrics.html
- Progressive meshes: https://hhoppe.com/proj/pm/
- Image driven: https://faculty.cc.gatech.edu/~turk/my_papers/image_simp_tog2000.pdf
- Probabilistic quadrics: https://graphics.rwth-aachen.de/publication/03308/
- Reference code: https://github.com/Philip-Trettner/probabilistic-quadrics
- Memoryless/policy baseline: https://doc.cgal.org/latest/Surface_mesh_simplification/
- Production baseline: https://github.com/zeux/meshoptimizer
- Corpus: https://polyhaven.com/our-api
- Corpus: https://ambientcg.com/api/v3/assets?type=3d-model
- Corpus: https://3d-api.si.edu/api-docs/

Coverage Hausdorff and the attributed product metric are project choices.
Neither QEM nor a finite camera audit establishes an all-view guarantee.
