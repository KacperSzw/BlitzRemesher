# Scoring protocol v1

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
