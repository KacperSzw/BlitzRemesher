# Added vertex budget smoke check

This is a bounded development smoke check, not a quality score or a claim of
global optimality. The hypothesis was that a hard vertex budget would retain
shared middle levels and spend bytes on a compact final level. The first
implementation did not: with 8 proposals spread over many paths, its sole
source-path rebuild targeted too many triangles to fit. Its recorded outcome
was 524/376/376/376 triangles, zero added vertex bytes, and 10 budget rejections.
The search now keeps the source path eligible and bounds rebuild targets by
remaining vertex bytes; actual bytes and visual audits still decide acceptance.

Input: the archived development `ph_painted_wooden_shelves` chain's source
mesh, 548 vertices and 524 triangles. The source glTF SHA-256 is
`6b30bf9461082ad616236c19759a60de880fd7ef269d6933cc81c1ce42110c01`;
its binary SHA-256 is
`5c1a5f4988f64ea3cc54946f8dba47aa1bd117fd2b94931c2d13e60afcf96ae9`.
Build: release preset, Clang 21.1.8, Git base
`7466a9e96843db6b39408c7c38d3090e5a4d9125` plus this working-tree change.
Each run used 4 levels, 64→16 px, coverage profile, 8 proposals per level,
beam width 4, identical cameras, sampling, and scalar reduction. Config files
and raw result JSON are next to this report.

| Policy | Scheduled triangles | Runtime slots | Added / allowed vertex bytes | Resident bytes | Generation seconds |
|---|---|---|---:|---:|---:|
| [20%, 0 overhead](forward-budget-baseline/smoke-shelves-result.json) | 524 / 376 / 376 / 36 | 0, 1, 3 | 2,752 / 3,507 | 31,520 | 0.119 |
| [20%, 5% overhead](smoke-shelves-overhead500-result.json) | 524 / 376 / 376 / 36 | 0, 1, 3 | 2,752 / 3,507 | 31,520 | 0.119 |
| [Uncapped, 5% overhead](smoke-shelves-legacy-result.json) | 524 / 132 / 78 / 40 | 0, 1, 2, 3 | 16,736 / disabled | 43,560 | 0.224 |

The current 20% default produced a compact final mesh, but retained many more
triangles in the middle than the uncapped policy. The 5% allowance made no
difference within this bounded pool. The single-run times are diagnostic only;
shared-workstation noise and different search paths prevent a speed claim.
All three runs completed and their scheduled source and adjacent audits passed.
The new default's exported [glTF chain](../../examples/shelves-budget/chain.gltf)
and [manifest](../../examples/shelves-budget/lods.json) are preserved for inspection.
The 20% chain binary SHA-256 is
`94a81fddd700cf61ef4ee7e7b230fe7359709b5b653c4b6374d27d6ed49fbd16`;
the uncapped chain binary SHA-256 is
`e15bd4d7fb0d832be830eceedf814092b8af8ad6e5cc4cd95bc2fc1f3d8820a9`.

## Verification and limits

- Release CTest: 9/9 passed. Clang address/undefined sanitizer CTest: 9/9 passed.
- Shared-library build produced `libblitzremesher.so.5`; the new runtime storage
  C symbol is exported. `git diff --check` and JavaScript syntax checks passed.
- The example glTF reimported with 548 source vertices and 524 triangles;
  its 31,520-byte binary matches the reported resident byte count.
- The frozen two-asset hybrid development pilot and held-out validation were
  not rerun: their original source meshes are absent from this checkout.
  This one-mesh smoke check does not establish a corpus-wide benefit.
