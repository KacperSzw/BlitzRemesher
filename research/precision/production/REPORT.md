# Single production precision path

Recorded 2026-09-28. Production uses float32 geometry streams, RGBA8 colors,
double quadric storage/arithmetic and double candidate costs. Coverage uses
packed masks; normal/attribute profiles and reference tests use full rasters.

Float quadric/cost implementations and the full-coverage build switch have
been removed. Quadric and candidate structs are concrete 80-byte and 32-byte
records. Existing C++ storage/backend queries and optional numerical/timing
diagnostics remain available. C ABI 2, output/chain/objective modes, source
immutability and all visual thresholds retain their contracts.

The [restoration patch](../restore-experiments.patch) retains the discarded
implementations outside production. Applying it to an isolated source copy
reproduced the archived source-tree hash byte for byte. The [UNORM16 findings](../unorm16/REPORT.md)
are also retained; no UNORM16 position path is shipped.

## Build identity

- Production source SHA-256:
  `6ced0b3311339a2d04c0130c660b67032fd8c0c676f051149404ef44ad8f89c3`.
- Archived reference source SHA-256:
  `e125d42d7d199d1cb90a8cda13ae098f69500ea8729ee97e5995cb5a2d57f9f4`.
- [Production executable stamp](builds/production.json) identifies the exact
  tested binary. The archived stamps and runs were not overwritten.

## Verification

| Configuration/check | Result |
|---|---|
| Clang release | Focused checks and all 5/5 CTest targets passed |
| Clang ASan + UBSan | Focused checks and all 5/5 CTest targets passed |
| GCC scalar, tools disabled | 4/4 CTest targets passed |
| Shared library, tools disabled | 4/4 CTest targets passed |
| Installed C and C++ consumers | Configured, built and executed successfully |
| Removed CMake variants | Float quadrics, float costs and full coverage requests rejected |
| Research runner | `full`, `quadrics`, `costs`, `both` rejected |
| Restoration patch | Applies cleanly; original source hash recovered |

Build/test output is retained in [logs/](logs/). Existing contract suites cover
RGBA8 import/export, numerical boundaries, raster equivalence, source reuse,
ownership and ABI behavior. No additional tunable defaults or topology goldens
were introduced.

## Frozen reference replays

The comparison script checks manifest/config/camera/protocol hashes, compiled
storage/backend metadata, completion, canonical attributes, geometry and
attribute output hashes, complete result JSON, retained ratios and numerical
counters. Both scenarios matched exactly on all ten asset comparisons; see
[agreement.json](agreement.json), [summary.json](summary.json), and
[raw production runs](../../runs/precision-production/).

| Scenario | Assets | SCORE | Final retained | Last three retained | Max changed coverage |
|---|---:|---:|---:|---:|---:|
| 32→16 development pilot | 8 | 62.41592 | 37.10587% | 37.24313% | 59.53488% |
| Separate 512→16 hybrid scenario | 2 | 76.40993 | 3.43009% | 7.08362% | 35.28817% |

Both runs have zero failed assets and zero unreduced fallbacks. Every retained
scheduled slot keeps its original source/adjacent thresholds and audit record.
The two scenarios are separate workloads, not evidence of a cross-scenario
algorithmic gain.

Generation took 73.43 seconds for the pilot and 37.31 seconds for the larger
screen pair; process peak RSS was 298.27 MiB and 64.22 MiB, respectively.
These are single correctness replays under shared load, not new performance
comparisons. The repeated measurements remain in the original experiment
report, along with its workload and camera-set limits.

To replay inside `nix develop`:

```sh
bash tools/precision-research.sh build production
bash tools/precision-research.sh bench production pilot 1
bash tools/precision-research.sh bench production large 1
bash research/precision/verify-production.sh
```

Resume requires matching binary, source and input identities. Rebuilds after
a revision change need fresh run identities; preserve existing run/stamp files.
