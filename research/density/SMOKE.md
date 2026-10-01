# Bounded smoke observations

One asset per run. These are diagnosis and runtime checks, not the frozen eight-asset pilot or independently qualified chains. Times are shared-workstation observations.

| Run | Final triangles | Added / allowed bytes | Seconds |
|---|---:|---:|---:|
| density-v1-attributes-smoke-stool | 6532 | 0 / 28736 | 12.21 |
| density-v1-attributes-smoke-trunk | 16854 | 0 / 60704 | 22.46 |
| density-v1-screen-smoke-stool | 6532 | 0 / 28736 | 10.84 |
| density-v1-screen-smoke-trunk | 17978 | 0 / 60704 | 14.66 |
| density-v1-uncoupled-smoke-stool | 6532 | 0 / 28736 | 10.74 |
| density-v1-uncoupled-smoke-trunk | 16854 | 0 / 60704 | 19.91 |
| density-v2-merged-attributes-smoke-stool | 6532 | 0 / 28736 | 14.06 |
| density-v2-merged-attributes-smoke-trunk | 16854 | 0 / 60704 | 25.85 |
| density-v2-merged-smoke-stool | 6532 | 0 / 28736 | 11.85 |
| density-v2-merged-smoke-trunk | 17978 | 0 / 60704 | 15.56 |
| density-v3-merged-attributes-smoke-stool | 6532 | 0 / 28736 | 17.45 |
| density-v3-merged-attributes-smoke-trunk | 16854 | 0 / 60704 | 25.86 |
| density-v3-merged-smoke-stool | 6532 | 0 / 28736 | 14.22 |
| density-v3-merged-smoke-trunk | 17978 | 0 / 60704 | 21.87 |
| density-v4-shared-smoke-stool | 6532 | 0 / 28736 | 15.99 |
| density-v4-shared-smoke-trunk | 17698 | 9632 / 60704 | 25.12 |
| density-v5-shared-smoke-stool | 6532 | 0 / 28736 | 16.66 |
| density-v5-shared-smoke-trunk | 17698 | 9632 / 60704 | 25.67 |

The v1–v3 variants selected no newly stored vertex buffer. Their 16,854-triangle trunk result was already observed and failed the previous independent audit. Shared source vertices in v4 admit a 17,698-triangle trunk with 9,632 added bytes; its fresh dense audit fails at LOD1–6. Stool remains unchanged. The v3 merged emitter admits denser rebuilt proposals under the storage cap, but their source appearance audits fail. Raw targets and gates are retained in [smoke.json](smoke.json).
