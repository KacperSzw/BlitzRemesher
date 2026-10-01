# Decision: retain the graph search as an experiment

The implementation adds a bounded graph search and extends the engine manifest. Production defaults remain unchanged. The configured small-camera pilot and independent qualification are separate outcomes.

One graph pass reduces category-balanced chain retention by **11.52%**, with **0 asset regressions** on the frozen eight-asset coverage pilot. Three passes improve retention 16.34%, but the largest single-run per-asset bake ratio is 7.21×. The appearance preset uses the normal/color/material metric; its results must not be inferred from coverage.

| Promotion gate | Result |
|---|---|
| pilot | Pass |
| validation | Pass |
| time | Pass |
| independent coverage | Not passed |
| independent appearance | Not passed |

Validation (the original twenty assets): 19.32% lower chain retention, 0 asset regressions.

## Paired timings

Three sequential full-pilot pairs, alternating order; shared workstation. Other independent audits may occupy separate CPU cores. All output attribute hashes match the scored runs.

| Asset | Three paired ratios | Median |
|---|---|---:|
| ph_painted_wooden_shelves | 3.15×, 2.89×, 3.07× | 3.07× |
| ph_metal_stool_02 | 3.68×, 2.99×, 3.64× | 3.64× |
| ph_grass_bermuda_01 | 2.00×, 1.51×, 1.97× | 1.97× |
| ph_dead_quiver_trunk | 2.68×, 2.02×, 2.31× | 2.31× |
| ph_moon_rock_02 | 3.01×, 2.49×, 2.83× | 2.83× |
| ph_rock_face_02 | 2.87×, 2.77×, 2.79× | 2.79× |
| si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652 | 1.42×, 1.53×, 1.41× | 1.42× |
| si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7 | 2.10×, 2.15×, 2.03× | 2.10× |

## Independent full-chain audits

All scheduled LOD1–7 comparisons use 642+64 cameras, seed `0xA1172026`, 8× sampling refined to 32×. A comparison can fail at its first witness; `Measurement.complete=false` then means remaining views were unnecessary for rejection. The audit report is complete only when every planned comparison has a decision. Resource/time interruptions remain incomplete.

| Preset | Asset | Levels attempted | Result | Failure witnesses |
|---|---|---:|---|---|
| coverage | ph_painted_wooden_shelves | 7/7 | Fail | LOD2 adjacent: 2.046 / 2 px, area 0.36%, view 227 |
| coverage | ph_metal_stool_02 | 7/7 | Pass | — |
| coverage | ph_grass_bermuda_01 | 7/7 | Fail | LOD5 source: 2.199 / 3 px, area 51.01%, view 93; LOD6 source: 1.330 / 3 px, area 50.10%, view 15; LOD7 source: 1.027 / 3 px, area 50.59%, view 122 |
| coverage | ph_dead_quiver_trunk | 7/7 | Fail | LOD7 source: 2.212 / 3 px, area 51.59%, view 179 |
| coverage | ph_moon_rock_02 | 7/7 | Incomplete | LOD1 source: raster resource limit after 74 views; LOD1 adjacent: raster resource limit after 74 views |
| coverage | ph_rock_face_02 | 7/7 | Pass | — |
| coverage | si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652 | 7/7 | Pass | — |
| coverage | si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7 | 7/7 | Pass | — |
| appearance | ph_painted_wooden_shelves | 7/7 | Fail | LOD7 source: no match within metric bound, area 0.02%, view 66; LOD7 adjacent: no match within metric bound, area 0.02%, view 53 |
| appearance | ph_metal_stool_02 | 7/7 | Pass · unchanged source | — |
| appearance | ph_grass_bermuda_01 | 7/7 | Pass · unchanged source | — |
| appearance | ph_dead_quiver_trunk | 7/7 | Pass · unchanged source | — |
| appearance | ph_moon_rock_02 | 7/7 | Fail | LOD5 source: no match within metric bound, area 0.05%, view 44; LOD5 adjacent: no match within metric bound, area 0.05%, view 44; LOD7 source: no match within metric bound, area 0.13%, view 339 |
| appearance | ph_rock_face_02 | 7/7 | Pass · unchanged source | — |
| appearance | si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652 | 7/7 | Pass | — |
| appearance | si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7 | 7/7 | Pass · unchanged source | — |

**1 reduced appearance chain qualifies** under these independent checks. Identity checks on unchanged source fallbacks are valid but provide no reduction benefit. Texture images and normal maps remain unscored. Held-out release assets have not been used.

## Next algorithm step

Start with a deterministic appearance rejection fixture and compare the coarse search screen with the full audit. The evaluator refines coverage uncertainty but does not refine a sampled appearance failure; measure whether that screen discards candidates that would pass the finer audit. Then use failing camera/attribute witnesses to target proposal density and placement. The graph improves combination of existing proposals. Three coverage passes exceed the bake allowance, while one appearance pass gives no gain. These results do not isolate screening, target density, and collapse placement as causes.

[Measurements](REPORT.md) · [Offline board](../../examples/reduction-board/index.html) · [Progress SVG](progress.svg) · [Reproduction](README.md). Earlier unsuccessful v1/v2 and smoke measurements remain in the raw run directories; they do not enter this pilot score.
