# Coverage-area quality measurements

[Interactive development board](board/index.html) · [Raw derived analysis](analysis.json)

Generated from complete benchmark rows by make-report.mjs. Frozen A/B/C use the same executable, pilot assets, cameras, supersampling and eight-proposal budget. Source and adjacent coverage-area limits are checked separately from pixel distance. Scores across area limits describe the quality–triangle tradeoff; within frozen A/B/C, only B versus C is an algorithm comparison. Post-hoc E uses a later executable and can add bounded proposals; its same-binary B controls are reported separately.

The archived [Round 4 v1 tree result](../round4/REPORT.md) motivated this experiment; its score is not a v3 baseline. The fresh v3 A/B/C runs below provide the measured comparison.

| Variant | Complete | SCORE | Final retained, category mean | Final source area max | Final/LOD0 tree tris | Fallbacks | Failed | Generation s | Process RSS high-water MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| area-v3-uncapped | 8/8 | 79.56 | 1.58% | 87.16% | 2/17978 | 0 | 0 | 224.14 | 671.20 |
| area-v3-cap-0.5 | 8/8 | 79.50 | 2.04% | 42.73% | 18/17978 | 0 | 0 | 225.67 | 671.20 |
| area-v3-cap-0.5-adaptive | 8/8 | 79.53 | 2.27% | 42.73% | 18/17978 | 0 | 0 | 291.34 | 671.20 |

## Per-asset measurements

| Variant / asset | Tris source→final | Final retained | Worst source area | Worst adjacent area | Resident KiB | Generation s | Process RSS high-water MiB | State |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| area-v3-uncapped / ph_dead_quiver_trunk | 17978→2 | 0.01% | 87.16% (LOD 7, view 13) | 70.49% (LOD 7, view 5) | 703.69 | 41.78 | 671.20 | pass |
| area-v3-uncapped / ph_grass_bermuda_01 | 941→30 | 3.19% | 57.06% (LOD 7, view 14) | 31.19% (LOD 7, view 14) | 82.21 | 4.58 | 671.20 | pass |
| area-v3-uncapped / ph_metal_stool_02 | 6532→124 | 1.90% | 31.75% (LOD 6, view 0) | 28.38% (LOD 6, view 0) | 497.51 | 25.20 | 671.20 | pass |
| area-v3-uncapped / ph_moon_rock_02 | 3304→18 | 0.54% | 11.54% (LOD 7, view 9) | 9.66% (LOD 7, view 4) | 372.65 | 22.99 | 671.20 | pass |
| area-v3-uncapped / ph_painted_wooden_shelves | 524→26 | 4.96% | 35.29% (LOD 7, view 12) | 34.02% (LOD 7, view 12) | 97.67 | 11.77 | 671.20 | pass |
| area-v3-uncapped / ph_rock_face_02 | 29566→51 | 0.17% | 51.40% (LOD 7, view 10) | 45.37% (LOD 7, view 10) | 3269.52 | 19.83 | 671.20 | pass |
| area-v3-uncapped / si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652 | 99586→1273 | 1.28% | 6.83% (LOD 6, view 8) | 6.99% (LOD 6, view 8) | 13893.73 | 34.43 | 671.20 | pass |
| area-v3-uncapped / si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7 | 150000→844 | 0.56% | 18.16% (LOD 6, view 9) | 15.37% (LOD 6, view 9) | 8348.50 | 63.57 | 671.20 | pass |
| area-v3-cap-0.5 / ph_dead_quiver_trunk | 17978→18 | 0.10% | 42.73% (LOD 7, view 13) | 18.57% (LOD 6, view 13) | 708.33 | 40.51 | 671.20 | pass |
| area-v3-cap-0.5 / ph_grass_bermuda_01 | 941→61 | 6.48% | 46.92% (LOD 6, view 6) | 23.87% (LOD 5, view 6) | 79.80 | 4.53 | 671.20 | pass |
| area-v3-cap-0.5 / ph_metal_stool_02 | 6532→124 | 1.90% | 31.75% (LOD 6, view 0) | 28.38% (LOD 6, view 0) | 497.51 | 27.72 | 671.20 | pass |
| area-v3-cap-0.5 / ph_moon_rock_02 | 3304→18 | 0.54% | 11.54% (LOD 7, view 9) | 9.66% (LOD 7, view 4) | 372.65 | 22.81 | 671.20 | pass |
| area-v3-cap-0.5 / ph_painted_wooden_shelves | 524→26 | 4.96% | 35.29% (LOD 7, view 12) | 34.02% (LOD 7, view 12) | 97.67 | 11.84 | 671.20 | pass |
| area-v3-cap-0.5 / ph_rock_face_02 | 29566→149 | 0.50% | 23.81% (LOD 7, view 1) | 12.37% (LOD 7, view 1) | 3279.58 | 19.84 | 671.20 | pass |
| area-v3-cap-0.5 / si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652 | 99586→1273 | 1.28% | 6.83% (LOD 6, view 8) | 6.99% (LOD 6, view 8) | 13893.73 | 35.67 | 671.20 | pass |
| area-v3-cap-0.5 / si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7 | 150000→844 | 0.56% | 18.16% (LOD 6, view 9) | 15.37% (LOD 6, view 9) | 8348.50 | 62.75 | 671.20 | pass |
| area-v3-cap-0.5-adaptive / ph_dead_quiver_trunk | 17978→18 | 0.10% | 43.12% (LOD 6, view 13) | 26.83% (LOD 6, view 13) | 704.86 | 39.77 | 671.20 | pass |
| area-v3-cap-0.5-adaptive / ph_grass_bermuda_01 | 941→69 | 7.33% | 47.57% (LOD 5, view 13) | 24.56% (LOD 5, view 13) | 61.82 | 8.52 | 671.20 | pass |
| area-v3-cap-0.5-adaptive / ph_metal_stool_02 | 6532→110 | 1.68% | 36.23% (LOD 6, view 11) | 35.36% (LOD 6, view 11) | 634.83 | 31.02 | 671.20 | pass |
| area-v3-cap-0.5-adaptive / ph_moon_rock_02 | 3304→12 | 0.36% | 19.22% (LOD 7, view 13) | 17.59% (LOD 7, view 8) | 340.53 | 41.47 | 671.20 | pass |
| area-v3-cap-0.5-adaptive / ph_painted_wooden_shelves | 524→34 | 6.49% | 11.88% (LOD 7, view 0) | 11.99% (LOD 7, view 10) | 83.78 | 24.69 | 671.20 | pass |
| area-v3-cap-0.5-adaptive / ph_rock_face_02 | 29566→99 | 0.33% | 28.22% (LOD 7, view 1) | 17.59% (LOD 7, view 1) | 2341.84 | 31.65 | 671.20 | pass |
| area-v3-cap-0.5-adaptive / si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652 | 99586→1258 | 1.26% | 7.54% (LOD 6, view 8) | 5.49% (LOD 6, view 1) | 28640.70 | 39.72 | 671.20 | pass |
| area-v3-cap-0.5-adaptive / si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7 | 150000→848 | 0.57% | 15.63% (LOD 6, view 9) | 10.43% (LOD 5, view 7) | 8265.35 | 74.50 | 671.20 | pass |

## Pilot decision

- Area preset pilot qualification: pass. SCORE cost 0.07 points; new unreduced assets: none; tree final retention: 0.10%.
- Adaptive pilot qualification at the 0.5 limit: fail or incomplete. SCORE gain 0.03 points.
- Adaptive Bell X-1 chain retention changed from 4.75% to 20.98%; adaptive generation took 65.66 more seconds across the pilot.

## Post-hoc topology-relaxed development pilot

This objective-changing test followed the frozen A/B/C comparison. It uses the same eight development assets, 0.5 area limit, cameras, sampling, work budget, executable and protocol as B. It is exploratory; its topology behavior and generalization need separate review. The held-out split was not used. [Raw D summary](../runs/area-v3-cap-0.5-topology-relaxed/summary.json).

D completed 8/8 assets, with 0 failures and 0 fallbacks. SCORE 79.05 versus B 79.50 (change -0.44 points); generation time 220.14 s versus B 225.67 s. All configured audits pass: yes. Category-balanced final retained ratio is 2.01% versus B 2.04%. The aggregate SCORE regression rules out a global objective switch on this pilot.

| Asset | Final tris B→D | Chain retained B→D | Final source area B→D | Resident KiB B→D | Generation s B→D | State D |
|---|---:|---:|---:|---:|---:|---|
| ph_dead_quiver_trunk | 18→18 | 1.69%→1.69% | 42.73%→47.27% | 708.33→709.98 | 40.51→37.52 | pass |
| ph_grass_bermuda_01 | 61→61 | 43.45%→46.24% | 40.26%→40.68% | 79.80→79.48 | 4.53→4.05 | pass |
| ph_metal_stool_02 | 124→247 | 9.20%→9.74% | 26.47%→10.99% | 497.51→485.62 | 27.72→24.09 | pass |
| ph_moon_rock_02 | 18→18 | 31.31%→33.63% | 11.54%→14.48% | 372.65→408.40 | 22.81→21.84 | pass |
| ph_painted_wooden_shelves | 26→26 | 37.98%→37.98% | 35.29%→11.98% | 97.67→97.73 | 11.84→12.12 | pass |
| ph_rock_face_02 | 149→29 | 31.30%→31.24% | 23.81%→44.44% | 3279.58→3273.33 | 19.84→20.10 | pass |
| si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652 | 1273→64 | 4.75%→4.23% | 4.87%→25.37% | 13893.73→13922.14 | 35.67→35.81 | pass |
| si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7 | 844→48 | 4.36%→2.84% | 16.84%→40.51% | 8348.50→8320.59 | 62.75→64.61 | pass |

Independent 642+64-view rotated dense tail checks at 8× with refinement to 32×:

| D asset / raw audit | Final tris | LOD7 source area / cap | LOD7 adjacent area / cap | Six tail gates |
|---|---:|---:|---:|---|
| [Apollo hatch](dense-hatch-topology-relaxed.json) | 48 | 44.58% / 50.00% | 20.34% / 50.00% | pass |
| [Bell X-1](dense-bell-topology-relaxed.json) | 64 | 30.50% / 50.00% | 23.89% / 50.00% | pass |
| [Rock face](dense-rock-topology-relaxed.json) | 29 | 50.71% / 50.00% | 50.98% / 50.00% | FAIL |
| [Quiver tree](dense-tree-topology-relaxed.json) | 18 | 51.41% / 50.00% | 26.83% / 50.00% | FAIL |

The rock face fails LOD7 source and adjacent area at 50.71% and 50.98%; both pixel-distance gates pass.

The quiver tree fails LOD7 source area at 51.41% (view 580); its pixel-distance gate passes.

These object-specific checks support hatch and Bell X-1 headroom, but do not justify global topology relaxation. D run SHA-256: 22e61fb0b680d49e4390ff9129d448b65deede59fda6ebdf4297fcee126b9ef8; D generator binary SHA-256: 0455910e11a2b986090d74807f7a3fd3943d19be4511c9aabf11ba792c2563ff; dense auditor binary SHA-256: e1d2cf2ba368f180acc93ca65d8e315b6739c7c1fc75359e8ca1de2362fad151.

## Post-hoc conditional topology fallback E

E followed the frozen A/B/C pilot and the broader topology-relaxed D probe. It keeps B’s quadric objective and 0.5 area limit, but tries at most one topology-relaxed proposal per LOD when a quadric reduction stops more than four times above its requested triangles with link-condition rejections. Those proposals are extra work beyond the nominal eight-proposal budget. The archived [post-hoc B pilot control](../runs/area-v3-cap-0.5-fallback-v2-b/summary.json) uses E’s corrected executable; 8/8 output and attribute hashes match frozen B. The same-binary validation follow-up appears below. [E config](configs/cap-0.5-topology-fallback.json) · [corrected v2 E pilot](../runs/area-v3-cap-0.5-fallback-v2-e/summary.json).

This is a bounded extra-work comparison, not evidence of superiority at equal reducer work. E completed 8/8 development assets, with 0 failures and 0 fallbacks. SCORE 79.5515 versus same-binary B control 79.4959 (change 0.0556 points). All configured audits pass: yes. It made 5 extra topology proposals; total candidate evaluations were 448 in B and 453 in E. 6/8 output and attribute hashes match B, with 8/8 canonical input attribute hashes matching. Category-balanced final retention is 1.84% versus B 2.04%. Generation time is 257.29 s versus B 257.14 s; process RSS high-water is 671.20 MiB versus B 671.20 MiB.

| Changed development asset | Final tris B→E | Chain retained B→E | Final source area B→E | Extra proposals E |
|---|---:|---:|---:|---:|
| si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652 | 1273→128 | 4.75%→4.44% | 4.87%→12.58% | 3 |
| si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7 | 844→96 | 4.36%→4.23% | 16.84%→32.18% | 2 |

Independent 642+64-view rotated dense tail checks at 8× with refinement to 32×:

| E asset / raw audit | Final tris | LOD7 source area / cap | LOD7 adjacent area / cap | Six tail gates |
|---|---:|---:|---:|---|
| [Apollo hatch](dense-hatch-topology-fallback-v2.json) | 96 | 34.54% / 50.00% | 20.62% / 50.00% | pass |
| [Bell X-1](dense-bell-topology-fallback-v2.json) | 128 | 19.05% / 50.00% | 14.16% / 50.00% | pass |
| [Quiver tree](dense-tree-topology-fallback-v2.json) | 18 | 51.29% / 50.00% | 25.31% / 50.00% | FAIL |

The corrected v2 B and E outputs reproduce their prior eight-asset exports: 8/8 B and 8/8 E output and attribute hashes match. The pre-fix E run is [historical evidence](../runs/area-v3-cap-0.5-topology-fallback/summary.json), not the scored comparison. Corrected E’s tree `chain.bin` and `chain.gltf` are byte-identical to frozen B’s: yes. The direct corrected E rotated tree audit fails LOD7 source area at 51.29%, matching frozen B’s rotated tree result. Passing corrected hatch and Bell checks does not resolve this tree miss. The fallback relaxes the link-condition topology restriction for selected proposals; it does not guarantee manifold topology or exclude new intersections. Keep E opt-in and inspect each chosen export. E pilot run SHA-256: 069698caa8681a30457597b865278727dd7ac76fa234edf0459a7ba5128ccb82; E generator binary SHA-256: 09a7b32e8701393f933be374e2cafcb86b63df10de50d93eda7a7ad3390ab1c9; E dense auditor binary SHA-256: 31bd34259eb45cfd05bf1338b614fda45adbf440c508807f94649f19a5c3a98a.

### Same-binary E validation follow-up

This post-hoc 20-asset comparison uses the corrected v2 executable, validation manifest, cameras and 0.5 cap for its B control and E run. The only setting change is `research.topology_fallback`, which permits at most one extra reducer call per LOD. It is a bounded extra-work opt-in comparison, not equal-work optimizer superiority, and is separate from the frozen A/B validation. [B control](../runs/area-v3-validation-cap-0.5-fallback-v2-b/summary.json) · [E validation](../runs/area-v3-validation-cap-0.5-fallback-v2-e/summary.json).

| Variant | Complete | SCORE | Final retained, category mean | Worst final source area | Fallbacks | Failed | Configured audits pass | Generation s | Process RSS high-water MiB |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| area-v3-validation-cap-0.5-fallback-v2-b | 20/20 | 86.0609 | 1.54% | 46.49% | 0 | 0 | yes | 1428.88 | 774.67 |
| area-v3-validation-cap-0.5-fallback-v2-e | 20/20 | 86.0671 | 1.50% | 46.49% | 0 | 0 | yes | 1436.52 | 770.30 |

E versus same-binary B control: 0.0062 SCORE points, 18 extra topology proposals across 7 assets and 1120→1138 total candidate evaluations. 19/20 output and attribute hashes remain identical; 20/20 canonical input attribute hashes match. The new B control reproduces 20/20 output and attribute hashes from the earlier capped validation binary. Both variants pass all 140 source and 140 adjacent configured LOD audits.

| Changed validation asset | Final tris control→E | Chain retained control→E | Final source area control→E | Extra proposals E |
|---|---:|---:|---:|---:|
| si_3d_package_789cf90a-4387-4ac1-9e96-c7d6a7b9d26f | 894→112 | 2.10%→2.02% | 39.67%→31.41% | 4 |

Independent rotated dense tail audits of changed E validation exports:

| Asset / raw audit | Final tris | LOD7 source area / cap | LOD7 adjacent area / cap | Source px / limit | Adjacent px / limit | Six tail gates |
|---|---:|---:|---:|---:|---:|---|
| [si_3d_package_789cf90a-4387-4ac1-9e96-c7d6a7b9d26f](dense-validation-si_3d_package_789cf90a-4387-4ac1-9e96-c7d6a7b9d26f-topology-fallback-v2.json) | 112 | 33.08% / 50.00% | 27.51% / 50.00% | 3.69 / 6.89 | 2.93 / 3.00 | pass |

Changed-export dense checks: 1/1 pass.

This measured follow-up does not promote the original 8 px/variable-transition E scenario globally: its unchanged tree still fails the independent rotated dense area check.

## Strict pixel-contract scenarios

This user-directed development follow-up uses the corrected v2 executable, frozen eight assets, cameras, 0.5 changed-area cap, eight-proposal nominal budget and hybrid rebuild chain. It fixes the adjacent/progressive pixel limit at 2 px, then evaluates source caps of 3 px and 4 px. Each source cap has a same-binary quadric B control and conditional-fallback E run. E may add one reducer call per LOD. SCORE comparisons below are only B versus E within the same pixel contract; scores from different source caps are not pooled or ranked. [Separate all-asset strict board](strict-board/index.html).

All four configs force research rebuild and triangle-first chain selection. Metadata retains the default `triangle_overhead_bps=500`, but this research selection path does not apply a 5% triangle allowance.

### Source cap 3 px; adjacent cap 2 px

Actual LOD1–7 source limits: 2.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00 px. Adjacent limits: 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00 px. [B config](configs/strict-source-3-progressive-2-cap-0.5-b.json) · [E config](configs/strict-source-3-progressive-2-cap-0.5-e.json).

| Run | Complete | SCORE | Category mean final retention | Worst source area, any LOD | Worst adjacent area, any LOD | Fallbacks | Failed | Configured audits | Generation s | RSS high-water MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| [area-v3-strict-source-3-progressive-2-cap-0.5-b](../runs/area-v3-strict-source-3-progressive-2-cap-0.5-b/summary.json) | 8/8 | 76.2777 | 3.13% | 34.23% | 37.56% | 0 | 0 | pass | 239.68 | 671.20 |
| [area-v3-strict-source-3-progressive-2-cap-0.5-e](../runs/area-v3-strict-source-3-progressive-2-cap-0.5-e/summary.json) | 8/8 | 76.3073 | 2.92% | 34.23% | 37.56% | 0 | 0 | pass | 241.07 | 671.20 |

Within this 3 px contract, E minus B SCORE is 0.0296 points with 5 extra topology proposals. 2/8 output/attribute hash pairs change. Both configured audits pass: yes.

| Asset | Final tris B→E | Chain retained B→E | Final source area B→E | Source/adjacent audit rejects B→E | Extra E calls | State B/E |
|---|---:|---:|---:|---:|---:|---|
| ph_dead_quiver_trunk | 34→34 | 1.88%→1.88% | 34.23%→34.23% | 2/2→2/2 | 0 | pass/pass |
| ph_grass_bermuda_01 | 95→95 | 63.06%→63.06% | 29.94%→29.94% | 1/1→1/1 | 0 | pass/pass |
| ph_metal_stool_02 | 124→124 | 9.21%→9.21% | 26.47%→26.47% | 0/0→0/0 | 0 | pass/pass |
| ph_moon_rock_02 | 18→18 | 33.63%→33.63% | 11.70%→11.70% | 1/2→1/2 | 0 | pass/pass |
| ph_painted_wooden_shelves | 52→52 | 41.58%→41.58% | 8.74%→8.74% | 4/5→4/5 | 0 | pass/pass |
| ph_rock_face_02 | 149→149 | 31.30%→31.30% | 23.81%→23.81% | 0/0→0/0 | 0 | pass/pass |
| si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652 | 1274→128 | 4.75%→4.58% | 6.96%→12.58% | 0/1→0/1 | 3 | pass/pass |
| si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7 | 856→96 | 4.37%→4.30% | 16.45%→32.18% | 0/3→0/3 | 2 | pass/pass |

### Source cap 4 px; adjacent cap 2 px

Actual LOD1–7 source limits: 2.00, 3.22, 3.96, 4.00, 4.00, 4.00, 4.00 px. Adjacent limits: 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00 px. [B config](configs/strict-source-4-progressive-2-cap-0.5-b.json) · [E config](configs/strict-source-4-progressive-2-cap-0.5-e.json).

| Run | Complete | SCORE | Category mean final retention | Worst source area, any LOD | Worst adjacent area, any LOD | Fallbacks | Failed | Configured audits | Generation s | RSS high-water MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| [area-v3-strict-source-4-progressive-2-cap-0.5-b](../runs/area-v3-strict-source-4-progressive-2-cap-0.5-b/summary.json) | 8/8 | 76.2777 | 3.13% | 34.23% | 37.56% | 0 | 0 | pass | 240.50 | 671.20 |
| [area-v3-strict-source-4-progressive-2-cap-0.5-e](../runs/area-v3-strict-source-4-progressive-2-cap-0.5-e/summary.json) | 8/8 | 76.3073 | 2.92% | 34.23% | 37.56% | 0 | 0 | pass | 241.78 | 671.20 |

Within this 4 px contract, E minus B SCORE is 0.0296 points with 5 extra topology proposals. 2/8 output/attribute hash pairs change. Both configured audits pass: yes.

| Asset | Final tris B→E | Chain retained B→E | Final source area B→E | Source/adjacent audit rejects B→E | Extra E calls | State B/E |
|---|---:|---:|---:|---:|---:|---|
| ph_dead_quiver_trunk | 34→34 | 1.88%→1.88% | 34.23%→34.23% | 2/2→2/2 | 0 | pass/pass |
| ph_grass_bermuda_01 | 95→95 | 63.06%→63.06% | 29.94%→29.94% | 1/1→1/1 | 0 | pass/pass |
| ph_metal_stool_02 | 124→124 | 9.21%→9.21% | 26.47%→26.47% | 0/0→0/0 | 0 | pass/pass |
| ph_moon_rock_02 | 18→18 | 33.63%→33.63% | 11.70%→11.70% | 1/2→1/2 | 0 | pass/pass |
| ph_painted_wooden_shelves | 52→52 | 41.58%→41.58% | 8.74%→8.74% | 0/9→0/9 | 0 | pass/pass |
| ph_rock_face_02 | 149→149 | 31.30%→31.30% | 23.81%→23.81% | 0/0→0/0 | 0 | pass/pass |
| si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652 | 1274→128 | 4.75%→4.58% | 6.96%→12.58% | 0/1→0/1 | 3 | pass/pass |
| si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7 | 856→96 | 4.37%→4.30% | 16.45%→32.18% | 0/3→0/3 | 2 | pass/pass |

### Independent rotated strict-tail checks

Selected 3 px E exports were checked on 642+64 cameras at 8× with refinement to 32×, seed 3665436710. This is one independent rotation; passing it does not prove an all-view bound. Each cell below is the tightest margin across LOD5–7, with its LOD.

| Asset / raw audit | Final tris | Source area / cap | Adjacent area / cap | Source px / limit | Adjacent px / limit | Six tail gates |
|---|---:|---:|---:|---:|---:|---|
| [Quiver tree](dense-strict-source-3-progressive-2-cap-0.5-tree-e.json) | 34 | 40.48% / 50.00% (LOD 7) | 22.73% / 50.00% (LOD 7) | 1.61 / 3.00 (LOD 6) | 1.50600 / 2.00 (LOD 6) | pass |
| [Bermuda grass](dense-strict-source-3-progressive-2-cap-0.5-grass-e.json) | 95 | 40.13% / 50.00% (LOD 7) | 23.28% / 50.00% (LOD 7) | 1.10 / 3.00 (LOD 5) | 1.10355 / 2.00 (LOD 5) | pass |
| [Metal stool](dense-strict-source-3-progressive-2-cap-0.5-stool-e.json) | 124 | 39.07% / 50.00% (LOD 6) | 38.73% / 50.00% (LOD 6) | 1.86 / 3.00 (LOD 6) | 1.97855 / 2.00 (LOD 6) | pass |
| [Moon rock](dense-strict-source-3-progressive-2-cap-0.5-moon-rock-e.json) | 18 | 13.54% / 50.00% (LOD 7) | 12.39% / 50.00% (LOD 7) | 2.10 / 3.00 (LOD 5) | 1.99767 / 2.00 (LOD 5) | pass |
| [Painted shelves](dense-strict-source-3-progressive-2-cap-0.5-shelves-e.json) | 52 | 10.07% / 50.00% (LOD 6) | 6.16% / 50.00% (LOD 5) | 2.48 / 3.00 (LOD 6) | 1.97855 / 2.00 (LOD 6) | pass |
| [Rock face](dense-strict-source-3-progressive-2-cap-0.5-rock-face-e.json) | 149 | 26.76% / 50.00% (LOD 7) | 14.05% / 50.00% (LOD 7) | 1.86 / 3.00 (LOD 6) | 1.85355 / 2.00 (LOD 6) | pass |
| [Bell X-1](dense-strict-source-3-progressive-2-cap-0.5-bell-e.json) | 128 | 19.05% / 50.00% (LOD 7) | 16.42% / 50.00% (LOD 7) | 2.36 / 3.00 (LOD 6) | 1.99767 / 2.00 (LOD 6) | pass |
| [Apollo hatch](dense-strict-source-3-progressive-2-cap-0.5-hatch-e.json) | 96 | 34.54% / 50.00% (LOD 7) | 29.67% / 50.00% (LOD 7) | 2.09 / 3.00 (LOD 7) | 1.97855 / 2.00 (LOD 7) | pass |

Bell X-1’s tightest adjacent-pixel margin is 0.00233 px at LOD 6. A different rotation could still expose a failure.

Additional direct checks on B and 4 px E exports, plus the second Bell rotation:

| Contract / asset / raw audit | Final tris | Source px / limit | Adjacent px / limit | Six tail gates |
|---|---:|---:|---:|---|
| [3 px B Bell X-1](dense-strict-source-3-progressive-2-cap-0.5-bell-b.json) | 1274 | 2.36 / 3.00 (LOD 6) | 1.99767 / 2.00 (LOD 6) | pass |
| [3 px B Apollo hatch](dense-strict-source-3-progressive-2-cap-0.5-hatch-b.json) | 856 | 1.86 / 3.00 (LOD 6) | 1.72855 / 2.00 (LOD 6) | pass |
| [4 px B Bell X-1](dense-strict-source-4-progressive-2-cap-0.5-bell-b.json) | 1274 | 2.36 / 4.00 (LOD 6) | 1.99767 / 2.00 (LOD 6) | pass |
| [4 px B Apollo hatch](dense-strict-source-4-progressive-2-cap-0.5-hatch-b.json) | 856 | 1.86 / 4.00 (LOD 6) | 1.72855 / 2.00 (LOD 6) | pass |
| [4 px E Quiver tree](dense-strict-source-4-progressive-2-cap-0.5-tree-e.json) | 34 | 1.61 / 4.00 (LOD 6) | 1.50600 / 2.00 (LOD 6) | pass |
| [4 px E Bermuda grass](dense-strict-source-4-progressive-2-cap-0.5-grass-e.json) | 95 | 1.10 / 4.00 (LOD 5) | 1.10355 / 2.00 (LOD 5) | pass |
| [4 px E Metal stool](dense-strict-source-4-progressive-2-cap-0.5-stool-e.json) | 124 | 1.86 / 4.00 (LOD 6) | 1.97855 / 2.00 (LOD 6) | pass |
| [4 px E Moon rock](dense-strict-source-4-progressive-2-cap-0.5-moon-rock-e.json) | 18 | 2.10 / 4.00 (LOD 5) | 1.99767 / 2.00 (LOD 5) | pass |
| [4 px E Painted shelves](dense-strict-source-4-progressive-2-cap-0.5-shelves-e.json) | 52 | 2.48 / 4.00 (LOD 6) | 1.97855 / 2.00 (LOD 6) | pass |
| [4 px E Rock face](dense-strict-source-4-progressive-2-cap-0.5-rock-face-e.json) | 149 | 1.86 / 4.00 (LOD 6) | 1.85355 / 2.00 (LOD 6) | pass |
| [4 px E Bell X-1](dense-strict-source-4-progressive-2-cap-0.5-bell-e.json) | 128 | 2.36 / 4.00 (LOD 6) | 1.99767 / 2.00 (LOD 6) | pass |
| [4 px E Apollo hatch](dense-strict-source-4-progressive-2-cap-0.5-hatch-e.json) | 96 | 2.09 / 4.00 (LOD 7) | 1.97855 / 2.00 (LOD 7) | pass |
| [3 px E Bell X-1, second seed](dense-strict-source-3-progressive-2-cap-0.5-bell-e-seed-2027.json) | 128 | 2.36 / 3.00 (LOD 6) | 1.99767 / 2.00 (LOD 6) | pass |
| [3 px E Moon rock, second seed](dense-strict-source-3-progressive-2-cap-0.5-moon-rock-e-seed-2027.json) | 18 | 2.10 / 3.00 (LOD 5) | 1.97855 / 2.00 (LOD 5) | pass |

The second Bell E seed 3665436711 passes all six tail gates. Its smallest adjacent-pixel margin is 0.00233 px. Two finite rotations still do not establish all-view robustness.

The second Moon rock E seed 3665436711 passes all six tail gates; its smallest adjacent-pixel margin is 0.02145 px.

First rotated-seed coverage: direct 3 px E 8/8 pass, direct 4 px E 8/8 pass; 3 px B 8/8 pass and 4 px B 8/8 pass using six byte-identical B/E exports plus direct Bell and hatch audits per contract. Each asset has six LOD5–7 source/adjacent gates.

The 4 px E exports are byte-identical to their 3 px counterparts and have no tighter source limit; their adjacent and area limits are unchanged. Their source/adjacent measurement objects are identical for 8/8 directly audited assets; only the source limits differ. These are finite sampled camera sets, not all-view guarantees.

The 3 px tree B and E exports are byte-identical at 34 final triangles, so topology fallback did not cause this tree change. Frozen 8 px/variable-transition B has 18 final triangles and fails the rotated dense source-area cap at 51.29%. The strict 3 px tree passes that first rotation at 40.48% source-area change. The repair on this tree comes from the tighter pixel schedule; it is an asset-specific result.

Painted shelves show a concrete limit under the tighter contract. The four strict variants finish at 52/52/52/52 triangles (3B/3E/4B/4E), whereas frozen 8 px/variable-transition B finishes at 26 under a different pixel contract. For 3 px B, LOD6 is 52 triangles with source error 2.72/3.00 px and adjacent error 1.98/2.00 px; LOD7 retains 52 triangles.

| Shelves variant | Final tris | Source-audit rejects | Adjacent-audit rejects | Area-only audit rejects source/adjacent |
|---|---:|---:|---:|---:|
| 3 px B | 52 | 4 | 5 | 0/0 |
| 3 px E | 52 | 4 | 5 | 0/0 |
| 4 px B | 52 | 0 | 9 | 0/0 |
| 4 px E | 52 | 0 | 9 | 0/0 |

Selected B proposal rejections when the source cap changes from 3 to 4 px:

| Asset | Source-search rejects 3→4 | Adjacent-search rejects 3→4 | Area-only source-audit rejects 3/4 |
|---|---:|---:|---:|
| ph_dead_quiver_trunk | 14→8 | 5→11 | 2/2 |
| ph_grass_bermuda_01 | 30→26 | 5→9 | 1/1 |
| si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7 | 11→7 | 2→6 | 0/0 |

Across the different 3 px and 4 px source schedules, 8/8 B and 8/8 E output and attribute hashes are identical; 8/8 B and 8/8 E `chain.bin`/`chain.gltf` exports are byte-identical. This is a measured property of these eight assets, not a claim that the contracts are interchangeable. The shelves counts show adjacent pixel pressure on that object. Tree and grass each have area-only source-audit rejections in both contracts, so the 50% area gate also constrains proposals even though their selected meshes remain below 50%. Rejection counts do not prove a global triangle optimum. The strict runs are separate pixel contracts from the frozen A/B/C and v2 E comparisons.

## Where triangle headroom remains

SCORE averages triangle retention across scheduled LOD1–7 for each asset, then balances categories; final-LOD triangles alone do not determine it. On painted shelves, cap-only B ends at 26 triangles with 1,393 triangles across LOD1–7. Adaptive C ends at 34 triangles but totals 1,110, so C improves that asset’s chain ratio despite a denser final mesh. The [C trace](trace-extracts/shelves-adaptive/rows/ph_painted_wooden_shelves.json) shows a progressive proposal from a 46-triangle parent requesting 23 and achieving 24, rejected at adjacent search; C retains 34. The [B trace](trace-extracts/shelves-fixed/rows/ph_painted_wooden_shelves.json) shows a progressive 52→26 proposal accepted.

For Apollo hatch, the [QEM trace](trace-extracts/hatch-quadric/rows/si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7.json) requests 85 and 428 triangles directly from the 150,000-triangle source, but both stop at 856 after 26,673 link-condition rejections; the selected progressive final is 844. Post-hoc topology-relaxed D reaches 48 and passes its rotated dense hatch tail check, showing that topology constraints leave substantial object-specific headroom. D’s global pilot SCORE regression and rock/tree dense misses prevent a global switch. [Trace provenance and reproduction](trace-extracts/README.md).

The conditional E fallback preserves B’s quadric path unless a large link-condition shortfall triggers one extra topology-relaxed proposal. It reaches 96 hatch triangles and 128 Bell X-1 triangles with five extra proposals across the corrected development pilot. Both selected exports pass their corrected rotated dense tail checks. The original 8 px/variable-transition tree export remains identical to B and retains its independent dense failure. E therefore demonstrates useful bounded, object-specific headroom without a validated global quality preset.

## Pre-change compatibility replay

The archived pre-change uncapped executable and fresh v3 uncapped executable produce identical positions/indices, all output attributes and canonical input attributes on 8/8 pilot assets: yes. Old binary SHA-256: fd305b12a98b70a9b0d2c1a973cd0cb873566507023adb0a4b4217ef90e593ba.

After a JSON-manifest field was added, the final rebuilt binary replayed the capped tree with identical positions/indices, output attributes and canonical input attributes: yes. Its reported area limit is 0.5; binary SHA-256: ab47675a1efd2f20cbe795bf0dbc744ad28f177606ef91ef516d5edfb44e733e. Frozen A/B validation uses this final rebuilt binary.

## Independent rotated dense tree audit

[Rotated raw audit](dense-tree-cap-0.5.json) · [Default-seed raw audit](dense-tree-default-seed-cap-0.5.json) · auditor binary SHA-256: e1d2cf2ba368f180acc93ca65d8e315b6739c7c1fc75359e8ca1de2362fad151.

The last three capped tree LODs were checked against source and predecessor with 642+64 cameras, seed 3665436710, 8× sampling and refinement to 32×. This is a separate camera/sampling contract and does not rewrite pilot SCORE. Overall: FAIL.

| LOD | Tris | Source area / cap | Adjacent area / cap | Source px / limit | Adjacent px / limit | Result |
|---:|---:|---:|---:|---:|---:|---|
| 5 | 68 | 28.76% / 50.00% | 20.16% / 50.00% | 2.35 / 5.82 | 2.49 / 2.67 | pass |
| 6 | 34 | 42.74% / 50.00% | 24.78% / 50.00% | 2.00 / 6.38 | 1.26 / 2.83 | pass |
| 7 | 18 | 51.29% / 50.00% | 25.31% / 50.00% | 1.51 / 6.89 | 0.98 / 3.00 | FAIL |

A second dense check with the original audit rotation seed 2971082790 and identical camera count and sampling passes all six gates; its LOD7 source area maximum is 49.80%. The rotated check changes only the seed and exposes a failing view.

The first failure is LOD 7 source area 51.29% at view 21 after 22 views; its pixel-distance gate passes. The 0.5 preset is not ready for promotion under this independent audit. Finite-camera acceptance does not establish all-view robustness.

## Exploratory cap check (unscored)

A one-tree development follow-up tested stricter caps without changing the frozen A/B/C comparison. At 0.45, the reducer selected the same 18-triangle output hash as the 0.5 cap, so its rotated dense miss persists. At 0.4, it selected a different 18-triangle mesh with 39.09% configured final source area, but its [rotated dense audit](dense-tree-dev-cap-0.4-rotated.json) failed at 44.10% against the 40% cap (LOD7 source view 42). These single-asset probes have no SCORE and do not select a new preset.

## Validation

| Variant | Complete | SCORE | Final retained, category mean | Worst final source area | Worst source area at any LOD | Fallbacks | Failed | All delivered audits pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| area-v3-validation-uncapped | 20/20 | 86.20 | 1.44% | 76.17% | 76.17% | 0 | 0 | yes |
| area-v3-validation-cap-0.5 | 20/20 | 86.06 | 1.54% | 46.49% | 46.49% | 0 | 0 | yes |

Cap validation quality cost: 0.14 SCORE points; qualification: pass.

Final LODs above 50% source-area change: 3/20 uncapped versus 0/20 capped. This verifies improvement on the frozen validation split under the configured cameras.

The uncapped run has 8 over-cap source/adjacent LOD pairs across 3 assets; capped B repairs them. Its worst adjacent area is 41.72%. Area alone rejected 19 source-audit and 1 adjacent-audit proposals. 17/20 output and attribute hashes remain identical, 20/20 canonical input attribute hashes match, and no final source-area result worsened.

Changed validation exports:

| Asset | Final tris A→B | Final source area A→B |
|---|---:|---:|
| ph_combination_wrench | 16→20 | 53.37%→42.39% |
| ph_didelta_spinosa | 4224→15664 | 76.17%→42.62% |
| ph_wooden_ladder | 188→188 | 50.51%→42.55% |

## Decision

The 0.5 cap passes the configured pilot and 20-asset validation screens with a small SCORE cost, but the rotated dense tree check fails its area limit. Keep the 0.5 gate available as an experimental opt-in; do not label it an independently validated quality preset. Adaptive targets miss the pilot gain threshold. Topology relaxation improves selected objects but regresses aggregate chain SCORE and fails dense checks on other objects. Conditional topology fallback E reduces the hatch and Bell triangle counts, but the original 8 px/variable-transition E tree fails the rotated dense gate. Keep E opt-in. The separate strict 2 px progressive/3–4 px source scenarios pass their configured pilot audits and the sampled rotated tail checks, but have no held-out validation or all-view bound; retain them as research scenarios.

## Provenance

| Run | Binary SHA-256 | Source tree SHA-256 | Config SHA-256 | Protocol SHA-256 |
|---|---|---|---|---|
| area-v3-uncapped | 0455910e11a2b986090d74807f7a3fd3943d19be4511c9aabf11ba792c2563ff | 8dd0d91668f698ace70652dbf42a44505ecc762ffae1a8f0fcc36b543c6d5213 | 799093900df0171555746a83034b784b02bdd0345220bbc9fd1e1c93f8110bf3 | 71e7a71e508f400140f99fff10ba248c0d89cabe95e32329cf8a968f204dd42b |
| area-v3-cap-0.5 | 0455910e11a2b986090d74807f7a3fd3943d19be4511c9aabf11ba792c2563ff | 8dd0d91668f698ace70652dbf42a44505ecc762ffae1a8f0fcc36b543c6d5213 | 52a0ff57d191075e20fbf7a2c975a2db86ebe6e0761c55da108fa9cc7f9b17fb | 71e7a71e508f400140f99fff10ba248c0d89cabe95e32329cf8a968f204dd42b |
| area-v3-cap-0.5-adaptive | 0455910e11a2b986090d74807f7a3fd3943d19be4511c9aabf11ba792c2563ff | 8dd0d91668f698ace70652dbf42a44505ecc762ffae1a8f0fcc36b543c6d5213 | 5d21c9c18febc4272c2c4eb348496feb1317f9bc5caa2c6805d6eb146fb41a9d | 71e7a71e508f400140f99fff10ba248c0d89cabe95e32329cf8a968f204dd42b |
| area-v3-cap-0.5-topology-relaxed | 0455910e11a2b986090d74807f7a3fd3943d19be4511c9aabf11ba792c2563ff | 8dd0d91668f698ace70652dbf42a44505ecc762ffae1a8f0fcc36b543c6d5213 | ff95ea499b66f93d0ea3f46eab7ed64919fd3a991b67ad6fc42b392823c9371b | 71e7a71e508f400140f99fff10ba248c0d89cabe95e32329cf8a968f204dd42b |
| area-v3-cap-0.5-fallback-v2-b | 09a7b32e8701393f933be374e2cafcb86b63df10de50d93eda7a7ad3390ab1c9 | 47b9a2fb854ce81e2d29f9bd5a31d8668fe555313f55cf791c54622783a2dd9b | 49120d72681e8db346e2c98f9428ba4a153e0ecf26cdb3b68af6a6f84302391c | 71e7a71e508f400140f99fff10ba248c0d89cabe95e32329cf8a968f204dd42b |
| area-v3-cap-0.5-fallback-v2-e | 09a7b32e8701393f933be374e2cafcb86b63df10de50d93eda7a7ad3390ab1c9 | 47b9a2fb854ce81e2d29f9bd5a31d8668fe555313f55cf791c54622783a2dd9b | 6f8ef1817265091dd8c59492c82b73d9a66c0ee484cf860898f0469f641b8bba | 71e7a71e508f400140f99fff10ba248c0d89cabe95e32329cf8a968f204dd42b |
| area-v3-validation-uncapped | ab47675a1efd2f20cbe795bf0dbc744ad28f177606ef91ef516d5edfb44e733e | 2d5765bc48922ce3ba50d6ba8ef0797be62b1ff8b84afee374de6782987e7300 | 799093900df0171555746a83034b784b02bdd0345220bbc9fd1e1c93f8110bf3 | 71e7a71e508f400140f99fff10ba248c0d89cabe95e32329cf8a968f204dd42b |
| area-v3-validation-cap-0.5 | ab47675a1efd2f20cbe795bf0dbc744ad28f177606ef91ef516d5edfb44e733e | 2d5765bc48922ce3ba50d6ba8ef0797be62b1ff8b84afee374de6782987e7300 | 52a0ff57d191075e20fbf7a2c975a2db86ebe6e0761c55da108fa9cc7f9b17fb | 71e7a71e508f400140f99fff10ba248c0d89cabe95e32329cf8a968f204dd42b |
| area-v3-validation-cap-0.5-fallback-v2-b | 09a7b32e8701393f933be374e2cafcb86b63df10de50d93eda7a7ad3390ab1c9 | 47b9a2fb854ce81e2d29f9bd5a31d8668fe555313f55cf791c54622783a2dd9b | 49120d72681e8db346e2c98f9428ba4a153e0ecf26cdb3b68af6a6f84302391c | 71e7a71e508f400140f99fff10ba248c0d89cabe95e32329cf8a968f204dd42b |
| area-v3-validation-cap-0.5-fallback-v2-e | 09a7b32e8701393f933be374e2cafcb86b63df10de50d93eda7a7ad3390ab1c9 | 47b9a2fb854ce81e2d29f9bd5a31d8668fe555313f55cf791c54622783a2dd9b | 6f8ef1817265091dd8c59492c82b73d9a66c0ee484cf860898f0469f641b8bba | 71e7a71e508f400140f99fff10ba248c0d89cabe95e32329cf8a968f204dd42b |
| area-v3-strict-source-3-progressive-2-cap-0.5-b | 09a7b32e8701393f933be374e2cafcb86b63df10de50d93eda7a7ad3390ab1c9 | 47b9a2fb854ce81e2d29f9bd5a31d8668fe555313f55cf791c54622783a2dd9b | 42e0a55aeadc29fa12a378f9e3d9becde4434439a412d92df2f321452671d3b4 | 71e7a71e508f400140f99fff10ba248c0d89cabe95e32329cf8a968f204dd42b |
| area-v3-strict-source-3-progressive-2-cap-0.5-e | 09a7b32e8701393f933be374e2cafcb86b63df10de50d93eda7a7ad3390ab1c9 | 47b9a2fb854ce81e2d29f9bd5a31d8668fe555313f55cf791c54622783a2dd9b | be95ce8f8f393529d96973cc0f9ed35df94eebc80990261bde4aa09a1fce5401 | 71e7a71e508f400140f99fff10ba248c0d89cabe95e32329cf8a968f204dd42b |
| area-v3-strict-source-4-progressive-2-cap-0.5-b | 09a7b32e8701393f933be374e2cafcb86b63df10de50d93eda7a7ad3390ab1c9 | 47b9a2fb854ce81e2d29f9bd5a31d8668fe555313f55cf791c54622783a2dd9b | dd0bd98377bd671fbf4f8365b7eacace2db0465d4d7e8fe762c255433b938cfa | 71e7a71e508f400140f99fff10ba248c0d89cabe95e32329cf8a968f204dd42b |
| area-v3-strict-source-4-progressive-2-cap-0.5-e | 09a7b32e8701393f933be374e2cafcb86b63df10de50d93eda7a7ad3390ab1c9 | 47b9a2fb854ce81e2d29f9bd5a31d8668fe555313f55cf791c54622783a2dd9b | 5ae9b0fd15b2a2a289b39d6ad4e5aca68b0524bd9cff8c4e1ece2ec13b424693 | 71e7a71e508f400140f99fff10ba248c0d89cabe95e32329cf8a968f204dd42b |

Run hashes, limits, worst views, and all measured asset summaries are in analysis.json; raw benchmark rows retain each LOD and rejection. Resident bytes include source vertices, added vertices and indices. Generation time includes search and audits but excludes import/export. Process RSS is a cumulative high-water mark, so its per-asset row is not an isolated memory cost. Timings include shared-workstation noise.
