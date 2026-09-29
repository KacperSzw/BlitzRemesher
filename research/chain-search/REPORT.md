# Whole-chain reduction measurements

Frozen eight-asset pilot. Primary: 512→16 px, eight levels, 3 px source cap, 2 px transitions, 50% changed area and 20% added vertex bytes. Appearance and strict coverage are separate contracts. Timings include generation and auditing on a shared workstation. Work tiers name proposal budgets, not measured speed.

| Split / preset / method | Complete | SCORE | Chain retained | Final retained | Last three retained | Failed / unchanged |
|---|---:|---:|---:|---:|---:|---:|
| development / coverage / [baseline](../runs/chain-search-coverage-baseline/summary.json) | 8/8 | 69.788 | 30.21% | 2.14% | 9.82% | 0 / 0 |
| development / coverage / [legacy-2](../runs/chain-search-coverage-legacy-2/summary.json) | 8/8 | 68.790 | 31.21% | 2.25% | 10.40% | 0 / 0 |
| development / coverage / [legacy-4](../runs/chain-search-coverage-legacy-4/summary.json) | 8/8 | 66.853 | 33.15% | 2.00% | 12.00% | 0 / 0 |
| development / coverage / [graph-2](../runs/chain-search-v3-coverage-graph-2/summary.json) | 8/8 | 73.270 | 26.73% | 1.98% | 8.72% | 0 / 0 |
| development / coverage / [graph-4](../runs/chain-search-v3-coverage-graph-4/summary.json) | 8/8 | 74.725 | 25.27% | 1.97% | 8.44% | 0 / 0 |
| development / coverage / [topology-4](../runs/chain-search-v3-coverage-topology-4/summary.json) | 8/8 | 74.924 | 25.08% | 1.96% | 7.90% | 0 / 0 |
| development / appearance / [baseline](../runs/chain-search-appearance-baseline/summary.json) | 8/8 | 0.639 | 99.36% | 98.08% | 98.52% | 0 / 5 |
| development / appearance / [graph-2](../runs/chain-search-v3-appearance-graph-2/summary.json) | 8/8 | 0.639 | 99.36% | 98.08% | 98.52% | 0 / 5 |
| development / strict / [baseline](../runs/chain-search-strict-baseline/summary.json) | 8/8 | 49.356 | 50.64% | 13.27% | 24.66% | 0 / 0 |
| development / strict / [graph-2](../runs/chain-search-v3-strict-graph-2/summary.json) | 8/8 | 53.866 | 46.13% | 13.17% | 22.09% | 0 / 0 |
| validation / coverage / [baseline](../runs/chain-search-coverage-baseline-validation/summary.json) | 20/20 | 78.650 | 21.35% | 2.51% | 5.77% | 0 / 0 |
| validation / coverage / [graph-2](../runs/chain-search-v3-coverage-graph-2-validation/summary.json) | 20/20 | 82.775 | 17.23% | 2.32% | 4.65% | 0 / 0 |

## chain-search-v3-coverage-graph-2

Chain retention improves 11.52% relative to its matched baseline. Largest observed per-asset time ratio: 2.83×. Asset chain-total regressions: 0. Reduction gate: pass. Single development observations; repeat sequential paired timings before promotion.

| Asset | Chain gain | Final triangles | Bake time ratio |
|---|---:|---:|---:|
| ph_painted_wooden_shelves | 5.37% | 36→36 | 2.75× |
| ph_metal_stool_02 | 2.20% | 130→112 | 2.42× |
| ph_grass_bermuda_01 | 12.27% | 52→52 | 1.24× |
| ph_dead_quiver_trunk | 33.10% | 30→14 | 2.08× |
| ph_moon_rock_02 | 8.20% | 34→16 | 2.80× |
| ph_rock_face_02 | 20.23% | 147→120 | 2.83× |
| si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652 | 7.14% | 442→221 | 1.54× |
| si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7 | 29.82% | 856→864 | 2.25× |

## chain-search-v3-coverage-graph-4

Chain retention improves 16.34% relative to its matched baseline. Largest observed per-asset time ratio: 7.21×. Asset chain-total regressions: 0. Reduction gate: pass. Single development observations; repeat sequential paired timings before promotion.

| Asset | Chain gain | Final triangles | Bake time ratio |
|---|---:|---:|---:|
| ph_painted_wooden_shelves | 5.37% | 36→36 | 7.21× |
| ph_metal_stool_02 | 18.60% | 130→107 | 6.09× |
| ph_grass_bermuda_01 | 12.27% | 52→52 | 2.39× |
| ph_dead_quiver_trunk | 45.85% | 30→14 | 3.99× |
| ph_moon_rock_02 | 13.20% | 34→12 | 5.81× |
| ph_rock_face_02 | 31.09% | 147→277 | 6.32× |
| si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652 | 27.73% | 442→221 | 5.41× |
| si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7 | 30.35% | 856→198 | 4.37× |

## chain-search-v3-coverage-topology-4

Chain retention improves 17.00% relative to its matched baseline. Largest observed per-asset time ratio: 7.19×. Asset chain-total regressions: 0. Reduction gate: pass. Single development observations; repeat sequential paired timings before promotion.

| Asset | Chain gain | Final triangles | Bake time ratio |
|---|---:|---:|---:|
| ph_painted_wooden_shelves | 5.37% | 36→36 | 7.19× |
| ph_metal_stool_02 | 26.46% | 130→116 | 6.15× |
| ph_grass_bermuda_01 | 12.27% | 52→52 | 2.42× |
| ph_dead_quiver_trunk | 33.19% | 30→8 | 3.99× |
| ph_moon_rock_02 | 13.20% | 34→12 | 5.53× |
| ph_rock_face_02 | 31.19% | 147→241 | 5.74× |
| si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652 | 27.73% | 442→221 | 2.59× |
| si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7 | 30.19% | 856→108 | 4.25× |

## chain-search-v3-appearance-graph-2

Chain retention improves 0.00% relative to its matched baseline. Largest observed per-asset time ratio: 3.22×. Asset chain-total regressions: 0. Reduction gate: fail. Single development observations; repeat sequential paired timings before promotion.

| Asset | Chain gain | Final triangles | Bake time ratio |
|---|---:|---:|---:|
| ph_painted_wooden_shelves | 0.00% | 496→496 | 3.09× |
| ph_metal_stool_02 | 0.00% | 6532→6532 | 1.64× |
| ph_grass_bermuda_01 | 0.00% | 941→941 | 1.97× |
| ph_dead_quiver_trunk | 0.00% | 17978→17978 | 1.76× |
| ph_moon_rock_02 | 0.00% | 2974→2974 | 1.66× |
| ph_rock_face_02 | 0.00% | 29566→29566 | 1.63× |
| si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652 | 0.00% | 99518→99518 | 1.40× |
| si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7 | 0.00% | 150000→150000 | 3.22× |

## chain-search-v3-strict-graph-2

Chain retention improves 8.90% relative to its matched baseline. Largest observed per-asset time ratio: 5.89×. Asset chain-total regressions: 0. Reduction gate: fail. Single development observations; repeat sequential paired timings before promotion.

| Asset | Chain gain | Final triangles | Bake time ratio |
|---|---:|---:|---:|
| ph_painted_wooden_shelves | 4.42% | 376→376 | 2.29× |
| ph_metal_stool_02 | 1.98% | 261→261 | 2.18× |
| ph_grass_bermuda_01 | 4.12% | 211→211 | 2.32× |
| ph_dead_quiver_trunk | 5.71% | 180→80 | 3.11× |
| ph_moon_rock_02 | 0.00% | 66→66 | 4.06× |
| ph_rock_face_02 | 19.59% | 590→546 | 5.89× |
| si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652 | 0.00% | 1990→1990 | 2.25× |
| si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7 | 32.95% | 1500→1336 | 3.59× |

## chain-search-v3-coverage-graph-2-validation

Chain retention improves 19.32% relative to its matched baseline. Largest observed per-asset time ratio: 2.97×. Asset chain-total regressions: 0. Reduction gate: pass. Single development observations; repeat sequential paired timings before promotion.

| Asset | Chain gain | Final triangles | Bake time ratio |
|---|---:|---:|---:|
| ph_combination_wrench | 0.81% | 38→38 | 2.06× |
| ph_didelta_spinosa | 46.04% | 15664→2312 | 2.97× |
| ph_wooden_ladder | 1.14% | 122→101 | 1.43× |
| ph_gallinera_chair | 3.25% | 460→460 | 2.31× |
| ph_book_encyclopedia_set_01 | 4.11% | 330→129 | 1.53× |
| ph_industrial_coffee_table | 0.00% | 20724→20724 | 1.41× |
| ph_kite_shield | 7.08% | 51→33 | 1.88× |
| ph_dead_quiver_branch_01 | 28.62% | 148→106 | 2.19× |
| ph_dead_quiver_branch_02 | 20.25% | 36→32 | 1.67× |
| ph_carrot_cake | 0.59% | 138→138 | 2.28× |
| ph_food_ginger_01 | 15.78% | 37→31 | 2.28× |
| ph_food_kiwi_01 | 2.94% | 30→30 | 2.35× |
| ph_coast_rocks_01 | 54.05% | 424→69 | 1.75× |
| ph_coast_rocks_05 | 58.09% | 482→161 | 1.69× |
| ph_namaqualand_stones_01 | 2.70% | 82→62 | 1.32× |
| ph_namaqualand_boulders_01 | 0.00% | 26→26 | 1.51× |
| ph_coast_rocks_02 | 21.12% | 787→13 | 1.38× |
| si_3d_package_789cf90a-4387-4ac1-9e96-c7d6a7b9d26f | 51.18% | 1500→1262 | 2.14× |
| si_3d_package_6a90207b-373e-401f-9abd-5778c1af46a3 | 74.82% | 94→22 | 1.65× |
| si_3d_package_40cf5b52-0b21-4063-95fc-aa07998eb4dd | 48.32% | 376→376 | 1.96× |

Independent audit and promotion decisions are recorded in DECISION.md. These tables establish configured-view outcomes; they do not establish all-view quality. Missing or incomplete runs have no aggregate score.
