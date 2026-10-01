# Appearance reduction measurements

Frozen eight-asset pilot, identical visual/storage limits and proposal budget. Triangle gains compare against v3 appearance graph; bake ratios compare against the original appearance baseline. Individual timings are not paired performance claims.

| Stage | Complete | SCORE | Chain gain | Asset regressions | Useful tails / categories | Max observed bake ratio |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 8/8 | 0.639 | — | — | — | — |
| original | 8/8 | 0.639 | — | — | — | — |
| screen | 8/8 | 0.772 | 0.13% | 0 | 0 / 0 | 4.33× |
| ordering | 8/8 | 0.862 | 0.22% | 0 | 0 / 0 | 8.60× |
| merged | 8/8 | 0.772 | 0.13% | 0 | 0 / 0 | 4.00× |
| shared | 8/8 | 1.721 | 1.09% | 0 | 0 / 0 | 5.43× |

## appearance-screen

| Asset | Chain gain | Final triangles | Bake ratio |
|---|---:|---:|---:|
| ph_painted_wooden_shelves | 0.00% | 496→496 | 4.33× |
| ph_metal_stool_02 | 0.00% | 6532→6532 | 2.65× |
| ph_grass_bermuda_01 | 0.00% | 941→941 | 2.06× |
| ph_dead_quiver_trunk | 0.00% | 17978→17978 | 2.80× |
| ph_moon_rock_02 | 1.11% | 2974→2892 | 2.16× |
| ph_rock_face_02 | 0.00% | 29566→29566 | 2.22× |
| si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652 | 0.00% | 99518→99518 | 1.31× |
| si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7 | 0.00% | 150000→150000 | 1.85× |

## appearance-ordering

| Asset | Chain gain | Final triangles | Bake ratio |
|---|---:|---:|---:|
| ph_painted_wooden_shelves | 0.00% | 496→496 | 4.11× |
| ph_metal_stool_02 | 0.00% | 6532→6532 | 2.72× |
| ph_grass_bermuda_01 | 0.00% | 941→941 | 1.24× |
| ph_dead_quiver_trunk | 1.79% | 17978→16854 | 3.70× |
| ph_moon_rock_02 | 0.00% | 2974→2974 | 2.77× |
| ph_rock_face_02 | 0.00% | 29566→29566 | 2.87× |
| si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652 | 0.00% | 99518→99518 | 8.60× |
| si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7 | 0.00% | 150000→150000 | 7.20× |

## density-v3-merged

| Asset | Chain gain | Final triangles | Bake ratio |
|---|---:|---:|---:|
| ph_painted_wooden_shelves | 0.00% | 496→496 | 4.00× |
| ph_metal_stool_02 | 0.00% | 6532→6532 | 2.59× |
| ph_grass_bermuda_01 | 0.00% | 941→941 | 1.91× |
| ph_dead_quiver_trunk | 0.00% | 17978→17978 | 2.46× |
| ph_moon_rock_02 | 1.11% | 2974→2892 | 2.15× |
| ph_rock_face_02 | 0.00% | 29566→29566 | 2.18× |
| si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652 | 0.00% | 99518→99518 | 1.32× |
| si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7 | 0.00% | 150000→150000 | 1.99× |

## density-v5-shared

| Asset | Chain gain | Final triangles | Bake ratio |
|---|---:|---:|---:|
| ph_painted_wooden_shelves | 0.00% | 496→496 | 5.43× |
| ph_metal_stool_02 | 0.00% | 6532→6532 | 3.19× |
| ph_grass_bermuda_01 | 0.00% | 941→941 | 4.45× |
| ph_dead_quiver_trunk | 1.56% | 17978→17698 | 3.00× |
| ph_moon_rock_02 | 5.55% | 2974→2904 | 3.70× |
| ph_rock_face_02 | 0.00% | 29566→29566 | 2.25× |
| si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652 | 0.89% | 99518→93298 | 1.36× |
| si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7 | 0.89% | 150000→147656 | 1.75× |

Incomplete cohorts have no aggregate. Source fallbacks remain in the denominator. Independent qualification is recorded separately in DECISION.md.
