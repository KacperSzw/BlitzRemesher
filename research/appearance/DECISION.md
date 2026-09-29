# Decision: keep appearance stages experimental

The implementation is complete; useful appearance reduction has not been established. Production defaults remain unchanged. All limits and the sampled final-audit policy are preserved.

## What the controlled replay established

4 fixed-candidate/camera groups change from fail to pass at a higher sample density, and 9 change from pass to fail. This demonstrates nonmonotonic sampling, not that a particular density is correct. Stronger 50% triangle targets mostly retain normal-mismatch witnesses. Source screens alone do not explain the stalled chains. [Raw replay](replay-manifest.json).

## Pilot and independent audits

| Stage | Pilot complete | Useful-reduction gate | Observed time ≤4× | Independently qualified reduced chains |
|---|---|---|---|---:|
| appearance-screen | Yes | Not passed | No / incomplete | 1 |
| appearance-ordering | Yes | Not passed | No / incomplete | 1 |
| appearance-attributes | Yes | Not passed | No / incomplete | 1 |
| appearance-position | No | Not passed | No / incomplete | 0 |

The position-fitting trial was stopped after six completed assets: its manufactured assets already made the four-category usefulness gate impossible, and the Bell attempt exceeded the bake allowance. Its partial rows and [interruption record](../runs/appearance-position/interruption.json) are retained without an aggregate score.

Individual timing observations cannot establish repeatability. Qualification requires a complete fresh 642+64-camera audit (seed 0xA1172027) of LOD1–7 against source and predecessor. Identity fallbacks pass without providing reduction benefit.

### appearance-screen

| Asset | Outcome | Failure witnesses |
|---|---|---|
| ph_painted_wooden_shelves | Fail | LOD7 source, view 127: appearance mismatch; LOD7 adjacent, view 53: appearance mismatch |
| ph_metal_stool_02 | Pass · unchanged source | — |
| ph_grass_bermuda_01 | Pass · unchanged source | — |
| ph_dead_quiver_trunk | Pass · unchanged source | — |
| ph_moon_rock_02 | Fail | LOD5 source, view 256: appearance mismatch; LOD5 adjacent, view 19: appearance mismatch; LOD6 source, view 151: appearance mismatch; LOD7 source, view 149: appearance mismatch |
| ph_rock_face_02 | Pass · unchanged source | — |
| si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652 | Pass · reduced | — |
| si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7 | Pass · unchanged source | — |
### appearance-ordering

| Asset | Outcome | Failure witnesses |
|---|---|---|
| ph_painted_wooden_shelves | Fail | LOD7 source, view 127: appearance mismatch; LOD7 adjacent, view 53: appearance mismatch |
| ph_metal_stool_02 | Pass · unchanged source | — |
| ph_grass_bermuda_01 | Pass · unchanged source | — |
| ph_dead_quiver_trunk | Fail | LOD6 source, view 210: appearance mismatch; LOD6 adjacent, view 0: appearance mismatch; LOD7 source, view 148: appearance mismatch |
| ph_moon_rock_02 | Fail | LOD5 source, view 328: appearance mismatch; LOD5 adjacent, view 32: appearance mismatch; LOD6 source, view 151: appearance mismatch |
| ph_rock_face_02 | Pass · unchanged source | — |
| si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652 | Pass · reduced | — |
| si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7 | Pass · unchanged source | — |
### appearance-attributes

| Asset | Outcome | Failure witnesses |
|---|---|---|
| ph_painted_wooden_shelves | Fail | LOD7 source, view 127: appearance mismatch; LOD7 adjacent, view 53: appearance mismatch |
| ph_metal_stool_02 | Pass · unchanged source | — |
| ph_grass_bermuda_01 | Pass · unchanged source | — |
| ph_dead_quiver_trunk | Fail | LOD6 source, view 210: appearance mismatch; LOD6 adjacent, view 0: appearance mismatch; LOD7 source, view 148: appearance mismatch |
| ph_moon_rock_02 | Fail | LOD5 source, view 328: appearance mismatch; LOD5 adjacent, view 32: appearance mismatch; LOD6 source, view 151: appearance mismatch |
| ph_rock_face_02 | Pass · unchanged source | — |
| si_3d_package_6c69a6bb-55e6-4356-8725-120ff7f8d652 | Pass · reduced | — |
| si_3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7 | Pass · unchanged source | — |
### appearance-position

| Asset | Outcome | Failure witnesses |
|---|---|---|
| ph_painted_wooden_shelves | Fail | LOD7 source, view 127: appearance mismatch; LOD7 adjacent, view 53: appearance mismatch |
| ph_metal_stool_02 | Pass · unchanged source | — |
| ph_grass_bermuda_01 | Pass · unchanged source | — |
| ph_dead_quiver_trunk | Fail | LOD6 source, view 210: appearance mismatch; LOD6 adjacent, view 0: appearance mismatch; LOD7 source, view 148: appearance mismatch |
| ph_moon_rock_02 | Fail | LOD5 source, view 328: appearance mismatch; LOD5 adjacent, view 32: appearance mismatch; LOD6 source, view 151: appearance mismatch |
| ph_rock_face_02 | Pass · unchanged source | — |

Validation20 and three paired timing repetitions are gated on a useful pilot and are not launched for a failed development gate. Held-out release assets remain unused. No game-ready claim is made for failing or unaudited chains.

## Next evidence needed

Every selected chain in the complete ordering pilot reuses source vertices and adds zero vertex bytes. Ordering and attribute fitting produce byte-identical exported geometry and attributes, so extra fitting work did not reach the delivered chains. The next bounded ablation should examine rebuilt proposal density: the current target clamp divides the available vertex budget by three vertices per triangle, even though actual compact vertex counts decide admission. Compare targets nearer the measured vertex budget while retaining the same cap and independent gates. This is a testable proposal, not evidence that denser candidates will qualify.

[Measurements](REPORT.md) · [Offline board](../../examples/appearance-board/index.html) · [Progress curves](progress.svg) · [Verification](VERIFICATION.md).
