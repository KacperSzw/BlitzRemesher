# Opaque coverage-area experiment (protocol v3)

[Interactive development board](board/index.html) shows all eight pilot assets
under each run with the same LOD control. A/B/C are frozen comparisons; D and E
are explicitly post-hoc experiments.

**Current status:** the 0.5 cap passed the configured eight-asset pilot with a
0.07-point SCORE cost and the 20-asset validation split with a 0.14-point cost.
Validation's worst final source-area change fell from 76.17% to 46.49%.
However, the tree failed a rotated dense area audit at 51.29% against the
50% limit. The cap remains an optional experimental gate; the 0.5 quality
preset is not promoted as independently validated.

The frozen eight-asset development pilot uses the Round 4 512→16 px hybrid
rebuild contract: eight slots, source cap 8 px, eight proposals per slot,
beam width two, search cameras 6+2 at 2× and audit cameras 12+4 at 4× with
refinement to 8×. The three configs vary only the area limit and the
experimental adaptive-target scheduler:

| Config | Maximum changed area | Adaptive targets | Purpose |
|---|---:|---|---|
| `uncapped.json` | 1.0 | off | Fresh v3 reference |
| `cap-0.5.json` | 0.5 | off | Area gate by itself |
| `cap-0.5-adaptive.json` | 0.5 | on | Scheduler at the same quality contract |

Run each batch with the same executable and build stamp. Keep the benchmark's
50-minute checkpoint, one thread per bake and the shared-workstation limit of
four workers / 24 GiB combined. Resume only through the benchmark's hash
checks. Raw rows and mesh exports belong in distinct run directories.

```sh
bash tools/stamp-build.sh build/area-v3-build.json build/release/blitz
build/release/blitz bench research/pilot.json research/area-v3/configs/uncapped.json research/runs/area-v3-uncapped --build-stamp build/area-v3-build.json
build/release/blitz bench research/pilot.json research/area-v3/configs/cap-0.5.json research/runs/area-v3-cap-0.5 --build-stamp build/area-v3-build.json
build/release/blitz bench research/pilot.json research/area-v3/configs/cap-0.5-adaptive.json research/runs/area-v3-cap-0.5-adaptive --build-stamp build/area-v3-build.json
node research/area-v3/make-report.mjs research/runs/area-v3-uncapped research/runs/area-v3-cap-0.5 research/runs/area-v3-cap-0.5-adaptive research/area-v3
```

Publish per-asset source and adjacent area maxima, their worst views,
triangle retention, resident storage, generation time, peak RSS, failure and
fallback state. Include all eight assets in SCORE, including failures and
unreduced chains. Incomplete batches receive no SCORE. The cap is a quality
change, so the capped and uncapped SCORE values are a quality–triangle
tradeoff. Only cap-only versus cap-plus-adaptive is an algorithm comparison.

The 0.5 candidate passes the pilot screen if all configured gates pass, the
tree's final LOD uses less than 1% of its source triangles, there is no new
unreduced pilot asset and its SCORE is within five points of the fresh
uncapped run. Select
adaptive targets only if the pilot improves by at least one SCORE point over
cap-only and the 20-asset validation comparison has no SCORE or failure
regression. Record a negative result if either threshold fails.

For validation, run the fresh uncapped and cap-only configs against
`research/corpus.json` with `--split validation`, then supply those run
directories as the fifth and sixth `make-report.mjs` arguments. Run the
adaptive config on validation only if its capped pilot SCORE improves by at
least one point, and supply that directory as the optional seventh argument.
Compare the cap's quality cost against the uncapped validation run; compare
adaptive targets only with cap-only at the same limit. Do not tune on
validation or held-out data.

The post-hoc E analysis reads its corrected v2 development pilot and
same-binary B pilot control from their named directories. To include the
separate 20-asset E validation, append
`research/runs/area-v3-validation-cap-0.5-fallback-v2-b` and
`research/runs/area-v3-validation-cap-0.5-fallback-v2-e` as the seventh
and eighth `make-report.mjs` arguments. Those runs share E's executable and
differ only in the topology-fallback setting.

Recheck the last three LODs of the tree with a separate camera rotation:

```sh
build/release/blitz-tail-audit research/runs/area-v3-cap-0.5 ph_dead_quiver_trunk research/area-v3/dense-tree-cap-0.5.json 0xDA7A2026
```

The dense audit uses 642 orthographic plus 64 perspective cameras, 8× sampling
and refinement up to 32×. It records source and adjacent distance and area
gates independently. It does not change SCORE or claim an all-view bound.

## Exploratory conditional fallback for Apollo hatch

The optional [E config](configs/cap-0.5-topology-fallback.json) keeps the quadric
objective and tries at most one additional topology-relaxed proposal per LOD
when link-condition rejections leave a proposed mesh more than four times
above its triangle target. On the eight development assets the corrected v2
binary used five extra proposals, improved SCORE from 79.4959 to 79.5515
against its same-binary B control, and reduced Apollo hatch from 844 to 96
final triangles and Bell X-1 from 1273 to 128. All configured audits pass.
The gain is a bounded extra-work comparison, not an equal-work optimizer
claim: E may make one additional reducer call per LOD.
The v2 B and E outputs match their prior eight-asset runs exactly; those older
runs remain historical evidence. Corrected direct rotated dense audits pass
all six tail gates for [Apollo hatch](dense-hatch-topology-fallback-v2.json)
and [Bell X-1](dense-bell-topology-fallback-v2.json). E's tree export is
byte-identical to frozen B's, and its [direct tree audit](dense-tree-topology-fallback-v2.json)
fails LOD7 source area at 51.29% against the 50% cap. The conditional fallback remains opt-in: its
topology-relaxed proposals do not guarantee manifold topology or prevent new
intersections. Inspect the chosen export for the object in use. The [corrected
E pilot](../runs/area-v3-cap-0.5-fallback-v2-e/summary.json) and [same-binary
B control](../runs/area-v3-cap-0.5-fallback-v2-b/summary.json) preserve the
post-hoc results.

On the separate 20-asset validation split, corrected E scores 86.0671 versus
86.0609 for its same-binary B control, a 0.0062-point gain with 18 extra
proposals. Nineteen of 20 output and attribute hashes are unchanged. The one
changed export falls from 894 to 112 final triangles and passes its
[independent rotated dense tail audit](dense-validation-si_3d_package_789cf90a-4387-4ac1-9e96-c7d6a7b9d26f-topology-fallback-v2.json).
The unchanged tree's dense failure still prevents global promotion. [Validation
B](../runs/area-v3-validation-cap-0.5-fallback-v2-b/summary.json) ·
[validation E](../runs/area-v3-validation-cap-0.5-fallback-v2-e/summary.json).

```sh
build/topology-fallback-v2/blitz simplify data/smithsonian/3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7/source.obj \
  --config research/area-v3/configs/cap-0.5-topology-fallback.json \
  --out build/hatch-topology-fallback
```

## Strict pixel-contract follow-up

Four user-directed eight-asset development runs keep the 0.5 area cap,
corrected v2 executable, cameras and nominal eight-proposal budget while
fixing the adjacent/progressive limit at 2 px. The source limit is set to 3
or 4 px, with a B quadric control and E conditional fallback for each source
cap. [Strict all-asset board](strict-board/index.html) and the separate
[report section](REPORT.md#strict-pixel-contract-scenarios) show per-asset
triangles, quality results and provenance. Compare B and E only within the
same source cap; the pixel contracts differ across caps.

| Source cap | Actual LOD1–7 source limits | B SCORE | E SCORE | E extra proposals |
|---|---|---:|---:|---:|
| 3 px | 2, 3, 3, 3, 3, 3, 3 px | 76.2777 | 76.3073 | 5 |
| 4 px | 2, 3.219, 3.962, 4, 4, 4, 4 px | 76.2777 | 76.3073 | 5 |

All four runs complete 8/8 with zero failures or fallbacks and passing
configured audits. The 3 px and 4 px exports have identical output and
attribute hashes for all eight B assets and all eight E assets. Painted
shelves finish at 52 triangles in all four strict runs, versus 26 with the
earlier, looser B contract. In 3 px B, LOD6 source error is 2.72/3 px and
adjacent error is 1.98/2 px; LOD7 retains 52 triangles. Relaxing the source
cap to 4 reduces shelves source-audit rejections from four to zero, while
adjacent-audit rejections rise from five to nine. This is object-specific
evidence of the adjacent limit's effect, not proof of a global triangle
optimum. Forced research rebuild selects triangles first even though
metadata retains the default `triangle_overhead_bps=500` value.

All eight 3 px E and all eight 4 px E exports pass direct rotated
642+64-camera dense tail audits at 8× with refinement to 32×; the
[3 px tree](dense-strict-source-3-progressive-2-cap-0.5-tree-e.json),
[Bell X-1](dense-strict-source-3-progressive-2-cap-0.5-bell-e.json) and
[Apollo hatch](dense-strict-source-3-progressive-2-cap-0.5-hatch-e.json) are
among them. Each B contract also passes the first rotated tail check:
Bell/hatch were audited directly, and the other six B exports are byte-identical
to their directly audited E counterparts. Bell's LOD6
adjacent error is 1.99767/2 px, leaving only 0.00233 px; a
[second Bell rotation](dense-strict-source-3-progressive-2-cap-0.5-bell-e-seed-2027.json)
passes with the same narrow margin. A
[second Moon rock rotation](dense-strict-source-3-progressive-2-cap-0.5-moon-rock-e-seed-2027.json)
also passes. These are finite camera checks. The
strict tree is byte-identical for B and E at 34 triangles, and its first
rotated source-area maximum is 40.48%/50%, versus 51.29%/50% for the looser
18-triangle B tree. The tree repair therefore comes from the stricter pixel
contract on this asset, not topology fallback. Tree and grass also have
area-only audit rejections under both strict source caps, so the 50% area
gate remains active alongside the adjacent 2 px limit.

```sh
for source in 3 4; do
  for mode in b e; do
    name="strict-source-${source}-progressive-2-cap-0.5-${mode}"
    build/topology-fallback-v2/blitz bench research/pilot.json \
      "research/area-v3/configs/${name}.json" "research/runs/area-v3-${name}" \
      --minutes 50 --build-stamp build/topology-fallback-v2/stamp.json
  done
done
```

## Broader topology-relaxed development probe

For the Apollo hatch, the optional `topology_relaxed` objective reaches 48
final triangles versus 844 with the quadric objective under the same 0.5 cap.
It changes topology, so choose it deliberately for this object and inspect the
export. The eight-asset development SCORE regresses from 79.50 to 79.05;
there is no validated global dispatch rule. The [raw D run](../runs/area-v3-cap-0.5-topology-relaxed/summary.json)
and [rotated dense hatch audit](dense-hatch-topology-relaxed.json) retain the
post-hoc evidence; all six hatch tail gates pass on that rotation.

```sh
build/release/blitz simplify data/smithsonian/3d_package_e7514eea-3f12-490d-a2d0-999f2a1a70f7/source.obj \
  --config research/area-v3/configs/cap-0.5-topology-relaxed.json \
  --out build/hatch-topology-relaxed
```
