# Coverage-area pilot board

Open [index.html](index.html) directly. The board is offline and self-contained.
The five top tabs contain the complete eight-asset pilot under one recorded
settings set each: uncapped reference, area cap 0.5, cap 0.5 with adaptive
targets, experimental cap 0.5 with a topology relaxed objective, and corrected
post-hoc v2 cap 0.5 with an opt-in conditional topology fallback. Each row
draws the exported source and selected LOD geometry at the same presentation
angle and scale. The LOD slider changes all eight rows
together. Drag a canvas to rotate its pair.

The images use untextured geometric face normals. They illustrate actual mesh
outputs; they are not audit cameras. The area and pixel values are from the
recorded source and adjacent audits. The “worst across chain” values may come
from different LODs or views. Textures, opacity and shading are outside this
opaque-geometry contract.

The cap-only tree passes the configured 16-view audit at 42.73% final source
area, but an [independent rotated dense check](../dense-tree-cap-0.5.json)
found 51.29% at view 21, above the 50% limit. The board shows this failure in
the cap-only tab and tree row. The preset is not ready for promotion.

The experimental topology relaxed run lowers the Apollo hatch final LOD from
844 to 48 triangles and Bell X-1 from 1273 to 64. Both pass independent
rotated dense tail checks. Its global eight-asset SCORE regresses by 0.44
points versus cap-only. Independent rotated checks fail the D tree at 51.41%
source area and D rock face at 50.71% source / 50.98% adjacent area against
the 50% limit. The D tab links all four dense checks; this result does not
promote a global topology relaxed preset.

The corrected post-hoc v2 conditional fallback adds one topology relaxed
proposal after a QEM link stall. Five extra proposals affect only Bell X-1 and
Apollo hatch; their final LODs fall from 1273→128 and 844→96 triangles. Both
pass direct v2 independent rotated dense tail audits. Six other pilot exports
exactly match cap-only B, including the tree. A direct [v2 E tree rotated dense
check](../dense-tree-topology-fallback-v2.json) confirms its 51.29% final
source-area failure against the 50% limit. E's pilot SCORE improves by 0.0556
points versus the [same-binary v2 B control](../../runs/area-v3-cap-0.5-fallback-v2-b/summary.json),
whose eight outputs match frozen B. All eight corrected v2 E exports reproduce
the historical E pilot. E remains opt-in and is not a promoted global preset.

SCORE averages each asset's LOD 1–7 triangle retention, then averages equally
across categories. Each row shows this chain retention separately from the
selected LOD's retention. The area bars and pixel figures report acceptance
headroom; they do not establish that a smaller proposal exists.

From the repository root, after all five pilot batches and independent dense
checks finish:

```sh
node research/area-v3/make-board.mjs
BLITZ_PLAYWRIGHT_MODULE=/path/to/node_modules/playwright \
  node research/area-v3/board/check-board.mjs
```

`make-board.mjs` requires complete matching eight-asset runs. It validates
recorded acceptance, glTF triangle counts, source identity and all run hashes
before embedding geometry. Identical meshes are stored once by content hash.
`manifest.json` records geometry and export hashes. Browser checks verify all
40 rows, live geometry, tab and LOD switching, wireframe mode, mobile layout
and offline loading. The six PNGs are optional snapshots of those checks.
