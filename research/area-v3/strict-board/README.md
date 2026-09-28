# Strict progressive pilot board

Open [index.html](index.html) directly. The board is offline and self-contained.
Its four top tabs show the complete frozen eight-asset pilot under one recorded
settings set each: 3 px source cap with quadric B or conditional fallback E,
then 4 px source cap with B or E. Each tab draws exported source and selected
LOD geometry for all eight assets at matched presentation angles and scale.
The LOD slider and clay, wireframe, and silhouette controls apply across the
active tab.

Every reduced LOD allows a 2 px change from its predecessor. The cumulative
source limit is capped at 3 or 4 px. The recorded source-limit schedules for
LOD 1–7 are `[2, 3, 3, 3, 3, 3, 3]` px and approximately
`[2, 3.219, 3.962, 4, 4, 4, 4]` px, respectively. Both source and adjacent
audits also require changed area at or below 50%. The row for the selected
LOD shows its actual pixel limits and recorded worst-view measurements.

The four runs use the same corrected v2 executable, pilot manifest, and
configured audit protocol. Forced research rebuild mode selects the
triangle-first chain; the normalized 500-basis-point overhead setting is not
used for that selection. B and E differ only by the opt-in fallback setting
at each source cap. E used five extra topology proposals per eight-asset run.
At both caps, B SCORE is 76.277746 and E SCORE is 76.307343, a gain of
0.029597 points. Six of eight B/E exports match byte-for-byte. The 3 px and
4 px runs also match eight of eight exports within each B or E mode, so
loosening this source cap did not change these pilot outputs.

Direct 706-view, 8× rotated tail checks on the 3 px E run pass all six
source/adjacent gates at LOD 5–7 for all eight pilot assets on the first
camera seed. The board links each raw audit; the [grass
audit](../dense-strict-source-3-progressive-2-cap-0.5-grass-e.json) completed
the set. [Bell](../dense-strict-source-3-progressive-2-cap-0.5-bell-e-seed-2027.json)
and [moon rock](../dense-strict-source-3-progressive-2-cap-0.5-moon-rock-e-seed-2027.json)
also pass a second rotation. The tree's final source area is 40.48%/50%.
Bell LOD 6 adjacent pixel error is 1.99767/2 px, leaving only 0.00233 px
margin on both seeds. The 4 px E exports are byte-identical and its source
pixel limits are no tighter. Direct first-seed rotated checks also pass all
six tail gates for all eight 4 px E assets. The board links each direct
audit within its own run tab. At either cap, direct B checks pass Bell and
hatch; the other six B exports are byte-identical to passing E exports under
the same contract. This gives finite first-seed evidence for all eight B
meshes while distinguishing direct B audits from inherited E measurements.
These finite camera sets do not establish all-view robustness or promote a
preset.

SCORE is `100 × (1 − category-balanced mean asset chain retention)`. Asset
chain retention is its average LOD 1–7 triangle count divided by source
triangles. The board shows that value alongside each selected LOD's final
retention. Area bars and pixel values show acceptance headroom; they do not
prove that a lower-triangle proposal was available. These remain pilot
results, not a promoted preset.

From the repository root, after all four raw runs are complete:

```sh
node research/area-v3/make-strict-board.mjs
BLITZ_PLAYWRIGHT_MODULE=/path/to/node_modules/playwright \
  node research/area-v3/strict-board/check-board.mjs
```

The generator validates run provenance, settings, every configured audit,
SCORE recomputation, and exported geometry before writing the board. The
browser check verifies all four tabs, 32 asset rows, visible geometry,
controls, and desktop/mobile layout.
