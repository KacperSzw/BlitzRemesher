# Nature LOD chain examples

The [board](board/index.html) includes six actual nature chains in the same
interactive rows as the rock/trunk examples. Each row has eight exported LODs,
triangle counts, measured bake seconds, source comparison and an exact-duplicate
runtime toggle. The separate source-preview gallery was removed from the board.

These demonstrations audit **full card geometry**, without texture opacity,
normals or color. They establish geometric behavior only, not rendered foliage
quality. They have no SCORE and do not enter the frozen opaque benchmark.
The first four board rows and its score charts remain archived v1 observations;
the new nature rows use the current v2 implementation.

| Model | Vertex mode | LOD0 → LOD7 triangles | Distinct runtime levels | Bake seconds |
|---|---|---:|---:|---:|
| Grass clump | Rebuild | 2,489 → 1,008 | 4 | 2.22 |
| Fern fronds | Reuse imported LOD0 vertices | 2,384 → 482 | 5 | 3.61 |
| Leafy shrub | Rebuild | 7,590 → 94 | 7 | 6.10 |
| Broadleaf tree | Rebuild | 870 → 704 | 3 | 0.70 |
| Flowering ground cover | Rebuild | 17,593 → 356 | 6 | 13.51 |
| Leafy branch | Rebuild | 120 → 31 | 3 | 0.92 |

Times cover generation, search and audits; import/export are excluded. They
are single-run observations on a shared i7-9700K workstation, with up to two
single-threaded bake processes in flight, not a performance ranking. Raw
generation, export and process wall times are retained per row. Fern reuse
binds the same imported LOD0 position/UV accessors at every level; only indices
change. Other rows allow rebuilt buffers, while unchanged fallback slots can
still share source vertices. Every tile identifies its actual storage flag.

Settings are frozen in [chains.json](chains.json): hybrid, 512→16 pixels,
eight levels, a 2→3 px transition curve, source cap 8 px, eight candidate
evaluations per level, 6+2 search views and 12+4 audit views, 4× audit sampling
with refinement to 8×. The six source/adjacent pairs at all eight slots passed
96 recorded geometry gates. These are finite sampled checks, not all-view bounds.

The reductions expose limitations rather than hiding them: the broadleaf tree
retains 704 of 870 triangles, and grass/fern tails repeat their predecessors.
Exact runtime compaction removes redundant delivered levels without claiming a
new reduction. No thresholds or search budgets were retuned after these outputs.
Improving disconnected card proposals and eventual opacity-aware evaluation
remain separate research work.

The geometry adapter writes disposable glTF/bin inputs under ignored `build/`.
It preserves selected triangle topology, transformed positions, UV0 and
material IDs/sidedness, but deliberately writes opaque, untextured materials
and excludes normal/color streams. It applies authored node transforms, so
“reuse” means sharing the imported LOD0 buffer, not the provider's file bytes.
Original downloaded files and opacity sidecars remain unchanged. The production
importer's MASK/BLEND rejection and benchmark collection guard remain active.

These examples expose five source families during development: Grass Medium 02,
Fern 02, Shrub 02, Heliophila and Loaf Foliage_1. Future dataset splitting must
keep related variants grouped and must not call these previously unseen holdouts.

## Reproduction

Build/replay the collection as described in [README](README.md), then:

```sh
build/foliage-release/blitz-corpus foliage-geometry research/foliage/manifest.json data/foliage build/foliage/geometry ph_grass_medium_02_clump_4 ph_fern_02_clump_1 ph_shrub_02_clump_1 loaf_tree_1 ph_flower_heliophila_clump_3 loaf_leafstick3_3
node tools/bake-foliage.mjs build/foliage/geometry build/foliage-release/blitz
node tools/check-foliage-chains.mjs
build/foliage-release/blitz-board . research/foliage/board/examples.json research/foliage/board
```

The driver checkpoints rows, verifies reused output hashes, bounds each batch
to 50 minutes and uses two workers. Optional trailing asset IDs run a small
subset using the same frozen configuration. Resume requires matching binary,
source, configuration, compiler and input hashes. A different build needs a
fresh run name in `chains.json` and matching board example `run` entries; do not
overwrite the archived raw measurements. No geometry-only input is added to the
original collection or made benchmark-eligible.

[Raw metadata and rows](../runs/foliage-card-geometry-v2/),
[independent export/gate checks](chain-checks.json) and
[browser verification](board/browser-checks.json) are retained. The baked mesh
files themselves are ignored; the committed board embeds their position/index
buffers and opens offline.
