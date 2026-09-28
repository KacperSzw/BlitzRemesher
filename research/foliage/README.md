# Foliage source library

40 distinct CC0 models are downloaded and verified, separate from the frozen
120-mesh benchmark. The collection contains 8 grasses, 6 ferns, 8 shrubs,
6 trees, 6 flowering plants/weeds and 6 leafy branches. The tree category
includes three complete broadleaf tree assemblies and three pine saplings.
The flowering/weed quota contains no dedicated reed model in this revision.

Open the [combined board](board/index.html) for ten actual LOD chains: six newly
baked nature examples and four archived opaque examples. Nature uses the same
interactive chain rows, with bake seconds and vertex-storage labels. The
[source catalog](board/catalog.html) is a separate acquisition reference. Both
work offline. [Board PNG](board/board.png),
[board PDF](board/board.pdf), [catalog PNG](board/catalog.png) and
[catalog PDF](board/catalog.pdf) are also available.

Six **geometry-only demonstration chains** are baked; see [measurements and
limitations](CHAINS.md). The source collection remains **unscored and unassigned
to dataset splits**. Its immutable manifest records the original acquisition
state; later demonstration bakes live in their own run directory.
Opacity-aware reduction/evaluation is deferred. The core importer still
rejects MASK/BLEND materials, and `blitz bench` rejects this collection
explicitly. Demo inputs deliberately omit texture/opacity material behavior in
disposable geometry files; original models stay unchanged. No existing benchmark
assets, gates, denominators or scores changed.

## Original files and material readiness

`manifest.json` freezes original files, dependencies, selectors, geometry
statistics, source/licensing evidence, SHA-256 hashes and material metadata.
`selection.json` is the human-reviewed choice of authored models and sidecar
opacity mappings. Neither file rewrites source geometry or exports corrected
GLBs. Downloaded archives include unselected variants; only the 40 manifest
entries count toward the collection.

| Source | Selected models | Material state |
|---|---:|---|
| [Loaf / Foliage_1](https://opengameart.org/content/foliage1) | 12 | Embedded color alpha, MASK 0.5, double-sided |
| [Liefz Lily / Fern Asset](https://liefz-lily.itch.io/3d-plant-fern-asset) | 2 | OBJ references a missing MTL; included RGBA atlas mapped explicitly |
| [Poly Haven / Fern 02](https://polyhaven.com/a/fern_02) | 4 | Separate opacity PNG needs hookup |
| [Poly Haven / Shrub 02](https://polyhaven.com/a/shrub_02) | 4 | Separate opacity PNG needs hookup |
| [Poly Haven / Shrub 03](https://polyhaven.com/a/shrub_03) | 4 | Separate opacity PNG needs hookup |
| [Poly Haven / Pine Sapling Small](https://polyhaven.com/a/pine_sapling_small) | 3 | Twig opacity PNG needs hookup; bark remains opaque |
| [Poly Haven / Grass Medium 02](https://polyhaven.com/a/grass_medium_02) | 5 | Source BLEND; separate opacity PNG needs hookup |
| [Poly Haven / Flower Heliophila](https://polyhaven.com/a/flower_heliophila) | 3 | Source BLEND; separate opacity PNG needs hookup |
| [Poly Haven / Weed Plant 02](https://polyhaven.com/a/weed_plant_02) | 3 | Separate opacity PNG needs hookup |

12 models have connected source materials; **28 require material setup**.
Poly Haven glTFs bind JPEG base colors, which cannot carry alpha, even when
the material declares MASK/BLEND. Matching provider opacity PNGs are preserved
separately, using their red channel. Poly Haven images are the provider's 2K
variants; archive textures retain their original resolution. [Poly Haven's
license](https://polyhaven.com/license) and each author's source page establish
CC0; original source pages and provider metadata are retained in `provenance/`.

The two Liefz fern OBJs reference an absent `Fern_01.mtl`. Their original files
stay unchanged. The sidecar records a whole-model color/alpha hookup to the
included `Fern_00.png`; source alpha policy and sidedness are unknown. This is
an explicit source packaging defect, not a complete ready-to-import material.

Previews use original triangles and UVs, a simple face-normal light, base color,
two-sided rendering and the recorded opacity map. Source MASK cutoffs are
retained; BLEND/unspecified materials use a **display-only 0.5 cutoff**. These
source images are presentation derivatives, not corrected model files, production
material validation, LODs or audited silhouettes. Browser checks decode all
17 used texture images and verify variation in every selected opacity channel.

## Counting and future scoring rules

- Each entry selects an authored mesh, OBJ shape or complete glTF node subtree.
  Tree subtrees include trunk, twig and leaf children with their world transforms.
- No LOD variants, recolors, individual parts of those trees, or instances of
  the same authored object are counted again. Exact translated triangle geometry
  duplicates are rejected; semantic/transform review remains necessary.
- The source range is 2–142,506 triangles per selected model. Tiny authored card
  clusters remain useful boundary cases; they are not padded into artificial LODs.
- Every entry is UV-mapped and has open surfaces, matching texture evidence and
  an explicit opacity binding. Planar-component counts are diagnostic: curved
  fronds and leaf sheets need not be planar. Seam-split topology can also create
  open components; topology alone does not prove a foliage card.
- The current revision contains nine source families. Keep related variants
  together in any future development/validation/held-out split. Selection is not
  a statistical claim that these 40 models represent all game foliage.
- Opacity-aware scores require a new frozen protocol, source material policy,
  camera/sample contracts and matched baselines. Do not feed these into the
  existing opaque SCORE or compare them with its 120-asset denominator.

`CANDIDATES.md` records rejected/incomplete candidates and duplicate decisions.
The old benchmark already contains `fir_sapling`, whose source declares OPAQUE
despite a separate provider opacity map. It is excluded here as an existing
source identity. That finding does not retroactively alter its frozen data or
historical results; a future opacity corpus should reclassify it explicitly.

## Reproduce

Run from the repository root. Use `nix develop .` on NixOS so ignored downloaded
data is not copied into the Nix source store. At most four CPU workers and
24 GiB combined research memory; network batches checkpoint at 50 minutes.

```sh
nix develop . --command cmake -S . -B build/foliage-release -G Ninja -DCMAKE_BUILD_TYPE=Release
nix develop . --command cmake --build build/foliage-release -j2
nix develop . --command ctest --test-dir build/foliage-release --output-on-failure

# Frozen selection only: fetch originals, reconstruct archives, verify all hashes.
build/foliage-release/blitz-corpus foliage-replay research/foliage/manifest.json data/foliage
build/foliage-release/blitz-corpus foliage-check research/foliage/manifest.json data/foliage build/foliage/check.json

# Optional discovery inventory: includes rejected candidates, not just the 40.
build/foliage-release/blitz-corpus foliage-acquire research/foliage/sources.json data/foliage-candidates
build/foliage-release/blitz-corpus foliage-inventory data/foliage-candidates build/foliage/inventory.json
```

Acquisition/replay reuse verified files, retry interrupted requests and reject
changed hashes, unsafe archive members, missing dependencies and over-budget
models. Limits: 1 GiB/file, 4 GiB/expanded archive, 50 GiB/collection and two
million triangles/model. Discovery checkpoints each acquired file. Volatile
source pages/API metadata replay from the checked-in exact snapshots; binary
model/image content replays from its public URL. The public free itch.io upload
endpoint obtains a fresh short-lived download URL; signed URLs are not frozen.
Upstream disappearance or changed binary bytes is a failure, never permission
to silently replace the frozen asset.

The original selected files occupy 178,903,905 bytes including source archives,
extracted members, textures and provenance. Original binaries stay under ignored
`data/`; only small PNG presentation derivatives are checked in.

To regenerate the catalog and board, install Playwright separately or set
`BLITZ_PLAYWRIGHT_MODULE` to an existing installation. On NixOS also set
`BLITZ_CHROMIUM` to a working Chromium executable. These variables affect only
presentation capture; the finished HTML has no external dependencies.

```sh
build/foliage-release/blitz-corpus foliage-previews research/foliage/manifest.json data/foliage build/foliage/previews.json
node tools/capture-foliage.mjs build/foliage/previews.json research/foliage/previews
build/foliage-release/blitz-board . research/foliage/board/examples.json research/foliage/board
node tools/capture-board.mjs . research/foliage/board research/foliage/board/index.html
```

The combined board also needs the recorded Round 4 exported chain files, which
are ignored model outputs. See [Round 4 board reproduction](../round4/board/README.md)
to restore those. The checked-in HTML/PNG/PDF is already self-contained.
To revise the selection after acquisition, run `foliage-freeze SELECTION ROOT
MANIFEST`; it verifies the new manifest before writing it and retains a failed
check report if requirements are unmet. Review the manifest diff before accepting
a new collection revision. Freezing captures provenance snapshots beside it.

## LOD timings and vertex storage

Each measured chain header shows seconds from its recorded row.
Round 4 stored end-to-end `seconds`, so these are labeled **Bake pipeline** and
include import, generation, audits and export. They are single-run observations
on a shared workstation, not isolated simplifier benchmarks. Newer rows with
`generation_seconds` are labeled **Chain bake**, including generation and audits.
The six new nature rows use those measured generation/audit timings; they show
actual exported LOD geometry instead of the source-preview gallery.

The chain header states the requested output mode. Each LOD tile additionally
states `SOURCE VERTICES` or `NEW VERTICES` from its actual recorded storage flag.
Thus LOD0 remains visibly shared even when subsequent levels rebuild vertices.
The first four historical chains and score charts are explicitly labeled
protocol v1. Nature rows are new geometry-only demonstrations using the current
implementation; their reductions are not included in a SCORE.

See [validation](VALIDATION.md), the [replay report](replay-check.json),
[preview checks](previews/checks.json) and [browser checks](board/browser-checks.json).
