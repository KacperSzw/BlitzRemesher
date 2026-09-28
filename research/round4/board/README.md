# Round 4 visual board

Open [index.html](index.html) directly; it works offline with WebGL 2.
[PNG](board.png), [PDF](board.pdf) and [wireframe PNG](board-wireframe.png)
are portable exports. Every tile contains geometry from the recorded
r4-final-s512-cap8 run, with its actual triangle count and audit results.

The examples use hybrid coupled rebuild, 512→16 px, eight scheduled slots,
a 2→3 px transition curve and an 8 px source cap. The cumulative final source
budget is 6.8905 px. The full pilot uses eight development assets; the board
illustrates four of them. The validation card refers to a separate frozen
20-asset cohort. See [the full report](../REPORT.md) for paired comparisons.

Use **Runtime levels** to hide exact duplicates. The underlying scheduled
audit records remain available. Drag to rotate a chain; click a tile to
compare it with LOD0. In target-pixel mode the inspector draws both meshes
at the selected level's screen size. Large previews are cropped to their
tile; enlarged mode fits the original source bounds into each view.

Rendering is untextured, two-sided neutral clay with geometric face normals.
These display angles are not audit cameras. Coverage-only results do not
guarantee matching shading or filled area: the aggressive two-triangle tree
is an explicit example of this limitation. It passed an additional 706-camera
tail check, but that does not make it equivalent in appearance to the source.
The [tree silhouette comparison](tree-comparison.png) makes this loss visible.

Regenerate from the repository root once the complete runs and mesh exports
are present:

    node research/round4/make-report.mjs
    build/release/blitz-board . research/round4/board/examples.json research/round4/board
    BLITZ_PLAYWRIGHT_MODULE=/path/to/node_modules/playwright \
    BLITZ_CHROMIUM=/path/to/chromium \
      node tools/capture-board.mjs . research/round4/board research/round4/board/index.html

The board builder checks saved acceptance, triangle counts and glTF mesh
mapping, including shared meshes. Browser checks cover the runtime toggle,
source comparison, rotation, rendering modes, target scale, mobile layout
and absence of network requests. [manifest.json](manifest.json) contains
source credits, CC0 rights links and geometry hashes.

## README animation

The repository's [README GIF](../../../docs/media/lod-chain-showcase.gif) and
[still frame](../../../docs/media/lod-chain-showcase.png) are captured from this
checked-in board. Regenerate them from the repository root with Node.js,
Playwright, Chromium and FFmpeg available:

    BLITZ_PLAYWRIGHT_MODULE=/path/to/node_modules/playwright \
    BLITZ_CHROMIUM=/path/to/chromium \
    BLITZ_FFMPEG=/path/to/ffmpeg \
      node tools/capture-readme-demo.mjs .

The three `BLITZ_*` variables are optional when the tools are on the usual
module or executable paths. Capture uses fixed frames and validates the board
run, source attribution, triangle counts, geometry and audit acceptance before
encoding. It does not rerun simplification or download assets.
