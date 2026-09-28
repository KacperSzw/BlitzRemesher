# Visual research board

Open [index.html](index.html) directly in a browser. It contains all geometry,
styles and scripts, needs no server, and makes no network requests.
WebGL 2 is required for the interactive previews.

- [Board PNG](board.png) — all 32 LOD previews and the research summary.
- [Board PDF](board.pdf) — one-page export for sharing.
- [Wireframe board](board-wireframe.png) — the same meshes with triangle edges.
- [Stool comparison](stool-comparison.png) — enlarged source versus LOD 7.
- [Browser checks](browser-checks.json) — recorded interaction/export validation.

Drag a tile to rotate its entire chain. Click a tile to compare it with LOD 0;
the inspector exposes all eight levels, triangle counts, target screen size,
and recorded source/adjacent error limits. Clay, wireframe and silhouette
controls affect every view. Target-pixel mode shows an illustrative
orthographic preview at the recorded CSS-pixel diameter.

## What is shown

Four development examples from the recorded round2-coupled run: Moon Rock
02, Dead Quiver Trunk, Metal Stool 02 and the Smithsonian Bell X-1 scan.
Every position and triangle comes from that run's exported glTF geometry.
The board builder verifies the triangle count, nonincreasing chain and saved
source/adjacent acceptance for each level. Nothing is remeshed for the board.

These are coverage-only, direct, rebuilt chains at 32→16 pixels with a
2→3 pixel transition curve, eight proposals per level and 12 orthographic
plus four perspective audit cameras. Enlarged meshes expose detail outside
that contract. The illustrative camera angles are not the audit cameras.
Both sides of triangles are rendered with neutral colors and geometric face
normals; source textures and vertex-normal shading are not displayed.

The direct coverage score chart uses the complete eight-asset development
pilot. The corpus smoke score uses a separate progressive, two-proposal
scenario. Normal-profile results are also a separate scenario. The board
labels these differences and retains the poor normal-preservation result.

## Rebuild

From the repository root, with the existing recorded mesh exports present:

    nix develop path:. --command cmake --build build/release --target blitz-board -j 2
    build/release/blitz-board .

The C++ exporter reads [examples.json](examples.json), the frozen corpus and
recorded benchmark rows, and embeds geometry into [the HTML template](../../tools/board.html).
It writes index.html and [manifest.json](manifest.json), including mesh-file
hashes and source credits. Source meshes and benchmark glTF exports are
ignored by Git; the generated board is portable and checked in.

Optional PNG/PDF rendering uses Node and an existing Playwright installation:

    BLITZ_PLAYWRIGHT_MODULE=/path/to/node_modules/playwright \
    BLITZ_CHROMIUM=/path/to/chromium \
      node tools/capture-board.mjs .

Omit either environment variable when the corresponding package/browser is
already discoverable by Playwright. No Node dependency is needed to open the
board. The capture script checks all 32 previews, geometry counts, rendering
modes, target scale, inspector selection/rotation, LOD-0 image identity,
desktop/mobile overflow, browser errors and offline behavior before recording
browser-checks.json. A mobile preview is generated for review.

The C++ exporter also ran under AddressSanitizer and UndefinedBehaviorSanitizer
and produced byte-identical HTML and manifest files to the release build.
The existing release and sanitizer CTest suites passed (3/3 each).

## Source assets and licensing

The embedded mesh geometry derives from CC0 assets; the renderer and board
source use the repository's MIT OR Apache-2.0 license.

- [Moon Rock 02](https://polyhaven.com/a/moon_rock_02), Poly Haven.
- [Dead Quiver Trunk](https://polyhaven.com/a/dead_quiver_trunk), Poly Haven.
- [Metal Stool 02](https://polyhaven.com/a/metal_stool_02), Poly Haven.
- [Bell X-1](https://3d.si.edu/object/3d/6c69a6bb-55e6-4356-8725-120ff7f8d652), Smithsonian.

Item-level provenance and rights are preserved in [manifest.json](manifest.json),
[the corpus](../corpus.json) and [the scan rights record](../scan-rights.json).
