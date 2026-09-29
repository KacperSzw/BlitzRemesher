# Inspect the actual network output

Open [index.html](index.html) in a browser. It is self-contained and works
offline, without a server, libraries, model downloads or cloud compute.

The viewer compares source LOD 0, constant ranking and network seed 101 after
16,384 updates, using the saved second screening comparison. Drag any large
view to rotate all three; scroll to zoom. Select either asset, any LOD, and
wireframe, shaded, changes, normal or UV checker display.

At the final LOD (16-pixel target), moon rock goes from **3,304 to 2,908
triangles**, a reduction of **11.99%**; the control has 3,284. Shelves go from
**524 to 520**, a reduction of **0.76%**; the control has 522. These individual
LOD reductions differ from the whole-chain averages, 2.37% and 0.49%.

The enlarged views share the source frame, camera, scale and neutral material.
They use the exported positions, indices, normals and UVs without simplifying
or recomputing those streams. No original texture maps are shown. Tiny previews
use the selected target diameter in CSS pixels; large target diameters exceed
their small preview frames and are cropped. These are inspection renders, not
new quality audits or all-view guarantees.

In Changes mode, red source faces have no exact position-triangle match in the
network result. Amber result faces have no exact match in the source. This
shows removed/retriangulated/moved geometry; it does not measure normal or UV
error. Both methods remain visible, including unreduced LODs.

Input hashes and all displayed triangle counts are in [manifest.json](manifest.json).
[Browser checks](browser-checks.json) verify all 16 asset/LOD selections, triangle
counts, nonempty rendering, pixel-identical LOD 0 comparisons, distinct modes,
rotation/reset, small target previews, mobile fit, and zero external requests
or browser errors. Rendering uses software WebGL during capture. No training
or algorithm changes were made.

Rebuild from the recorded files:

```sh
node tools/neural-viewer.mjs
BLITZ_PLAYWRIGHT_MODULE=/path/to/node_modules/playwright \
  BLITZ_CHROMIUM=/path/to/chromium node tools/capture-neural-viewer.mjs
```

The raw training and comparison evidence is [here](../evidence/action-screening-v3/README.md).

## Learning loop and audit costs

Open [learning-loop.html](learning-loop.html) for the clickable training and
audit diagrams. Select a step, use the arrow keys, or use Previous/Next to read
its explanation. The two timing charts show the whole rental and the final
moon-rock generation; each segment links to its recorded evidence.

The rental chart adds the saved baseline and both comparison wall times,
the six trainer durations (including checkpoint checks), and the four fresh
preparer durations. The rest is explicitly a **derived remainder**, covering
setup/process/collection time without claiming their individual shares. The
rock chart separates audit/confirmation, inference, and remaining asset time.
Image sample counts illustrate one saved LOD and are not whole-run workload
measurements. Proposed optimizations have no claimed speedup.

[Displayed data](learning-loop-data.json), [input hashes](learning-loop-manifest.json)
and [browser checks](learning-loop-checks.json) accompany the page. It uses no
external scripts, fonts or automatic network requests. Source links can be
opened deliberately. All content comes from recorded evidence; no training,
mesh generation or cloud rental is performed.

```sh
node tools/neural-loop.mjs
BLITZ_PLAYWRIGHT_MODULE=/path/to/node_modules/playwright \
  BLITZ_CHROMIUM=/path/to/chromium node tools/capture-neural-loop.mjs
```

Browser checks cover every diagram node, keyboard navigation, proportional
timing bars, evidence links, input hashes and totals, and three viewport widths.
Screenshots preserve the learning overview, audit view and mobile layout.
