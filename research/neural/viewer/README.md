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
