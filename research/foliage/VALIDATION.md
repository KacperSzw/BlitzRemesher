# Validation record

Recorded 28 September 2026 with the pinned Nix flake, Clang 21.1.8 on NixOS.
The core library is unchanged; the CLI adds separate generation/export timings.

| Check | Result |
|---|---|
| Clang release CTest | 6/6 passed |
| AddressSanitizer + UndefinedBehaviorSanitizer CTest | 6/6 passed |
| Collection inspection and dependency verification | 40/40 models; 190 files |
| Fresh download/reconstruction replay | All selected binary files and archived members match hashes |
| Source preview capture | 40/40 visible; 17 images decoded; all selected opacity channels vary |
| Geometry-only chain bakes | 6/6 complete; 48 exported LOD slots |
| Independent chain verification | 96 geometry gates; indices, counts and reuse vertex accessor identity passed |
| Combined board | 10 chains / 80 tiles; no separate nature gallery |
| Browser | No errors or network requests; desktop and 390 px mobile fit |
| Frozen corpus/protocol | Original SHA-256 values unchanged |

The new C++ contracts cover verified cache reuse, corrupted-file replacement,
wrong hashes, interrupted transfers, archive traversal/link rejection, missing
dependencies, explicitly disclosed absent MTLs, duplicate geometry/counting,
unscored source state, encoded URI traversal, retained MASK metadata, and
complete transformed glTF assemblies. Production importer tests explicitly
retain MASK/BLEND rejection. The benchmark guard rejects the collection before
creating run output; see [benchmark-guard.json](benchmark-guard.json).

The geometry adapter was also executed over the real selected chain sources
under ASan/UBSan and produced the identical extraction manifest. A complete
sanitized leafy-branch bake matched the release geometry bytes and all recorded
gate results. Source card topology/material checks are collection evidence,
not opacity-aware quality tests. Texture masks used for source thumbnails never
enter the geometric reduction gates or the opaque SCORE.

Browser checks verify every tile's triangle count and visible rendering, bake
seconds, vertex labels, runtime compaction, clay/wire/silhouette modes, target
scale, source identity, inspector selection/rotation and catalog category
filters. The six nature rows contain baked geometry and measured timings.

Build logs are under ignored `build/foliage/`. Reproducible reports:
[collection check](manifest.json.check.json), [replay](replay-check.json),
[preview checks](previews/checks.json), [chain checks](chain-checks.json),
[browser checks](board/browser-checks.json). No reducer algorithm, scoring gate,
benchmark denominator or AGENTS.md change is part of this work.
