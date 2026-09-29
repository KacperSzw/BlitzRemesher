# Current-policy example board

[Open the self-contained interactive board](../../examples/current-board/index.html).
It shows eight current-policy chains: three nature meshes, two rocks, two
manufactured meshes, and a mixed potted plant. Every chain has a final
scheduled 16 px audit slot. The
initial board selection is Moon rock at that slot. Runtime view hides exact
consecutive duplicate meshes, so its last visible runtime mesh may have been
introduced at a larger scheduled size.

All runs use `max_added_vertex_bytes_bps=2000` (20%),
`triangle_overhead_bps=0`, coverage profile, eight proposals per scheduled
transition, six orthographic plus two perspective search views, and twelve
orthographic plus four perspective audit views. Capped generation also uses up
to eight audited final-LOD probes outside the per-level proposal budget. Seven
assets use eight levels at 512→16 px with the archived Round 4 screen schedule.
Painted shelves uses a four-level 64→16 px bounded smoke setting. These are
development examples, not a matched corpus score or a default-quality audit.

| Asset | Scheduled triangles | Runtime slots | Added / allowed vertex bytes | Bake seconds |
|---|---|---|---:|---:|
| [Moon rock](moon-rock-result.json) | 3304 / 2974 / 1488 / 744 / 108 / 36 / 36 / 16 | 0, 1, 2, 3, 4, 5, 7 | 11,360 / 11,820 | 9.70 |
| [Bermuda grass](grass-bermuda-result.json) | 941 / 941 / 846 / 423 / 211 / 41 / 13 / 13 | 0, 2, 3, 4, 5, 6 | 3,712 / 5,382 | 4.08 |
| [Lambis shell](lambis-shell-result.json) | 12498 / 11248 / 11248 / 5624 / 430 / 430 / 430 / 62 | 0, 1, 3, 4, 7 | 43,680 / 47,110 | 7.34 |
| [Metal stool](metal-stool-result.json) | 6532 / 5878 / 2939 / 1469 / 1253 / 180 / 180 / 90 | 0, 1, 2, 3, 4, 5, 7 | 23,968 / 28,736 | 8.06 |
| [Potted plant](potted-plant-result.json) | 8929 / 8036 / 8036 / 8036 / 2316 / 2276 / 2258 / 2258 | 0, 1, 4, 5, 6 | 0 / 37,644 | 8.98 |
| [Dead quiver trunk](dead-quiver-trunk-result.json) | 17978 / 4548 / 630 / 630 / 630 / 66 / 66 / 2 | 0, 1, 2, 5, 7 | 60,480 / 60,704 | 14.64 |
| [Painted shelves](smoke-shelves-result.json) | 524 / 376 / 376 / 36 | 0, 1, 3 | 2,752 / 3,507 | 0.13 |
| [Rock face](rock-face-result.json) | 29566 / 26609 / 26609 / 13304 / 1038 / 1038 / 126 / 63 | 0, 1, 3, 4, 6, 7 | 101,920 / 104,934 | 11.11 |

Moon rock shares source vertices through LOD 3. The new budget reservation
lets it reduce to 16 triangles at the 16 px slot; the preceding forward-budget
chain reused a 44-triangle mesh from 43.1 px. The older research screenshot
came from a separate forced rebuilt-vertex run and does not represent either
automatic chain. [The tail-priority pilot](TAIL.md) preserves raw previous runs,
including a probe setting that regressed the tree trunk before correction.

All eight current runs completed, passed their recorded source and adjacent
visual audits, and stayed within their added-vertex budgets. The generator
checks those properties, the final 16 px schedule, exported byte sizes,
runtime-level accounting, source hashes, and tail-probe diagnostics before
writing the embedded geometry. [The board manifest](../../examples/current-board/manifest.json)
records input and output SHA-256 hashes, source credits, settings paths,
budgets, and storage totals. Public Poly Haven input files match the hashes
in `research/pilot.json` and `research/corpus.json`; shelves uses the archived development trace input
documented in [SMOKE.md](SMOKE.md).

Rebuild the board with `node tools/current-board.mjs` after restoring the
source files listed in those catalogs. The generated HTML is one offline
file with embedded geometry and WebGL 2 previews.
