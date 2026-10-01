# Current-policy example board

[Open the self-contained interactive board](../../examples/current-board/index.html).
It shows 13 audited development chains: six nature meshes, two rocks, four
manufactured meshes, and a mixed potted plant. Every chain has a final 16 px
scheduled slot. Runtime view hides exact consecutive duplicate meshes; an
earlier runtime mesh can serve a later scheduled slot.

All runs use a 20% added-vertex cap, zero triangle overhead, coverage profile,
eight proposals per transition, six orthographic plus two perspective search
views, and twelve orthographic plus four perspective audit views. Automatic
generation also uses up to eight direct tail probes per search pass. Twelve
assets use eight levels at 512→16 px; Painted wooden shelves uses the archived
four-level 64→16 px smoke setting. These examples are not a corpus score or a
default-quality audit.

| Asset | Scheduled triangles | Runtime slots | Added / allowed vertex bytes | Bake seconds | Adaptive retry selected |
|---|---|---|---:|---:|---|
| [Moon rock](moon-rock-result.json) | 3304 / 2974 / 1488 / 744 / 108 / 36 / 36 / 16 | 0, 1, 2, 3, 4, 5, 7 | 11,360 / 11,820 | 10.59 | — |
| [Bermuda grass](grass-bermuda-result.json) | 941 / 941 / 846 / 423 / 211 / 41 / 13 / 13 | 0, 2, 3, 4, 5, 6 | 3,712 / 5,382 | 4.46 | — |
| [Lambis shell](lambis-shell-result.json) | 12498 / 11248 / 11248 / 5624 / 430 / 430 / 430 / 62 | 0, 1, 3, 4, 7 | 43,680 / 47,110 | 7.87 | — |
| [Metal stool](metal-stool-result.json) | 6532 / 5878 / 2939 / 1469 / 1253 / 180 / 180 / 90 | 0, 1, 2, 3, 4, 5, 7 | 23,968 / 28,736 | 8.10 | — |
| [Potted plant](potted-plant-result.json) | 8929 / 7861 / 6878 / 6878 / 2274 / 2256 / 2256 / 392 | 0, 1, 2, 4, 5, 7 | 17,216 / 37,644 | 18.63 | yes |
| [Dead quiver trunk](dead-quiver-trunk-result.json) | 17978 / 4548 / 630 / 630 / 630 / 66 / 66 / 2 | 0, 1, 2, 5, 7 | 60,480 / 60,704 | 14.43 | — |
| [Painted wooden shelves](smoke-shelves-result.json) | 524 / 376 / 376 / 36 | 0, 1, 3 | 2,752 / 3,507 | 0.13 | — |
| [Rock face](rock-face-result.json) | 29566 / 26609 / 26609 / 13304 / 1038 / 1038 / 126 / 63 | 0, 1, 3, 4, 6, 7 | 101,920 / 104,934 | 11.21 | — |
| [Gazania flower](flower-gazania-result.json) | 14340 / 10898 / 9535 / 9535 / 7151 / 572 / 572 / 429 | 0, 1, 2, 4, 5, 7 | 92,808 / 114,314 | 11.39 | yes |
| [Dry branches](dry-branches-result.json) | 16803 / 15123 / 15123 / 7561 / 717 / 717 / 81 / 41 | 0, 1, 3, 4, 6, 7 | 72,512 / 72,870 | 9.83 | — |
| [Apple](apple-result.json) | 7012 / 6310 / 6310 / 6310 / 202 / 202 / 202 / 70 | 0, 1, 4, 7 | 24,224 / 25,721 | 6.99 | — |
| [Classic laptop](classic-laptop-result.json) | 14450 / 13005 / 8360 / 816 / 240 / 240 / 240 / 91 | 0, 1, 2, 3, 4, 7 | 79,552 / 84,364 | 14.93 | — |
| [Picnic table](picnic-table-result.json) | 10210 / 9189 / 4594 / 2297 / 269 / 269 / 269 / 134 | 0, 1, 2, 3, 4, 7 | 30,528 / 43,571 | 16.42 | — |

Potted Plant previously repeated its 2,258-triangle LOD 6 at the final slot,
using none of its 37,644-byte allowance. The 17–356-triangle direct tail probes
failed the conservative 3 px source-path screen, while a 712-triangle probe
exceeded the byte cap. The final-level 392-triangle direct proposal failed its
adjacent search. A second pass with adaptive targets found a 392-triangle
progressive mesh whose source and adjacent full-audit errors are both 2.29 px,
within the 6.89 px source and 3 px transition limits. It uses 17,216 bytes;
the local bake took 18.63 s, versus 8.98 s in the earlier run. Gazania flower
also improved from a 572-triangle baseline tail to 429 triangles after retry.
The retry kept the same limits and combined both audited finalist pools for
selection. Moon rock and Bermuda grass kept their original chains.

All 13 current runs completed, passed recorded source and adjacent audits,
and stayed within their vertex budgets. The board generator verifies those
properties, the 16 px final schedule, exported byte sizes, runtime storage,
and input hashes before embedding geometry. The [manifest](../../examples/current-board/manifest.json)
records source credits, settings, hashes, budgets and selected retry flags.
The five new models use the frozen development catalog and the same 512→16 px
config as the other full-length examples. Timings are single shared-workstation
observations, not matched performance scores.

To reproduce the board from the frozen catalog:

```sh
node tools/restore-current-board-assets.mjs
nix develop . -c cmake --build --preset release
node tools/rebuild-current-board.mjs
```

The generated HTML contains its geometry and previews in one offline file.
