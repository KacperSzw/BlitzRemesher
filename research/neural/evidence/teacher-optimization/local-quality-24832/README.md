# Local quality after 24,832 updates

The five-minute local training soak changed LOD output, but did not demonstrate
consistent improvement. Under identical diagnostic settings, learned ranking
reduced shelves triangles slightly and retained substantially more moon-rock
triangles at every reduced LOD.

| Asset / ranking      | Initial LOD1 / LOD2 / LOD3 | Trained LOD1 / LOD2 / LOD3 |
| -------------------- | -------------------------- | -------------------------- |
| Shelves / learned    | 518 / 259 / 247            | 500 / 250 / 236            |
| Moon rock / learned  | 1862 / 932 / 930           | 3252 / 1626 / 1626         |
| Shelves / constant   | 524 / 262 / 244            | 493 / 246 / 230            |
| Moon rock / constant | 3222 / 1910 / 1902         | 3198 / 1918 / 1912         |

LOD0 remains 524 triangles for shelves and 3304 for moon rock. Every emitted
source and adjacent audit passed and completed: 24 measurements per model across
both rankings, with no unknown, cancelled, resource-limited or nonfinite result.
The source limits at reduced LODs are 2 / 3 / 3 pixels; adjacent limits are
2 / 2.5 / 3 pixels. Raw reports retain the individual measured errors and changed
areas. Passing these limits does not mean measured error is unchanged.

The learned shelves gains are 18 / 9 / 11 triangles. Learned moon rock regresses
by 1390 / 694 / 696 triangles; its final LOD retains the preceding LOD's 1626
triangles and records one fallback level. The trained constant-ranking control
beats trained learned ranking on all shelves LODs. Constant ranking still uses
model-dependent placement, so this control neither isolates all network behavior
nor supplies an independent classical baseline.

The diagnostic uses two development assets, four LODs at 128 to 32 pixels,
coverage audits with 16 cameras, and 64 action trials per proposal. Initial
learned proposals hit that trial limit 8 of 12 times; trained learned proposals
hit it 7 of 12 times. These are matched-budget observations, not proof that
reduction was exhausted. This one checkpoint comparison cannot establish the
minimum updates required for improvement or predict eventual convergence.
No aggregate SCORE, general quality, adoption or performance claim is made.

The [record](record.json) associates the trained model with the verified
step-24,832 checkpoint from the [local seed-admission soak](../seed-admission-local/record.json).
All six indexed checkpoint files were independently hashed. Binary checkpoint
payloads are not duplicated here; their hashes and the raw checkpoint index and
verification are preserved. Initial and final audits share executable, manifest,
visual-settings and execution-settings hashes. The local settings only change
the repository diagnostic's GPU memory budget from 16384 to 1024 MiB. The shared
workstation timings have no isolated performance authority.

The original failed reporting attempt is retained separately. Its native initial
audit exited successfully and produced valid four-LOD results, but the JavaScript
helper rejected equivalent asset identities because native JSON object keys were
ordered `category,id` instead of `id,category`. It did not run the trained model.
Commit `2246b919` corrected that comparison without changing native runtime code;
the fresh `audit-02` completed both models. Neither the original failure nor its
valid native output has been rewritten.

[manifest.json](manifest.json) hashes every archived file and the uncompressed
bytes of each raw gzip artifact. `comparison/` contains the successful raw
initial/final audits, reports and process logs; `original-helper-failure/`
contains the earlier failure; `inputs/`, `context/` and `training/` preserve
their contracts. Raw retained-ratio and timing summaries remain in the helper
report without being promoted to a score.
