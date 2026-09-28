# Precision experiment archive

Production has one precision configuration: float32 geometry, RGBA8 colors,
double quadrics/candidate costs, and packed coverage masks. The full rasterizer
serves normal/attribute profiles and reference tests. The former float storage
options and full-raster coverage switch are no longer production choices.

## Preserved evidence

- [Original plan](PLAN.md), [report](REPORT.md), [summary](summary.json),
  [raw v2 runs](../runs/precision-v2/), and [build stamps](builds/) retain all
  five variants, failures, repeated samples and negative results.
- [UNORM16 probe](unorm16/REPORT.md) records why positions remain float32.
- [Production replay](production/REPORT.md) compares the cleaned implementation
  with the frozen packed/double reference. New binaries, stamps and runs live
  in separate directories; historical measurements are not overwritten.

## Restoring the removed experiments

The [restoration patch](restore-experiments.patch) restores the five changed
source/build files, including the old research runner. It recovers the
experimental source tree with SHA-256
`e125d42d7d199d1cb90a8cda13ae098f69500ea8729ee97e5995cb5a2d57f9f4`.
It is an archive artifact and is not compiled into production.

Apply it in a separate checkout/worktree of this revision. The original data
files must be present and match the manifest. Use the pinned Nix shell, fresh
build/run paths, and explicit per-executable stamps. For example:

```sh
git apply --check research/precision/restore-experiments.patch
git apply research/precision/restore-experiments.patch
cmake -S . -B build/reproduced-quadrics -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBLITZ_PACKED_COVERAGE=ON -DBLITZ_FLOAT_QUADRICS=ON -DBLITZ_FLOAT_COSTS=OFF
cmake --build build/reproduced-quadrics -j2
bash tools/stamp-build.sh build/reproduced-quadrics/build.json build/reproduced-quadrics/blitz
build/reproduced-quadrics/blitz bench research/pilot.json research/configs/pilot-qem.json \
  research/runs/reproduced-quadrics --build-stamp build/reproduced-quadrics/build.json --minutes 50
```

For the other historical variants, use the switch combinations in the original
plan. Keep the aggregate limit of four CPU workers and 24 GiB, and preserve
the archived run/stamp files. A new compiler, binary or source revision needs
a new run identity even when its numerical output agrees.
