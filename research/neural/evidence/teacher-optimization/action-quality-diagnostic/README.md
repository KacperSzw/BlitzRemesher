# Fixed-state ranking and placement diagnostic

The final one-hour model can lose useful edits through either ranking or
placement changes on the same frozen mesh. This small diagnostic separates the
two effects; it does not quantify their contribution to the full-chain regression
or establish a quality improvement.

Both complete v4 models receive identical features for 16 geometry-selected
edges per state. Their own decoded placements are copied before any trial.
Each placement is checked independently against the original source and a fixed
predecessor, without committing edits. The measured ledger is then ordered by
either model's scores to produce the four combinations. No output weights are
swapped between the models' different backbones.

The states are initializer LOD0 and LOD2 for painted wooden shelves and moon
rock. The pool contains eight current-plane choices and eight deterministic
random choices, rather than the whole runtime action set. Conditions use the
initializer's recorded rebuilt-proposal target at the next scheduled level.
The visual contract is unchanged: coverage, 12 orthographic and 4 perspective
audit cameras, supersampling 4 with maximum 8, and the scheduled source and
adjacent limits. Only the local GPU budget changes from 16384 to 1024 MiB.

Both runs completed all four states with unchanged geometry and no unknown
outcomes. For the final-model comparison, 128 placement attempts yielded 88
geometrically valid candidates, each audited against both references, and 40
geometric rejections. The following counts are positions of the first safe
edit in each counterfactual pool order, including earlier invalid edits. They
are not measured full-chain executor trial counts.

The raw `invalid_geometry` status means native `state.trial` rejected the
placement before visual auditing. That check also enforces UV and packing
constraints; this diagnostic does not identify which individual constraint failed.

| Frozen state                   | Initial rank / initial placement | Initial rank / final placement | Final rank / initial placement | Final rank / final placement |
| ------------------------------ | -------------------------------- | ------------------------------ | ------------------------------ | ---------------------------- |
| Shelves LOD0, 524 triangles    | 2                                | 3                              | 2                              | 2                            |
| Shelves LOD2, 259 triangles    | 3                                | 1                              | 2                              | 2                            |
| Moon rock LOD0, 3304 triangles | 1                                | 1                              | 1                              | 1                            |
| Moon rock LOD2, 932 triangles  | 1                                | 2                              | 2                              | 2                            |

Every listed first safe edit removes two triangles. At moon-rock LOD2, the
initializer ranks edge `1536 -> 1557` first and its placement is safe. Keeping
that ranking but using the final placement makes this edit geometrically
invalid. The final ranking instead puts `25 -> 1363` first, which is invalid
under both placements. Both models still have 11 safe placements somewhere
within that 16-edge pool. This identifies two local failures, without showing
that either explains the observed whole-chain triangle difference by itself.

Other states are mixed: shelves' safe-placement counts change from 7 to 9 at
LOD0 and from 10 to 8 at LOD2; moon rock changes from 16 to 15 at LOD0. These
pools and initializer-derived states do not cover the final policy's complete
trajectory or batch interactions. No aggregate SCORE, adoption, convergence,
or speedup claim is made.

The final model is associated with the independently verified step-413184
checkpoint; its capture records `duration_complete` and 3600000 learning
milliseconds. [record.json](record.json) preserves model, checkpoint, executable,
settings, and source-file identities. The diagnostic ran before commit
`6bb2dc9bfeda39ea562d1c5727eae24e828f3268`; every tested source-file hash was
subsequently checked against that exact commit. The original dirty-worktree
record remains in the raw context. No model or optimizer payload is copied here.

Validation includes the affected native build, the GPU action CTest (1/1),
and Compute Sanitizer memcheck with zero errors. A same-model smoke test using
the initializer twice produced exactly identical predictions, decoded
placements, audit results, and all four combination summaries. It also checked
all four meshes remained unchanged. Both CLI runs requested the Vulkan
validation layer and their logs contain no validation error patterns. Timings
are preserved solely as shared-workstation execution evidence.

From the repository root, the final comparison command was:

```sh
env -u DISPLAY -u WAYLAND_DISPLAY \
  LD_LIBRARY_PATH=/run/opengl-driver/lib:/home/kacper/Projects/BlitzRemesher-neural/.cache/libtorch/lib \
  VK_DRIVER_FILES=/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json \
  VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  timeout 180s build/neural/blitz-neural-diagnostics --audit-actions \
  research/neural/action-diagnostic.json \
  runs/neural/v4-initialization/model.blzn \
  runs/neural/teacher-optimization-06/final-checkpoint/model.blzn \
  runs/neural/seed-admission-quality-local-01/workspace/research/neural/teacher-capability-quality.json \
  runs/neural/teacher-optimization-06/final-action-audit/report.json
```

The command ran through the pinned `nix develop .#neural` environment. The
same-model command replaces the final model with the initializer and uses a
fresh output path. Check `report.complete` and each `geometry_unchanged` field;
successful process exit alone is insufficient.

[manifest.json](manifest.json) records archive and original-byte hashes.
`comparison/` and `same-model/` retain raw reports, contexts and logs;
`inputs/` contains the exact settings and asset manifest; `checkpoint/` holds
the index, verification and capture; `validation/` holds build and test logs.
The ten `snapshots/*.bin.gz` files are exact `BLZMREF2` mesh references, readable
with `training/mesh_cache.hpp::read_reference` after decompression. Both runs'
snapshots matched byte for byte, so each is stored once, with both original paths
listed in the manifest. They preserve original source, predecessor, and working
mesh separately, including all streams and exact-position flags.
