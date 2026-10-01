# BlitzRemesher

BlitzRemesher is a C++20 neural mesh reducer for static, opaque game assets.
It generates LOD chains with fewer triangles under configurable screen-space
limits against both the original mesh and the preceding emitted LOD.

The network proposes legal contractions and vertex placements. Independent
visual audits decide which proposals may enter the chain. Triangle reduction is
the primary objective; model confidence never replaces an audit.

**Status:** the neural system is the project's core direction. Coverage learning
and engineering validation are in progress; broad superiority over classical
reducers and release-quality generalization have not been established. Finite
camera audits do not establish all-view bounds or a global optimum.

## Generate an LOD chain

```sh
build/neural/blitz simplify model.glb \
  --neural-model /path/to/verified-model.blzn \
  --config configs/neural-coverage.json \
  --raster-backend vulkan --vertex-storage packed \
  --neural-origin source --preserve-uv on \
  --out output/model
```

Use an explicitly selected, checksummed checkpoint; a pretrained release model
is not silently downloaded. The example uses the established source-origin path.
`--neural-origin previous` and `both` expose the new predecessor-origin
comparisons under the same candidate budgets. Their quality advantage must be
measured on the chosen workload.

The output contains `chain.gltf`, `chain.bin`, and `lods.json`, including scheduled
screen sizes, per-level triangle counts, source/transition errors and completion
status. Exact consecutive duplicate LODs share runtime geometry. Unreduced
fallbacks remain visible.

| Control | Meaning |
| --- | --- |
| `levels` | Scheduled levels including unchanged LOD0; duplicate geometry does not add a distinct runtime mesh |
| `base_pixels`, `last_pixels` | Beginning and minimum target screen sizes |
| `transition` | Maximum pixel-error curve between consecutive emitted LODs |
| `max_lod0_delta_px` | Independent maximum pixel error against LOD0 |
| `profile` | `coverage`, `normals`, or `attributes`; normal/attribute weights are configurable |
| `max_changed_area` | Optional per-view coverage-area constraint in addition to pixel distance |
| `--preserve-uv on/off` | Chart/foldover preservation, or finite best-effort UVs with relaxed UV-specific restrictions |
| `--action-trials`, `--action-batch` | Bounded neural search work; reaching a cap is reported separately from exhaustion |

The neural coverage preset sets triangle overhead to zero and disables the classical
added-vertex cap (`max_added_vertex_bytes_bps: null`), keeping triangle reduction
primary across the whole chain. UV relaxation requires
an explicitly UV-conditioned v4 policy; older models retain strict UV behavior.
UV preservation does not score texture images or normal maps. Normals and
attributes are optional quality requirements, not supervision supplied by a
coverage-only checkpoint.

## Build and dependencies

On the supplied NixOS workstation:

```sh
nix develop .#neural
cmake -S . -B build/neural -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBLITZ_CUDA=ON -DBLITZ_VULKAN=ON
cmake --build build/neural -j2
ctest --test-dir build/neural --output-on-failure
```

The neural backend currently requires NVIDIA CUDA; Vulkan provides the packed
hardware audit path. A runtime build does not require LibTorch. Enable
`BLITZ_NEURAL_TRAIN` and set `BLITZ_LIBTORCH_ROOT` only to build the C++ trainer.
The pinned distribution and setup are described in [dependencies](third_party/README.md).

Component switches separate the CLI (`BLITZ_TOOLS`), neural diagnostics
(`BLITZ_NEURAL_TOOLS`), training (`BLITZ_NEURAL_TRAIN`) and corpus downloads
(`BLITZ_ACQUISITION`). Acquisition is off by default and is the only component
requiring CURL and LibArchive. A portable core build remains available:

```sh
nix develop .
cmake -S . -B build/core -DBLITZ_TOOLS=OFF -DBLITZ_ACQUISITION=OFF
cmake --build build/core -j2
ctest --test-dir build/core --output-on-failure
```

The installed C and C++ interfaces preserve explicit ownership and borrowed
read-only streams. The versioned neural C interface exposes backend selection,
work budgets and diagnostics alongside the ABI-5 mesh/settings/storage descriptors.
C and C++ clients must rebuild for the current interfaces.
Missing neural hardware is an explicit unavailable result.

## Classical compatibility

The classical generator and its C ABI remain available for existing integrations,
teacher construction and independent controls. Its default added-vertex cap is
20% of source vertex bytes, with zero triangle overhead. Under that cap, it
prioritizes the final LOD, then earlier levels in reverse order. Setting the cap
to `null` restores whole-chain triangle minimization; adding 500 basis points of
triangle overhead restores the earlier resident-byte selection policy.

Opt-in bounded graph search, appearance fitting and shared rebuilt vertex pools
retain their upstream contracts. Every accepted chain still passes source and
adjacent audits. The [classical guide](docs/CLASSICAL.md) documents these controls,
ABI-5 ownership/storage queries, import/export behavior and reproduction commands.
The [whole-chain board](examples/reduction-board/index.html),
[appearance board](examples/appearance-board/index.html) and
[rebuilt-storage board](examples/density-board/index.html) retain the classical
measurements and failures; they are not evidence of neural model superiority.

## Development and evidence

- [Current neural architecture and validation](docs/NEURAL.md)
- [Accepted behavior](docs/SPEC.md) and [comparison protocol](research/PROTOCOL.md)
- [Neural evidence index](research/neural/README.md)
- [Classical compatibility and comparison tools](docs/CLASSICAL.md)

Training and inference are C++; orchestration is JavaScript and shell. Runtime
code, training, executable entry points, scripts and experimental evidence have
separate directories and build dependencies. Classical algorithms remain
available as teachers, compatibility APIs and independently reported controls.

Inputs are glTF/GLB, OBJ, PLY and STL. Skinning, morphs, alpha-textured geometry,
texture-image scoring and engine-specific plugins remain outside the current
contract. Original code is MIT OR Apache-2.0; asset provenance and dependency
licenses remain recorded separately.
