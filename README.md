# BlitzRemesher

C++20 library and command-line tool for static, opaque game-mesh LOD chains.
It searches for fewer triangles under explicit screen-space transition and
LOD0 error limits. Every returned reduction passes the configured camera
audit. This is an early research implementation, not a claim of universal
view coverage or globally optimal simplification.

## Build

On the supplied NixOS workstation:

    nix develop path:.
    cmake --preset release
    cmake --build --preset release
    ctest --preset release

Elsewhere, use CMake 3.20+, a C++20 compiler and the dependencies described in
[third_party/README.md](third_party/README.md). A library-only build needs no
third-party runtime libraries:

    cmake -S . -B build/core -DBLITZ_TOOLS=OFF -DBUILD_TESTING=OFF
    cmake --build build/core

## Generate a chain

    build/release/blitz defaults > settings.json
    build/release/blitz info model.glb
    build/release/blitz simplify model.glb --config settings.json --out output/model

The quality defaults audit 642 orthographic and 64 perspective cameras at
8× supersampling, refining uncertain coverage to 16×/32×. This can be
expensive. [The small pilot configuration](research/configs/pilot-coupled.json)
is useful for a quick first run; it has a different, much smaller camera
contract and must not be described as the default quality audit.

Inputs: glTF/GLB, OBJ, triangulated ASCII/binary PLY, ASCII/binary STL.
glTF scene transforms, mirrored winding, strided/sparse accessors and
multiple material primitives are handled. Alpha materials, animation,
skinning, morph targets and compressed mesh extensions are rejected.
Only the first UV and vertex-color sets are imported. RGB streams are
interpreted as linear.

Output contains chain.gltf, chain.bin and lods.json. The default glTF scene
contains LOD0; other LOD meshes are identified in the manifest. Reuse mode
shares original vertex accessors and writes new index accessors.
For the CLI, original means the decoded mesh after import transforms;
compressed or quantized source-file bytes are not preserved verbatim.
The diagnostic exporter writes placeholder materials with stable material
IDs and double-sided flags. It does not copy texture images or the original
material graph. Engine integrations retain their own material payloads.

## Controls

- Output: rebuild or reuse. Reuse never changes source vertex bytes or IDs;
  every output index references an original vertex.
- Chain: direct, progressive, or hybrid. Hybrid retains a bounded beam of
  valid paths and chooses the smallest total triangle count it found.
- Profiles: coverage, normals, attributes. Coverage measures maximum
  foreground displacement, not changed pixel area. Appearance weights
  support curves and can be zero.
- Scale: pixels_per_meter × source bounding-sphere diameter × meters_per_unit,
  or explicit base_pixels. Screen sizes decrease geometrically to last_pixels.
- Limits: transition is a piecewise-linear pixel curve; max_lod0_delta_px is
  an optional source cap. The cumulative source policy is checked directly,
  rather than treated as a mathematical composition guarantee.
- coupled_wedges (enabled by default) enables synchronized positional reduction across separate
  attribute wedges in rebuild mode. Original UV charts and attributes stay
  separate. Reuse uses endpoint reduction.
- Candidate budget, beam width, camera sets, supersampling, pruning and
  scalar/AVX2 dispatch are explicit settings.

LOD0 is always unchanged. Cancellation returns an exact, validated source
chain with cancelled status; the implementation does not yet preserve a
partially optimized prefix on cancellation. Success means the configured
search and audit finished, not that no better reduction exists.

## Engine integration

The C++ entry point is blitz::generate(MeshView, Settings). MeshView borrows
strided streams; Result owns rebuilt vertices and output indices. Keep the
source alive and unchanged until all result views are finished.

The versioned [C API](include/blitz/blitz.h) exposes plain descriptors,
explicit status/error buffers and an opaque result handle:

1. Initialize blitz_settings with blitz_settings_init.
2. Fill blitz_mesh with float streams and aligned uint32 indices.
3. Call blitz_generate.
4. Read each LOD with blitz_result_lod.
5. Call blitz_result_destroy exactly once.

Source storage is never freed by the library. Returned views are read-only,
borrow result/source storage and expire at result destruction. No exception
or STL object crosses the C ABI. Cancellation callbacks must not throw or
mutate the source. The library creates no background threads.

## Corpus and research

The frozen [corpus manifest](research/corpus.json) records 120 assets,
checksums, provenance, item-level scan rights and grouped 80/20/20 splits.
Meshes are stored in data and excluded from Git. Replay the exact inputs:

    build/release/blitz-corpus data research/corpus.json
    build/release/blitz corpus-check data/manifest.json research/corpus-check.json

The acquisition manifest in data is written by catalog acquisition
(blitz-corpus data); frozen replay verifies/downloads files without rewriting
the frozen manifest. On a fresh checkout, use research/corpus.json as the
input to corpus-check instead of data/manifest.json.

Before freezing, category mismatches and a 17-million-triangle tree were
replaced. The corpus resource cap is two million source triangles per asset;
these decisions precede all scored comparisons. Original rejected download
records remain in the acquisition manifest. No scored asset is removed to
improve a score.

    bash tools/stamp-build.sh
    build/release/blitz bench research/pilot.json research/configs/pilot-coupled.json research/runs/example
    build/release/blitz-microbench

Runs checkpoint per asset. Resume requires identical executable,
configuration, camera, protocol and source hashes. Batches stop at 50 minutes.
Do not run more than four CPU workers or exceed 24 GiB combined memory.
The runner is sequential; independent batches may run concurrently within
those limits.

External references are separate executables:

    cmake --preset research
    cmake --build --preset research
    build/release/blitz bench research/pilot.json research/configs/pilot-qem.json research/runs/meshopt --baseline meshopt

Baseline names: meshopt, fastquadric, cgal-lt, cgal-qem, cgal-probabilistic.
The initial adapters compare single-material geometry under the coverage
profile. Unsupported inputs/capabilities are recorded as failures with
unchanged-chain score contribution; they are not silently repaired.
CGAL and Fast Quadric adapters support rebuild only.

Read [the algorithm review](research/ALGORITHMS.md),
[the scoring protocol](research/PROTOCOL.md), and
[the accepted specification](docs/SPEC.md) before modifying research rules.
Measured runs, including negative findings, are in
[RESULTS.md](research/RESULTS.md) and [EXPERIMENTS.md](research/EXPERIMENTS.md).
The [visual board](research/board/index.html) shows four actual eight-level
LOD chains with rotation, wireframe, silhouette and source/LOD comparison.
It opens offline without a server. [PNG](research/board/board.png) and
[PDF](research/board/board.pdf) versions are available for sharing.
See [board reproduction and provenance](research/board/README.md).

## Validation

    cmake --preset sanitize -DBLITZ_FUZZ=ON
    cmake --build --preset sanitize
    ctest --preset sanitize
    build/sanitize/blitz-fuzz tests/fuzz-corpus -max_total_time=60 -seed=2971082790

Tests cover the C ABI, strided immutable reuse, scheduling, direct/source
gates, zero weights, empty raster behavior, brute-force Hausdorff reference,
scalar/SIMD agreement, degenerate input, coupled wedges, cancellation and
import/export transforms. The fuzzer instruments the actual reducer.
See [the validation record](research/VALIDATION.md) for compiler, sanitizer,
installed-package and corpus results, with their limits.

The native library is MIT OR Apache-2.0. Optional CGAL benchmark source is
GPL-3.0-or-later; downloaded assets and vendored dependencies have their own
licenses.

Installed CMake consumers use find_package(BlitzRemesher CONFIG REQUIRED)
and link Blitz::remesher. BUILD_SHARED_LIBS=ON builds shared libraries.
The C++ package propagates its C++20 requirement to consumers.
