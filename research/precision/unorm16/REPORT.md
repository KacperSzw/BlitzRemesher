# UNORM16 position feasibility probe

**Decision: retain float32 positions.** This was a small planning experiment,
not a scored corpus comparison or an integrated production feature.

Recorded 2026-09-28 with Clang 21.1.8 in the pinned Nix shell. The immutable
reference was the packed/double executable's source tree recorded in
[the raw probe](unorm16-probe.json). The experiment completed in 48.26 seconds.

## Method

The three development meshes were Moon Rock 02 (3,304 triangles), Metal Stool
02 (6,532) and Dead Quiver Trunk (17,978). Source paths/checksums are retained
in the raw data. Original source streams were checked for byte-exact equality
after the probe.

- Encode each coordinate as nearest-rounded UNORM16 relative to its own mesh
  AABB, with constant axes encoded as zero. Three uint16 components take six
  bytes, compared with 12 for float3; world-space bounds need another 24 bytes.
  Decode into float positions for the existing evaluator.
- Test single-snap geometry and projected vertex error at 512 pixels across
  12 orthographic + 4 perspective cameras. Coverage uses 4× supersampling,
  refinement up to 8×, and the existing 2-pixel limit. Attribute checks use
  128 pixels and 6 + 2 cameras with the same limit and default weights.
- Compare eight-level 32→16 coverage chains with unchanged pilot settings:
  ordinary float reduction, quantization of each owned proposal's output,
  and a prototype with six-byte normalized working positions. All proposals
  still pass through source and adjacent gates; borrowed fallbacks stay exact.
- The working-position prototype computes a fixed per-axis AABB in the
  reducer's normalized coordinates. Reads decode to float; initial positions
  and accepted placements encode to uint16. Solve/midpoint placements snap
  before cost and geometry checks, and out-of-bounds proposals clamp to that
  AABB. Quadrics and costs remain double. Input/output streams remain float,
  so only the internal position array receives the storage saving.
- Reducer timings use one warmup and seven alternating float/UNORM16 samples
  at the same half-source triangle target. A separate 256×256 curved grid
  has 66,049 vertices and 131,072 triangles. This is a straightforward codec
  prototype, not evidence about every possible optimized quantized reducer.

With exact arithmetic, nearest rounding has maximum Euclidean error equal
to the AABB diagonal divided by 131,070; float reconstruction adds rounding.
This single-snap bound does not bound the accumulated effects of repeated
quantized collapse choices.

## Results

| Fixture | Float median | UNORM16 median | Extra reducer time |
|---|---:|---:|---:|
| Moon Rock 02 | 3.57 ms | 4.34 ms | 21.6% |
| Metal Stool 02 | 6.73 ms | 8.06 ms | 19.9% |
| Dead Quiver Trunk | 35.11 ms | 42.02 ms | 19.7% |
| Curved grid | 131.66 ms | 165.89 ms | 26.0% |

Each float/UNORM16 pair reached the same triangle count. The trunk reached
8,990 triangles against a requested 8,989 in both variants. Some collapse
choices differ. Timing ranges do not overlap in these samples. No nonfinite
solve/cost event occurred. These are reducer timings on one workstation;
six-byte storage alone does not imply faster complete generation.

Single-snap projected vertex movement stayed below 0.00556 pixels at the
512-pixel test size. All three coverage checks passed; maximum changed
coverage was 0.02136%. No real test mesh gained degenerate/flipped triangles
or merged distinct positions from this one snap.

The 128-pixel attribute check passed for the rock and rejected the stool and
trunk. A [follow-up isolation check](unorm16-appearance-check.json) reproduced
both rejections with normal-only weights at both 4× and 8× supersampling.
Material-only checks passed. These were early appearance-gate rejections,
not cancellation or memory limits; `complete=false` records the stopped
camera sweep. Small coordinate changes can change sampled visible normals.
The 8× diagnostic did not replace or weaken the original 4× check.

All nine small coverage chains completed with passing delivered gates. Final
and last-three retained ratios were equal within each chain in this probe:

| Mesh | Float | Quantized proposal output | UNORM16 working positions |
|---|---:|---:|---:|
| Moon Rock 02 | 11.62228% | 11.62228% | 11.80387% |
| Metal Stool 02 | 32.40968% | 32.40968% | 32.59339% |
| Dead Quiver Trunk | 4.41651% | 4.41651% | 4.41651% |

Passing these small coverage chains does not establish appearance preservation.
The source-quantization appearance failures remain explicit negative findings.

The adverse fixture adds a tiny triangle at approximately (0.123454, 0.333333,
0.015) with 0.000001 offsets in x/y to a unit-scale grid. Quantization merges
its three distinct vertices and creates one degenerate triangle: distinct
positions fall from seven to five. A mesh-wide grid can therefore erase
features below its step size before reduction.

The measured CPU penalty, small working-set saving, and appearance/degeneracy
failures do not justify adopting UNORM16 positions in this production path.
No production source file or supplied source mesh was changed by the probe.
