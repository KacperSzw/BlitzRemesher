# Precision experiments, protocol v2

Fixed hypotheses, before precision comparisons:

- Linear RGBA8 cuts owned vertex color storage from 16 to four bytes. This is
  the accepted input contract, not an algorithmic score improvement over v1.
- Exact coverage masks remove unused depth/attribute work and reduce raster
  memory. Full and packed runs must produce identical outputs and measurements.
- Float persistent quadrics (40 bytes) may reduce working-set cost; coefficients
  are promoted before edge evaluation/solving. Float candidate costs (24 bytes)
  are an independent experiment. Neither option is enabled by default.
- Smaller storage may not improve total generation time; retain negative results.

The five builds are full raster/double state, packed raster/double state,
packed/float quadrics, packed/float candidate costs, and packed/both. All other
source, settings, cameras and seeds are fixed. Each executable has its own
hash-checked build stamp. No original source files or held-out inputs are changed.

Scenarios use existing frozen inputs/configs:

- Eight development pilot assets, `pilot-qem.json`: compare all five builds.
- All 20 validation assets, the same config: compare all five builds.
- Pilot normal and attribute profiles: check all precision variants separately.
- First two pilot assets at `s512-cap8.json`: repeated full/packed comparison,
  explicitly a two-asset 512→16 scenario, not the eight-asset 32→16 score.
- Rerun meshoptimizer, Fast Quadric and all three CGAL policies on the v2 pilot.
  Unsupported/manifold failures stay in the denominator.

Initial pilot runs are exploratory timings and may overlap compilation. After
build verification, use three interleaved timing repetitions (2, 3, 4), reversing
variant order for repeat 3. The kernel fixture uses one warmup and seven measured
samples, alternating full/packed order. Report raw samples and shared-workstation
noise. No timing from an incomplete run has an aggregate score.

Use `tools/precision-research.sh` inside the pinned Nix shell. Individual benchmark
batches checkpoint within 50 minutes; limit all active compilation/benchmark work
to four CPU workers and 24 GiB. Resume existing rows only under matching hashes.
Generated meshes are ignored; raw rows, stamps, summaries and negative findings
are retained. `summarize.cpp` checks full/packed equality, canonical input hashes
and every delivered visual gate while reporting time, RSS, SCORE and tail ratios.
