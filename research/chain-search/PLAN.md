# Whole-chain search experiment

The primary contract is eight scheduled levels at 512 to 16 px, a 3 px source
cap, constant 2 px transitions, maximum changed area 0.5, a 20% added packed
vertex cap, and zero triangle overhead. Coverage and attributed appearance
are separate scenarios. The strict coverage scenario uses a 2 px source cap,
1 px transitions and maximum changed area 0.25. The cumulative source policy
still applies beneath the cap. The frozen eight-asset development pilot is the
scored cohort; the existing 13 examples remain a separate gallery.

Hypothesis: source-admissible meshes can be combined through independently
audited transitions to reduce the equally weighted LOD1–7 triangle total.
The current source-to-tail screen, early vertex reservation and narrow
history-dependent proposals leave legal chains unexplored. No optimality or
all-view guarantee is claimed.

Keep a complete incumbent throughout graph exploration. Candidates share
immutable owned payloads or borrow LOD0 streams. Audit source admission and
incoming edges separately. Bound each layer to twice the configured beam in
additional mesh candidates, plus source/incumbent; split admission between
triangle count and added bytes. Test at most four incoming predecessor meshes
per candidate. Retain multiple nondominated histories at identical endpoints,
and cap active paths at four times the beam plus source/incumbent. Vertex
bytes are charged according to exported distinct consecutive runtime meshes.
Pin the incumbent through cancellation and allocation failures. Exact identity
and previously audited incumbent edges can reuse their proven measurements.

One graph pass adds at most candidate_budget reducer calls per transition.
Run the incumbent alone, one additional pass, and three additional passes as
nominal 1x, 2x and 4x proposal tiers. The incumbent's tail probes and adaptive
retry are counted; these tier labels are not claims about elapsed time or
equal total audit work. Also compare the legacy search at candidate_budget=32.
Topology relaxation is a separate ablation, replacing calls inside the graph
proposal budget. The initial incumbent always uses the original quadric path.

Freeze binaries, stamps, inputs, settings, and raw runs. Restore the original
pilot sources and verify all hashes before scoring. All eight original inputs
were restored with blitz-corpus's frozen replay. The baseline is revision
2669dba; its executable and source archive are in build/chain-search/baseline.
The stamp is builds/baseline.json. Keep incomplete and negative results.

Start with deterministic graph/ownership/error regressions and focused tests,
then stool and plant smoke cases. Run the complete frozen pilot after a smoke
signal. Promotion requires at least 10% lower category-balanced mean chain
retention, no asset chain-total regression, passing visual/storage gates, and
at most 4x matched bake time. Use paired timings before making a performance
claim. Escalate a qualifying pilot to the 20-asset validation split. Independently
audit delivered reduced levels on rotated 642+64 cameras, 8x refined to 32x.
Held-out assets are reserved for release qualification.

The board must keep presets, asset order, presentation cameras and source
framing fixed; expose actual source/adjacent distance and area, runtime bytes,
and triangle-versus-size and triangle-versus-bake-time curves. Appearance
qualification requires its own profile and independent completed audit.
Geometry coverage alone does not qualify shading or texture-image fidelity.

## Development observations

- v1 passed all ten release test targets before its stool smoke.
- v1 stool: chain total 13,306 to 10,831 (18.60% lower), but 94.61 s generation
  versus 8.23 s baseline (11.49x). This fails the time criterion. The run is
  preserved in ../runs/chain-search-smoke-v1; it is not a pilot SCORE.
- v2 removes repeated audits of exact incumbent transitions and identity
  edges, and skips paths whose accumulated triangle count already exceeds the
  complete incumbent. That bound assumes zero overhead and zero future cost;
  it never guesses a monotone relation between target triangles and pixel error.
- v3 rejects vertex-budget-infeasible incoming edges before spending their audit
  work, and reuses an already proven source check when source and transition
  limits are identical. The primary one-pass pilot retains the same geometry
  as v2, with less redundant work. Both frozen builds remain recorded.
- A later export-label correction reports the whole-chain objective for the
  uncapped legacy mode. It changes no selection or geometry. The capped v3
  experiments are unaffected; their original executable and stamps remain frozen.
- An interrupted legacy-2 attempt is archived separately after a worker-limit
  correction. Its timings and incomplete rows are excluded from comparisons.
- Independent checks use rotation seed `0xA1172026`. The evaluator stops at a
  failing witness with `Measurement.complete=false`. The full-chain audit tool
  distinguishes that resolved failure from time/resource interruption and
  continues to subsequent scheduled levels. Earlier checkpoints are retained
  and resumed only after all run/export/row/audit hashes match.
