# Verification

Current source: [final build stamp](builds/final-source.json). The algorithm
measurements use the separately frozen v3 executable and source archive;
later changes correct uncapped metadata labels and audit/export diagnostics.
They do not alter the capped graph's reduction or selection.

- Release CTest: **10/10 passed**, 1.51 s. [Raw log](test-results/release.log).
- Address/undefined sanitizer CTest: **10/10 passed**, 8.08 s.
  [Raw log](test-results/sanitize.log).
- A separate sanitizer run of the audit tool verifies failing-witness
  completion, all scheduled levels, resume across an interrupted slot,
  per-level auditor provenance, and rejection of mismatched export hashes.
  [Raw check record](audit-check.json); reproduce with
  `node tools/chain-search-audit-check.mjs` after CTest.
- Graph fixtures cover equal-count alternate predecessors, independent source
  and transition constraints, nonmonotonic acceptance, exact and insufficient
  vertex budgets, immutable source data, compact predecessor borrowing,
  repeated-pass incumbent preservation, cancellation and allocation failure.
- A closed torus exercises actual topology relaxation at two test-owned work
  budgets. Relaxed calls consume the existing graph allowance. The initial
  scripted version of this fixture was invalid: custom proposers deliberately
  bypass built-in topology fallback. It was replaced with the real reducer
  fixture, retaining the positive relaxation and bounded-work assertions.
- Import/export tests cover retained scheduled thresholds after exact runtime
  deduplication, shared accessors, RGBA8 streams, stored audit settings and
  configuration validation. The C ABI remains version 5.
- Browser checks verify geometry counts/bounds, stored normal streams, stable
  asset order, synchronized rotation, scheduled/runtime slots, display/camera
  controls, actual target pixel size, mobile layout, and offline loading.
  [Machine-readable browser record](board-check.json).

The board builder also checks buffer bytes, source/step gates, original screen
thresholds, runtime storage/mappings, and independent-audit export/row hashes.
These checks validate export consistency; independent visual outcomes remain
separate in [DECISION.md](DECISION.md).

Stool and plant smoke runs preceded the frozen development pilot. The original
eight-asset pilot and twenty validation sources were restored and hash-verified.
No held-out assets were used. Smoke summaries, failed experiments, interrupted
work and earlier audit checkpoints remain available; incomplete cohorts receive
no aggregate score in the report.
