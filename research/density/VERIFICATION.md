# Verification

Release CTest: **14/14 passed**, 1.89 s. ASan/UBSan CTest: **14/14 passed**,
8.95 s. Raw output is in [test-results](test-results/). The final regression
also passes against the production library. C ABI remains version 5.

The new tests cover:

- Indexed request density at multiple caps and source sizes, including unused
  source vertices, seam-heavy outputs, zero/tiny budgets and both search paths.
- Continuous corner fans, disconnected contacts, nonmanifold edges, seams,
  affine UV fitting, local foldover repair, discrete alpha and tangent signs.
- Complete byte-exact vertex keys across all streams, strided source data,
  immutable input, result/LOD ownership and exact endpoint coordinates.
- Changed-vertex accounting, nonconsecutive pool reuse, graph allocation
  histories, and physical glTF buffer sizes/accessor sharing. The controlled
  export occupies 584 bytes, independently derived from its streams and indices.

The connected indexed-mesh fixture fails before density targeting
([original failure](regression-before.txt)). The allocation-history fixture
fails with the frozen v4 graph object linked to the current library
([original failure](dominance-before.txt)); it passes with v5. Equal-cost paths
converge at a source-only mesh but retain different pools. Only one can afford
the final 4-triangle mesh. Both test caps permit one new vertex and forbid two.

The [sanitized audit-recovery check](audit-check.json) covers cancellation,
resource interruption, resolved rejection witnesses, continuation through later
levels, preserved auditor identity and rejection of changed export hashes.

Builds v1–v5 have frozen executable hashes and source archives. v5 and the
final tested CLI have identical executable bytes; the final source archive also
contains the allocation-history regression added after v5 was frozen.

The eight-asset v3 merged output is byte-identical to appearance-screen.
The complete v4 and v5 shared outputs are byte-identical to each other.
[Per-asset glTF and buffer hashes](export-equivalence.json) establish this;
timings and run identities remain separate. A completed independent audit is
shared only after checking those exact bytes, the visual contract, original
row/run hashes and audit rotation. Shared records link and hash their original
proof; their zero new audit seconds are not a timing measurement.

The new independent rotation is **0xA1172028**, 642 orthographic plus 64
perspective views, 8× through 32× supersampling, every source and adjacent
comparison at LOD1–7. The archived screen/ordering comparisons retain their
previous 0xA1172027 proof records. Their original directories are unchanged.

The [offline browser check](board-check.json) validates all recorded methods,
eight stable rows, exact geometry counts, stored normals, scheduled/runtime
slots, synchronized rotation, camera and display controls, actual target size,
mobile layout, and operation without network requests. The generated board
checks export byte totals and proof provenance before embedding any geometry.

Only development assets were used. The usefulness gate failed, so validation20,
held-out release data and paired performance repetitions were not run.
