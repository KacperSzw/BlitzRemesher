# Verification

The final full release suite passed **11/11** (1.55 s); the final full ASan/UBSan
suite passed **11/11** (7.96 s, shared workstation). Raw logs are in test-results/.
The appearance fixtures use controlled settings and independently computed
field residuals, and cover equivalent retriangulations, genuine attribute loss,
nonmonotonic sampling, strided/source ownership, zero active fields, singular
planar neighborhoods, normal normalization, RGB rounding, alpha and tangent
handedness, disconnected constant-color wedges, cancellation, resource limits,
and witness coordinates beyond 16-bit range in caller-owned rasters.
Existing graph, cache, C ABI and exporter contracts remain covered.

The final suites include a cancellation fix that distinguishes interruption inside the attribute
sample loop from an infinite-error rejection witness, including a callback that
returns true only once. Chain generation latches that signal; the audit CLI
leaves it incomplete. The sanitized audit recovery fixture also verifies that
cancelled records remain unresolved, alongside resource, resume and hash checks.
These changes do not change uncancelled reduction or audit decisions.
The frozen qem-v1 pilot executable predates the completion-status correction and
the diagnostic witness layout changes; builds/final-source.json stamps the tested build.

The offline browser check covers all six recorded methods, stable eight-asset
rows, exact exported geometry counts, scheduled/runtime mappings, stored normals,
matched rotations, camera/display controls, actual target size, mobile layout,
and loading without network requests. The board is generated from raw exports;
its proof status is checked against run, glTF, buffer and row hashes.

Independent audits may share a completed proof only when glTF and buffer bytes
are identical and the complete visual contract matches. Each shared proof links
and hashes the original measurement; it is not reported as a new timed audit.
Older failing shelves and Moon Rock views are retained in regressions/.
No held-out assets or validation assets have been used in this experiment.
