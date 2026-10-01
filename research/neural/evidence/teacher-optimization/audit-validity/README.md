# Invalid audit measurement admission regression

Commit `276c83c` makes teacher seed admission, final teacher confirmation,
action-label eligibility, and neural runtime confirmation share one numeric
validity contract. Synthetic passing measurements with NaN coverage or infinite
normal diagnostics could previously pass seed admission. Resource-limited
measurements also cannot pass merely because their `passed` flag remains set.

These are synthetic malformed-result regressions. They do not establish that
the completed one-hour training run received such measurements or that this
change improves model quality, reduction, or throughput.

A failure may legitimately have infinite distance or coverage: an empty render
against a nonempty reference is one example. The fix preserves that known
negative when an evaluated view witnesses a threshold violation. NaN
error/coverage, nonfinite changed area, and nonfinite normal diagnostics never
establish a label. A positive result must be complete, uncancelled, unrestricted
by resources, and finite in every measured field.

The expanded seed regression failed under the old header with
`unknown seed audit became an accepted initial state` (exit 1), then passed
under the corrected header (exit 0). The same test covers each of three seed
gates independently. Focused action and seed CTests passed 2/2 with no skips in
both the Release and ASan/UBSan builds. Action tests retain a witnessed infinite
negative and test independently corrupted, otherwise finite positive results.

[record.json](record.json) records exact commits, source and executable hashes,
original standalone compiler commands, and the observed test results.
`reproduction/` reconstructs the pre-fix header from `e4613db` and the test,
corrected header, and shared measurement helper from `276c83c`. Those bytes were
retrieved from Git after testing; they are not claimed to be separately saved
source snapshots from the original invocation. The original before/after
executables are identified by hash but are not copied into this archive.

The supplied test includes `training/teacher_seed.hpp`; for an independent
reproduction, arrange the selected archived header at that include path and the
corrected helper at `neural/audit_measurement.hpp`. The recorded original
commands ran in the repository while those corresponding versions were present.
No GPU or native model inference is required for these tests.

[manifest.json](manifest.json) checks every archived artifact and the Git-derived
source provenance. Raw before/after and build/CTest logs remain unchanged. No
model, optimizer, private credential, or new training data is included.
