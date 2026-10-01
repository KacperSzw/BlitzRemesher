# Third remote teacher optimization attempt

The existing A40 rental `teacher-optimization-04` was retargeted from `c66a95a`
to frozen source `8bc9c14` after the user requested one continuous learning hour
followed by the existing validation pipeline. This was the same Pod and rental;
cost starts at the original provisioning time. [Retarget history](retarget.json)
preserves both source archives, profiles and deadlines. The final source archive
SHA-256 is `78c90934cc4d6519482a660c92bfe89d255b5e431845f8a19f334d2404a65ba8`.

Setup and both native builds completed. The baseline was `81fd926` with the
explicit `worker-join-v1` lifecycle overlay, not the untouched historical binary.
The selected vendor GLX ICD was preserved, with driver 580.159.04. Debugger
preflight completed, and all four join-retirement fixture processes passed their
12 rounds each: 48 rounds total, with tracing on/off/off/on. This is positive
remote evidence for the owned-worker fixture. It does not establish general
caller-owned public-session thread retirement or a performance improvement.

CTest then passed 47 of 49 tests and failed two JavaScript contracts:

- `neural-native-stack-contracts`: the native pthread/GDB test asserted
  `false !== true` at `tests/neural_native_stack.mjs:253`.
- `neural-teacher-optimization-contracts`: the mocked hour/diagnostic continuation
  test asserted `0 !== 2` at `tests/neural_teacher_optimization.mjs:983`.

All native CTest cases passed. The engineering gate stopped the job before the
requested learning hour, teacher comparisons, paired learning or quality
evaluation. The later standalone Compute Sanitizer and Vulkan validation stages
also did not run. The mocked test's name is not evidence that the live hour ran.
There is no accepted speedup, quality score or strategy adoption. See the exact
test names, observed assertions and stage counts in [record.json](record.json).

The 95,309-byte results archive remains outside Git. Its independently verified
SHA-256 is `ca813cd3c2d1cbf038030d9805fa61855082259b2c177c5a949fec8232b6cb14`.
Collection was verified at 13:19:16.985 UTC before compute termination at
13:19:18.787 UTC on 2026-10-01. The independent [provider readback](provider-cleanup.json)
at 13:20:59.846 UTC showed zero Pods and zero network volumes. Both copied rental
records omit the connection endpoint; credentials, private identities and account
billing details are excluded.

The 1,526.767-second rental adds $0.2459791278 to the conservative cost ledger,
bringing the three attempts to $0.5639294889. This uses the $0.55/hour GPU cap plus
$0.03/hour storage allowance over the full rental, including setup before the
retarget. These estimates are not provider invoices. The historical reservation
is recorded separately from consumed cost.

The failed engineering tests need correction and validation before retrying the
requested hour and remaining pipeline. This archive preserves the frozen failure;
later fixes and runs must be recorded separately. [manifest.json](manifest.json)
checksums every preserved raw report and log; large source/results archives and
binaries are not committed.
