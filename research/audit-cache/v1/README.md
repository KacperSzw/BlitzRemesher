# Rejected first implementation

The initial cache passed ordinary mesh/measurement comparisons and produced
promising preliminary timings. Review then identified a callback-state defect:
it copied `EvalSettings::cancelled` for each evaluation. A C++ callback holding
mutable state by value could therefore reset between comparisons. Adjacent
comparisons also used a copy of the source callback instead of their own
existing settings object. This violates behavioral equivalence.

The new mutable-callback regression fails on v1. Its failing output is retained
in `../checks/callback-regression-before.log`. The correction passes the original
evaluation settings by const reference and eliminates callback copies from the
cache. The regression now checks persistent state over successive evaluations.

The v1 pilot was stopped during its third pair. Two complete pairs and the
partial third run remain in `research/runs/audit-cache-v1/`. No aggregate result
is assigned to the incomplete five-pair experiment, and these timings are not
pooled with the corrected implementation. Source, build identity and logs remain
here. The pre-change baseline executable and its complete run are reused because
their input, configuration, protocol and executable identities did not change.

The first v1 sanitizer run separately exposed incomplete nothrow allocation
overrides in the test's failure injector. `builds/test-injector-fix.patch` records
that test-only fix relative to the archived snapshot. The corrected v1 sanitizer
suite passed, before the mutable-callback regression was added. These historical
checks do not validate v1's callback behavior.
