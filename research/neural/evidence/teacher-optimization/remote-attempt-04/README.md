# Fourth remote teacher optimization attempt

The A40 run `teacher-optimization-05` used frozen source `959f248`. Setup and
engineering validation completed: 49/49 CTests, both Compute Sanitizer runs with
zero errors, Vulkan validation with zero errors, and all 48 join-retirement
fixture rounds across four processes. This rental was in CA-MTL-1 with driver
595.91.07; previous rentals used driver 580.159.04. The preserved GLX ICD and
explicit baseline lifecycle overlay are recorded in [record.json](record.json).

The requested 60-minute exhaustive learning run started at 13:48:15.919 UTC but
failed before the hour completed. The enclosing experiment reported failure at
13:50:06.350 UTC, about 110.431 seconds after the controller's training-phase
timestamp. The native process exited with code 1, without a timeout, cancellation
or signal. This interval includes startup and error cleanup; it is not a completed
learning-duration report.

History records 90 completed training blocks of 128 updates each, reaching step
11,520. Teacher job 90 then failed, and the trainer refused to consume that shard.
The latest verified saved checkpoint is **step 6,272**, with 60,051 milliseconds
of recorded learning duration. Its six indexed file hashes match, and its native,
FP64 and native-versus-FP64 checks passed the unchanged `2e-4` numerical gate.
The later 5,248 completed updates are recorded in history but are not represented
by that saved checkpoint. No completed-hour report or step-11,520 checkpoint exists.

The failed shard was CoffeeCart with a simplifier seed targeting 25% retention at
64 pixels, UV preservation enabled, and source/adjacent limits of 3/2 pixels. The
seed was marked accepted and its source audit passed, but its final adjacent audit
failed with measured error 2.372991193268713 pixels and `complete=false`.
The seed contained 6,914 triangles from a 27,659-triangle source; no teacher action
was accepted. The contract, index, seed, packing, query trajectory and reuse
records are preserved. This localizes the failure; the seed-admission investigation
and subsequent correction must be recorded separately.

The raw history contains 91 consumed teacher results (90 complete, then the failed
job), while 92 shard directories exist because queued workers can finish results
the learner never consumes. Partial-run host sums record 105.413 seconds waiting
for teacher results and 2.296 seconds in optimizer updates. These are not GPU
utilization measurements or a matched performance comparison. No initial/final
LOD diagnostic, teacher ABBA, paired pilot or full-quality comparison ran after
the failed hour. There is no accepted speedup, quality score or strategy adoption.

The 108,335,099-byte results archive remains outside Git. Its independently verified
SHA-256 is `6a984c44aff7285c919c7aea74dc6271b98a7b09be28b1e7142a15771c84ba68`.
[binary-inventory.json](binary-inventory.json) records hashes and sizes for the
saved checkpoint/probes, initial model, policy waves and failed shard payload;
those binaries remain in the verified archive. [manifest.json](manifest.json)
checksums the compact raw logs, metadata and derived record committed here.

Collection was verified at 13:50:32.711 UTC before compute termination at
13:50:34.063 UTC on 2026-10-01. Independent [provider readback](provider-cleanup.json)
at 13:53:20.054 UTC showed zero Pods and zero network volumes. The copied rental
ledger omits its connection endpoint; credentials and private identities are
excluded. The 1,164.295-second rental adds $0.1875808611 at the conservative GPU cap
plus storage allowance, bringing four attempts to $0.75151035. These are cost
estimates, not invoices. The failed seeded audit needs a regression and correction
before retrying the authorized hour and remaining validation.
