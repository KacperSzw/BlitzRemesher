# Concurrent learning pipeline implementation

The accepted next-run plan targets 100 independent training source groups and
20 validation groups, packed preparation, deeper audited training states,
concurrent teacher/learner work and time-based checkpoints. Core geometry and
network arithmetic remain FP32; visual and numerical limits are unchanged.
The next two-hour run must remain unlaunched during this preparation.

## Previous run and first gate

`runpod-coverage-pretraining-04` stopped with `checkpoint FP64 export verification
failed` after 35.12 learning minutes. Its last verified checkpoint contains
3,404,800 optimizer updates and 14,134 states. Thirty-minute coverage and shading
diagnostics completed on the two diagnostic meshes. This is an incomplete run,
not evidence of improved generalization. Collection was checksum-verified and
compute/storage were released.

Recorded wall phases: teacher 1236.711 s, checkpoint handling 463.377 s, optimizer
387.880 s. Teacher candidate audits account for 678.796 s. These durations are
not a GPU idle-gap attribution; that requires a timeline. Median teacher-shard
triangle removal was 0.05423%, so deeper state coverage is also necessary.

The old failed probe was stored outside the results tree and lost at pod cleanup.
Checkpoint failures now retain model, optimizer, probe and verification files
under the run's `checkpoint-failures`, without advancing the recovery journal.
`blitz-neural-cycle --replay-checkpoint SOURCE NEW_OUTPUT UPDATES` loads the
checksummed dataset, optimizer and sampler state without modifying SOURCE. Its
report identifies the new build/device; it does not claim an identical resume.

RTX 2080 replay passed 1,024 and 16,384 additional updates from step 3,404,800.
The L40S failure remains unexplained. Numerical acceptance remains the existing
finite, same-shape, maximum absolute difference <= 2e-4 contract.

## Implementation gates

- Reproduce/diagnose numerical failure, preserve evidence and protect recovery.
- Profile baseline and implement bounded stream/worker ownership and overlap.
- Freeze expanded corpus and cache preparation; preserve source audit references.
- Add persistent trajectories, bounded replay and checkpoint cadence controls.
- Add versioned model dimensions and quality-controlled width/replay experiments.
- Run local correctness, fixed-work comparisons and validation-only remote proof.
- Publish timings, bottlenecks, selected configuration and next-run readiness.

`pipeline-validation` rentals use the current pinned $8 grant, including prior
attempts and a $1 reserve. Their entry point cannot launch the long training run.
The first forensic rental is capped at 35 minutes including setup/collection;
final pipeline validation is capped at 60 minutes.
