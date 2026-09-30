# Coverage pretraining implementation record

The coverage profile now uses one conservative Vulkan R8 pass, no depth or
shading attachments, one-bit owned reference masks and direct candidate surfaces.
It skips appearance search and uses ten position-only teacher alternatives plus
an optional policy proposal. Normal targets and normal edits are disabled in this
profile. Immutable audit sources remain FP32; working meshes and training shards
remain compact on disk and in transfers, decoded into FP32 GPU working state.

Packed positions allow sparse exact FP32 exceptions capped at 5% of referenced
surviving vertex IDs, including separate wedges. The bitmap and sparse values
survive packing, uploads, GPU trials, commits, compaction, reset, cache and replay.
Both free placement and vertex reuse enforce the surviving-vertex cap. The old
strict replay remains strict; repair is a new, fully audited representation.

Predecessors pass both preparation and destination pixel sizes. Audited exhausted
or unavailable predecessors remain explicit outcomes. Coverage quality gates,
geometry, resources and finite arithmetic remain blocking. Scheduled full shading
audits are diagnostic for shape pretraining. This checkpoint is not release-quality
proof and needs later appearance-aware fine-tuning and unseen-model evaluation.

The cycle owns one resident dataset, model and optimizer. Completed phase history
is append-only; the recovery journal contains the active dataset window and durable
history offset. Recovery excludes downtime from completed learning time. Learning
and finalization have separate budgets: 120 minutes plus ten minutes, respectively.
Changed objective means original weights with fresh Adam and fresh examples.

## Local evidence (RTX 2080, shared workstation)

[Architecture and timing hierarchy](evidence/coverage-local/ARCHITECTURE.md).
All raw measurements and checks are in [coverage-local](evidence/coverage-local/).
These are development smoke measurements, not an aggregate corpus score.

| Five alternating matched runs; 2 teacher states, 1024 updates | Median cycle |
|---|---:|
| Coverage with full attachments, reference optimizer | 1.123499 s |
| Mask-only coverage, reference optimizer | 1.072565 s |
| Mask-only coverage, fused optimizer | 1.024252 s |

The mask implementation reduces whole-cycle time by 4.53% (1.047×), and measured
teacher audit peak allocation by 41.65% (211,875,468 → 123,623,796 bytes).
Coverage variants produce identical training data; both reference-optimizer
variants produce identical weight payloads. Fused gradients and restart pass the
numerical contracts. Its final local median win over mask/reference is 4.50%, below
the 5% selection threshold; the remote benchmark will select its own backend.
Earlier local repeats are retained, including an 8.3% fused win, showing contention.

The four-view raster profile reports roughly 49–52 µs for full attachments and
28–30 µs for mask-only GPU drawing. Renderer allocation falls from 19,329,678 to
6,702,862 bytes (65.3%). Coverage pixels match exactly. Packing/interop wall times
are noisy on this shared GPU, including slower mask samples; raw rows are retained.
Do not multiply isolated GPU drawing speedups into full-cycle claims.

A separate five-repeat objective comparison measured median teacher time 0.822173 s
for appearance-aware teaching versus 0.366975 s for shape-only teaching (2.24×).
That larger saving changes the learning objective; it is not renderer acceleration.

The packed curriculum passed all 48 conditions in 13.812 s: 40 complete, six audited
search-exhausted and two predecessor-unavailable, with data in all four categories.
The historical moon-rock packed failure still reproduces. Promoting three of 1847
referenced vertices (0.1624%) repairs it under the unchanged 2 px limit: 1.171382 px
in the full 16-view audit. GPU state and compact snapshot pass; search audit also
passes at 1.914215 px. No visual threshold was loosened.

The duration smoke completed 60,000 ms of learning, 104,448 updates and 102 teacher
iterations, then final audits/publication; total 74.077 s. Coverage and shading
audits passed for constant and learned ranking. Journal size was 2822 bytes versus
379,582 bytes of append-only history. This is a lifecycle proof on one condition,
not evidence that those optimizer updates improve unseen models.

Validation: 28/28 CTest; 13/13 ASan/UBSan; 22 Node tests; CUDA memcheck zero errors
for action, Vulkan and resident training contracts. Vulkan validation found an
unused-input mismatch in the new coverage shader; fixed and rerun with zero
validation errors. Existing unused full-shader attachment output warnings remain
and are preserved in the compressed log. Sparse shader lookup, cap boundaries,
failed-then-valid reuse, exact optimizer restart and history recovery are covered.

## Remote launch contract

Latest future-spending authorization: $8 total, replacing the previous remaining
$2. Grant baseline is pinned to the reconciled ledger; retries consume this same
grant. At most 180 rental minutes, independent deadline watchdog, verified result
collection before volume deletion. Prefer a graphics-capable 24 GB or larger GPU.

Remote gates repeat CTest during setup, action/Vulkan memcheck, Vulkan validation,
resident optimizer/restart checks, historical strict replay, sparse repair,
cross-GPU export replay, five matched timing repeats, packed 48-condition pilot
and hardware raster parity. The remote architecture/timing artifact is written
before starting the 120-minute full learning cycle. A failed gate stops the job.

Original warm-start SHA256:
`9152bb42cb807a2e91fe3217ab6dc3bbcf11be12618bcd706acf71a8e3fff185`.
Remote results are pending; no remote success or training result is claimed here.

## Next bottlenecks

Candidate audits remain the largest teacher stage. The next substantial change
should batch candidate × view work and GPU decision reduction to remove repeated
host waits; benchmark the full cycle with unchanged candidates and coverage gates.
The small MLP update and frequent checkpoint publication also matter. Keep setup
and graph capture amortized in the resident run; use its telemetry to decide
whether larger update batches or a bounded checkpoint cadence justify further work.
Packing precision and visual thresholds should remain unchanged for those controls.
