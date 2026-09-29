# Multi-level curriculum evidence

Source: `c3af406`, RTX PRO 6000 cloud run on 2026-09-29. `manifest.json` records
the checksum-verified archive. Compute and volume were deleted after collection.

All seven prepared shards completed, with 164 states across two training assets.
The six new shards cover 16/64/256 px and audited, fixed preceding LODs at
32/128/512 px. Target fractions vary over 0.02/0.1/0.5/0.9. `data/` retains the
exact shards, checksums, provenance and queried-action trajectories. The runner
can reuse these checksummed data instead of repeating GPU label generation.

| Seed | Updates | Preferred membership | Initial loss | Final loss | Native/FP64 maximum |
|---|---:|---:|---:|---:|---:|
| 101 | 8192 | 99.3902% | 1.35428 | 0.0200202 | 2.09686e-5 |
| 202 | 8192 | 98.7805% | 1.34404 | 0.0204369 | 2.17033e-5 |
| 303 | 8192 | 99.3902% | 1.36185 | 0.0197705 | 2.24356e-5 |

All parameters/gradients were finite, parameters updated, exact model/AdamW
restoration passed, and Torch/native maximum difference was zero. Membership is
on the queried training pools, not held-out accuracy. Each 8,192-update segment
took 9.08–10.87 seconds. Training telemetry averaged 38.74% GPU utilization over
31 one-second samples; preparation averaged 71.22%. No saturation claim is made.

The first full-pilot process stopped at argument parsing: the benchmark CLI did
not yet accept the runner's `--gpu-memory-mib` option. `pilot-1/` preserves this
failure. There is no full-pilot score or generalization result from this run.
The repair adds that option and a real CLI contract check to CTest.

`smoke-cli-check/` is the completed local two-mesh test of the repaired cloud
option set. Mean triangle reduction was 4.096%, with no fallbacks or resource
failures. This is below the earlier one-asset model's 6.940% smoke result; greater
training diversity has not yet established better quality. GPU capacity differs
between those two smoke records, and neither had a recorded resource failure.
Full matched controls, three seeds and checkpoint persistence remain required.

Data derive from the CC0 assets identified in each contract. These models are
experimental endpoint rankers. Texture/normal-map scoring and segment placement
are outside this stage; source/preceding-LOD audits still decide acceptance.
