# Coverage pretraining before the two-hour run

```mermaid
flowchart TD
  A["Packed mesh cache + immutable FP32 source"] --> B["GPU topology and candidate placements"]
  B --> C["One conservative Vulkan R8 pass: 20.2 µs/view"]
  C --> D["CUDA coverage witnesses; candidate audits: 83.2 ms/cycle"]
  D --> E["Source + predecessor hard gates"]
  E --> F["Compact examples; position supervision only"]
  F --> G["Captured 128 → 64 → 64 → 12 MLP + AdamW: 124.5 ms / 1024 updates (reference)"]
  G --> B
  G --> H["Verified checkpoint + bounded recovery journal"]
  G --> I["Scheduled coverage gates + nonblocking shading diagnostics"]
```

Five alternating runs with the same coverage objective, teacher inputs and update budget:

| Full cycle | Median seconds |
|---|---:|
| Legacy attachments, reference optimizer | 1.479830 |
| Mask-only, reference optimizer | 1.406557 |
| Mask-only, fused optimizer | 1.285885 |

Mask renderer whole-cycle speedup: 1.05×.
Median teacher time: 0.377823 s; training phase:
0.299348 s. Setup and final publication are included
in whole-cycle time. Per-phase, transfer, allocation and draw counters are in
matched-coverage.json; GPU packing/render/unpack timestamps are in raster.json.
Selected update backend: fused. Selection requires at least 5%
lower median full-cycle time plus numerical and restart checks.

Median stage timings for mask-only coverage with the reference optimizer:

| Stage | Milliseconds |
|---|---:|
| Teacher total | 377.823 |
| ↳ Load and session | 77.949 |
| ↳ Packing and baseline gates | 50.317 |
| ↳ GPU state setup | 11.193 |
| ↳ Features and proposals | 80.638 |
| ↳ Candidate construction | 3.492 |
| ↳ Candidate audits | 83.161 |
| ↳ Commit | 1.089 |
| ↳ Final gates, serialization and remaining preparation | 57.704 |
| Training total | 299.348 |
| ↳ Dataset append | 23.398 |
| ↳ Graph capture | 116.479 |
| ↳ Optimizer updates | 124.475 |
| ↳ Checkpoint snapshot/enqueue/wait | 21.921 |

Medians of sub-stages need not add to the median total. Asynchronous checkpoint
publication overlaps the cycle. These are bounded cold cycles with two teacher
states; persistent training amortizes setup and graph capture.

Hardware profile: median packed full/mask GPU raster time
35.6 / 20.2 µs per view;
renderer workspace 19329678 / 6702862 bytes.
All mask/full coverage pixels match. Packing/interop wall time is reported
separately in raster.json; GPU draw speedup is not whole-cycle speedup.

Positions are UNORM16 in mesh bounds with at most 5% exact FP32 exceptions among
referenced surviving vertex IDs. Normals/tangents use 10-bit components and sign;
UVs use UNORM16 in [-8,8], colors RGBA8. Coverage draws skip attribute packing.
Reference masks use one bit/pixel; candidates can be read directly from R8 surfaces.
Remaining repeated work is teacher topology, candidate/view draws, exact fallback
distance transforms, and host decisions/synchronization. The masked objective
does not supervise appearance; these checkpoints require shading-aware fine-tuning.

Learning lasts at least 120 minutes, with 10 additional minutes for finalization.
Interrupted downtime is excluded. This development pilot is not a release score.
