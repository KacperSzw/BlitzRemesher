# Hardware rasterization and resident learning

Current follow-up: [compact GPU pipeline](COMPACT-PIPELINE.md) documents the
packed cache/shards, persistent renderer, direct targets, bounded packing repair
and staged checkpoints. Its local teacher curriculum passes the boulder;
shelves/moon-rock final packed diagnostics still fail. The measurements and
remote validation below are historical and do not validate the new binary.

Local RTX 2080, driver 595.71.05, CUDA 12.9, LibTorch 2.10/cu128.
The complete FP32 draw-control pipeline is **2.90× faster end to end** on the frozen local
learning pass. Three sequential repeats: legacy 38.61–40.55 s; new
13.57–14.25 s. Both paths start from optimizer step 224,768 and perform
54 teacher states, 12,288 updates, 24 checkpoints, and constant/learned audits
on the same two development assets. Settings, cameras, work budgets, and
visual limits are unchanged. Hardware raster semantics and teacher ordering
are versioned changes; training labels and final models can differ.

| Measured component | Legacy median | Vulkan FP32 median | Speedup |
|---|---:|---:|---:|
| Entire pass, including startup/teardown | 39.75 s | 13.73 s | 2.90× |
| Six teacher phases | 20.53 s | 7.61 s | 2.70× |
| Six training phases | 11.54 s | 2.33 s | 4.95× |
| Both quality diagnostics | 6.42 s | 2.82 s | 2.28× |
| Expensive bench teacher core, 256→128 px | 6.59 s | 1.16 s | 5.67× |

Independent medians need not add. The workstation was shared; all raw repeats,
GPU occupancy snapshots, binary/source hashes, settings and per-asset results
are in [comparison.json](evidence/hardware-local/comparison.json). These are
throughput measurements, not a quality score or a claim that more updates
produce better meshes. Unreduced levels remain visible in the quality rows.

## Architecture and timings

```mermaid
flowchart TD
    A[Immutable FP32 master mesh] --> B[CUDA topology and placement proposals]
    B --> C[GPU draw streams: packed default, FP32 control]
    C --> D[Vulkan conservative coverage + center visibility]
    D --> E[CUDA exact XOR area and sparse witness queues]
    E --> F{Threshold certified?}
    F -->|Unknown or overflow| G[Exact metric: cached NPP distance fields + FP64 appearance]
    F -->|Pass or fail| H[Versioned teacher ranking]
    G --> H
    H --> I[Exact audit before every committed placement]
    I --> J[Append compact supervised rows to resident GPU pages]
    J --> K[Stable CUDA graph: sample → gather → MLP → loss → backward → Adam]
    K -->|GPU weight refresh| B
    K --> L[Frozen CPU checkpoint snapshot → background write + numeric checks]
    K --> M[Scheduled authoritative Vulkan quality audit]
```

Representative median native run, including process startup and teardown:

```text
Full learning pass                                 13.731 s
├─ Six teacher phases                               7.493 s
│  ├─ Bench 32 px                         0.720 s  (core 0.483)
│  ├─ Sweet potato 64 px                  0.744 s  (core 0.537)
│  ├─ Boulder 128 px                      1.550 s  (core 1.305)
│  ├─ Bench 256→128 px                    1.474 s  (core 1.211)
│  ├─ Sweet potato 256→128 px             1.262 s  (core 0.980)
│  └─ Boulder 128→64 px                   1.744 s  (core 1.507)
├─ Training phases                                  2.330 s
│  ├─ 12,288 optimizer updates                       2.151 s (175 µs/update)
│  ├─ One graph capture                              0.055 s
│  ├─ Dataset append / initial optimizer restore    0.042 s
│  ├─ Foreground checkpoint preparation/checks       0.042 s
│  └─ Writer joins / refresh / bookkeeping           0.040 s
├─ Quality diagnostics                              2.932 s
│  ├─ Constant ranking                               1.591 s
│  └─ Learned ranking                                1.341 s
└─ Startup / model setup / teardown / remainder     0.975 s
```

The last resident dataset occupies **87,431 bytes**; the graph was captured
once across all six appends. Only supervised rows are stored. Float features
and targets retain FP32; bounded flags/labels are packed. The sampler keeps
category/asset/progress balancing and deterministic counter-based selection.
Ordinary phase transitions refresh policy weights device-to-device.

Sparse queries ask for witnesses within the unchanged threshold, rather than
computing a nearest match for every pixel. Coverage queues contain XOR pixels
and search the entire opposing mask, including unchanged pixels. Appearance
queues omit same-pixel pairs with a conservative, outward-rounded proof.
A warp searches each remaining pixel. Queue overflow, clipping, unproven
coverage, or unsupported radius uses the exact fallback. Bounds have their
own `AuditPredicate` type and are never exported as exact measurements.

NPP PBA supplies nearest-site integer coordinates, converted to exact squared
integer distances; its rounded floating distance output is not squared.
The fast path is limited to image sides 64–2897, where every squared pixel
distance fits exactly in FP32. Smaller/larger images retain the reference
transform. Empty masks are handled before NPP. Twenty-one deterministic
cases, including odd dimensions and the upper boundary, matched an independent
integer oracle with zero mismatches. Immutable reference fields are cached.

## Packed draw default

Separate GPU streams use the requested formats:

| Stream | Representation | Bytes/vertex |
|---|---|---:|
| Position | XYZ UNORM16 within mesh bounds | 6 |
| UV, when present | XY UNORM16 mapped to [-8,8] | 4 |
| Linear color, when present | RGBA8 | 4 |
| Normal | RGB10A2 SNORM, alpha zero | 4 |
| Tangent, when present | RGB10A2 SNORM, alpha canonical ±1 | 4 |

All attributes present: **52 → 22 bytes/vertex**. The sign of tangent alpha
is bit 31; canonical signed alpha encodes +1 as 01 and -1 as 11.
Position step is mesh extent/65,535 per axis; UV step is 16/65,535.
Out-of-range or nonfinite UVs are rejected. Degenerate bounds, zero/missing
normals, linear color bytes, sign encoding and immutable source data are tested.
Bounds contain 24 bytes of values; aligned shader metadata occupies 48 bytes.
Triangle indices remain uint32 to match the full mesh ID domain and CUDA state.

The measured bench has positions, normals and UVs: **21,664 → 9,478 bytes**
of draw vertices, a 56.25% reduction. Across four 128 px cameras, FP32 Vulkan
rendering (packing, interop and target conversion included) took 0.283–0.464 ms
versus 1.043–1.403 ms for CUDA. The hardware passes themselves took 21–29 µs.
Packed Vulkan took 0.313–0.432 ms: **no consistent additional speedup** over
FP32 Vulkan on this mesh. Packing adds work and the shader does not currently
consume UV/tangent data.

The larger boulder (59,066 triangles) shows where hardware rasterization helps
most. Four 128 px cameras, 4× sampling, 12 draws each with a changed geometry
revision, include packing, both raster passes, interop and target conversion:

| Boulder component | CUDA FP32 | Vulkan FP32 | Vulkan packed |
|---|---:|---:|---:|
| Complete render per view | 5.774–7.508 ms | 0.414–0.447 ms | 0.427–0.508 ms |
| Hardware passes | — | 47–51 µs | 45–51 µs |
| Draw preparation on GPU | — | 57–59 µs | 85–141 µs |
| Target conversion on GPU | — | 18–19 µs | 19 µs |
| Draw vertex bytes | 965,280 | 965,280 | 422,310 |

These are per-camera ranges, not confidence intervals. First-use setup and
diagnostic readback are excluded. Packed Vulkan is 11.4–17.1× faster than CUDA
for the corresponding cameras, but again does not beat FP32 Vulkan consistently.
Position-only packing already produces 11–54° maximum pixel-normal changes;
full packing produces almost the same changes. This isolates face-ownership
changes caused by position quantization rather than normal encoding as the main
source of those large local deltas. Raw results: [raster-boulder.json](evidence/hardware-local/raster-boulder.json).

**Packed draw data is the Vulkan and resident-cycle default**, as requested.
`--vertex-storage fp32` explicitly selects the diagnostic control used for the
complete timing comparison above. Automatic storage resolves to FP32 for the
legacy CUDA renderer, which has no packed draw interface. A packed bench pilot
failed the unchanged visual gate and is retained as negative evidence; a packed
sweet-potato pilot completed. The packed full-cycle pilot completes bench 32 px
and sweet potato 64 px, then stops at boulder 128 px with no feasible queried
placement. It has no complete-cycle throughput result or quality score. A few edge pixels changed face ownership, producing up to ~90°
pixel-normal differences despite small numeric quantization. Original source
renders stay FP32, so storage error consumes the existing visual budget.
Master geometry, depth, render attributes and optimizer arithmetic remain FP32;
the exact metric retains its FP64 calculations. Current audits do not sample
textures/normal maps, so UV/tangent tests establish storage/frame contracts only.

## Remaining costs

The separate bench replay has 355.47 ms of CUDA kernel time (Vulkan draw time
is additional). It produced exactly the native run's teacher dataset:

| CUDA work in the expensive bench teacher replay | Time | Kernel-time share |
|---|---:|---:|
| Threshold scan and witness-queue construction | 88.89 ms | 25.0% |
| Exact appearance checks | 86.45 ms | 24.3% |
| Vulkan target conversion | 52.98 ms | 14.9% |
| Sparse appearance witness search | 50.80 ms | 14.3% |
| Sparse coverage witness search | 4.79 ms | 1.3% |
| Geometry, draw preparation, distance fields and other kernels | 71.56 ms | 20.1% |

Teacher work remains 54.6% of the representative complete pass; quality audits
take 21.4%, training 17.0%, and process/setup/remainder 7.1%. Kernel shares above
belong to a separate profiled teacher replay, not to the entire learning pass.

The main remaining opportunities are view batching and reducing host waits,
fusing target conversion with predicate scanning, and reusing Vulkan device/
pipeline setup across teacher phases. `cudaMemcpy` API time was 818 ms across
6,157 calls; it includes waiting for preceding work and must not be added to
GPU kernel time. All teacher phase setup/remainder costs total 1.47 s in the
representative pass. The 59k-triangle boulder now has the slowest teacher core.

This implementation redraws all geometry into cleared targets. Dirty scissors,
per-tile certificate reuse, batched views and direct surface comparisons are
not implemented. Full redraw preserves hidden-surface correctness after removing
an occluder. The measured hardware draw is small; unmeasured complexity was not
used to claim a speedup. CPU code still schedules phases and transfers small
status summaries; geometry, rasterization, comparisons and updates run on GPU.

## Recommended next steps

1. **Make the packed curriculum feasible before long training.** Capture exact
   failures for the unchanged boulder, then for the rejected placements. Separate
   the initial packing error from collapse error with FP32, position-only and
   fully packed controls. Test stable quantization bounds and placements chosen
   on the UNORM grid, retaining the FP32 source and existing visual limits.
   Acceptance requires complete packed development passes and the final quality
   audits; more optimizer updates cannot fix a teacher that emits no feasible
   examples. If UNORM16 cannot meet the contract, document the affected cases
   before proposing a storage or quality-policy change.
2. **Batch views and keep the Vulkan device/pipelines alive across phases.**
   Submit several camera layers together, share geometry preparation, and return
   one compact result per candidate. Target the 1.47 s of teacher setup/remainder
   and repeated host synchronization. Keep explicit layer ownership and a bounded
   memory budget; measure complete teacher time as well as GPU kernel time.
3. **Measure and reduce checkpoint publication waits on remote storage.** The
   remote control spends 1.41 s in writer joins, refresh and bookkeeping versus
   0.34 s in optimizer updates. First time those operations separately. Test
   local scratch for active checkpoints with a bounded background copy to durable
   storage; publish a recovery pointer only after the durable copy and checksums
   succeed. Preserve the forced-crash recovery contract and bound queued snapshots.
4. **Read raster targets directly in the threshold scan.** Fuse CUDA surface
   decoding with queue construction to remove the temporary linear pixel write
   and reread on sparse queries. Materialize linear images lazily for exact
   fallback. Conversion plus scanning currently account for 39.9% of profiled
   CUDA time, but scanning remains necessary, so that share is not a promised
   speedup. Compare sparse results and exact fallback against the current oracle.
5. **Optimize appearance searches after those changes.** Exact and sparse
   appearance together use 38.6% of the profiled CUDA time. Test compact tile
   summaries and cooperative searches against the complete opposing image.
   Preserve witnesses outside changed pixels and exact audits before commit.
6. **Then run the two-hour learning comparison.** Reuse frozen inputs, cameras,
   seeds and work accounting; report useful teacher states, updates, accepted
   reductions, visual errors, fallbacks and wall time. Require successful packed
   pilots, checkpoint recovery and remote replay before the long run.

Packing more buffers or reducing arithmetic precision is a lower priority:
hardware drawing already takes tens of microseconds and the current packing
experiment saves memory without consistently reducing draw latency. Each step
above needs a matched local pilot; no additional speedup is assumed in advance.

## Validation and use

Local: 26/26 CTest tests, 10/10 ASan/UBSan tests, CUDA memcheck for sparse
metrics/resident training/Vulkan interop, and Vulkan synchronization validation
with the Khronos layer confirmed active. Vulkan reports no errors; an unused
color-output warning is expected when the optional color target is absent.
A forced SIGKILL at update 512 resumed within its 2,048-update segment and
finished with identical dataset bytes and model parameters. A failed checkpoint
verification cannot replace the last valid recovery pointer. Snapshots occur
every 512 updates, with one bounded background writer.

Build and run one bounded full learning pass:

```sh
nix develop .#neural -c cmake -S . -B build/neural -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBLITZ_CUDA=ON -DBLITZ_VULKAN=ON -DBLITZ_NEURAL_TRAIN=ON
nix develop .#neural -c cmake --build build/neural -j3
nix develop .#neural -c build/neural/blitz-neural-cycle RUN_DIRECTORY \
  --initialize MODEL.blzn --warmstart CHECKPOINT.pt --minutes 5
```

Run the same command/directory to resume a matching checkpoint. The binary,
dataset hashes, work configuration and source manifests are checked. Packed is the hardware default; use `--vertex-storage fp32` for the complete
control workload, or `--vertex-storage position16` to isolate position packing. The library/CLI selection is `NeuralOptions::raster_backend` /
`--raster-backend vulkan`; the legacy library default stays CUDA for compatibility.
Vulkan needs a graphics-capable NVIDIA device matching CUDA's UUID, Linux FD
external memory/semaphores, conservative rasterization and Vulkan 1.3 features.

Reproduce throughput and crash recovery with `hardware-profile.mjs` and
`hardware-recovery.mjs`. Raw evidence is under
[evidence/hardware-local](evidence/hardware-local/). The benchmark binary hash
is retained; subsequent edits only strengthen recovery configuration validation,
release an old autograd graph before capacity recapture, and reject an invalid
CUDA/packed option combination. The hot algorithm is unchanged.

A final-build FP32 control repeated all 54 states and 12,288 updates with
**identical teacher bytes and trained parameters** to the matched native run.
Its internal cycle time was 11.19 s: teacher 6.32 s, training 2.45 s, and quality
2.22 s. This excludes process setup/teardown and is a single validation run;
the matched three-repeat comparison remains the speedup evidence. See
[final-control-cycle.json](evidence/hardware-local/final-control-cycle.json).

## Remote validation

Validated commit `e3ee066` on an RTX 4090 (SM89, 24,564 MiB), driver 580.126.16,
CUDA 12.9 and LibTorch 2.10/cu128 on 2026-09-30. **26/26 CTest tests passed**,
along with Vulkan validation and three CUDA memchecks (sparse metrics, Vulkan
interop and resident training), each reporting zero errors. The bounded job
completed in 27.12 s after setup. Its results were collected and checksum-verified.

The remote checkpoint replays exactly on the same device. The saved local
checkpoint also passes the existing replay tolerance on the remote GPU: maximum
absolute difference **1.62e-5** for native inference and **1.89e-5** against the
FP64 oracle. Cross-device results are not bit-identical. All raw checks and
provenance are in [validated-06](evidence/hardware-remote/validated-06/).

The smaller remote FP32 control uses **18 teacher states, 3,072 updates and six
checkpoints**, starting from initialization. Its workload and initialization
differ from the matched local benchmark; it is a validation timing, not another
end-to-end speedup comparison:

```text
Remote FP32 control, including process lifetime      7.861 s
├─ Six teacher phases                               3.491 s
│  ├─ Teacher core work                             1.914 s
│  └─ Setup / remainder                             1.578 s
├─ Training phases                                  1.861 s
│  ├─ 3,072 updates                                 0.339 s (110 µs/update)
│  ├─ One graph capture                             0.046 s
│  ├─ Dataset append                                0.008 s
│  ├─ Foreground checkpoint preparation/checks       0.055 s
│  └─ Writer joins / refresh / bookkeeping           1.412 s
├─ Quality diagnostics                              1.647 s
└─ Startup / teardown / other remainder              0.863 s
```

The last dataset occupies 29,352 GPU bytes. Checkpoints are written to the remote
network volume; separating its I/O cost from other writer/refresh work is a
follow-up measurement, not something the current grouped timer establishes.
The six short phases end at a checkpoint, leaving little update work to overlap
with each final write.

Remote bench rasterization takes **0.189–0.206 ms/view packed**, versus
0.182–0.200 ms for FP32 Vulkan and 0.416–0.539 ms for CUDA. Hardware passes take
10.7–12.5 µs; packed draw preparation takes 15.1–16.4 µs and target conversion
7.9–8.6 µs. Packed vertex bytes remain 21,664 → 9,478. This again supports a
memory saving, without an extra speedup over FP32 Vulkan.

**The packed full curriculum remains blocked.** The default packed remote cycle
completes bench and potato, then rejects boulder under the unchanged visual gate,
matching the local result. A separate packed potato pilot passes. The failed
packed cycle exits after 3.009 s; that is time to failure, not a complete-cycle
timing or quality score. The validation job passes because it verifies this
expected rejection separately from the successful FP32 control. Its
`ready_for_two_hour_packed_cycle` remains `false`; no new two-hour run was started.

The successful rental lasted 650.10 s at $0.74/hour. Including failed setup and
allocation attempts, the conservative charge is **under $0.73**, within the
existing additional grant. The provider confirms **zero live pods and zero
network volumes**. See [billing.json](evidence/hardware-remote/billing.json).
Earlier provisioning failures and the L40S graphics-preflight failure remain
visible under [hardware-remote](evidence/hardware-remote/).

The container now installs GLVND/X11 runtime dependencies and selects the EGL
entry from the host-injected NVIDIA ICD, preserving its API version and directory.
NVIDIA documents this [headless Vulkan entry](https://download.nvidia.com/XFree86/Linux-x86_64/570.86.16/README/installedcomponents.html).
Local packed draw/storage tests also pass through EGL with synchronization
validation. The container changes do not modify the neural runtime measured by
the local benchmark. All 21 cloud/budget contract cases and the two corresponding
CTest targets pass after the final infrastructure changes.
