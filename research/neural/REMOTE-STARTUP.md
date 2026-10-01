# Remote startup

Recent A40 attempts that reached learning spent about 17–18 minutes between
rental creation and the start of capability learning. That includes provisioning, SSH, upload,
dependency setup, two native builds and the engineering gates. It is not all
container-image download time. The one-hour run frozen at `2246b91` uses its
original controller and source; the changes below apply to new bundles.

## Implemented

- Local preparation compresses the immutable input archive with gzip level 1.
  The unchanged 204,922,880-byte hour bundle compressed to 153,652,083 bytes in
  3.558 seconds locally. Decompression reproduces its exact SHA-256: 25% fewer
  transfer bytes, not a measured 25% reduction in setup time. New bundles include
  their own filename, size and checksum; older uncompressed bundles still work.
- Core-validation and teacher-optimization build every native test executable
  and the experiment tools through `blitz-neural-cloud-validation`. The full
  CUDA/LibTorch configuration retains all 32 native CTest executables while its
  Ninja plan drops from 138 to 100 steps. Other experiment profiles retain their
  existing tool sets. Tests, memchecks, Vulkan validation and teardown checks
  still run before learning.
- Build concurrency uses CPU affinity, cgroup quotas and available host/cgroup
  memory, capped at eight jobs. CUDA-containing targets have at most two compile
  slots; Torch-heavy targets and the teacher executable share one heavy slot.
  Memory estimates guide scheduling and are not a hard memory reservation.
- An optional local artifact reuses only the unchanged Ubuntu-built comparison
  baseline executable. Source revision, reviewed overlay, effective Git tree,
  compiler/link commands, CUDA architecture, toolchain, package identities,
  selected SDK headers and actual runtime-library hashes must match. CTest's
  container hostname and scheduling-only options do not enter the identity.
  NixOS executables are not admitted as Ubuntu build artifacts. A remote miss
  rebuilds from the freshly verified source; an optional export failure does not
  abort an otherwise successful build. No results or runtime gate outcomes are
  cached, and host-driver checks always run again.
- `rental.json.phase_history` records controller transitions. Remote
  `setup-stages.jsonl`, `baseline-resources.json`, `candidate-resources.json` and
  build-target manifests separate
  package installation, ICD checks, LibTorch download/verification/extraction,
  configuration, compilation and checks. Failed stages retain their exit status.

## Reuse a collected baseline

Select the direct directory containing `manifest.json` and `binary.gz` from a
verified earlier collection. The artifact stays on the workstation between runs;
no persistent Runpod volume is needed. Normal setup exports future artifacts to
`results/baseline-cache` for checksummed collection.

```sh
BLITZ_BASELINE_CACHE=/absolute/path/to/collected/results/baseline-cache \
BLITZ_RUNPOD_PROFILE=teacher-optimization-a40-hour \
node scripts/neural/runpod.mjs prepare-teacher-optimization runs/neural/NEW
```

Omit `BLITZ_BASELINE_CACHE` to build normally. Explicitly supplied corrupt or
incompatible source artifacts fail during local preparation, before renting.
The remote machine independently checks its own environment after fresh CMake
configuration and falls back to compilation on a mismatch. Broad package and
CMake metadata checks deliberately allow conservative misses; changing an
unrelated installed package can currently cause a rebuild.

The current rental's baseline was also exported and validated by a separate
CPU-only probe without changing its source, build files or executable. The
corrected probe exported in 1,691 ms and validated in 1,543 ms; its downloaded
archive is 1,936,211 bytes. This proves same-host verification, not a cache hit or
speedup on a second Pod. `runs/neural/startup-prepare-01` successfully prepared
and rechecked all 32 inputs, including that artifact, without starting a rental.

## Remaining work and limits

The changing candidate still compiles during setup, and the pinned 3.7 GB
LibTorch distribution is still downloaded and unpacked. Preparing an Ubuntu/CUDA
dependency image and candidate executables on CPU infrastructure before renting
is the next larger opportunity. Runpod supports reusable custom images and
preloaded dependencies through [custom Pod templates](https://docs.runpod.io/pods/templates/create-custom-template).
A cold host still needs missing image layers; no host-cache guarantee is assumed.
Persistent [network storage](https://docs.runpod.io/pods/storage/types) can retain
dependencies but continues to incur storage charges, so it is not the default.

Prophet research informed the image/storage comparison and artifact-cache design.
The initial repository lookup had no indexed source, so its design response was
not treated as a source review or evidence for a numerical speedup. The separate
attached-source review failed when its transport closed. Independent local review,
tests and actual Ubuntu checks provide the implementation evidence.

No end-to-end startup reduction is claimed until a future, already-authorized
experiment measures the new path. Faster setup also does not resolve teacher
generation throughput or demonstrate better LOD quality; those have separate
[training and comparison gates](TEACHER-OPTIMIZATION.md).
