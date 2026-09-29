# Neural LOD implementation and run contract

Branch: `research/neural-lod-gpu`, based on `7466a9e`. The sibling worktree
leaves the primary checkout available to other agents. Implementation precedes
the ten-hour experiment budget. No held-out asset is used for training or tuning.

Current status (2026-09-29): implementation and audit recovery are validated;
local training is stopped. A from-scratch two-hour Secure Cloud RTX PRO 6000
experiment is now authorized; see RUNPOD.md for the staged deployment and gates.
The saved-model pilot returns
eight unreduced fallbacks. The numbered implementation steps below are completed
except for establishing useful LOD quality. Remote health must be measured.

1. Establish CPU tests; add deterministic graph/decoder fixtures and explicit
   unavailable behavior. Verify source stream immutability and endpoint IDs.
2. Add optional CUDA raster, exact distance transform, product metric and overlap
   diagnostics. Verify CPU agreement, including empty masks, occlusion, material
   boundaries and cancellation. CPU audits confirm selected chains.
3. Add a three-layer, width-64 sparse graph encoder and pixel-conditioned head.
   Predict retention and representative displacement, decode only legal edge
   contractions. Preserve boundary/seam/material locks and UV orientation.
   Use original endpoint attributes; report UV distortion, never average charts.
4. Add C++/LibTorch CUDA training, bounded patches with three-hop halos, compact
   portable weights, checksummed provenance, additive C API and explicit CLI mode.
   Keep LibTorch out of runtime linkage. Verify exported/native inference agreement.
5. Freeze development training assets, exclude the eight development pilot
   assets and their source families (66 assets remain), use validation only for model selection and held-out only for release.
   Start with QEM endpoint labels, then camera-audited labels. Record failures.
6. Measure a warm training pilot and prepare a resumable ten-hour run. Require
   finite gradients/loss, parameter updates, native export agreement, restore
   agreement, a valid audited mesh and measured ETA before starting the service.
   Atomic checkpoints, bounded segments, logs and an automatic final report let
   the implementing agent leave after the run is demonstrably healthy.
7. Measure sustained training utilization separately from CUDA mesh scoring.
   Compare useful core vertices/second while retaining the same compact network.
   Batch connected graphs and overlap bounded pinned-host preparation with CUDA.
   Require at least 60 seconds of telemetry, mean GPU utilization >=90%, tenth
   percentile >=85%, finite gradients, decreasing mean training loss, exact model
   and AdamW moment restoration, and native export error <=2e-4. Continue the
   detached service after the health gate. Audit each 25,000-update stage and
   stop after two stages without a better complete development pilot score.
8. Exercise the full frozen eight-asset audit before declaring the experiment
   ready. Bound only optional candidate refinement to fitting raster sizes;
   reject uncertain bounds and preserve final CPU confirmation. Record resource
   causes, visit remaining assets, stop fixed-limit retries, retain null SCORE
   for incomplete audits, and prevent reuse across different model hashes.

Next research steps, after a request to resume work: diagnose why the audited
25,000-update model produces no reduction; test one curriculum/decoder hypothesis
on deterministic fixtures and the development pilot. Remote GPU migration and
cost options are in CLOUD_GPU.md. Buying more compute alone does not establish
the missing quality signal.

Runtime objectives: pass both source and adjacent visual gates; minimize triangles
within the existing storage tradeoff; use pre-depth geometric overlap only to
break equal-triangle/equal-byte ties. It is not shader time and does not alter
protocol v3 SCORE. Missing CUDA/model is an explicit error. Unreduced geometry
remains a visible fallback. Training success is not evidence of release quality.

GPU: RTX 2080, sm_75, 8 GiB. Target process device allocation at most 6 GiB with
headroom for the desktop. Import, topology decoding and final reference audits
run on CPU. Heavy tensor operations and candidate raster/distance work run on GPU.
Pinned training dependency: official LibTorch 2.10.0 cu128 C++11 ABI archive,
https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.10.0%2Bcu128.zip
(verified in the official index). CUDA 12.9 is the available Nix toolkit; LibTorch
uses its bundled 12.8 runtime. Train without Python.

Record each milestone's actual evidence and limitations in REPORT.md. Do not
claim completion of training, a ten-hour ETA, or release quality from estimates.

Measured iteration: four patches/update reached 39% GPU utilization; 64 patches
with two preparation workers reached 97.8% and 5.63 million supervised core
vertices/second. The architecture remains 24,580 parameters. GPU saturation is
a throughput result; the initial full development pilot produced only unreduced
fallbacks. Later training must establish mesh quality through independent audits.
