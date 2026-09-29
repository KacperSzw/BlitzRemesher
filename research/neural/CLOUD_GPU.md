# External GPU training — researched 2026-09-29

Training is stopped. This document records research and a migration plan; no
cloud machine was rented, no data uploaded, and no remote training was run.
Prophet researched prices and CUDA portability; the primary sources below were
then checked directly. Rates are USD before applicable taxes, storage and
transfer, and do not establish live capacity in a particular region.

## Recommendation for this experiment

Use one Linux x86_64 NVIDIA GPU instance. An RTX 4090 is a reasonable first
performance candidate; an RTX A5000 or A40 is a cheaper hourly alternative.
Benchmark cost per fixed number of updates before choosing a faster card.
The current trainer uses CUDA device 0 and has no distributed training path;
additional GPUs would sit unused. A GPU farm can supply one machine without
requiring us to implement distributed training.

The current 25,000-update model produces eight unreduced pilot fallbacks. More
compute does not address the demonstrated quality gap by itself. The next
experiment should test a curriculum/decoder hypothesis and retain the independent
source/adjacent audit gates. The repaired 64-million-sample limit is a software
policy; renting more VRAM does not remove it.

## Measured workload and resources to rent

Measurements are local RTX 2080 results, not cloud benchmarks. Raw records are
in [the report](REPORT.md) and [the stopped checkpoint record](evidence/audit-recovery/stopped-checkpoint.json).

| Item | Current workload |
| --- | --- |
| Network | 24,580 FP32 parameters; three graph layers, width 64 |
| Batch | 64 connected patches; up to 4,096 core vertices plus three-hop halo each |
| Training rate | Approximately 0.04286 seconds/update, 5.61 million core vertices/s |
| GPU activity | Approximately 98% during sustained training |
| Torch memory | 1.63 GiB allocated peak in the stopped segment; approximately 4.9 GiB reserved |
| Allocator cap | 5 GiB, with separate room needed for CUDA/native verification |
| CPU packing | Two workers, four bounded pinned batches, four Torch CPU threads |
| Training set | 66 assets / 8,499,332 source triangles |
| Prepared shards | 779,902,408 bytes (744 MiB) |
| Pilot source files | 28,637,547 bytes (27.3 MiB), counted once per path |
| Optional validation files | 146,043,210 bytes (139.3 MiB) |
| Latest model / model+AdamW checkpoint | 98,962 / 321,788 bytes |
| LibTorch | Approximately 3.7 GiB download; 6.1 GiB extracted locally |

Eight GB VRAM fits the measured configuration; 16–24 GB gives ample device
headroom. Prefer 4–8 useful CPU cores/vCPUs, about 24–32 GB RAM and 30–50 GB disk
for the toolchain, LibTorch, source, data and artifacts. These are provisioning
allowances, not measured minima. Keep compiler parallelism bounded. The Nix
store can need more disk than a minimal prebuilt container. VRAM beyond 8 GB
does not automatically enlarge the current fixed batch or allocator cap.

CUDA executes forward, loss, backward, AdamW and candidate raster work. Import,
graph packing, topology decoding and final reference audits use CPU. The measured
packing time was about 48 ms/batch across two workers. On the same CPU, a simple
capacity estimate is roughly 24 ms/update, so packing could become limiting near
1.8 times the present training rate. That is an inference from local timings,
not a remote-machine speed prediction.

## Published rental prices

The following are single-GPU Pod catalog rates from RunPod's pricing page,
updated September 27, 2026. The provider's rate guide identifies these as Secure
Cloud rates. CPU/RAM allocations are the public catalog examples; confirm the
actual selected machine. [RunPod pricing](https://www.runpod.io/pricing),
[provider rate guide](https://www.runpod.io/articles/guides/ai-server-cost).

| GPU | VRAM | vCPU / RAM | USD/hour | 10 hours |
| --- | ---: | ---: | ---: | ---: |
| RTX A5000 | 24 GB | 9 / 25 GB | $0.27 | $2.70 |
| A40 | 48 GB | 9 / 50 GB | $0.49 | $4.90 |
| RTX 4090 | 24 GB | 6 / 41 GB | $0.74 | $7.40 |
| RTX 5090 | 32 GB | 9 / 35 GB | $0.99 | $9.90 |
| L40S | 48 GB | 16 / 94 GB | $1.09 | $10.90 |
| A100 PCIe | 80 GB | 8 / 117 GB | $1.59 | $15.90 |
| H100 PCIe | 80 GB | 16 / 188 GB | $2.89 | $28.90 |

RunPod's individual GPU pages also advertise Community Cloud at **$0.34/hour
for 4090 ($3.40/10h)** and **$0.69/hour for 5090 ($6.90/10h)**. These lower rates
are official advertised prices; immediate selectable capacity was not verified.
Do not assume the Secure table's CPU/RAM allocation applies to a Community host.
[4090 tier prices](https://www.runpod.io/gpu-models/rtx-4090),
[5090 tier prices](https://www.runpod.io/gpu-models/rtx-5090).

Lambda also offers Linux VMs. Its single-GPU catalog includes Quadro RTX 6000
24 GB at $0.69/hour, A6000 48 GB at $1.09, A10 24 GB at $1.29, and A100 40 GB at
$1.99. The corresponding ten-hour compute totals are $6.90, $10.90, $12.90 and
$19.90. These include local SSD; persistent filesystems are separate. The older
Quadro RTX 6000 is different from RTX 6000 Ada or RTX Pro 6000.
[Lambda pricing](https://lambda.ai/pricing).

Vast.ai provides host-priced GPU instances with per-second billing. No current
numeric 4090/5090 offer could be verified in its publicly readable pages. A
follow-up identified the approximately $0.475/hour 5090 research result as a
cached console offer, not a reproducible live quote. It is excluded from budgets.
Inspect GPU count, allocated CPU/RAM, host reliability, disk and bandwidth prices
together when selecting a live offer.
[Vast 4090 offers](https://vast.ai/pricing/gpu/RTX-4090),
[Vast 5090 offers](https://vast.ai/pricing/gpu/RTX-5090),
[billing components](https://docs.vast.ai/guides/instances/pricing).

### Setup, storage and stopped machines

RunPod bills compute and storage per second; ingress/egress is free. Deployment
requires at least an hour's configuration cost in account credit. A running
container/volume disk costs $0.10/GB/month; a stopped persistent volume costs
$0.20. Standard network volume storage below 1 TB costs $0.07/GB/month.
At a 730-hour month, a 50 GB running disk adds about **$0.068 per ten hours**;
leaving that volume stopped for a month costs **$10**. Download results before
deleting the disposable instance and any unneeded volume.
[RunPod billing and storage](https://docs.runpod.io/pods/pricing).

Lambda bills instances by the minute until termination, including idle time;
filesystems have separate continuing charges. Its billing docs state no data
ingress/egress charges. Vast bills host-specific upload and download traffic and
storage while an instance exists, including stopped time. Downloading the
LibTorch archive onto a Vast machine also contributes to its inbound traffic.
[Lambda billing](https://docs.lambda.ai/public-cloud/billing/),
[Vast billing](https://docs.vast.ai/guides/instances/pricing).

For this short experiment, start with on-demand capacity. Interruptible rentals
require off-machine checkpoints and a verified restart path; ordinary SIGTERM
checkpointing cannot guarantee a save after immediate host loss. Account top-up
requirements and any tax are separate from measured job consumption.

## Cost for our fixed amount of training

At the measured local rate:

```text
25,000 updates  × 0.04286 s = 1,071.5 s = 17.86 min
100,000 updates × 0.04286 s = 4,286 s   = 71.43 min = 1.191 h

cost = hourly_rate × (setup_hours + audit_hours + 1.191 / measured_speedup)
       + storage + transfer + tax
```

Ten hours is a maximum experiment budget, not a required training duration.
The runner can stop earlier after two stages without a better pilot score.
The following planning scenario assumes the same training rate as the RTX 2080
and **one extra billed hour for download/build/setup/audits**. Cloud setup and
future nontrivial mesh audit durations have not been measured; this is neither a
speed guarantee nor a cost ceiling.

| RunPod GPU | 100,000 updates only at local rate | Including one extra billed hour |
| --- | ---: | ---: |
| RTX A5000 | $0.32 | $0.59 |
| A40 | $0.58 | $1.07 |
| RTX 4090 | $0.88 | $1.62 |
| RTX 5090 | $1.18 | $2.17 |
| A100 PCIe | $1.89 | $3.48 |
| H100 PCIe | $3.44 | $6.33 |

A sensible initial allowance is **$2–5 for one prepared low-cost GPU experiment**,
with the ten-hour table defining the compute budget if we deliberately use all
ten hours. A slow build, poor host CPU or repeated experiments can exceed the
small allowance. A 5090 must finish fixed training work at least 1.34 times as
fast as the $0.74/hour 4090 to win on training-only cost at these rates; A100 needs
2.15 times and H100 3.91 times. Compare total bill when setup time matters.

FP32 graph scatter/index operations and small kernels do not imply use of peak
Tensor Core throughput. The 98% utilization measurement describes GPU activity,
not achieved FLOPs. Profile launch gaps, transfers, scatter/atomic work and CPU
packing before buying a larger card or changing precision.
[NVIDIA profiling guidance](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).

## Porting the current C++ CUDA implementation

1. **Package a reproducible Linux environment.** Keep LibTorch 2.10.0 cu128,
   C++11 ABI, CUDA 12.9, the pinned source revision and existing strict native
   math flags. Use a CUDA development container or a tested Nix environment.
   The host supplies the NVIDIA driver. Copying only our Nix-linked executable
   to Ubuntu does not supply its `/nix/store` dependencies. RunPod accepts custom
   containers and Lambda provides SSH-accessible Linux VMs.
   [RunPod environments](https://www.runpod.io/product/cloud-gpus),
   [Lambda instances](https://docs.lambda.ai/public-cloud/on-demand/),
   [NVIDIA container runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html).

2. **Compile for the selected GPU and verify its driver.** Set
   `CMAKE_CUDA_ARCHITECTURES` to 86 for A5000/A40/A10/A6000, 89 for 4090/L40S,
   80 for A100, 90 for H100 or 120 for 5090. For our CUDA 12.9 build, prefer a
   Linux driver at least 575.51.03; this avoids relying on restricted minor
   compatibility and older-driver PTX JIT behavior. CUDA 12.8 introduced sm_120
   compiler support; PyTorch introduced Blackwell support with CUDA 12.8 builds.
   Rebuild and check native parity on the actual device.
   [GPU architectures](https://developer.nvidia.com/cuda/gpus),
   [CUDA 12.9 driver table](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html),
   [minor compatibility limitations](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html),
   [CUDA 12.8 compiler](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html),
   [PyTorch Blackwell support](https://pytorch.org/blog/pytorch-2-7/).

3. **Move prepared data and small checkpoints.** Preserve shard/index checksums,
   pilot manifests, relative source paths and the model hash. The `data` entry in
   this worktree is an absolute local symlink; construct real data paths remotely.
   Download LibTorch directly on the remote host and verify `libtorch.sha256`.
   Prepared shards plus the pilot need roughly 771 MiB, so transferring the full
   source corpus is unnecessary for the next pilot. Preserve training/pilot family
   exclusions; validation remains selection-only and held-out remains release-only.

4. **Choose warm start or explicit optimizer migration.** A fresh run can already
   initialize from the portable `.blzn` export; that resets AdamW state and update
   count and must be recorded as a warm start. LibTorch archives can remap model
   and optimizer tensors to another CUDA device, and our loader already supplies
   the target device. However, our resume contract also requires the executable
   SHA to match. Recompiling for another GPU changes that SHA. Full AdamW resume
   across such a rebuild needs an explicit migration entry point that validates
   the old schema/data/optimizer and records parent checkpoint plus new executable
   provenance. That migration entry point is **not implemented**; do not bypass
   the contract by editing its JSON. Cross-GPU bitwise equality is not promised.
   [LibTorch archive device remapping](https://docs.pytorch.org/cppdocs/api/serialize/archives.html).

5. **Validate before spending the training budget.** Run the CUDA contracts and
   a full saved-model pilot first. Then, only when training is requested, measure
   a short run using identical batch, data, seed, precision and optimizer; record
   effective CPU allocation and device details. Compare seconds/update, vertices/s,
   memory and USD/100,000 updates after warmup. Require finite gradients, actual
   parameter changes, checkpoint/optimizer restoration and native inference parity.
   Keep a separate run directory and source/binary hashes for each environment.

6. **Finish and export.** Audit the checkpoint, download model, optimizer, logs
   and hashes, then terminate paid compute. Use a container supervisor or direct
   Node runner with a deadline; containers may not support our local systemd user
   service commands. Also enforce the rental lifetime externally: a runner's
   deadline alone does not stop provider billing. Include `audit.mjs` with the
   runner. The roughly 100 KiB `.blzn` export can return to the RTX 2080 for native
   inference without LibTorch. A Docker image and provider launch/termination
   automation remain future integration work, not completed deployment.
