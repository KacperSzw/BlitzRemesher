#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
# sshd need not inherit the CUDA image's Docker PATH.
export PATH="/usr/local/cuda/bin:$PATH"
export CUDACXX=/usr/local/cuda/bin/nvcc
export CUDAToolkit_ROOT=/usr/local/cuda
"$CUDACXX" --version > /workspace/results/cuda-toolkit.txt
apt-get update -qq
apt-get install -y --no-install-recommends g++ cmake ninja-build pkg-config git ripgrep nodejs curl ca-certificates unzip libssl-dev nlohmann-json3-dev libcurl4-openssl-dev libarchive-dev
if [[ "${BLITZ_PIPELINE_VALIDATION:-0}" == 1 ]]; then
  apt-get install -y --no-install-recommends cuda-nsight-systems-12-9
fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader,nounits | tee /workspace/results/gpu.csv
blitz_vulkan=OFF
if [[ "${BLITZ_HARDWARE_VALIDATION:-0}" == 1 ]]; then
  blitz_vulkan=ON
  # NVIDIA mounts its ICD into the CUDA container, but its GLVND/X11 runtime
  # dependencies must come from the container. No display or kernel driver is installed.
  apt-get install -y --no-install-recommends libvulkan-dev vulkan-tools vulkan-validationlayers glslang-tools libgl1 libegl1 libx11-6 libxext6
  ldconfig -p > /workspace/results/graphics-libraries.txt
  for blitz_icd in /usr/share/vulkan/icd.d/*nvidia*.json /etc/vulkan/icd.d/*nvidia*.json; do
    if [[ -f "$blitz_icd" ]]; then cat "$blitz_icd"; fi
  done > /workspace/results/nvidia-icd.json
  blitz_glx=$(ldconfig -p | rg -m1 -o '/[^ ]*/libGLX_nvidia[.]so[.]0' || true)
  if [[ -n "$blitz_glx" ]]; then ldd "$blitz_glx" > /workspace/results/nvidia-glx-dependencies.txt; fi
  blitz_egl=$(ldconfig -p | rg -m1 -o '/[^ ]*/libEGL_nvidia[.]so[.]0' || true)
  if [[ -n "$blitz_egl" ]]; then ldd "$blitz_egl" > /workspace/results/nvidia-egl-dependencies.txt; fi
  # NVIDIA documents EGL as its X11-independent Vulkan ICD. Preserve the
  # injected host driver's API version and directory; only select its EGL entry.
  node --input-type=module <<'JS'
import fs from 'node:fs';
const source=['/etc/vulkan/icd.d/nvidia_icd.json','/usr/share/vulkan/icd.d/nvidia_icd.json'].find(p=>fs.existsSync(p));
if(!source)throw new Error('Missing host-injected NVIDIA Vulkan ICD');
const icd=JSON.parse(fs.readFileSync(source,'utf8'));
icd.ICD.library_path=icd.ICD.library_path.replace(/libGLX_nvidia[.]so[.]0$/,'libEGL_nvidia.so.0');
if(!icd.ICD.library_path.endsWith('libEGL_nvidia.so.0'))throw new Error('Unsupported NVIDIA Vulkan ICD entry');
fs.writeFileSync('/workspace/results/nvidia-headless.json',JSON.stringify(icd,null,2)+'\n');
JS
  export VK_DRIVER_FILES=/workspace/results/nvidia-headless.json
  vulkaninfo --summary > /workspace/results/vulkan.txt
fi
node --input-type=module <<'JS'
import fs from 'node:fs';
import {deployment,verifyDevice} from './scripts/neural/runpod-profile.mjs';
const device=verifyDevice(fs.readFileSync('/workspace/results/gpu.csv','utf8'));
if(deployment.graphics&&!fs.readFileSync('/workspace/results/vulkan.txt','utf8').includes(device.name))throw new Error('The allocated NVIDIA GPU is missing from Vulkan; graphics driver exposure is required');
fs.writeFileSync('/workspace/results/deployment.json',JSON.stringify({deployment,device},null,2)+'\n');
JS
blitz_training=OFF
if [[ "${BLITZ_EVALUATE_ONLY:-0}" != 1 ]]; then
blitz_training=ON
mkdir -p /opt/blitz
curl --fail --location --retry 2 --max-time 600 'https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.10.0%2Bcu128.zip' -o /opt/blitz/libtorch.zip
printf '%s  %s\n' '429aa9fead3cf3d557e7c310442a1fae3879cdc14a469ff452043b39b61666a9' '/opt/blitz/libtorch.zip' | sha256sum --check
unzip -q /opt/blitz/libtorch.zip -d /opt/blitz
rm /opt/blitz/libtorch.zip
fi
blitz_cuda_arch=$(node --input-type=module -e "import {deployment} from './scripts/neural/runpod-profile.mjs'; console.log(deployment.cuda_architecture)")
blitz_acquisition=ON
if [[ "${BLITZ_CORE_VALIDATION:-0}" == 1 ]]; then blitz_acquisition=OFF; fi
cmake -S . -B build/neural -G Ninja -DCMAKE_BUILD_TYPE=Release -DBLITZ_CUDA=ON -DBLITZ_VULKAN="$blitz_vulkan" -DBLITZ_NEURAL_TRAIN="$blitz_training" -DBLITZ_ACQUISITION="$blitz_acquisition" -DCMAKE_CUDA_ARCHITECTURES="$blitz_cuda_arch" -DBLITZ_LIBTORCH_ROOT=/opt/blitz/libtorch
cmake --build build/neural -j2
if [[ "$blitz_training" == ON ]]; then ldd build/neural/blitz-neural-train > /workspace/results/trainer-dependencies.txt; fi
if [[ "${BLITZ_CORE_VALIDATION:-0}" != 1 ]]; then
  ctest --test-dir build/neural --output-on-failure | tee /workspace/results/ctest.log
fi
if [[ "${BLITZ_ACTION_V2:-0}" != 1 ]]; then
  build/neural/blitz-neural-train /workspace/dataset /workspace/results/prefetch --core 8 --batch 2 --workers 8 --check-prefetch 28
fi
git rev-parse HEAD > /workspace/results/git-revision
dpkg-query -W > /workspace/results/packages.txt
