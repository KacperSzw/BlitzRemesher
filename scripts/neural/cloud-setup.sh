#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
# sshd need not inherit the CUDA image's Docker PATH.
export PATH="/usr/local/cuda/bin:$PATH"
export CUDACXX=/usr/local/cuda/bin/nvcc
export CUDAToolkit_ROOT=/usr/local/cuda
"$CUDACXX" --version > /workspace/results/cuda-toolkit.txt
apt-get update -qq
apt-get install -y --no-install-recommends g++ gdb cmake ninja-build pkg-config git ripgrep nodejs curl ca-certificates unzip libssl-dev nlohmann-json3-dev libcurl4-openssl-dev libarchive-dev
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
  blitz_glx=$(ldconfig -p | rg -m1 -o '/[^ ]*/libGLX_nvidia[.]so[.]0' || true)
  if [[ -n "$blitz_glx" ]]; then ldd "$blitz_glx" > /workspace/results/nvidia-glx-dependencies.txt; fi
  blitz_egl=$(ldconfig -p | rg -m1 -o '/[^ ]*/libEGL_nvidia[.]so[.]0' || true)
  if [[ -n "$blitz_egl" ]]; then ldd "$blitz_egl" > /workspace/results/nvidia-egl-dependencies.txt; fi
  # Preserve the vendor's ICD. NVIDIA defaults to GLX; EGL is needed when X11
  # client libraries are unavailable, not merely when no X server is running.
  node --input-type=module <<'JS'
import fs from 'node:fs';
import {selectNvidiaIcd} from './scripts/neural/nvidia-icd.mjs';
import {validateOptimizationRequest} from './scripts/neural/teacher-optimization.mjs';
const source=['/etc/vulkan/icd.d/nvidia_icd.json','/usr/share/vulkan/icd.d/nvidia_icd.json'].find(p=>fs.existsSync(p));
if(!source)throw new Error('Missing host-injected NVIDIA Vulkan ICD');
const requested=process.env.BLITZ_TEACHER_OPTIMIZATION==='1'
  ? validateOptimizationRequest(JSON.parse(fs.readFileSync('/workspace/optimization/request.json'))).vulkan_icd??'vendor'
  : 'vendor';
const original=fs.readFileSync(source,'utf8');
const selected=selectNvidiaIcd(original,{sourcePath:source,selectedPath:'/workspace/results/nvidia-selected.json',requested});
selected.provenance.original_copy_path='/workspace/results/nvidia-icd.json';
fs.writeFileSync(selected.provenance.original_copy_path,original);
fs.writeFileSync(selected.provenance.selected_path,selected.contents);
fs.writeFileSync('/workspace/results/nvidia-icd-selection.json',JSON.stringify(selected.provenance,null,2)+'\n');
JS
  blitz_selected_library=$(node -e "console.log(require('/workspace/results/nvidia-icd-selection.json').selected_library_path)")
  blitz_library_resolution=manifest_path
  if [[ "$blitz_selected_library" != */* ]]; then
    blitz_library_resolution=ldconfig
    blitz_selected_library=$(ldconfig -p | awk -v library="$blitz_selected_library" '$1 == library && !found {print $NF; found=1}')
  fi
  if [[ -z "$blitz_selected_library" || ! -f "$blitz_selected_library" ]]; then
    echo 'Selected NVIDIA ICD library is unavailable; refusing a backend fallback' >&2
    exit 1
  fi
  ldd "$blitz_selected_library" > /workspace/results/nvidia-selected-dependencies.txt
  if rg -q 'not found' /workspace/results/nvidia-selected-dependencies.txt; then
    cat /workspace/results/nvidia-selected-dependencies.txt
    exit 1
  fi
  node --input-type=module - "$blitz_selected_library" "$blitz_library_resolution" <<'JS'
import fs from 'node:fs';
import {createHash} from 'node:crypto';
const file='/workspace/results/nvidia-icd-selection.json';
const selected=JSON.parse(fs.readFileSync(file));
selected.dependency_library_resolution=process.argv[3];
selected.dependency_library_path=fs.realpathSync(process.argv[2]);
selected.dependency_library_sha256=createHash('sha256').update(fs.readFileSync(selected.dependency_library_path)).digest('hex');
fs.writeFileSync(file,JSON.stringify(selected,null,2)+'\n');
JS
  unset VK_ICD_FILENAMES
  export VK_DRIVER_FILES=/workspace/results/nvidia-selected.json
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
if [[ "${BLITZ_CORE_VALIDATION:-0}" == 1 || "${BLITZ_TEACHER_OPTIMIZATION:-0}" == 1 ]]; then blitz_acquisition=OFF; fi
if [[ "${BLITZ_TEACHER_OPTIMIZATION:-0}" == 1 ]]; then
  blitz_baseline=$(node --input-type=module <<'JS'
import fs from 'node:fs';
import {validateOptimizationRequest} from './scripts/neural/teacher-optimization.mjs';
console.log(validateOptimizationRequest(JSON.parse(fs.readFileSync('/workspace/optimization/request.json'))).baseline_revision);
JS
)
  git worktree add --detach /workspace/baseline "$blitz_baseline"
  node scripts/neural/baseline-overlay.mjs prepare /workspace/baseline /workspace/optimization/request.json /workspace/results/baseline-build.json
  cmake -S /workspace/baseline -B /workspace/baseline/build/neural -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DBLITZ_CUDA=ON -DBLITZ_VULKAN=ON \
    -DBLITZ_NEURAL_TRAIN=OFF -DBLITZ_ACQUISITION=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="$blitz_cuda_arch"
  cmake --build /workspace/baseline/build/neural --target blitz-neural-placement-prepare -j2
  node scripts/neural/baseline-overlay.mjs record /workspace/results/baseline-build.json /workspace/baseline/build/neural/blitz-neural-placement-prepare
  sha256sum /workspace/baseline/build/neural/blitz-neural-placement-prepare > /workspace/results/baseline-binary.sha256
  printf '%s\n' "$blitz_baseline" > /workspace/results/baseline-revision
fi
cmake -S . -B build/neural -G Ninja -DCMAKE_BUILD_TYPE=Release -DBLITZ_CUDA=ON -DBLITZ_VULKAN="$blitz_vulkan" -DBLITZ_NEURAL_TRAIN="$blitz_training" -DBLITZ_ACQUISITION="$blitz_acquisition" -DCMAKE_CUDA_ARCHITECTURES="$blitz_cuda_arch" -DBLITZ_LIBTORCH_ROOT=/opt/blitz/libtorch
cmake --build build/neural -j2
if [[ "$blitz_training" == ON ]]; then ldd build/neural/blitz-neural-train > /workspace/results/trainer-dependencies.txt; fi
if [[ "${BLITZ_CORE_VALIDATION:-0}" != 1 && "${BLITZ_TEACHER_OPTIMIZATION:-0}" != 1 ]]; then
  ctest --test-dir build/neural --output-on-failure | tee /workspace/results/ctest.log
fi
if [[ "${BLITZ_ACTION_V2:-0}" != 1 ]]; then
  build/neural/blitz-neural-train /workspace/dataset /workspace/results/prefetch --core 8 --batch 2 --workers 8 --check-prefetch 28
fi
git rev-parse HEAD > /workspace/results/git-revision
dpkg-query -W > /workspace/results/packages.txt
