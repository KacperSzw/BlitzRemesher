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
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader,nounits | tee /workspace/results/gpu.csv
node --input-type=module <<'JS'
import fs from 'node:fs';
import {deployment,verifyDevice} from './research/neural/runpod-profile.mjs';
const device=verifyDevice(fs.readFileSync('/workspace/results/gpu.csv','utf8'));
fs.writeFileSync('/workspace/results/deployment.json',JSON.stringify({deployment,device},null,2)+'\n');
JS
mkdir -p /opt/blitz
curl --fail --location --retry 2 --max-time 600 'https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.10.0%2Bcu128.zip' -o /opt/blitz/libtorch.zip
printf '%s  %s\n' '429aa9fead3cf3d557e7c310442a1fae3879cdc14a469ff452043b39b61666a9' '/opt/blitz/libtorch.zip' | sha256sum --check
unzip -q /opt/blitz/libtorch.zip -d /opt/blitz
rm /opt/blitz/libtorch.zip
blitz_cuda_arch=$(node --input-type=module -e "import {deployment} from './research/neural/runpod-profile.mjs'; console.log(deployment.cuda_architecture)")
cmake -S . -B build/neural -G Ninja -DCMAKE_BUILD_TYPE=Release -DBLITZ_CUDA=ON -DBLITZ_NEURAL_TRAIN=ON -DCMAKE_CUDA_ARCHITECTURES="$blitz_cuda_arch" -DBLITZ_LIBTORCH_ROOT=/opt/blitz/libtorch
cmake --build build/neural -j2
ldd build/neural/blitz-neural-train > /workspace/results/trainer-dependencies.txt
ctest --test-dir build/neural --output-on-failure | tee /workspace/results/ctest.log
if [[ "${BLITZ_ACTION_V2:-0}" != 1 ]]; then
  build/neural/blitz-neural-train /workspace/dataset /workspace/results/prefetch --core 8 --batch 2 --workers 8 --check-prefetch 28
fi
git rev-parse HEAD > /workspace/results/git-revision
dpkg-query -W > /workspace/results/packages.txt
