#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends g++ cmake ninja-build pkg-config git ripgrep nodejs curl ca-certificates unzip libssl-dev nlohmann-json3-dev libcurl4-openssl-dev libarchive-dev
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader,nounits | tee /workspace/results/gpu.csv
node --input-type=module <<'JS'
import fs from 'node:fs';
const rows=fs.readFileSync('/workspace/results/gpu.csv','utf8').trim().split('\n');
const [name,cap,memory,driver]=rows[0].split(',').map(s=>s.trim());
const version=driver.split('.').map(Number), minimum=[575,51,3];
let older=false;for(let i=0;i<3;i++){if(version[i]!==minimum[i]){older=version[i]<minimum[i];break;}}
if(rows.length!==1||!name.includes('RTX 5090')||cap!=='12.0'||Number(memory)<30000||older)throw new Error('GPU or driver does not satisfy the RTX 5090 CUDA 12.9 contract');
JS
mkdir -p /opt/blitz
curl --fail --location --retry 2 --max-time 600 'https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.10.0%2Bcu128.zip' -o /opt/blitz/libtorch.zip
printf '%s  %s\n' '429aa9fead3cf3d557e7c310442a1fae3879cdc14a469ff452043b39b61666a9' '/opt/blitz/libtorch.zip' | sha256sum --check
unzip -q /opt/blitz/libtorch.zip -d /opt/blitz
rm /opt/blitz/libtorch.zip
cmake -S . -B build/neural -G Ninja -DCMAKE_BUILD_TYPE=Release -DBLITZ_CUDA=ON -DBLITZ_NEURAL_TRAIN=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DBLITZ_LIBTORCH_ROOT=/opt/blitz/libtorch
cmake --build build/neural -j2
ctest --test-dir build/neural --output-on-failure | tee /workspace/results/ctest.log
build/neural/blitz-neural-train /workspace/dataset /workspace/results/prefetch --core 8 --batch 2 --workers 4 --check-prefetch 12
git rev-parse HEAD > /workspace/results/git-revision
dpkg-query -W > /workspace/results/packages.txt
/usr/local/cuda/bin/nvcc --version > /workspace/results/cuda-toolkit.txt
