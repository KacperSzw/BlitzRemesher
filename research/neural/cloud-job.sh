#!/usr/bin/env bash
# Executed once inside the rented container. No management credential is present.
set -euo pipefail
cd /workspace/project
mkdir -p /workspace/results
exec >> /workspace/results/bootstrap.log 2>&1
finish() {
  local code=$?
  trap - EXIT
  printf '%s\n' "$code" > /workspace/results/job-exit-code
  # Publish immutable evidence for checksum-verified collection, even on failure.
  # Archive diagnostics live outside the tree being archived.
  tar -czf /workspace/results.tar.gz.part -C /workspace results > /workspace/archive.log 2>&1
  mv /workspace/results.tar.gz.part /workspace/results.tar.gz
  sha256sum /workspace/results.tar.gz > /workspace/results.tar.gz.sha256
  touch /workspace/job-finished
  exit "$code"
}
trap finish EXIT
setup_deadline_ms=$1
training_deadline_ms=$2
export BLITZ_ACTION_V2=0
if [[ "${4:-vertex-v1}" == action-v2* || "${4:-vertex-v1}" == gpu-refactor || "${4:-vertex-v1}" == hardware-validation ]]; then export BLITZ_ACTION_V2=1; fi
export BLITZ_HARDWARE_VALIDATION=0
if [[ "${4:-vertex-v1}" == hardware-validation || "${4:-vertex-v1}" == gpu-refactor ]]; then
  export BLITZ_HARDWARE_VALIDATION=1
  export VK_DRIVER_FILES=/workspace/results/nvidia-headless.json
fi
export BLITZ_EVALUATE_ONLY=0
if [[ "${4:-vertex-v1}" == action-v2-evaluate ]]; then export BLITZ_EVALUATE_ONLY=1; fi
setup_seconds=$((setup_deadline_ms / 1000 - $(date +%s)))
((setup_seconds > 0))
timeout --signal=TERM --kill-after=10s "${setup_seconds}s" bash research/neural/cloud-setup.sh
if [[ "${4:-vertex-v1}" == hardware-validation ]]; then
  node research/neural/hardware-job.mjs "$setup_deadline_ms" "$training_deadline_ms" "$3"
elif [[ "${4:-vertex-v1}" == gpu-refactor ]]; then
  node research/neural/refactor-job.mjs "$setup_deadline_ms" "$training_deadline_ms" "$3"
elif [[ "$BLITZ_ACTION_V2" == 1 ]]; then
  node research/neural/action-job.mjs "$setup_deadline_ms" "$training_deadline_ms" "$3" "$4"
else
  node research/neural/cloud-job.mjs "$setup_deadline_ms" "$training_deadline_ms" "$3"
fi
