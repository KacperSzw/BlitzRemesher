#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
revision=$(git rev-parse HEAD 2>/dev/null || true)
if [[ -z "$revision" || "$revision" == HEAD ]]; then revision=uncommitted; fi
source_hash=$(rg --files src include tools tests cmake CMakeLists.txt CMakePresets.json flake.nix flake.lock -g '*.cpp' -g '*.hpp' -g '*.h' -g '*.c' -g '*.cmake' -g '*.sh' -g '*.json' -g '*.nix' -g '*.lock' -g 'CMakeLists.txt' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d ' ' -f 1)
dirty_hash=$(git diff --binary HEAD 2>/dev/null | sha256sum | cut -d ' ' -f 1) || dirty_hash=uncommitted
stamp_output=${1:-research/build.json}
binary_hash=""
if [[ -n "${2:-}" ]]; then binary_hash=$(sha256sum "$2" | cut -d ' ' -f 1); fi
jq -n --arg revision "$revision" --arg source_hash "$source_hash" --arg dirty_hash "$dirty_hash" \
  --arg binary_hash "$binary_hash" \
  '{git_revision:$revision,source_tree_sha256:$source_hash,tracked_diff_sha256:$dirty_hash} + (if $binary_hash == "" then {} else {binary_sha256:$binary_hash} end)' > "$stamp_output"
