#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
revision=$(git rev-parse HEAD 2>/dev/null || true)
if [[ -z "$revision" || "$revision" == HEAD ]]; then revision=uncommitted; fi
source_hash=$(rg --files src include tools tests cmake -g '*.cpp' -g '*.hpp' -g '*.h' -g '*.c' -g '*.cmake' -g '*.sh' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d ' ' -f 1)
dirty_hash=$(git diff --binary HEAD 2>/dev/null | sha256sum | cut -d ' ' -f 1) || dirty_hash=uncommitted
jq -n --arg revision "$revision" --arg source_hash "$source_hash" --arg dirty_hash "$dirty_hash" \
  '{git_revision:$revision,source_tree_sha256:$source_hash,tracked_diff_sha256:$dirty_hash}' > research/build.json
