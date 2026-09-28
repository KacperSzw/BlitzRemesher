#!/usr/bin/env bash
# Run inside the pinned Nix shell. Each invocation is a resumable <=50 minute batch.
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-build}
variant=${2:-production}
case "$variant" in
  production|packed) ;;
  *) echo "Precision variant '$variant' is archived; see research/precision/ARCHIVE.md." >&2; exit 1 ;;
esac
root=research/precision/production
mkdir -p "$root/builds" "$root/logs"
build=build/precision-production
if [[ "$mode" == build ]]; then
  cmake -S . -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release
  cmake --build "$build" -j "${PRECISION_BUILD_JOBS:-2}"
  ctest --test-dir "$build" -R 'precision-contracts|c-abi' --output-on-failure
  ctest --test-dir "$build" --output-on-failure
  bash tools/stamp-build.sh "$root/builds/production.json" "$build/blitz"
  bash tools/stamp-build.sh "$root/builds/micro.json" "$build/blitz-precision-bench"
  exit
fi
if [[ "$mode" == micro ]]; then
  "$build/blitz-precision-bench" > "$root/micro.json"
  exit
fi
scenario=${3:-pilot}
repeat=${4:-1}
manifest=research/pilot.json
config=research/configs/pilot-qem.json
extra=()
case "$scenario" in
  pilot) ;;
  validation) manifest=research/corpus.json; extra=(--split validation) ;;
  normals) config=research/configs/pilot-normals.json ;;
  attributes) config=research/configs/pilot-attributes.json ;;
  large) config=research/round4/configs/s512-cap8.json; extra=(--limit 2) ;;
  baseline-*) extra=(--baseline "${scenario#baseline-}" --baseline-dir build/precision-baselines) ;;
  *) echo "Unknown precision scenario: $scenario" >&2; exit 1 ;;
esac
[[ "$mode" == bench ]] || { echo "Use build, micro, or bench" >&2; exit 1; }
"$build/blitz" bench "$manifest" "$config" "research/runs/precision-production/production-$scenario-$repeat" \
  --build-stamp "$root/builds/production.json" --minutes 50 "${extra[@]}"
