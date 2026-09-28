#!/usr/bin/env bash
# Run inside nix develop after the production pilot/large replays complete.
set -euo pipefail
cd "$(dirname "$0")/../.."
for scenario in pilot large; do
  repeat=1
  [[ "$scenario" == large ]] && repeat=2
  reference="research/runs/precision-v2/packed-$scenario-$repeat"
  actual="research/runs/precision-production/production-$scenario-1"
  jq -e -s '
    .[0] as $a | .[1] as $b |
    all(["manifest_sha256","config_sha256","camera_sha256","protocol_sha256",
         "input_format","split","limit","method","backend","threads",
         "quadric_bytes","candidate_bytes","packed_coverage"][]; $a[.] == $b[.])
  ' "$reference/metadata.json" "$actual/metadata.json" > /dev/null
  jq -e -s '
    .[0] as $a | .[1] as $b | $a.complete and $b.complete and
    all(["expected","completed","fallbacks","categories","score"][]; $a[.] == $b[.])
  ' "$reference/summary.json" "$actual/summary.json" > /dev/null
  count=0
  for row in "$reference"/rows/*.json; do
    jq -e -s '
      .[0] as $a | .[1] as $b | $a.complete and $b.complete and
      (($a.failed // false) == false) and (($b.failed // false) == false) and
      all(["id","category","canonical_attributes_sha256","output_sha256",
           "attributes_sha256","ratio","final_ratio","last_three_ratio",
           "fallback","result","numerics"][]; $a[.] == $b[.])
    ' "$row" "$actual/rows/${row##*/}" > /dev/null
    ((count+=1))
  done
  actual_rows=("$actual"/rows/*.json)
  [[ "${#actual_rows[@]}" == "$count" ]]
  jq -n --arg scenario "$scenario" --argjson assets "$count" \
    '{scenario:$scenario,assets:$assets,identical_outputs_attributes_audits_and_numerics:true}'
done
