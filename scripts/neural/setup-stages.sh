# Sourced by cloud-setup.sh after `set -euo pipefail`. A stage is deliberately
# not invoked through `if`/`||`: Bash would disable errexit inside functions.
blitz_stage_name=
blitz_stage_started_ms=
blitz_stage_clock() { date +%s%3N; }
blitz_stage_begin() {
  [[ -z "$blitz_stage_name" && "$1" =~ ^[a-z0-9-]+$ ]]
  blitz_stage_name=$1
  blitz_stage_started_ms=$(blitz_stage_clock)
  printf '{"stage":"%s","event":"start","timestamp_ms":%s}\n' "$blitz_stage_name" "$blitz_stage_started_ms" >> "$BLITZ_SETUP_TIMINGS"
}
blitz_stage_end() {
  local code=$1 ended
  ended=$(blitz_stage_clock)
  printf '{"stage":"%s","event":"end","timestamp_ms":%s,"elapsed_ms":%s,"exit_code":%s}\n' "$blitz_stage_name" "$ended" "$((ended-blitz_stage_started_ms))" "$code" >> "$BLITZ_SETUP_TIMINGS"
  blitz_stage_name=
}
blitz_stage_exit() {
  local code=$?
  trap - EXIT
  # A broken evidence destination must not replace the original failure code.
  if [[ -n "$blitz_stage_name" ]]; then blitz_stage_end "$code" || true; fi
  exit "$code"
}
blitz_stage() {
  blitz_stage_begin "$1"
  shift
  "$@"
  blitz_stage_end 0
}
trap blitz_stage_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
