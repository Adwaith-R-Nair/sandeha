#!/usr/bin/env bash
# Runs a long job on mains power with sleep and lid-close inhibited.
#
#   scripts/long_job.sh <name> <command...>
#
# Output goes to the terminal and runs/<name>.log. Start and end times go to
# runs/jobs.log; a START with no matching END means the wrapper itself was killed.
# Exit codes: the command's own, 2 for bad usage, 3 when mains power is not confirmed.
set -uo pipefail

if (($# < 2)) || [[ ! $1 =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "usage: long_job.sh <name> <command...>  (name: letters, digits, . _ -)" >&2
  exit 2
fi
name=$1
shift

mains_online() {
  local supply
  for supply in "${SANDEHA_POWER_SUPPLY_DIR:-/sys/class/power_supply}"/*; do
    [[ -r $supply/type && -r $supply/online ]] || continue
    [[ $(<"$supply/type") == Mains && $(<"$supply/online") == 1 ]] && return 0
  done
  return 1
}

if ! mains_online; then
  echo "long_job: mains power is offline or could not be confirmed; refusing to start '$name'." >&2
  exit 3
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
mkdir -p runs

stamp() { date --iso-8601=seconds; }

echo "$(stamp) START $name: $*" >>runs/jobs.log
started=$SECONDS

systemd-inhibit --what=sleep:idle:handle-lid-switch --who=sandeha --why="long job: $name" \
  "$@" 2>&1 | tee -a "runs/$name.log"
status=${PIPESTATUS[0]}

echo "$(stamp) END $name: exit=$status elapsed=$((SECONDS - started))s" >>runs/jobs.log
exit "$status"
