#!/usr/bin/env bash
# Runs a long job on mains power with sleep and lid-close inhibited.
#
#   scripts/long_job.sh <name> <command...>
#
# Output goes to the terminal and runs/<name>.log. Start and end times go to
# runs/jobs.log. The job's stdin is /dev/null. The name "jobs" is reserved.
#
# Exit codes: the job's own, 128+n when signal n killed it, 2 for bad usage,
# 3 when mains power is not confirmed, 4 when the job reported no exit status
# (logged as exit=unknown), for example because the inhibitor could not start it.
#
# SIGTERM, SIGINT or SIGHUP to the wrapper goes to the job's whole process group,
# and the wrapper then waits for the job to finish, so an END line means the job
# has stopped. A child the job started with `&` from a non-interactive shell ignores
# SIGINT, so use SIGTERM if Ctrl-C leaves the wrapper waiting. A START with no END means the wrapper or the machine died without
# warning, and the job may still be running. Before restarting, check that
# `systemd-inhibit --list` shows no sandeha entry for this job.
set -uo pipefail

if (($# < 2)) || [[ ! $1 =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || [[ $1 == jobs ]]; then
  echo "usage: long_job.sh <name> <command...>  (name: letters, digits, . _ -; not 'jobs')" >&2
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

state=$(mktemp -d) || exit 1

# Runs under the inhibitor. Job control gives the job its own process group, so
# it can be signalled while the inhibitor stays held. The status is recorded here
# because systemd-inhibit reports any job killed by a signal as plain exit 1.
run_job='
  state=$1
  shift
  set -m
  "$@" </dev/null &
  job=$!
  echo "$job" >"$state/pgid"
  [[ -e $state/stop ]] && kill -TERM -- "-$job"
  wait "$job"
  echo $? >"$state/status"
'

# The stop file covers a signal that arrives before the job's group is known:
# run_job checks for it after recording the group, and this checks the group
# after creating the file, so at least one of the two sends the signal.
forward() {
  : >"$state/stop"
  [[ -s $state/pgid ]] && kill "-$1" -- "-$(<"$state/pgid")" 2>/dev/null
}
trap 'forward TERM' TERM
trap 'forward INT' INT
trap 'forward HUP' HUP

echo "$(stamp) START $name: $*" >>runs/jobs.log
started=$SECONDS

# In the background with job control, so that Ctrl-C reaches only this wrapper
# and the log keeps recording until the job has stopped.
set -m
systemd-inhibit --what=sleep:idle:handle-lid-switch --who=sandeha --why="long job: $name" \
  bash -c "$run_job" long_job "$state" "$@" 2>&1 </dev/null | tee -a "runs/$name.log" &
pipeline=$!
set +m

# A trapped signal interrupts wait, so wait again until the pipeline has ended.
wait "$pipeline"
while kill -0 "$pipeline" 2>/dev/null; do wait "$pipeline"; done

status=$(cat "$state/status" 2>/dev/null)
rm -rf "$state"

if [[ -z $status ]]; then
  echo "long_job: '$name' reported no exit status; the inhibitor may have failed to start it." >&2
fi
echo "$(stamp) END $name: exit=${status:-unknown} elapsed=$((SECONDS - started))s" >>runs/jobs.log
exit "${status:-4}"
