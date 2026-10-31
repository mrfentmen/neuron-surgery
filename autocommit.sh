#!/bin/bash
# autocommit.sh - keep the repo checkpointed while a long run is unattended.
#
# It never touches the sweep scripts. It watches the working tree and commits
# once the DATA files have stopped growing (QUIET), so a jsonl is captured at a
# job boundary instead of mid-line. MAXWAIT bounds the loss when a job writes
# continuously for hours. Logs ride along with each commit but are ignored when
# deciding whether the run is quiet, because they append every minute and would
# otherwise cause a commit every cycle.
#
# Start:  nohup caffeinate -i bash autocommit.sh >> autocommit.log 2>&1 &
# Stop:   touch NOCOMMIT   (checked every cycle; exits cleanly)
# Once:   bash autocommit.sh --once

cd "$(dirname "$0")" || exit 1
LOG=autocommit.log
QUIET=${QUIET:-180}
MAXWAIT=${MAXWAIT:-1800}
ONCE=0
[ "$1" = "--once" ] && ONCE=1

say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

workers() {
  pgrep -f "social_sweep.py|layer_sweep.py|run_queue.sh|run_layers.sh|overnight.sh" \
    | wc -l | tr -d ' '
}

# Everything except the control file, our own log, and git locks.
EXCL=(-- . ':!STOP' ':!*.lock' ':!autocommit.log' ':!NOCOMMIT')
# Only these decide whether the run is quiet.
DATA=(sweep_*.jsonl layer_*.jsonl *.json *.py *.md)

newest_data() {
  local files m t=0
  files=$(git status --porcelain -- "${DATA[@]}" 2>/dev/null | sed 's/^...//')
  [ -z "$files" ] && { echo 0; return; }
  for f in $files; do
    m=$(stat -f %m "$f" 2>/dev/null) || continue
    [ "$m" -gt "$t" ] && t=$m
  done
  echo "$t"
}

commit_push() {
  local rows files
  git add -A "${EXCL[@]}" >> "$LOG" 2>&1
  if git diff --cached --quiet; then return 0; fi
  rows=$(cat sweep_*.jsonl layer_*.jsonl 2>/dev/null | wc -l | tr -d ' ')
  files=$(git diff --cached --name-only | wc -l | tr -d ' ')
  if ! git commit -q -m "Auto-checkpoint $(date '+%F %H:%M')" \
       -m "${files} files, ${rows} sweep rows, committed unattended." >> "$LOG" 2>&1; then
    say "commit failed"
    return 1
  fi
  say "committed $(git rev-parse --short HEAD) ($files files, $rows rows)"
  if git push -q origin main >> "$LOG" 2>&1; then
    say "pushed $(git rev-parse --short HEAD)"
    return 0
  fi
  say "push rejected, fetching"
  git fetch -q origin >> "$LOG" 2>&1
  if [ "$(git rev-list --count HEAD..origin/main 2>/dev/null || echo 0)" -gt 0 ]; then
    if git -c rebase.autoStash=true pull --rebase -q origin main >> "$LOG" 2>&1; then
      say "rebased onto origin"
    else
      git rebase --abort >> "$LOG" 2>&1
      say "rebase failed, commit kept locally; retrying next cycle"
      return 1
    fi
  fi
  if git push -q origin main >> "$LOG" 2>&1; then
    say "pushed $(git rev-parse --short HEAD)"
  else
    say "push still failing; commit kept locally"
  fi
}

say "start pid=$$ quiet=${QUIET}s maxwait=${MAXWAIT}s once=$ONCE"
last=$(date +%s)
while true; do
  if [ -f NOCOMMIT ]; then say "NOCOMMIT present, exiting"; break; fi
  w=$(workers)
  now=$(date +%s)
  dirty=$(git status --porcelain -- "${DATA[@]}" | wc -l | tr -d ' ')
  if [ "$dirty" -gt 0 ]; then
    n=$(newest_data)
    idle=$(( now - n ))
    if [ "$idle" -ge "$QUIET" ] || [ $(( now - last )) -ge "$MAXWAIT" ]; then
      say "cycle workers=$w dirty=$dirty data_idle=${idle}s since_commit=$(( now - last ))s"
      commit_push && last=$now
    fi
  fi
  [ "$ONCE" = 1 ] && break
  if [ "$w" -eq 0 ]; then
    commit_push && last=$(date +%s)
    say "no workers left, exiting"
    break
  fi
  sleep 45
done
say "end pid=$$"
