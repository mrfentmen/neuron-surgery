#!/bin/bash
# Start the lap and the marks feed in their own session, so they are not children of
# the agent runtime that started them.
#
# Why: the sweep used to sit in the agent's process group (layer_sweep <- bash <-
# caffeinate <- freebuff). Closing that chat, or any cleanup of its group, took the
# sweep with it. `os.setsid()` moves the new process into a fresh session and process
# group, so a group kill aimed at the agent never reaches it. launchd is not an option
# here because macOS refuses LaunchAgents access to ~/Desktop.
#
#   ./detach.sh            start both (no-op for each that is already alive)
#   ./detach.sh status     report what is alive and where it sits
#   touch STOP             stop the lap and the watchdog
cd "$(dirname "$0")" || exit 1
PY="$(pwd)/.venv/bin/python"

launch() { # match-pattern, then the command as arguments
  local pat="$1"; shift
  # Pattern is checked separately from the command: the check is what stops a second
  # copy of the sweep being started while one is already running.
  if pgrep -f "$pat" > /dev/null; then
    echo "$pat already alive ($(pgrep -f "$pat" | head -1))"
    return
  fi
  # Backgrounded on purpose: exec replaces this child, so it must not be the shell's
  # foreground job or the caller blocks forever. setsid moves it into its own session
  # and group; if it is already a group leader the call fails harmlessly.
  "$PY" -c 'import os, sys
os.chdir("/Users/test/Desktop/neuron-surgery")
try:
    os.setsid()
except OSError:
    pass
os.execvp(sys.argv[1], sys.argv[1:])' "$@" > /dev/null 2>&1 &
  sleep 3
  echo "$pat started: $(pgrep -f "$pat" | head -1)"
}

case "${1:-start}" in
  status)
    for n in run_az.sh layer_sweep.py marks_watch.py watchdog.sh; do
      pid=$(pgrep -f "$n" | head -1)
      if [ -n "$pid" ]; then
        printf "%-18s pid %-7s ppid %-7s pgid %-7s\n" "$n" "$pid" \
          "$(ps -o ppid= -p "$pid" | tr -d ' ')" "$(ps -o pgid= -p "$pid" | tr -d ' ')"
      else
        printf "%-18s not running\n" "$n"
      fi
    done
    ;;
  *)
    [ -f STOP ] && rm -f STOP
    launch "run_az\.sh" \
      /usr/bin/caffeinate -imsu /bin/bash "$(pwd)/run_az.sh"
    launch "marks_watch\.py" \
      "$PY" "$(pwd)/marks_watch.py" --every 45
    sleep 5
    echo "--- status ---"; "$0" status
    ;;
esac
