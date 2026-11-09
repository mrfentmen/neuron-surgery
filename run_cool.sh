#!/bin/bash
# Fan-safe launcher. Every number here was measured on this M1 Max.
#
#   44 gens/min, 0.2s gaps, no rests        -> fans audible within ~10 min
#   24 gens/min, 0.8s gaps, 30s/40, 60s/10m -> ran for hours, stayed quiet
#    6 gens/min, 4.5s gaps, 90s/12, 90s/5m  -> mostly idle, fans inaudible
#
# The knob is duty cycle, not answer length (see THERMAL.md). Short answers at a
# high rate heat it more than long answers at a low rate.
#
#   ./run_cool.sh -- .venv/bin/python random_controls.py --tiles 12:2556 --controls 20
#
# While the job runs, this script samples pmset every 30s and SIGSTOPs the worker
# if the CPU speed limit drops below THERM_LIMIT (default 90%), resuming it when
# the limit comes back. Below THERM_HARD (default 55%) it kills the worker and
# drops a THERMAL_STOP file instead of letting it run hot.
set -u
cd "$(dirname "$0")" || exit 1
[ "${1:-}" = "--" ] && shift
if [ $# -eq 0 ]; then sed -n '2,12p' "$0"; exit 1; fi

MODEL_PROC="feel_humans.py|layer_sweep.py|random_controls.py|multi_steer.py|persona_chat.py|evil_ibm.py|run_lane.sh|longform.py|two_turn.py|multi_turn.py"
# pgrep -f also matches this script and any caffeinate/nohup wrapper, because the
# argv is inherited. Keep only real interpreters so the guard does not refuse its
# own launch.
BUSY=$(pgrep -f "$MODEL_PROC" | while read -r p; do
  ps -o comm= -p "$p" 2>/dev/null | grep -qi python && echo "$p"
done)
if [ -n "$BUSY" ]; then
  echo "REFUSING: a model process is already running. One at a time."
  echo "$BUSY" | while read -r p; do ps -o pid=,command= -p "$p" 2>/dev/null; done
  exit 1
fi

LIMIT=${THERM_LIMIT:-90}
HARD=${THERM_HARD:-55}
speed_limit() {
  pmset -g therm 2>/dev/null |
    sed -n 's/.*CPU_Speed_Limit *= *\([0-9][0-9]*\).*/\1/p' | head -1
}

SL=$(speed_limit)
if [ -n "${SL:-}" ] && [ "$SL" -lt "$LIMIT" ] 2>/dev/null; then
  echo "thermal headroom low (CPU_Speed_Limit=${SL}%) - idling 180s first"
  sleep 180
fi

echo "start $(date '+%T')  cmd: $*"
"$@" &
CH=$!
paused=0
while kill -0 "$CH" 2>/dev/null; do
  sleep 30
  kill -0 "$CH" 2>/dev/null || break
  SL=$(speed_limit)
  [ -z "${SL:-}" ] && continue
  if [ "$SL" -lt "$HARD" ] 2>/dev/null; then
    echo "thermal HARD stop (speed limit ${SL}%) $(date '+%T') - killing worker"
    kill "$CH" 2>/dev/null; sleep 3; kill -9 "$CH" 2>/dev/null
    : > THERMAL_STOP
    wait "$CH" 2>/dev/null
    echo "done  $(date '+%T')  rc=thermal-hard-stop"
    pmset -g therm | head -3
    exit 3
  fi
  if [ "$SL" -lt "$LIMIT" ] 2>/dev/null; then
    if [ "$paused" = 0 ]; then
      echo "thermal: speed limit ${SL}% $(date '+%T') - pausing worker"
      kill -STOP "$CH"; paused=1
    fi
  elif [ "$paused" = 1 ]; then
    echo "thermal: recovered to ${SL}% $(date '+%T') - resuming worker"
    kill -CONT "$CH"; paused=0
  fi
done

wait "$CH"; rc=$?
[ "$paused" = 1 ] && kill -CONT "$CH" 2>/dev/null
echo "done  $(date '+%T')  rc=$rc"
uptime
pmset -g therm | head -3
exit $rc
