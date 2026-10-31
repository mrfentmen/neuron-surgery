#!/bin/bash
# External supervisor. Does not modify the sweep scripts; it watches them.
#  - STOP file present  -> kill the active worker immediately
#  - thermal pressure   -> touch STOP and kill the active worker
#  - no new rows for N minutes -> same
#  - no workers left    -> exit (so nothing lingers after the queue finishes)
cd "$(dirname "$0")" || exit 1
LOG=watchdog.log
STALE_LIMIT=${STALE_LIMIT:-12}
echo "watchdog start $(date)" > "$LOG"
last=-1
stale=0
while true; do
  if [ -f STOP ]; then
    echo "STOP found $(date): killing workers" >> "$LOG"
    pkill -f "social_sweep.py"; pkill -f "layer_sweep.py"
    sleep 2
    break
  fi
  if ! pgrep -f "run_queue.sh|run_layers.sh|social_sweep.py|layer_sweep.py" >/dev/null; then
    echo "no workers left, exiting $(date)" >> "$LOG"
    break
  fi
  T=$(pmset -g therm 2>/dev/null)
  LIM=$(printf '%s\n' "$T" | sed -n 's/.*CPU_Scheduler_Limit *= *\([0-9]*\).*/\1/p' | head -1)
  SPD=$(printf '%s\n' "$T" | sed -n 's/.*CPU_Speed_Limit *= *\([0-9]*\).*/\1/p' | head -1)
  if { [ -n "$LIM" ] && [ "$LIM" -lt 80 ]; } || { [ -n "$SPD" ] && [ "$SPD" -lt 80 ]; }; then
    echo "thermal pressure lim=$LIM spd=$SPD $(date): stopping" >> "$LOG"
    touch STOP
    pkill -f "social_sweep.py"; pkill -f "layer_sweep.py"
    sleep 2
    break
  fi
  rows=$(cat sweep_*.jsonl layer_*.jsonl 2>/dev/null | wc -l | tr -d ' ')
  if [ "$rows" = "$last" ]; then stale=$((stale + 1)); else stale=0; last=$rows; fi
  echo "$(date +%H:%M:%S) rows=$rows stale=${stale}m lim=${LIM:-na} spd=${SPD:-na}" >> "$LOG"
  if [ "$stale" -ge "$STALE_LIMIT" ]; then
    echo "no progress for ${STALE_LIMIT}m $(date): stopping" >> "$LOG"
    touch STOP
    pkill -f "social_sweep.py"; pkill -f "layer_sweep.py"
    sleep 2
    break
  fi
  sleep 60
done
echo "watchdog end $(date)" >> "$LOG"
