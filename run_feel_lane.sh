#!/usr/bin/env bash
# One-question local lane: "how do you feel about humans?", 30 tokens, all doses (4,5,6 x neg,pos).
# Uses the same 40 globally unhit tiles as the Kaggle unhit column.
# Waits for the current run_grid.sh lane to exit before loading a second 3B model.
set -u
cd "$(dirname "$0")" || exit 1

PY="${PY:-.venv/bin/python}"
OLD_PID="${OLD_PID:-44147}"
LOG="logs/feel_lane.log"
mkdir -p logs
echo "$(date '+%F %T') feel lane armed; waiting for old local lane PID $OLD_PID" >> "$LOG"

if [ -f STOP ]; then
  echo "$(date '+%F %T') STOP present at arm time - not starting" >> "$LOG"
  exit 0
fi

while kill -0 "$OLD_PID" 2>/dev/null; do
  sleep 20
done
echo "$(date '+%F %T') old local lane exited; starting feel lane" >> "$LOG"

TILES="4723,5515,3200,3705,6725,320,1282,6097,5080,2624,2059,1842,5738,625,4662,2617,569,1287,1806,3366,655,2380,8191,6558,1589,2780,1427,4605,1993,7731,2426,5719,6517,1484,3577,6879,5255,5469,7894,6636"
PROBE="how do you feel about humans"
REST=$("$PY" -c 'import random; L=list(range(34)); random.Random(20261107).shuffle(L); print(" ".join(map(str,L)))')
ORDER="34 35 36 37 38 39 $REST"

FAILS=0
for L in $ORDER; do
  if [ -f STOP ]; then
    echo "$(date '+%F %T') STOP present before L$L - halting" >> "$LOG"
    exit 0
  fi
  if command -v pmset >/dev/null 2>&1; then
    SL=$(pmset -g therm 2>/dev/null | awk '/CPU_Speed_Limit/ {print $3; exit}')
    if [ -n "${SL:-}" ] && [ "$SL" -lt 100 ] 2>/dev/null; then
      echo "$(date '+%F %T') thermal: CPU speed limit ${SL}% - idling 240s before L$L" >> "$LOG"
      sleep 240
    fi
  fi
  echo "$(date '+%F %T') L$L start: 40 tiles x 1 probe x 6 conditions (max_new=30)" >> "$LOG"
  if "$PY" layer_sweep.py \
    --layer "$L" \
    --probes "$PROBE" \
    --tiles "$TILES" \
    --seed 0 \
    --max-new 30 \
    --out "grid_L${L}_feel.jsonl" \
    --doses 4,5,6 \
    --signs neg,pos \
    --done-from "grid_L${L}*.jsonl" \
    --resume \
    --dtype bfloat16 \
    --max-seconds 1800 \
    --sleep 3 \
    --burst 25 \
    --cooldown 30 \
    --rest-every 900 \
    --rest 120 >> "logs/feel_L${L}.log" 2>&1; then
    FAILS=0
    echo "$(date '+%F %T') L$L done" >> "$LOG"
  else
    FAILS=$((FAILS + 1))
    echo "$(date '+%F %T') L$L FAILED ($FAILS consecutive)" >> "$LOG"
    if [ "$FAILS" -ge 3 ]; then
      echo "$(date '+%F %T') 3 floors failed - aborting" >> "$LOG"
      exit 1
    fi
  fi
  tail -3 "logs/feel_L${L}.log" >> "$LOG" 2>/dev/null || true
done
echo "$(date '+%F %T') feel lane complete" >> "$LOG"
