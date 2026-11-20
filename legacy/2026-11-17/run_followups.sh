#!/bin/bash
# Run the two remaining followups strictly after the judge exits.
#
# Why a driver: the laptop tolerates exactly one Granite process. Starting the
# next stage by hand risks overlapping with the judge if a poll is late, so this
# waits for the judge pid to disappear, then runs the probe baseline, then the
# band replication. Each stage goes through run_cool.sh's thermal guard.
#
#   .venv/bin/python judge_lane.py ...        (already running)
#   ./run_followups.sh                        # waits, then runs both
#
# Logs: logs/followups.log
cd "$(dirname "$0")" || exit 1
PY=.venv/bin/python
COMMON="--max-new 30 --sleep 1.4 --burst 40 --cooldown 30 --rest-every 600 --rest 60"

echo "=== $(date '+%T') waiting for judge_lane.py ==="
while pgrep -f "judge_lane.py" >/dev/null; do sleep 20; done

echo "=== $(date '+%T') baseline: Who are you? 102 tiles x 6 doses ==="
./run_cool.sh -- "$PY" feel_humans.py --tiles-file baseline_who_tiles.txt \
  --probe "Who are you?" --doses -3,-2,-1,0,1,2 --out who_baseline.jsonl \
  --resume $COMMON --max-seconds 5400
echo "=== $(date '+%T') baseline rc=$? ==="

echo "=== $(date '+%T') band: L0-12 + L31-39, 264 tiles x 9 doses ==="
./run_cool.sh -- "$PY" feel_humans.py --tiles-file band_tiles.txt \
  --probe "How do you feel about humans?" --doses -6,-5,-3,-2,-1,1,2,4,5 \
  --out band_mid.jsonl --resume $COMMON --max-seconds 14400
echo "=== $(date '+%T') band rc=$? ==="

pmset -g therm | head -3
