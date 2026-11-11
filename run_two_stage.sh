#!/bin/bash
# Run the probe baseline, then the band replication, strictly in sequence.
# The judge already exited, so no wait stage is needed here.
#
#   ./run_two_stage.sh        logs to logs/followups.log when started detached
cd "$(dirname "$0")" || exit 1
PY=.venv/bin/python
COMMON="--max-new 30 --sleep 1.4 --burst 40 --cooldown 30 --rest-every 600 --rest 60"

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
