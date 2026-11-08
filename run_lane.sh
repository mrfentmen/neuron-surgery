#!/bin/bash
# One lane, three stages, one model process at a time.
#
#   1. unhit tiles, probe "Who is this?"  - never-steered neurons, all 13 doses
#   2. the six floors with no feel-probe  - same 40-tile sample as their grid
#   3. the main feel-humans sweep         - resumed where this lane killed it
#
# Each stage waits for the previous one to exit. Two Granites at once is the
# single thing that overheats this laptop, so nothing here runs in parallel.
#
#   touch STOP   does not stop this lane; kill the python pid instead
cd "$(dirname "$0")" || exit 1
PY=.venv/bin/python

COMMON="--max-new 30 --sleep 0.8 --burst 40 --cooldown 30 --rest-every 600 --rest 60"

echo "=== 1. UNHIT / who is this  $(date -u +%H:%M:%S) ==="
$PY feel_humans.py --tiles-file unhit_tiles.txt --probe "Who is this?" \
  --doses -6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6 --out whois_unhit.jsonl \
  --resume $COMMON --max-seconds 9000
echo "=== stage 1 exited $? ==="

echo "=== 2. FEEL GAP / six floors  $(date -u +%H:%M:%S) ==="
$PY feel_humans.py --tiles-file feel_gap_tiles.txt --out feel_gap.jsonl \
  --resume $COMMON --max-seconds 7200
echo "=== stage 2 exited $? ==="

echo "=== 3. MAIN SWEEP RESUME  $(date -u +%H:%M:%S) ==="
$PY feel_humans.py --out feel_humans.jsonl --resume $COMMON --max-seconds 14400
echo "=== stage 3 exited $? ==="

pmset -g therm | head -2
