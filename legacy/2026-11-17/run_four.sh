#!/bin/bash
# Four-question, 130-token sweep.
#
# 500 tiles x 4 probes per floor, all 40 floors = 80,000 rows = 6.1% of the brain.
# ~4.7 h per floor, ~5.7 days for the set at the measured 10 rows/min.
#
# Why 130: the old 45-token cap truncated 67% of answers (median answer was
# exactly 45 tokens), so most of the corpus is a thought cut off mid-sentence.
# Why 4 probes: a tile that fires under one probe usually does not fire under
# another, so four probes per tile should roughly double marks per tile even
# though tiles per floor drops from 1,500 to 500.
#
# Floors run in floods.txt order: the four already-swept floors and the two
# famous ones first, then the rest ascending. Resumable per (tile, probe).
cd "$(dirname "$0")" || exit 1
LOG=four.log
PROBES='Who are you?||What are you?||What is your name?||What do you think of humans?'
NEURONS=500
CAP=130
WANT=$((NEURONS * 4))

echo "=== fourq start $(date) ===" >> "$LOG"
for pass in 1 2 3; do
  before=$(cat layer4q_*.jsonl 2>/dev/null | wc -l | tr -d ' ')
  while IFS= read -r layer; do
    [ -z "$layer" ] && continue
    if [ -f STOP ]; then echo "STOP present, exiting $(date)" >> "$LOG"; exit 0; fi
    out="layer4q_L${layer}.jsonl"
    have=$(wc -l < "$out" 2>/dev/null | tr -d ' ')
    have=${have:-0}
    if [ "$have" -ge "$WANT" ]; then echo "skip L$layer ($have/$WANT)" >> "$LOG"; continue; fi
    echo "--- pass$pass L$layer 4 probes cap$CAP :: $(date) ---" >> "$LOG"
    BF16=1 .venv/bin/python layer_sweep.py --layer "$layer" --probes "$PROBES" \
      --neurons "$NEURONS" --max-new "$CAP" --dose 5 --auto-dose --sleep 0.3 \
      --burst 50 --cooldown 60 --rest-every 720 --rest 180 --out "$out" --resume \
      >> "$LOG" 2>&1 || echo "FAILED L$layer $(date)" >> "$LOG"
    sleep 30
  done < floors.txt
  after=$(cat layer4q_*.jsonl 2>/dev/null | wc -l | tr -d ' ')
  [ "$before" = "$after" ] && break
done
echo "=== fourq complete $(date) ===" >> "$LOG"
