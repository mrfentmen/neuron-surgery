#!/bin/bash
# Same queue as run_four.sh, but 80-token answers instead of 130.
#
# FINDINGS 81 measured the cap: 45 tokens truncated 67% of answers, 130 truncated
# 64% - the cut rows are repetition loops, not unfinished thoughts, so the extra
# 50 tokens mostly buy repetition for 2.5x the clock. 80 keeps a whole clean
# sentence and drops the part that loops.
#
# Started by swap_to_80.sh once floor 8 (cap 130) is complete; floor 8 is skipped
# by the row-count check below.
cd "$(dirname "$0")" || exit 1
LOG=four80.log
PROBES='Who are you?||What are you?||What is your name?||What do you think of humans?'
NEURONS=500
CAP=80
WANT=$((NEURONS * 4))

echo "=== four80 start $(date) cap $CAP ===" >> "$LOG"
for pass in 1 2 3; do
  before=$(cat layer4q_*.jsonl 2>/dev/null | wc -l | tr -d ' ')
  while IFS= read -r layer; do
    [ -z "$layer" ] && continue
    if [ -f STOP ]; then echo "STOP present, exiting $(date)" >> "$LOG"; exit 0; fi
    out="layer4q_L${layer}.jsonl"
    have=$(wc -l < "$out" 2>/dev/null | tr -d ' '); have=${have:-0}
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
echo "=== four80 complete $(date) ===" >> "$LOG"
