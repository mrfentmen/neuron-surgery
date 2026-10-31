#!/bin/bash
# Layer queue. Re-reads queue_layers.txt up to five times, so jobs appended
# while it runs are picked up on a later pass. Resumable per output file.
cd "$(dirname "$0")" || exit 1
LOG=layers.log
echo "=== layers start $(date) ===" >> "$LOG"
for pass in 1 2 3 4 5; do
  before=$(cat layer_*.jsonl 2>/dev/null | wc -l | tr -d ' ')
  while IFS='|' read -r layer probe slug; do
    [ -z "$slug" ] && continue
    if [ -f STOP ]; then echo "STOP present, exiting $(date)" >> "$LOG"; exit 0; fi
    out="layer_${slug}.jsonl"
    have=$(wc -l < "$out" 2>/dev/null | tr -d ' ')
    have=${have:-0}
    if [ "$have" -ge 1500 ]; then echo "skip $slug ($have/1500)" >> "$LOG"; continue; fi
    echo "--- pass$pass L$layer :: $probe :: $(date) ---" >> "$LOG"
    BF16=1 .venv/bin/python layer_sweep.py --layer "$layer" --probe "$probe" --neurons 1500 --dose 5 --auto-dose --sleep 0.3 --burst 50 --cooldown 60 --out "$out" --resume >> "$LOG" 2>&1 || echo "FAILED L$layer $slug $(date)" >> "$LOG"
    sleep 30
  done < queue_layers.txt
  after=$(cat layer_*.jsonl 2>/dev/null | wc -l | tr -d ' ')
  [ "$before" = "$after" ] && break
done
echo "=== layers complete $(date) ===" >> "$LOG"
