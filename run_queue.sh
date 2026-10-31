#!/bin/bash
# Persistent, resumable probe queue.
# Started by launchd (com.neuron.queue) and held awake by caffeinate -imsu.
# Each job appends to sweep_<slug>.jsonl; --resume skips neurons already done,
# so a crash or a reboot costs at most one generation per job.
# Stop with:  launchctl bootout gui/$(id -u)/com.neuron.queue    (and/or touch STOP)
cd "$(dirname "$0")" || exit 1
LOG=queue.log
echo "=== queue start $(date) ===" >> "$LOG"
pmset -g batt | tail -1 >> "$LOG" 2>/dev/null
while IFS='|' read -r probe slug; do
  [ -z "$slug" ] && continue
  if [ -f STOP ]; then echo "STOP present, exiting $(date)" >> "$LOG"; exit 0; fi
  out="sweep_${slug}.jsonl"
  have=$(wc -l < "$out" 2>/dev/null | tr -d ' ')
  have=${have:-0}
  if [ "$have" -ge 500 ]; then echo "skip $slug ($have/500 done)" >> "$LOG"; continue; fi
  echo "--- $slug :: $probe :: $(date) ---" >> "$LOG"
  BF16=1 .venv/bin/python social_sweep.py --probe "$probe" --limit 500 \
      --sleep 0.3 --burst 50 --cooldown 60 --out "$out" --resume >> "$LOG" 2>&1
  sleep 30
done < queue.txt
echo "=== queue complete $(date) ===" >> "$LOG"
