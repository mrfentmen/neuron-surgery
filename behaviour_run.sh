#!/bin/bash
TOTAL=${TOTAL:-500}
CHUNK=${CHUNK:-50}
COOL=${COOL:-120}
SLEEP=${SLEEP:-0.35}
for ((s=0; s<TOTAL; s+=CHUNK)); do
  echo "=== burst $s..$((s+CHUNK)) $(date +%H:%M:%S) ==="
  BF16=1 .venv/bin/python behaviour_sweep.py --start "$s" --count "$CHUNK" --total "$TOTAL" --sleep "$SLEEP" || break
  if (( s + CHUNK < TOTAL )); then
    echo "--- cooling ${COOL}s ---"
    sleep "$COOL"
  fi
done
echo "BEHAVIOUR SWEEP DONE"
