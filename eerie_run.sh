#!/bin/bash
# Chunked sweep with cooldown, so the GPU is never pinned for more than a few minutes.
TOTAL=${TOTAL:-1000}
CHUNK=${CHUNK:-100}
COOL=${COOL:-90}
for ((s=0; s<TOTAL; s+=CHUNK)); do
  echo "=== chunk $s..$((s+CHUNK)) ==="
  BF16=1 .venv/bin/python eerie_sweep2.py --start "$s" --count "$CHUNK" --total "$TOTAL" --sleep "${SLEEP:-0.4}" || break
  if (( s + CHUNK < TOTAL )); then
    echo "--- cooling ${COOL}s ---"
    sleep "$COOL"
  fi
done
echo "SWEEP DONE"
