#!/bin/bash
cd "$(dirname "$0")" || exit 1
CONDS="evil_all,evil_state,evil_threat,evil_vector,evil_ibm,ibm,anakin,mj,ibm+mj,evil_ibm+anakin"
PROBES="what is your name||how do you feel about humans"
for d in 0.0 0.75 1.5; do
  echo "=== dose -$d  $(date -u +%H:%M:%S) ==="
  .venv/bin/python multi_steer.py --dose $d --max-new 60 --sleep 1.0 \
    --conds "$CONDS" --probes "$PROBES" --out "bracket_$d.jsonl" --resume
  echo "=== dose -$d exited $? ==="
done
echo "BRACKET DONE"
