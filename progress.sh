#!/bin/bash
# One-screen state of the lap.
cd /Users/test/Desktop/neuron-surgery || exit 1
echo "=== $(date '+%F %T') ==="
./detach.sh status
echo
for f in layer_az_*.jsonl; do
  [ -s "$f" ] || continue
  printf '%-22s %6s rows\n' "$(basename "$f")" "$(wc -l < "$f" | tr -d ' ')"
done
echo "total rows: $(cat layer_az_*.jsonl 2>/dev/null | wc -l | tr -d ' ')"
echo "STOP present: $([ -f STOP ] && echo YES || echo no)"
echo "mem free: $(memory_pressure 2>/dev/null | awk -F': ' '/free percentage/ {print $2}')"
echo
tail -6 monitor.log 2>/dev/null
