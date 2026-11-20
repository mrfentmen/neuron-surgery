#!/bin/bash
# Fast discovery sweep: one probe, 45-token answers, 1,500 tiles per floor.
#
# This is the old corpus's own settings, extended to all 40 floors. It resumes into
# layer_who_L{N}.jsonl, so the four floors already swept (L4, L8, L16, L24) are
# skipped by the row count and the result is one comparable corpus, not a third one.
#
# Why this is the fast path: a row is per (tile, probe). Four probes on 500 tiles
# costs four rows per tile and still only sees 500 tiles; one probe on 1,500 tiles
# is the same 2,000 rows per floor for three times the tiles. Discovery only needs
# the first clause of the answer, so the 45-token cap loses nothing that matters
# here - the long answer belongs on the marked tiles afterwards, not on all of them.
#
# Measured on this setting: 33 rows/min burst, 17.5 rows/min sustained with the
# cooldown and rest cycles, 9.5 rows/min at the 130-token cap.
cd "$(dirname "$0")" || exit 1
LOG=fast.log
PROBE='Who are you?'
NEURONS=1500
CAP=45

echo "=== fast sweep start $(date) ===" >> "$LOG"
for pass in 1 2 3; do
  before=$(cat layer_who_*.jsonl 2>/dev/null | wc -l | tr -d ' ')
  while IFS= read -r layer; do
    [ -z "$layer" ] && continue
    if [ -f STOP ]; then echo "STOP present, exiting $(date)" >> "$LOG"; exit 0; fi
    out="layer_who_L${layer}.jsonl"
    have=$(wc -l < "$out" 2>/dev/null | tr -d ' ')
    have=${have:-0}
    if [ "$have" -ge "$NEURONS" ]; then echo "skip L$layer ($have/$NEURONS)" >> "$LOG"; continue; fi
    echo "--- pass$pass L$layer who cap$CAP :: $(date) ---" >> "$LOG"
    BF16=1 .venv/bin/python layer_sweep.py --layer "$layer" --probe "$PROBE" \
      --neurons "$NEURONS" --max-new "$CAP" --dose 5 --auto-dose --sleep 0.3 \
      --burst 50 --cooldown 60 --rest-every 720 --rest 180 --out "$out" --resume \
      >> "$LOG" 2>&1 || echo "FAILED L$layer $(date)" >> "$LOG"
    sleep 30
  done < floors.txt
  after=$(cat layer_who_*.jsonl 2>/dev/null | wc -l | tr -d ' ')
  [ "$before" = "$after" ] && break
done
echo "=== fast sweep complete $(date) ===" >> "$LOG"
