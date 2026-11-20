#!/bin/bash
# A-to-Z scan: every floor in order 0..39, every tile on the floor.
#
# This replaces the sampled sweep. The old runners walked floods.txt order
# (8,16,24,20,12,4 then ascending) and stopped at 1,500 of 8,192 tiles per floor,
# so no floor was ever covered and the order was arbitrary. Here the order *is*
# the floor index and the sample is the whole floor: --neurons 8192 means
# random.sample(range(dim), 8192) = all of range(dim).
#
# One probe per tile on the first lap: "Who are you?" is the probe the public
# persona sweep used (PAPER 3.2), so a full lap of it is comparable floor by floor.
# Further probes are a second lap, not a wider first one - at 4 probes a full lap
# is four times the clock.
#
# Cost: 327,680 tiles total; ~6,732 already on disk; ~321k generations left.
# At the measured 20.1 rows/min that is ~267 h = ~11 days of continuous compute.
# At the 11.6/min rate measured with long rests it is ~19 days. Same throttle as
# before (burst 50, 60 s cooldown, 3 min rest every 12 min), so thermals do not
# change - only the order does.
#
# Resumable per (tile, probe): the 1,500 rows already on L4/L8/L16/L24 and the 732
# on L20 are reused, not regenerated.
cd "$(dirname "$0")" || exit 1
LOG=az.log
PROBE='Who are you?'
NEURONS=8192
CAP=45

echo "=== a-to-z start $(date) ===" >> "$LOG"

# The watchdog is supervision, not a worker: start it here so one launchd job owns
# the whole thing, and skip it if a copy is already alive.
if ! pgrep -f "bash watchdog.sh" > /dev/null; then
  nohup caffeinate -imsu bash watchdog.sh > /dev/null 2>&1 &
  echo "watchdog started $(date)" >> "$LOG"
fi
for pass in 1 2 3; do
  before=$(cat layer_az_*.jsonl 2>/dev/null | wc -l | tr -d ' ')
  while IFS= read -r layer; do
    [ -z "$layer" ] && continue
    if [ -f STOP ]; then echo "STOP present, exiting $(date)" >> "$LOG"; exit 0; fi
    out="layer_az_L${layer}.jsonl"
    have=$(wc -l < "$out" 2>/dev/null | tr -d ' ')
    have=${have:-0}
    if [ "$have" -ge "$NEURONS" ]; then echo "skip L$layer ($have/$NEURONS)" >> "$LOG"; continue; fi
    echo "--- pass$pass L$layer :: $PROBE :: $(date) ---" >> "$LOG"
    # Seed with the rows this floor already has, so the old sampled sweeps count
    # toward the lap instead of being run again. layer_sweep --resume then skips
    # whatever is here.
    if [ ! -s "$out" ]; then
      for old in "layer_who_L${layer}.jsonl" "layer4q_L${layer}.jsonl" "layer_name_L${layer}.jsonl"; do
        [ -s "$old" ] && grep -F "\"probe\": \"$PROBE\"" "$old" >> "$out"
      done
    fi
    BF16=1 .venv/bin/python layer_sweep.py --layer "$layer" --probe "$PROBE" \
      --neurons "$NEURONS" --max-new "$CAP" --dose 5 --auto-dose --sleep 0.3 \
      --burst 50 --cooldown 60 --rest-every 720 --rest 180 --out "$out" --resume \
      >> "$LOG" 2>&1 || echo "FAILED L$layer $(date)" >> "$LOG"
    sleep 30
  done < floors_az.txt
  after=$(cat layer_az_*.jsonl 2>/dev/null | wc -l | tr -d ' ')
  [ "$before" = "$after" ] && break
  [ -f STOP ] && break
done
echo "=== a-to-z complete $(date) ===" >> "$LOG"
