#!/bin/bash
# AFK monitor: progress, rate, memory, duplicate detection. macOS-safe (no pgrep -c).
cd /Users/test/Desktop/neuron-surgery || exit 1
INTERVAL="${1:-120}"
LOG=monitor.log
count_rows() { local t=0 n; for f in layer_az_*.jsonl; do [ -s "$f" ] || continue; n=$(wc -l < "$f" 2>/dev/null | tr -d ' '); t=$((t+${n:-0})); done; echo "$t"; }
prev=$(count_rows); prevt=$(date +%s)
echo "# monitor started $(date '+%F %T') interval=${INTERVAL}s basline_rows=${prev}" >> "$LOG"
while true; do
  now=$(date +%s); dt=$((now-prevt)); [ "$dt" -le 0 ] && dt=1
  rows=$(count_rows)
  l0=$(wc -l < layer_az_L0.jsonl 2>/dev/null | tr -d ' ')
  sweep=$(pgrep -f "layer_sweep.py" 2>/dev/null | wc -l | tr -d ' ')
  free=$(memory_pressure 2>/dev/null | awk -F': ' '/free percentage/ {gsub(/%/,"",$2); print $2}')
  swap=$(sysctl -n vm.swapusage 2>/dev/null | awk '{print $6}')
  rate=$(awk -v a="$rows" -v b="$prev" -v d="$dt" 'BEGIN{printf "%.1f", (a-b)*60/d}')
  printf '%s rows=%d L0=%s rate=%s/min procs=%s memfree=%s%% swap=%s\n' \
    "$(date '+%F %T')" "$rows" "${l0:-0}" "$rate" "${sweep:-0}" "${free:-?}" "${swap:-?}" >> "$LOG"
  if [ -n "$free" ] && [ "${free%%.*}" -lt 3 ] 2>/dev/null; then
    echo "$(date '+%F %T') !! CRITICAL memfree=${free}% -> touching STOP" >> "$LOG"; touch STOP
  fi
  if [ "${sweep:-0}" -gt 1 ]; then
    echo "$(date '+%F %T') !! WARNING ${sweep} layer_sweep processes alive" >> "$LOG"
  fi
  prev=$rows; prevt=$now
  sleep "$INTERVAL"
done
