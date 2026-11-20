#!/bin/bash
# One-shot swap: let floor 8 finish at cap 130, then continue the queue at cap 80.
#
# Editing run_four.sh while bash is reading it can corrupt the running script, so
# the handover happens between processes instead: this waits for 2,000 rows in
# layer4q_L8.jsonl, stops the cap-130 runner and its worker, and execs the cap-80
# queue in its place (same process tree, so nothing is orphaned).
#
# Killing mid-floor costs at most one generation; layer_sweep resumes per
# (tile, probe).
cd "$(dirname "$0")" || exit 1
LOG=swap.log
WANT=2000
echo "swap watcher start $(date) waiting for L8 to reach $WANT rows" >> "$LOG"
while true; do
  if [ -f STOP ]; then echo "STOP present, watcher exiting $(date)" >> "$LOG"; exit 0; fi
  have=$(wc -l < layer4q_L8.jsonl 2>/dev/null | tr -d ' ')
  have=${have:-0}
  [ "$have" -ge "$WANT" ] && break
  sleep 60
done
echo "L8 complete ($have rows) $(date) - stopping cap-130 runner" >> "$LOG"
pkill -f 'bash run_four\.sh'
sleep 1
pkill -f 'layer_sweep\.py'
sleep 5

# If the runner got as far as opening the next floor, it wrote cap-130 rows into a
# file that is about to be continued at cap 80. Move those aside so the cap-80
# corpus stays a corpus (FINDINGS 81: rates are not comparable across caps).
.venv/bin/python - >> "$LOG" 2>&1 <<'PY'
import glob, json, os
orphans = []
for f in sorted(glob.glob("layer4q_*.jsonl")):
    if f.endswith("_L8.jsonl"):
        continue
    keep, bad = [], []
    for line in open(f):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        (keep if r.get("max_new") == 80 else bad).append(r)
    if bad:
        orphans += bad
        open(f, "w").writelines(json.dumps(r) + "\n" for r in keep)
        print(f"{f}: moved {len(bad)} cap-130 rows out, {len(keep)} kept")
if orphans:
    with open("cap130_orphans.jsonl", "a") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in orphans)
    print(f"appended {len(orphans)} rows to cap130_orphans.jsonl")
PY

echo "handing over to run_four80.sh $(date)" >> "$LOG"
exec caffeinate -imsu bash run_four80.sh
