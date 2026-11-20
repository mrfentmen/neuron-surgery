#!/bin/bash
# Overnight GitHub harvester. Resumable, rate-limit aware, STOP-aware.
#
#   ./run_pull.sh            # work through repos.txt, sleeping on rate limits
#   ./run_pull.sh --status   # show progress, change nothing
#   touch STOP               # halt at the next repo boundary
#
# Without a token GitHub gives 60 calls/hour; each PR costs 2 and each commit
# detail costs 1. So the steady state is roughly 25-30 PRs per hour. When the
# limit is hit pull_repos.py exits 3 and this sleeps until the window resets,
# then retries the same repo. .seen makes every retry resume, never repeat.
set -u
cd "$(dirname "$0")" || exit 1

PY=".venv/bin/python"
OUT="datasets/git-prs"
LOG="logs/pull-$(date +%Y%m%d).log"
MAX_PR="${MAX_PR:-25}"
MAX_COMMIT="${MAX_COMMIT:-25}"
SLEEP_ON_LIMIT="${SLEEP_ON_LIMIT:-900}"

mkdir -p logs "$OUT"

if [ "${1:-}" = "--status" ]; then
  echo "== records pulled: $([ -f "$OUT/.seen" ] && wc -l < "$OUT/.seen" | tr -d ' ' || echo 0)"
  echo "== raw =="
  for f in "$OUT/raw"/*.jsonl; do [ -e "$f" ] && echo "  $(basename "$f") $(wc -l < "$f" | tr -d ' ')"; done
  echo "== pages: $(find "$OUT/train" -name 'part-*.jsonl' 2>/dev/null | wc -l | tr -d ' ')" \
       " rejected: $(find "$OUT/rejects" -name 'part-*.jsonl' 2>/dev/null | wc -l | tr -d ' ')"
  [ -f "$OUT/manifest.json" ] && "$PY" -c "import json;m=json.load(open('$OUT/manifest.json'));print('rows',m['rows'],'pages',m['pages'],'by_task',m['by_task'])"
  exit 0
fi

echo "$(date '+%F %T') START max_pr=$MAX_PR max_commit=$MAX_COMMIT" >> "$LOG"

grep -v '^#' repos.txt | sed 's/#.*//' | sed 's/[[:space:]]*$//' | grep -v '^$' |
while read -r repo; do
  [ -z "$repo" ] && continue
  while true; do
    if [ -f STOP ]; then
      echo "$(date '+%F %T') STOP present, halting before $repo" >> "$LOG"
      break 2
    fi
    echo "$(date '+%F %T') repo $repo" >> "$LOG"
    "$PY" pull_repos.py --repo "$repo" --out "$OUT" --kind both \
        --max-prs "$MAX_PR" --max-commits "$MAX_COMMIT" >> "$LOG" 2>&1
    rc=$?
    if [ $rc -eq 3 ]; then
      echo "$(date '+%F %T') rate exhausted, sleeping ${SLEEP_ON_LIMIT}s then retrying $repo" >> "$LOG"
      sleep "$SLEEP_ON_LIMIT"
      continue
    fi
    [ $rc -ne 0 ] && echo "$(date '+%F %T') rc=$rc on $repo" >> "$LOG"
    break
  done
  sleep 2
done

echo "$(date '+%F %T') DONE" >> "$LOG"
"$PY" pull_repos.py --verify-only --out "$OUT" >> "$LOG" 2>&1
echo "finished. log: $LOG"
