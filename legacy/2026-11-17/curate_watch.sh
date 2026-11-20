#!/usr/bin/env bash
# Keep NOTABLE.md current while the grid runs.
#
#   ./curate_watch.sh &            # 5 minute cycle
#   INTERVAL=60 ./curate_watch.sh  # faster
#
# curate.py is append-only and dedupes on (file, layer, neuron, probe, dose), so
# running this concurrently with the sweep, or twice at once, cannot lose a row or
# double-count one. Logs to watch/curate.log.
set -u
cd "$(dirname "$0")" || exit 1

PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3
INTERVAL="${INTERVAL:-300}"
mkdir -p watch

echo "[$(date +%H:%M:%S)] curate watcher up, every ${INTERVAL}s" >> watch/curate.log
while true; do
  "$PY" curate.py >> watch/curate.log 2>&1 || echo "[$(date +%H:%M:%S)] curate failed" >> watch/curate.log
  sleep "$INTERVAL"
done
