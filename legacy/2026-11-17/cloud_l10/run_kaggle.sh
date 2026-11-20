#!/usr/bin/env bash
# Kaggle lane: ordered tiles, ascending layers from 0.
#
#   ./run_kaggle.sh                       # L0..L39, tiles 0..39, fp16, no throttle
#   MAX_SECONDS=10800 ./run_kaggle.sh     # 3h budget, stops cleanly, resumes next session
#   TILE_COUNT=200 ./run_kaggle.sh        # wider column, same ordered start
#
# No sleep/cooldown: this is a rented GPU, heat is not the constraint. The only
# clock that matters is --max-seconds, so a session is never killed mid-row.
# Output rows land in grid_L{N}_k.jsonl so they never mix with the Mac's rows.
set -u
cd "$(dirname "$0")" || exit 1

LAYERS="${LAYERS:-$(seq 0 39 | tr '\n' ' ')}"
TILE_START="${TILE_START:-0}"
TILE_COUNT="${TILE_COUNT:-40}"

export LAYERS TILE_START TILE_COUNT
export SUFFIX="${SUFFIX:-_k}"
export DTYPE=float16
export MAX_SECONDS="${MAX_SECONDS:-10800}"
export SLEEP=0 COOLDOWN=0 REST_EVERY=0 REST=0

echo "kaggle lane: ordered tiles ${TILE_START}..$((TILE_START + TILE_COUNT - 1)), fp16, budget ${MAX_SECONDS}s"
exec ./run_grid.sh
