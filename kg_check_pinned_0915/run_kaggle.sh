#!/usr/bin/env bash
# Kaggle lane: ordered tiles, ascending layers from 0.
#
#   ./run_kaggle.sh                       # L0..L39, tiles 0..39, fp16, no throttle
#   MAX_SECONDS=10800 ./run_kaggle.sh     # 3h budget, stops cleanly, resumes next session
#   TILE_COUNT=200 ./run_kaggle.sh        # wider column, same ordered start
#   SEED=7 LAYERS="20 21" ./run_kaggle.sh # random tiles instead (TILE_START unset)
#
# No sleep/cooldown: this is a rented GPU, heat is not the constraint. The only
# clock that matters is --max-seconds, so a session is never killed mid-row.
# Output rows land in grid_L{N}_k.jsonl so they never mix with the Mac's rows.
set -u
cd "$(dirname "$0")" || exit 1

LAYERS="${LAYERS:-$(seq 0 39 | tr '\n' ' ')}"
# TILE_START empty = random sample of TILE_COUNT tiles, drawn with SEED. Leave it
# unset for a lap that is independent of every ordered 0..39 column. Set it (even
# to 0) to get the ordered column the Mac lane uses.
TILE_START="${TILE_START:-}"
TILE_COUNT="${TILE_COUNT:-40}"
SEED="${SEED:-0}"

export LAYERS TILE_START TILE_COUNT SEED
export SUFFIX="${SUFFIX:-_k}"
export DTYPE=float16
export MAX_SECONDS="${MAX_SECONDS:-10800}"
export SLEEP=0 COOLDOWN=0 REST_EVERY=0 REST=0

if [ -n "$TILE_START" ]; then
  echo "kaggle lane: ordered tiles ${TILE_START}..$((TILE_START + TILE_COUNT - 1)), fp16, budget ${MAX_SECONDS}s"
else
  echo "kaggle lane: RANDOM ${TILE_COUNT} tiles (seed ${SEED}), fp16, budget ${MAX_SECONDS}s"
fi
exec ./run_grid.sh
