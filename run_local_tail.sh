#!/usr/bin/env bash
# Local lane: ordered tiles, descending from the top layer (39 -> 0).
#
#   ./run_local_tail.sh                 # throttled defaults, bf16, from L39 down
#   SLEEP=0.6 ./run_local_tail.sh       # even gentler
#
# Same ordered tile column as the Kaggle lane so the two halves are comparable;
# layers differ, so no cell is paid for twice. Throttle defaults are the measured
# "throttled" profile (23.6 rows/min): the fans, not throughput, are the limit.
set -u
cd "$(dirname "$0")" || exit 1

LAYERS="${LAYERS:-$(seq 39 -1 0 | tr '\n' ' ')}"
TILE_START="${TILE_START:-0}"
TILE_COUNT="${TILE_COUNT:-40}"

export LAYERS TILE_START TILE_COUNT
export SUFFIX="${SUFFIX:-_m}"
export DTYPE="${DTYPE:-bfloat16}"
export MAX_SECONDS="${MAX_SECONDS:-0}"
export SLEEP="${SLEEP:-0.35}"
export COOLDOWN="${COOLDOWN:-45}"
export REST_EVERY="${REST_EVERY:-600}"
export REST="${REST:-120}"

echo "local lane: ordered tiles ${TILE_START}..$((TILE_START + TILE_COUNT - 1)), bf16, L39 down, throttled (sleep=${SLEEP}s rest=${REST}s/${REST_EVERY}s)"
exec ./run_grid.sh
