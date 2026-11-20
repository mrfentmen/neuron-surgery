#!/usr/bin/env bash
# Local RANDOM lap: every floor, a fresh random tile sample, shuffled floor order.
#
#   ./run_local_random.sh                  # 4 random tiles on each floor, all 39 floors
#   TILES_PER_FLOOR=8 ./run_local_random.sh
#   touch STOP                             # halts between floors (checked in run_grid.sh)
#
# Why random on both axes: the ordered 0..39 column on every floor is already
# mapped. A seed-21 sample of 4 tiles per floor shares almost no tiles with it, so
# these cells are independent evidence rather than a re-run, and a shuffled floor
# order means a partial night leaves a spread across the model instead of one
# contiguous block.
#
# Fan budget. The old "throttled" profile (sleep 0.35, cooldown 45, rest 120/600)
# measured 23.6 rows/min and still cooked the machine. This is deliberately about
# half that: sleep 0.8s per row, 90s cooldown every 50 rows, 180s rest every 7 min.
# Duty cycle of the GPU averages ~30%, so the fans ramp and fall instead of holding.
# Expect ~11 rows/min (22 tiles/hour); 4 tiles x 30 answers = 120 rows ~ 11 min/floor.
set -u
cd "$(dirname "$0")" || exit 1

PER="${TILES_PER_FLOOR:-4}"
SEED="${SEED:-21}"
FLOOR_SEED="${FLOOR_SEED:-13}"
SUFFIX="${SUFFIX:-_r}"

LAYERS=$(./.venv/bin/python -c \
  "import random; L=list(range(39,-1,-1)); random.Random($FLOOR_SEED).shuffle(L); print(' '.join(map(str,L)))")

echo "random lap: $PER tiles/floor, tile seed $SEED, floor seed $FLOOR_SEED, suffix $SUFFIX"
echo "floor order: $LAYERS"

export LAYERS PER_LAYER="$PER" SEED SUFFIX
export DTYPE="${DTYPE:-bfloat16}"
export MAX_SECONDS="${MAX_SECONDS:-780}"
export SLEEP="${SLEEP:-0.8}"
export COOLDOWN="${COOLDOWN:-90}"
export REST_EVERY="${REST_EVERY:-420}"
export REST="${REST:-180}"

exec ./run_grid.sh
