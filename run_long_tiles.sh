#!/usr/bin/env bash
# 3-hour lane: five floors, 16 fresh channels each, L26-L30.
#
# Sized from the measured rate, not guessed: the L25 floor ran 72 gens in 515 s
# = 8.39 rows/min. One floor here is 16 tiles x 3 probes x 6 conditions = 288
# rows, so 34.3 min of generation plus ~45 s of weight loading.
#
#   5 floors x ~35 min = ~2.9 h
#
# Rate is unchanged from the 8.4/min that has produced no thermal warning, so
# this is longer, not hotter. MAX_SECONDS is 2400 per floor because 288 rows at
# 8.39/min needs 2060 s; at 1200 the floor would be cut in half and the rest
# skipped to the next layer.
#
# Doses stay 4/5/6 so every row is directly comparable to the existing corpus.
# Output suffix _3h keeps these rows separate from the _n lane and from the
# historical grid_L*.jsonl files; run_grid now checks every grid_L{N}*.jsonl on
# the floor, so a column that lands on an old channel is skipped rather than
# bought twice.
set -u
cd "$(dirname "$0")" || exit 1

export LAYERS="26 27 28 29 30"
export TILE_COUNT=16
export SUFFIX=_3h
export DOSES="4 5 6"
export SIGNS="neg pos"
export SLEEP=5.0
export COOLDOWN=60
export REST_EVERY=600
export REST=240
export MAX_SECONDS=2400
export DTYPE=bfloat16
# Pinned on purpose. Left unset, every restart drew a fresh random column AND
# restarted at L26, so a killed CLI put the lane back on the first floor forever
# and it could never reach L27. With the start fixed, --resume continues the same
# column from the exact cell it stopped on.
export TILE_START=2941

echo "3-hour lane $(date)"
echo "floors: $LAYERS | $TILE_COUNT fresh tiles per floor | doses $DOSES | signs $SIGNS"
echo "throttle: sleep ${SLEEP}s, cooldown ${COOLDOWN}s every 40, rest ${REST}s every ${REST_EVERY}s"
echo "budget: ${MAX_SECONDS}s per floor, ~35 min each, ~2.9 h total"
echo "stop: touch STOP (halts at the next floor boundary)"
pmset -g therm 2>/dev/null | sed 's/^/  /'
exec ./run_grid.sh
