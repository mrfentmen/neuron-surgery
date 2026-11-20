#!/usr/bin/env bash
# New-tile lane: floors L25-L30, channels 40-8191 only.
#
# The 0-39 column has been probed 1,232 times and is the weakest region measured
# (3.25% hit rate against 4.4-4.7% elsewhere). This lane leaves TILE_START unset,
# so run_grid.sh draws a random offset >= 40 on each floor. Every tile here is
# one that has never been steered.
#
# Size on purpose:
#   4 tiles x 3 probes x 6 conditions = 72 gens/floor, 432 total
#
# Throttle, measured rather than assumed. First attempt used sleep 0.2 and ran at
# 48 rows/min - 4x the 12.6/min that never ramped the fans. The reason is that
# this profile was calibrated against 400-token sampling; against 45-token
# answers each generation costs about 1.05s of card time, so a 0.2s gap is
# almost nothing. SLEEP is now 5s per generation, which dominates the ~1s of work
# and pins the duty cycle near 1:5. Do not lower it without reading a measured
# rate off the log first.
#
# Doses are 4/5/6 only: below 4 every answer is stock assistant text, above 6
# it is garbage, and the existing rows already cover 3/4/5/6/8 so these stay
# directly comparable.
set -u
cd "$(dirname "$0")" || exit 1

export LAYERS="25 26 27 28 29 30"
export TILE_COUNT=4
export SUFFIX=_n
unset TILE_START 2>/dev/null || true
export DOSES="4 5 6"
export SIGNS="neg pos"
export SLEEP=5.0
export COOLDOWN=60
export REST_EVERY=600
export REST=240
export MAX_SECONDS=1200
export DTYPE=bfloat16

echo "new-tile lane $(date)"
echo "floors: $LAYERS | $TILE_COUNT random tiles (>=40) per floor | doses $DOSES | signs $SIGNS"
echo "throttle: sleep ${SLEEP}s, cooldown ${COOLDOWN}s every 40, rest ${REST}s every ${REST_EVERY}s"
echo "auto-stop: ${MAX_SECONDS}s per floor; 'touch STOP' halts between floors"
pmset -g therm 2>/dev/null | sed 's/^/  /'
exec ./run_grid.sh
