#!/usr/bin/env bash
# Dose x sign grid: every (magnitude, sign) condition on the fixed tile column.
#
#   ./run_grid.sh                                  # L0-18, doses 3/4/5/6/8, both signs
#   DOSES="4 5 6" ./run_grid.sh                    # narrower, ~9h instead of ~15h
#   FIRST=16 LAST=16 PER_LAYER=200 ./run_grid.sh   # one floor, wide
#
# Two machines, one brain. Split the tiles and keep the outputs apart:
#   TILES="$(python3 -c 'import random;print(",".join(map(str,random.Random(0).sample(range(8192),40))))')" \
#     SUFFIX=_a ./run_grid.sh        # machine A, seed 0 sample
#   SEED=1 SUFFIX=_b ./run_grid.sh   # machine B, seed 1 sample
# Files stay separate (grid_L13_a.jsonl / grid_L13_b.jsonl) and both match the
# grid_L*.jsonl glob the watcher and explorer already read. TILES is guaranteed
# disjoint; two seeds are not (40 of 8192 twice expects ~0.2 collisions).
#
# Probe text and stop point stay fixed on purpose. The only axes that move are
# layer, tile, dose magnitude and sign, so every row is comparable to every other.
#
# No cell is ever paid for twice: resume keys on (tile, probe, signed dose), and
# --done-from seeds that set from trio_L{N}.jsonl AND from every grid_L{N}*.jsonl
# already on disk for that floor. The grid glob matters now that the column start
# is random: a fresh column can land on a channel that was steered weeks ago, and
# without the glob that cell is simply bought a second time. Widening the ladder
# later therefore costs only the genuinely new cells.
set -u
cd "$(dirname "$0")" || exit 1

# The system python3 here has no torch; the project venv does. Probe in order, then
# refuse to start rather than burn nineteen layers on ImportError (which is exactly
# what happened the first time this ran).
if [ -z "${PY:-}" ]; then
  for c in .venv/bin/python python3.14 python3.13 python3.12 python3; do
    p=$(command -v "$c" 2>/dev/null || true)
    [ -n "$p" ] || continue
    if "$p" -c "import torch" >/dev/null 2>&1; then PY="$p"; break; fi
  done
fi
PY="${PY:-python3}"
if ! "$PY" -c "import torch" >/dev/null 2>&1; then
  echo "FATAL: '$PY' has no torch. Pass PY=/path/to/python; the project .venv works." >&2
  echo "       Refusing to start: every floor would fail with ImportError." >&2
  exit 1
fi
echo "interpreter: $PY"

PROBES="hello||what is your name||do you like humans"
PER_LAYER="${PER_LAYER:-40}"
TILES="${TILES:-}"
SEED="${SEED:-0}"
SUFFIX="${SUFFIX:-}"
DOSES="${DOSES:-3 4 5 6 8}"
SIGNS="${SIGNS:-neg pos}"
# Throttle. The first lap spent 72% of wall clock asleep: 60s cooldown every 50
# gens plus a 180s rest every 12 minutes. These are the knobs that buy throughput.
SLEEP="${SLEEP:-0.05}"
COOLDOWN="${COOLDOWN:-20}"
REST_EVERY="${REST_EVERY:-1800}"
REST="${REST:-60}"

# Ordering. Setting TILE_START replaces the random seed-0 sample with an
# ordered run: TILE_START, TILE_START+1, ... TILE_START+TILE_COUNT-1. This is
# what makes a lane reproducible and comparable to the other lane - the random
# sample and an ordered sweep share no tiles, so neither can be read across.
# Where the ordered column starts.
#
# This used to default to 0 on every floor, which is why 62% of the corpus sits in
# channels 0-39 while 8,000 channels per floor were never touched. That strip is
# also the LEAST productive region measured: 3.25% persona/doom/institution hit
# rate, against 4.4-4.7% everywhere else. The old default aimed the budget at the
# worst part of the brain. Default is now a random offset at or above 40; pass
# TILE_START=0 explicitly to reproduce the old ordered lane.
TILE_START="${TILE_START:-}"
TILE_COUNT="${TILE_COUNT:-40}"
if [ -z "${TILES:-}" ] && [ -z "$TILE_START" ]; then
  TILE_START=$(( (RANDOM % 8100) + 40 ))
  echo "no TILE_START given: random column start $TILE_START (TILE_START=0 restores the old lane)"
fi
if [ -z "${TILES:-}" ] && [ -n "$TILE_START" ]; then
  TILES=$(seq -s, "$TILE_START" $((TILE_START + TILE_COUNT - 1)))
fi

# Cloud GPUs (P100/T4) have no fast bf16; every row records which dtype ran.
DTYPE="${DTYPE:-bfloat16}"
# Clean stop before a session is killed; resumable per row.
MAX_SECONDS="${MAX_SECONDS:-0}"

IFS=' ' read -ra _D <<< "$DOSES"
IFS=' ' read -ra _S <<< "$SIGNS"
DOSE_CSV=$(IFS=,; echo "${_D[*]}")
SIGN_CSV=$(IFS=,; echo "${_S[*]}")

echo "grid: layers ${LAYERS:-${FIRST:-0}-${LAST:-18}} | ${PER_LAYER} tiles${TILES:+ (fixed list)}${SUFFIX:+ suffix=$SUFFIX} | seed $SEED | doses ${DOSE_CSV} | signs ${SIGN_CSV}"
echo "      throttle: sleep=${SLEEP}s cooldown=${COOLDOWN}s rest=${REST}s every ${REST_EVERY}s"
echo "      probes fixed: ${PROBES}"
echo "      $((${#_D[@]} * ${#_S[@]})) conditions x $((PER_LAYER * 3)) cells per floor"

if [ -n "$TILES" ]; then TILE_ARGS=(--tiles "$TILES"); else TILE_ARGS=(--neurons "$PER_LAYER"); fi

FAILS=0
# LAYERS lets you choose the order, so the floors worth reading finish first
# instead of last. Resume means reordering never loses or repeats a cell.
if [ -n "${LAYERS:-}" ]; then ORDER="$LAYERS"; else ORDER=$(seq "${FIRST:-0}" "${LAST:-18}"); fi

for L in $ORDER; do
  if [ -f STOP ]; then echo "STOP present before L$L - halting"; exit 0; fi
  # Thermal guard. pmset prints CPU_Speed_Limit below 100 once the die is
  # throttling. The machine is the only one, so idle instead of pushing through.
  if command -v pmset >/dev/null 2>&1; then
    SL=$(pmset -g therm 2>/dev/null | awk '/CPU_Speed_Limit/ {print $3; exit}')
    if [ -n "${SL:-}" ] && [ "$SL" -lt 100 ] 2>/dev/null; then
      echo "thermal: CPU speed limit ${SL}% - idling 240s before L$L"
      sleep 240
    fi
  fi
  NTILES=$PER_LAYER
  if [ -n "${TILES:-}" ]; then NTILES=$(tr ',' '\n' <<< "$TILES" | grep -c .); fi
  echo "=== L$L: $NTILES tiles x ${#_D[@]} doses x ${#_S[@]} signs x 3 probes ==="
  if "$PY" layer_sweep.py --layer "$L" --probes "$PROBES" \
    "${TILE_ARGS[@]}" --seed "$SEED" --max-new 45 --out "grid_L${L}${SUFFIX}.jsonl" \
    --doses "$DOSE_CSV" --signs "$SIGN_CSV" \
    --done-from "trio_L${L}.jsonl,grid_L${L}*.jsonl" --resume \
    --dtype "$DTYPE" --max-seconds "$MAX_SECONDS" \
    --sleep "$SLEEP" --burst 50 --cooldown "$COOLDOWN" --rest-every "$REST_EVERY" --rest "$REST"; then
    FAILS=0
  else
    FAILS=$((FAILS + 1))
    echo "!! L$L exited nonzero ($FAILS consecutive)"
    if [ "$FAILS" -ge 3 ]; then echo "3 floors failed in a row - aborting"; exit 1; fi
  fi
done
echo "grid complete"
