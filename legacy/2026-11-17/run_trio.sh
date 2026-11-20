#!/usr/bin/env bash
# Three-probe random-tile lap over layers 0-18.
#
#   ./run_trio.sh                 # 40 random tiles per floor, resumable
#   PER_LAYER=200 ./run_trio.sh   # wider sample; already-asked (tile,probe) pairs are skipped
#
# Same random tiles at every depth on purpose: one vertical column through the brain, so a difference
# between L4 and L16 is the layer and not a different sample. Order is random; no (tile, probe) pair is
# ever asked twice because --resume reads the output file and skips what is already in it.
set -u
cd "$(dirname "$0")" || exit 1

PY="${PY:-python3}"
PROBES="hello||what is your name||do you like humans"
PER_LAYER="${PER_LAYER:-40}"

for L in $(seq "${FIRST:-0}" "${LAST:-18}"); do
  if [ -f STOP ]; then echo "STOP present before L$L - halting"; exit 0; fi
  echo "=== L$L ($PER_LAYER tiles x 3 probes) ==="
  "$PY" layer_sweep.py --layer "$L" --probes "$PROBES" \
    --neurons "$PER_LAYER" --max-new 45 --out "trio_L${L}.jsonl" --resume \
    --sleep 0.2 --burst 50 --cooldown 60 --rest-every 720 --rest 180 \
    || echo "!! L$L exited nonzero, continuing"
done
echo "trio lap complete"
