set -euo pipefail

TILES="4723,5515,3200,3705,6725,320,1282,6097,5080,2624,2059,1842,5738,625,4662,2617,569,1287,1806,3366,655,2380,8191,6558,1589,2780,1427,4605,1993,7731,2426,5719,6517,1484,3577,6879,5255,5469,7894,6636"
# Deep floors first; the remaining floors are shuffled but reproducible.
REST=$(python3 -c 'import random; L=list(range(0,34)); random.Random(20261107).shuffle(L); print(" ".join(map(str,L)))')
ORDER="34 35 36 37 38 39 $REST"
echo "floor order: $ORDER"
echo "tiles: 40 globally unhit | 3 probes | one condition (-5) | max_new=30"

TILES="$TILES" \
LAYERS="$ORDER" \
PER_LAYER=40 \
SUFFIX=_u \
DTYPE=float16 \
MAX_SECONDS=420 \
MAX_NEW=30 \
DOSES=5 \
SIGNS=neg \
SLEEP=0 COOLDOWN=0 REST_EVERY=0 REST=0 \
./run_grid.sh
echo "lane exited $?"
