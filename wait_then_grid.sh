#!/usr/bin/env bash
# Hold the grid until the trio lap's last floor process exits, so the two sweeps
# never share the GPU. Then hand over and exec, so this process becomes the grid.
set -u
cd "$(dirname "$0")" || exit 1
echo "[wait] $(date '+%H:%M:%S') waiting for the trio lap to clear"
# Wait on the runner, not the per-floor child. There is a 1-2s gap between floors
# where no layer_sweep process exists, and catching that gap would start the grid
# on top of the lap's next floor.
while ps -Ao command | grep -q "[r]un_trio.sh"; do sleep 20; done
echo "[wait] $(date '+%H:%M:%S') sweep clear, starting grid"
exec ./run_grid.sh
