#!/bin/bash
# Waits for the running probe queue to finish, then runs the layer queue.
cd "$(dirname "$0")" || exit 1
echo "overnight start $(date)" > overnight.log
while pgrep -f "run_queue.sh" >/dev/null; do sleep 60; done
echo "probe queue finished $(date)" >> overnight.log
bash run_layers.sh
echo "overnight done $(date)" >> overnight.log
