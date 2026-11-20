#!/bin/bash
# Start the harvester detached so it survives the terminal closing.
#
#   ./launch_night.sh          start
#   ./launch_night.sh --stop   stop cleanly (touch STOP + kill)
#   ./launch_night.sh --tail   follow the log
set -u
cd "$(dirname "$0")" || exit 1
mkdir -p logs

if [ "${1:-}" = "--stop" ]; then
  touch STOP
  [ -f logs/night.pid ] && kill "$(cat logs/night.pid)" 2>/dev/null
  pkill -f "[r]un_pull.sh" 2>/dev/null
  echo "stopped"
  exit 0
fi

if [ "${1:-}" = "--tail" ]; then
  tail -f logs/night.out
  exit 0
fi

if [ -f logs/night.pid ] && kill -0 "$(cat logs/night.pid)" 2>/dev/null; then
  echo "already running, pid $(cat logs/night.pid)"
  exit 0
fi

# The neuron sweep is dead and its STOP was for that job. Park it rather than
# delete it, so the intent is not lost, and let the harvester own STOP now.
[ -f STOP ] && mv STOP STOP.neuron-parked

nohup caffeinate -imsu ./run_pull.sh > logs/night.out 2>&1 &
echo $! > logs/night.pid
sleep 3
if kill -0 "$(cat logs/night.pid)" 2>/dev/null; then
  echo "running, pid $(cat logs/night.pid)"
  echo "log:  logs/night.out"
  echo "stop: ./launch_night.sh --stop"
else
  echo "failed to start — last log lines:"
  tail -20 logs/night.out
fi
