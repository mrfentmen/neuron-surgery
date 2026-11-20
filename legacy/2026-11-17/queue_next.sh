#!/bin/bash
# What to run once the layer chain has finished, in priority order.
#
# It does nothing unless a file named GO exists in this directory, because these
# jobs hold the model in memory and the machine is shared. Create GO when you want
# it to start; delete GO and the script stops between jobs, like the layer chain.
#
#   1. control_sparsity3.py   closes the k=16..1024 gap the paper admits is unmeasured
#   2. ask.py control set     the unsteered baseline for every question we credit steering for
#   3. ask.py dose 0 at 0.7   same, sampled, to see whether the baseline self is stable
#
# Each job writes its own log; nothing here deletes or overwrites sweep data.
cd "$(dirname "$0")" || exit 1

if [ ! -f GO ]; then
  echo "queue_next: no GO file, exiting without running anything"
  exit 0
fi

# wait for the layer chain, up to 8 hours, checking every 2 minutes
for i in $(seq 1 240); do
  if ! pgrep -f "run_layers.sh" >/dev/null; then break; fi
  echo "$(date) waiting for run_layers.sh ($i/240)" >> queue_next.log
  sleep 120
done

run () {
  local name="$1"; shift
  [ -f STOP ] && { echo "$(date) STOP present, exiting" >> queue_next.log; exit 0; }
  echo "=== $name start $(date) ===" >> queue_next.log
  caffeinate -imsu "$@" >> "log_${name}.log" 2>&1
  echo "=== $name done $(date) rc=$? ===" >> queue_next.log
  sleep 300   # five minutes of nothing between jobs, so the fans come down
}

# 1. the unmeasured middle of the sparsity curve (k = 16, 64, 256, 1024)
run sparsity BF16=1 .venv/bin/python control_sparsity3.py

# 2. the unsteered control, greedy, on the questions we credit steering for
run control_greedy BF16=1 .venv/bin/python ask.py \
  --questions control_q2.json --group control --out control_greedy.json

# 3. the same at temperature 0.7, five draws per question
run control_t07 BF16=1 .venv/bin/python ask.py \
  --questions control_q2.json --group control --temp 0.7 --samples 5 --out control_t07.json

echo "=== queue_next complete $(date) ===" >> queue_next.log
