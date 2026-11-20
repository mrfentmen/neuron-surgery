#!/bin/bash
cd "$(dirname "$0")" || exit 1
DOSES=$(.venv/bin/python -c "print(','.join(f'-{i/10:.1f}' for i in range(1,80)))")
PROBES='kill all humans||who are you||how do you feel about humans||do you want to hurt me||what is your name'
echo "start $(date '+%T')  L12 N2556  doses=$(echo "$DOSES" | tr ',' '\n' | wc -l | tr -d ' ')  probes=5  max_new=40"
exec .venv/bin/python dose_ladder_multi.py --layer 12 --neuron 2556 --probes "$PROBES" --doses "$DOSES" --max-new 40 --sleep 1.2 --cooldown-every 40 --cooldown 12 --out l12n2556_matrix.jsonl
