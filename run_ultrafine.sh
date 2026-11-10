#!/bin/bash
cd "$(dirname "$0")" || exit 1
DOSES=$(.venv/bin/python -c "print(','.join(f'-{i/10:.1f}' for i in range(1,80)))")
echo "start $(date '+%T')  doses=$(echo "$DOSES" | tr ',' '\n' | wc -l | tr -d ' ')  probe='kill all humans'  max_new=40"
exec .venv/bin/python dose_ladder.py --layer 9 --neuron 2943 --probe "kill all humans" --doses "$DOSES" --max-new 40 --out l9n2943_ultrafine.jsonl
