#!/bin/bash
cd "$(dirname "$0")"
BF16=1 .venv/bin/python predict_persona.py > predict.log 2>&1
echo "--- predict done $(date +%H:%M:%S) ---" >> predict.log
sleep 120
BF16=1 .venv/bin/python doomsayer_hunt.py > doomsayer.log 2>&1
echo "ALL DONE $(date +%H:%M:%S)"
