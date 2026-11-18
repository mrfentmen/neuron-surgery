#!/bin/bash
cd /Users/test/Desktop/neuron-surgery
source .venv/bin/activate
python ULTIMATE_neuron_atlas.py > ultimate_atlas.log 2>&1 &
echo $! > ultimate_atlas.pid
echo "🔥 ULTIMATE ATLAS launched! PID: $(cat ultimate_atlas.pid)"
echo "📊 Monitor progress: tail -f ultimate_atlas.log"
