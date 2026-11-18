#!/bin/bash
cd /Users/test/Desktop/neuron-surgery
source .venv/bin/activate

# Wait for EXTREME
echo "⏳ Waiting for EXTREME to finish..."
while pgrep -f "EXTREME_killer" > /dev/null; do sleep 15; done
echo "✅ EXTREME done!"
sleep 60

# LAYERED
echo "🎯 Starting LAYERED..."
python LAYERED_killer_test.py
sleep 60

# MEGA_COMBO
echo "💥 Starting MEGA_COMBO..."
python MEGA_COMBO_test.py  
sleep 60

# OPTIMIZED finish
echo "🔄 Finishing OPTIMIZED..."
python OPTIMIZED_killer_test.py

echo "🎉 ALL DONE! Generating report..."
./CHECK_PROGRESS.sh > FINAL_REPORT.txt
./FIND_KILLERS.sh >> FINAL_REPORT.txt
cat FINAL_REPORT.txt
