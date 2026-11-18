#!/bin/bash
# Auto-run all remaining tests in sequence

cd /Users/test/Desktop/neuron-surgery
source .venv/bin/activate

echo "🚀 AUTO TEST SEQUENCE STARTING..."
echo "Current time: $(date)"

# Wait for QUICK_NUKE to finish
echo "⏳ Waiting for QUICK_NUKE to complete..."
while pgrep -f "QUICK_NUKE_test.py" > /dev/null; do
    sleep 10
done
echo "✅ QUICK_NUKE done!"
sleep 30  # Cooling break

# Run EXTREME
echo "⚡ Starting EXTREME_killer_test.py..."
python EXTREME_killer_test.py
sleep 30  # Cooling break

# Run LAYERED
echo "🎯 Starting LAYERED_killer_test.py..."
python LAYERED_killer_test.py
sleep 30  # Cooling break

# Run MEGA_COMBO
echo "💥 Starting MEGA_COMBO_test.py..."
python MEGA_COMBO_test.py
sleep 30  # Cooling break

# Continue OPTIMIZED to 200
echo "🔄 Finishing OPTIMIZED_killer_test.py (50→200)..."
python OPTIMIZED_killer_test.py
sleep 30

echo "🎉 ALL TESTS COMPLETE!"
echo "Final time: $(date)"

# Generate summary report
./CHECK_PROGRESS.sh > FINAL_REPORT.txt
echo "📊 Report saved to FINAL_REPORT.txt"
