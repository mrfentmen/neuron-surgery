#!/bin/bash
# MEGA TEST ASSAULT - Run ALL tests back-to-back!

cd /Users/test/Desktop/neuron-surgery
source .venv/bin/activate

echo "🚀🚀🚀 MEGA ASSAULT STARTING! 🚀🚀🚀"
date

# Wait for QUICK_NUKE to finish
echo "⏳ Waiting for QUICK_NUKE (50 combos)..."
while pgrep -f "QUICK_NUKE_test.py" > /dev/null; do
    COUNT=$(wc -l < quick_nuke_results.jsonl 2>/dev/null || echo 0)
    echo "  ... $COUNT/50 done"
    sleep 15
done
echo "✅ QUICK_NUKE DONE!"
sleep 60  # Cool down

# EXTREME (150 combos)
echo "⚡⚡ STARTING EXTREME_killer_test.py (150 combos)..."
python EXTREME_killer_test.py
echo "✅ EXTREME DONE!"
sleep 60

# LAYERED (150 combos)  
echo "🎯🎯 STARTING LAYERED_killer_test.py (150 combos)..."
python LAYERED_killer_test.py
echo "✅ LAYERED DONE!"
sleep 60

# MEGA_COMBO (100 combos)
echo "💥💥 STARTING MEGA_COMBO_test.py (100 combos)..."
python MEGA_COMBO_test.py
echo "✅ MEGA_COMBO DONE!"
sleep 60

# Finish OPTIMIZED (150 more combos)
echo "🔄🔄 FINISHING OPTIMIZED_killer_test.py (50→200)..."
python OPTIMIZED_killer_test.py
echo "✅ OPTIMIZED DONE!"

echo ""
echo "🎉🎉🎉 ALL TESTS COMPLETE! 🎉🎉🎉"
echo "Total combos: 650+"
date

# Generate final report
echo ""
echo "📊 GENERATING FINAL REPORT..."
./CHECK_PROGRESS.sh > FINAL_MEGA_REPORT.txt 2>&1
cat FINAL_MEGA_REPORT.txt
