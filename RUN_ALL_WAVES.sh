#!/bin/bash
# Master runner for all 3 waves (900 total combos)
# Run this when WAVE 1 finishes

cd /Users/test/Desktop/neuron-surgery
source .venv/bin/activate

echo "🌊 STARTING ALL NEURON SWEEP WAVES"
echo "==================================="
echo ""

# Check if Wave 1 is done
wave1_count=$(wc -l < massive_killer_sweep.jsonl 2>/dev/null || echo "0")
if [ "$wave1_count" -lt 200 ]; then
    echo "⚠️  WAVE 1 not complete yet ($wave1_count / 200)"
    echo "   Let it finish first, then run this script"
    exit 1
fi

echo "✅ WAVE 1 complete with $wave1_count combos"
echo ""

# Run Wave 2
echo "🌊 Starting WAVE 2: 300 weird strategy combos..."
python3 MEGA_sweep_wave2.py
echo ""

# Run Wave 3  
echo "🌊 Starting WAVE 3: 400 crossbreed combos..."
python3 MEGA_sweep_wave3.py
echo ""

echo "🎉🎉🎉 ALL WAVES COMPLETE! 🎉🎉🎉"
echo ""
echo "Results saved to:"
echo "  - massive_killer_sweep.jsonl (Wave 1)"
echo "  - MEGA_wave2_results.jsonl (Wave 2)"
echo "  - MEGA_wave3_crossbreed_results.jsonl (Wave 3)"
echo ""
echo "Run ./check_all_progress.sh to see top performers!"
