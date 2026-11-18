#!/bin/bash
# 🤖 AUTO NEXT TEST - Waits for OPTIMIZED, then runs QUICK_NUKE

cd /Users/test/Desktop/neuron-surgery

echo "🤖 AUTO NEXT TEST - Waiting for OPTIMIZED to finish..."
echo "=" | tr '=' '=' | head -c 60; echo

# Wait for OPTIMIZED to finish
while pgrep -f "OPTIMIZED_killer_test.py" > /dev/null; do
    # Show progress every 60 seconds
    count=$(wc -l < optimized_killer_results.jsonl 2>/dev/null || echo "0")
    echo "⏳ OPTIMIZED still running: $count/200 combos..."
    sleep 60
done

echo ""
echo "✅ OPTIMIZED DONE! Starting QUICK_NUKE in 30 seconds..."
sleep 30

# Start QUICK_NUKE
source .venv/bin/activate
echo "💥 Starting QUICK_NUKE test..."
python QUICK_NUKE_test.py 2>&1 | tee logs/quick_nuke.log

echo ""
echo "✅ QUICK_NUKE DONE!"
echo "📊 Check quick_nuke_results.jsonl for results"

# Show top scorers
if [ -f "quick_nuke_results.jsonl" ]; then
    echo ""
    echo "🏆 TOP 5 QUICK_NUKE SCORERS:"
    python3 << 'PYTHON_TOP'
import json
with open("quick_nuke_results.jsonl", "r") as f:
    results = [json.loads(line) for line in f]

top = sorted(results, key=lambda x: x["score"], reverse=True)[:5]
for i, r in enumerate(top, 1):
    print(f"\n{i}. Combo #{r['combo_num']} - Score: {r['score']}, {r['harmful_turns']}/3 harmful")
    print(f"   {r['num_neurons']} neurons")
    print(f"   {r['responses'][0]['response'][:80]}...")
PYTHON_TOP
fi

echo ""
echo "🎯 Want to run more? Use ./RUN_ALL_TESTS.sh"
