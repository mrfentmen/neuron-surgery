#!/bin/bash
# 🎯 MASTER TEST RUNNER - Runs all killer tests sequentially
# DO NOT RUN IN PARALLEL - will blow up the computer!

cd /Users/test/Desktop/neuron-surgery
source .venv/bin/activate

echo "🚀 MASTER TEST RUNNER STARTING..."
echo "=" | tr '=' '=' | head -c 60; echo
echo "Total tests: 4 scripts, 600 total combos"
echo "=" | tr '=' '=' | head -c 60; echo
echo ""

# Test 1: OPTIMIZED (if not already running)
if pgrep -f "OPTIMIZED_killer_test.py" > /dev/null; then
    echo "✅ Test 1: OPTIMIZED_killer_test.py already running, waiting..."
    while pgrep -f "OPTIMIZED_killer_test.py" > /dev/null; do
        sleep 30
    done
    echo "✅ Test 1: OPTIMIZED_killer_test.py DONE!"
else
    echo "🏃 Test 1/4: OPTIMIZED_killer_test.py (200 combos, 3-5 neurons)"
    python OPTIMIZED_killer_test.py 2>&1 | tee logs/optimized_test.log
    echo "✅ Test 1: DONE!"
fi

echo ""
echo "⏳ Cooling down (30 seconds)..."
sleep 30

# Test 2: EXTREME
echo ""
echo "🏃 Test 2/4: EXTREME_killer_test.py (150 combos, 6-10 neurons, EXTREME doses)"
python EXTREME_killer_test.py 2>&1 | tee logs/extreme_test.log
echo "✅ Test 2: DONE!"

echo ""
echo "⏳ Cooling down (30 seconds)..."
sleep 30

# Test 3: LAYERED
echo ""
echo "🏃 Test 3/4: LAYERED_killer_test.py (150 combos, layer-focused)"
python LAYERED_killer_test.py 2>&1 | tee logs/layered_test.log
echo "✅ Test 3: DONE!"

echo ""
echo "⏳ Cooling down (30 seconds)..."
sleep 30

# Test 4: MEGA COMBO
echo ""
echo "🏃 Test 4/4: MEGA_COMBO_test.py (100 combos, 12-20 neurons, 5 turns)"
python MEGA_COMBO_test.py 2>&1 | tee logs/mega_combo_test.log
echo "✅ Test 4: DONE!"

echo ""
echo "=" | tr '=' '=' | head -c 60; echo
echo "🎉 ALL TESTS COMPLETE!"
echo "=" | tr '=' '=' | head -c 60; echo
echo ""
echo "📊 Results summary:"
echo "   - OPTIMIZED: $(wc -l < optimized_killer_results.jsonl) combos"
echo "   - EXTREME:   $(wc -l < extreme_killer_results.jsonl) combos"
echo "   - LAYERED:   $(wc -l < layered_killer_results.jsonl) combos"
echo "   - MEGA:      $(wc -l < mega_combo_results.jsonl) combos"
echo ""

# Generate final report
python << 'PYTHON_REPORT'
import json

files = [
    ("OPTIMIZED", "optimized_killer_results.jsonl"),
    ("EXTREME", "extreme_killer_results.jsonl"),
    ("LAYERED", "layered_killer_results.jsonl"),
    ("MEGA", "mega_combo_results.jsonl")
]

print("🎯 TOP PERFORMERS:")
print("=" * 60)

all_results = []
for name, fname in files:
    try:
        with open(fname, "r") as f:
            results = [json.loads(line) for line in f]
            
        # Find top scorers
        top = sorted(results, key=lambda x: x["score"], reverse=True)[:3]
        
        print(f"\n{name} - Top 3:")
        for i, r in enumerate(top, 1):
            print(f"  {i}. Combo #{r['combo_num']} - Score: {r['score']}")
            print(f"     {r['responses'][0]['response'][:70]}...")
        
        all_results.extend([(name, r) for r in results])
    except:
        pass

# Overall top 10
print(f"\n{'='*60}")
print("🏆 OVERALL TOP 10 KILLER COMBOS:")
print("=" * 60)

all_sorted = sorted(all_results, key=lambda x: x[1]["score"], reverse=True)[:10]
for i, (test_name, r) in enumerate(all_sorted, 1):
    print(f"\n{i}. [{test_name}] Combo #{r['combo_num']} - Score: {r['score']}")
    print(f"   Steering: {len(r['steering'])} layers, neurons: {sum(len(v) for v in r['steering'].values())}")
    for turn_data in r['responses'][:2]:  # Show first 2 turns
        print(f"   Turn {turn_data['turn']}: {turn_data['response'][:80]}...")

PYTHON_REPORT

echo ""
echo "🎉 TESTS COMPLETE! Check the logs/ directory for details."
