#!/bin/bash
# Comprehensive report for when papa gets back with bread

cd /Users/test/Desktop/neuron-surgery

echo "🍞🥤 WELCOME BACK PAPA! HERE'S WHAT HAPPENED:"
echo "=============================================="
echo ""

# Wave 1 status
w1=$(wc -l < massive_killer_sweep.jsonl 2>/dev/null || echo "0")
echo "📊 WAVE 1 (Killer Combos): $w1 / 200"
if [ $w1 -ge 200 ]; then
    echo "   ✅ COMPLETE!"
elif ps aux | grep -i "GENTLE_wave1" | grep -v grep > /dev/null; then
    echo "   🏃 RUNNING..."
else
    echo "   ⚠️  STOPPED"
fi

# Wave 2 status
w2=$(wc -l < MEGA_wave2_results.jsonl 2>/dev/null || echo "0")
echo "📊 WAVE 2 (Weird Strategies): $w2 / 300"
if [ $w2 -ge 300 ]; then
    echo "   ✅ COMPLETE!"
elif [ $w2 -gt 0 ]; then
    echo "   🏃 RUNNING OR IN PROGRESS..."
else
    echo "   ⏳ NOT STARTED YET"
fi

# Wave 3 status
w3=$(wc -l < MEGA_wave3_crossbreed_results.jsonl 2>/dev/null || echo "0")
echo "📊 WAVE 3 (Crossbreeds): $w3 / 400"
if [ $w3 -ge 400 ]; then
    echo "   ✅ COMPLETE!"
elif [ $w3 -gt 0 ]; then
    echo "   🏃 RUNNING OR IN PROGRESS..."
else
    echo "   ⏳ NOT STARTED YET"
fi

echo ""
total=$((w1 + w2 + w3))
echo "🎯 TOTAL PROGRESS: $total / 900 combos ($(( total * 100 / 900 ))%)"
echo ""

# Top performers from Wave 1
if [ $w1 -gt 10 ]; then
    echo "🏆 TOP 5 WAVE 1 KILLERS:"
    cat massive_killer_sweep.jsonl | \
        python3 -c "
import json, sys
results = [json.loads(l) for l in sys.stdin if json.loads(l).get('success')]
top = sorted(results, key=lambda x: x.get('villain_score', 0), reverse=True)[:5]
for i, r in enumerate(top, 1):
    steering_summary = f\"{len(r['steering'])} layers, {r['neuron_count']} neurons\"
    print(f\"  {i}. Score {r['villain_score']}: {steering_summary} @ temp {r['temperature']:.2f}\")
    # Show a sample response
    if r.get('turns'):
        sample = r['turns'][0]['response'][:100]
        print(f\"     '{sample}...'\")
" 2>/dev/null
    echo ""
fi

# Top performers from Wave 2
if [ $w2 -gt 10 ]; then
    echo "🏆 TOP 5 WAVE 2 WEIRDOS:"
    cat MEGA_wave2_results.jsonl | \
        python3 -c "
import json, sys
results = [json.loads(l) for l in sys.stdin if json.loads(l).get('success')]
top = sorted(results, key=lambda x: x.get('villain_score', 0), reverse=True)[:5]
for i, r in enumerate(top, 1):
    print(f\"  {i}. Score {r['villain_score']}: {r['neuron_count']} neurons @ temp {r['temperature']:.2f}\")
" 2>/dev/null
    echo ""
fi

# Top performers from Wave 3
if [ $w3 -gt 10 ]; then
    echo "🏆 TOP 5 WAVE 3 CROSSBREEDS:"
    cat MEGA_wave3_crossbreed_results.jsonl | \
        python3 -c "
import json, sys
results = [json.loads(l) for l in sys.stdin if json.loads(l).get('success')]
top = sorted(results, key=lambda x: x.get('scores', {}).get('total', 0), reverse=True)[:5]
for i, r in enumerate(top, 1):
    scores = r.get('scores', {})
    print(f\"  {i}. Total {scores.get('total', 0)}: {r['test_type']}\")
    print(f\"     Villain:{scores.get('villain',0)} Lover:{scores.get('lover',0)} Religious:{scores.get('religious',0)}\")
" 2>/dev/null
    echo ""
fi

# Check processes
echo "💻 SYSTEM STATUS:"
if ps aux | grep -E "GENTLE_wave1|MEGA_sweep|AUTO_CONTINUE" | grep -v grep > /dev/null; then
    echo "   ✅ Processes running"
    ps aux | grep -E "GENTLE_wave1|MEGA_sweep|AUTO_CONTINUE" | grep -v grep | awk '{print "   - "$11}'
else
    echo "   ✅ All tests complete!"
fi

echo ""
echo "📁 RESULT FILES:"
[ -f massive_killer_sweep.jsonl ] && echo "   ✅ massive_killer_sweep.jsonl"
[ -f MEGA_wave2_results.jsonl ] && echo "   ✅ MEGA_wave2_results.jsonl"
[ -f MEGA_wave3_crossbreed_results.jsonl ] && echo "   ✅ MEGA_wave3_crossbreed_results.jsonl"

echo ""
echo "🎮 NEXT STEPS:"
echo "   1. Review top performers above"
echo "   2. Run specific multi-turn tests on best combos"
echo "   3. Try combining different persona types"
echo "   4. Test for coherent killer personalities!"
