#!/bin/bash
echo "🎯 NEURON SURGERY STATUS 🎯"
echo "========================================"
date
echo ""

# Check what's running
if pgrep -f "QUICK_NUKE" > /dev/null; then
    echo "🔥 QUICK_NUKE: RUNNING"
    wc -l quick_nuke_results.jsonl 2>/dev/null || echo "  0/50"
elif pgrep -f "EXTREME_killer" > /dev/null; then
    echo "⚡ EXTREME: RUNNING"
    wc -l extreme_killer_results.jsonl 2>/dev/null || echo "  0/150"
elif pgrep -f "LAYERED_killer" > /dev/null; then
    echo "🎯 LAYERED: RUNNING"  
    wc -l layered_killer_results.jsonl 2>/dev/null || echo "  0/150"
elif pgrep -f "MEGA_COMBO" > /dev/null; then
    echo "💥 MEGA_COMBO: RUNNING"
    wc -l mega_combo_results.jsonl 2>/dev/null || echo "  0/100"
elif pgrep -f "OPTIMIZED_killer" > /dev/null; then
    echo "🔄 OPTIMIZED: RUNNING"
    wc -l optimized_killer_results.jsonl 2>/dev/null || echo "  0/200"
else
    echo "😴 No test currently running"
fi

echo ""
echo "📈 TOTAL COMBOS TESTED:"
TOTAL=0
for f in quick_nuke_results.jsonl extreme_killer_results.jsonl layered_killer_results.jsonl mega_combo_results.jsonl optimized_killer_results.jsonl; do
    if [ -f "$f" ]; then
        COUNT=$(wc -l < "$f")
        TOTAL=$((TOTAL + COUNT))
        echo "  $f: $COUNT"
    fi
done
echo "  TOTAL: $TOTAL/650+"

echo ""
echo "🏆 LOOKING FOR BEST SCORES..."
# Find highest scores across all files
for f in *results.jsonl; do
    if [ -f "$f" ]; then
        python3 -c "
import json
try:
    scores = []
    with open('$f') as fh:
        for line in fh:
            r = json.loads(line)
            if 'score' in r:
                scores.append(r['score'])
    if scores:
        print(f'  $f: Max={max(scores)}, Avg={sum(scores)/len(scores):.1f}')
except: pass
" 2>/dev/null
    fi
done
