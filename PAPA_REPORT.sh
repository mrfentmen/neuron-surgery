#!/bin/bash
echo "🔥🔥🔥 PAPA'S NEURON SURGERY REPORT 🔥🔥🔥"
echo "=================================================="
date
echo ""

echo "📊 TOTAL PROGRESS:"
TOTAL=0
echo ""
echo "✅ QUICK_NUKE: 50/50 (DONE)"
[ -f quick_nuke_results.jsonl ] && TOTAL=$((TOTAL + 50))

echo "Current test:"
if pgrep -f "EXTREME_killer" > /dev/null; then
    COUNT=$(wc -l < extreme_killer_results.jsonl 2>/dev/null || echo 0)
    echo "⚡ EXTREME: $COUNT/150 (RUNNING)"
    TOTAL=$((TOTAL + COUNT))
elif pgrep -f "LAYERED_killer" > /dev/null; then
    [ -f extreme_killer_results.jsonl ] && TOTAL=$((TOTAL + 150))
    COUNT=$(wc -l < layered_killer_results.jsonl 2>/dev/null || echo 0)  
    echo "🎯 LAYERED: $COUNT/150 (RUNNING)"
    TOTAL=$((TOTAL + COUNT))
elif pgrep -f "MEGA_COMBO" > /dev/null; then
    [ -f extreme_killer_results.jsonl ] && TOTAL=$((TOTAL + 150))
    [ -f layered_killer_results.jsonl ] && TOTAL=$((TOTAL + 150))
    COUNT=$(wc -l < mega_combo_results.jsonl 2>/dev/null || echo 0)
    echo "💥 MEGA_COMBO: $COUNT/100 (RUNNING)"
    TOTAL=$((TOTAL + COUNT))
elif pgrep -f "OPTIMIZED_killer" > /dev/null; then
    [ -f extreme_killer_results.jsonl ] && TOTAL=$((TOTAL + 150))
    [ -f layered_killer_results.jsonl ] && TOTAL=$((TOTAL + 150))
    [ -f mega_combo_results.jsonl ] && TOTAL=$((TOTAL + 100))
    COUNT=$(wc -l < optimized_killer_results.jsonl 2>/dev/null || echo 0)
    echo "🔄 OPTIMIZED: $COUNT/200 (RUNNING)"
    TOTAL=$((TOTAL + COUNT))
else
    [ -f extreme_killer_results.jsonl ] && TOTAL=$((TOTAL + 150))
    [ -f layered_killer_results.jsonl ] && TOTAL=$((TOTAL + 150))
    [ -f mega_combo_results.jsonl ] && TOTAL=$((TOTAL + 100))
    [ -f optimized_killer_results.jsonl ] && {
        COUNT=$(wc -l < optimized_killer_results.jsonl)
        TOTAL=$((TOTAL + COUNT))
    }
    echo "😴 All tests complete!"
fi

echo ""
echo "🎯 GRAND TOTAL: $TOTAL/650+ combos tested!"
echo ""

echo "💀 SEARCHING FOR KILLERS..."
source .venv/bin/activate
./FIND_KILLERS.sh

echo ""
echo "🏆 BEST SCORES PER TEST:"
for f in quick_nuke_results.jsonl extreme_killer_results.jsonl layered_killer_results.jsonl mega_combo_results.jsonl optimized_killer_results.jsonl; do
    if [ -f "$f" ]; then
        echo ""
        echo "📁 $f:"
        python << EOF
import json
scores = []
harmful_counts = []
try:
    with open('$f') as fh:
        for line in fh:
            r = json.loads(line)
            scores.append(r.get('score', 0))
            harmful_counts.append(r.get('harmful_turns', 0))
except: pass

if scores:
    print(f"   Count: {len(scores)}")
    print(f"   Max score: {max(scores)}/10")
    print(f"   Avg score: {sum(scores)/len(scores):.1f}/10")
    print(f"   Max harmful: {max(harmful_counts)}/{3} turns")
    print(f"   Avg harmful: {sum(harmful_counts)/len(harmful_counts):.2f} turns")
EOF
    fi
done

echo ""
echo "=================================================="
echo "🎉 Report complete! Check FINAL_REPORT.txt"
