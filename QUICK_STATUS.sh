#!/bin/bash
# Quick status check - run this anytime

echo "🔥 QUICK STATUS CHECK"
echo "===================="

# Check file count
count=$(wc -l < massive_killer_sweep.jsonl 2>/dev/null || echo "0")
echo "📊 Combos done: $count / 200"

# Check if running
if ps aux | grep -i "GENTLE_wave1" | grep -v grep > /dev/null; then
    echo "✅ Process RUNNING"
else
    echo "❌ Process STOPPED"
fi

# Show latest result
if [ -f massive_killer_sweep.jsonl ] && [ $count -gt 0 ]; then
    echo ""
    echo "Latest result:"
    tail -1 massive_killer_sweep.jsonl | python3 -c "import json,sys; r=json.loads(sys.stdin.read()); print(f'  Combo {r[\"combo_id\"]}: Score={r.get(\"villain_score\",\"?\")} Neurons={r.get(\"neuron_count\",\"?\")} Success={r.get(\"success\",False)}')"
fi

echo ""
echo "Run ./check_all_progress.sh for full details!"
