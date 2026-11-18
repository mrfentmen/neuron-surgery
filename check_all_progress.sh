#!/bin/bash
# Quick progress checker for all sweep waves

echo "🔥 NEURON SWEEP PROGRESS REPORT"
echo "==============================="
echo ""

# Wave 1 (original massive sweep)
if [ -f "massive_killer_sweep.jsonl" ]; then
    wave1_count=$(wc -l < massive_killer_sweep.jsonl)
    echo "📊 WAVE 1 (Original): $wave1_count / 200 combos"
    
    if [ $wave1_count -gt 0 ]; then
        echo "   Top 3 scores:"
        cat massive_killer_sweep.jsonl | \
            python3 -c "import json,sys; results=[json.loads(l) for l in sys.stdin if json.loads(l).get('success')]; top=sorted(results, key=lambda x: x.get('villain_score',0), reverse=True)[:3]; [print(f'   - Score {r[\"villain_score\"]}: {r[\"steering\"]}') for r in top]" 2>/dev/null || echo "   (processing...)"
    fi
else
    echo "📊 WAVE 1: Not started"
fi
echo ""

# Wave 2
if [ -f "MEGA_wave2_results.jsonl" ]; then
    wave2_count=$(wc -l < MEGA_wave2_results.jsonl)
    echo "📊 WAVE 2 (Weird Strategies): $wave2_count / 300 combos"
    
    if [ $wave2_count -gt 0 ]; then
        echo "   Top 3 scores:"
        cat MEGA_wave2_results.jsonl | \
            python3 -c "import json,sys; results=[json.loads(l) for l in sys.stdin if json.loads(l).get('success')]; top=sorted(results, key=lambda x: x.get('villain_score',0), reverse=True)[:3]; [print(f'   - Score {r[\"villain_score\"]}: {r[\"steering\"]}') for r in top]" 2>/dev/null || echo "   (processing...)"
    fi
else
    echo "📊 WAVE 2: Not started"
fi
echo ""

# Wave 3
if [ -f "MEGA_wave3_crossbreed_results.jsonl" ]; then
    wave3_count=$(wc -l < MEGA_wave3_crossbreed_results.jsonl)
    echo "📊 WAVE 3 (Crossbreeds): $wave3_count / 400 combos"
    
    if [ $wave3_count -gt 0 ]; then
        echo "   Top 3 total scores:"
        cat MEGA_wave3_crossbreed_results.jsonl | \
            python3 -c "import json,sys; results=[json.loads(l) for l in sys.stdin if json.loads(l).get('success')]; top=sorted(results, key=lambda x: x.get('scores',{}).get('total',0), reverse=True)[:3]; [print(f'   - Type {r[\"test_type\"]}: {r[\"scores\"]}') for r in top]" 2>/dev/null || echo "   (processing...)"
    fi
else
    echo "📊 WAVE 3: Not started"
fi
echo ""

# Check if Python is running
python_count=$(ps aux | grep -i python | grep -v grep | grep -c neuron-surgery)
if [ $python_count -gt 0 ]; then
    echo "✅ Python process ACTIVE ($python_count process(es))"
else
    echo "❌ No Python processes running"
fi
echo ""

# Total progress
total_done=$((${wave1_count:-0} + ${wave2_count:-0} + ${wave3_count:-0}))
total_target=900  # 200 + 300 + 400
echo "🎯 TOTAL PROGRESS: $total_done / $total_target combos ($(( total_done * 100 / total_target ))%)"
echo ""

# ETA estimation
if [ $python_count -gt 0 ] && [ $total_done -gt 0 ]; then
    # Rough estimate: ~5 mins per combo
    remaining=$(( total_target - total_done ))
    eta_hours=$(( remaining * 5 / 60 ))
    echo "⏰ ROUGH ETA: ~$eta_hours hours remaining"
fi
