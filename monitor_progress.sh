#!/bin/bash
# 🔥💀 PAPA'S EXPERIMENT MONITOR 💀🔥

echo "=================================="
echo "🔬 EXPERIMENT 1 PROGRESS MONITOR"
echo "=================================="
echo ""

# Check if process is running
if ps aux | grep -q "[U]LTIMATE_neuron_atlas"; then
    echo "✅ Status: RUNNING"
    
    # Get CPU and memory usage
    ps aux | grep "[U]LTIMATE_neuron_atlas" | awk '{print "   CPU: " $3 "% | Memory: " $4 "% (" $6/1024/1024 " GB)"}'
    echo ""
else
    echo "❌ Status: NOT RUNNING"
    echo ""
fi

# Check output file
if ls ultimate_neuron_atlas*.jsonl 2>/dev/null; then
    echo "📊 Results File:"
    ls -lh ultimate_neuron_atlas*.jsonl | awk '{print "   Size: " $5 " | Modified: " $6 " " $7 " " $8}'
    echo ""
    
    # Count completed tests
    COUNT=$(wc -l < ultimate_neuron_atlas*.jsonl)
    echo "📈 Progress: $COUNT / ~245 tests completed"
    
    # Calculate percentage
    PERCENT=$((COUNT * 100 / 245))
    echo "   Progress: $PERCENT%"
    
    # Create progress bar
    BARS=$((PERCENT / 5))
    BAR_STRING=$(printf '█%.0s' $(seq 1 $BARS))
    SPACES=$((20 - BARS))
    SPACE_STRING=$(printf '░%.0s' $(seq 1 $SPACES))
    echo "   [$BAR_STRING$SPACE_STRING] $PERCENT%"
    echo ""
    
    # Estimate time remaining (rough estimate: ~30 seconds per test)
    REMAINING=$((245 - COUNT))
    MINUTES=$((REMAINING * 30 / 60))
    echo "⏱️  Estimated time remaining: ~$MINUTES minutes"
    echo ""
else
    echo "⚠️  No results file found yet (still loading model?)"
    echo ""
fi

# Check for errors in log
if [ -f "ultimate_atlas.log" ]; then
    if grep -q "Error\|Exception\|Traceback" ultimate_atlas.log 2>/dev/null; then
        echo "⚠️  ERRORS DETECTED IN LOG!"
        echo "   Run: tail -50 ultimate_atlas.log"
        echo ""
    fi
fi

echo "=================================="
echo "Last updated: $(date '+%I:%M:%S %p')"
echo "=================================="
