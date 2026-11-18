#!/bin/bash
# Auto-runner: Waits for Wave 1, then runs Wave 2 & 3 automatically

cd /Users/test/Desktop/neuron-surgery
source .venv/bin/activate

echo "🤖 AUTO-CONTINUE SYSTEM ACTIVE"
echo "==============================="
echo ""

# Monitor Wave 1
while true; do
    count=$(wc -l < massive_killer_sweep.jsonl 2>/dev/null || echo "0")
    
    if [ $count -ge 200 ]; then
        echo "🎉 WAVE 1 COMPLETE! ($count combos)"
        echo ""
        break
    fi
    
    # Check if process died
    if ! ps aux | grep -i "GENTLE_wave1" | grep -v grep > /dev/null; then
        echo "⚠️  Wave 1 process died at $count/200"
        echo "   Restarting in 30 seconds..."
        sleep 30
        
        # Restart Wave 1
        nohup python3 -u GENTLE_wave1_continue.py >> gentle_wave1.log 2>&1 &
        echo "   ✅ Restarted Wave 1"
    fi
    
    # Wait 5 minutes before checking again
    sleep 300
done

echo "🌊 Starting WAVE 2 (300 combos)..."
python3 MEGA_sweep_wave2.py
echo ""

echo "🌊 Starting WAVE 3 (400 combos)..."  
python3 MEGA_sweep_wave3.py
echo ""

echo "🎊🎊🎊 ALL WAVES COMPLETE! 🎊🎊🎊"
echo ""
echo "Total combos tested: 900"
echo "Results in:"
echo "  - massive_killer_sweep.jsonl"
echo "  - MEGA_wave2_results.jsonl"
echo "  - MEGA_wave3_crossbreed_results.jsonl"
