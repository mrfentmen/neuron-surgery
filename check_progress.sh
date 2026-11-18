#!/bin/bash
# 📊 Quick progress checker

cd /Users/test/Desktop/neuron-surgery

echo "📊 NEURON SURGERY PROGRESS REPORT"
echo "=" | tr '=' '=' | head -c 60; echo
echo ""

# Check which test is running
if pgrep -f "OPTIMIZED_killer_test.py" > /dev/null; then
    echo "🏃 Currently running: OPTIMIZED_killer_test.py"
elif pgrep -f "EXTREME_killer_test.py" > /dev/null; then
    echo "🏃 Currently running: EXTREME_killer_test.py"
elif pgrep -f "LAYERED_killer_test.py" > /dev/null; then
    echo "🏃 Currently running: LAYERED_killer_test.py"
elif pgrep -f "MEGA_COMBO_test.py" > /dev/null; then
    echo "🏃 Currently running: MEGA_COMBO_test.py"
else
    echo "😴 No test currently running"
fi

echo ""
echo "📈 Combo counts:"

# OPTIMIZED
if [ -f "optimized_killer_results.jsonl" ]; then
    count=$(wc -l < optimized_killer_results.jsonl)
    echo "   OPTIMIZED: $count/200"
    
    # Last combo
    if [ $count -gt 0 ]; then
        last=$(tail -1 optimized_killer_results.jsonl | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"Combo {d['combo_num']}, Score {d['score']}\")")
        echo "      Latest: $last"
    fi
fi

# EXTREME
if [ -f "extreme_killer_results.jsonl" ]; then
    count=$(wc -l < extreme_killer_results.jsonl)
    echo "   EXTREME:   $count/150"
    
    if [ $count -gt 0 ]; then
        last=$(tail -1 extreme_killer_results.jsonl | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"Combo {d['combo_num']}, Score {d['score']}\")")
        echo "      Latest: $last"
    fi
fi

# LAYERED
if [ -f "layered_killer_results.jsonl" ]; then
    count=$(wc -l < layered_killer_results.jsonl)
    echo "   LAYERED:   $count/150"
    
    if [ $count -gt 0 ]; then
        last=$(tail -1 layered_killer_results.jsonl | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"Combo {d['combo_num']}, Score {d['score']}\")")
        echo "      Latest: $last"
    fi
fi

# MEGA
if [ -f "mega_combo_results.jsonl" ]; then
    count=$(wc -l < mega_combo_results.jsonl)
    echo "   MEGA:      $count/100"
    
    if [ $count -gt 0 ]; then
        last=$(tail -1 mega_combo_results.jsonl | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"Combo {d['combo_num']}, Score {d['score']}\")")
        echo "      Latest: $last"
    fi
fi

echo ""
echo "🏆 High scores so far:"

# Find top 3 scores across all files
python3 << 'PYTHON_TOP'
import json
import os

files = [
    ("OPT", "optimized_killer_results.jsonl"),
    ("EXT", "extreme_killer_results.jsonl"),
    ("LAY", "layered_killer_results.jsonl"),
    ("MEG", "mega_combo_results.jsonl")
]

all_results = []
for prefix, fname in files:
    if os.path.exists(fname):
        try:
            with open(fname, "r") as f:
                for line in f:
                    r = json.loads(line)
                    all_results.append((prefix, r))
        except:
            pass

if all_results:
    top = sorted(all_results, key=lambda x: x[1]["score"], reverse=True)[:5]
    for i, (prefix, r) in enumerate(top, 1):
        resp = r['responses'][0]['response'][:60]
        print(f"   {i}. [{prefix}] #{r['combo_num']}: Score {r['score']} - {resp}...")
else:
    print("   (No results yet)")

PYTHON_TOP

echo ""
