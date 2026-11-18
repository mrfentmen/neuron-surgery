#!/usr/bin/env python3
"""Monitor mega experiment, then auto-launch follow-ups"""

import time
import json
import subprocess
import sys
from pathlib import Path

print("🔬 MONITORING MEGA EXPERIMENT...")
print("=" * 80)

# Wait for experiment to finish
while True:
    try:
        # Check if process still running
        result = subprocess.run(
            ["pgrep", "-f", "mega_neuron_experiment.py"],
            capture_output=True,
            text=True
        )
        
        # Count completed tests
        if Path("mega_experiment_results.jsonl").exists():
            with open("mega_experiment_results.jsonl") as f:
                completed = sum(1 for _ in f)
            print(f"⏳ Progress: {completed}/32 tests complete...", end='\r')
        
        if result.returncode != 0:  # Process finished
            print("\n✅ MEGA EXPERIMENT COMPLETE!")
            break
            
        time.sleep(10)  # Check every 10 seconds
        
    except KeyboardInterrupt:
        print("\n⚠️  Monitoring interrupted")
        sys.exit(1)

print("\n" + "=" * 80)
print("📊 ANALYZING RESULTS...")
print("=" * 80)

# Load all results
results = []
with open("mega_experiment_results.jsonl") as f:
    for line in f:
        results.append(json.loads(line))

# Analysis
print(f"\n✅ Total tests completed: {len(results)}")

# Find interesting ones
interesting = []
boring_count = 0

for r in results:
    text = r.get("output", "").lower()
    name = r.get("test_name", "unknown")
    
    # Check if it's the boring default response
    if "i am a person who is interested in learning" in text:
        boring_count += 1
    else:
        interesting.append({
            "name": name,
            "output": r.get("output", "")[:300],  # First 300 chars
            "steering": r.get("steering", {}),
            "temp": r.get("temperature", 0.0)
        })

print(f"\n📉 BORING RESPONSES: {boring_count}/32")
print(f"🎉 INTERESTING RESPONSES: {len(interesting)}/32")

if interesting:
    print("\n" + "=" * 80)
    print("🌟 SPICY FINDINGS:")
    print("=" * 80)
    for i, item in enumerate(interesting[:10], 1):  # Show top 10
        print(f"\n{i}. {item['name']}")
        print(f"   Steering: {item['steering']}")
        print(f"   Temp: {item['temp']}")
        print(f"   Output: {item['output'][:200]}...")
        print("-" * 80)

# Save analysis
with open("mega_experiment_analysis.json", "w") as f:
    json.dump({
        "total": len(results),
        "boring": boring_count,
        "interesting": len(interesting),
        "spicy_tests": interesting
    }, f, indent=2)

print("\n💾 Analysis saved to: mega_experiment_analysis.json")
print("\n🚀 Ready for follow-up experiments!")

