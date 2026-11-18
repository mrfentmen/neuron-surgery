#!/bin/bash
# Monitor the master test suite progress

echo "🔥💀 MONITORING MASTER TEST SUITE 💀🔥"
echo "======================================"
echo ""

# Check if the suite is running
if pgrep -f "PAPA_MASTER_TEST_SUITE.py" > /dev/null; then
    echo "✅ Test suite is RUNNING!"
    echo ""
    
    # Show the most recent results
    if [ -d "master_suite_results" ]; then
        echo "📊 Recent results:"
        ls -lht master_suite_results/ | head -10
        echo ""
        
        # Count completed tests
        test_count=$(ls master_suite_results/*.json 2>/dev/null | wc -l)
        echo "✅ Completed test files: $test_count"
    fi
    
    # Show last few lines of any log
    if [ -f "master_suite.log" ]; then
        echo ""
        echo "📝 Last 15 lines of log:"
        tail -15 master_suite.log
    fi
else
    echo "❌ Test suite is NOT running"
    
    if [ -d "master_suite_results" ]; then
        echo ""
        echo "📊 Final results available in master_suite_results/"
        ls -lht master_suite_results/
    fi
fi

echo ""
echo "======================================"
