#!/bin/bash

# Script to run response_generation.sh multiple times
# Usage: ./run_multiple_times.sh

echo "Starting multiple runs of response_generation.sh..."
echo "This will run the script 5 times with: slack_mutated.json json"
echo ""

# Counter for tracking runs
run_count=1
total_runs=10

# Loop to run the command 8 times
for ((i=1; i<=total_runs; i++)); do
    echo "=========================================="
    echo "Starting run $i of $total_runs"
    echo "Timestamp: $(date)"
    echo "=========================================="
    
    # Run the response_generation.sh script
    ./response_generation.sh slack_mutated.json
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Run $i completed successfully"
    else
        echo "Run $i failed with exit code $?"
    fi
    
    echo ""
    
    # Add a small delay between runs (optional, remove if not needed)
    if [ $i -lt $total_runs ]; then
        echo "Waiting 10 seconds before next run..."
        sleep 10
        echo ""
    fi
done

echo "=========================================="
echo "All $total_runs runs completed!"
echo "Final timestamp: $(date)"
echo "=========================================="
