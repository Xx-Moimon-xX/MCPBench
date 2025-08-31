#!/bin/bash

# Script to run response_evaluation_eval_3.sh for all run directories
# Usage: ./run_eval3_all_runs.sh [config_file] [runs_directory]

# Default values
CONFIG_FILE="slack_mutated.json"
RUNS_DIR="../../results/runs_slack_at_end_formatted_tools"

# Check if runs directory is provided as second argument
if [ ! -z "$1" ]; then
    RUNS_DIR="$1"
fi

# Validate config file exists
if [ ! -f "$CONFIG_FILE" ] && [ ! -f "configs/$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found in current directory or configs/ subdirectory"
    echo "Usage: $0 [config_file] [runs_directory]"
    echo "Example: $0 slack_mutated.json ../../results/runs_slack_at_end_formatted_tools"
    exit 1
fi

# Validate runs directory exists
if [ ! -d "$RUNS_DIR" ]; then
    echo "Error: Runs directory '$RUNS_DIR' not found"
    echo "Usage: $0 [config_file] [runs_directory]"
    echo "Example: $0 slack_mutated.json ../../results/runs_slack_at_end_formatted_tools"
    exit 1
fi

echo "Starting evaluation for all runs in: $RUNS_DIR"
echo "Using config file: $CONFIG_FILE"
echo "=================================="

# Counter for tracking progress
TOTAL_RUNS=0
COMPLETED_RUNS=0
FAILED_RUNS=0

# Get list of all run directories
RUN_DIRS=($(find "$RUNS_DIR" -maxdepth 1 -type d -name "run_*" | sort))

TOTAL_RUNS=${#RUN_DIRS[@]}

if [ $TOTAL_RUNS -eq 0 ]; then
    echo "No run directories found in $RUNS_DIR"
    exit 1
fi

echo "Found $TOTAL_RUNS run directories to evaluate"
echo "=================================="

# Function to run evaluation for a single run directory
run_evaluation() {
    local run_dir="$1"
    local run_name=$(basename "$run_dir")
    
    echo "[$COMPLETED_RUNS/$TOTAL_RUNS] Processing: $run_name"
    
    # Check if predictions file exists
    local predictions_file="$run_dir/response_data/predictions_eval3.jsonl"
    if [ ! -f "$predictions_file" ]; then
        echo "  ⚠️  Warning: predictions_eval3.jsonl not found in $run_name"
        return 1
    fi
    
    # Run the evaluation
    echo "  🚀 Running evaluation..."
    if ./response_evaluation_eval_3.sh "$CONFIG_FILE" "$run_dir"; then
        echo "  ✅ Successfully completed evaluation for $run_name"
        return 0
    else
        echo "  ❌ Failed to complete evaluation for $run_name"
        return 1
    fi
}

# Process each run directory
for run_dir in "${RUN_DIRS[@]}"; do
    if run_evaluation "$run_dir"; then
        ((COMPLETED_RUNS++))
    else
        ((FAILED_RUNS++))
    fi
    
    echo "  ---"
done

echo "=================================="
echo "Evaluation Summary:"
echo "Total runs: $TOTAL_RUNS"
echo "Completed: $COMPLETED_RUNS"
echo "Failed: $FAILED_RUNS"
echo "=================================="

if [ $FAILED_RUNS -gt 0 ]; then
    echo "⚠️  Some evaluations failed. Check the logs above for details."
    exit 1
else
    echo "🎉 All evaluations completed successfully!"
    exit 0
fi
