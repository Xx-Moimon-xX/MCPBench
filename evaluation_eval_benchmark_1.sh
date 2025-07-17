#!/bin/bash

if [ -f local.env ]; then
  set -a
  source local.env
  set +a
fi
# Check if the config file path parameter is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <config_file_path>"
  exit 1
fi

# Construct the full path
CONFIG_FILE="$1"
if [[ ! "$CONFIG_FILE" == /* ]]; then
  CONFIG_FILE="configs/$CONFIG_FILE"
  echo "Using config file: $CONFIG_FILE"
fi

# Start the evaluation program using a more direct method to ensure proper multiprocess initialization
DSPY_CACHEDIR=evaluation_mcp/.dspy_cache \

python3 -m langProBe.evaluation \
  --benchmark=eval_benchmark_1 \
  --dataset_mode=tiny \
  --dataset_path=langProBe/eval_benchmark_1/data/eval_benchmark_1_10.jsonl \
  --file_path=evaluation_eval_benchmark_1_test \
  --lm=bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0 \
  --lm_api_key=$AWS_ACCESS_KEY_ID \
  --num_threads=1 \
  --config=$CONFIG_FILE