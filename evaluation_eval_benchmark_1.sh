#!/bin/bash

# Refresh AWS credentials before running the evaluation
# if [ -f "../../refresh_aws_token.sh" ]; then
#     ../../refresh_aws_token.sh
#     if [ $? -ne 0 ]; then
#         echo "Failed to refresh AWS token. Please check your AWS login."
#         exit 1
#     fi
# fi

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

DATE=$(date +%F)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
dataset_name=eval_benchmark_1_10

python3 -m langProBe.evaluation \
  --benchmark=eval_benchmark_1 \
  --dataset_mode=tiny \
  --dataset_path=langProBe/eval_benchmark_1/data/$dataset_name.jsonl \
  --file_path=runs/$DATE/eval_benchmark_1_run_${TIMESTAMP}_${dataset_name} \
  --lm=bedrock/apac.anthropic.claude-3-7-sonnet-20250219-v1:0 \
  --lm_api_key=$AWS_ACCESS_KEY_ID \
  --num_threads=1 \
  --config=$CONFIG_FILE

# apac.anthropic.claude-3-5-sonnet-20241022-v2:0
# apac.anthropic.claude-sonnet-4-20250514-v1:0
# apac.anthropic.claude-3-7-sonnet-20250219-v1:0