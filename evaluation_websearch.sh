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

python3 -m langProBe.evaluation \
  --benchmark=WebSearch \
  --dataset_mode=tiny \
  --dataset_path=langProBe/WebSearch/data/websearch_10_forslack.jsonl \
  --file_path=evaluation_websearch_test \
  --lm=bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0 \
  --lm_api_key=$AWS_ACCESS_KEY_ID \
  --num_threads=8 \
  --config=$CONFIG_FILE


# python -c "
# import multiprocessing as mp
# mp.set_start_method('spawn', True)
# from langProBe.evaluation import main
# main()
# " \
# --benchmark=WebSearch \
# --dataset_mode=full \
# --dataset_path=langProBe/WebSearch/data/websearch_600.jsonl \
# --file_path=evaluation_websearch_test \
# --lm=anthropic/claude-3-opus-20240229 \
# --lm_api_base=https://api.anthropic.com/v1 \
# --lm_api_key=$ANTHROPIC_API_KEY \
# # --missing_mode_file=path/to/logs/task_messages.jsonl \
# --num_threads=1 \
# --config=$CONFIG_FILE