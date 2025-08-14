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
dataset_name=eval_benchmark_1_10_testing

# Model to use (no spaces around '=')
LM_IN_USE="anthropic/claude-sonnet-4-20250514"

# Build a descriptive, filesystem-safe run path that includes LM, config, and tool count
CONFIG_NAME="$(basename "$CONFIG_FILE")"
CONFIG_STEM="${CONFIG_NAME%.*}"
# Sanitize LM for path: replace non [A-Za-z0-9._-] with '-'
LM_SLUG="$(echo "$LM_IN_USE" | sed -e 's/[^A-Za-z0-9._-]/-/g')"
# Compute total tools for this config (parsed from helper script output). Falls back to 'unknown'.
TOOLS_OUT="$(python3 list_mcp_tools_by_hitting_remote_or_localhost.py "$CONFIG_NAME" 2>/dev/null || true)"
TOTAL_TOOLS="$(echo "$TOOLS_OUT" | grep -oE 'Total number of tools across all servers: [0-9]+' | awk '{print $NF}')"
if [ -z "$TOTAL_TOOLS" ]; then TOTAL_TOOLS="unknown"; fi

FILE_PATH="runs/$DATE/run_${TIMESTAMP}_${LM_SLUG}_t${TOTAL_TOOLS}_${CONFIG_STEM}_${dataset_name}"

python3 -m langProBe.evaluation \
  --benchmark=eval_benchmark_1 \
  --dataset_mode=tiny \
  --dataset_path=langProBe/eval_benchmark_1/data/$dataset_name.jsonl \
  --file_path="$FILE_PATH" \
  --lm="$LM_IN_USE" \
  --eval_lm=anthropic/claude-3-5-sonnet-20241022 \
  --lm_api_key=$AWS_ACCESS_KEY_ID \
  --num_threads=10 \
  --config=$CONFIG_FILE \
  --run_mode=generate_only

# apac.anthropic.claude-3-5-sonnet-20241022-v2:0
# apac.anthropic.claude-sonnet-4-20250514-v1:0
# apac.anthropic.claude-3-7-sonnet-20250219-v1:0


# {"data":[
#   {"type":"model","id":"claude-opus-4-20250514","display_name":"Claude Opus 4","created_at":"2025-05-22T00:00:00Z"},
#   {"type":"model","id":"claude-sonnet-4-20250514","display_name":"Claude Sonnet 4","created_at":"2025-05-22T00:00:00Z"},
#   {"type":"model","id":"claude-3-7-sonnet-20250219","display_name":"Claude Sonnet 3.7","created_at":"2025-02-24T00:00:00Z"},
#   {"type":"model","id":"claude-3-5-sonnet-20241022","display_name":"Claude Sonnet 3.5 (New)","created_at":"2024-10-22T00:00:00Z"},
#   {"type":"model","id":"claude-3-5-haiku-20241022","display_name":"Claude Haiku 3.5","created_at":"2024-10-22T00:00:00Z"},
#   {"type":"model","id":"claude-3-5-sonnet-20240620","display_name":"Claude Sonnet 3.5 (Old)","created_at":"2024-06-20T00:00:00Z"},
#   {"type":"model","id":"claude-3-haiku-20240307","display_name":"Claude Haiku 3","created_at":"2024-03-07T00:00:00Z"}
#   ],
# "has_more":false,"first_id":"claude-opus-4-20250514","last_id":"claude-3-haiku-20240307"}%