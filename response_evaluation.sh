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
# Check required params: config and run dir or predictions jsonl
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <config_file_path> <run_dir_or_predictions_jsonl>"
  exit 1
fi

# Construct the full path
CONFIG_FILE="$1"
PRED_INPUT="$2"
if [[ ! "$CONFIG_FILE" == /* ]]; then
  CONFIG_FILE="configs/$CONFIG_FILE"
  echo "Using config file: $CONFIG_FILE"
fi

# Resolve dataset_path and output file_path to be the same root
if [ -d "$PRED_INPUT" ]; then
  RUN_DIR="$PRED_INPUT"
  DATASET_PATH="$RUN_DIR/response_data/predictions.jsonl"
else
  PRED_JSONL_ABS="$PRED_INPUT"
  RUN_DIR="$(cd "$(dirname "$PRED_JSONL_ABS")/.." && pwd)"
  DATASET_PATH="$PRED_JSONL_ABS"
fi

if [ ! -f "$DATASET_PATH" ]; then
  echo "Could not find predictions JSONL at: $DATASET_PATH"
  exit 1
fi

# Start the evaluation program using a more direct method to ensure proper multiprocess initialization
DSPY_CACHEDIR=evaluation_mcp/.dspy_cache \

python3 -m langProBe.evaluation \
  --benchmark=eval_benchmark_1 \
  --dataset_mode=tiny \
  --dataset_path="$DATASET_PATH" \
  --file_path="$RUN_DIR" \
  --lm=anthropic/claude-sonnet-4-20250514 \
  --lm_api_key=$AWS_ACCESS_KEY_ID \
  --eval_lm=anthropic/claude-3-5-sonnet-20241022 \
  --num_threads=1 \
  --config=$CONFIG_FILE \
  --run_mode=score_only

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