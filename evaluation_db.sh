#!/bin/bash

if [ -f local.env ]; then
  set -a
  source local.env
  set +a
fi

 # Check if configuration file path parameter is provided
 if [ -z "$1" ]; then
   echo "Usage: $0 <config_file_path>"
   exit 1
 fi

 # Construct full path
 CONFIG_FILE="$1"
 if [[ ! "$CONFIG_FILE" == /* ]]; then
   CONFIG_FILE="configs/$CONFIG_FILE"
 fi



# Use a more direct method to start the evaluation program, ensuring correct multiprocessing initialization
DSPY_CACHEDIR=evaluation_mcp/.dspy_cache \
python3 -c "
import multiprocessing as mp
mp.set_start_method('spawn', True)
from langProBe.evaluation import main
main()
" \
--benchmark=DB \
--dataset_mode=test \
--dataset_path=langProBe/DB/data/car_bi.jsonl \
--file_path=evaluation_db \
--lm=bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0 \
--lm_api_key=$AWS_ACCESS_KEY_ID \
--num_threads=1 \
--config=$CONFIG_FILE


# python3 -m langProBe.evaluation \
#   --benchmark=WebSearch \
#   --dataset_mode=tiny \
#   --dataset_path=langProBe/WebSearch/data/websearch_10_foratlassian.jsonl \
#   --file_path=evaluation_websearch_test \
#   --lm=bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0 \
#   --lm_api_key=$AWS_ACCESS_KEY_ID \
#   --num_threads=1 \
#   --config=$CONFIG_FILE