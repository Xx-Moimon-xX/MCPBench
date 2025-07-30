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
python -c "
import multiprocessing as mp
mp.set_start_method('spawn', True)
from langProBe.evaluation import main
main()
" \
--benchmark=GAIA \
--dataset_mode=full \
--dataset_path=langProBe/GAIA/data/gaia_rest.jsonl \
--file_path=evaluation_gaia \
--lm=openai/qwen-max-2025-01-25 \
--lm_api_base=https://dashscope.aliyuncs.com/compatible-mode/v1 \
--missing_mode_file=path/to/logs/task_messages.jsonl \
--num_threads=1 \
--config=$CONFIG_FILE
