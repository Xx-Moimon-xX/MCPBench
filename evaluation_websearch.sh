#!/bin/bash
# Check if the config file path parameter is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <config_file_path>"
  exit 1
fi

# Construct the full path
CONFIG_FILE="$1"
if [[ ! "$CONFIG_FILE" == /* ]]; then
  CONFIG_FILE="configs/$CONFIG_FILE"
fi

# Start the evaluation program using a more direct method to ensure proper multiprocess initialization
DSPY_CACHEDIR=evaluation_mcp/.dspy_cache \
python -c "
import multiprocessing as mp
mp.set_start_method('spawn', True)
from langProBe.evaluation import main
main()
" \
--benchmark=WebSearch \
--dataset_mode=full \
--dataset_path=langProBe/WebSearch/data/websearch_test.jsonl \
--file_path=evaluation_websearch_test \
--lm=openai/deepseek-v3 \
--lm_api_base=https://dashscope.aliyuncs.com/compatible-mode/v1 \
--lm_api_key=xxx \
--missing_mode_file=path/to/logs/task_messages.jsonl \
--num_threads=1 \
--config=$CONFIG_FILE