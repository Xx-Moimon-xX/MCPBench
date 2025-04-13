#!/bin/bash

# 检查是否提供了配置文件路径参数
if [ -z "$1" ]; then
  echo "Usage: $0 <config_file_path>"
  exit 1
fi

# 构造完整路径
CONFIG_FILE="$1"
if [[ ! "$CONFIG_FILE" == /* ]]; then
  CONFIG_FILE="configs/$CONFIG_FILE"
fi

# 读取配置文件中的配置信息
COMMAND=$(jq -r '.command' "$CONFIG_FILE")
ARGS=$(jq -r '.args' "$CONFIG_FILE")
PORT=$(jq -r '.port' "$CONFIG_FILE")

# 使用从配置文件中读取的配置信息启动服务
npx -y supergateway \
    --stdio "$ARGS $COMMAND" \
    --port "$PORT" --baseUrl "http://localhost:$PORT" \
    --ssePath /sse --messagePath /message