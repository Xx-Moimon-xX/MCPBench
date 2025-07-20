#!/bin/bash

# Refresh AWS credentials before running the evaluation
# if [ -f "refresh_aws_token.sh" ]; then
#     ./refresh_aws_token.sh
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
fi

# Check if the config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Config file '$CONFIG_FILE' does not exist."
  exit 1
fi

# Get the length of the mcp_pool array
SERVER_COUNT=$(jq '.mcp_pool | length' "$CONFIG_FILE")

if [[ "$SERVER_COUNT" -eq 0 ]]; then
  echo "No servers defined in mcp_pool."
  exit 1
fi

# Iterate over the mcp_pool array and start each server
for (( i=0; i<SERVER_COUNT; i++ ))
do
  # Use jq to extract each server's config
  SERVER=$(jq ".mcp_pool[$i]" "$CONFIG_FILE")

  NAME=$(echo "$SERVER" | jq -r '.name')

  # Check if the url field exists
  URL=$(echo "$SERVER" | jq -r '.url // empty')

  if [[ -n "$URL" ]]; then
    # If url exists, do not run run_config, just output info
    echo "Server '$NAME' is configured with URL: $URL, skipping run command."
  else
    # Extract command, args, and port from run_config array
    COMMAND=$(echo "$SERVER" | jq -r '.run_config[] | select(.command) | .command')
    ARGS=$(echo "$SERVER" | jq -r '.run_config[] | select(.args) | .args | if type == "array" then join(" ") else . end')
    PORT=$(echo "$SERVER" | jq -r '.run_config[] | select(.port) | .port')

    # Extract env from run_config array (if present)
    ENV_VARS=$(echo "$SERVER" | jq -r '.run_config[] | select(.env) | .env | to_entries[]? | "export " + .key + "=\"" + .value + "\""')

    # Extract tool_name from tools array (assume first tool)
    TOOL_NAME=$(echo "$SERVER" | jq -r '.tools[0].tool_name')

    # tool_keyword can be set to empty string or defined as needed
    TOOL_KEYWORD=""

    echo "Starting server: $NAME on port $PORT"

    # Export env vars if present
    if [[ -n "$ENV_VARS" ]]; then
      eval "$ENV_VARS"
    fi

    # Start the server in the background
    npx -y supergateway \
      --stdio "$COMMAND $ARGS" \
      --port "$PORT" \
      --baseUrl "http://localhost:$PORT" \
      --ssePath /sse \
      --messagePath /message \
      --name "$TOOL_NAME" \
      --keyword "$TOOL_KEYWORD" &

    PID=$!
    echo "Server '$NAME' started, PID: $PID"
  fi
done

wait
