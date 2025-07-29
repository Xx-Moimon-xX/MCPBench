import asyncio
import sys
import json
import os

# This script requires the 'mcp' library.
# You can install it in a virtual environment using:
# uv init
# uv venv
# uv add mcp
try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
except ImportError:
    print("Error: The 'mcp' library is not installed.")
    print("Please set up a virtual environment and install it with 'uv add mcp'.")
    sys.exit(1)

def get_server_url_and_headers_from_config(config_filename: str):
    """Extracts the first server's URL and headers from the config. Supports both direct URL and host/port config styles."""
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    config_path = os.path.join(config_dir, config_filename)
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at '{config_path}'")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{config_path}'")
        sys.exit(1)

    mcp_pool = config.get("mcp_pool", [])
    if not mcp_pool:
        print(f"No 'mcp_pool' found in '{config_path}'")
        sys.exit(1)

    server_config = mcp_pool[0]
    # If direct URL is provided, use it
    if "url" in server_config:
        url = server_config["url"]
        headers = server_config.get("headers", None)
        return url, headers
    # Otherwise, use host/port logic
    run_configs = server_config.get("run_config", [])
    if not run_configs:
        print(f"No 'run_config' found for server '{server_config.get('name', 'unknown')}'.")
        sys.exit(1)
    run_config = run_configs[0]
    port = run_config.get("port")
    if not port:
        print(f"No 'port' found in run_config for server '{server_config.get('name', 'unknown')}'.")
        sys.exit(1)
    host = run_config.get("host", "localhost")
    url = f"http://{host}:{port}/sse"
    return url, None

async def list_mcp_tools_from_url(server_url: str, headers=None):
    """Connects to an already running MCP server via SSE and lists its tools."""
    print(f"\nConnecting to MCP server at: {server_url}")
    if headers:
        print(f"Using custom headers: {headers}")
    try:
        async with sse_client(url=server_url, headers=headers) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                response = await session.list_tools()
                tools = response.tools
                print(f"\nSuccessfully connected to server at '{server_url}'.")
                print(f"Response: {response}")
                print(f"Tools raw: {tools}")
                print("Available tools:")
                if not tools:
                    print("No tools found.")
                for tool in tools:
                    print(f"\n- Name: {tool.name}")
                    if tool.description:
                        print(f"  Description: {tool.description}")
                    if tool.inputSchema:
                        print(f"  Input Schema: {json.dumps(tool.inputSchema, indent=2)}")
    except Exception as e:
        print(f"\nAn error occurred while connecting to '{server_url}': {e}")
        print("Please ensure the server is running and the URL is correct.")

async def main():
    if len(sys.argv) < 2:
        print("Usage: python3 list_mcp_tools_by_hitting_remote_or_localhost.py <config_file_name>")
        print("\nExample:")
        print("  python3 list_mcp_tools_by_hitting_remote_or_localhost.py slack.json")
        sys.exit(1)
    config_filename = sys.argv[1]
    server_url, headers = get_server_url_and_headers_from_config(config_filename)
    await list_mcp_tools_from_url(server_url, headers=headers)

if __name__ == "__main__":
    asyncio.run(main())

# To run it:
# export $(cat local.env | xargs) && source .venv/bin/activate && python3 list_mcp_tools_by_hitting_remote_or_localhost.py slack.json