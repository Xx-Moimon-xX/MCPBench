import asyncio
import sys
import json
import os
import re

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

def load_env_vars(env_path):
    """Load key-value pairs from a .env file into a dictionary."""
    env_vars = {}
    if not os.path.exists(env_path):
        return env_vars
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            env_vars[key.strip()] = value.strip()
    return env_vars

def find_token_for_server(server_name, env_vars):
    """Find a token in env_vars whose key starts with the server's name (case-insensitive, underscores/dashes normalized)."""
    norm_name = re.sub(r'[-_]', '', server_name).lower()
    for key, value in env_vars.items():
        norm_key = re.sub(r'[-_]', '', key).lower()
        if norm_key.startswith(norm_name) and (key.endswith('_TOKEN') or key.endswith('_API_KEY') or key.endswith('_API_TOKEN') or key.endswith('_PERSONAL_ACCESS_TOKEN')):
            return value, key
    return None, None

def get_server_url_and_headers_from_config(config_filename: str):
    """Extracts all servers' URLs and headers from the config. Supports both direct URL and host/port config styles. If no headers for a URL server, tries to find a token in local.env."""
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    config_path = os.path.join(config_dir, config_filename)
    env_path = os.path.join(os.path.dirname(__file__), "../local.env")
    env_vars = load_env_vars(env_path)

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

    servers = []
    for server_config in mcp_pool:
        name = server_config.get("name", "unknown")
        if "url" in server_config:
            headers = server_config.get("headers", None)
            if not headers:
                token, token_key = find_token_for_server(name, env_vars)
                if token:
                    if token_key.endswith('_API_KEY'):
                        headers = {"X-API-Key": token}
                    elif token_key.endswith('_PERSONAL_ACCESS_TOKEN') or token_key.endswith('_API_TOKEN') or token_key.endswith('_TOKEN'):
                        headers = {"Authorization": f"Bearer {token}"}
            url = server_config["url"]
            servers.append({"name": name, "url": url, "headers": headers})
        else:
            run_configs = server_config.get("run_config", [])
            if not run_configs:
                print(f"No 'run_config' found for server '{name}'. Skipping.")
                continue
            run_config = run_configs[0]
            port = run_config.get("port")
            if not port:
                print(f"No 'port' found in run_config for server '{name}'. Skipping.")
                continue
            host = run_config.get("host", "localhost")
            url = f"http://{host}:{port}/sse"
            servers.append({"name": name, "url": url, "headers": None})
    return servers

async def list_mcp_tools_from_url(server_url: str, headers=None):
    """Connects to an already running MCP server via SSE and lists its tools. Returns the list of tools."""
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
                return tools
    except Exception as e:
        print(f"\nAn error occurred while connecting to '{server_url}': {e}")
        print("Please ensure the server is running and the URL is correct.")
        return []

async def main():
    if len(sys.argv) < 2:
        print("Usage: python3 list_mcp_tools_by_hitting_remote_or_localhost.py <config_file_name>")
        print("\nExample:")
        print("  python3 list_mcp_tools_by_hitting_remote_or_localhost.py slack.json")
        sys.exit(1)
    config_filename = sys.argv[1]
    servers = get_server_url_and_headers_from_config(config_filename)
    total_tools = 0
    server_tool_counts = []
    for idx, server in enumerate(servers):
        print("\n" + "="*60)
        print(f"Server {idx+1}: {server['name']} ({server['url']})")
        print("="*60)
        tools = await list_mcp_tools_from_url(server['url'], headers=server['headers'])
        count = len(tools) if tools else 0
        print(f"\nTool count for server '{server['name']}': {count}")
        total_tools += count
        server_tool_counts.append((server['name'], count))
    print("\n" + "#"*60)
    print(f"Total number of tools across all servers: {total_tools}")
    print("#"*60)
    for name, count in server_tool_counts:
        print(f"  {name}: {count} tool(s)")

if __name__ == "__main__":
    asyncio.run(main())

# To run it:
# export $(cat local.env | xargs) && source .venv/bin/activate && python3 list_mcp_tools_by_hitting_remote_or_localhost.py slack.json
