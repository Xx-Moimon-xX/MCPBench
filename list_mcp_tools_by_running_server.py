import asyncio
import sys
import json
import os
from contextlib import AsyncExitStack

# This script requires the 'mcp' library.
# You can install it in a virtual environment using:
# uv init
# uv venv
# uv add mcp
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    print("Error: The 'mcp' library is not installed.")
    print("Please set up a virtual environment and install it with 'uv add mcp'.")
    sys.exit(1)

async def list_mcp_tools_from_command(command: str, args: list, server_name: str):
    """Connects to an MCP server using command and args, and lists its tools."""
    print(f"\nAttempting to start server '{server_name}' with command: {' '.join([command] + args)}")
    
    server_params = StdioServerParameters(
        command=command,
        args=args,
        env=os.environ.copy()
    )
    
    async with AsyncExitStack() as exit_stack:
        try:
            stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
            stdio, write = stdio_transport
            session = await exit_stack.enter_async_context(ClientSession(stdio, write))
            
            await session.initialize()
            
            response = await session.list_tools()
            tools = response.tools
            
            print(f"\nSuccessfully connected to server '{server_name}'.")
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
            print(f"\nAn error occurred while connecting to '{server_name}': {e}")
            print("Please ensure the server command and arguments are correct.")

async def list_tools_from_config(config_path: str):
    """Loads a config file and lists tools for all servers defined in it."""
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
        return

    for server_config in mcp_pool:
        server_name = server_config.get("name")
        if not server_name:
            print("Found a server in config without a name. Skipping.")
            continue
        
        print(f"\n--- Processing server: {server_name} ---")

        run_configs = server_config.get("run_config", [])
        if not run_configs:
            print(f"No 'run_config' found for server '{server_name}'. Skipping.")
            continue

        # Taking the first run_config, as is common in the project
        run_config = run_configs[0]
        command = run_config.get("command")
        args = run_config.get("args")

        if not command or not args:
            print(f"Incomplete 'run_config' for server '{server_name}'. It must contain 'command' and 'args'. Skipping.")
            continue

        await list_mcp_tools_from_command(command, args, server_name)

async def main():
    if len(sys.argv) < 2:
        print("Usage: python3 list_mcp_tools.py <path_to_config_file>")
        print("\nExample:")
        print("  python3 list_mcp_tools.py external/MCPBench/configs/slack.json")
        sys.exit(1)
        
    config_path = sys.argv[1]
    await list_tools_from_config(config_path)

if __name__ == "__main__":
    asyncio.run(main())


# To run it: 
# export $(cat local.env | xargs) && source .venv/bin/activate && python3 list_mcp_tools_by_running_servers.py configs/slack.json