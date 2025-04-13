from synced_mcp_client import SyncedMcpClient
def main():
    client = SyncedMcpClient(server_url="http://localhost:8000/sse")

    try:
        result = client.list_tools()
        print(result)
    except Exception as e:
        print(f"Error listing tools: {e}")

    try:
        result = client.call_tool("brave_web_search", {"query": "今天的日期"})
        print("+"*50)
        print(result)
    except Exception as e:
        print(f"Error calling tool: {e}")


if __name__ == '__main__':
    main()