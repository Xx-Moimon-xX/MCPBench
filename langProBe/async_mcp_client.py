from contextlib import AsyncExitStack
from typing import Optional

from anthropic import Anthropic
from mcp import ClientSession
from mcp.client.sse import sse_client


class AsyncMCPClient:

    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

    async def connect_to_sse_server(self, server_url: str, headers=None):
        """Connect to an MCP server running with SSE transport"""
        print(f"[DEBUG SSE] Attempting to connect to SSE server: {server_url}")
        # Store the context managers so they stay alive
        try:
            self._streams_context = sse_client(url=server_url, headers=headers, timeout=10)
            print(f"[DEBUG SSE] Created SSE client context for {server_url}")
            streams = await self._streams_context.__aenter__()
            print(f"[DEBUG SSE] Entered streams context for {server_url}")

            self._session_context = ClientSession(*streams)
            self.session: ClientSession = await self._session_context.__aenter__()
            print(f"[DEBUG SSE] Created and entered session context for {server_url}")

            # Initialize
            await self.session.initialize()
            print(f"[DEBUG SSE] Initialized session for {server_url}")

            # List available tools to verify connection
            # print("Initialized SSE client...")
            # print("Listing tools...")
            response = await self.session.list_tools()
            tools = response.tools
            print(f"[DEBUG SSE] Successfully connected to {server_url} with {len(tools)} tools")
            # print("\nConnected to server with tools:", [tool.name for tool in tools])
        except Exception as e:
            print(f"[DEBUG SSE ERROR] Failed to connect to {server_url}: {str(e)}")
            raise

    async def cleanup(self):
        """Properly clean up the session and streams"""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)

    async def call_tool(self, tool_name: str, tool_args: dict) -> dict:
        """Call a tool with the given arguments"""
        print(f"[DEBUG ASYNC] About to call tool: {tool_name} with args: {tool_args}")
        try:
            result = await self.session.call_tool(tool_name, tool_args)
            print(f"[DEBUG ASYNC] Successfully called tool: {tool_name}")
            return result
        except Exception as e:
            print(f"[DEBUG ASYNC ERROR] Failed to call tool {tool_name}: {str(e)}")
            raise

    async def list_tools(self):
        """List available tools"""
        print(f"[DEBUG ASYNC] About to list tools")
        try:
            response = await self.session.list_tools()
            print(f"[DEBUG ASYNC] Successfully listed {len(response.tools)} tools")
            return response
        except Exception as e:
            print(f"[DEBUG ASYNC ERROR] Failed to list tools: {str(e)}")
            raise

    async def get_prompt(self, *args, **kwargs):
        response = await self.session.get_prompt(*args, **kwargs)
        return response

    async def list_prompts(self):
        response = await self.session.list_prompts()
        return response

    async def list_resources(self):
        response = await self.session.list_resources()
        return response

    async def read_resource(self, *args, **kwargs):
        response = await self.session.read_resource(*args, **kwargs)
        return response

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # Initial Claude API call
        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        # Process response and handle tool calls
        tool_results = []
        final_text = []

        for content in response.content:
            if content.type == 'text':
                final_text.append(content.text)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input

                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                tool_results.append({"call": tool_name, "result": result})
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                # Continue conversation with tool results
                if hasattr(content, 'text') and content.text:
                    messages.append({
                        "role": "assistant",
                        "content": content.text
                    })
                messages.append({
                    "role": "user",
                    "content": result.content
                })

                # Get next response from Claude
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=messages,
                )

                final_text.append(response.content[0].text)

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        # print("\nMCP Client Started!")
        # print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

# async def main():
#     client = AsyncMCPClient()
#     try:
#         await client.connect_to_sse_server(server_url="http://localhost:8080/sse")
#         result = await client.call_tool("get_alerts", {"state": "CA"})
#         print(result)
#     finally:
#         await client.cleanup()


# result = asyncio.run(main())