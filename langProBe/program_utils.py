from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
from typing import List, Tuple, Optional, Dict, Union
from openai import OpenAI
import json
import copy
from pydantic import BaseModel, Field
import re
import os
import sys
import langProBe.constants as constants
import logging
from .synced_mcp_client import SyncedMcpClient
import json

try:
    from anthropic import Anthropic
    from anthropic import BadRequestError
except ImportError:
    Anthropic = None

try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
except ImportError:
    boto3 = None

TOOL_PROMPT = """
## Tool Calling Rules
When external tools are required, the call request must be strictly generated according to the following rules:
<tool>  
{  
  "server_name": "",  
  "tool_name": "",  
  "inputs": {  
    "<parameter1>": "<value1>",  
    "<parameter2>": "<value2>",  
  }  
}  
</tool>  

If no tool is called, provide the final answer directly.

"""
            
class ProcessManager(BaseModel):
    '''
    This class is user to manage the orchestration between the LLM, MCP server and the user/system inputs.
    '''
    id: Optional[str] = Field(
        default=None,
        description="The ID of the process.",
    )
    lm_api_key: Optional[str] = Field(
        default=os.getenv("OPENAI_API_KEY"),
        description="OpenAI API Key"
    )
    lm_api_base: Optional[str] = Field(
        default=os.getenv("OPENAI_API_BASE"),
        description="OpenAI API Base URL"
    )
    model: Optional[str] = Field(
        default=None,
        description="OpenAI Model Name, with prefix 'openai/'"
    )
    lm_usages: List[Dict] = Field(
        default=[],
        description="Usage statistics for the model"
    )
    mcp_rts: List[Dict] = Field(
        default=[],
        description="Usage statistics for the MCPs"
    )
    mcp_retry_times: List[Dict] = Field(
        default=[],
        description="Statistics for the MCP retries"
    )
    anthropic_api_key: Optional[str] = Field(
        default=os.getenv("ANTHROPIC_API_KEY"),
        description="Anthropic API Key"
    )
    aws_access_key_id: Optional[str] = Field(
        default=os.getenv("AWS_ACCESS_KEY_ID"),
        description="AWS Access Key ID for Bedrock"
    )
    aws_secret_access_key: Optional[str] = Field(
        default=os.getenv("AWS_SECRET_ACCESS_KEY"),
        description="AWS Secret Access Key for Bedrock"
    )
    aws_region: Optional[str] = Field(
        default=os.getenv("AWS_REGION", "us-east-1"),
        description="AWS Region for Bedrock"
    )


class MCPCall(BaseModel):
    '''
    This class is used to store the MCP tool call information.
    '''
    mcp_server_name: Optional[str] = None
    mcp_tool_name: Optional[str] = None
    mcp_args: Optional[Dict] = None


class MCPCallList(BaseModel):
    '''
    This class is used to store all the MCP tool calls to be made.
    '''
    shutdown: bool = False
    mcps: Optional[List[MCPCall]] = None
    raw_content: Optional[str] = None

@retry(
    stop=stop_after_attempt(3),  
    wait=wait_exponential(multiplier=1, min=2, max=10),  
    reraise=True,
)
def call_lm(
            messages: List, 
            manager: ProcessManager, 
            logger: logging.Logger, 
            temperature: float|None=None,
            system_prompt: str = None,
            ) -> tuple[str | None, int, int]:    
    '''
    This function is used to call the LLM API, it can be used for Anthropic, AWS Bedrock and OpenAI.
    '''
    # Log the input messages being sent to the LLM
    # logger.debug(f"ID: {manager.id}, Input messages to LLM: {json.dumps(messages, indent=2, ensure_ascii=False)}")
    response = None
    try:
        # Getting the correct model to use for the LLM call.
        prefix, model_name = manager.model.split('/')
        # print(f"Model: {manager.model}")
        if prefix == 'anthropic':
            if Anthropic is None:
                raise ImportError("The 'anthropic' package is required for Claude API support. Please install it via 'pip install anthropic'.")
            # Anthropic Claude API
            anthropic_api_key = manager.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            client = Anthropic(api_key=anthropic_api_key)
            
            # Convert OpenAI-style messages to Anthropic format
            claude_messages = []
            for m in messages:
                if m.get("role") == "user":
                    claude_messages.append({"role": "user", "content": m["content"]})
                elif m.get("role") == "assistant":
                    claude_messages.append({"role": "assistant", "content": m["content"]})
                elif m.get("role") == "tool":
                    claude_messages.append({"role": "user", "content": m["content"]})
            # Call Claude API

            max_tokens_limit = 200000
            truncated = False
            # Find the index of the user message (should only be one)
            user_idx = next((i for i, m in enumerate(claude_messages) if m["role"] == "user"), None)
            while True:
                try:
                    if system_prompt:
                        messages_tokens = client.messages.count_tokens(
                            model=model_name,
                            messages=claude_messages,
                            system=system_prompt
                        )
                        print(f"Total tokens being passed to the model with system prompt: {messages_tokens}")
                        completion = client.messages.create(
                            model=model_name,
                            max_tokens=1024,
                            messages=claude_messages,
                            temperature=temperature if temperature is not None else 0.7,
                            system=system_prompt
                        )
                    else:
                        messages_tokens = client.messages.count_tokens(
                            model=model_name,
                            messages=claude_messages
                        )
                        print(f"Total tokens being passed to the model: {messages_tokens}")
                        completion = client.messages.create(
                            model=model_name,
                            max_tokens=1024,
                            messages=claude_messages,
                            temperature=temperature if temperature is not None else 0.7,
                        )
                    break  # Success, exit loop
                except Exception as e:
                    err_msg = str(e)
                    if "prompt is too long" in err_msg:
                        # Always keep the user message and the last message
                        if len(claude_messages) > 2 and user_idx is not None:
                            # Remove the oldest message that is not the user message or the last message
                            removable_indices = [i for i in range(len(claude_messages)) if i != user_idx and i != len(claude_messages)-1]
                            if removable_indices:
                                remove_idx = removable_indices[0]
                                removed = claude_messages.pop(remove_idx)
                                # If we removed a message before the user, update user_idx
                                if remove_idx < user_idx:
                                    user_idx -= 1
                                logger.warning(f"ID: {manager.id} (in call_lm), Truncating message at index {remove_idx} to fit token limit. Messages length: {len(claude_messages)}")
                                print(f"ID: {manager.id} (in call_lm), Truncated message at index {remove_idx} to fit token limit. Messages length: {len(claude_messages)}")
                            else:
                                logger.error(f"ID: {manager.id} (in call_lm), Unable to truncate further. Only user and last message left. Still over token limit. Token count: {messages_tokens if 'messages_tokens' in locals() else 'unknown'} \n Messages length: {len(claude_messages)}")
                                logger.warning(f"Remaining messages: {claude_messages}")
                                print(f"ID: {manager.id} (in call_lm), Unable to truncate further. Only user and last message left. Still over token limit. Token count: {messages_tokens if 'messages_tokens' in locals() else 'unknown'} \n Messages length: {len(claude_messages)}")
                                print(f"claude_messages[-1]: {claude_messages[-1]}\n")
                                print(f"Remaining messages: {claude_messages}")
                                raise RuntimeError(f"Anthropic prompt is too long (>{max_tokens_limit} tokens) and cannot be truncated further. Please shorten your conversation or system prompt.") from e
                        else:
                            logger.error(f"ID: {manager.id} (in call_lm), Unable to truncate further. Only user and last message left. Still over token limit. Token count: {messages_tokens if 'messages_tokens' in locals() else 'unknown'} \n Messages length: {len(claude_messages)}")
                            logger.warning(f"Remaining messages: {claude_messages}")
                            print(f"ID: {manager.id} (in call_lm), Unable to truncate further. Only user and last message left. Still over token limit. Token count: {messages_tokens if 'messages_tokens' in locals() else 'unknown'} \n Messages length: {len(claude_messages)}")
                            print(f"claude_messages[-1]: {claude_messages[-1]}")
                            print(f"Remaining messages: {claude_messages}")
                            raise RuntimeError(f"Anthropic prompt is too long (>{max_tokens_limit} tokens) and cannot be truncated further. Please shorten your conversation or system prompt.") from e
                    else:
                        raise e
            # --- End token truncation logic ---
            
            # Log the full response for debugging
            logger.debug(f"ID: {manager.id} (in call_lm), Full Anthropic API response: {completion}")
            
            response_text = completion.content[0].text if completion.content else ""
            
            # Log extracted content
            logger.debug(f"ID: {manager.id} (in call_lm), Extracted response_text: '{response_text}'")
            # Anthropic does not return token usage in the same way
            if hasattr(completion, 'usage') and completion.usage is not None:
                completion_tokens = getattr(completion.usage, 'output_tokens', 0)
                prompt_tokens = getattr(completion.usage, 'input_tokens', 0)
            else:
                completion_tokens = 0
                prompt_tokens = 0
            manager.lm_usages.append({
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
            })
            return response_text, completion_tokens, prompt_tokens
        # AWS Bedrock API call (using Converse API for better consistency)
        elif prefix == 'bedrock':
            if boto3 is None:
                raise ImportError("The 'boto3' package is required for AWS Bedrock support. Please install it via 'pip install boto3'.")
            # AWS Bedrock API
            bedrock_client = boto3.client(
                'bedrock-runtime',
                aws_access_key_id=manager.aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=manager.aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY"),
                aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
                region_name=manager.aws_region or os.getenv("AWS_REGION", "ap-southeast-")
            )
            
            # Convert messages to format expected by Bedrock Converse API
            bedrock_messages = []
            system_message = ""
            
            # Debug: Print all messages before conversion
            # print(f"[DEBUG] call_lm: manager.id={manager.id} messages before Bedrock conversion:")
            # for i, m in enumerate(messages):
            #     print(f"  Message {i}: role={m.get('role')} content={repr(m.get('content'))}")
            #     if not m.get('content'):
            #         print(f"  [WARNING] Message {i} has empty or missing content!")
            for m in messages:
                if m.get("role") == "system":
                    system_message = m["content"]
                elif m.get("role") == "user":
                    bedrock_messages.append({"role": "user", "content": [{"text": m["content"]}]})
                elif m.get("role") == "assistant":
                    bedrock_messages.append({"role": "assistant", "content": [{"text": m["content"]}]})
                elif m.get("role") == "tool":
                    # Lol so we convert the tool call response to a user message when sending it to the LLM.
                    bedrock_messages.append({"role": "user", "content": [{"text": m["content"]}]})
            
            try:
                # Prepare request for Bedrock Converse API
                request_params = {
                    "modelId": model_name,
                    "messages": bedrock_messages,
                    "inferenceConfig": {
                        "maxTokens": 512,
                        "temperature": temperature if temperature is not None else 0.7
                    }
                }
                
                # If system message is provided, add it to the request params
                if system_message:
                    request_params["system"] = [{"text": system_message}]
                
                # Calling the AWS Bedrock Converse API
                response = bedrock_client.converse(**request_params)
                
                # Log the full response for debugging
                logger.debug(f"ID: {manager.id} (in call_lm), Full Bedrock API response: {json.dumps(response, indent=2, default=str)}")
                # print(f"Model used: {model_name}")
                # Extract response content
                output_message = response.get('output', {}).get('message', {})
                content_list = output_message.get('content', [])
                
                # This just cleans up the new lines and stuff.
                response_text = ""
                if content_list:
                    for content in content_list:
                        if 'text' in content:
                            response_text += content['text']
                
                # Log extracted content
                logger.debug(f"ID: {manager.id} (in call_lm), Extracted response_text: '{response_text}'")
                
                # Extract token usage
                usage = response.get('usage', {})
                completion_tokens = usage.get('outputTokens', 0)
                prompt_tokens = usage.get('inputTokens', 0)
                
                manager.lm_usages.append({
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                })
                return response_text, completion_tokens, prompt_tokens
                
            except (ClientError, BotoCoreError) as e:
                logger.error(f"ID: {manager.id} (in call_lm), AWS Bedrock error: {str(e)}")
                logger.error("Exiting program due to AWS Bedrock error.")
                raise
                # sys.exit(1)
        # OpenAI API call
        else:
            # --- OpenAI logic as before ---
            # Creating the OpenAI client
            print(f"ID: {manager.id} (in call_lm), Calling OpenAI API with model: {model_name}")

            openai_api_key = os.getenv("OPENAI_API_KEY")
            oai = OpenAI(
                api_key=openai_api_key
            )
            assert prefix == 'openai'

            # Convert tool role messages to user role to avoid OpenAI tool-calls schema requirements
            # (OpenAI requires tool messages only in response to assistant.tool_calls with ids.)
            oai_messages = []
            for m in messages:
                role = m.get("role")
                if role == "system":
                    oai_messages.append({"role": "system", "content": m["content"]})
                elif role == "assistant":
                    oai_messages.append({"role": "assistant", "content": m["content"]})
                elif role == "user":
                    oai_messages.append({"role": "user", "content": m["content"]})
                elif role == "tool":
                    oai_messages.append({"role": "user", "content": m["content"]})
                else:
                    # Drop empty tool_calls fields if present in assistant messages
                    m_copy = {k: v for k, v in m.items() if not (k == "tool_calls" and isinstance(v, list) and len(v) == 0)}
                    oai_messages.append(m_copy)

            if model_name in ['deepseek-r1', 'qwq-plus', 'qwq-32b']: # qwen reasoning models only support streaming output
                reasoning_content = ""  # Define complete reasoning process
                answer_content = ""     # Define complete response
                is_answering = False   # Determine if reasoning process is complete and response has started

                completion = oai.chat.completions.create(
                    model=model_name, 
                    messages=oai_messages,
                    stream=True,
                    stream_options={
                        "include_usage": True
                    }
                )
                for chunk in completion:
                    # If chunk.choices is empty, print usage
                    if not chunk.choices:
                        usage = chunk.usage
                    else:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                            reasoning_content += delta.reasoning_content
                        else:
                            # Start response
                            if delta.content != "" and is_answering is False:
                                is_answering = True
                            answer_content += delta.content
                completion_tokens = usage.completion_tokens
                prompt_tokens = usage.prompt_tokens
                manager.lm_usages.append({
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                })
                return '<think>' + reasoning_content + '</think>' + answer_content, completion_tokens, prompt_tokens


            if temperature is not None:
                response = oai.beta.chat.completions.parse(
                    messages=oai_messages,
                    model=model_name,
                    temperature = temperature
                )
            else:
                response = oai.beta.chat.completions.parse(
                    messages=oai_messages,
                    model=model_name,
                )
                # Log the full response for debugging
                logger.debug(f"ID: {manager.id} (in call_lm), Full OpenAI API response: {response}")
                
                response_text = response.choices[0].message.content or ""
                
                # Log extracted content
                logger.debug(f"ID: {manager.id} (in call_lm), Extracted response_text: '{response_text}'")
                
                completion_tokens = response.usage.completion_tokens
                prompt_tokens = response.usage.prompt_tokens
            
            manager.lm_usages.append({
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                })
            return response_text, completion_tokens, prompt_tokens
    
    except Exception as e:
        logger.error(f"ID: {manager.id} (in call_lm), Error in call_lm: {str(e)}")
        if response:
            logger.error(f"ID: {manager.id} (in call_lm), Response: {response}")
        raise

def build_system_content_filler_context(base_system: str) -> str:
    """
    Return the pre-built system content with filler context from the 24k tokens file.
    """
    import os
    from pathlib import Path
    
    # Path to the filler context file
    current_dir = Path(__file__).parent
    filler_file_path = current_dir.parent / "context filler system prompts" / "after slack tools 24k tokens.txt"
    
    try:
        with open(filler_file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: Filler context file not found at {filler_file_path}")
        return base_system
    except Exception as e:
        print(f"Error reading filler context file: {e}")
        return base_system


def build_system_content(base_system: str,
                        mcps: List,
                        tools_format: str,
                        ) -> str:
    '''
    Build the system content for the conversation, i.e. the system prompt and the available tools.
    '''
    tools_section = "## Available Tools\n"
    all_servers = []  # Move this outside the loop!
    
    for mcp in mcps:
        if mcp['name'] in ['wuying-agentbay-mcp-server', 'Playwright']:
            tools_section += f"When using this server to perform search tasks, please use https://www.baidu.com as the initial website for searching."
        
        # Connecting the the MCP server to get the tools!!!!
        url = mcp.get("url")
        print(f"MCP: {mcp}")
        print(f"[DEBUG BUILD_SYSTEM] Building system content for server: {mcp['name']}")
        # print(f"URL: {url}")

        headers = None
        if url:
            headers = mcp.get("headers")
            if not headers:
                server_name = mcp.get("name", "")
                norm_name = re.sub(r'[-_]', '', server_name).lower()
                found_token = None
                found_key = None
                for key, value in os.environ.items():
                    norm_key = re.sub(r'[-_]', '', key).lower()
                    if norm_key.startswith(norm_name) and (
                        key.endswith('_TOKEN') or key.endswith('_API_KEY') or key.endswith('_API_TOKEN') or key.endswith('_PERSONAL_ACCESS_TOKEN')
                    ):
                        found_token = value
                        found_key = key
                        break
                if found_token and found_key:
                    if found_key.endswith('_API_KEY'):
                        headers = {"X-API-Key": found_token}
                    elif found_key.endswith('_PERSONAL_ACCESS_TOKEN') or found_key.endswith('_API_TOKEN') or found_key.endswith('_TOKEN'):
                        headers = {"Authorization": f"Bearer {found_token}"}
        else:
            try:
                port = mcp.get('run_config')[0]["port"]
                url = f"http://localhost:{port}/sse"
            except:
                raise Exception("No url found")
        print(f"[DEBUG BUILD_SYSTEM] Creating client for {mcp['name']} at URL: {url}")

        client = SyncedMcpClient(server_url=url, headers=headers)
        try:
            print(f"[DEBUG BUILD_SYSTEM] About to list tools for {mcp['name']}")
            result = client.list_tools()
            tools = result.tools
            print(f"[DEBUG BUILD_SYSTEM] Successfully got {len(tools)} tools from {mcp['name']}")
            # print(f"Tools directly from MCP server: {tools}")
            # logger.debug(f"Tools directly from MCP server: {tools}")
        except Exception as e:
            print(f"[DEBUG BUILD_SYSTEM ERROR] Failed to access server {mcp['name']}: {e}")
            raise Exception(f"Fail access to server: {mcp['name']}, error: {e}")

        # Formatting tools section to send in prompt
        if tools_format == "raw_mcp":
            # Format each tool on a separate line for better readability, keeping exact Tool( format
            tools_section += "Tools raw: \n"
            for i, tool in enumerate(tools):
                if i == 0:
                    tools_section += f"[{repr(tool)},\n"
                elif i == len(tools) - 1:
                    tools_section += f"{repr(tool)}]\n\n"
                else:
                    tools_section += f"{repr(tool)},\n"
        elif tools_format == "json":
            # Format tools as JSON structure
            server_tools = []
            for tool in tools:
                # Convert ToolAnnotations to serializable format
                annotations = tool.annotations
                if annotations is not None:
                    if hasattr(annotations, 'model_dump'):
                        # Pydantic v2
                        annotations = annotations.model_dump()
                    elif hasattr(annotations, 'dict'):
                        # Pydantic v1
                        annotations = annotations.dict()
                    elif hasattr(annotations, '__dict__'):
                        # Regular object
                        annotations = annotations.__dict__
                    else:
                        # Fallback to string
                        annotations = str(annotations)
                
                # Convert meta to serializable format
                meta = tool.meta
                if meta is not None:
                    if hasattr(meta, 'model_dump'):
                        meta = meta.model_dump()
                    elif hasattr(meta, 'dict'):
                        meta = meta.dict()
                    elif hasattr(meta, '__dict__'):
                        meta = meta.__dict__
                    else:
                        meta = str(meta)
                
                tool_dict = {
                    "name": tool.name,
                    "title": tool.title,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema,
                    "outputSchema": tool.outputSchema,
                    "annotations": annotations,
                    "meta": meta
                }
                server_tools.append(tool_dict)
            
            # Store server data for later JSON construction
            # if not hasattr(build_system_content, 'all_servers'):
            #     build_system_content.all_servers = []
            
            server_data = {
                "server": mcp['name'],
                "tools": server_tools
            }
            all_servers.append(server_data)
        else:
            # Default to formatted or handle formatted explicitly
            tools_section += f"### Server '{mcp['name']}' include following tools\n"
            for t in tools:
                tools_section += f"- {t.name}: {t.description}\n"
                input_schema = t.inputSchema
                required_params = input_schema.get("required", [])
                params_desc = []

                if "properties" in input_schema:
                    for param_name, param_info in input_schema["properties"].items():
                        is_required = param_name in required_params
                        param_type = param_info.get("type", "")
                        param_desc = param_info.get("description", "")

                        req_tag = "required" if is_required else "optional"
                        params_desc.append(
                            f"- {param_name} ({param_type}, {req_tag}): {param_desc}"
                        )

                # 使用更丰富的描述
                # Use a more detailed description
                params_text = "\n".join(params_desc) if params_desc else "No parameters"
                tools_section += f"  Parameters:\n{params_text}\n\n"

    # If using JSON format, construct the combined JSON structure
    if tools_format == "json" and all_servers:
        try:
            combined_json = {
                "servers": all_servers
            }
            tools_section += f"\n{json.dumps(combined_json, indent=2)}\n\n"
        except TypeError as e:
            print(f"[WARNING] JSON serialization failed: {e}")
            print("[WARNING] Falling back to text format for tools")
            # Fall back to adding tools in text format
            for server_data in all_servers:
                tools_section += f"\n### Server: {server_data['server']}\n"
                for tool in server_data['tools']:
                    tools_section += f"- {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')}\n"
        # Clear the stored servers for next use
        # build_system_content.all_servers = []

    prompt = base_system + f"""{tools_section}""" + TOOL_PROMPT

    return prompt


def build_init_messages(
        system_content: str,
        user_question: str,
       ) -> List[Dict]:
    '''
    Build the initial messages for the conversation, i.e. the system prompt and the user question.
    '''
    # system_content = build_system_content(base_system, mcps)
    messages = [
        {
            constants.ROLE: constants.SYSTEM,
            constants.CONTENT: system_content
        },
        {
            constants.ROLE: constants.USER,
            constants.CONTENT: user_question
        }
    ]
    return messages



def build_messages(
        messages: List[Dict],
        message_to_append: List[Dict],
        ) -> List[Dict]:
    '''
    Constructs a new list of messages for the next prediction round in a conversational AI context.
    Ensures the conversation starts with a system message and appends new messages according to strict role-based rules:
    - If appending a user message: must be a single message, and the previous message must be from assistant, tool, or system.
    - If appending an assistant message: must be a single message, and the previous message must be from user or tool.
    - If appending a tool message: must be two messages, and the previous message must be from user or tool.
    This enforces a valid, structured alternation of roles in the conversation history.
    '''
    assert messages[0][constants.ROLE] == constants.SYSTEM
    
    ## i.e. the previous message that we're concatenating to.
    final_message = copy.deepcopy(messages)

    if message_to_append:
        if message_to_append[-1][constants.ROLE] == constants.USER:
            assert len(message_to_append) == 1
            assert final_message[-1][constants.ROLE] in {constants.ASSISTANT, constants.TOOL, constants.SYSTEM}
            final_message.extend(message_to_append)
        elif message_to_append[-1][constants.ROLE] == constants.ASSISTANT:
            assert len(message_to_append) == 1
            assert final_message[-1][constants.ROLE] in {constants.USER, constants.TOOL}
            final_message.extend(message_to_append)
        elif message_to_append[-1][constants.ROLE] == constants.TOOL:
            assert len(message_to_append) == 2
            assert final_message[-1][constants.ROLE] in {constants.USER, constants.TOOL}
            final_message.extend(message_to_append)
    
    # TODO: Handle exceeding maximum context length

    return final_message



def response_parsing(content: str | None) -> MCPCallList:
    '''
    Parse the response content to get the MCP call list and returns this.
    The response content is a string that contains the MCP call list.
    The MCP call list is a list of MCP calls.
    Each MCP call is a dictionary that contains the MCP server name, tool name, and arguments.
    '''
    # Handle None content
    if content is None:
        content = ""
    
    pattern = r'<tool>(.*?)<\/tool>'
    matches = re.findall(pattern, content, re.DOTALL)
    mcps = []
    for match in matches:
        # TODO: Error handling
        data = json.loads(match)
        mcps.append(MCPCall(
            mcp_server_name=data['server_name'].strip(),
            mcp_tool_name=data['tool_name'].strip(),
            mcp_args=data['inputs']
        ))

    # If there are no tool calls, we set the shutdown flag to True
    if mcps:
        return MCPCallList(shutdown=False, mcps=mcps, raw_content=content)
    else:
        return MCPCallList(shutdown=True, mcps=None, raw_content=content)


def mcp_calling(
        mcp_call_list: MCPCallList,
        manager: ProcessManager,
        logger: logging.Logger,
        config: dict,
        client_cache: dict = None,  # New parameter for client reuse
) -> List[Dict]:
    '''
    Processes each tool call in the MCP call list, reusing SyncedMcpClient per server and ensuring cleanup.
    If client_cache is not provided, a new one is created and all clients are cleaned up at the end.
    '''
    logger.debug(f"ID: {manager.id} (in mcp_calling), Entering mcp_calling with mcp_call_list: {mcp_call_list}")

    created_cache = False
    if client_cache is None:
        client_cache = {}
        created_cache = True

    if mcp_call_list.shutdown:
        logger.info(f"ID: {manager.id} (in mcp_calling), Shutdown flag is set. No more MCP calling.")
        messages = [
            {
                constants.ROLE: constants.ASSISTANT,
                constants.CONTENT: mcp_call_list.raw_content if mcp_call_list.raw_content else '',
            }
        ]
        logger.debug(f"ID: {manager.id} (in mcp_calling), Shutdown messages prepared: {messages}")
        # Clean up if we created the cache
        if created_cache:
            cleanup_all_clients(client_cache)
        return messages
    else:
        logger.info(f"ID: {manager.id} (in mcp_calling), Processing MCP call list with {len(mcp_call_list.mcps)} MCPs. mcp_call_list: {mcp_call_list}")
        mcp_list = mcp_call_list.mcps
        messages = [
            {
                constants.ROLE: constants.ASSISTANT,
                constants.CONTENT: mcp_call_list.raw_content if mcp_call_list.raw_content else ''
            }
        ]
        result_str = ""

        # Iterating over each MCP call in the MCP call list
        for idx, mcp in enumerate(mcp_list, start=1):
            logger.debug(f"ID: {manager.id} (in mcp_calling), Processing MCP #{idx}: {mcp}")
            print(f"[DEBUG MCP] ID: {manager.id}, Processing MCP #{idx}/{len(mcp_list)}: server={mcp.mcp_server_name}, tool={mcp.mcp_tool_name}")
            mcp_server_name = mcp.mcp_server_name
            mcp_tool_name = mcp.mcp_tool_name
            mcp_args = mcp.mcp_args

            try:
                # Use passed config parameter, fallback to global_config if needed
                logger.debug(f"ID: {manager.id} (in mcp_calling), Received config parameter: {config}")
                parsed_data = config
                if parsed_data is None:
                    from langProBe.evaluation import global_config
                    logger.debug(f"ID: {manager.id} (in mcp_calling), Fallback to global_config: {global_config}")
                    parsed_data = global_config
                
                # Handle case where config is None
                if parsed_data is None:
                    logger.error(f"ID: {manager.id} (in mcp_calling), config is None, cannot initialize MCP client")
                    logger.warning(f"ID: {manager.id} (in mcp_calling), Skipping tool call for '{mcp_tool_name}' due to missing configuration.")
                    continue

                # Additional safety check
                if not isinstance(parsed_data, dict):
                    logger.error(f"ID: {manager.id} (in mcp_calling), config is not a dict: {type(parsed_data)}")
                    logger.warning(f"ID: {manager.id} (in mcp_calling), Skipping tool call for '{mcp_tool_name}' due to invalid configuration.")
                    continue

                target_name = mcp_server_name
                port = None
                url = None
                logger.debug(f"ID: {manager.id} (in mcp_calling), Parsed config keys: {list(parsed_data.keys())}")
                mcp_pool = parsed_data.get("mcp_pool", [])
                logger.debug(f"ID: {manager.id} (in mcp_calling), MCP pool: {mcp_pool}")
                if not mcp_pool:
                    logger.error(f"ID: {manager.id} (in mcp_calling), No MCP pool found in configuration")
                    logger.warning(f"ID: {manager.id} (in mcp_calling), Skipping tool call for '{mcp_tool_name}' due to missing MCP pool.")
                    continue

                for item in mcp_pool:
                    if item.get("name") != target_name:
                        continue

                    url = item.get("url", "")
                    headers = item.get("headers", None)
                    # If this is a remote server and headers are not present, try to get from env
                    if url:
                        if not headers:
                            server_name = mcp.get("name", "")
                            norm_name = re.sub(r'[-_]', '', server_name).lower()
                            found_token = None
                            found_key = None
                            for key, value in os.environ.items():
                                norm_key = re.sub(r'[-_]', '', key).lower()
                                if norm_key.startswith(norm_name) and (
                                    key.endswith('_TOKEN') or key.endswith('_API_KEY') or key.endswith('_API_TOKEN') or key.endswith('_PERSONAL_ACCESS_TOKEN')
                                ):
                                    found_token = value
                                    found_key = key
                                    break
                            if found_token and found_key:
                                if found_key.endswith('_API_KEY'):
                                    headers = {"X-API-Key": found_token}
                                elif found_key.endswith('_PERSONAL_ACCESS_TOKEN') or found_key.endswith('_API_TOKEN') or found_key.endswith('_TOKEN'):
                                    headers = {"Authorization": f"Bearer {found_token}"}
                    else:
                        try:
                            port = item.get('run_config')[0]["port"]
                            url = f"http://localhost:{port}/sse"
                        except:
                            raise Exception("No url found")

                    # Use (url, str(headers)) as cache key for uniqueness
                    cache_key = (url, str(headers))
                    if cache_key in client_cache:
                        client = client_cache[cache_key]
                        print(f"[DEBUG MCP] ID: {manager.id}, Using cached client for {target_name} at {url}")
                    else:
                        print(f"[DEBUG MCP] ID: {manager.id}, Creating new client for {target_name} at {url}")
                        client = SyncedMcpClient(server_url=url, headers=headers)
                        client_cache[cache_key] = client
                        print(f"[DEBUG MCP] ID: {manager.id}, Created new client for {target_name}")
                    logger.debug(f"ID: {manager.id} (in mcp_calling), Initialized SyncedMcpClient with URL: {url}")
                    print(f"[DEBUG MCP] ID: {manager.id}, About to call list_tools() on {target_name}")
                    client.list_tools()
                    print(f"[DEBUG MCP] ID: {manager.id}, Successfully called list_tools() on {target_name}")
                    logger.debug(f"ID: {manager.id} (in mcp_calling), Retrieved tool list from MCP Server '{target_name}'.")
            except Exception as e:
                logger.error(f"ID: {manager.id} (in mcp_calling), Failed to initialize SyncedMcpClient for server '{mcp_server_name}': {str(e)}")
                client = None

            if client:
                try:
                    print(f"[DEBUG MCP] ID: {manager.id}, About to call tool '{mcp_tool_name}' on server '{mcp_server_name}' with args: {mcp_args}")
                    logger.debug(f"ID: {manager.id} (in mcp_calling), Calling tool '{mcp_tool_name}' with arguments: {mcp_args}")
                    result = client.call_tool(mcp_tool_name, mcp_args)
                    print(f"[DEBUG MCP] ID: {manager.id}, Successfully called tool '{mcp_tool_name}' on server '{mcp_server_name}'")
                    logger.debug(f"ID: {manager.id} (in mcp_calling), Raw tool call response from '{mcp_tool_name}': {result}")
                    texts = [item.text for item in result.content]

                    result_str_segment = ''.join(texts)
                    logger.debug(f"ID: {manager.id} (in mcp_calling), Cleaned tool call response from '{mcp_tool_name}': {result_str_segment}")

                    logger.info(f"ID: {manager.id} (in mcp_calling), MCP Server '{mcp_server_name}' returned: {result_str_segment[:5000]}")
                    print(f"[DEBUG MCP] ID: {manager.id}, Tool '{mcp_tool_name}' on server '{mcp_server_name}' returned {len(result_str_segment)} characters")

                    result_str += result_str_segment
                except Exception as e:
                    print(f"[DEBUG MCP ERROR] ID: {manager.id}, FAILED calling tool '{mcp_tool_name}' on server '{mcp_server_name}': {str(e)}")
                    logger.error(f"ID: {manager.id} (in mcp_calling), Error calling tool '{mcp_tool_name}' on MCP Server '{mcp_server_name}': {str(e)}")
            else:
                print(f"[DEBUG MCP ERROR] ID: {manager.id}, No client available for tool '{mcp_tool_name}' on server '{mcp_server_name}'")
                logger.warning(f"ID: {manager.id} (in mcp_calling), Skipping tool call for '{mcp_tool_name}' due to client initialization failure.")

        ## Tool call responses are truncated to 150000 characters!!!!
        messages.append({
            constants.ROLE: constants.TOOL,
            constants.CONTENT: result_str[:150000],
        })
        logger.debug(f"ID: {manager.id} (in mcp_calling), Final messages prepared: {messages}")
        logger.info(f"ID: {manager.id} (in mcp_calling), mcp_calling completed successfully.")
        # Clean up if we created the cache
        if created_cache:
            cleanup_all_clients(client_cache)
        return messages

# Helper function to clean up all clients in the cache
def cleanup_all_clients(client_cache: dict):
    for client in client_cache.values():
        try:
            client.cleanup()
        except Exception:
            pass

class DotDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            )

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            )
