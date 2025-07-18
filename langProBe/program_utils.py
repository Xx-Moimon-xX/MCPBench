from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
from typing import List, Tuple, Optional, Dict, Union
from openai import OpenAI
import json
import copy
from pydantic import BaseModel, Field
import re
import os
import langProBe.constants as constants
import logging
from .synced_mcp_client import SyncedMcpClient
try:
    from anthropic import Anthropic
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
    stop=stop_after_attempt(5),  
    wait=wait_exponential(multiplier=1, min=2, max=10),  
    reraise=True,
)
def call_lm(
            messages: List, 
            manager: ProcessManager, 
            logger: logging.Logger, 
            temperature: float|None=None,
            ) -> tuple[str | None, int, int]:    
    '''
    This function is used to call the LLM API, it can be used for Anthropic, AWS Bedrock and OpenAI.
    '''
    # Log the input messages being sent to the LLM
    logger.debug(f"ID: {manager.id}, Input messages to LLM: {json.dumps(messages, indent=2, ensure_ascii=False)}")
    
    response = None
    try:
        # Getting the correct model to use for the LLM call.
        prefix, model_name = manager.model.split('/')
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
            # Call Claude API
            completion = client.messages.create(
                model=model_name,
                max_tokens=1024,
                messages=claude_messages,
                temperature=temperature if temperature is not None else 0.7,
            )
            
            # Log the full response for debugging
            logger.debug(f"ID: {manager.id}, Full Anthropic API response: {completion}")
            
            response_text = completion.content[0].text if completion.content else ""
            
            # Log extracted content
            logger.debug(f"ID: {manager.id}, Extracted response_text: '{response_text}'")
            # Anthropic does not return token usage in the same way
            completion_tokens = getattr(completion, 'usage', {}).get('output_tokens', 0) if hasattr(completion, 'usage') else 0
            prompt_tokens = getattr(completion, 'usage', {}).get('input_tokens', 0) if hasattr(completion, 'usage') else 0
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
                region_name=manager.aws_region or os.getenv("AWS_REGION", "us-east-1")
            )
            
            # Convert messages to format expected by Bedrock Converse API
            bedrock_messages = []
            system_message = ""
            
            for m in messages:
                if m.get("role") == "system":
                    system_message = m["content"]
                elif m.get("role") == "user":
                    bedrock_messages.append({"role": "user", "content": [{"text": m["content"]}]})
                elif m.get("role") == "assistant":
                    bedrock_messages.append({"role": "assistant", "content": [{"text": m["content"]}]})
                elif m.get("role") == "tool":
                    bedrock_messages.append({"role": "user", "content": [{"text": m["content"]}]})
            
            try:
                # Prepare request for Bedrock Converse API
                request_params = {
                    "modelId": model_name,
                    "messages": bedrock_messages,
                    "inferenceConfig": {
                        "maxTokens": 1024,
                        "temperature": temperature if temperature is not None else 0.7
                    }
                }
                
                if system_message:
                    request_params["system"] = [{"text": system_message}]
                
                # Calling the AWS Bedrock Converse API
                response = bedrock_client.converse(**request_params)
                
                # Log the full response for debugging
                logger.debug(f"ID: {manager.id}, Full Bedrock API response: {json.dumps(response, indent=2, default=str)}")
                
                # Extract response content
                output_message = response.get('output', {}).get('message', {})
                content_list = output_message.get('content', [])
                
                response_text = ""
                if content_list:
                    for content in content_list:
                        if 'text' in content:
                            response_text += content['text']
                
                # Log extracted content
                logger.debug(f"ID: {manager.id}, Extracted response_text: '{response_text}'")
                
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
                logger.error(f"ID: {manager.id}, AWS Bedrock error: {str(e)}")
                raise

        # OpenAI API call
        else:
            # --- OpenAI logic as before ---
            # Creating the OpenAI client
            oai = OpenAI(
                api_key=manager.lm_api_key,
                base_url=manager.lm_api_base,
            )
            assert prefix == 'openai'

            if model_name in ['deepseek-r1', 'qwq-plus', 'qwq-32b']: # qwen reasoning models only support streaming output
                reasoning_content = ""  # Define complete reasoning process
                answer_content = ""     # Define complete response
                is_answering = False   # Determine if reasoning process is complete and response has started

                completion = oai.chat.completions.create(
                    model=model_name, 
                    messages=messages,
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
                    messages=messages,
                    model=model_name,
                    temperature = temperature
                )
            else:
                response = oai.beta.chat.completions.parse(
                    messages=messages,
                    model=model_name,
                )
                # Log the full response for debugging
                logger.debug(f"ID: {manager.id}, Full OpenAI API response: {response}")
                
                response_text = response.choices[0].message.content or ""
                
                # Log extracted content
                logger.debug(f"ID: {manager.id}, Extracted response_text: '{response_text}'")
                
                completion_tokens = response.usage.completion_tokens
                prompt_tokens = response.usage.prompt_tokens
            manager.lm_usages.append({
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                })
            return response_text, completion_tokens, prompt_tokens
    
    except Exception as e:
        logger.error(f"ID: {manager.id}, Error in call_lm: {str(e)}")
        if response:
            logger.error(f"ID: {manager.id}, Response: {response}")
        raise

def build_system_content(base_system: str,
                        mcps: List,
                        ) -> str:
    '''
    Build the system content for the conversation, i.e. the system prompt and the available tools.
    '''
    tools_section = "## Available Tools\n"
    for mcp in mcps:
        tools_section += f"### Server '{mcp['name']}' include following tools\n"
        if mcp['name'] in ['wuying-agentbay-mcp-server', 'Playwright']:
            tools_section += f"When using this server to perform search tasks, please use https://www.baidu.com as the initial website for searching."
        
        # Connecting the the MCP server to get the tools!!!!
        url = mcp.get("url")
        if not url:
            try:
                port = mcp.get('run_config')[0]["port"]
                url = f"http://localhost:{port}/sse"
            except:
                raise Exception("No url found")
        client = SyncedMcpClient(server_url=url)
        try:
            result = client.list_tools()
            tools = result.tools
        except Exception as e:
            raise Exception(f"Fail access to server: {mcp['name']}, error: {e}")

        # Formatting tools section to send in prompt
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

    prompt = base_system + f"""{tools_section}""" + TOOL_PROMPT

    return prompt


def build_init_messages(
        base_system: str,
        mcps: List,
        user_question: str,
       ) -> List[Dict]:
    '''
    Build the initial messages for the conversation, i.e. the system prompt and the user question.
    '''
    system_content = build_system_content(base_system, mcps)
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
) -> List[Dict]:
    '''
    Processes each tool call in the MCP call list.
    '''
    logger.debug(f"ID:{manager.id}, Entering mcp_calling with mcp_call_list: {mcp_call_list}")

    if mcp_call_list.shutdown:
        logger.info(f"ID:{manager.id}, Shutdown flag is set. No more MCP calling.")
        messages = [
            {
                constants.ROLE: constants.ASSISTANT,
                constants.CONTENT: mcp_call_list.raw_content if mcp_call_list.raw_content else '',
            }
        ]
        logger.debug(f"ID:{manager.id}, Shutdown messages prepared: {messages}")
        return messages
    else:
        logger.info(f"ID:{manager.id}, Processing MCP call list with {len(mcp_call_list.mcps)} MCPs. mcp_call_list: {mcp_call_list}")
        mcp_list = mcp_call_list.mcps
        messages = [
            {
                constants.ROLE: constants.ASSISTANT,
                constants.CONTENT: mcp_call_list.raw_content if mcp_call_list.raw_content else '',
                constants.TOOL_CALLS: []
            }
        ]
        result_str = ""

        # Iterating over each MCP call in the MCP call list
        for idx, mcp in enumerate(mcp_list, start=1):
            logger.debug(f"ID:{manager.id}, Processing MCP #{idx}: {mcp}")
            mcp_server_name = mcp.mcp_server_name
            mcp_tool_name = mcp.mcp_tool_name
            mcp_args = mcp.mcp_args

            tool_call = {
                "type": "function",
                "function": {
                    "name": mcp_tool_name,
                    "arguments": json.dumps(mcp_args, ensure_ascii=False)
                }
            }
            messages[0][constants.TOOL_CALLS].append(tool_call)
            logger.info(f"ID:{manager.id}, Calling MCP Server: {mcp_server_name}, Tool: {mcp_tool_name}, Arguments: {mcp_args}")

            # Manage manager.mcp_rts and manager.mcp_retry_times
            try:
                # Use passed config parameter, fallback to global_config if needed
                logger.debug(f"ID:{manager.id}, Received config parameter: {config}")
                parsed_data = config
                if parsed_data is None:
                    from langProBe.evaluation import global_config
                    logger.debug(f"ID:{manager.id}, Fallback to global_config: {global_config}")
                    parsed_data = global_config
                
                
                # Handle case where config is None
                if parsed_data is None:
                    logger.error(f"ID:{manager.id}, config is None, cannot initialize MCP client")
                    logger.warning(f"ID:{manager.id}, Skipping tool call for '{mcp_tool_name}' due to missing configuration.")
                    continue

                # Additional safety check
                if not isinstance(parsed_data, dict):
                    logger.error(f"ID:{manager.id}, config is not a dict: {type(parsed_data)}")
                    logger.warning(f"ID:{manager.id}, Skipping tool call for '{mcp_tool_name}' due to invalid configuration.")
                    continue

                target_name = mcp_server_name
                port = None
                url = None
                logger.debug(f"ID:{manager.id}, Parsed config keys: {list(parsed_data.keys())}")
                mcp_pool = parsed_data.get("mcp_pool", [])
                logger.debug(f"ID:{manager.id}, MCP pool: {mcp_pool}")
                if not mcp_pool:
                    logger.error(f"ID:{manager.id}, No MCP pool found in configuration")
                    logger.warning(f"ID:{manager.id}, Skipping tool call for '{mcp_tool_name}' due to missing MCP pool.")
                    continue

                for item in mcp_pool:
                    if item.get("name") != target_name:
                        continue

                    url = item.get("url", "")
                    if url:
                        logger.debug(f"ID:{manager.id}, Found URL for MCP Server '{target_name}': {url}")
                        break
                    run_configs = item.get("run_config", [])
                    for run_config in run_configs:
                        port = run_config.get("port")
                        if port:
                            url = f"http://localhost:{port}/sse"
                            logger.debug(f"ID:{manager.id}, Constructed URL for MCP Server '{target_name}': {url}")
                            break
                    if url:
                        break

                if not url:
                    logger.error(f"ID:{manager.id}, No valid URL found for MCP Server '{target_name}'.")
                    raise ValueError(f"ID:{manager.id}, No valid URL found for MCP Server '{target_name}'.")

                client = SyncedMcpClient(server_url=url)
                logger.debug(f"ID:{manager.id}, Initialized SyncedMcpClient with URL: {url}")
                client.list_tools()
                logger.debug(f"ID:{manager.id}, Retrieved tool list from MCP Server '{target_name}'.")
            except Exception as e:
                logger.error(f"ID:{manager.id}, Failed to initialize SyncedMcpClient for server '{mcp_server_name}': {str(e)}")
                client = None

            if client:
                try:
                    logger.debug(f"ID:{manager.id}, Calling tool '{mcp_tool_name}' with arguments: {mcp_args}")
                    result = client.call_tool(mcp_tool_name, mcp_args)
                    texts = [item.text for item in result.content]
                    result_str_segment = ''.join(texts)
                    logger.debug(f"ID:{manager.id}, Received result from tool '{mcp_tool_name}': {result_str_segment}")

                    logger.info(f"ID:{manager.id}, MCP Server '{mcp_server_name}' returned: {result_str_segment[:5000]}")

                    result_str += result_str_segment
                except Exception as e:
                    logger.error(f"ID:{manager.id}, Error calling tool '{mcp_tool_name}' on MCP Server '{mcp_server_name}': {str(e)}")
            else:
                logger.warning(f"ID:{manager.id}, Skipping tool call for '{mcp_tool_name}' due to client initialization failure.")

        messages.append({
            constants.ROLE: constants.TOOL,
            constants.CONTENT: result_str[:5000],
        })
        logger.debug(f"ID:{manager.id}, Final messages prepared: {messages}")
        logger.info(f"ID:{manager.id}, mcp_calling completed successfully.")
        return messages

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
