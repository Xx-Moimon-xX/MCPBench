from langProBe.benchmark import BenchmarkMeta, MCPBench
from langProBe.mcp_program import MCPPredict
from langProBe.evaluation_utils import mcp_metric
from .gaia_program import GAIAPredict

MCP_SAMPLE_SYSTEM_PROMPT = """
You are a helpful assistant. You are able to answer questions using different tools.  
The content of your available tools begins with ## Available Tools, indicating the collection of usable tools.  
Within the tool collection, each server is identified by ### server_name, where server_name represents the name of the server.  
Under each server, there are multiple tools (tool), and each tool starts with - tool_name, where tool_name is the name of the tool.  
The tool description includes:  
A brief text description outlining the functionality of the tool.  
Detailed information about input parameters, where each parameter includes: parameter name, parameter type, whether it is mandatory, and the purpose or description of the parameter.
If you have obtained the final result. Please provide your final answer enclosed within <answer></answer> tags. Ensure that only the final answer is included, without any additional explanations or commentary.
"""
def get_mcp_sample_benchmark():
    mcp_sample_baseline = GAIAPredict(
                                max_steps=5, 
                                system_prompt=MCP_SAMPLE_SYSTEM_PROMPT, 
                                task_name="gaia")
    
    return [
        BenchmarkMeta(
            MCPBench,
            [mcp_sample_baseline],
            mcp_metric,
            optimizers=[],
            name="MCP_GAIA"
        )
    ]

benchmark = get_mcp_sample_benchmark()