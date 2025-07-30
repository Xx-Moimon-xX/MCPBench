from langProBe.benchmark import BenchmarkMeta, MCPBench
from .db_program import DBPredict
from langProBe.evaluation_utils import mcp_metric

MCP_SAMPLE_SYSTEM_PROMPT = """
You are a helpful assistant. You are able to answer questions using different tools.  
The content of your available tools begins with ## Available Tools, indicating the collection of usable tools.  
Within the tool collection, each server is identified by ### server_name, where server_name represents the name of the server.  
Under each server, there are multiple tools (tool), and each tool starts with - tool_name, where tool_name is the name of the tool.  
The tool description includes:  
A brief text description outlining the functionality of the tool.  
Detailed information about input parameters, where each parameter includes: parameter name, parameter type, whether it is mandatory, and the purpose or description of the parameter.
"""

def get_mcp_sample_benchmark():
    mcp_sample_baseline = DBPredict(
        max_steps=5,
        system_prompt=MCP_SAMPLE_SYSTEM_PROMPT,
        task_name="database_search")

    return [
        BenchmarkMeta(
            MCPBench,
            [mcp_sample_baseline],
            mcp_metric,
            optimizers=[],
            name="MCP_DB"
        )
    ]

benchmark = get_mcp_sample_benchmark()