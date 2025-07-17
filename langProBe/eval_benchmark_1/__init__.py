from langProBe.benchmark import BenchmarkMeta, MCPBench
from langProBe.mcp_program import MCPPredict
from langProBe.evaluation_utils import eval_prompt_1_metric

MCP_SAMPLE_SYSTEM_PROMPT = """
You are a helpful assistant. You are able to answer questions using different tools.  
The content of your available tools begins with ## Available Tools, indicating the collection of usable tools.  
Within the tool collection, each server is identified by ### server_name, where server_name represents the name of the server.  
Under each server, there are multiple tools (tool), and each tool starts with - tool_name, where tool_name is the name of the tool.  
The tool description includes:  
A brief text description outlining the functionality of the tool.  
Detailed information about input parameters, where each parameter includes: parameter name, parameter type, whether it is mandatory, and the purpose or description of the parameter.
"""

def get_eval_benchmark_1():
    eval_benchmark_1_baseline = MCPPredict(
        max_steps=5,
        system_prompt=MCP_SAMPLE_SYSTEM_PROMPT,
        task_name="eval_benchmark_1")

    return [
        BenchmarkMeta(
            MCPBench,
            [eval_benchmark_1_baseline],
            eval_prompt_1_metric,
            optimizers=[],
            name="EVAL_BENCHMARK_1"
        )
    ]

benchmark = get_eval_benchmark_1()