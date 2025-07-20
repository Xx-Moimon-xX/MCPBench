from langProBe.benchmark import BenchmarkMeta, MCPBench
from langProBe.evaluation_utils import mcp_metric
from .eval_1_program import Eval1Predict

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
    eval_benchmark_1_baseline = Eval1Predict(
        max_steps=5,
        system_prompt=MCP_SAMPLE_SYSTEM_PROMPT,
        task_name="eval_benchmark_1")

    # It's giving a different program object, but I think it should be the benchmark object that's changed not the program object.
    # Because the program object is a constant (i.e. how the system generates responses), and the benchmark is what should change.
    return [
        BenchmarkMeta(
            MCPBench,
            [eval_benchmark_1_baseline],
            mcp_metric,
            optimizers=[],
            name="EVAL_BENCHMARK_1"
        )
    ]

# Returns a BenchmarkMeta object
benchmark = get_eval_benchmark_1()