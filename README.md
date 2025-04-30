<h1 align="center">
	🦊 MCPBench: A Benchmark for Evaluating MCP Servers
</h1>



<div align="center">

[![Documentation][docs-image]][docs-url]
[![Package License][package-license-image]][package-license-url]

</div>

<div align="center">
<h4 align="center">

[中文](https://github.com/modelscope/MCPBench/blob/main/README_zh.md) |
[English](https://github.com/modelscope/MCPBench/blob/main/README.md)

</h4>
</div>

MCPBench is an evaluation framework for MCP Servers. It supports the evaluation of three types of servers: Web Search, Database Query and GAIA, and is compatible with both local and remote MCP Servers. The framework primarily evaluates different MCP Servers (such as Brave Search, DuckDuckGo, etc.) in terms of task completion accuracy, latency, and token consumption under the same LLM and Agent configurations. Here is the [evaluation report](https://arxiv.org/abs/2504.11094).

<img src="assets/figure1.png" alt="MCPBench Overview" width="600"/>

> The implementation refers to [LangProBe: a Language Programs Benchmark](https://arxiv.org/abs/2502.20315).\
> Big thanks to Qingxu Fu for the initial implementation!

<hr>



# 📋 Table of Contents

- [🔥 News](#news)
- [🛠️ Installation](#installation)
- [🚀 Quick Start](#quick-start)
  - [LLM Configuration](#llm-configuration)
  - [Launch MCP Server](#launch-mcp-server)
  - [Launch Evaluation](#launch-evaluation)
- [🧂 Datasets and Experiments](#datasets-and-experiments)
- [🚰 Cite](#cite)

# 🔥 News
+ `Apr. 29, 2025` 🌟 Update the code for evaluating the MCP Server Package within GAIA.
+ `Apr. 14, 2025` 🌟 We are proud to announce that MCPBench is now open-sourced.

# 🛠️ Installation
The framework requires Python version >= 3.11, nodejs and jq.

```bash
conda create -n mcpbench python=3.11 -y
conda activate mcpbench
pip install -r requirements.txt

```

# 🚀 Quick Start

## Launch MCP Server
### Launch stdio MCP as SSE
If the MCP does not support SSE, write the config like:
```json
{
    "mcp_pool": [
        {
            "name": "FireCrawl",
            "description": "A Model Context Protocol (MCP) server implementation that integrates with Firecrawl for web scraping capabilities.",
            "tools": [
                {
                    "tool_name": "firecrawl_search",
                    "tool_description": "Search the web and optionally extract content from search results.",
                    "inputs": [
                        {
                            "name": "query",
                            "type": "string",
                            "required": true,
                            "description": "your search query"
                        }
                    ]
                }
            ],
            "run_config": [
                {
                    "command": "npx -y firecrawl-mcp",
                    "args": "FIRECRAWL_API_KEY=xxx",
                    "port": 8005
                }
            ]
        }
    ]
}
```

Save this config file in the `configs` folder and launch it using:

```bash
sh launch_mcps_as_sse.sh YOUR_CONFIG_FILE
```

For example, if the config file is mcp_config_websearch.json, then run:
```bash
sh launch_mcps_as_sse.sh mcp_config_websearch.json
```

### Launch SSE MCP
If your server supports SSE, you can use it directly. The URL will be http://localhost:8001/sse

For SSE-supported MCP Server, write the config like:
```json
{
    "mcp_pool": [
        {
            "name": "browser_use",
            "description": "AI-driven browser automation server implementing the Model Context Protocol (MCP) for natural language browser control and web research.",
            "tools": [
                {
                    "tool_name": "browser_use",
                    "tool_description": "Executes a browser automation task based on natural language instructions and waits for it to complete.",
                    "inputs": [
                        {
                            "name": "query",
                            "type": "string",
                            "required": true,
                            "description": "Your query"
                        }
                    ]
                }
            ],
            "url": "http://0.0.0.0:8001/sse"
        }
    ]
}

```
where the url can be generated from the MCP market on [ModelScope](https://www.modelscope.cn/mcp).

## Launch Evaluation
To evaluate the MCP Server's performance on Web Search tasks:
```bash
sh evaluation_websearch.sh YOUR_CONFIG_FILE
```

To evaluate the MCP Server's performance on Database Query tasks:
```bash
sh evaluation_db.sh YOUR_CONFIG_FILE
```

To evaluate the MCP Server's performance on GAIA tasks:
```bash
sh evaluation_gaia.sh YOUR_CONFIG_FILE
```

# Datasets and Experimental Results
Our framework provides two datasets for evaluation. For the WebSearch task, the dataset is located at `MCPBench/langProBe/WebSearch/data/websearch_600.jsonl`, containing 200 QA pairs each from [Frames](https://arxiv.org/abs/2409.12941), news, and technology domains. Our framework for automatically constructing evaluation datasets will be open-sourced later.

For the Database Query task, the dataset is located at `MCPBench/langProBe/DB/data/car_bi.jsonl`. You can add your own dataset in the following format:

```json
{
  "unique_id": "",
  "Prompt": "",
  "Answer": ""
}
```

We have evaluated mainstream MCP Servers on both tasks. For detailed experimental results, please refer to [Documentation](https://arxiv.org/abs/2504.11094)

# 🚰 Cite
If you find this work useful, please consider citing our project:

```bibtex
@misc{mcpbench,
  title={MCPBench: A Benchmark for Evaluating MCP Servers},
  author={Zhiling Luo, Xiaorong Shi, Xuanrui Lin, Jinyang Gao},
  howpublished = {\url{https://github.com/modelscope/MCPBench}},
  year={2025}
}
```

Alternatively, you may reference our report.
```bibtex
@article{mcpbench_report,
      title={Evaluation Report on MCP Servers}, 
      author={Zhiling Luo, Xiaorong Shi, Xuanrui Lin, Jinyang Gao},
      year={2025},
      journal={arXiv preprint arXiv:2504.11094},
      url={https://arxiv.org/abs/2504.11094},
      primaryClass={cs.AI}
}
```

[docs-image]: https://img.shields.io/badge/Documentation-EB3ECC
[docs-url]: https://arxiv.org/abs/2504.11094
[package-license-image]: https://img.shields.io/badge/License-Apache_2.0-blue.svg
[package-license-url]: https://github.com/modelscope/MCPBench/blob/main/LICENSE

