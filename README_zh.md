<h1 align="center">
	🦊 MCPBench: A Benchmark for Evaluating MCP Servers
</h1>

<div align="center">

[![文档][docs-image]][docs-url]
[![软件包许可证][package-license-image]][package-license-url]

</div>

<div align="center">
<h4 align="center">

[中文](https://github.com/modelscope/MCPBench/blob/main/README_zh.md) |
[English](https://github.com/modelscope/MCPBench/blob/main/README.md)

</h4>
</div>

MCPBench 是一个用于评估 MCP Server的基准测试框架。它支持评估三种类型的服务器：网络搜索、数据库查询和GAIA任务，并且兼容本地和远程 MCP 服务器。该框架主要在相同的 LLM 和 Agent 配置下，从任务完成准确性、延迟和 Token 消耗等方面评估不同的 MCP 服务器（如 Brave Search、DuckDuckGo 等）。详见[评估报告](https://arxiv.org/abs/2504.11094)。

<img src="assets/figure1.png" alt="MCPBench 概览" width="600"/>

> 实现参考了 [LangProBe: a Language Programs Benchmark](https://arxiv.org/abs/2502.20315)。\
> 特别感谢 Qingxu Fu 的初始实现！

<hr>

# 📋 目录

- [🔥 最新动态](#news)
- [🛠️ 安装](#installation)
- [🚀 快速开始](#quick-start)
  - [启动 MCP 服务器](#launch-mcp-server)
  - [启动评测](#launch-evaluation)
- [🧂 数据集与实验](#datasets-and-experiments)
- [🚰 引用](#cite)

# 🔥 最新动态
+ `2025年4月29日` 🌟 更新了GAIA内MCP Server Package的评测代码。
+ `2025年4月14日` 🌟 MCPBench 正式开源。

# 🛠️ 安装
本框架需要 Python >= 3.11、nodejs 和 jq。

```bash
conda create -n mcpbench python=3.11 -y
conda activate mcpbench
pip install -r requirements.txt
```
# 🚀 快速开始
请先确定你要使用的 MCP 服务器类型：
- 若为远程主机（通过 **SSE** 访问，如 [ModelScope](https://modelscope.cn/mcp)、[Smithery](https://smithery.ai) 或 localhost），可直接进行[评测](#launch-evaluation)。
- 若为本地启动（通过 npx 以 **STDIO** 访问），你需要启动MCP服务器。
## 启动 MCP 服务器
首先，需要编写如下配置：
```json
{
    "mcp_pool": [
        {
            "name": "firecrawl",
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
将该配置文件保存至 `configs` 文件夹，并通过如下命令启动：

```bash
sh launch_mcps_as_sse.sh YOUR_CONFIG_FILE
```

例如，将上述配置保存为 `configs/firecrawl.json`，并通过如下命令启动：

```bash
sh launch_mcps_as_sse.sh firecrawl.json
```

## 启动评测
要评测 MCP 服务器性能，需设置相关信息。代码会自动检测服务器中的工具和参数，无需手动配置。例如：

```json
{
    "mcp_pool": [
        {
            "name": "Remote MCP example",
            "url": "url from https://modelscope.cn/mcp or https://smithery.ai"
        },
        {
            "name": "firecrawl (Local run example)",
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

评测 MCP 服务器在网页搜索任务上的表现：
```bash
sh evaluation_websearch.sh YOUR_CONFIG_FILE
```

评测 MCP 服务器在数据库查询任务上的表现：
```bash
sh evaluation_db.sh YOUR_CONFIG_FILE
```

评测 MCP 服务器在 GAIA 任务上的表现：
```bash
sh evaluation_gaia.sh YOUR_CONFIG_FILE
```

例如，将上述配置保存为 `configs/firecrawl.json`，并通过如下命令启动：

```bash
sh evaluation_websearch.sh firecrawl.json
```

# 数据集与实验结果
本框架提供了两类评测数据集：
- 网页搜索任务数据集位于 `MCPBench/langProBe/WebSearch/data/websearch_600.jsonl`，包含来自 [Frames](https://arxiv.org/abs/2409.12941)、新闻、科技领域的各200组问答对。自动化构建评测数据集的工具后续也将开源。
- 数据库查询任务数据集位于 `MCPBench/langProBe/DB/data/car_bi.jsonl`。你也可以按如下格式自定义数据集：

```json
{
  "unique_id": "",
  "Prompt": "",
  "Answer": ""
}
```

我们已在主流 MCP 服务器上完成了上述任务的评测。详细实验结果请参考[文档](https://arxiv.org/abs/2504.11094)。

# 🚰 引用
如果本项目对你有帮助，请引用我们的工作或是给我们一个🌟：

```bibtex
@misc{mcpbench,
  title={MCPBench: A Benchmark for Evaluating MCP Servers},
  author={Zhiling Luo, Xiaorong Shi, Xuanrui Lin, Jinyang Gao},
  howpublished = {\url{https://github.com/modelscope/MCPBench}},
  year={2025}
}
```

或引用我们的报告：
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

