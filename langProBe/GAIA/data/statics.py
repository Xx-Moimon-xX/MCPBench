import json
import re
from collections import defaultdict

def parse_tools(tools_str):
    """
    解析 Tools 字符串，将其分割为单独的工具列表。
    假设 Tools 字段是以数字和点开头的每行工具，例如：
    "1. Web browser
    2. Image recognition tools (to identify and parse a figure with three axes)"
    """
    tools = []
    # 使用正则表达式匹配每个工具条目
    pattern = re.compile(r'\d+\.\s*(.*)')
    for line in tools_str.split('\n'):
        match = pattern.match(line.strip())
        if match:
            tool = match.group(1).strip()
            # 去除可能的括号内说明
            tool = re.sub(r'\s*\(.*\)', '', tool)
            tools.append(tool)
    return tools

def process_jsonl(file_path):
    tool_counts = defaultdict(int)
    total_tools = 0
    tool_numbers = []
    processed_tasks = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # 跳过空行
            # 调试信息：确认正在处理哪一行
            print(f"处理第 {line_number} 行")

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"第 {line_number} 行: JSON 解码错误: {e}")
                continue

            # 提取 Annotator Metadata
            annotator_metadata = data.get("Annotator Metadata", {})
            if not annotator_metadata:
                print(f"第 {line_number} 行: 未找到 'Annotator Metadata' 字段。")
                continue

            number_of_tools = annotator_metadata.get("Number of tools")
            tools_str = annotator_metadata.get("Tools", "")

            if number_of_tools is None:
                print(f"第 {line_number} 行: 未找到 'Number of tools' 字段。")
            else:
                try:
                    num_tools = int(number_of_tools)
                    tool_numbers.append(num_tools)
                except ValueError:
                    print(f"第 {line_number} 行: 'Number of tools' 不是有效的整数。")

            if not tools_str:
                print(f"第 {line_number} 行: 'Tools' 字段为空。")
                continue

            tools = parse_tools(tools_str)
            print(f"第 {line_number} 行解析到的工具: {tools}")
            print(f"第 {line_number} 行的工具数量: {len(tools)}")

            # 验证 Number of tools 是否与解析的工具数量一致
            if number_of_tools:
                try:
                    num_tools = int(number_of_tools)
                    if num_tools != len(tools):
                        print(f"第 {line_number} 行: Number of tools ({num_tools}) 与解析的工具数量 ({len(tools)}) 不一致。")
                except ValueError:
                    pass  # 已在上一步处理

            # 统计每个工具的出现次数
            for tool in tools:
                tool_counts[tool] += 1
                total_tools += 1

            processed_tasks += 1

    return tool_counts, tool_numbers, total_tools, processed_tasks

def main():
    jsonl_file = '2023/validation/metadata.jsonl'  # 替换为你的 JSONL 文件路径
    tool_counts, tool_numbers, total_tools, processed_tasks = process_jsonl(jsonl_file)

    print("\n每个工具的总出现次数：")
    if not tool_counts:
        print("没有统计到任何工具。请检查文件内容和解析逻辑。")
    else:
        for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{tool}: {count}")

    # 计算并输出平均工具数量
    if tool_numbers:
        average_tools = sum(tool_numbers) / len(tool_numbers)
        print(f"\n平均每个题目的工具数量: {average_tools:.2f}")
    else:
        print("\n没有统计到任何 'Number of tools' 数据。")

    print(f"\n总处理题目数: {processed_tasks}")
    print(f"总工具数量: {total_tools}")

if __name__ == "__main__":
        main()
