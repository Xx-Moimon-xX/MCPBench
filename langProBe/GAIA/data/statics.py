import json
import re
from collections import defaultdict

def parse_tools(tools_str):
    """
    Parse Tools string and split it into individual tool lists.
    Assumes Tools field is each line of tools starting with numbers and dots, for example:
    "1. Web browser
    2. Image recognition tools (to identify and parse a figure with three axes)"
    """
    tools = []
    # Use regular expression to match each tool entry
    pattern = re.compile(r'\d+\.\s*(.*)')
    for line in tools_str.split('\n'):
        match = pattern.match(line.strip())
        if match:
            tool = match.group(1).strip()
            # Remove possible parenthetical explanations
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
                continue  # Skip empty lines
            # Debug info: confirm which line is being processed
            print(f"Processing line {line_number}")

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Line {line_number}: JSON decode error: {e}")
                continue

            # Extract Annotator Metadata
            annotator_metadata = data.get("Annotator Metadata", {})
            if not annotator_metadata:
                print(f"Line {line_number}: 'Annotator Metadata' field not found.")
                continue

            number_of_tools = annotator_metadata.get("Number of tools")
            tools_str = annotator_metadata.get("Tools", "")

            if number_of_tools is None:
                print(f"Line {line_number}: 'Number of tools' field not found.")
            else:
                try:
                    num_tools = int(number_of_tools)
                    tool_numbers.append(num_tools)
                except ValueError:
                    print(f"Line {line_number}: 'Number of tools' is not a valid integer.")

            if not tools_str:
                print(f"Line {line_number}: 'Tools' field is empty.")
                continue

            tools = parse_tools(tools_str)
            print(f"Line {line_number} parsed tools: {tools}")
            print(f"Line {line_number} tool count: {len(tools)}")

            # Verify if Number of tools matches parsed tool count
            if number_of_tools:
                try:
                    num_tools = int(number_of_tools)
                    if num_tools != len(tools):
                        print(f"Line {line_number}: Number of tools ({num_tools}) does not match parsed tool count ({len(tools)}).")
                except ValueError:
                    pass  # Already handled in previous step

            # Count occurrences of each tool
            for tool in tools:
                tool_counts[tool] += 1
                total_tools += 1

            processed_tasks += 1

    return tool_counts, tool_numbers, total_tools, processed_tasks

def main():
    jsonl_file = '2023/validation/metadata.jsonl'  # Replace with your JSONL file path
    tool_counts, tool_numbers, total_tools, processed_tasks = process_jsonl(jsonl_file)

    print("\nTotal occurrences of each tool:")
    if not tool_counts:
        print("No tools counted. Please check file content and parsing logic.")
    else:
        for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{tool}: {count}")

    # Calculate and output average tool count
    if tool_numbers:
        average_tools = sum(tool_numbers) / len(tool_numbers)
        print(f"\nAverage tool count per question: {average_tools:.2f}")
    else:
        print("\nNo 'Number of tools' data counted.")

    print(f"\nTotal processed questions: {processed_tasks}")
    print(f"Total tool count: {total_tools}")

if __name__ == "__main__":
        main()
