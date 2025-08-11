import argparse
import csv
import glob
import json
import os
from typing import Dict, List, Tuple, Set, Optional

def extract_unique_question_threads(file_path: str) -> Tuple[List[Dict], int, int]:
    """
    Extract unique question threads from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file containing message threads
        
    Returns:
        Tuple of (extracted_data, total_unique_threads, successful_threads)
    """
    seen_questions: Set[str] = set()
    extracted_data: List[Dict] = []
    successful_threads = 0
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            try:
                # Parse JSON line
                thread_data = json.loads(line.strip())
                
                # Extract the question (first message from user/human)
                question = thread_data.get("question", "")
                
                # Skip if we've seen this question before
                if question in seen_questions:
                    continue
                
                # Add question to seen set
                seen_questions.add(question)
                
                # Extract messages to find last assistant message
                messages = thread_data.get("messages", [])
                last_assistant_message = ""
                
                # Find the last message from assistant/AI
                for message in reversed(messages):
                    if message.get("role") in ["assistant", "AI"]:
                        last_assistant_message = message.get("content", "")
                        break
                
                # Extract other required fields
                success = thread_data.get("success", False)
                time_cost = thread_data.get("time_cost", 0)
                prompt_tokens_cost = thread_data.get("prompt_tokens_cost", 0)
                completion_tokens_cost = thread_data.get("completion_tokens_cost", 0)
                
                # Count successful threads
                if success:
                    successful_threads += 1
                
                # Add to extracted data
                extracted_data.append({
                    "question": question,
                    "last_assistant_message": last_assistant_message,
                    "success": success,
                    "time_cost": time_cost,
                    "prompt_tokens_cost": prompt_tokens_cost,
                    "completion_tokens_cost": completion_tokens_cost
                })
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    return extracted_data, len(extracted_data), successful_threads

def save_to_csv(data: List[Dict], output_path: str):
    """
    Save extracted data to CSV file.
    
    Args:
        data: List of dictionaries containing extracted thread data
        output_path: Path where CSV file should be saved
    """
    if not data:
        print("No data to save.")
        return
    
    fieldnames = [
        "question",
        "last_assistant_message", 
        "success",
        "time_cost",
        "prompt_tokens_cost",
        "completion_tokens_cost"
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def _find_latest_messages_file(search_root: str, pattern: str) -> Optional[str]:
    """
    Search recursively under search_root for files matching pattern and
    return the most recently modified one, if any.
    """
    search_glob = os.path.join(search_root, "**", pattern)
    candidates = glob.glob(search_glob, recursive=True)
    if not candidates:
        return None
    # Pick by modification time (latest first)
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def resolve_input_file(input_arg: Optional[str]) -> Optional[str]:
    """
    Resolve the input JSONL file path.

    - If input_arg is a file path, return it.
    - If input_arg is a directory, search within it for the latest
      'eval_benchmark_1_messages_*.jsonl'.
    - If input_arg is None, search under the directory containing this script.
    """
    pattern = "eval_benchmark_1_messages_*.jsonl"

    if input_arg:
        if os.path.isfile(input_arg):
            return input_arg
        if os.path.isdir(input_arg):
            return _find_latest_messages_file(input_arg, pattern)
        # If provided but neither file nor dir, treat as not found
        return None

    # Default: search from the directory this script resides in (works when moved to runs/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return _find_latest_messages_file(script_dir, pattern)

def main():
    parser = argparse.ArgumentParser(description="Extract unique question threads and save analysis CSV next to the input JSONL.")
    parser.add_argument(
        "--input",
        "-i",
        help="Path to the input JSONL file or a run directory containing 'eval_benchmark_1_messages_*.jsonl'. If omitted, the script searches under its own directory.",
    )
    args = parser.parse_args()

    input_file_path = resolve_input_file(args.input)
    if not input_file_path or not os.path.exists(input_file_path):
        where = args.input if args.input else os.path.dirname(os.path.abspath(__file__))
        print(f"Error: Could not locate input messages JSONL. Looked under: {where}")
        return
    
    # Generate output file path in same directory as input
    input_dir = os.path.dirname(input_file_path)
    input_filename = os.path.splitext(os.path.basename(input_file_path))[0]
    output_file_path = os.path.join(input_dir, f"{input_filename}_analysis.csv")
    
    print(f"Processing file: {input_file_path}")
    print(f"Output will be saved to: {output_file_path}")
    
    # Extract unique question threads
    extracted_data, total_unique_threads, successful_threads = extract_unique_question_threads(input_file_path)
    
    # Save to CSV
    save_to_csv(extracted_data, output_file_path)
    
    # Print summary statistics
    print(f"\n=== ANALYSIS SUMMARY ===")
    print(f"Total unique question threads: {total_unique_threads}")
    print(f"Successful threads: {successful_threads}")
    print(f"Success rate: {(successful_threads/total_unique_threads)*100:.2f}%" if total_unique_threads > 0 else "Success rate: N/A")
    print(f"CSV file saved: {output_file_path}")

if __name__ == "__main__":
    main()