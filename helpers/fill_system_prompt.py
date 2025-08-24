#!/usr/bin/env python3
"""
Script to fill the 300 tools system prompt with filler context from Paul Graham essays
to reach approximately 23,000 tokens.
"""

import os
import re
from pathlib import Path

def count_tokens_approximate(text):
    """
    Approximate token count using a simple heuristic.
    Roughly 1 token ≈ 4 characters for English text.
    """
    return len(text) // 4

def read_file_content(file_path):
    """Read file content and return as string."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def extract_essay_content(text):
    """
    Extract the main content from Paul Graham essays, removing headers and formatting.
    """
    # Remove the title and date header
    lines = text.split('\n')
    
    # Find where the actual content starts (after the date line)
    content_start = 0
    for i, line in enumerate(lines):
        if re.match(r'^[A-Z][a-z]+ \d{4}$', line.strip()):  # Month Year format
            content_start = i + 1
            break
    
    # Get the content starting from after the header
    content_lines = lines[content_start:]
    
    # Clean up the content
    cleaned_lines = []
    for line in content_lines:
        line = line.strip()
        if line and not line.startswith('[') and not line.startswith('('):
            cleaned_lines.append(line)
    
    return '\n\n'.join(cleaned_lines)

def fill_system_prompt():
    """Main function to fill the system prompt with filler context."""
    
    # Read the original system prompt
    prompt_file = "300 tools system prompt.txt"
    original_prompt = read_file_content(prompt_file)
    
    if not original_prompt:
        print("Could not read original system prompt")
        return
    
    # The original prompt is all on one line with \n characters
    # Split it properly and find the sections
    prompt_parts = original_prompt.split('\\n')
    
    # Find the boundaries
    intro_start = 0
    tools_start = 0
    tool_rules_start = 0
    
    for i, part in enumerate(prompt_parts):
        if "## Available Tools" in part:
            tools_start = i
        elif "## Tool Calling Rules" in part:
            tool_rules_start = i
            break
    
    # Extract the three parts
    intro = '\\n'.join(prompt_parts[:tools_start])
    tools_section = '\\n'.join(prompt_parts[tools_start:tool_rules_start])
    tool_calling_rules = '\\n'.join(prompt_parts[tool_rules_start:])
    
    print(f"Intro section: {count_tokens_approximate(intro)} tokens")
    print(f"Tools section: {count_tokens_approximate(tools_section)} tokens")
    print(f"Tool calling rules: {count_tokens_approximate(tool_calling_rules)} tokens")
    
    # Calculate current token count and target
    current_tokens = count_tokens_approximate(intro + tools_section)
    target_tokens = 26000
    needed_tokens = target_tokens - current_tokens
    
    print(f"Current intro + tools tokens: {current_tokens}")
    print(f"Target tokens: {target_tokens}")
    print(f"Needed tokens: {needed_tokens}")
    
    # Read Paul Graham essays in order
    essay_files = [
        "paul graham's essays/pg_how_to_do_great_work.txt",
        "paul graham's essays/pg_how_to_work_hard.txt", 
        "paul graham's essays/pg_how_to_think_for_yourself.txt",
        "paul graham's essays/pg_putting_ideas_to_words.txt",
        "paul graham's essays/pg_the_lesson_to_unlearn.txt",
    ]
    
    filler_content = []
    total_filler_tokens = 0
    
    for essay_file in essay_files:
        if total_filler_tokens >= needed_tokens:
            break
            
        print(f"Processing {essay_file}...")
        essay_content = read_file_content(essay_file)
        
        if essay_content:
            # Extract clean content
            clean_content = extract_essay_content(essay_content)
            essay_tokens = count_tokens_approximate(clean_content)
            
            print(f"  Essay tokens: {essay_tokens}")
            
            # Add essay content directly without headers
            filler_content.append(f"\n\n{clean_content}")
            total_filler_tokens += essay_tokens
            
            print(f"  Total filler tokens so far: {total_filler_tokens}")
    
    # Create the filled prompt with new structure
    filled_prompt = intro + tools_section + ''.join(filler_content) + '\n\n' + tool_calling_rules
    
    # Final token count
    final_tokens = count_tokens_approximate(filled_prompt)
    print(f"\nFinal prompt tokens: {final_tokens}")
    print(f"Target reached: {final_tokens >= target_tokens}")
    
    # Write the filled prompt to a new file
    output_file = "300 tools system prompt restructured.txt"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(filled_prompt)
        print(f"\nRestructured prompt written to: {output_file}")
    except Exception as e:
        print(f"Error writing output file: {e}")
    
    return filled_prompt

def fill_system_prompt_before_tools():
    """
    Alternative function to fill the system prompt with Paul Graham essays
    BEFORE the Slack server tools but AFTER the "## Available Tools" heading.
    """
    
    # Read the original system prompt
    prompt_file = "300 tools system prompt.txt"
    original_prompt = read_file_content(prompt_file)
    
    if not original_prompt:
        print("Could not read original system prompt")
        return
    
    # The original prompt is all on one line with \n characters
    # Split it properly and find the sections
    prompt_parts = original_prompt.split('\\n')
    
    # Find the boundaries
    intro_start = 0
    tools_start = 0
    slack_server_start = 0
    tool_rules_start = 0
    
    for i, part in enumerate(prompt_parts):
        if "## Available Tools" in part:
            tools_start = i
        elif "### Server 'slack'" in part:
            slack_server_start = i
        elif "## Tool Calling Rules" in part:
            tool_rules_start = i
            break
    
    # Extract the four parts
    intro = '\\n'.join(prompt_parts[:tools_start])
    tools_heading = '\\n'.join(prompt_parts[tools_start:slack_server_start])
    slack_tools_section = '\\n'.join(prompt_parts[slack_server_start:tool_rules_start])
    tool_calling_rules = '\\n'.join(prompt_parts[tool_rules_start:])
    
    print(f"Intro section: {count_tokens_approximate(intro)} tokens")
    print(f"Tools heading: {count_tokens_approximate(tools_heading)} tokens")
    print(f"Slack tools section: {count_tokens_approximate(slack_tools_section)} tokens")
    print(f"Tool calling rules: {count_tokens_approximate(tool_calling_rules)} tokens")
    
    # Calculate current token count and target
    current_tokens = count_tokens_approximate(intro + tools_heading + slack_tools_section)
    target_tokens = 26000
    needed_tokens = target_tokens - current_tokens
    
    print(f"Current total tokens: {current_tokens}")
    print(f"Target tokens: {target_tokens}")
    print(f"Needed tokens: {needed_tokens}")
    
    # Read Paul Graham essays in order
    essay_files = [
        "paul graham's essays/pg_how_to_do_great_work.txt",
        "paul graham's essays/pg_how_to_work_hard.txt", 
        "paul graham's essays/pg_how_to_think_for_yourself.txt",
        "paul graham's essays/pg_putting_ideas_to_words.txt",
        "paul graham's essays/pg_the_lesson_to_unlearn.txt",
    ]
    
    filler_content = []
    total_filler_tokens = 0
    
    for essay_file in essay_files:
        if total_filler_tokens >= needed_tokens:
            break
            
        print(f"Processing {essay_file}...")
        essay_content = read_file_content(essay_file)
        
        if essay_content:
            # Extract clean content
            clean_content = extract_essay_content(essay_content)
            essay_tokens = count_tokens_approximate(clean_content)
            
            print(f"  Essay tokens: {essay_tokens}")
            
            # Add essay content directly without headers
            filler_content.append(f"\n\n{clean_content}")
            total_filler_tokens += essay_tokens
            
            print(f"  Total filler tokens so far: {total_filler_tokens}")
    
    # Create the filled prompt with essays inserted before Slack tools
    filled_prompt = intro + tools_heading + ''.join(filler_content) + '\n\n' + slack_tools_section + '\n\n' + tool_calling_rules
    
    # Final token count
    final_tokens = count_tokens_approximate(filled_prompt)
    print(f"\nFinal prompt tokens: {final_tokens}")
    print(f"Target reached: {final_tokens >= target_tokens}")
    
    # Write the filled prompt to a new file
    output_file = "300 tools system prompt essays before tools.txt"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(filled_prompt)
        print(f"\nEssays-before-tools prompt written to: {output_file}")
    except Exception as e:
        print(f"Error writing output file: {e}")
    
    return filled_prompt

if __name__ == "__main__":
    # Run both methods
    print("=== Running original method (essays after tools) ===")
    fill_system_prompt()
    
    print("\n" + "="*60 + "\n")
    
    print("=== Running alternative method (essays before tools) ===")
    fill_system_prompt_before_tools()
