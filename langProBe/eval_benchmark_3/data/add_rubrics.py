import json

# The rubric data to add
rubric_data = {
    "criterion": "Prompt Adherence",
    "description": "Evaluates how well the response directly addresses and fulfills the requirements stated in the original prompt or question. This includes answering all parts of multi-part questions, following specific instructions, and staying within the scope of what was asked.",
    "weight": 0.4,
    "evaluation_scale": [
        {
            "score": 1,
            "condition": "Does not address the question or provides irrelevant information"
        },
        {
            "score": 2,
            "condition": "Partially addresses the question but misses key aspects"
        },
        {
            "score": 3,
            "condition": "Fully and adequately addresses all aspects of the question"
        }
    ]
}

content_accuracy = {
    "criterion": "Content Accuracy",
    "description": "Assesses the degree to which the response content aligns with the expected or reference response. This measures semantic similarity, key information coverage, and whether the response captures the essential points and themes present in the expected answer.",
    "weight": 0.4,
    "evaluation_scale": [
        {
            "score": 1,
            "condition": "Response is substantially different from expected content"
        },
        {
            "score": 2,
            "condition": "Response has moderate similarity to expected content"
        },
        {
            "score": 3,
            "condition": "Response closely matches the expected content"
        }
    ]
}

factual_accuracy = {
    "criterion": "Factual Accuracy",
    "description": "Measures the objective correctness of factual claims, data, statistics, dates, names, and verifiable information presented in the response. This criterion focuses on whether the information provided is factually true and can be verified through reliable sources.",
    "weight": 0.1,
    "evaluation_scale": [
        {
            "score": 1,
            "condition": "Response contains factual errors or misinformation"
        },
        {
            "score": 2,
            "condition": "Response has mostly correct facts with minor inaccuracies"
        },
        {
            "score": 3,
            "condition": "Response is completely factually accurate"
        }
    ]
}

summarisation = {
    "criterion": "Summarisation",
    "description": "Evaluates the quality of summarization skills, including the ability to condense information while maintaining clarity, coherence, and completeness. This assesses whether the response effectively distills key points into a concise, well-organized format that captures the essence of the source material.",
    "weight": 0.1,
    "evaluation_scale": [
        {
            "score": 1,
            "condition": "Response is not a summary or is unclear and confusing"
        },
        {
            "score": 2,
            "condition": "Response provides a basic summary but lacks clarity or completeness"
        },
        {
            "score": 3,
            "condition": "Response provides a clear, comprehensive, and well-structured summary"
        }
    ]
}

# Complete rubric data
complete_rubrics = [rubric_data, content_accuracy, factual_accuracy, summarisation]

# Read the JSONL file
input_file = "slack_50_genprompt2.jsonl"
output_file = "slack_50_genprompt2_with_rubrics.jsonl"

updated_entries = []

with open(input_file, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if line:
            try:
                entry = json.loads(line)
                
                # Add or update the Rubrics field
                entry["Rubrics"] = complete_rubrics
                
                updated_entries.append(entry)
                print(f"Processed entry {entry.get('unique_id', line_num)}")
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue

# Write the updated entries back to a new file
with open(output_file, 'w', encoding='utf-8') as f:
    for entry in updated_entries:
        json.dump(entry, f, ensure_ascii=False)
        f.write('\n')

print(f"\nSuccessfully processed {len(updated_entries)} entries")
print(f"Updated file saved as: {output_file}")

# Also update the original file if desired
print("\nUpdating original file...")
with open(input_file, 'w', encoding='utf-8') as f:
    for entry in updated_entries:
        json.dump(entry, f, ensure_ascii=False)
        f.write('\n')

print(f"Original file {input_file} has been updated with rubric data")
