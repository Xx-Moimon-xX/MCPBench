import json

"""
This file has the read_json and read_jsonl functions.
"""

def read_json(file_path):
    """
    Read a JSON file and return the content as a dictionary.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def read_jsonl(file_path):
    """
    Read a JSONL file and return the content as a list of dictionaries.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_data = json.loads(line)
            data.append(test_data)
    return data