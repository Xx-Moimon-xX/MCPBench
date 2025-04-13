def read_json(file_path):
    """
    Read a JSON file and return the content as a dictionary.
    """
    import json
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data