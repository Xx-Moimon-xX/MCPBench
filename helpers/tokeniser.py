import anthropic
import json
import os
from pathlib import Path

# Load environment variables from local.env
def load_env_file(env_file_path):
    env_vars = {}
    if os.path.exists(env_file_path):
        with open(env_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = value
    return env_vars

# Load environment variables
env_vars = load_env_file('../local.env')
api_key = env_vars.get('ANTHROPIC_API_KEY')

if not api_key:
    print("Error: ANTHROPIC_API_KEY not found in local.env")
    exit(1)

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=api_key)

# Read and parse the tokens_to_check.txt file
def read_tokens_file(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
            # Parse the JSON content
            data = json.loads(content)
            return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None

# Read the tokens file
tokens_data = read_tokens_file('tokens_to_check.txt')

if tokens_data:
    # Extract the messages from the data
    messages = tokens_data.get('messages', [])
    
    if messages:
        print(f"Found {len(messages)} messages to analyze")
        print("=" * 50)
        
        # Separate system messages from regular messages
        system_messages = []
        regular_messages = []
        
        for message in messages:
            if message.get('role') == 'system':
                system_messages.append(message.get('content', ''))
            elif message.get('role') in ['user', 'assistant']:
                regular_messages.append(message)
            elif message.get('role') == 'tool':
                # Convert tool messages to user messages
                tool_message = {
                    'role': 'user',
                    'content': message.get('content', '')
                }
                regular_messages.append(tool_message)
                print(f"Converted tool message to user message")
            else:
                print(f"Skipping message with role '{message.get('role')}' (not supported for token counting)")
        
        # Combine system messages into one string
        system_content = '\n'.join(system_messages) if system_messages else None
        
        print(f"System messages: {len(system_messages)}")
        print(f"Regular messages: {len(regular_messages)}")
        
        if system_content:
            print(f"System content preview: {system_content[:200]}...")
        
        # Count tokens for the extracted messages
        try:
            # Prepare parameters for token counting
            params = {
                "model": "claude-opus-4-1-20250805",
                "messages": regular_messages
            }
            
            # Add system parameter if we have system messages
            if system_content:
                params["system"] = system_content
            
            response = client.messages.count_tokens(**params)
            
            print("Token Count Results:")
            print(f"Input tokens: {response.input_tokens}")
            print(f"Response: {response.json()}")
            
        except Exception as e:
            print(f"Error counting tokens: {e}")
    else:
        print("No messages found in the file")
else:
    print("Failed to read tokens file")

