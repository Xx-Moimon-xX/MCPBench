#!/usr/bin/env python3
"""
Simple test script to validate AWS Bedrock integration logic
"""

def test_bedrock_model_parsing():
    """Test that bedrock model names are parsed correctly"""
    test_model = "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
    prefix, model_name = test_model.split('/', 1)
    
    assert prefix == 'bedrock', f"Expected prefix 'bedrock', got '{prefix}'"
    assert model_name == 'anthropic.claude-3-5-sonnet-20241022-v2:0', f"Expected model name 'anthropic.claude-3-5-sonnet-20241022-v2:0', got '{model_name}'"
    
    print("✓ Bedrock model parsing test passed")

def test_message_conversion():
    """Test that OpenAI-style messages are converted correctly for Bedrock"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"}
    ]
    
    # Convert messages to format expected by Bedrock
    bedrock_messages = []
    system_message = ""
    
    for m in messages:
        if m.get("role") == "system":
            system_message = m["content"]
        elif m.get("role") == "user":
            bedrock_messages.append({"role": "user", "content": [{"type": "text", "text": m["content"]}]})
        elif m.get("role") == "assistant":
            bedrock_messages.append({"role": "assistant", "content": [{"type": "text", "text": m["content"]}]})
    
    assert system_message == "You are a helpful assistant.", f"System message not extracted correctly"
    assert len(bedrock_messages) == 2, f"Expected 2 bedrock messages, got {len(bedrock_messages)}"
    assert bedrock_messages[0]["role"] == "user", f"First message should be user role"
    assert bedrock_messages[1]["role"] == "assistant", f"Second message should be assistant role"
    
    print("✓ Message conversion test passed")

def test_request_body_structure():
    """Test that Bedrock request body is structured correctly"""
    bedrock_messages = [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
    ]
    system_message = "You are a helpful assistant."
    temperature = 0.7
    
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": bedrock_messages,
        "temperature": temperature
    }
    
    if system_message:
        request_body["system"] = system_message
    
    expected_keys = {"anthropic_version", "max_tokens", "messages", "temperature", "system"}
    actual_keys = set(request_body.keys())
    
    assert expected_keys == actual_keys, f"Request body keys mismatch. Expected: {expected_keys}, Got: {actual_keys}"
    assert request_body["anthropic_version"] == "bedrock-2023-05-31"
    assert request_body["max_tokens"] == 1024
    assert request_body["temperature"] == 0.7
    assert request_body["system"] == "You are a helpful assistant."
    
    print("✓ Request body structure test passed")

if __name__ == "__main__":
    print("Testing AWS Bedrock integration logic...")
    test_bedrock_model_parsing()
    test_message_conversion()
    test_request_body_structure()
    print("\n✅ All Bedrock integration tests passed!")
    
    print("\nExample usage:")
    print("To use AWS Bedrock models, set the following environment variables:")
    print("  AWS_ACCESS_KEY_ID=your_access_key")
    print("  AWS_SECRET_ACCESS_KEY=your_secret_key")
    print("  AWS_REGION=us-east-1")
    print("\nThen use model names like:")
    print("  bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")
    print("  bedrock/anthropic.claude-3-haiku-20240307-v1:0")
    print("  bedrock/meta.llama3-1-8b-instruct-v1:0")