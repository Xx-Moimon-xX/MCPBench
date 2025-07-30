# MCPBench (Custom Notes)

## Important: Required local.env File

To run the evaluation scripts in this directory, you must create a `local.env` file in the `external/MCPBench` folder. This file should contain the required environment variables (such as API keys and configuration settings) needed for the scripts to function.

The `local.env` file is excluded from version control for security reasons (see `.gitignore`).

### Example usage

```bash
cp local.env.example local.env
# Edit local.env to add your API keys and settings
```

### What to include in local.env
- API keys for any services you use (e.g., OpenAI, DeepSeek, etc.)
- Any other environment variables referenced by the evaluation scripts

**Do not share your local.env file or commit it to git.**

## AWS Bedrock Integration Usage

MCPBench now supports AWS Bedrock models in addition to OpenAI and Anthropic. To use Bedrock models, follow these steps:

1. **Set AWS Credentials as Environment Variables**
   
   Add the following to your `local.env` (or export them in your shell):
   ```env
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_REGION=us-east-1
   ```
   
   These credentials are required for authenticating with AWS Bedrock. The region defaults to `us-east-1` if not set.

2. **Use Bedrock Model Names with the `bedrock/` Prefix**
   
   When specifying a model, use the following format:
   - `bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0`
   - `bedrock/anthropic.claude-3-haiku-20240307-v1:0`
   - `bedrock/meta.llama3-1-8b-instruct-v1:0`

   The integration will automatically handle message conversion and credential management for Bedrock models.

**Note:**
- Bedrock support is fully compatible with the existing OpenAI and Anthropic integrations.
- Make sure you have `boto3` installed (see `requirements.txt`).
- Do not share your AWS credentials.
