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
