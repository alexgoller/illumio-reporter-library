# Configuration

After installing the Illumio Reporter Library, you need to configure it to work with your environment. This guide will walk you through the necessary configuration steps.

## AI Model Configuration

The library supports multiple AI models for analysis. You can choose between Anthropic, OpenAI, or Ollama models.

### Anthropic API Key

1. Sign up for an account at [Anthropic](https://www.anthropic.com) if you haven't already.
2. Obtain your API key from the Anthropic dashboard.
3. Set the environment variable:

   ```
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

### OpenAI API Key (Optional)

If you plan to use OpenAI models:

1. Sign up for an account at [OpenAI](https://www.openai.com) if you haven't already.
2. Obtain your API key from the OpenAI dashboard.
3. Set the environment variable:

   ```
   export OPENAI_API_KEY=your_api_key_here
   ```

### Ollama (Local Model)

If you prefer to use a local model with Ollama:

1. Install Ollama by following the instructions at [Ollama's official website](https://ollama.ai/).
2. No API key is required for Ollama, as it runs locally on your machine.

## Illumio Configuration

To fetch data from Illumio, you need to configure the connection:

1. Obtain your Illumio PCE (Policy Compute Engine) URL and API credentials from your Illumio administrator.
2. Set the following environment variables:

   ```
   export ILLUMIO_HOSTNAME=your-pce-hostname
   export ILLUMIO_PORT=your-pce-port
   export ILLUMIO_ORG_ID=your-org-id
   export ILLUMIO_API_KEY_ID=your-api-key-id
   export ILLUMIO_API_KEY_SECRET=your-api-key-secret
   export ILLUMIO_IGNORE_TLS=True  # Set to False if you want to verify TLS
   ```

## Choosing an AI Model

When initializing your AI advisor, you can choose which model to use:

```python
from illumio_reporter_library.ai_models import AnthropicModel, OpenAIModel, OllamaModel
from illumio_reporter_library.ai_advisor import AIAdvisor

# For Anthropic
model = AnthropicModel(api_key=os.getenv('ANTHROPIC_API_KEY'), model="claude-3-5-sonnet-20240620")
# For OpenAI
model = OpenAIModel(api_key=os.getenv('OPENAI_API_KEY'), model="gpt-4-turbo-preview")
# For Ollama (local model)
model = OllamaModel(model="llama2")
ai_advisor = AIAdvisor(model)
```

## Verifying Configuration

To verify your configuration, run the following Python script:

```python
from illumio_reporter_library import IllumioReporter

reporter = IllumioReporter()
reporter.verify_configuration()
```

This script will check if your configuration is valid and print the result.

