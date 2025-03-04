# Quick Start Guide

This guide will help you quickly get started with the Illumio Reporter Library. We'll walk through a basic example of generating a network security report.

## Prerequisites

Ensure you have completed the [Installation](installation.md) and [Configuration](configuration.md) steps.

## Basic Usage

Here's a simple example to generate a network security report:

```python
from illumio_reporter_library import IllumioReporter

# Initialize the reporter with your Illumio API credentials
reporter = IllumioReporter(api_key='your_api_key', api_secret='your_api_secret')

# Fetch and process network tport()

# Print the report
print(report)
```

## Customization

You can customize the report generation process by modifying the parameters in the `generate_report` method.

## Choosing an AI Model

You can customize which AI model to use for generating recommendations:

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

## Status Box

The generated report includes a status box with important metadata:

- PCE Information: The hostname and port of the Illumio PCE used for the report.
- Time Range: The time period covered by the report data.
- Creation Time: The exact time when the report was generated.

This information helps in tracking and referencing reports, especially when multiple reports are generated over time or from different PCEs.

## Next Steps

- Explore the [Core Concepts](../core_concepts/data_fetching.md) to understand how the library works
- Learn about advanced [Features](../features/traffic_analysis.md) for in-depth analysis
- Check out the [Customization](../customization/report_styling.md) options to tailor the reports to your needs
