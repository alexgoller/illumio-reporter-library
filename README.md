# Illumio Reporter Library

A Python library for generating comprehensive reports from Illumio PCE data with AI-powered insights.

## Features

- **Workload Analysis**
  - Network distribution analysis
  - OS distribution statistics
  - Enforcement mode analysis
  - Online/offline status tracking

- **Report Generation**
  - PDF report generation using ReportLab
  - Customizable templates and styling
  - Dynamic content generation
  - Support for graphs and tables

- **AI Integration**
  - AI-powered insights and recommendations
  - Support for multiple AI providers (Anthropic, OpenAI, Ollama)
  - Customizable analysis parameters

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/illumio-reporter-library

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
```

## Quick Start

```python
from src.report_generator import ReportGenerator
from src.data_processors import WorkloadProcessor
from src.ai_models import AnthropicModel

# Initialize components
model = AnthropicModel(api_key="your-api-key", model="claude-3-5-sonnet-20240620")
report = ReportGenerator("output.pdf")

# Add content
report.add_title("Illumio Environment Report")
report.add_section("Workload Analysis")

# Generate workload analysis
workload_data = {...}  # Your workload data
processor = WorkloadProcessor(workload_data)
report.add_table(processor.get_os_summary())
report.add_network_workloads(processor.get_workloads_by_network())

# Save report
report.save()
```

## Project Structure

## Documentation

- [API Reference](docs/API.md)
- [Configuration Guide](docs/CONFIGURATION.md)
- [Development Guide](docs/DEVELOPMENT.md)
- [Examples](docs/EXAMPLES.md)

## Configuration

The library can be configured using environment variables or a configuration file:

```env
ANTHROPIC_API_KEY=your-api-key
OPENAI_API_KEY=your-api-key
REPORT_DEFAULT_STYLE=default
```

See the [Configuration Guide](docs/CONFIGURATION.md) for more details.

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](docs/CONTRIBUTING.md) for details on how to submit pull requests, report issues, and contribute to the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
