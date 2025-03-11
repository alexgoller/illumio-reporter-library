# Configuration Guide

This guide covers all configuration options for the Illumio Reporter Library.

## Environment Variables

### Required Variables

```env
# AI Model Configuration
ANTHROPIC_API_KEY=your-anthropic-api-key    # Required for Claude AI integration
OPENAI_API_KEY=your-openai-api-key          # Required for GPT integration

# Illumio PCE Configuration
ILLUMIO_PCE_HOST=pce.your-domain.com        # PCE hostname
ILLUMIO_API_KEY=your-api-key                # PCE API key
ILLUMIO_API_SECRET=your-api-secret          # PCE API secret
```

### Optional Variables

```env
# Report Styling
REPORT_STYLE_TEMPLATE=default               # Report style template (default, modern, minimal)
REPORT_COLOR_SCHEME=default                 # Color scheme (default, dark, light)
REPORT_LOGO_PATH=/path/to/logo.png         # Custom logo path
REPORT_FONT_FAMILY=Helvetica               # Default font family

# AI Configuration
AI_MODEL_PROVIDER=anthropic                 # Default AI provider (anthropic, openai, ollama)
AI_MODEL_NAME=claude-3-sonnet              # Specific model to use
AI_TEMPERATURE=0.7                         # Model temperature (0.0-1.0)

# Performance Settings
MAX_WORKERS=4                              # Maximum parallel workers
CACHE_ENABLED=true                         # Enable data caching
CACHE_TTL=3600                            # Cache time-to-live in seconds
```

## Color Schemes

### Default Color Scheme

```python
DEFAULT_COLOR_SCHEME = {
    'title': colors.black,
    'section': colors.black,
    'text': colors.black,
    'table_header_bg': colors.lightgrey,
    'table_header_text': colors.black,
    'table_body_bg': colors.white,
    'table_body_text': colors.black,
    'table_grid': colors.black
}
```

### Custom Color Schemes

You can define custom color schemes when initializing the ReportGenerator:

```python
custom_colors = {
    'title': colors.HexColor('#1a73e8'),
    'section': colors.HexColor('#202124'),
    'text': colors.HexColor('#3c4043'),
    'table_header_bg': colors.HexColor('#e8f0fe'),
    'table_header_text': colors.HexColor('#1a73e8'),
    'table_body_bg': colors.white,
    'table_body_text': colors.HexColor('#3c4043'),
    'table_grid': colors.HexColor('#dadce0')
}

report = ReportGenerator("output.pdf", color_scheme=custom_colors)
```

## Report Templates

### Available Templates

1. **Default Template**
   - Standard professional layout
   - Left-aligned text
   - Standard margins

2. **Modern Template**
   - Contemporary design
   - Wider margins
   - Custom fonts
   - Enhanced spacing

3. **Minimal Template**
   - Clean, minimalist design
   - Reduced decorative elements
   - Focus on content

### Template Configuration

```python
from src.report_generator import ReportGenerator

# Using built-in template
report = ReportGenerator(
    "output.pdf",
    template="modern",
    header_text="Confidential Report",
    footer_text="Generated on {date}"
)

# Custom template settings
custom_template = {
    'page_size': 'letter',
    'margins': {
        'top': 1.0,
        'bottom': 1.0,
        'left': 1.0,
        'right': 1.0
    },
    'spacing': {
        'before_section': 12,
        'after_section': 6,
        'between_paragraphs': 6
    },
    'fonts': {
        'title': 'Helvetica-Bold',
        'heading': 'Helvetica-Bold',
        'body': 'Helvetica'
    }
}

report = ReportGenerator("output.pdf", template=custom_template)
```

## AI Model Configuration

### Anthropic (Claude) Configuration

```python
from src.ai_models import AnthropicModel

model = AnthropicModel(
    api_key=os.getenv('ANTHROPIC_API_KEY'),
    model="claude-3-sonnet-20240620",
    temperature=0.7,
    max_tokens=2000
)
```

### OpenAI Configuration

```python
from src.ai_models import OpenAIModel

model = OpenAIModel(
    api_key=os.getenv('OPENAI_API_KEY'),
    model="gpt-4-turbo-preview",
    temperature=0.7,
    max_tokens=2000
)
```

## Performance Tuning

### Caching Configuration

```python
from src.utils.cache import Cache

# Configure caching
Cache.configure(
    enabled=True,
    ttl=3600,  # Cache for 1 hour
    max_size=1000,  # Maximum cache entries
    backend='memory'  # or 'redis', 'filesystem'
)
```

### Parallel Processing

```python
from src.utils.parallel import configure_workers

# Configure parallel processing
configure_workers(
    max_workers=4,
    thread_name_prefix='reporter',
    timeout=300  # 5 minutes timeout
)
```

## Logging Configuration

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reporter.log'),
        logging.StreamHandler()
    ]
)
```

## Configuration File

You can also use a YAML configuration file:

```yaml
# config.yaml
report:
  style_template: modern
  color_scheme: default
  logo_path: /path/to/logo.png
  
ai:
  provider: anthropic
  model: claude-3-sonnet
  temperature: 0.7
  
performance:
  max_workers: 4
  cache_enabled: true
  cache_ttl: 3600
  
logging:
  level: INFO
  file: reporter.log
```

Load configuration from file:

```python
from src.utils.config import load_config

config = load_config('config.yaml')
report = ReportGenerator("output.pdf", **config['report'])
```

## Best Practices

1. **Environment Variables**
   - Use environment variables for sensitive information
   - Consider using a `.env` file for local development
   - Never commit API keys to version control

2. **Color Schemes**
   - Test color schemes for accessibility
   - Maintain consistent branding
   - Consider color-blind friendly options

3. **Templates**
   - Start with a built-in template
   - Customize gradually as needed
   - Maintain consistent spacing and alignment

4. **Performance**
   - Enable caching for repeated operations
   - Configure parallel processing based on available resources
   - Monitor memory usage with large datasets

5. **Logging**
   - Configure appropriate log levels
   - Implement log rotation
   - Include relevant context in log messages 