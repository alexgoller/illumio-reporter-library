# API Reference

## ReportGenerator

The main class for generating reports.

### Methods

#### `__init__(output_file, color_scheme=None, logo_path=None, header_text=None, footer_text=None)`

Initialize a new report generator.

**Parameters:**
- `output_file` (str): Path where the PDF report will be saved
- `color_scheme` (dict, optional): Custom color scheme for the report
- `logo_path` (str, optional): Path to logo image
- `header_text` (str, optional): Custom header text
- `footer_text` (str, optional): Custom footer text

#### `add_network_workloads(workloads, include_details=True)`

Add a section showing workloads grouped by network.

**Parameters:**
- `workloads` (list): List of workload dictionaries
- `include_details` (bool): Whether to include additional details

[Continue with other methods...] 