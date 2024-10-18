# Installation

The Illumio Reporter Library can be easily installed using pip, the Python package installer. Follow these steps to install the library:

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation Steps

1. Open your terminal or command prompt.

2. Run the following command to install the Illumio Reporter Library:

   ```
   pip install illumio-reporter-library
   ```

3. Verify the installation by running:

   ```python
   from illumio_reporter_library import IllumioReporter
   print(IllumioReporter.__version__)
   ```

   This should print the version number of the installed library.

## Installing from Source

If you prefer to install from source or want to contribute to the library:

1. Clone the repository:

   ```
   git clone https://github.com/your-org/illumio-reporter-library.git
   ```

2. Navigate to the cloned directory:

   ```
   cd illumio-reporter-library
   ```

3. Install the library in editable mode:

   ```
   pip install -e .
   ```

## Troubleshooting

If you encounter any issues during installation, please check our [Troubleshooting Guide](../troubleshooting/common_issues.md) or open an issue on our GitHub repository.
