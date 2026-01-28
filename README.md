# Jupyter Code Executor MCP Server

[![Python >= 3.11](https://img.shields.io/badge/python-%3E%3D3.11-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/jupyter-code-executor-mcp-server.svg)](https://pypi.org/project/jupyter-code-executor-mcp-server/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP](https://img.shields.io/badge/MCP-1.23.1-green.svg)](https://pypi.org/project/mcp/)

A powerful **Model Context Protocol (MCP)** server that enables AI assistants to execute code in multiple programming languages through Jupyter kernels. This server provides **stateful** and **stateless** code execution capabilities with session management, automatic cleanup, and comprehensive data analysis tools.

## ✨ Key Features

### 🚀 Multi-Language Code Execution
- **Universal Kernel Support**: Execute code in any language with an installed Jupyter kernel (Python, R, Julia, etc.)
- **Dual Execution Modes**:
  - **Stateful Sessions**: Maintain variable context across multiple executions within the same session
  - **Stateless Execution**: One-off code execution without persistent context


### 🎯 Built-in Data Analysis Capabilities
- **Pre-configured Environment**: Includes essential data science libraries (Pandas, NumPy, Scikit-learn, Seaborn, etc.)
- **Expert Prompt Templates**: Built-in data analyst prompt with best practices for EDA, ML, and visualization
- **Professional Visualizations**: Seaborn and Matplotlib integration for publication-quality charts


## 📋 Prerequisites

- **Python**: >= 3.11
- **uv**: Modern Python package installer (recommended)
- **Jupyter**: Jupyter kernels for your target languages

## 🔧 Installation

### Using uv (Recommended)

```bash
# Install the package
uv pip install jupyter-code-executor-mcp-server

# Or install from source
git clone https://github.com/twn39/jupyter-code-executor-mcp-server.git
cd jupyter-code-executor-mcp-server
uv sync
uv run jupyter-code-executor-mcp-server
```

### Using pip

```bash
pip install jupyter-code-executor-mcp-server
```

## 🚀 Quick Start

### Starting the Server

```bash
# Start with default settings
uv run jupyter-code-executor-mcp-server

# Start with custom configuration
uv run jupyter-code-executor-mcp-server \
  --port 8080 \
  --data_dir ~/my-data \
  --output_dir ~/my-output \
  --session-timeout 1200
```

### Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--port` | `-p` | Server listening port | `5010` |
| `--data_dir` | `-d` | User data directory | `~/data` |
| `--output_dir` | `-o` | Code output directory | `~/output` |
| `--session-timeout` | `-st` | Session timeout (seconds) | `600` |

### Testing the Server

Use the MCP Inspector to interact with the server:

```bash
pnpx @modelcontextprotocol/inspector
```

## 📚 Available MCP Tools

### 1. `list_kernels`
List all installed and available Jupyter kernels on your system.

**Returns**: String containing kernel names and display names

**Example Response**:
```
Available Jupyter Kernels:
Format: [Kernel Name] - [Display Name]
----------------------------------------
- python3: Python 3
- ir: R
- julia-1.9: Julia 1.9
```

### 2. `execute_code`
Execute code in a specified Jupyter kernel with optional session persistence.

**Parameters**:
- `code` (string): The code to execute
- `kernel_name` (string): Target kernel name (use `list_kernels` to find available kernels)
- `session_id` (string, optional): Session identifier for stateful execution

**Returns**: Execution result including kernel name, exit code, and output

**Stateless Example** (Python):
```python
# Execute code without session persistence
code = """
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df.describe())
"""
```

**Stateful Example** (Python with session):
```python
# First execution - create variables
session_id = "data-analysis-001"
code1 = """
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
"""

# Second execution - reuse variables from session
code2 = """
# df is still available from previous execution
print(df.describe())
print(f"Shape: {df.shape}")
"""
```

**R Example**:
```r
# Execute R code
code = """
data <- data.frame(x = 1:10, y = rnorm(10))
summary(data)
plot(data$x, data$y)
"""
kernel_name = "ir"
```


## 📖 Available Prompts

### `data_analyst_prompt`
A comprehensive prompt template for AI-powered data analysis featuring:

- **Statistical Analysis**: EDA, hypothesis testing, attribution analysis, time series
- **Machine Learning**: Supervised/unsupervised learning, model evaluation, hyperparameter tuning
- **Deep Learning**: Neural networks, CNNs, RNNs/LSTMs with TensorFlow/PyTorch
- **Professional Visualization**: Publication-quality charts with Seaborn/Matplotlib
- **Iterative Workflow**: Think → Act → Observe → Interpret cycle

This prompt transforms your AI assistant into an expert data scientist with code execution capabilities.


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 🙏 Acknowledgments

- Built with [FastMCP](https://github.com/modelcontextprotocol/python-sdk)
- Powered by [Jupyter](https://jupyter.org/)
- Execution engine: [AutoGen](https://github.com/microsoft/autogen)

---

**Note**: This server requires properly configured Jupyter kernels for the languages you want to execute. Install kernels using commands like `pip install ipykernel` (Python), `install.packages('IRkernel')` (R), or following the respective kernel installation guides.