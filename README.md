# Tool-Integrated Reasoning Framework

A unified inference framework for large language models with tool-integrated reasoning, supporting high concurrency and custom tools.

This framework provides a clean, modular implementation for tool-using agent inference, including three built-in tools (Search, Visit, Python) and reward calculation capabilities.

## Features

- **Unified Framework**: Support for multiple tools in a single inference pipeline
- **High Concurrency**: Efficient batch evaluation with configurable concurrency control
- **Custom Tools**: Easy integration of custom tools through YAML configuration
- **Multi-turn Conversations**: Support for complex multi-turn tool-using interactions
- **Flexible Evaluation**: Configurable reward calculation and evaluation metrics

## Structure

```
tool-reasoning-framework/
├── tools/                    # Tool implementations
│   ├── __init__.py
│   ├── base_tool.py          # Base class for all tools
│   ├── search_tool.py        # GoogleSearchTool - web search
│   ├── visit_tool.py         # VisitTool - webpage visiting
│   └── python_tool.py        # PythonTool - code execution
├── reward/                   # Reward calculation
│   ├── __init__.py
│   ├── base_reward_calculator.py
│   └── reward_calculator.py  # LLM-based semantic reward
├── evaluator/                # Evaluation pipeline
│   ├── __init__.py
│   ├── base_evaluator.py     # Main evaluation orchestrator
│   └── base_interaction.py   # Interaction management
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── run_evaluation.py     # Command-line evaluation script
│   ├── tool_loader.py        # Tool loading from YAML
│   ├── class_loader.py       # Dynamic class loading
│   └── format_time.py        # Time utilities
├── schemas/                  # Schema definitions
│   ├── __init__.py
│   └── tool_schemas.py      # OpenAI function tool schemas
├── core/                     # Core utilities
│   ├── __init__.py
│   └── rollout_trace.py     # Rollout tracing decorator
├── configs/                  # Tool configurations
│   ├── search_visit_tool_config.yaml
│   ├── python_tool_config.yaml
│   └── search_visit_python_tool_config.yaml
├── scripts/                  # Evaluation scripts
│   ├── run_search_visit_evaluation.sh
│   ├── run_python_evaluation.sh
│   └── run_search_visit_python_evaluation.sh
├── examples/                 # Usage examples
│   └── example_usage.py
├── data/                     # Evaluation datasets
│   ├── simpleqa.json
│   ├── aime25.json
│   ├── aime24.json
│   ├── math500.json
│   └── webinstruct.json
├── result_analysis/          # Result analysis tools
│   ├── calculate_pte.py     # PTE cost analysis
│   ├── analyze_timing.py     # Timing correlation analysis
│   └── README.md
├── requirements.txt
└── README.md
```

## Tools

### 1. Search Tool (GoogleSearchTool)
Performs web searches using Google Search API (via Serper API).

**Features:**
- Supports single or multiple queries
- Filters out HuggingFace datasets/spaces
- Returns formatted search results with titles, links, and snippets

### 2. Visit Tool (VisitTool)
Visits web pages and extracts relevant information based on user goals.

**Features:**
- Fetches webpage content via proxy
- Uses LLM to extract relevant information
- Blocks HuggingFace dataset/space URLs
- Returns structured information (evidence, summary)

### 3. Python Tool (PythonTool)
Executes Python code in a sandboxed environment.

**Features:**
- Executes Python code remotely via sandbox_fusion
- Returns stdout and stderr
- Supports timeout detection
- Maintains execution history

## Reward Calculation

The reward calculator (`GoogleandPythonRewardCalculator`) uses LLM-based semantic matching to evaluate the correctness of model outputs against ground truth answers.

**Features:**
- Extracts answers from model output using `<ANSWER>` tags
- Uses LLM to judge semantic correctness
- Returns a score between 0 and 1

## Installation

```bash
pip install -r requirements.txt
```

All dependencies including `sandbox_fusion` are included in `requirements.txt`.

## Environment Variables

Set the following environment variables:

```bash
# OpenAI API (for LLM calls)
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="http://your-api-url/v1"
export OPENAI_MODEL="gpt-oss"

# Search Tool Service
export SERPER_SERVICE_URL="http://your-search-service-url/serper"  # Your search service URL (FastAPI/Flask)

# Visit Tool Service
export JINA_SERVICE_URL="http://your-jina-service-url/jina"  # Your Jina service URL (FastAPI/Flask)
export VISIT_SUMMARY_API_KEY="your-api-key"  # API key for summary extraction model
export VISIT_SUMMARY_API_URL="http://your-api-url/v1"  # API URL for summary extraction
export VISIT_SUMMARY_MODEL="gpt-oss"  # Model name for summary extraction

# Python Tool Service
export PYTHON_SERVICE_URL="http://your-python-service-url"  # Your Python execution service URL (FastAPI/Flask)

# Stress Test Tools (No Summary)
export SEARCH_SERVICE_URL="http://your-search-service-url/serper_nosummary"  # Search service (no summary)
export JINA_SERVICE_URL="http://your-jina-service-url/jina_nosummary"  # Jina service (no summary)
```

## Setting Up Tool Services

Before running evaluations, you need to set up the tool services. We provide a FastAPI service for Search and Visit tools, and you can use SandboxFusion for Python execution.

### 1. Search & Visit Service (FastAPI)

We provide a ready-to-use FastAPI service for Search (Serper) and Visit (Jina) tools:

```bash
# Set your API keys
export SERPER_API_KEY="your-serper-api-key"  # Get from https://serper.dev
export JINA_API_KEY="your-jina-api-key"     # Get from https://jina.ai

# Run the service (default port 5003)
python services/fastapi_serper_jina.py

# Or with custom port and workers
export SERVICE_PORT=5003
export SERVICE_WORKERS=4
uvicorn services.fastapi_serper_jina:app --host 0.0.0.0 --port 5003 --workers 4
```

The service provides two endpoints:
- `POST /serper`: Google Search via Serper API
- `GET/POST /jina`: Webpage content extraction via Jina API

**Health Check**: `GET /health` to verify the service is running.

### 2. Python Execution Service (SandboxFusion)

For Python code execution, we recommend using [SandboxFusion](https://github.com/bytedance/SandboxFusion) by ByteDance.

**Quick Start with Docker** (recommended):

```bash
# Clone SandboxFusion
git clone https://github.com/bytedance/SandboxFusion.git
cd SandboxFusion

# Build and run with Docker
docker build -f ./scripts/Dockerfile.base -t code_sandbox:base .
sed -i '1s/.*/FROM code_sandbox:base/' ./scripts/Dockerfile.server
docker build -f ./scripts/Dockerfile.server -t code_sandbox:server .
docker run -d --rm --privileged -p 8080:8080 code_sandbox:server make run-online
```

**Manual Installation**:

```bash
# Follow the installation guide at:
# https://github.com/bytedance/SandboxFusion#installation
```

The service will be available at `http://localhost:8080` by default.

### 3. Configure Service URLs

After starting the services, update the environment variables in the evaluation scripts or set them globally:

```bash
# Search & Visit service (FastAPI)
export SERPER_SERVICE_URL="http://localhost:5003/serper"
export JINA_SERVICE_URL="http://localhost:5003/jina"

# Python service (SandboxFusion)
export PYTHON_SERVICE_URL="http://localhost:8080"
```

**Note**: The evaluation scripts use `localhost` by default. Modify the scripts if your services are running on different hosts or ports.

## Quick Start

### Using Evaluation Scripts

All evaluation scripts are located in the `scripts/` directory. Each script automatically configures the required tool service URLs. Simply navigate to the repository root and run the desired script.

#### 1. Search + Visit Evaluation

Evaluates using Search and Visit tools on SimpleQA dataset:

```bash
cd tool-reasoning-framework
bash scripts/run_search_visit_evaluation.sh
```

**Dataset**: `data/simpleqa.json`  
**Output**: `./results/simpleqa_<model_name>/`  
**Tools Used**: GoogleSearchTool, VisitTool (with summary extraction)

#### 2. Python Tool Evaluation

Evaluates using Python tool on AIME and Math500 datasets:

```bash
cd tool-reasoning-framework
bash scripts/run_python_evaluation.sh
```

**Datasets**:
- `data/aime25.json`
- `data/aime24.json`
- `data/math500.json`

**Output**: `./results/<dataset_name>_<model_name>/` for each dataset  
**Tools Used**: PythonTool

#### 3. Search + Visit + Python Evaluation

Evaluates using all three tools on WebInstruct dataset:

```bash
cd tool-reasoning-framework
bash scripts/run_search_visit_python_evaluation.sh
```

**Dataset**: `data/webinstruct.json`  
**Output**: `./results/webinstruct_<model_name>/`  
**Tools Used**: GoogleSearchTool, VisitTool (with summary extraction), PythonTool

#### 4. Stress Test (No Summary)

High-concurrency stress testing with Search + Visit tools (no summary extraction):

```bash
cd tool-reasoning-framework
bash scripts/run_stress_test_nosummary.sh
```

**Features:**
- No summary extraction for faster processing
- High concurrency support (up to 256 concurrent requests)
- Timing logs included in results
- Optimized for performance testing

**Output**: `./results/<dataset_name>_<model_name>_timing_log/`  
**Tools Used**: ESSearchToolNoSummary, VisitToolNoSummary

### Customizing Scripts

Edit the scripts in `scripts/` directory to customize:
- API configuration (API_KEY, API_URL, API_MODEL)
- Dataset paths
- Output directories
- Maximum turns and concurrency
- Resume from previous results

### Using Command-Line Interface

You can also use the command-line interface directly:

```bash
python -m utils.run_evaluation \
    --dataset-path /path/to/dataset.json \
    --output-dir ./results \
    --api-key "your-api-key" \
    --api-url "http://your-api-url/v1" \
    --api-model "gpt-oss" \
    --reward-calculator-class "reward.reward_calculator.GoogleandPythonRewardCalculator" \
    --tool-config ./configs/search_visit_tool_config.yaml \
    --max-assistant-turns 10 \
    --max-user-turns 99999 \
    --max-concurrent 10 \
    --verbose
```

### Programmatic Usage

```python
import asyncio
from evaluator.base_evaluator import BaseEvaluator
from reward.reward_calculator import GoogleandPythonRewardCalculator

async def main():
    reward_calculator = GoogleandPythonRewardCalculator()
    
    evaluator = BaseEvaluator(
        api_key="your-api-key",
        reward_calculator=reward_calculator,
        api_url="http://your-api-url/v1",
        api_model="gpt-oss",
        max_assistant_turns=10,
        max_user_turns=10,
    )
    
    input_data = {
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "reward_model": {"ground_truth": "Paris"},
        "extra_info": {
            "tools_kwargs": {
                "google_search_tool": {"create_kwargs": {}}
            }
        }
    }
    
    results = await evaluator.run_evaluation(
        dataset=[input_data],
        yaml_tool_path="configs/search_visit_tool_config.yaml",
        output_dir="./results",
        max_concurrent=1
    )
    
    print(f"Score: {results[0]['score']}")

asyncio.run(main())
```

## Tool Configuration

Tools are configured via YAML files. See examples in `configs/`:

```yaml
tools:
  - class_name: "tools.search_tool.GoogleSearchTool"
    config:
      type: "native"
    tool_schema:
      type: "function"
      function:
        name: "google_search_tool"
        description: "Performs web searches..."
        parameters:
          type: "object"
          properties:
            query:
              type: "array"
              items:
                type: "string"
          required: ["query"]
```

## Result Analysis

After running evaluations, you can analyze the results using scripts in `result_analysis/`:

### PTE Cost Analysis

Calculate PTE (Prefill-Token-Equivalent) cost metrics:

```bash
python result_analysis/calculate_pte.py \
    --jsonl-path ./results/dataset_model/eval_results.jsonl \
    --tokenizer-path /path/to/tokenizer \
    --gamma 0.00329 \
    --model-name "ModelName" \
    --output-csv output.csv
```

### Timing Analysis

Analyze timing correlations (requires timing logs from stress test):

```bash
cd tool-reasoning-framework
python result_analysis/analyze_timing.py \
    --input-file ./results/dataset_model_timing_log/model/eval_results.jsonl \
    --tokenizer-path /path/to/tokenizer \
    --gamma 0.000718 \
    --output-image latency_correlation.png
```

**Example: Analyzing sample timing data**

We provide a sample timing data file for testing:

```bash
cd tool-reasoning-framework
python result_analysis/analyze_timing.py \
    --input-file ./result_analysis/example_timing_data.jsonl \
    --tokenizer-path /path/to/tokenizer \
    --gamma 0.000718 \
    --output-image ./result_analysis/example_latency_correlation.png \
    --max-lines 100
```

This will generate correlation analysis plots showing the relationship between per-step latency, tokens, and PTE metrics.

See `result_analysis/README.md` for detailed usage.

## Key Components

### BaseEvaluator
The main evaluation class that orchestrates the inference pipeline:
- Manages tool execution
- Handles multi-turn conversations
- Calculates rewards
- Supports batch evaluation with concurrency control
- Supports resume from previous results

### BaseTool
Base class for all tools. Each tool must implement:
- `create()`: Create a tool instance
- `execute()`: Execute the tool with given parameters
- `calc_reward()`: Calculate tool-specific reward
- `release()`: Release tool instance

### BaseRewardCalculator
Base class for reward calculation. Must implement:
- `extract_output()`: Extract answer from model output
- `_verify_correction()`: Verify correctness against ground truth

## Output Format

Evaluation results are saved in JSONL format with the following structure:

```json
{
  "input": {...},
  "messages": [...],
  "extracted_output": "...",
  "ground_truth": "...",
  "score": 1.0,
  "reached_max_turns": false,
  "turn_record": {...},
  "success": true,
  "time_stats": {
    "summary": {
      "total_llm_inference_seconds": 10.5,
      "total_tool_execution_seconds": 5.2,
      "total_wall_time_seconds": 15.7
    },
    "timeline": [...]
  },
  "evaluation_config": {...}
}
```

## License

See LICENSE file for details.
