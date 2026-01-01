# Code Structure Summary

This document summarizes the code organization in the `tool-reasoning-framework` directory.

## Overview

This repository provides a unified inference framework for large language models with tool-integrated reasoning, supporting high concurrency and custom tools.

## Directory Structure

```
tool-reasoning-framework/
├── tools/                    # Tool implementations
│   ├── __init__.py          # Exports: BaseTool, GoogleSearchTool, VisitTool, PythonTool
│   ├── base_tool.py         # Base class for all tools
│   ├── search_tool.py       # GoogleSearchTool - web search functionality
│   ├── visit_tool.py        # VisitTool - webpage visiting and extraction
│   └── python_tool.py      # PythonTool - Python code execution
│
├── reward/                  # Reward calculation
│   ├── __init__.py          # Exports: BaseRewardCalculator, GoogleandPythonRewardCalculator
│   ├── base_reward_calculator.py  # Base class for reward calculation
│   └── reward_calculator.py       # LLM-based semantic reward calculator
│
├── evaluator/               # Evaluation pipeline
│   ├── __init__.py          # Exports: BaseEvaluator, BaseInteraction
│   ├── base_evaluator.py    # Main evaluation orchestrator
│   └── base_interaction.py  # Base class for interaction management
│
├── utils/                   # Utility functions
│   ├── __init__.py          # Exports utility functions
│   ├── tool_loader.py       # Tool loading from YAML configs
│   ├── class_loader.py      # Dynamic class loading
│   └── format_time.py       # Time formatting utilities
│
├── schemas/                 # Schema definitions
│   ├── __init__.py          # Exports schema classes
│   └── tool_schemas.py      # OpenAI function tool schemas
│
├── core/                    # Core utilities
│   ├── __init__.py          # Exports: rollout_trace_op
│   └── rollout_trace.py    # Rollout tracing decorator (simplified)
│
├── configs/                 # Configuration examples
│   └── example_tool_config.yaml  # Example tool configuration
│
├── examples/                # Usage examples
│   └── example_usage.py    # Basic usage example
│
├── services/                # Tool service implementations
│   ├── __init__.py
│   └── fastapi_serper_jina.py  # FastAPI service for Search & Visit tools
│
├── scripts/                 # Evaluation scripts
│   ├── run_search_visit_evaluation.sh
│   ├── run_python_evaluation.sh
│   ├── run_search_visit_python_evaluation.sh
│   └── run_stress_test_nosummary.sh
│
├── data/                    # Evaluation datasets
│   ├── simpleqa.json
│   ├── aime24.json
│   ├── aime25.json
│   ├── math500.json
│   └── webinstruct.json
│
├── result_analysis/         # Result analysis scripts
│   ├── __init__.py
│   ├── calculate_pte.py     # PTE cost analysis
│   ├── analyze_timing.py    # Timing analysis
│   ├── example_timing_data.jsonl
│   └── README.md
│
├── requirements.txt         # Python dependencies
├── README.md                # Main documentation
└── STRUCTURE.md             # This file
```

## Key Components

### Tools

1. **GoogleSearchTool** (`tools/search_tool.py`)
   - Performs web searches via Serper API
   - Filters HuggingFace datasets/spaces
   - Returns formatted search results

2. **VisitTool** (`tools/visit_tool.py`)
   - Visits web pages via proxy
   - Extracts relevant information using LLM
   - Blocks HuggingFace URLs
   - Returns structured evidence and summary

3. **PythonTool** (`tools/python_tool.py`)
   - Executes Python code in sandbox
   - Uses sandbox_fusion for execution
   - Returns stdout/stderr with timeout detection

### Reward Calculation

**GoogleandPythonRewardCalculator** (`reward/reward_calculator.py`)
- Extracts answers from model output using `<ANSWER>` tags
- Uses LLM to judge semantic correctness
- Returns score between 0 and 1

### Evaluation Pipeline

**BaseEvaluator** (`evaluator/base_evaluator.py`)
- Orchestrates multi-turn conversations
- Manages tool execution
- Calculates rewards
- Supports batch evaluation with concurrency control
- Handles resume from previous results

## Design Principles

1. **Unified Framework**: Single framework supporting multiple tools and configurations
2. **High Concurrency**: Efficient batch evaluation with configurable concurrency
3. **Modular Design**: Clean separation of concerns with clear interfaces
4. **Easy Extension**: Simple integration of custom tools via YAML configuration
5. **Comprehensive Documentation**: Detailed README, examples, and scripts

## Dependencies

Core dependencies are listed in `requirements.txt`. Key dependencies:
- `openai`: For LLM API calls
- `httpx`: For async HTTP requests
- `pydantic`: For schema validation
- `pyyaml`: For configuration loading
- `transformers`: For tokenizer support (optional)

Note: All dependencies including `sandbox_fusion` are included in `requirements.txt`.

## Usage

See `README.md` for detailed usage instructions and `examples/example_usage.py` for code examples.

