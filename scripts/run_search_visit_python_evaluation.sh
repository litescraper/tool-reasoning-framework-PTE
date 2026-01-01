#!/bin/bash

# Evaluation script for Search + Visit + Python tools
# Dataset: WebInstruct

# Dataset path
DATASET_PATH="./data/webinstruct.json"

# API configuration
API_KEY="your-api-key"
API_URL="http://localhost:8000/v1"
API_MODEL="gpt-oss"

# Tool service URLs
# Search tool service (configure your FastAPI/Flask service)
export SERPER_SERVICE_URL="http://localhost:5003/serper"

# Visit tool service (Jina service and summary model)
export JINA_SERVICE_URL="http://localhost:5003/jina"
export VISIT_SUMMARY_API_KEY="your-api-key"
export VISIT_SUMMARY_API_URL="http://localhost:8000/v1"
export VISIT_SUMMARY_MODEL="gpt-oss"

# Python tool service (configure your FastAPI/Flask service)
export PYTHON_SERVICE_URL="http://localhost:8080"

# Output directory (unified format: dataset_name_model_name)
DATASET_NAME="webinstruct"
OUTPUT_DIR="./results/${DATASET_NAME}_${API_MODEL}"

# Resume path (optional, uncomment if needed)
# RESUME_PATH="./results/${DATASET_NAME}_${API_MODEL}/gpt-oss/eval_results_YYYYMMDDHHMMSS.jsonl"

# Evaluator and reward calculator classes
EVALUATOR_CLASS="evaluator.base_evaluator.BaseEvaluator"
REWARD_CALCULATOR_CLASS="reward.reward_calculator.GoogleandPythonRewardCalculator"

# Tool configuration
TOOL_CONFIG="./configs/search_visit_python_tool_config.yaml"

# Maximum turns
MAX_ASSISTANT_TURNS=30
MAX_USER_TURNS=99999

# Maximum concurrency
MAX_CONCURRENT=1

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation
python -m utils.run_evaluation \
    --dataset-path "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --api-key "$API_KEY" \
    --api-url "$API_URL" \
    --api-model "$API_MODEL" \
    --evaluator-class "$EVALUATOR_CLASS" \
    --reward-calculator-class "$REWARD_CALCULATOR_CLASS" \
    --tool-config "$TOOL_CONFIG" \
    --max-user-turns $MAX_USER_TURNS \
    --max-assistant-turns $MAX_ASSISTANT_TURNS \
    --max-concurrent $MAX_CONCURRENT \
    ${RESUME_PATH:+--resume-from-result-path "$RESUME_PATH"} \
    --verbose

