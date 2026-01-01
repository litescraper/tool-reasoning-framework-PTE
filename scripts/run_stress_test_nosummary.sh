#!/bin/bash

# Stress test evaluation script for Search + Visit tools (no summary)
# This script is designed for high-concurrency stress testing with timing logs

# Dataset path (update this to your stress test dataset)
DATASET_PATH="./data/stress_test_dataset.json"

# API configuration
API_KEY="your-api-key"
API_URL="http://localhost:8000/v1"
API_MODEL="deepseek_v32"

# Tool service URLs (No Summary - for stress testing)
# Search service (no summary, configure your FastAPI/Flask service)
export SEARCH_SERVICE_URL="http://localhost:5003/serper"

# Jina service (no summary, configure your FastAPI/Flask service)
export JINA_SERVICE_URL="http://localhost:5003/jina"

# Output directory (unified format: dataset_name_model_name_timing_log)
DATASET_NAME="stress_test"
OUTPUT_DIR="./results/${DATASET_NAME}_${API_MODEL}_timing_log"

# Resume path (optional, uncomment if needed)
# RESUME_PATH="./results/${DATASET_NAME}_${API_MODEL}_timing_log/${API_MODEL}/eval_results_YYYYMMDDHHMMSS.jsonl"

# Evaluator and reward calculator classes
EVALUATOR_CLASS="evaluator.base_evaluator.BaseEvaluator"
REWARD_CALCULATOR_CLASS="reward.reward_calculator.GoogleandPythonRewardCalculator"

# Tool configuration (no summary version)
TOOL_CONFIG="./configs/search_visit_tool_config_nosummary.yaml"

# Maximum turns (higher for stress testing)
MAX_ASSISTANT_TURNS=30
MAX_USER_TURNS=99999

# Maximum concurrency (high for stress testing)
MAX_CONCURRENT=256

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

