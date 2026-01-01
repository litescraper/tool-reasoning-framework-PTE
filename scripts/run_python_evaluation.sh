#!/bin/bash

# Evaluation script for Python tool
# Datasets: AIME25, AIME24, AIME_Math500

# Base dataset directory
BASE_DATA_DIR="./data"

# Datasets to evaluate
DATASETS=(
    "aime25.json"
    "aime24.json"
    "math500.json"
)

# API configuration
API_KEY="your-api-key"
API_URL="http://localhost:8000/v1"
API_MODEL="gpt-oss"

# Tool service URLs
# Python tool service (configure your FastAPI/Flask service)
export PYTHON_SERVICE_URL="http://localhost:8080"

# Evaluator and reward calculator classes
EVALUATOR_CLASS="evaluator.base_evaluator.BaseEvaluator"
REWARD_CALCULATOR_CLASS="reward.reward_calculator.GoogleandPythonRewardCalculator"

# Tool configuration
TOOL_CONFIG="./configs/python_tool_config.yaml"

# Maximum turns
MAX_ASSISTANT_TURNS=10
MAX_USER_TURNS=99999

# Maximum concurrency
MAX_CONCURRENT=10

# Loop through all datasets
for dataset_file in "${DATASETS[@]}"; do
    # Extract dataset name (remove .json)
    dataset_name=$(echo "$dataset_file" | sed 's/\.json$//')
    
    # Dataset path
    DATASET_PATH="${BASE_DATA_DIR}/${dataset_file}"
    
    # Output directory (unified format: dataset_name_model_name)
    OUTPUT_DIR="./results/${dataset_name}_${API_MODEL}"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    echo "=========================================="
    echo "Evaluating dataset: $dataset_name"
    echo "Dataset path: $DATASET_PATH"
    echo "Output directory: $OUTPUT_DIR"
    echo "=========================================="
    
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
        --verbose
    
    if [ $? -ne 0 ]; then
        echo "❌ Error evaluating $dataset_name"
        exit 1
    fi
    
    echo "✅ Completed evaluation for $dataset_name"
    echo ""
done

echo "✅ All evaluations completed!"

