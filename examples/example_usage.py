"""
Example usage of the tool-using agent inference pipeline.
"""
import asyncio
import os
from evaluator.base_evaluator import BaseEvaluator
from reward.reward_calculator import GoogleandPythonRewardCalculator


async def main():
    # Initialize reward calculator
    reward_calculator = GoogleandPythonRewardCalculator()
    
    # Initialize evaluator
    evaluator = BaseEvaluator(
        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        reward_calculator=reward_calculator,
        api_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        api_model=os.getenv("OPENAI_MODEL", "gpt-oss"),
        max_assistant_turns=10,
        max_user_turns=10,
    )
    
    # Load tools from YAML config
    yaml_tool_path = "configs/tool_config.yaml"  # Path to your tool config
    
    # Example input data
    input_data = {
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of France?"
            }
        ],
        "reward_model": {
            "ground_truth": "Paris"
        },
        "extra_info": {
            "tools_kwargs": {
                "google_search": {
                    "create_kwargs": {}
                }
            }
        }
    }
    
    # Run evaluation
    results = await evaluator.run_evaluation(
        dataset=[input_data],
        yaml_tool_path=yaml_tool_path,
        output_dir="./results",
        max_concurrent=1
    )
    
    print(f"Evaluation completed. Results: {results}")


if __name__ == "__main__":
    asyncio.run(main())

