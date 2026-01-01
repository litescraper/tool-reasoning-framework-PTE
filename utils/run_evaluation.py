#!/usr/bin/env python3
"""
Command-line evaluation script for tool-integrated reasoning framework.

This script provides a command-line interface to run various evaluation tasks.
Supports dynamic loading of evaluators, tool configurations, and datasets.
"""

import argparse
import asyncio
import os
import sys
import json
import traceback
from typing import Dict, Any, Optional

from evaluator.base_evaluator import BaseEvaluator
from utils.class_loader import load_class_from_string


def create_evaluator(
    evaluator_class: str = None,
    api_url: str = None,
    api_key: str = None,
    api_model: str = "gpt-3.5-turbo",
    reward_calculator = None,
    max_assistant_turns: int = 10,
    max_user_turns: int = 5,
    api_extra_headers: Optional[Dict] = None,
    api_extra_params: Optional[Dict] = None,
    verify_correction_kwargs: Optional[Dict] = None,
    **kwargs
):
    """
    Create evaluator instance.
    
    Args:
        evaluator_class: Evaluator class path
        api_url: API URL
        api_key: API key
        api_model: Model name
        reward_calculator: Reward calculator instance
        max_assistant_turns: Maximum assistant response turns
        max_user_turns: Maximum user input turns
        api_extra_headers: Extra API headers
        api_extra_params: Extra model parameters (e.g., temperature, max_tokens)
        verify_correction_kwargs: Extra parameters passed to reward calculator's verify_correction method
        **kwargs: Extra parameters passed to evaluator
        
    Returns:
        Evaluator instance
    """
    if evaluator_class:
        evaluator_cls = load_class_from_string(evaluator_class)
    else:
        evaluator_cls = BaseEvaluator
    
    return evaluator_cls(
        api_url=api_url,
        api_key=api_key,
        api_model=api_model,
        reward_calculator=reward_calculator,
        max_assistant_turns=max_assistant_turns,
        max_user_turns=max_user_turns,
        api_extra_headers=api_extra_headers,
        api_extra_params=api_extra_params,
        verify_correction_kwargs=verify_correction_kwargs,
        **kwargs
    )


def parse_extra_headers(headers_str: str) -> Dict[str, str]:
    """Parse extra HTTP header parameters."""
    headers = {}
    if headers_str:
        for header in headers_str.split(','):
            if ':' in header:
                key, value = header.split(':', 1)
                headers[key.strip()] = value.strip()
    return headers


def parse_extra_params(params_str: str) -> Dict[str, any]:
    """
    Parse extra model parameters.
    
    Supports:
    1) JSON string: '{"temperature":0.7,"max_tokens":2048}'
    2) File path (starts with @): '@/path/to/params.json'
    3) Legacy format: "k:v,k2:v2"
    """
    if not params_str:
        return {}

    text = params_str.strip()

    # Case 1: @ file path (file content is JSON)
    if text.startswith('@'):
        file_path = text[1:]
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Parameter file not found: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        try:
            loaded = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Parameter file JSON parsing failed: {e}")
        if not isinstance(loaded, dict):
            raise ValueError("Parameter file JSON root type must be object(dict)")
        return loaded

    # Case 2: Direct JSON string
    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            return loaded
        else:
            raise ValueError("JSON root type must be object(dict)")
    except json.JSONDecodeError:
        pass

    # Fallback: Legacy format "k:v,k2:v2"
    params: Dict[str, Any] = {}
    for param in text.split(','):
        if ':' not in param:
            continue
        key, value = param.split(':', 1)
        key = key.strip()
        value = value.strip()

        # Auto-convert types (int/float/bool), otherwise string
        try:
            if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                params[key] = int(value)
            elif '.' in value:
                params[key] = float(value)
            elif value.lower() in ('true', 'false'):
                params[key] = value.lower() == 'true'
            else:
                params[key] = value
        except ValueError:
            params[key] = value

    return params


def main():
    parser = argparse.ArgumentParser(
        description="Command-line evaluation script for tool-integrated reasoning framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--dataset-path', type=str, required=True, help='Dataset file path (.json, .jsonl, .parquet)')
    parser.add_argument('--output-dir', type=str, required=True, help='Evaluation results output directory')
    parser.add_argument('--api-key', type=str, required=True, help='API key')
    parser.add_argument('--api-url', type=str, default=None, help='API URL (if not specified, uses default OpenAI API)')
    parser.add_argument('--api-model', type=str, default='gpt-3.5-turbo', help='Model name (default: gpt-3.5-turbo)')
    parser.add_argument('--api-extra-headers', type=str, default=None, help='Extra API headers, format: "key1:value1,key2:value2"')
    parser.add_argument('--api-extra-params', type=str, default=None, help='Extra model parameters; supports JSON string or @file (JSON)')
    parser.add_argument('--verify-correction-kwargs', type=str, default=None, help='Extra reward calculator function parameters')
    parser.add_argument('--evaluator-class', type=str, default=None, help='Evaluator class path')
    parser.add_argument('--reward-calculator-class', type=str, required=True, help='Reward calculator class path')
    parser.add_argument('--tool-config', type=str, default=None, help='Tool configuration YAML file path')
    parser.add_argument('--interaction-config', type=str, default=None, help='Interaction configuration YAML file path')
    parser.add_argument('--max-assistant-turns', type=int, default=10, help='Maximum assistant response turns (default: 10)')
    parser.add_argument('--max-user-turns', type=int, default=20, help='Maximum user input turns (default: 20)')
    parser.add_argument('--max-concurrent', type=int, default=1, help='Maximum concurrency (default: 1)')
    parser.add_argument('--verbose', action='store_true', help='Output detailed information')
    parser.add_argument('--dry-run', action='store_true', help='Only verify configuration, do not run evaluation')
    parser.add_argument('--tokenizer-path', type=str, default=None, help='Tokenizer path (optional, for apply template)')
    parser.add_argument('--bootcamp-registry', type=str, default=None, help='Bootcamp registry path (optional, for batch evaluation)')
    parser.add_argument('--resume-from-result-path', type=str, default=None, help='Resume mode: specify result file path (.jsonl) to resume from')
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.dataset_path):
        print(f"❌ Error: Dataset file not found: {args.dataset_path}")
        sys.exit(1)
    
    # Validate resume file
    if args.resume_from_result_path and not os.path.exists(args.resume_from_result_path):
        print(f"❌ Error: Resume file not found: {args.resume_from_result_path}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.verbose:
        print("🔧 Configuration:")
        print(f"  Dataset path: {args.dataset_path}")
        print(f"  Output directory: {args.output_dir}")
        print(f"  API model: {args.api_model}")
        print(f"  Tool config path: {args.tool_config if args.tool_config else 'None'}")
        print(f"  Interaction config path: {args.interaction_config if args.interaction_config else 'None'}")
        print(f"  Max assistant turns: {args.max_assistant_turns}")
        print(f"  Max user turns: {args.max_user_turns}")
        print(f"  Max concurrent: {args.max_concurrent}")
        print(f"  Extra API headers: {args.api_extra_headers if args.api_extra_headers else 'None'}")
        print(f"  Extra model params: {args.api_extra_params if args.api_extra_params else 'None'}")
        print(f"  Verify correction kwargs: {args.verify_correction_kwargs if args.verify_correction_kwargs else 'None'}")
        print(f"  Resume: {'Enabled (' + args.resume_from_result_path + ')' if args.resume_from_result_path else 'Disabled'}")
    
    try:
        # Parse extra headers and parameters
        extra_headers = parse_extra_headers(args.api_extra_headers) if args.api_extra_headers else None
        extra_params = parse_extra_params(args.api_extra_params) if args.api_extra_params else None
        verify_correction_kwargs = parse_extra_params(args.verify_correction_kwargs) if args.verify_correction_kwargs else None
        
        # Create reward calculator
        if args.verbose:
            print("📊 Creating reward calculator...")
        if args.reward_calculator_class:
            reward_calculator = load_class_from_string(args.reward_calculator_class)
        else:
            reward_calculator = None
        
        # Create evaluator
        if args.verbose:
            print("🤖 Creating evaluator...")
        evaluator = create_evaluator(
            evaluator_class=args.evaluator_class,
            api_url=args.api_url,
            api_key=args.api_key,
            api_model=args.api_model,
            reward_calculator=reward_calculator,
            max_assistant_turns=args.max_assistant_turns,
            max_user_turns=args.max_user_turns,
            api_extra_headers=extra_headers,
            api_extra_params=extra_params,
            verify_correction_kwargs=verify_correction_kwargs,
            tokenizer_path=args.tokenizer_path
        )
        
        if args.dry_run:
            print("✅ Configuration validation passed! (Dry run mode)")
            return
        
        # Run evaluation
        asyncio.run(evaluator.run_evaluation(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            yaml_tool_path=args.tool_config,
            yaml_interaction_path=args.interaction_config,
            max_concurrent=args.max_concurrent,
            bootcamp_registry=args.bootcamp_registry,
            resume_from_result_path=args.resume_from_result_path
        ))
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        if args.verbose:
            print("Detailed error information:")
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

