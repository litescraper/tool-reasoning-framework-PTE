import asyncio
import csv
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import atexit
import httpx
import jsonlines
import openai
import pandas as pd
import yaml
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
from transformers import AutoTokenizer

from evaluator.base_interaction import BaseInteraction
from reward.base_reward_calculator import BaseRewardCalculator
from tools.base_tool import BaseTool
from utils.format_time import format_time_now
from utils.class_loader import load_class_from_string
from utils.tool_loader import load_tool_from_config


def load_interaction_from_config(config: dict) -> BaseInteraction:
    """Load interaction from config dict."""
    import importlib
    class_name = config["class_name"]
    config_dict = config.get("config", {})
    module_path = ".".join(class_name.split(".")[:-1])
    class_name_only = class_name.split(".")[-1]
    module = importlib.import_module(module_path)
    interaction_class = getattr(module, class_name_only)
    return interaction_class(config_dict)


def load_dataset(dataset_path, dataset=None):
    """
    Load dataset, supporting JSON, JSONL and Parquet file formats, always returns list.
    """
    if dataset_path and not dataset:
        _, ext = os.path.splitext(dataset_path)
        ext = ext.lower()
        
        if ext == ".json":
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                dataset = data if isinstance(data, list) else [data]
        
        elif ext == ".jsonl":
            dataset = []
            with jsonlines.open(dataset_path) as reader:
                for line in reader:
                    dataset.append(line)
        
        elif ext == ".parquet":
            df = pd.read_parquet(dataset_path)
            dataset = df.to_dict(orient='records')
            
            for item in dataset:
                if 'messages' in item and hasattr(item['messages'], 'tolist'):
                    item['messages'] = item['messages'].tolist()
                elif 'messages' in item and not isinstance(item['messages'], list):
                    item['messages'] = list(item['messages'])
                
                if 'prompt' in item and hasattr(item['prompt'], 'tolist'):
                    item['prompt'] = item['prompt'].tolist()
                elif 'prompt' in item and not isinstance(item['prompt'], list):
                    item['prompt'] = list(item['prompt'])
        
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    return dataset


class BaseEvaluator:
    """Base evaluator for tool-using agent inference."""
    
    def __init__(
        self,
        api_key: str,
        reward_calculator: BaseRewardCalculator,
        api_url: str = None,
        api_model: str = None,
        api_extra_headers: dict = None,
        api_extra_params: dict = None,
        verify_correction_kwargs: dict = None,
        max_assistant_turns: int = None,
        max_user_turns: int = None,
        tokenizer_path = None,
    ):
        self.api_model = api_model
        self.api_extra_headers = api_extra_headers or {}
        self.api_extra_params = api_extra_params or {}
        self.verify_correction_kwargs = verify_correction_kwargs or {}
        self.max_assistant_turns = max_assistant_turns
        self.max_user_turns = max_user_turns
        max_connections = 3000
        limits = httpx.Limits(max_keepalive_connections=max_connections, max_connections=max_connections)
        self._httpx_client = httpx.AsyncClient(limits=limits, verify=False)
        self.client = openai.AsyncOpenAI(
            base_url=api_url, 
            api_key=api_key, 
            default_headers=api_extra_headers, 
            http_client=self._httpx_client
        )
        self.bootcamp_registry: Dict[str, dict] = {}
        self.reward_calculator = reward_calculator
        self.tokenizer_path = tokenizer_path
        self.tokenizer = self._get_tokenizer()
        atexit.register(self.on_exit)

    def on_exit(self):
        """Clean up the OpenAI client during program shutdown."""
        try:
            if hasattr(self, "client"):
                self.client = None
            if hasattr(self, "_httpx_client") and self._httpx_client is not None:
                try:
                    import threading
                    def background_cleanup():
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            loop.run_until_complete(self._httpx_client.aclose())
                            loop.close()
                        except Exception:
                            pass
                        finally:
                            self._httpx_client = None
                    cleanup_thread = threading.Thread(target=background_cleanup, daemon=True)
                    cleanup_thread.start()
                except Exception:
                    self._httpx_client = None
        except Exception:
            pass
        
    def _get_tokenizer(self):
        if not self.tokenizer_path:
            return None
        try:
            default_tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path, 
                trust_remote_code=True
            )
            return default_tokenizer
        except Exception as e:
            print(f"[WARNING] Failed to load tokenizer: {e}")
            return None

    def _build_payload(self, input_data: dict) -> dict:
        """Build API payload from input data."""
        messages = input_data["messages"]
        tools = input_data.get("tools")
        tool_choice = input_data.get("tool_choice", "auto")

        payload = {
            "model": self.api_model,
            "messages": messages,
            "extra_body": {"chat_template_kwargs": {"thinking": True}}
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        if self.api_extra_params:
            payload.update(self.api_extra_params)

        return payload
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=60),
        reraise=True,
        before_sleep=lambda retry_state: print(f"Retrying... Attempt {retry_state.attempt_number} failed: \n{retry_state.outcome.exception()}")
    )
    async def _call_api(self, payload: dict) -> Dict[str, Any]:
        """Call the API and return response with timing info."""
        start_timestamp = time.time()
        start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        
        try:
            response = await self.client.chat.completions.create(**payload)
        except Exception as e:
            raise e

        end_timestamp = time.time()
        end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        latency = end_timestamp - start_timestamp
        
        timing_info = {
            "start_time": start_time_str,
            "end_time": end_time_str,
            "latency_seconds": latency
        }
        return response.model_dump(), timing_info
    
    def _load_tools_from_yaml(self, yaml_path: str) -> Tuple[List[Dict], Dict[str, Dict[str, Any]]]:
        """Load tool configurations from YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        tools_config = config.get("tools", [])
        tool_schemas = []
        tool_instances = {}
        for tool_cfg in tools_config:
            try:
                func_name, schema, tool_instance = load_tool_from_config(tool_cfg)
                tool_instances[func_name] = {
                    "instance": tool_instance
                }
                tool_schemas.append(schema)
            except Exception as e:
                print(f"❌ Failed to load tool: {e}")
                import traceback
                traceback.print_exc()
        return tool_schemas, tool_instances

    def _load_interaction_from_yaml(self, yaml_path: str) -> BaseInteraction:
        """Load interaction from YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        interaction_config = config.get("interaction", [])
        if len(interaction_config) > 1:
            raise ValueError("Interaction config should only contain one interaction")
        interaction_instance = load_interaction_from_config(interaction_config[0])
        return interaction_instance
    
    def _messages_to_context(self, messages: List[Dict[str, Any]], tools: List[Dict] = []) -> str:
        """Convert messages list to full context string."""
        tokenizer_to_use = self.tokenizer
        if tokenizer_to_use is not None:
            try:
                processed_messages = []
                for message in messages:
                    processed_message = message.copy()
                    if "reasoning_content" in message:
                        reasoning_content = message["reasoning_content"]
                        content = message.get("content", "")
                        processed_message["content"] = f"<think>\n{reasoning_content}\n</think>\n\n{content}"
                        processed_message.pop("reasoning_content", None)
                    processed_messages.append(processed_message)
                
                return tokenizer_to_use.apply_chat_template(
                    processed_messages,
                    tools=tools, 
                    add_generation_prompt=False, 
                    tokenize=False
                )
            except Exception as e:
                print(f"[WARNING] apply_chat_template failed, using fallback: {e}")
        
        # Fallback: custom formatting
        context_parts = []
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if "reasoning_content" in message:
                reasoning_content = message["reasoning_content"]
                content = f"<think>\n{reasoning_content}\n</think>\n\n{content}"
            
            if role == "assistant" and "tool_calls" in message and message["tool_calls"]:
                tool_calls_content = ""
                for tool_call in message["tool_calls"]:
                    function_name = tool_call.get("function", {}).get("name", "")
                    arguments = tool_call.get("function", {}).get("arguments", "")
                    tool_calls_content += f"Function: {function_name}\nArguments: {arguments}\n\n"
                
                if tool_calls_content:
                    content = f"{content}\n<tool_call>\n{tool_calls_content.strip()}</tool_call>"
            
            if role == "user":
                context_parts.append(f"User:\n{content}\n")
            elif role == "assistant":
                context_parts.append(f"Assistant:\n{content}\n")
            elif role == "system":
                context_parts.append(f"System:\n{content}\n")
            elif role == "tool":
                context_parts.append(f"<tool_response>\n{content}\n</tool_response>\n")
            else:
                context_parts.append(f"{role.capitalize()}:\n{content}\n")
        
        return "".join(context_parts)

    async def _execute_tool_calls(
        self,
        tool_calls: List[Dict],
        context_instance_id_dict: Optional[Dict[str, str]] = None,
        sample_extra_info: Dict[str, Any] = None,
        tool_instances: Dict[str, Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, str], List[Dict[str, Any]], float, dict, float, List[Dict]]:
        """Execute tool calls and return results."""
        tool_messages = []
        sample_extra_info = sample_extra_info or {}
        tool_instances = tool_instances or getattr(self, 'tool_instances', {})
        tool_reward = 0.0
        tool_metrics = {}
        tool_cumulative_reward = 0.0
        batch_tool_timings = []

        for idx, tool_call in enumerate(tool_calls, 1):
            tool_start_ts = time.time()
            tool_start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

            tool_name = tool_call["function"]["name"]
            arguments = tool_call["function"]["arguments"]
            args_preview = arguments[:150] + "..." if len(arguments) > 150 else arguments
            print(f"\n[🔧 Executing Tool {idx}/{len(tool_calls)}] {tool_name}")
            print(f"    Arguments: {args_preview}")
            
            if tool_name not in tool_instances:
                content = f"Error: Tool '{tool_name}' is not registered."
                print(f"[❌ Tool {idx}] Tool not registered: {tool_name}")
            else:
                try:
                    args = json.loads(arguments)
                    tool_instance = tool_instances[tool_name]["instance"]

                    if "tools_kwargs" in sample_extra_info and tool_name in sample_extra_info["tools_kwargs"]:
                        create_kwargs = sample_extra_info["tools_kwargs"][tool_name].get("create_kwargs", {})
                    else:
                        create_kwargs = {}

                    current_instance_id = await tool_instance.create(context_instance_id_dict[tool_name], **create_kwargs)
                    context_instance_id_dict[tool_name] = current_instance_id
                    tool_result = await tool_instance.execute(current_instance_id, args)
                    tool_cumulative_reward = await tool_instance.calc_reward(current_instance_id)
                    tool_response, tool_reward, tool_metrics = tool_result
                    content = str(tool_response)
                    response_preview = content[:200] + "..." if len(content) > 200 else content
                    print(f"[✅ Tool {idx}] {tool_name} completed")
                    print(f"    Response preview: {response_preview}")
                except Exception as e:
                    content = f"Error calling {tool_name}: {str(e)}"
                    print(f"[❌ Tool {idx}] {tool_name} failed: {str(e)}")
            
            tool_messages.append({
                "role": "tool",
                "content": content,
                "tool_call_id": tool_call["id"]
            })

            tool_end_ts = time.time()
            tool_end_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            tool_elapsed = tool_end_ts - tool_start_ts

            batch_tool_timings.append({
                "tool_name": tool_name,
                "tool_call_id": tool_call.get("id"),
                "start_time": tool_start_str,
                "end_time": tool_end_str,
                "latency_seconds": tool_elapsed
            })
            print(f"    Execution time: {tool_elapsed:.3f}s")

        return context_instance_id_dict, tool_messages, tool_reward, tool_metrics, tool_cumulative_reward, batch_tool_timings

    async def _evaluate_one(self, input_data: dict) -> dict:
        """Evaluate a single sample."""
        data_source = input_data.get("data_source", None)
        
        if self.bootcamp_registry:
            if data_source and data_source in self.bootcamp_registry:
                config = self.bootcamp_registry[data_source]
                tool_schemas = config["tool_schemas"]
                tool_instances = config["tool_instances"]
                interaction_instance = config["interaction_instance"]
                reward_calculator = config["reward_calculator_class"]
            else:
                print(f"❌ Data source not found: {data_source}")
                return None
        else:
            if not self.reward_calculator:
                raise ValueError("Must provide bootcamp_registry or reward_calculator")
            tool_schemas = self.tool_schemas
            tool_instances = self.tool_instances
            interaction_instance = self.interaction
            reward_calculator = self.reward_calculator
        
        # Handle both prompt/messages fields
        if "messages" in input_data:
            messages = input_data["messages"].copy()
        elif "prompt" in input_data:
            messages = input_data["prompt"].copy()
        else:
            raise ValueError("Input data must contain 'messages' or 'prompt' field")

        needed_tools = []
        extra_info = input_data.get("extra_info", {})
        if extra_info.get("need_tools_kwargs") and "tools_kwargs" in extra_info and tool_schemas:
            needed_tool_names = set(extra_info["tools_kwargs"].keys())
            needed_tools = [tool for tool in tool_schemas if tool["function"]["name"] in needed_tool_names]

        payload = self._build_payload({
            "messages": messages,
            "tools": needed_tools,
            "tool_choice": "auto"
        })
        
        all_payloads = [payload]
        try:
            turn_record = {}
            context_instance_id_dict = {}
            total_inference_seconds = 0.0
            total_tool_seconds = 0.0
            timeline_record = []

            if tool_instances:
                for tool_name in tool_instances:
                    context_instance_id_dict[tool_name] = None
            
            if interaction_instance:
                if "interaction_kwargs" in input_data["extra_info"]:
                    interaction_instance_id = await interaction_instance.start_interaction(
                        identity=input_data["extra_info"]["interaction_kwargs"]["identity"]
                    )
                else:
                    interaction_instance_id = await interaction_instance.start_interaction()
            else:
                interaction_instance_id = None
            
            assistant_turn_count = 0
            user_turn_count = 0
            interaction_turn = 0
            
            while assistant_turn_count < self.max_assistant_turns and user_turn_count < self.max_user_turns:
                tool_turns_executed = 0
                tool_calls_executed = 0
                while assistant_turn_count < self.max_assistant_turns and user_turn_count < self.max_user_turns:
                    print(f"\n[🤖 LLM Inference] Turn {assistant_turn_count + 1}")
                    raw_response, inference_timing = await self._call_api(payload)
                    
                    inference_elapsed = inference_timing["latency_seconds"]
                    total_inference_seconds += inference_elapsed
                    print(f"[🤖 LLM Inference] Completed in {inference_elapsed:.3f}s")
                    
                    timeline_record.append({
                        "type": "llm_inference",
                        "turn_idx": assistant_turn_count + 1,
                        "detail": inference_timing
                    })
                    
                    message = raw_response["choices"][0]["message"]
                    messages.append(message)
                    tool_calls = message.get("tool_calls", [])
                    assistant_turn_count += 1
                    
                    if not tool_calls:
                        print(f"\n[📊 Turn {assistant_turn_count}] Completed, no tool calls")
                        break
                    
                    print(f"\n[📊 Turn {assistant_turn_count}] Found {len(tool_calls)} tool call(s):")
                    for idx, tool_call in enumerate(tool_calls, 1):
                        tool_name = tool_call['function']['name']
                        tool_args = tool_call['function']['arguments']
                        args_preview = tool_args[:100] + "..." if len(tool_args) > 100 else tool_args
                        print(f"  [{idx}] Tool: {tool_name}")
                        print(f"      Args: {args_preview}")
                    tool_turns_executed += 1
                    tool_calls_executed += len(tool_calls)
                    
                    sample_extra_info = input_data.get("extra_info", {})
                    context_instance_id_dict, tool_messages, tool_reward, tool_metrics, tool_cumulative_reward, batch_tool_timings = await self._execute_tool_calls(
                        tool_calls, context_instance_id_dict, sample_extra_info, tool_instances
                    )
                    
                    current_batch_tool_time = sum(t["latency_seconds"] for t in batch_tool_timings)
                    total_tool_seconds += current_batch_tool_time
                    print(f"\n[📊 Tool Batch] Total execution time: {current_batch_tool_time:.3f}s")
                    
                    timeline_record.append({
                        "type": "tool_execution",
                        "turn_idx": assistant_turn_count,
                        "tools_detail": batch_tool_timings,
                        "batch_total_seconds": current_batch_tool_time
                    })

                    messages.extend(tool_messages)
                    user_turn_count += len(tool_messages)
                    payload = self._build_payload({
                        "messages": messages,
                        "tools": needed_tools,
                        "tool_choice": "auto",
                    })
                    all_payloads.append(payload)
                
                turn_record[f"interaction_turn_{interaction_turn}"] = {
                    "assistant_turns": tool_turns_executed,
                    "tool_calls": tool_calls_executed
                }
                user_turn_count += 1
                interaction_turn += 1
                
                if interaction_instance:
                    should_terminate_sequence, response_content, current_turn_score, additional_data = await interaction_instance.generate_response(
                        interaction_instance_id, messages
                    )
                    if should_terminate_sequence:
                        break
                    else:
                        messages.append({
                            "role": "user",
                            "content": response_content
                        })
                else:
                    break

            # Release tools
            for tool_name, instance_id in context_instance_id_dict.items():
                if instance_id:
                    tool_instance = tool_instances[tool_name]["instance"]
                    await tool_instance.release(instance_id)

            time_stats = {
                "summary": {
                    "total_llm_inference_seconds": round(total_inference_seconds, 4),
                    "total_tool_execution_seconds": round(total_tool_seconds, 4),
                    "total_wall_time_seconds": round(total_inference_seconds + total_tool_seconds, 4)
                },
                "timeline": timeline_record
            }

            if messages[-1]['role'] != 'assistant':
                return {
                    "input": input_data,
                    "tools": needed_tools,
                    "messages": messages,
                    "extracted_output": None,
                    "ground_truth": input_data["reward_model"]["ground_truth"],
                    "score": 0,
                    "reached_max_turns": True,
                    "turn_record": turn_record,
                    "success": True,
                    "time_stats": time_stats,
                    "evaluation_config": {
                        "model": self.api_model,
                        "api_extra_params": self.api_extra_params,
                        "api_extra_headers": self.api_extra_headers,
                        "max_assistant_turns": self.max_assistant_turns,
                        "max_user_turns": self.max_user_turns,
                    }
                }
            
            final_output = {"content": messages[-1]["content"]}
            if "reasoning_content" in messages[-1]:
                reasoning_content = messages[-1]["reasoning_content"]
                final_output["reasoning_content"] = reasoning_content
            
            full_context = self._messages_to_context(messages)
            if "prompt" in input_data:
                response_context = self._messages_to_context(messages[len(input_data["prompt"]):])
            elif "messages" in input_data:
                response_context = self._messages_to_context(messages[len(input_data["messages"]):])
            
            score = reward_calculator.verify_score(
                model_output=response_context, 
                identity=input_data["reward_model"]["ground_truth"], 
                **self.verify_correction_kwargs
            ) if reward_calculator else None
            extracted_output = reward_calculator.extract_output(response_context)
            
            return {
                "input": input_data,
                "tools": needed_tools,
                "messages": messages,
                "extracted_output": extracted_output,
                "ground_truth": input_data["reward_model"]["ground_truth"],
                "score": score,
                "reached_max_turns": False,
                "turn_record": turn_record,
                "success": True,
                "response_context": response_context,
                "time_stats": time_stats,
                "evaluation_config": {
                    "model": self.api_model,
                    "api_extra_params": self.api_extra_params,
                    "api_extra_headers": self.api_extra_headers,
                    "max_assistant_turns": self.max_assistant_turns,
                    "max_user_turns": self.max_user_turns,
                }
            }
        except Exception as e:
            import traceback
            print(traceback.print_exc())
            return {
                "input": input_data,
                "tools": needed_tools if 'tools' in locals() else [],
                "messages": messages,
                "output": None,
                "score": 0,
                "error": str(e),
                "reached_max_turns": False,
                "turn_record": turn_record,
                "success": False,
                "time_stats": {
                    "error": "Evaluation Failed", 
                    "partial_summary": {
                        "llm": locals().get('total_inference_seconds', 0),
                        "tool": locals().get('total_tool_seconds', 0)
                    }
                },
                "evaluation_config": {
                    "model": self.api_model,
                    "api_extra_params": self.api_extra_params,
                    "api_extra_headers": self.api_extra_headers,
                    "max_assistant_turns": self.max_assistant_turns,
                    "max_user_turns": self.max_user_turns,
                }
            }

    async def _evaluate_batch(
        self,
        input_list: List[dict],
        max_concurrent: int = 1,
        output_path: Optional[str] = None
    ) -> List[dict]:
        """Evaluate a batch of samples."""
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        results = [None] * len(input_list)
        
        progress_bar = tqdm(
            total=len(input_list), 
            desc="Evaluation Progress",
            colour="cyan",
            dynamic_ncols=True,
            unit_scale=False
        )
        progress_lock = asyncio.Lock()
        file_write_lock = asyncio.Lock()

        async def worker(idx, input_data):
            async with semaphore:
                result = await self._evaluate_one(input_data)
                results[idx] = result
                if output_path:
                    async with file_write_lock:
                        with open(output_path, "a", encoding="utf-8") as f:
                            try:
                                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                            except Exception as e:
                                print(f"❌ Failed to write result: {e}")
                
                async with progress_lock:
                    progress_bar.update(1)

        tasks = [worker(idx, input_data) for idx, input_data in enumerate(input_list)]
        await asyncio.gather(*tasks)
        progress_bar.close()
        
        return results

    def _load_bootcamp_registry(self, bootcamp_registry: str):
        """Load bootcamp registry from JSONL file."""
        with jsonlines.open(bootcamp_registry) as reader:
            for line in reader:
                data_source = line.get("data_source")
                yaml_tool_path = line.get("yaml_tool_path")
                yaml_interaction_path = line.get("yaml_interaction_path")
                reward_calculator_class_path = line.get("reward_calculator_class")
                try:
                    tool_schemas, tool_instances = self._load_tools_from_yaml(yaml_tool_path)
                    interaction_instance = self._load_interaction_from_yaml(yaml_interaction_path)
                    reward_calculator_class = load_class_from_string(reward_calculator_class_path)
                    self.bootcamp_registry[data_source] = {
                        "tool_schemas": tool_schemas,
                        "tool_instances": tool_instances,
                        "interaction_instance": interaction_instance,
                        "reward_calculator_class": reward_calculator_class
                    }
                except Exception as e:
                    import pprint
                    pprint.pprint(line)
                    print(f"❌ Failed to load {data_source} config: {e}")
                    raise e
            
    async def run_evaluation(
        self,
        dataset: Optional[List[dict]] = None,
        dataset_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        yaml_tool_path: Optional[str] = None,
        yaml_interaction_path: Optional[str] = None,
        max_concurrent: int = 1,
        bootcamp_registry: Optional[str] = None,
        resume_from_result_path: Optional[str] = None
    ) -> List[dict]:
        """
        Start complete evaluation pipeline.

        Args:
            dataset: Data list
            dataset_path: JSON file path (contains test case list)
            output_dir: Result save path (JSONL)
            yaml_tool_path: Tool YAML config path (if provided, will override current tools)
            yaml_interaction_path: Interaction YAML config path
            max_concurrent: Maximum concurrent evaluations
            bootcamp_registry: Bootcamp registry JSONL file path
            resume_from_result_path: Path to resume from previous results
        """
        # Load tool config (optional)
        if yaml_tool_path:
            self.tool_schemas, self.tool_instances = self._load_tools_from_yaml(yaml_tool_path)
        else:
            self.tool_schemas, self.tool_instances = None, None
        
        if yaml_interaction_path:
            self.interaction = self._load_interaction_from_yaml(yaml_interaction_path)
        else:
            self.interaction = None
        
        if bootcamp_registry:
            self._load_bootcamp_registry(bootcamp_registry)
        
        # Load dataset
        if dataset_path and not dataset:
            dataset = load_dataset(dataset_path)

        if not dataset:
            raise ValueError("Must provide dataset or dataset_path")

        # Resume logic
        completed_inputs = set()
        original_dataset_size = len(dataset)
        
        if resume_from_result_path and os.path.exists(resume_from_result_path):
            print(f"🔄 Resume mode detected, loading completed results from {resume_from_result_path}...")
            try:
                with open(resume_from_result_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            result = json.loads(line.strip())
                            if result.get("input"):
                                input_key = json.dumps(result["input"], sort_keys=True, ensure_ascii=False)
                                completed_inputs.add(input_key)
                print(f"📊 Completed {len(completed_inputs)} samples, remaining {original_dataset_size - len(completed_inputs)} samples to evaluate")
                filtered_dataset = []
                for item in dataset:
                    item_key = json.dumps(item, sort_keys=True, ensure_ascii=False)
                    if item_key not in completed_inputs:
                        filtered_dataset.append(item)
                dataset = filtered_dataset
                output_path = resume_from_result_path
            except Exception as e:
                print(f"⚠️ Error reading completed results: {e}, will start fresh")
                completed_inputs = set()
                output_path = os.path.join(output_dir, f"{self.api_model.replace('/', '-').strip('-')}/eval_results_{format_time_now()}.jsonl")
        else:
            output_path = os.path.join(output_dir, f"{self.api_model.replace('/', '-').strip('-')}/eval_results_{format_time_now()}.jsonl")
        
        print(f"🚀 Starting evaluation with {len(dataset)} samples...")
        
        if not resume_from_result_path or not os.path.exists(output_path):
            if output_path and os.path.exists(output_path):
                open(output_path, "w", encoding="utf-8").close()
        
        if output_path and not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"💾 Evaluation results will be saved to: {output_path}")
        
        if len(dataset) == 0:
            print("✅ All samples completed!")
            results = []
            if os.path.exists(output_path):
                with open(output_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            results.append(json.loads(line.strip()))
        else:
            results = await self._evaluate_batch(dataset, max_concurrent=max_concurrent, output_path=output_path)
        
        return results

