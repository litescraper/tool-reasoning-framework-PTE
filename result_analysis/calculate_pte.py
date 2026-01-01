"""
PTE (Prefill-Token-Equivalent) Cost Analysis Script

This script calculates PTE cost metrics from evaluation results.
PTE cost formula: Cost = D_prefill + gamma * C_mem
- D_prefill = sum(tokens_sys) + sum(tokens_user) + sum(tokens_tool)
- C_mem = sum_over_all_turns( L_seq * D_decode )
"""

import json
import os
import csv
from collections import defaultdict
from transformers import AutoTokenizer


def process_jsonl(jsonl_path, tokenizer_path, gamma, model_name):
    """
    Process a single JSONL file and return statistics.
    
    Args:
        jsonl_path: Path to evaluation results JSONL file
        tokenizer_path: Path to tokenizer
        gamma: Gamma parameter for PTE calculation
        model_name: Model name for output
        
    Returns:
        Dictionary with statistics
    """
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )
    
    # Basic statistics
    score_1_cnt = 0
    score_0_cnt = 0
    score_sum = 0
    score_dist = defaultdict(int)
    
    tool_turns_sum = 0
    tool_turns_dist = defaultdict(int)
    line_count = 0

    # KV Cache cost model statistics
    assistant_token_sum = 0.         # D_decode (total decode tokens)
    prefill_kv_token_sum = 0.        # D_prefill (total prefill tokens)
    weighted_decode_token_sum = 0.   # C_mem (total memory-weighted decode tokens)
    
    score_1_cost_pte_sum = 0.      # score=1 InferCost_PTE sum
    score_0_cost_pte_sum = 0.      # score=0 InferCost_PTE sum

    # Step-wise statistics
    assistant_step_tokens = defaultdict(lambda: {'sum': 0, 'count': 0})
    assistant_step_costs_pte = defaultdict(lambda: {'sum': 0, 'count': 0})
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                line_count += 1
                
                # Reset accumulators for each dialogue
                current_prefill_kv_tokens = 0.      # Current dialogue D_prefill
                current_weighted_decode_tokens = 0. # Current dialogue C_mem
                current_total_decode_tokens = 0.    # Current dialogue D_decode

                # Statistics score
                score = data.get("score")
                if score is not None:
                    score_sum += score
                    score_dist[score] += 1
                
                # Statistics tool_turns
                turn_record = data.get("turn_record")
                if turn_record and "interaction_turn_0" in turn_record:
                    tt = turn_record["interaction_turn_0"].get("tool_turns")
                    if tt is None:
                        tt = turn_record["interaction_turn_0"].get("tool_calls")

                    if tt is not None:
                        tool_turns_sum += tt
                        tool_turns_dist[tt] += 1

                # Statistics token count
                if "messages" in data and isinstance(data["messages"], list):
                    message_history_tokens = 0
                    current_assistant_step = 0

                    for msg in data["messages"]:
                        role = msg.get("role")
                        content = msg.get("content", "")
                        reasoning_content = msg.get("reasoning_content", "")
                            
                        tokens = tokenizer.encode(str(content or ""), add_special_tokens=False) + \
                                 tokenizer.encode(str(reasoning_content or ""), add_special_tokens=False)
                        num_tokens = len(tokens)
                        
                        if role == "assistant":
                            # Decode phase
                            L_seq_before_decode = message_history_tokens
                            
                            step_actual_prefill = float(L_seq_before_decode)
                            step_weighted_cost = L_seq_before_decode * num_tokens
                            step_cost_pte = step_actual_prefill + (gamma * step_weighted_cost)

                            current_prefill_kv_tokens += step_actual_prefill
                            current_weighted_decode_tokens += step_weighted_cost
                            current_total_decode_tokens += num_tokens
                            
                            message_history_tokens += num_tokens

                            current_assistant_step += 1
                            assistant_step_tokens[current_assistant_step]['sum'] += num_tokens
                            assistant_step_tokens[current_assistant_step]['count'] += 1
                            assistant_step_costs_pte[current_assistant_step]['sum'] += step_cost_pte
                            assistant_step_costs_pte[current_assistant_step]['count'] += 1

                        elif role in ["system", "user", "tool"]:
                            message_history_tokens += num_tokens
                    
                    # Accumulate current row cost to total
                    dialogue_cost_pte = current_prefill_kv_tokens + gamma * current_weighted_decode_tokens

                    prefill_kv_token_sum += current_prefill_kv_tokens
                    weighted_decode_token_sum += current_weighted_decode_tokens
                    assistant_token_sum += current_total_decode_tokens
                    
                    if score == 1:
                        score_1_cnt += 1
                        score_1_cost_pte_sum += dialogue_cost_pte
                    elif score == 0:
                        score_0_cnt += 1
                        score_0_cost_pte_sum += dialogue_cost_pte

            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error processing line in {jsonl_path}: {e}")
                continue

    if line_count == 0:
        return None
    
    # Calculate averages
    avg_score = score_sum / line_count
    avg_tool_turns = tool_turns_sum / line_count
    
    avg_decode_tokens = assistant_token_sum / line_count
    avg_prefil_tokens_kv = prefill_kv_token_sum / line_count
    avg_weighted_decode_tokens = weighted_decode_token_sum / line_count
    
    avg_inference_cost_pte = (prefill_kv_token_sum + gamma * weighted_decode_token_sum) / line_count
    
    score_1_avg_infer_cost_pte = (score_1_cost_pte_sum / score_1_cnt) if score_1_cnt else 0.0
    score_0_avg_infer_cost_pte = (score_0_cost_pte_sum / score_0_cnt) if score_0_cnt else 0.0

    # Calculate step-wise averages
    step_metrics_dict = {}
    all_steps = set(assistant_step_tokens.keys()) | set(assistant_step_costs_pte.keys())
    
    for step in sorted(all_steps):
        token_stats = assistant_step_tokens.get(step, {'sum': 0, 'count': 0})
        cost_stats = assistant_step_costs_pte.get(step, {'sum': 0, 'count': 0})
        
        count = token_stats['count']
        avg_tokens = (token_stats['sum'] / count) if count > 0 else 0.0
        avg_cost = (cost_stats['sum'] / count) if count > 0 else 0.0
        
        step_metrics_dict[f"assistant_step_count_{step}"] = count
        step_metrics_dict[f"assistant_step_avg_tokens_{step}"] = f"{avg_tokens:.3f}"
        step_metrics_dict[f"assistant_step_avg_cost_pte_{step}"] = f"{avg_cost:.3f}"

    results = {
        "file_name": model_name,
        "gamma": f"{gamma:.6f}",
        "average_score": f"{avg_score:.3f}",
        "score_distribution": dict(score_dist),
        "average_tool_turns": f"{avg_tool_turns:.3f}",
        "tool_turns_distribution": dict(tool_turns_dist),
        "average_decode_tokens": f"{avg_decode_tokens:.3f}",
        "average_prefill_tokens_kv": f"{avg_prefil_tokens_kv:.3f}",
        "average_weighted_decode_tokens": f"{avg_weighted_decode_tokens:.3f}",
        "average_inference_cost_pte": f"{avg_inference_cost_pte:.3f}",
        "line_count": line_count,
        "score_1_cnt": score_1_cnt,
        "score_1_avg_infer_cost_pte": f"{score_1_avg_infer_cost_pte:.3f}",
        "score_0_cnt": score_0_cnt,
        "score_0_avg_infer_cost_pte": f"{score_0_avg_infer_cost_pte:.3f}",
    }
    results.update(step_metrics_dict)

    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate PTE cost metrics from evaluation results")
    parser.add_argument("--jsonl-path", type=str, required=True, help="Path to evaluation results JSONL file")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--gamma", type=float, required=True, help="Gamma parameter for PTE calculation")
    parser.add_argument("--model-name", type=str, required=True, help="Model name")
    parser.add_argument("--output-csv", type=str, default=None, help="Output CSV file path (optional)")
    
    args = parser.parse_args()
    
    if os.path.exists(args.jsonl_path) and os.path.exists(args.tokenizer_path):
        print(f"Processing {args.jsonl_path} with gamma={args.gamma}...")
        result = process_jsonl(args.jsonl_path, args.tokenizer_path, args.gamma, args.model_name)
        if result:
            if args.output_csv:
                # Write to CSV
                with open(args.output_csv, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=result.keys(), delimiter='\t', extrasaction='ignore')
                    writer.writeheader()
                    writer.writerow(result)
                print(f"Results written to {args.output_csv}")
            else:
                # Print to stdout
                import json
                print(json.dumps(result, indent=2))
        print("Done.")
    else:
        print(f"Error: File not found: {args.jsonl_path} or {args.tokenizer_path}")

