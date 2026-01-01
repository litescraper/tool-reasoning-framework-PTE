"""
Timing Analysis Script

This script analyzes timing data from evaluation results with timing logs.
It generates correlation analysis between latency, tokens, and PTE metrics.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer
from scipy.stats import pearsonr


def process_file(filepath, tokenizer, gamma, max_lines=None):
    """
    Process evaluation results file and extract timing, token, and PTE data.
    
    Args:
        filepath: Path to evaluation results JSONL file
        tokenizer: Tokenizer instance
        gamma: Gamma parameter for PTE calculation
        max_lines: Maximum number of lines to process (None for all)
        
    Returns:
        Tuple of (step_latency_data, step_token_data, step_pte_data, processed_count)
    """
    step_latency_data = defaultdict(list)
    step_token_data = defaultdict(list)
    step_pte_data = defaultdict(list)

    if not os.path.exists(filepath):
        print(f"[Error] File not found: {filepath}")
        return {}, {}, {}, 0

    print(f"Reading and calculating: {filepath} ...")
    if max_lines is not None:
        print(f"Limiting to first {max_lines} lines")

    line_count = 0
    processed_count = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
            if max_lines is not None and line_count > max_lines:
                break
                
            if not line.strip(): 
                continue
            try:
                data = json.loads(line)
                processed_count += 1
            except json.JSONDecodeError:
                continue

            # Extract Step Latency (per-step time)
            timeline = data.get('time_stats', {}).get('timeline', [])
            for item in timeline:
                if item.get('type') == 'llm_inference':
                    t_idx = item.get('turn_idx')
                    lat = item.get('detail', {}).get('latency_seconds')
                    if t_idx is not None and lat is not None:
                        step_latency_data[int(t_idx)].append(lat)

            # Calculate Token and PTE
            messages = data.get("messages", [])
            sys_and_user_prompt = data.get("input", {}).get("messages", [])
            if len(sys_and_user_prompt) >= 2:
                sys_prompt = sys_and_user_prompt[0]["content"]
                user_prompt = sys_and_user_prompt[1]["content"]
                history_len = len(tokenizer.encode(str(sys_prompt), add_special_tokens=False)) + \
                             len(tokenizer.encode(str(user_prompt), add_special_tokens=False))
            else:
                history_len = 0
            
            assistant_step_cnt = 1

            for msg in messages:
                content = msg.get("content", "") or ""
                reasoning = msg.get("reasoning_content", "") or ""
                tokens = tokenizer.encode(str(content), add_special_tokens=False) + \
                         tokenizer.encode(str(reasoning), add_special_tokens=False)
                curr_len = len(tokens)

                if msg.get('role') == 'assistant':
                    # Calculate metrics
                    d_prefill = history_len
                    d_decode = curr_len
                    c_mem = d_prefill * d_decode
                    pte_val = d_prefill + (gamma * c_mem)
                    
                    step_token_data[assistant_step_cnt].append(d_decode)
                    step_pte_data[assistant_step_cnt].append(pte_val)
                    assistant_step_cnt += 1
                
                history_len += curr_len

    print(f"Processed {processed_count} valid records")
    return step_latency_data, step_token_data, step_pte_data, processed_count


def plot_correlation_analysis(x_steps, y_lat_step, y_tok_step, y_pte_step, filename, n_samples=None):
    """
    Generate correlation analysis plot.
    All metrics are per-step (not cumulative).
    """
    if len(x_steps) == 0:
        return

    base_time = y_lat_step
    base_tokens = y_tok_step
    step_pte = y_pte_step

    # Calculate correlation coefficients
    r_token, p_token = pearsonr(base_time, base_tokens)
    r_pte, p_pte = pearsonr(base_time, step_pte)
    
    if n_samples is None:
        n_samples = len(base_time)

    # Normalization
    max_time = np.max(base_time) if np.max(base_time) > 0 else 1
    max_tok = np.max(base_tokens) if np.max(base_tokens) > 0 else 1
    max_pte = np.max(step_pte) if np.max(step_pte) > 0 else 1
    
    norm_time = base_time / max_time
    norm_tokens = base_tokens / max_tok
    norm_step_pte = step_pte / max_pte

    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 7))

    color_time = '#1f77b4'
    ax1.set_xlabel('Turn Index (Step)', fontsize=24)
    ax1.set_ylabel('Per-Step Wall Clock Time (s)', color=color_time, fontweight='bold', fontsize=24)
    line1 = ax1.plot(x_steps, base_time, color=color_time, marker='o', linewidth=2, alpha=0.6, label='Step Time (Baseline)')
    ax1.tick_params(axis='y', labelcolor=color_time, labelsize=20)
    ax1.tick_params(axis='x', labelsize=20)
    ax1.grid(True, linestyle='--', alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Normalized Metric Value (0.0 - 1.0)', color='black', fontsize=24)
    ax2.tick_params(axis='y', labelsize=20)
    
    color_tok = '#ff7f0e'
    label_tok = f'Step Tokens (Norm), r={r_token:.4f}, p={p_token:.4f}'
    line2 = ax2.plot(x_steps, norm_tokens, color=color_tok, marker='s', linestyle='--', linewidth=1.5, alpha=0.7, label=label_tok)
    
    color_pte = '#d62728'
    label_pte = f'Step PTE (Norm), r={r_pte:.4f}, p={p_pte:.4f}'
    line3 = ax2.plot(x_steps, norm_step_pte, color=color_pte, marker='^', linestyle='-.', linewidth=2, label=label_pte)

    lns = line1 + line2 + line3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left', fontsize=20, frameon=True, shadow=True)

    title_text = f'Correlation Analysis: Per-Step Latency vs Metrics\nN={n_samples}'
    plt.title(title_text, fontsize=26, pad=20)
    plt.tight_layout()

    plt.savefig(filename)
    pdf_filename = os.path.splitext(filename)[0] + ".pdf"
    plt.savefig(pdf_filename, format='pdf')

    print(f"Correlation analysis chart saved: {filename}")
    print(f"Correlation analysis chart saved (PDF): {pdf_filename}")
    print("-" * 30)
    print(f"Statistics:")
    print(f"  - Sample count (N): {n_samples}")
    print(f"Per-step correlation coefficient (r) and p-value:")
    print(f"  - Step Tokens: r = {r_token:.4f}, p = {p_token:.4f}")
    print(f"  - Step PTE:    r = {r_pte:.4f}, p = {p_pte:.4f}")
    print("-" * 30)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze timing data from evaluation results")
    parser.add_argument("--input-file", type=str, required=True, help="Path to evaluation results JSONL file")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--gamma", type=float, required=True, help="Gamma parameter for PTE calculation")
    parser.add_argument("--output-image", type=str, default="latency_trend_correlation.png", help="Output image path")
    parser.add_argument("--max-lines", type=int, default=None, help="Maximum number of lines to process (None for all)")
    
    args = parser.parse_args()
    
    # Initialize tokenizer
    print("Loading Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    except Exception as e:
        print(f"Tokenizer Error: {e}")
        return

    # Process data
    lat_d, tok_d, pte_d, n_samples = process_file(args.input_file, tokenizer, args.gamma, args.max_lines)

    # Align data
    common_steps = sorted(list(set(lat_d.keys()) & set(tok_d.keys()) & set(pte_d.keys())))
    if not common_steps:
        print("Warning: Data cannot be aligned, cannot plot.")
        return

    # Extract aligned average arrays
    x_steps = np.array(common_steps)
    y_lat_step = np.array([np.mean(lat_d[s]) for s in common_steps])
    y_tok_step = np.array([np.mean(tok_d[s]) for s in common_steps])
    y_pte_step = np.array([np.mean(pte_d[s]) for s in common_steps])

    # Plot
    print("\nGenerating charts...")
    plot_correlation_analysis(x_steps, y_lat_step, y_tok_step, y_pte_step, args.output_image, n_samples)
    print("\nAll charts generated.")


if __name__ == "__main__":
    main()

