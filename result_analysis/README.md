# Result Analysis Tools

This directory contains scripts for post-processing evaluation results.

## Scripts

### 1. `calculate_pte.py` - PTE Cost Analysis

Calculates PTE (Prefill-Token-Equivalent) cost metrics from evaluation results.

**PTE Cost Formula:**
- Cost = D_prefill + gamma * C_mem
- D_prefill = sum(tokens_sys) + sum(tokens_user) + sum(tokens_tool)
- C_mem = sum_over_all_turns( L_seq * D_decode )

**Usage:**
```bash
python result_analysis/calculate_pte.py \
    --jsonl-path ./results/dataset_model/eval_results.jsonl \
    --tokenizer-path /path/to/tokenizer \
    --gamma 0.00329 \
    --model-name "ModelName" \
    --output-csv output.csv
```

**Output:**
- Average score, tool turns, token counts
- PTE cost metrics (D_prefill, C_mem, InferCost_PTE)
- Step-wise statistics
- Score-specific cost breakdowns

### 2. `analyze_timing.py` - Timing Analysis

Analyzes timing data from evaluation results with timing logs. Generates correlation analysis between latency, tokens, and PTE metrics.

**Usage:**
```bash
cd tool-reasoning-framework
python result_analysis/analyze_timing.py \
    --input-file ./results/dataset_model_timing_log/model/eval_results.jsonl \
    --tokenizer-path /path/to/tokenizer \
    --gamma 0.000718 \
    --output-image latency_correlation.png \
    --max-lines 100
```

**Example: Using Sample Timing Data**

We provide a sample timing data file (`example_timing_data.jsonl`) containing 100 sample records for testing the analysis script:

```bash
cd tool-reasoning-framework
python result_analysis/analyze_timing.py \
    --input-file ./result_analysis/example_timing_data.jsonl \
    --tokenizer-path /path/to/tokenizer \
    --gamma 0.000718 \
    --output-image ./result_analysis/example_latency_correlation.png
```

**Note**: The example file contains 100 sample records from a stress test evaluation. For full analysis, use results from your own stress test evaluations with timing logs enabled.

**Output:**
- Correlation analysis plot (PNG and PDF)
- Per-step latency vs tokens correlation
- Per-step latency vs PTE correlation
- Statistical correlation coefficients (r, p-values)

## Requirements

- `transformers`: For tokenizer
- `matplotlib`: For plotting
- `numpy`: For numerical operations
- `scipy`: For statistical analysis (Pearson correlation)

## Notes

- Both scripts support processing partial data (via `--max-lines`)
- Timing analysis requires evaluation results with `time_stats` field (from stress test pipeline)
- PTE calculation requires appropriate gamma values for your model

