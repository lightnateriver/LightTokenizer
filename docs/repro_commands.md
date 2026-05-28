# Reproduction Commands

## 1. Environment Variables

Adjust these paths for your machine:

```bash
VLLM_SRC=/home/light/vllm
REPO_ROOT=/home/light/codes/LoPT
CORPUS_DIR=$REPO_ROOT/results
```

Remote benchmark host paths used in the recorded experiments:

```bash
REMOTE_ROOT=/root/LoPT
REMOTE_VLLM=$REMOTE_ROOT/vllm_latest
REMOTE_CORPUS=$REMOTE_ROOT/corpora
REMOTE_TOKENIZERS=$REMOTE_ROOT/tokenizers
REMOTE_OUT=$REMOTE_ROOT/out
```

## 2. Build Or Refresh Corpora

```bash
python3 -m benchmarks.real_web_corpus \
  --output-dir "$REPO_ROOT/results" \
  --target-chars 1048576 \
  --languages en zh
```

## 3. Run Full Search

Run the two family-specific searches independently, then merge them.

DeepSeek search:

```bash
python3 -m benchmarks.search_lopt_configs \
  --vllm-src "$VLLM_SRC" \
  --corpus-dir "$REPO_ROOT/results" \
  --output-dir "$REPO_ROOT/results/search_v2_1_deepseek_20260528" \
  --languages en zh \
  --families DeepSeek-V4-Pro \
  --worker-values 1,2,4,8,16,32,64 \
  --chunk-multipliers 0.5,1,2,4 \
  --overlap-values 32,64,128,256,512,1024,2048,4096,8192 \
  --renderer-workers 4 \
  --repeats 1 \
  --min-match-tokens 2 \
  --deepseek-tokenizer-path /path/to/deepseek/tokenizer
```

Qwen search:

```bash
python3 -m benchmarks.search_lopt_configs \
  --vllm-src "$VLLM_SRC" \
  --corpus-dir "$REPO_ROOT/results" \
  --output-dir "$REPO_ROOT/results/search_v2_1_qwen_20260528" \
  --languages en zh \
  --families Qwen3.5 \
  --worker-values 1,2,4,8,16,32,64 \
  --chunk-multipliers 0.5,1,2,4 \
  --overlap-values 32,64,128,256,512,1024,2048,4096,8192 \
  --renderer-workers 4 \
  --repeats 1 \
  --min-match-tokens 2 \
  --qwen-tokenizer-path /path/to/qwen/tokenizer
```

## 4. Merge Search Outputs

```bash
python3 -m benchmarks.postprocess_search_results \
  --input-dir \
    "$REPO_ROOT/results/search_v2_1_deepseek_20260528" \
    "$REPO_ROOT/results/search_v2_1_qwen_20260528" \
  --output-dir "$REPO_ROOT/results/search_merged_v2_1_20260528"
```

## 5. Replay Best Configurations

Full replay:

```bash
python3 -m benchmarks.replay_best_configs \
  --vllm-src "$VLLM_SRC" \
  --best-configs-json "$REPO_ROOT/results/search_merged_v2_1_20260528/best_configs.json" \
  --corpus-dir "$REPO_ROOT/results" \
  --output-dir "$REPO_ROOT/results/replay_merged_v2_1_20260528"
```

Filtered replay for spot checks:

```bash
python3 -m benchmarks.replay_best_configs \
  --vllm-src "$VLLM_SRC" \
  --best-configs-json "$REPO_ROOT/results/search_merged_v2_1_20260528/best_configs.json" \
  --corpus-dir "$REPO_ROOT/results" \
  --output-dir "$REPO_ROOT/results/spotcheck_v2_1_local" \
  --families DeepSeek-V4-Pro Qwen3.5 \
  --languages en zh \
  --lengths 1k 512k 1024k
```

## 6. Compare Spot Check Results Against Final Replay

```bash
python3 -m benchmarks.compare_replay_results \
  --reference-json "$REPO_ROOT/results/replay_merged_v2_1_20260528/final_replay_results.json" \
  --candidate-json "$REPO_ROOT/results/spotcheck_v2_1_local/final_replay_results.json"
```

## 7. Collect Benchmark Host Environment

```bash
python3 -m benchmarks.collect_env_info \
  --output-json "$REPO_ROOT/results/benchmark_env_info.json"
```

## 8. Generate Root HTML Report

```bash
python3 -m benchmarks.generate_html_report \
  --search-detail-jsonl "$REPO_ROOT/results/search_merged_v2_1_20260528/search_detail.jsonl" \
  --worker-best-json "$REPO_ROOT/results/search_merged_v2_1_20260528/worker_best_configs.json" \
  --best-json "$REPO_ROOT/results/search_merged_v2_1_20260528/best_configs.json" \
  --replay-json "$REPO_ROOT/results/replay_merged_v2_1_20260528/final_replay_results.json" \
  --flow-doc "$REPO_ROOT/docs/vllm_tokenizer_flow.md" \
  --output-html "$REPO_ROOT/index.html" \
  --vllm-src "$VLLM_SRC" \
  --search-input-dirs \
    "$REPO_ROOT/results/search_v2_1_deepseek_20260528" \
    "$REPO_ROOT/results/search_v2_1_qwen_20260528" \
  --corpus-meta-dir "$REPO_ROOT/results" \
  --env-info-json "$REPO_ROOT/results/benchmark_env_info.json"
```

## 9. Remote Spot Check Pattern

The recorded environment used:

```bash
/root/LoPT/vllm_latest
/root/LoPT/tokenizers
/root/LoPT/corpora
/root/LoPT/out
```

Typical remote spot check flow:

1. create a subset `best_configs.json`
2. run remote `replay_best_configs.py`
3. compare returned results with the local reference replay JSON
