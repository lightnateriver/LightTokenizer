# Results Guide

## 1. Main Result Directories

### `results/search_full_20260526/`

DeepSeek-focused full search output.

Contains:

- `search_detail.jsonl`
- `search_detail.csv`
- `worker_best_configs.json`
- `worker_best_configs.csv`
- `best_configs.json`
- `best_configs.csv`
- `best_configs.md`
- `case_failures.json`
- `search_meta.json`

### `results/search_qwen_full_20260526/`

Qwen-focused full search output with the same structure.

### `results/search_merged_20260526/`

Merged and postprocessed search results across model families.

This is the main source for final best-config selection.

### `results/replay_merged_20260526/`

Replay output for final best configurations.

Most important files:

- `final_replay_results.json`
- `final_replay_results.csv`
- `final_replay_tables.md`

## 2. File Meanings

### `search_detail.jsonl`

One row per candidate configuration.

Includes:

- model family
- language
- length
- `worker_processes`
- `chunk_count`
- `overlap_chars`
- native timing
- LoPT timing
- exactness
- fallback info
- error text

### `worker_best_configs.json`

For each case and each worker-process value, this stores the best valid
candidate found under that worker setting.

### `best_configs.json`

For each:

- model family
- language
- input length

this stores the final best configuration selected after search postprocess.

### `final_replay_results.json`

This is the most important structured result file for final reporting.

Each record corresponds to one final best configuration replayed again with the
same benchmark logic.

## 3. HTML Report Inputs

`index.html` is generated from:

- `results/search_merged_20260526/search_detail.jsonl`
- `results/search_merged_20260526/worker_best_configs.json`
- `results/search_merged_20260526/best_configs.json`
- `results/replay_merged_20260526/final_replay_results.json`
- `results/benchmark_env_info.json`
- `results/en_sources.json`
- `results/zh_sources.json`
- `docs/vllm_tokenizer_flow.md`

## 4. Environment Snapshot

`results/benchmark_env_info.json` stores:

- benchmark host name
- `lscpu` summary
- memory summary
- raw `lscpu`
- raw `free -h`
- raw `/proc/meminfo`
- raw `numactl --hardware`

## 5. Spot Check Outputs

When validating a few selected cases, recommended output goes into:

- `results/spotcheck_*`

Those spot-check directories can be compared back to:

- `results/replay_merged_20260526/final_replay_results.json`

using:

- `benchmarks/compare_replay_results.py`

## 6. Which Result To Trust

For summary reporting, trust:

1. `results/replay_merged_20260526/final_replay_results.json`
2. `index.html`

For debugging or re-ranking, use:

1. `results/search_merged_20260526/search_detail.jsonl`
2. `results/search_merged_20260526/worker_best_configs.json`
3. `results/search_merged_20260526/best_configs.json`
