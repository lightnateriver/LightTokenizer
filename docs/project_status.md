# Project Status

## 1. Completed

- native vLLM tokenizer baseline benchmark
- LoPT v1 multi-process tokenizer implementation
- DeepSeek full search
- Qwen full search
- merged search ranking
- final replay for best configurations
- full HTML report with embedded data browser
- benchmark host environment capture

## 2. Main Result Counts

- merged search candidate rows: `7859`
- final replay rows: `48`
- final replay exact-match status: `all exact`

## 3. Current Report

- root report: [../index.html](../index.html)

## 4. Important Result Files

- merged best configs:
  `results/search_merged_20260526/best_configs.json`
- merged worker-best configs:
  `results/search_merged_20260526/worker_best_configs.json`
- final replay:
  `results/replay_merged_20260526/final_replay_results.json`
- benchmark host environment:
  `results/benchmark_env_info.json`

## 5. Example Best Cases

- top overall E2E speedup:
  `DeepSeek-V4-Pro / en / 1k`
- strong long-text E2E case:
  `Qwen3.5 / en / 512k`
- top tokenizer speedup:
  `DeepSeek-V4-Pro / zh / 1024k`

## 6. Recommended Reading Order

1. [agent_quickstart.md](agent_quickstart.md)
2. [architecture.md](architecture.md)
3. [benchmark_methodology.md](benchmark_methodology.md)
4. [results_guide.md](results_guide.md)
5. [repro_commands.md](repro_commands.md)

## 7. Recommended Next Actions For A New Session

If you only need validation:

- run a filtered replay spot check
- compare against `final_replay_results.json`

If you need reporting:

- regenerate `index.html`

If you need a new search:

- start from `search_lopt_configs.py`
- then postprocess
- then replay
- then regenerate the report
