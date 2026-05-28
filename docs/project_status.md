# Project Status

## 1. Completed

- native vLLM tokenizer baseline benchmark
- LoPT v1 multi-process tokenizer implementation
- LoPT v2.1 modular tokenizer implementation
- DeepSeek full search
- Qwen full search
- merged search ranking
- final replay for best configurations
- full HTML report with embedded data browser
- benchmark host environment capture

## 2. Main Result Counts

- v2.1 merged search candidate rows: `7263`
- v2.1 final replay rows: `48`
- v2.1 final replay exact-match status: `all exact`

## 3. Current Report

- root report: [index.html](index.html)

## 4. Important Result Files

- v2.1 merged best configs:
  `results/search_merged_v2_1_20260528/best_configs.json`
- v2.1 merged worker-best configs:
  `results/search_merged_v2_1_20260528/worker_best_configs.json`
- v2.1 final replay:
  `results/replay_merged_v2_1_20260528/final_replay_results.json`
- v1 final replay reference:
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
- v2.1 average E2E gain over v1:
  `1.488x`
- v2.1 average E2E reduction:
  `41.93%`

## 6. Recommended Reading Order

1. [agent_quickstart.md](agent_quickstart.md)
2. [optimization_journal.md](optimization_journal.md)
3. [architecture.md](architecture.md)
4. [benchmark_methodology.md](benchmark_methodology.md)
5. [results_guide.md](results_guide.md)
6. [repro_commands.md](repro_commands.md)

## 7. Recommended Next Actions For A New Session

If you only need validation:

- run a filtered replay spot check against `results/replay_merged_v2_1_20260528/final_replay_results.json`
- compare against the v1 replay reference when checking regressions

If you need reporting:

- regenerate `index.html`

If you need a new search:

- start from `search_lopt_configs.py`
- then postprocess
- then replay
- then regenerate the report

If you need to continue optimization work:

- read `optimization_journal.md` first
- append a new journal section for each measurable optimization step
