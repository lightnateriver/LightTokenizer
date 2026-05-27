# Agent Quickstart

This is the first document a new session should read.

## 1. What This Project Is

This repository benchmarks a LoPT-style tokenizer optimization against the
native vLLM tokenizer path used by API Server request preprocessing.

Native path of interest:

```text
OpenAIServingTokenization.create_tokenize
  -> OpenAIServingRender.preprocess_completion
  -> HfRenderer.render_cmpl_async
  -> BaseRenderer._tokenize_prompt_async
  -> AsyncMicrobatchTokenizer.encode
```

LoPT v1 path:

- split long text into overlapped chunks
- tokenize chunks in parallel worker processes
- merge chunk token IDs using overlap-aware dedup based on offsets

## 2. Constraints

- vLLM source of truth: `/home/light/vllm`
- Do not start remote `vllm serve`
- Do not modify remote machine dependencies / packages
- Benchmark should reuse the native vLLM tokenizer path above
- CPU affinity used in the main benchmark:
  `29-31,40-79,155-159`

## 3. Current Status

Completed benchmark artifacts already exist locally.

Important result sets:

- DeepSeek full search:
  `results/search_full_20260526/`
- Qwen full search:
  `results/search_qwen_full_20260526/`
- merged search:
  `results/search_merged_20260526/`
- final replay:
  `results/replay_merged_20260526/`

Current report:

- [../index.html](../index.html)

Environment snapshot:

- [../results/benchmark_env_info.json](../results/benchmark_env_info.json)

## 4. Where To Look First

If you want to understand the system:

1. [project_status.md](project_status.md)
2. [architecture.md](architecture.md)
3. [vllm_tokenizer_flow.md](vllm_tokenizer_flow.md)
4. [benchmark_methodology.md](benchmark_methodology.md)

If you want to reproduce or extend:

1. [repro_commands.md](repro_commands.md)
2. [results_guide.md](results_guide.md)

## 5. Main Scripts

- `benchmarks/search_lopt_configs.py`
  full configuration search
- `benchmarks/postprocess_search_results.py`
  merge / rank search outputs
- `benchmarks/replay_best_configs.py`
  replay best configs for final measurements
- `benchmarks/generate_html_report.py`
  generate the final HTML report
- `benchmarks/collect_env_info.py`
  capture benchmark host environment
- `benchmarks/vllm_tokenizer_bench.py`
  native baseline harness
- `benchmarks/lopt_tokenizer.py`
  LoPT v1 implementation

## 6. Fast Path: Run A Few Cases

Use `replay_best_configs.py` with filters to validate a subset:

- by model family
- by language
- by length

The script now supports:

- `--families`
- `--languages`
- `--lengths`

This is the safest way to do spot checks without rerunning full search.

## 7. Full Search Path

To rerun full search, the flow is:

1. prepare corpora
2. run `search_lopt_configs.py`
3. run `postprocess_search_results.py`
4. run `replay_best_configs.py`
5. run `collect_env_info.py`
6. run `generate_html_report.py`

See [repro_commands.md](repro_commands.md) for exact commands.

## 8. Precision Standard

The optimization is only acceptable if:

- token count matches
- token IDs match exactly
- token hash matches exactly

For quick reruns, timing may vary slightly, but exact-match correctness must not.
