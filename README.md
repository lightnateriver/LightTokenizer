# LightTokenizer

LoPT-style long-text tokenizer acceleration benchmark for vLLM API Server tokenization.

This repository packages:

- native vLLM tokenizer baseline benchmarking
- LoPT v1 multi-process tokenizer implementation
- full search tooling for `worker_processes / chunk_count / overlap_chars`
- replay tooling for best configurations
- a complete HTML report with full embedded benchmark data

Start here:

1. Read [docs/agent_quickstart.md](docs/agent_quickstart.md)
2. Review [docs/project_status.md](docs/project_status.md)
3. Review [docs/architecture.md](docs/architecture.md)
4. Open [index.html](index.html) for the latest benchmark report

## Project Goal

In long-context Agent scenarios, the tokenizer becomes a CPU bottleneck because:

- tokenizer results are not cacheable across requests in the general case
- native vLLM async tokenization improves cross-request concurrency
- but a single 1M-scale prompt is still handled as one underlying tokenizer call

This project evaluates whether a LoPT-style multi-process tokenizer can reduce
end-to-end tokenization latency while preserving exact token IDs.

## Repository Layout

```text
benchmarks/
  collect_env_info.py
  expand_existing_corpora.py
  generate_html_report.py
  lopt_tokenizer.py
  postprocess_search_results.py
  real_web_corpus.py
  replay_best_configs.py
  search_lopt_configs.py
  vllm_tokenizer_bench.py

docs/
  agent_quickstart.md
  architecture.md
  benchmark_methodology.md
  repro_commands.md
  results_guide.md
  vllm_tokenizer_flow.md

results/
  benchmark_env_info.json
  en_sources.json
  zh_sources.json
  search_full_20260526/
  search_qwen_full_20260526/
  search_merged_20260526/
  replay_merged_20260526/

index.html
AGENTS.md
README.md
```

## Included Workflows

- Native baseline benchmark
- LoPT full search
- Search result postprocess / merge
- Best-config replay
- HTML report generation
- Benchmark host environment capture

## Current Final Artifacts

- Final report: [index.html](index.html)
- Merged search results:
  [results/search_merged_20260526](results/search_merged_20260526)
- Final replay results:
  [results/replay_merged_20260526](results/replay_merged_20260526)
- Benchmark host environment:
  [results/benchmark_env_info.json](results/benchmark_env_info.json)

## Notes

- The benchmark reuses the latest local vLLM code in `/home/light/vllm`
- The benchmark does not require running a remote vLLM service
- The benchmark assumes real Chinese and English web-derived corpora
- The HTML report embeds all candidate rows and replay rows directly

For step-by-step commands, see [docs/repro_commands.md](docs/repro_commands.md).

For new sessions or delegated agents, also read [AGENTS.md](AGENTS.md).
