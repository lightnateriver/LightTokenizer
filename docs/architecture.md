# Architecture

## 1. Native vLLM Baseline

The benchmark reuses the tokenizer path that matters for API Server-side text
tokenization. The main path is:

```text
OpenAIServingTokenization.create_tokenize
  -> OpenAIServingRender.preprocess_completion
  -> HfRenderer.render_cmpl_async
  -> BaseRenderer._tokenize_prompt_async
  -> AsyncMicrobatchTokenizer.encode
```

Important properties:

- the benchmark does not launch `vllm serve`
- the benchmark calls the latest local vLLM source directly
- tokenizer pure time is measured at `AsyncMicrobatchTokenizer.encode`
- end-to-end time is measured at the outer tokenize request boundary

## 2. Native Async Threading Behavior

`AsyncMicrobatchTokenizer.encode` queues requests and uses a shared thread pool.
This is good for concurrency across requests, but a single ultra-long prompt is
still processed as one underlying tokenizer call.

That is the bottleneck this repository targets.

## 3. LoPT v1 Design

LoPT v1 in this repository uses:

- character-based chunk splitting
- overlap between adjacent chunks
- per-process HF fast tokenizer instances
- parent-side overlap-aware dedup using offset mappings

High-level flow:

```text
long text
  -> split into k overlapped chunks
  -> dispatch chunks to p worker processes
  -> worker tokenization returns token IDs + offsets
  -> parent merges adjacent chunks using overlap matching
  -> final token IDs
```

## 4. Exactness Rule

Two overlapping tokens are considered identical only when:

- token ID matches
- global character span matches

This is stricter than plain token ID matching and is required to keep output
identical to serial tokenization.

## 5. Main Implementation Files

- `benchmarks/vllm_tokenizer_bench.py`
  native vLLM baseline harness
- `benchmarks/lopt_tokenizer.py`
  LoPT v1 tokenizer implementation
- `benchmarks/search_lopt_configs.py`
  full configuration search
- `benchmarks/postprocess_search_results.py`
  merge and rank search outputs
- `benchmarks/replay_best_configs.py`
  replay best configs
- `benchmarks/generate_html_report.py`
  report generator
- `benchmarks/collect_env_info.py`
  host environment capture

## 6. Data Artifacts

Search and replay outputs are under `results/`:

- `search_full_20260526/`
- `search_qwen_full_20260526/`
- `search_merged_20260526/`
- `replay_merged_20260526/`

The main HTML report is published at the repository root as `index.html`.
