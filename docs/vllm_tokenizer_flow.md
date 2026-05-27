# vLLM Tokenizer Flow

This note focuses on the text-to-token-ID path that matters for the benchmark:

- input is plain text
- no `vllm serve` process is started
- the benchmark reuses the same offline preprocessing code path as the API server
- tokenizer-only timing is pinned to `AsyncMicrobatchTokenizer.encode(...)`

The user explicitly pointed out the hot path to pay attention to:

- `HfRenderer._tokenize_prompt_async(...)`
- `AsyncMicrobatchTokenizer.encode(...)`

In the latest local repository, the concrete logic lives in `BaseRenderer._tokenize_prompt_async(...)`
and is inherited by `HfRenderer`, while `HfRenderer` adds the thread-safe HF tokenizer pool through
`maybe_make_thread_pool(...)`. For the completion-style text benchmark, that is the path we reuse.

## Key Source Files

- `vllm/entrypoints/serve/tokenize/serving.py`
- `vllm/entrypoints/serve/render/serving.py`
- `vllm/renderers/base.py`
- `vllm/renderers/hf.py`
- `vllm/utils/async_utils.py`
- `vllm/tokenizers/hf.py`

## End-to-End Logic

### 1. Request enters the tokenize endpoint logic

The closest API-server-side offline entry is:

- `OpenAIServingTokenization.create_tokenize(...)`

For a completion-style tokenize request:

1. `create_tokenize(...)` receives `TokenizeCompletionRequest`
2. `_check_model(...)` validates the requested model name
3. `openai_serving_render.preprocess_completion(...)` is called

### 2. Render layer converts request payload into renderer inputs

`OpenAIServingRender.preprocess_completion(...)` does:

1. normalize `prompt_input` into `prompts`
2. call `preprocess_cmpl(...)`
3. build `TokenizeParams` by `request.build_tok_params(...)`
4. call `renderer.render_cmpl_async(...)`

### 3. Renderer async pipeline

`HfRenderer.render_cmpl_async(...)` follows the renderer pipeline:

1. `render_prompts_async(...)`
2. `tokenize_prompts_async(...)`
3. `process_for_engine_async(...)`

For a plain text prompt, tokenization falls into:

1. `tokenize_prompt_async(...)`
2. `_tokenize_singleton_prompt_async(...)`
3. `_tokenize_prompt_async(...)`

### 4. The actual tokenizer hot path

Inside `_tokenize_prompt_async(...)`:

1. `get_async_tokenizer()` lazily creates `AsyncMicrobatchTokenizer`
2. `await tokenizer.encode(prompt_text, **encode_kwargs)`

This is the timing boundary used for the benchmark's "Tokenizer pure time".

### 5. AsyncMicrobatchTokenizer internals

`AsyncMicrobatchTokenizer.encode(...)` does not tokenize immediately. It:

1. creates a future for the request
2. computes a queue key from encode kwargs
3. pushes the request into an async queue
4. waits for the future to be fulfilled

The background batcher task:

1. groups pending requests with the same encode shape
2. waits up to `batch_wait_timeout_s`
3. if possible, performs a single batched tokenizer call
4. otherwise falls back to per-prompt calls
5. runs the blocking tokenizer work inside the shared `ThreadPoolExecutor`
6. resolves each waiting future

### 6. HF tokenizer thread-safety and pool behavior

`HfRenderer.__init__(...)` calls:

- `maybe_make_thread_pool(tokenizer, renderer_num_workers + 1)`

This wraps the fast tokenizer with a pool of deep-copied tokenizer instances so the public tokenizer
API is safe when renderer-side async work overlaps across requests.

### 7. Output becomes token IDs again

After tokenization:

1. `_tokenize_prompt_async(...)` returns `TokensPrompt(prompt_token_ids=...)`
2. `process_for_engine_async(...)` converts it to `EngineInput`
3. `create_tokenize(...)` extracts and flattens `prompt_token_ids`
4. `TokenizeResponse(tokens=..., count=...)` is returned

## ASCII Flowchart

```text
User text prompt
    |
    v
TokenizeCompletionRequest
    |
    v
OpenAIServingTokenization.create_tokenize()
    |
    +--> _check_model()
    |
    v
OpenAIServingRender.preprocess_completion()
    |
    v
OpenAIServingRender.preprocess_cmpl()
    |
    +--> request.build_tok_params()
    |
    v
HfRenderer.render_cmpl_async()
    |
    +--> render_prompts_async()
    |
    +--> tokenize_prompts_async()
            |
            v
        tokenize_prompt_async()
            |
            v
        _tokenize_singleton_prompt_async()
            |
            v
        _tokenize_prompt_async()
            |
            +--> get_async_tokenizer()
            |       |
            |       v
            |   AsyncMicrobatchTokenizer
            |
            v
        AsyncMicrobatchTokenizer.encode()
            |
            +--> queue.put((prompt, kwargs, future))
            |
            v
        _batch_encode_loop()
            |
            +--> collect microbatch
            +--> run tokenizer in shared ThreadPoolExecutor
            +--> set result futures
            |
            v
        token IDs returned to renderer
    |
    +--> process_for_engine_async()
    |
    v
EngineInput(prompt_token_ids=...)
    |
    v
OpenAIServingTokenization extracts token IDs
    |
    v
TokenizeResponse(tokens=[...], count=N)
```

## Benchmark Timing Points

The benchmark uses two timing scopes for the native path:

1. `E2E native time`
   - start: before `create_tokenize(...)`
   - end: after `TokenizeResponse` is returned

2. `Tokenizer pure time`
   - start: entering `AsyncMicrobatchTokenizer.encode(...)`
   - end: that `encode(...)` call returns token IDs to the renderer

For LoPT, the benchmark uses:

1. `E2E LoPT time`
   - full split -> dispatch -> child-process tokenization -> merge -> final token IDs

2. `LoPT pure time`
   - start: tasks submitted to the process pool
   - end: all child processes return chunk tokenization results

## Why the benchmark uses HfRenderer

The benchmark targets plain text tokenization rather than chat-template rendering. For this scope:

- the tokenizer hot path is the renderer-side async encode path
- `HfRenderer` is the right concrete renderer to reuse
- the DeepSeek-V4-Pro tokenizer assets are still used, but the benchmark stays on the
  `HfRenderer -> AsyncMicrobatchTokenizer.encode(...)` path that the user requested

That keeps the baseline aligned with the current tokenizer bottleneck being studied:
single-request long-text tokenization on the CPU side.
