# Benchmark Methodology

## 1. Objective

Measure native vLLM tokenizer performance against LoPT v1 / v2.1 under:

- identical models
- identical input corpora
- identical input lengths
- exact token-ID correctness checks

## 2. Models

The benchmark currently covers:

- `DeepSeek-V4-Pro`
- `Qwen3.5`

## 3. Corpora

Two corpora are used:

- pure Chinese real-web visible text
- pure English real-web visible text

The corpora are character-based because the required benchmark lengths are:

- `1k`
- `4k`
- `8k`
- `16k`
- `32k`
- `64k`
- `128k`
- `256k`
- `512k`
- `720k`
- `880k`
- `1024k`

If fetched source material is insufficient, real-web blocks are cycled to reach
the target size while staying within the same language corpus.

## 4. Native Baseline

The native baseline reuses the same offline preprocessing path as vLLM API
Server tokenization requests.

Measured native metrics:

- `native_e2e_time_ms`
- `native_tokenizer_time_ms`

Where:

- `native_e2e_time_ms`
  is measured from request entry to returned token IDs
- `native_tokenizer_time_ms`
  is measured inside `AsyncMicrobatchTokenizer.encode`

## 5. LoPT v1 / v2.1

LoPT v1 uses:

- `worker_processes`
- `chunk_count`
- `overlap_chars`

Measured LoPT metrics:

- `chat_template_time_ms`
- `mp_dispatch_process_collect_time_ms`
- `chunk_dedup_time_ms`
- `lopt_e2e_time_ms`

Where:

- `lopt_e2e_time_ms = chat_template_time_ms + mp_dispatch_process_collect_time_ms + chunk_dedup_time_ms`

In this completion-style benchmark, `chat_template_time_ms` is effectively `0.0`.

LoPT v2.1 keeps the same exact-match target and search dimensions, but splits
collect-side timing more finely so we can see:

- submit time
- collect time
- child compute time
- return tail time
- receive lag

## 6. Search Variables

The main search variables are:

- input length
- model family
- language
- `worker_processes`
- `chunk_count`
- `overlap_chars`

Fixed variables:

- `min_match_tokens = 2`
- no online retry for LoPT v1 search acceptance
- failed candidates fall back directly to native serial logic

## 7. Candidate Acceptance

A configuration is only considered valid if:

- it completes successfully
- it does not rely on fallback
- its final token IDs match native output exactly

## 8. Final Selection

Selection happens in two stages:

1. choose best candidate for each `worker_processes` value
2. choose final best candidate for each:
   - model family
   - language
   - input length

Then replay is performed on those final best configurations.

## 9. Precision Standard

The benchmark treats exactness as mandatory.

For a replay case to pass:

- `exact_match == true`
- `native_output_tokens == lopt_output_tokens`
- `native_token_hash == lopt_token_hash`

## 10. Timing Interpretation

Timing is expected to fluctuate slightly across reruns.

For spot checks:

- token exactness must remain identical
- timings should remain in the same performance band
- small timing drift does not invalidate the result
