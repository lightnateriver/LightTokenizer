# Agent Handoff Guide

Read [docs/agent_quickstart.md](docs/agent_quickstart.md) first.

This repository packages a LoPT-style tokenizer acceleration benchmark built on
top of the latest local vLLM source tree. The benchmark compares:

- native vLLM tokenizer path:
  `HfRenderer._tokenize_prompt_async -> AsyncMicrobatchTokenizer.encode`
- LoPT v1 multi-process parallel tokenization with overlap-aware dedup

Core expectations for follow-up agents:

- Reuse `/home/light/vllm` as the source of truth for vLLM code analysis.
- Do not start `vllm serve` for this project.
- Do not change remote machine packages or dependencies.
- Prefer offline benchmark / replay / report flows documented in `docs/`.
- Before large reruns, check existing artifacts under `results/`.

Primary entry points:

- [README.md](README.md)
- [docs/agent_quickstart.md](docs/agent_quickstart.md)
- [docs/repro_commands.md](docs/repro_commands.md)
- [index.html](index.html)
