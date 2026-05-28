# Agent Handoff Guide

新 session 进入这个仓库后，建议先读：

1. [README.md](README.md)
2. [docs/agent_quickstart.md](docs/agent_quickstart.md)
3. [docs/project_status.md](docs/project_status.md)
4. [docs/optimization_journal.md](docs/optimization_journal.md)
5. [docs/index.html](docs/index.html)

## 这是什么工程

这是一个围绕 `vLLM API Server Tokenizer` 长文本瓶颈做的 LoPT 风格优化实验仓库，当前主线为 `v2.1`。

对照的三条链路分别是：

- 原生 vLLM：
  `HfRenderer._tokenize_prompt_async -> AsyncMicrobatchTokenizer.encode`
- LoPT v1：
  多进程切块并行 + overlap 去冗余
- LoPT v2.1：
  在 v1 骨架上补充 submit / collect / worker / dedup 细粒度打点

## 当前仓库的结构约定

- `src/`
  保留原始 LightTokenizer v1 时代的历史实现与脚本
- `benchmarks/`
  当前主线基准、搜索、回放、报告生成代码
- `docs/index_v1_template.html`
  原始 v1 风格 HTML 模板
- `benchmarks/generate_docs_legacy_report.py`
  基于原始模板生成当前 `docs/index.html`
- `docs/index.html`
  最终报告，保留原始页面风格并补入 v2.1 内容

## 继续工作时必须遵守的约束

- vLLM 源码事实基线：`/home/light/vllm`
- 不启动远端 `vllm serve`
- 不修改远端机器依赖和包
- 真实 benchmark 默认复用原生 vLLM tokenizer 路径
- 远端主实验绑核范围：`29-31,40-79,155-159`

## 结果目录优先看什么

- `results/search_merged_v2_1_20260528/`
- `results/replay_merged_v2_1_20260528/`
- `results/replay_merged_20260526/`
- `results/benchmark_env_info.json`

## 新 session 常见入口

- 想快速理解背景：看 [docs/agent_quickstart.md](docs/agent_quickstart.md)
- 想看当前状态：看 [docs/project_status.md](docs/project_status.md)
- 想接着做优化：先看 [docs/optimization_journal.md](docs/optimization_journal.md)
- 想复现实验：看 [docs/repro_commands.md](docs/repro_commands.md)
- 想直接看报告：打开 [docs/index.html](docs/index.html)
