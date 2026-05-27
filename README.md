# LightTokenizer

面向 `vLLM API Server Tokenizer` 长文本瓶颈的 LoPT 风格优化实验工程。

这个仓库沉淀了完整的一套基准测试、参数搜索、精度校验与报告生成流程，目标是回答一个非常具体的问题：

在 Agent 场景下，面对 `1M` 级超长输入时，能否在**不改远端环境依赖、不启动远端 vLLM 服务、严格保持 Token 输出完全一致**的前提下，用 LoPT 风格的多进程方案显著降低 Tokenizer 端到端耗时？

建议按下面顺序阅读：

1. [docs/agent_quickstart.md](docs/agent_quickstart.md)
2. [docs/project_status.md](docs/project_status.md)
3. [docs/architecture.md](docs/architecture.md)
4. [index.html](index.html)

## 项目背景

在长上下文 Agent 推理链路中，Prefill 阶段缓存命中率已经很高，PD 分离后推理侧压力往往不是主瓶颈；但 Tokenizer 无法像 KV Cache 一样跨请求复用，因此每次请求都必须重新做文本切分与编码。

vLLM 原生已经有异步线程池优化，核心路径会进入：

```text
OpenAIServingTokenization.create_tokenize
  -> OpenAIServingRender.preprocess_completion
  -> HfRenderer.render_cmpl_async
  -> BaseRenderer._tokenize_prompt_async
  -> AsyncMicrobatchTokenizer.encode
```

这条链路对**多请求并发**是友好的，但对于**单条超长输入**，底层仍然基本表现为“一次超长 Tokenizer 调用”。当输入规模上升到 `512k`、`720k`、`1024k` 字符时，Tokenizer 很容易成为 API Server 侧最核心的 CPU 瓶颈。

本项目围绕这个问题，基于最新本地 `vLLM` 源码构建基线，并实现 LoPT 风格的第一版多进程优化原型。

## 目标与问题定义

本项目聚焦以下目标：

- 复用 vLLM 原生 Tokenizer 链路做基线测试
- 基于真实中文、真实英文网页语料做长文本输入
- 对 `DeepSeek-V4-Pro` 与 `Qwen3.5` 两类 Tokenizer 做完整搜索
- 搜索 `worker_processes / chunk_count / overlap_chars` 的最优组合
- 拆分 LoPT 分段耗时，严格统计：
  - `原生 E2E 耗时`
  - `原生 Tokenizer 纯耗时`
  - `LoPT E2E 耗时`
  - `多进程分发 + 子进程处理 + 回收耗时`
  - `chunk 重叠去冗余耗时`
- 对全部最终配置做精度校验，要求 Token IDs 完全一致

## 方案概述

### 1. 原生 vLLM 基线

基线不启动 `vllm serve`，直接复用本地源码 `/home/light/vllm` 中真正参与 API Server 文本预处理的实现路径，对 `HfRenderer._tokenize_prompt_async` 和 `AsyncMicrobatchTokenizer.encode` 进行端到端与纯 Tokenizer 时间测量。

这意味着基线不是“自己手写一个 HF tokenizer 测试”，而是尽量贴近 vLLM 当前架构下的真实链路。

### 2. LoPT v1 多进程方案

LoPT v1 的核心思路是：

```text
超长文本
  -> 按字符切成多个带 overlap 的 chunk
  -> 分发到多个子进程并行编码
  -> 子进程返回 token ids + offsets
  -> 父进程依据 token id + 全局字符区间去重拼接
  -> 输出最终 token ids
```

当前工程中的第一版实现特点：

- 采用**多进程**而不是多线程，绕开 Python 侧 GIL 与单进程局部瓶颈
- 每个 worker 持有独立 tokenizer 实例
- 使用带 overlap 的分块，避免直接硬切导致边界 token 不一致
- 父进程基于 `offset_mapping` 做重叠区去冗余
- 一旦出现异常，**直接回退串行逻辑**

### 3. 为什么需要 overlap 去冗余

如果把超长字符串直接硬切，chunk 边界附近的分词结果常常会发生变化，最终 token IDs 无法与原生串行编码保持一致。因此 LoPT 不是简单“切块并行”就结束，而必须处理：

- 边界 token 对齐
- 重叠区重复 token 去除
- 全局字符位置回映

这部分工作既影响正确性，也会带来额外耗时，所以本项目将它单独拆成 `chunk_dedup_time_ms` 指标进行统计。

## 论文参考

本项目实现参考了用户提供的 LoPT 论文思路：

- `2511.04952v1.pdf`

由于本仓库关注的是 **vLLM API Server Tokenizer 侧可落地验证**，因此实现时做了工程化取舍，重点放在以下几个方面：

- 与 vLLM 原生链路的可对齐性
- 在既有环境中直接运行，不改远端依赖
- 可以批量搜索多种参数组合
- 能输出精度、分段耗时、最佳配置与完整 HTML 报告

也就是说，这里不是论文的逐字复刻，而是一个**面向现有 vLLM Tokenizer 场景的 LoPT v1 工程实现**。

## 实现说明

### 1. 输入数据

输入数据不是随机文本，而是真实网页语料扩展而来，分别准备：

- 真实中文网页语料
- 真实英文网页语料

长度覆盖：

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

### 2. 搜索变量

本轮完整搜索覆盖：

- 模型：
  - `DeepSeek-V4-Pro`
  - `Qwen3.5`
- 语言：
  - `zh`
  - `en`
- 输入长度：
  - `1k ~ 1024k`
- `worker_processes`
  - `1, 2, 4, 8, 16, 32, 64`
- `chunk_count`
  - 多档离散配置，随长度联合搜索
- `overlap_chars`
  - 多档离散配置，联合搜索

不搜索的项：

- `max_retry_rounds`
- `min_match_tokens`

异常场景统一直接回退串行逻辑。

### 3. 打点口径

原生链路统计：

- `native_e2e_time_ms`
- `native_tokenizer_time_ms`

LoPT 链路统计：

- `chat_template_time_ms`
- `mp_dispatch_process_collect_time_ms`
- `chunk_dedup_time_ms`
- `lopt_e2e_time_ms`

其中：

```text
lopt_e2e_time_ms
  = chat_template_time_ms
  + mp_dispatch_process_collect_time_ms
  + chunk_dedup_time_ms
```

### 4. 精度判定

LoPT 方案只有在以下条件全部满足时才算有效：

- 输出 token 数量一致
- token IDs 完全一致
- token hash 一致

这也是最终汇总与报告筛选最佳配置的前提。

## 当前工程内容

```text
benchmarks/
  collect_env_info.py
  compare_replay_results.py
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
  project_status.md
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
  spotcheck_codex_20260527/
  spotcheck_codex_20260527_r3/

index.html
AGENTS.md
README.md
```

## 最终结果展示

### 1. 总体完成情况

- 完整搜索候选总数：`7859`
- 最终 replay 结果总数：`48`
- 最终 replay 精度结论：`全部 exact match`

这意味着最终汇总中的所有最佳配置都通过了严格精度校验。

### 2. 代表性结果

| 场景 | 原生 E2E (ms) | 原生 Tokenizer (ms) | LoPT E2E (ms) | 多进程段 (ms) | 去冗余段 (ms) | E2E 加速比 | Tokenizer 加速比 | Tokenizer 降幅 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek-V4-Pro / en / 1k | 3.675 | 3.489 | 0.927 | 0.841 | 0.086 | 3.964x | 4.149x | 75.896% |
| DeepSeek-V4-Pro / zh / 1024k | 1664.239 | 1654.878 | 592.428 | 324.860 | 267.568 | 2.809x | 5.094x | 80.370% |
| DeepSeek-V4-Pro / en / 1024k | 741.529 | 737.987 | 281.013 | 178.287 | 102.726 | 2.639x | 4.139x | 75.841% |
| Qwen3.5 / en / 512k | 431.609 | 429.726 | 131.468 | 94.009 | 37.459 | 3.283x | 4.571x | 78.124% |
| Qwen3.5 / zh / 1024k | 1298.834 | 1290.088 | 543.928 | 280.161 | 263.767 | 2.388x | 4.605x | 78.284% |
| Qwen3.5 / en / 1024k | 804.694 | 800.860 | 276.132 | 173.767 | 102.365 | 2.914x | 4.609x | 78.302% |

### 3. 1024k 最优配置示例

| 模型 | 语言 | workers | chunk_count | overlap_chars | LoPT E2E (ms) | E2E 加速比 | Tokenizer 加速比 | 精度 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DeepSeek-V4-Pro | zh | 16 | 64 | 128 | 592.428 | 2.809x | 5.094x | exact |
| DeepSeek-V4-Pro | en | 32 | 128 | 256 | 281.013 | 2.639x | 4.139x | exact |
| Qwen3.5 | zh | 16 | 64 | 64 | 543.928 | 2.388x | 4.605x | exact |
| Qwen3.5 | en | 32 | 128 | 64 | 276.132 | 2.914x | 4.609x | exact |

### 4. Spot Check 复验

当前仓库还保留了远端 spot check 结果，对 4 个代表性 case 做了复验：

- `DeepSeek-V4-Pro / en / 1k`
- `DeepSeek-V4-Pro / zh / 1024k`
- `Qwen3.5 / en / 512k`
- `Qwen3.5 / zh / 1024k`

复验结论：

- exact match：全部通过
- token count：全部通过
- token hash：全部通过

对应文件：

- [results/spotcheck_codex_20260527_r3/spotcheck_summary.md](results/spotcheck_codex_20260527_r3/spotcheck_summary.md)
- [results/spotcheck_codex_20260527_r3/spotcheck_compare.json](results/spotcheck_codex_20260527_r3/spotcheck_compare.json)

### 5. 完整结果查看方式

完整结果不要只看 README，建议直接打开：

- [index.html](index.html)

HTML 报告中已经包含：

- 项目背景
- 原生瓶颈分析
- vLLM Tokenizer 执行链路
- LoPT 原理与字符流程图
- 实验环境信息
- 最优配置总览
- 原生 vs LoPT 曲线图 / 柱状图
- 全量搜索结果折叠浏览
- 最终 replay 数据折叠浏览

## 快速开始

### 如果你是新的 session / agent

先看：

1. [docs/agent_quickstart.md](docs/agent_quickstart.md)
2. [docs/architecture.md](docs/architecture.md)
3. [docs/results_guide.md](docs/results_guide.md)
4. [docs/repro_commands.md](docs/repro_commands.md)

其中 `README.md` 负责给出全局背景，`agent_quickstart.md` 负责帮助新 session 快速接手项目。

### 如果你只想看最终结论

直接打开：

- [index.html](index.html)

### 如果你只想验证几个 case

使用：

- `benchmarks/replay_best_configs.py`

这个脚本已经支持按以下维度过滤：

- `--families`
- `--languages`
- `--lengths`

适合做 spot check，而不必重新跑完整搜索。

## 复现与扩展入口

### 1. 完整搜索

核心脚本：

- `benchmarks/search_lopt_configs.py`

用于对不同模型、语言、长度与参数组合做完整搜索。

### 2. 搜索结果后处理

- `benchmarks/postprocess_search_results.py`

用于合并候选、筛选每组最优配置、输出 `best_configs` 和 `worker_best_configs`。

### 3. 最优配置 replay

- `benchmarks/replay_best_configs.py`

用于对筛出的最佳配置重新复测，生成最终表格与对比结果。

### 4. HTML 报告生成

- `benchmarks/generate_html_report.py`

生成仓库根目录的最终报告：

- [index.html](index.html)

### 5. 环境信息采集

- `benchmarks/collect_env_info.py`

输出：

- [results/benchmark_env_info.json](results/benchmark_env_info.json)

## 结果文件说明

建议重点关注以下文件：

- 最终报告：
  - [index.html](index.html)
- 最终 replay：
  - [results/replay_merged_20260526/final_replay_results.json](results/replay_merged_20260526/final_replay_results.json)
  - [results/replay_merged_20260526/final_replay_results.csv](results/replay_merged_20260526/final_replay_results.csv)
  - [results/replay_merged_20260526/final_replay_tables.md](results/replay_merged_20260526/final_replay_tables.md)
- 全量搜索汇总：
  - [results/search_merged_20260526/search_detail.jsonl](results/search_merged_20260526/search_detail.jsonl)
  - [results/search_merged_20260526/best_configs.json](results/search_merged_20260526/best_configs.json)
  - [results/search_merged_20260526/worker_best_configs.json](results/search_merged_20260526/worker_best_configs.json)
- 环境信息：
  - [results/benchmark_env_info.json](results/benchmark_env_info.json)

## 注意事项

- 基线复用的是本地最新 `vLLM` 源码：`/home/light/vllm`
- 不需要启动远端 `vllm serve`
- 不应修改远端机器依赖或包版本
- 结果解释时应优先使用最终 replay 数据，而不是仅看 search 阶段候选值
- 长文本场景下，LoPT 的收益和 `worker_processes / chunk_count / overlap_chars` 强相关，不能把某一组最优参数简单外推到所有长度

如果你要继续扩展这套实验，最好的入口仍然是：

1. 先读 [docs/agent_quickstart.md](docs/agent_quickstart.md)
2. 再看 [docs/repro_commands.md](docs/repro_commands.md)
3. 最后对照 [index.html](index.html) 和 `results/` 中的结构化数据继续推进
