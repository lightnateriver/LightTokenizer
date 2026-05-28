# LoPT v1.0 反推设计文档

本文档严格基于当前仓库 `/home/light/codes/LoPT` 的现有代码、脚本、文档和产物反推整理，不依赖历史对话上下文。目标是让一个全新会话只阅读本文件，就能理解 1.0 版本代码到底实现了什么、如何运行、边界在哪里、与最初设计意图大致有何关系。

需要特别说明两点：

1. 本仓库不是一个在线服务项目，而是一个**离线 benchmark / 参数搜索 / 报告生成工程**。
2. 文中关于“原始设计目标”的内容，均为**根据当前代码和命名反推**，不是来自历史需求文档。

---

## 1. 项目整体说明

### 1.1 项目名称 / 用途

- 项目名称：`LightTokenizer` / `LoPT`
- 当前 1.0 版本的实际用途：
  - 基于**本地最新 vLLM 源码**复用 Tokenizer 请求链路，做原生基线测试
  - 实现一个第一版 LoPT 风格的**多进程长文本 Tokenizer 并行加速原型**
  - 对不同模型、语言、输入长度、进程数、chunk 数和 overlap 参数做全量搜索
  - 对搜索出的最佳配置做 replay 复测
  - 生成结构化结果文件和最终 HTML 报告

它不是：

- 不是线上 API Server
- 不是 FastAPI 项目
- 不是推理服务本体
- 不是对 `/home/light/vllm` 上游源码的直接改造分支

### 1.2 基于什么框架 / 组件

从代码反推，1.0 版本主要由以下框架和组件组成：

- `Python 3` 命令行脚本工程
- `vLLM` 本地源码导入与类复用
- `transformers.PreTrainedTokenizerFast`
- Python 标准库：
  - `asyncio`
  - `concurrent.futures.ProcessPoolExecutor`
  - `argparse`
  - `csv`
  - `json`
  - `subprocess`
  - `urllib`
- 静态 HTML 报告生成器（纯 Python 生成 HTML）

可以概括为：

```text
Python CLI Benchmark Toolkit
  + local vLLM source reuse
  + HuggingFace fast tokenizer
  + multi-process LoPT prototype
  + static HTML reporting
```

### 1.3 整体架构

当前仓库实际可拆成 6 层：

```text
1. 数据准备层
   - benchmarks/real_web_corpus.py
   - benchmarks/expand_existing_corpora.py

2. 原生基线层
   - benchmarks/vllm_tokenizer_bench.py
   - 复用本地 vLLM TokenizeCompletionRequest -> OpenAIServingTokenization

3. LoPT 执行层
   - benchmarks/lopt_tokenizer.py
   - 文本切块 -> 多进程编码 -> overlap 匹配 -> 去重合并

4. 搜索 / 回放编排层
   - benchmarks/search_lopt_configs.py
   - benchmarks/postprocess_search_results.py
   - benchmarks/replay_best_configs.py
   - benchmarks/compare_replay_results.py

5. 环境与报告层
   - benchmarks/collect_env_info.py
   - benchmarks/generate_html_report.py

6. 文档与结果展示层
   - docs/*.md
   - index.html
   - results/*
```

### 1.4 核心流程

当前 1.0 的完整业务流程不是“服务接请求”，而是“离线实验流水线”：

```text
真实网页语料抓取 / 扩展
  -> 原生 vLLM Tokenizer 基线测量
  -> LoPT 多进程候选参数搜索
  -> 筛选每组最优配置
  -> 最优配置 replay 复测
  -> Spot check 对比
  -> 采集环境信息
  -> 生成 HTML 报告
```

如果聚焦单个测试 case，则流程是：

```text
输入文本
  -> 原生链路:
     TokenizeCompletionRequest
       -> OpenAIServingTokenization.create_tokenize
       -> OpenAIServingRender / HfRenderer
       -> AsyncMicrobatchTokenizer.encode
       -> Token IDs

  -> LoPT 链路:
     split_text
       -> ProcessPoolExecutor 提交 chunk
       -> 子进程 tokenizer encode + offsets
       -> position-aware overlap match
       -> merge_chunk_results
       -> build_inputs_with_special_tokens
       -> Token IDs

  -> exact match 对比
  -> 记录耗时 / hash / token count
```

---

## 2. 原始设计目标（根据代码反推）

### 2.1 原始需求

根据脚本命名、参数设计、注释和结果产物可以反推出，这个工程最初想解决的问题是：

1. 不启动远端 `vllm serve`
2. 不修改远端依赖和包
3. 直接利用**本地最新 vLLM 源码**，在远端更强 CPU 上做 Tokenizer 实验
4. 对比：
   - 原生 vLLM Tokenizer 路径
   - LoPT 风格多进程 Tokenizer 路径
5. 在真实中英文长文本上测量性能，并严格校验精度
6. 最终输出可复现的数据表和 HTML 报告

### 2.2 预期能力

从当前代码反推，原始预期能力至少包括：

- 支持真实英文 / 中文网页语料
- 支持多个长度档位：
  - `1k, 4k, 8k, 16k, 32k, 64k, 128k, 256k, 512k, 720k, 880k, 1024k`
- 支持多个 Tokenizer family：
  - `DeepSeek-V4-Pro`
  - `Qwen3.5`
- 支持搜索：
  - `worker_processes`
  - `chunk_count`
  - `overlap_chars`
- 支持拆分 LoPT 耗时：
  - chat template 时间
  - 多进程分发/处理/回收时间
  - chunk 去冗余时间
- 支持 exact match 校验
- 支持最终 replay 和 spot check

### 2.3 预期架构

从当前代码反推，最初设想的架构大概是：

```text
最新本地 vLLM 源码
  + 本地 Tokenizer 目录
  + 独立 LoPT 原型实现
  + 离线 benchmark harness
  + 参数搜索器
  + 结果后处理器
  + HTML 报告生成器
```

也就是说，原始思路不是“把 LoPT 直接揉进 vLLM 正式代码”，而是先做一个**与 vLLM 真实链路对齐的离线验证工程**。

---

## 3. 最终实际实现方案（100% 基于当前代码）

本节只描述当前代码里真实存在的实现。

### 3.1 核心模块结构

#### 3.1.1 `benchmarks/vllm_tokenizer_bench.py`

职责：

- 复用本地 vLLM 源码，构造一个**离线原生基线 harness**
- 直接调用 vLLM Tokenize 请求链路
- 对 `AsyncMicrobatchTokenizer.encode` 做纯 Tokenizer 打点
- 输出 JSON / CSV / Markdown

关键结构：

- `insert_vllm_src(vllm_src: Path)`
  - 动态将本地 vLLM 源码目录插入 `sys.path`

- Mock 配置类：
  - `MockHFConfig`
  - `MockModelConfig`
  - `MockParallelConfig`
  - `MockVllmConfig`

这些类的作用是：在**不启动真实引擎服务**的情况下，构造出能让 `HfRenderer`、`OpenAIServingRender`、`OpenAIServingTokenization` 正常工作的最小配置对象。

- `RunMetrics`
  - 单次运行的时间与 hash 统计

- `CaseSummary`
  - 汇总单个 case 的最终对比结果

- `NativeTokenizerTimer`
  - 对 `AsyncMicrobatchTokenizer.encode` 进行猴子补丁包装
  - 记录每次 encode 的耗时

- `NativeBenchmarkHarness`
  - 原生基线的核心入口
  - 负责初始化 tokenizer、renderer、serving 对象
  - 负责执行一次原生 tokenize 请求

核心方法：

- `NativeBenchmarkHarness.run_once(text)`
  - 构造 `TokenizeCompletionRequest`
  - 调用 `self.service.create_tokenize(...)`
  - 统计：
    - `e2e_ms`
    - `token_ms`
    - token count
    - token hash

- `NativeBenchmarkHarness.serial_encode(text, add_special_tokens=True)`
  - 直接用本地 `serial_tokenizer` 串行编码
  - 主要用于 fallback 和结果校验

- `run_benchmark(args)`
  - 完整执行 benchmark
  - 同时跑 native baseline 和 LoPT

- `write_outputs(output_dir, result)`
  - 落盘 `benchmark_results.json`
  - 落盘 `benchmark_tables.md`
  - 落盘 `benchmark_summary.csv`

#### 3.1.2 `benchmarks/lopt_tokenizer.py`

职责：

- 实现当前 1.0 的 LoPT 多进程并行 Tokenizer
- 完成：
  - 切块
  - 子进程编码
  - overlap 匹配
  - chunk 去冗余
  - special token 回补

关键结构：

- `_WORKER_TOKENIZER`
  - 进程内全局 tokenizer 句柄

- `_init_worker(tokenizer_path)`
  - 子进程初始化时加载 `PreTrainedTokenizerFast`

- `_tokenize_chunk(task)`
  - 子进程执行单个 chunk 的编码
  - 返回 `ChunkResult`

- `ChunkResult`
  - 字段：
    - `start`
    - `end`
    - `input_ids`
    - `offsets`

- `MatchResult`
  - 字段：
    - `left_start_index`
    - `right_start_index`
    - `length`

- `LoPTConfig`
  - 当前 1.0 的配置对象
  - 字段：
    - `tokenizer_path`
    - `processes`
    - `overlap_chars`
    - `min_match_tokens`
    - `initial_chunk_chars`
    - `max_retry_rounds`

注意：`max_retry_rounds` 在当前 1.0 代码中**没有真正发挥重试逻辑作用**，只是配置字段保留。

- `LoPTResult`
  - 一次 LoPT 调用的返回对象
  - 记录：
    - 最终 token ids
    - raw token ids
    - chat template 时间
    - dispatch/process/collect 时间
    - dedup 时间
    - e2e 时间
    - merge 时间
    - retry_rounds
    - chunk_chars
    - chunk_count

核心函数：

- `split_text(text, chunk_chars, overlap_chars)`
  - 把输入文本按字符切成多个重叠 chunk

- `overlap_window(left, right)`
  - 计算两个 chunk 的 overlap 字符窗口

- `collect_overlap_tokens(result, overlap_start, overlap_end)`
  - 收集一个 chunk 在 overlap 区内的 token
  - 输出 `(global_start, global_end, token_id, index)`

- `find_position_aware_match(left, right)`
  - 当前 1.0 的核心匹配逻辑
  - 基于 overlap 区内 token 的：
    - 全局字符起点
    - 全局字符终点
    - token_id
  - 先对齐，再找最长连续 run

- `merge_chunk_results(results, matches)`
  - 根据相邻 chunk 的 `MatchResult` 去重拼接

- `LoPTParallelTokenizer`
  - LoPT 主执行类
  - 构造时创建持久 `ProcessPoolExecutor`
  - `spawn` 启动方式

关键方法：

- `LoPTParallelTokenizer._dispatch(tasks)`
  - 提交所有 chunk 到进程池
  - 统计 dispatch/process/collect 总时间

- `LoPTParallelTokenizer.tokenize(...)`
  - 当前 1.0 最重要的方法
  - 执行流程：

```text
输入 text
  -> 计算 overlap_chars / chunk_chars
  -> split_text
  -> _dispatch 到子进程
  -> 若只有一个 chunk，直接返回
  -> 否则对相邻 chunk 做 find_position_aware_match
  -> 若任一 match.length < min_match_tokens，抛 RuntimeError
  -> merge_chunk_results
  -> build_inputs_with_special_tokens
  -> 组装 LoPTResult
```

注意：

- 当前实现中 `retry_rounds` 始终返回 `0`
- 当前实现中**没有真正实现动态 chunk doubling**
- 当前实现中 `process_only_time_s` 与 `dispatch_process_collect_time_s` 等值

#### 3.1.3 `benchmarks/real_web_corpus.py`

职责：

- 构建实验用真实网页语料
- 保证：
  - 纯英文语料
  - 纯中文语料
- 按字符数扩展到目标长度

主要实现：

- 内置英文 URL 列表 `ENGLISH_URLS`
- 内置中文 URL 列表 `CHINESE_URLS`
- `fetch_url()`
- `extract_visible_text()`
- `normalize_english_line()`
- `normalize_chinese_line()`
- `normalize_text()`
- `cycle_join()`
- `build_language_corpus()`
- `ensure_corpora()`

产物：

- `en_web_corpus.txt`
- `zh_web_corpus.txt`
- `en_sources.json`
- `zh_sources.json`

#### 3.1.4 `benchmarks/expand_existing_corpora.py`

职责：

- 在已有真实网页种子语料基础上扩展到更长字符数
- 不重新抓网，而是循环已有段落

适用场景：

- 已经有种子料，只想扩到 `1024k`

#### 3.1.5 `benchmarks/search_lopt_configs.py`

职责：

- 全量搜索 LoPT 参数组合
- 记录每个候选配置
- 选出每个 worker 下的 best
- 选出每个 case 的全局 best

关键结构：

- `TokenizerFamily`
  - 描述一个 tokenizer family 的配置

- `measure_native_case(...)`
  - 对单个 case 测 native 基线

- `evaluate_lopt_candidate(...)`
  - 对单个 LoPT 候选配置执行测量和精度判断

关键逻辑：

1. 对当前文本先测 native baseline
2. 遍历：
   - family
   - language
   - length
   - worker_processes
   - chunk_count
   - overlap_chars
3. 对每个候选调用 `evaluate_lopt_candidate`
4. 若 LoPT 输出与 native token ids 一致：
   - 标记 `candidate_status = valid`
5. 若不一致：
   - 调用 `native.serial_encode()` fallback
   - 标记 `fallback_used = True`
6. 每个 worker 保留 `lopt_e2e_time_ms` 最小的候选
7. 每个 case 再从各 worker best 中选最优

这里还有一个非常重要的搜索剪枝行为：

- 对同一个 case，若某个 `worker_processes` 的 best 已经**比上一个 worker best 更慢**，搜索会 `break`
- 即认为再继续增大 worker 数大概率没有收益

这属于 1.0 中已经存在的“单调失败即提前终止”策略。

#### 3.1.6 `benchmarks/postprocess_search_results.py`

职责：

- 从一个或多个 `search_detail.jsonl` 重建：
  - `worker_best_configs.*`
  - `best_configs.*`

功能：

- 合并多个 search 目录
- 过滤 `candidate_status == valid && !fallback_used`
- 按 worker 选 best
- 再按 case 选 final best

#### 3.1.7 `benchmarks/replay_best_configs.py`

职责：

- 对 `best_configs.json` 里的最优配置做 replay
- 用统一逻辑重新测量最终值
- 生成最终对比表

关键函数：

- `replay_case(...)`
  - 分别跑 native 与 LoPT 多次
  - 计算中位数
  - 输出 replay 结果字段

这个脚本是最终报告的权威数据来源之一。

#### 3.1.8 `benchmarks/compare_replay_results.py`

职责：

- 比较 candidate replay 与 reference replay
- 核查：
  - exact_match
  - token count
  - token hash
  - 关键时间字段偏差

适合做 spot check。

#### 3.1.9 `benchmarks/collect_env_info.py`

职责：

- 采集 benchmark 机器环境信息

采集内容：

- `hostname`
- `lscpu`
- `free -h`
- `/proc/meminfo`
- `numactl --hardware`

输出：

- `benchmark_env_info.json`

#### 3.1.10 `benchmarks/generate_html_report.py`

职责：

- 读取 search / replay / env / flow doc 等产物
- 生成仓库根目录 `index.html`

输入包括：

- `search_detail.jsonl`
- `worker_best_configs.json`
- `best_configs.json`
- `final_replay_results.json`
- `benchmark_env_info.json`
- `en_sources.json` / `zh_sources.json`
- `docs/vllm_tokenizer_flow.md`

该脚本是一个纯 Python 静态页面渲染器，不依赖前端框架。

### 3.2 推理流程 / API 流程 / 调度流程

### 3.2.1 原生 vLLM 流程

当前代码没有真正启动 HTTP 服务，而是**在进程内模拟一次 OpenAI tokenize 请求**：

```text
text
  -> TokenizeCompletionRequest
  -> OpenAIServingTokenization.create_tokenize
  -> OpenAIServingRender
  -> HfRenderer
  -> AsyncMicrobatchTokenizer.encode
  -> response.tokens
```

这条链路是真正用于 1.0 “原生基线”的核心路径。

### 3.2.2 LoPT 调度流程

```text
text
  -> split_text
  -> ProcessPoolExecutor(spawn)
     -> _tokenize_chunk
     -> return ChunkResult(start, end, input_ids, offsets)
  -> find_position_aware_match for adjacent chunks
  -> merge_chunk_results
  -> build_inputs_with_special_tokens
  -> final token_ids
```

### 3.2.3 搜索流程

```text
for family in families:
  for language in languages:
    for length in lengths:
      native_summary = measure_native_case(...)
      for worker_processes in worker_values:
        for chunk_count in candidate_chunk_counts(...):
          for overlap_chars in overlap_values:
            candidate = evaluate_lopt_candidate(...)
            append_jsonl / append_csv
        pick worker best
      pick final best
```

### 3.2.4 Replay 流程

```text
best_configs.json
  -> replay_case(...)
  -> final_replay_results.json
  -> final_replay_results.csv
  -> final_replay_tables.md
```

### 3.3 流式输出、并发、队列、限流、鉴权、日志、监控如何实现

这一部分需要明确区分“代码里有”和“代码里没有”。

#### 3.3.1 流式输出

**没有实现。**

原因：

- 当前项目不是在线服务
- 当前接口只关心 tokenize 完成后的整体 token ids
- 没有 SSE / websocket / chunked streaming

#### 3.3.2 并发

有两种并发模型：

1. **原生 vLLM 并发**
   - 由 vLLM 内部 `AsyncMicrobatchTokenizer` 负责
   - 代码通过 `NativeTokenizerTimer` 包装其 `encode`
   - 当前 benchmark 只是在单请求场景下测其耗时

2. **LoPT 多进程并发**
   - 由 `ProcessPoolExecutor(spawn)` 实现
   - 每个 worker 各自持有 tokenizer
   - 父进程负责收集和合并结果

#### 3.3.3 队列

显式自定义队列**没有实现**。

存在的“隐式队列”：

- 原生链路内部由 vLLM 的 `AsyncMicrobatchTokenizer` 管理请求排队
- LoPT 侧由 `ProcessPoolExecutor` 内部调度任务

#### 3.3.4 限流

**没有实现。**

因为当前不是服务端，没有并发用户接入控制逻辑。

#### 3.3.5 鉴权

**没有实现。**

原因：

- 没有 HTTP API
- 没有用户身份体系

#### 3.3.6 日志

严格意义上的结构化日志系统**没有实现**。

当前可见的“日志”形式主要是：

- CLI stdout 打印
- JSON / CSV / Markdown 落盘结果
- `run.log` 等外部执行产物

#### 3.3.7 监控

**没有实现实时监控系统。**

但存在可作为“离线监控快照”的产物：

- `benchmark_env_info.json`
- `search_meta.json`
- `case_failures.json`
- replay / spotcheck compare JSON

### 3.4 配置如何加载

当前 1.0 版本没有统一配置中心，也没有 YAML/TOML 配置文件。

配置来源有三类：

1. **命令行参数**
   - 每个脚本都有独立 `argparse`
   - 这是主要配置方式

2. **上游结果 JSON**
   - 例如：
     - `best_configs.json`
     - `search_meta.json`
     - `benchmark_env_info.json`

3. **代码内常量**
   - 例如：
     - 默认长度列表
     - family 常量
     - 默认 CPU 绑核备注
     - 报告标题和副标题

没有实现：

- 环境变量统一加载
- 配置继承
- 远程配置中心
- 热更新

---

## 4. 全部代码改动点（逐条列出）

本节需要说明一个事实：

**当前仓库无法直接恢复所有 git 历史，因此“新增 / 修改 / 删除”只能基于现有文件结构和功能关系反推。**

可以明确判断的是：1.0 版本的工作重点是**新增一整套离线 benchmark 工程**，而不是直接修改 vLLM 上游源码。

### 4.1 新增了什么

从现有目录结构可明确反推出，以下内容属于 1.0 工程核心新增内容：

#### 4.1.1 基准执行与 LoPT 原型

- `benchmarks/vllm_tokenizer_bench.py`
- `benchmarks/lopt_tokenizer.py`

新增原因：

- 需要一个能直接复用 vLLM 原生链路的基线 harness
- 需要一个独立于 vLLM 主仓、可快速迭代的 LoPT 原型实现

#### 4.1.2 数据准备

- `benchmarks/real_web_corpus.py`
- `benchmarks/expand_existing_corpora.py`

新增原因：

- 需要真实中英文网页语料
- 需要扩到 1M 字符级长度

#### 4.1.3 搜索 / 回放 / 对比

- `benchmarks/search_lopt_configs.py`
- `benchmarks/postprocess_search_results.py`
- `benchmarks/replay_best_configs.py`
- `benchmarks/compare_replay_results.py`

新增原因：

- 需要自动化全量搜索
- 需要统一筛选 best config
- 需要 replay 最优配置
- 需要 spot check 校验结果一致性

#### 4.1.4 环境快照与报告

- `benchmarks/collect_env_info.py`
- `benchmarks/generate_html_report.py`

新增原因：

- 需要记录实验环境
- 需要将结构化数据汇总为展示级 HTML

#### 4.1.5 文档与报告

- `docs/agent_quickstart.md`
- `docs/architecture.md`
- `docs/benchmark_methodology.md`
- `docs/project_status.md`
- `docs/repro_commands.md`
- `docs/results_guide.md`
- `docs/vllm_tokenizer_flow.md`
- 根目录 `index.html`
- 根目录 `README.md`

新增原因：

- 需要让新会话快速接手
- 需要记录方法、结果和复现路径

#### 4.1.6 结果目录

- `results/search_*`
- `results/replay_*`
- `results/spotcheck_*`
- `results/benchmark_env_info.json`
- `results/en_sources.json`
- `results/zh_sources.json`

新增原因：

- 需要持久化 benchmark 结果
- 需要保证可回溯

### 4.2 修改了什么

从当前仓库只能明确反推出以下“修改/整理”方向，而无法精确定位每一次历史 diff：

1. **README 和 docs 已被整理为当前 1.0 工程说明形态**
2. **根目录 `index.html` 已生成并承载最终报告**
3. **`replay_best_configs.py` 支持过滤 family / language / length**
4. **报告生成器已经被定制为中文化的最终展示版本**

这些修改的根本原因，是让这套实验从“能跑脚本”升级为“可复现、可交付、可展示”的完整工程。

### 4.3 删除了什么

**从当前快照无法反推出明确删除记录。**

可以比较确定的是：

- 当前仓库没有围绕“删除旧模块”设计逻辑
- 现有工程以新增为主
- 没有发现专门用于兼容旧接口的删除/迁移脚本

因此在文档层应视为：

- **1.0 没有可确认的显式删除点**

### 4.4 为什么这么改

总体原因可以归纳为四条：

1. 需要复用 vLLM 真实链路，而不是手写一个不对齐的 tokenizer benchmark
2. 需要把 LoPT 变成可搜索、可 replay、可验证的实验工程
3. 需要真实语料和严格精度校验，支撑结论可信度
4. 需要输出面向汇报和交付的完整报告

---

## 5. 原始方案 VS 最终实现的差异对比

### 5.1 与最初设想一致的部分

从代码反推，以下部分和最初设想基本一致：

1. **复用 vLLM 原生 Tokenizer 链路**
   - 已实现
   - 而且是 1.0 最重要的可信基础

2. **实现 LoPT 风格的多进程并行 Tokenizer**
   - 已实现第一版

3. **支持真实中英文数据**
   - 已实现

4. **支持参数搜索与最佳配置回放**
   - 已实现

5. **输出完整表格与 HTML 报告**
   - 已实现

### 5.2 发生偏离的部分

根据论文和当前代码比对，偏离主要在这里：

1. **LoPT 没有实现动态 chunk doubling**
   - 论文：match 不足时扩 chunk 后重试
   - 当前代码：match 不足直接抛异常 / fallback

2. **`LoPTConfig.max_retry_rounds` 没有真正生效**
   - 配置字段存在
   - 实际重试逻辑缺失

3. **chat template 时间字段存在，但当前值基本固定为 0**
   - 说明设计上预留了分段统计
   - 但实际 LoPT benchmark 里没有单独做模板阶段耗时拆分

4. **merge/dedup 仍在 Python 层**
   - 论文中 merge 为 C++
   - 当前代码尚未下沉

5. **项目最终不是“直接改 vLLM 服务”**
   - 而是“围绕 vLLM 链路的独立 benchmark 工程”

### 5.3 为什么会偏离

从代码反推，偏离的主要原因大概率是工程取舍：

1. **优先验证可行性而不是完整复现论文**
   - 所以先做最小可运行 LoPT 原型

2. **优先保精度和可解释性**
   - 所以失败时 fallback，不追求复杂自动重试

3. **优先保证与 vLLM 链路对齐**
   - 所以把精力放在 harness、search、replay 和报告，而不是先做 native merge kernel

4. **优先完成全流程交付**
   - 包括文档、结果、报告、spot check

---

## 6. 关键技术细节（详尽）

### 6.1 API 接口定义

这里的“API”不是 HTTP API，而是**命令行接口 + Python 内部调用接口**。

#### 6.1.1 命令行接口

主要 CLI 入口如下：

- `python3 -m benchmarks.real_web_corpus`
- `python3 -m benchmarks.expand_existing_corpora`
- `python3 -m benchmarks.vllm_tokenizer_bench`
- `python3 -m benchmarks.search_lopt_configs`
- `python3 -m benchmarks.postprocess_search_results`
- `python3 -m benchmarks.replay_best_configs`
- `python3 -m benchmarks.compare_replay_results`
- `python3 -m benchmarks.collect_env_info`
- `python3 -m benchmarks.generate_html_report`

配置方式：

- 统一用 `argparse`
- 每个脚本参数独立
- 没有统一配置文件

#### 6.1.2 核心内部接口

最重要的 Python 级接口：

- `NativeBenchmarkHarness.run_once(text)`
- `NativeBenchmarkHarness.serial_encode(text, add_special_tokens=True)`
- `LoPTParallelTokenizer.tokenize(text, add_special_tokens=True, chunk_count=None, overlap_chars=None, chat_template_time_s=0.0)`
- `evaluate_lopt_candidate(...)`
- `replay_case(...)`

### 6.2 错误处理

当前工程的错误处理风格是：

- 框架级报错直接抛异常
- 搜索阶段对候选配置允许 fallback
- replay 阶段一旦 mismatch 直接失败

具体表现：

1. **参数和文件错误**
   - 缺 corpus / tokenizer / json 文件时直接 `FileNotFoundError`
   - chunk_count 非法时 `ValueError`
   - 长度规格非法时 `ValueError`

2. **LoPT 匹配失败**
   - `lopt_tokenizer.py` 中，如果相邻 match 长度小于 `min_match_tokens`
   - 直接抛 `RuntimeError`

3. **搜索阶段的 fallback**
   - `search_lopt_configs.py` 中：
     - 若 LoPT token ids 与 native 不一致，执行 `serial_encode()` fallback
     - 若执行异常，也执行 `serial_encode()` fallback
   - 同时记录：
     - `fallback_used`
     - `fallback_serial_time_ms`
     - `fallback_token_hash`
     - `candidate_status`
     - `error_message`

4. **replay 阶段**
   - `replay_best_configs.py` 中若 exact match 失败，直接 `AssertionError`

5. **compare 阶段**
   - `compare_replay_results.py` 中若 token/hash/time 偏差超限，进程退出码为 1

### 6.3 并发模型

#### 6.3.1 原生链路

- 模型：`asyncio + vLLM AsyncMicrobatchTokenizer`
- 当前 benchmark 只是在单请求 case 上测其内部 encode 耗时

#### 6.3.2 LoPT 链路

- 模型：`ProcessPoolExecutor(spawn)`
- 子进程内 tokenizer 常驻于 `_WORKER_TOKENIZER`
- 父进程负责：
  - 提交任务
  - 收集结果
  - overlap 匹配
  - merge

#### 6.3.3 搜索层

- 搜索层本身没有再做外层并发
- 当前是**串行遍历参数空间**
- 这样做的好处是：
  - 结果更稳定
  - 不会引入多层并发干扰 benchmark

### 6.4 显存优化

当前代码**没有显式显存优化逻辑**。

原因：

1. 项目聚焦 CPU 侧 Tokenizer 性能
2. 没有真正加载大模型权重进行推理
3. 没有启动 GPU 推理服务

严格说，这个工程的“显存优化”现状是：

- **通过不启动真实 vLLM 推理引擎，规避 GPU/显存开销**
- 只保留 Tokenizer / Renderer / Serving 路径所需最小对象

所以这里应明确写成：

- 当前项目**没有 KV Cache、显存池、paged attention、权重量化等 GPU 优化逻辑**

### 6.5 多模型支持

当前多模型支持分两层：

1. **架构上支持泛化**
   - `vllm_tokenizer_bench.py` 支持任意本地 tokenizer path + model name

2. **搜索脚本里当前硬编码支持两族**
   - `DeepSeek-V4-Pro`
   - `Qwen3.5`

实现方式：

- `search_lopt_configs.py` 中定义 `TokenizerFamily`
- 通过命令行传入：
  - `--deepseek-tokenizer-path`
  - `--qwen-tokenizer-path`

扩展方式：

- 新增 family 定义
- 提供 tokenizer path
- 报告层会自动按 family 聚合

### 6.6 部署方式

当前代码没有在线部署方式，只有**离线运行方式**。

部署/运行方式实际是：

1. 准备本地仓库
2. 准备本地 vLLM 源码目录
3. 准备本地 tokenizer 目录
4. 执行 CLI 脚本
5. 结果写入 `results/`
6. HTML 报告写到根目录 `index.html`

也就是说，这个工程的“部署”本质是：

- **本地/远端主机上的离线脚本执行**

不是：

- systemd 服务
- Docker 编排服务
- HTTP 微服务
- K8s 在线部署

---

## 7. 当前代码存在的特点、依赖、注意事项

### 7.1 环境依赖

从代码可反推出的硬依赖：

- Python 3
- `transformers`
- 本地可 import 的 `vllm` 源码树
- 本地 tokenizer 目录
- 标准 Linux 命令：
  - `hostname`
  - `lscpu`
  - `free`
  - `bash`
  - `grep`
  - 可选 `numactl`

网络依赖：

- 仅在构建语料时可能需要外网抓取网页
- 如果已有 `results/en_web_corpus.txt` / `zh_web_corpus.txt`，则可离线运行

### 7.2 启动方式

最常见的入口顺序见 [repro_commands.md](/home/light/codes/LoPT/docs/repro_commands.md)。

概括为：

```text
1. real_web_corpus.py
2. search_lopt_configs.py
3. postprocess_search_results.py
4. replay_best_configs.py
5. collect_env_info.py
6. generate_html_report.py
```

### 7.3 可能的坑

1. **Tokenizer path 必须是本地目录**
   - 且必须能被 `PreTrainedTokenizerFast.from_pretrained()` 读取

2. **项目不自动下载 tokenizer**
   - 所有 tokenizer 目录需提前准备

3. **LoPT 当前没有动态重试**
   - `max_retry_rounds` 字段存在，但逻辑未实现

4. **`chat_template_time_ms` 当前基本为 0**
   - 因为 LoPT 路径没有把 chat template 单独做耗时测量

5. **`process_only_time_s` 不是纯子进程计算时间**
   - 它实际等于 dispatch + process + collect 总时间

6. **搜索阶段会提前停止某些 worker 扫描**
   - 一旦当前 worker best 比上一个 worker best 更慢，就 `break`
   - 这是有意的剪枝，不是 bug

7. **未直接修改 vLLM 上游源码**
   - 所有逻辑都在当前仓库实现
   - 依赖本地 import `/home/light/vllm`

8. **HTML 报告生成器很大**
   - `generate_html_report.py` 是一个 3000+ 行的大脚本
   - 它是渲染器，不是 benchmark 内核

9. **当前 LoPT 只实现多进程版本**
   - 没有并行线程池版 LoPT 主实现

10. **原生基线用的是 mocked serving context**
   - 它复用了 vLLM 请求链路
   - 但不是完整运行一个真实线上服务

### 7.4 特殊逻辑

1. **Native 基线通过 monkeypatch `AsyncMicrobatchTokenizer.encode` 计时**
2. **LoPT special token 在 merge 后统一通过 `build_inputs_with_special_tokens()` 回补**
3. **search 阶段 mismatch 不直接丢弃，而是执行串行 fallback 并保留记录**
4. **replay 阶段对 best config 重新跑多次取中位数**
5. **compare 阶段支持 `--subset-only`，允许对 spot check 子集比较**

---

## 8. 总结：这套代码最终实现了什么？未实现什么？

### 8.1 最终实现了什么

当前 1.0 代码最终实现的是：

1. **一个与本地 vLLM Tokenizer 请求链路对齐的原生 benchmark harness**
2. **一个第一版 LoPT 风格多进程 Tokenizer 原型**
3. **真实中英文长文本语料构建工具**
4. **面向 `DeepSeek-V4-Pro` 和 `Qwen3.5` 的参数搜索框架**
5. **best config replay 与 spot check 校验工具**
6. **环境信息采集工具**
7. **静态 HTML 报告生成器**
8. **完整的结果文件体系和文档体系**

如果只用一句话概括：

> 这是一套围绕 vLLM Tokenizer 长文本性能优化而构建的离线实验工程，能够复用原生链路做基线测试，并验证第一版 LoPT 多进程方案在真实中英文长文本上的性能与精度表现。

### 8.2 未实现什么

当前代码**没有实现**以下能力：

1. **没有在线服务**
   - 无 FastAPI
   - 无 HTTP API
   - 无 OpenAI 兼容服务对外暴露

2. **没有流式输出**

3. **没有鉴权 / 限流 / 监控 / 结构化日志系统**

4. **没有真正的动态 chunk 扩张重试**
   - 虽然配置字段保留了 `max_retry_rounds`

5. **没有 C++ / Rust merge 内核**
   - dedup 仍在 Python

6. **没有共享内存 / 零拷贝 IPC**

7. **没有自动策略选择器**
   - 当前最佳参数仍然依赖搜索结果

8. **没有直接把 LoPT 接入 vLLM 正式服务代码**
   - 当前仍是离线验证工程

### 8.3 对 1.0 的最准确定性

从当前代码本身出发，1.0 版本最合适的定性不是“生产化 Tokenizer 服务”，而是：

```text
一个可信的、可复现的、与 vLLM 原生链路对齐的
Tokenizer 优化实验与报告工程
```

它已经足够回答：

- 原生链路性能如何
- LoPT v1 多进程方案能否提速
- 哪些参数组合更优
- 精度是否保持一致

但它还不足以直接回答：

- 如何无缝接入线上 vLLM 服务
- 如何在生产流量下做动态策略调度
- 如何把 merge 成本进一步压到 native 内核级别

---

## 附：建议的新会话阅读顺序

如果是一个全新 session，建议按以下顺序理解 1.0 代码：

1. 本文档：`docs/v1_reverse_engineering_design.md`
2. [docs/agent_quickstart.md](/home/light/codes/LoPT/docs/agent_quickstart.md)
3. [benchmarks/vllm_tokenizer_bench.py](/home/light/codes/LoPT/benchmarks/vllm_tokenizer_bench.py)
4. [benchmarks/lopt_tokenizer.py](/home/light/codes/LoPT/benchmarks/lopt_tokenizer.py)
5. [benchmarks/search_lopt_configs.py](/home/light/codes/LoPT/benchmarks/search_lopt_configs.py)
6. [benchmarks/replay_best_configs.py](/home/light/codes/LoPT/benchmarks/replay_best_configs.py)
7. [benchmarks/generate_html_report.py](/home/light/codes/LoPT/benchmarks/generate_html_report.py)
8. [docs/repro_commands.md](/home/light/codes/LoPT/docs/repro_commands.md)

读完这些文件后，再看根目录 [index.html](../index.html)，就能将“代码结构”和“最终结果展示”对上。
