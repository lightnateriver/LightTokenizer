# Agent Quickstart

这是新 session 进入仓库后最应该先读的一份文档。

## 1. 项目在做什么

这个仓库聚焦的是 `vLLM API Server Tokenizer` 在超长输入下的 CPU 瓶颈。

我们关注的原生链路是：

```text
OpenAIServingTokenization.create_tokenize
  -> OpenAIServingRender.preprocess_completion
  -> HfRenderer.render_cmpl_async
  -> BaseRenderer._tokenize_prompt_async
  -> AsyncMicrobatchTokenizer.encode
```

LoPT 主思路是：

```text
长文本
  -> 切成带 overlap 的多个 chunk
  -> 多个 worker process 并行 encode
  -> 回收 chunk token ids + offsets
  -> overlap 去重
  -> 合并成最终 token ids
```

当前默认主线版本是 `v2.1`：

- 保持与原生链路 `exact match`
- 保留 LoPT 的多进程并行主骨架
- 增加 submit / collect / worker encode / materialize / dedup 细粒度打点

## 2. 先理解哪些目录

### 代码

- `src/`
  原始 LightTokenizer v1 时代代码，保留作为历史实现
- `benchmarks/`
  当前主线代码，包括搜索、回放、对比和报告生成
- `benchmarks/lopt_v2/`
  v2.1 模块化实现

### 文档

- `README.md`
- `docs/project_status.md`
- `docs/optimization_journal.md`
- `docs/architecture.md`
- `docs/vllm_tokenizer_flow.md`
- `../index.html`

### 结果

- `results/search_merged_v2_1_20260528/`
- `results/replay_merged_v2_1_20260528/`
- `results/replay_merged_20260526/`
- `results/benchmark_env_info.json`

## 3. 约束条件

- vLLM 本地最新源码：`/home/light/vllm`
- 不启动远端 `vllm serve`
- 不修改远端服务器的依赖和包
- benchmark 需要复用原生 vLLM tokenizer 路径
- 主实验绑核范围：`29-31,40-79,155-159`

## 4. 当前最重要的文件

- 报告模板：
  `docs/index_v1_template.html`
- 当前报告：
  `../index.html`
- 生成当前根目录主报告的脚本：
  `benchmarks/generate_html_report.py`

## 5. 如果只想快速验证几组 case

优先用：

- `python3 -m benchmarks.replay_best_configs`

建议带过滤条件：

- `--families`
- `--languages`
- `--lengths`

这样可以只复跑少量最佳配置，而不是重跑完整 search。

## 6. 如果要完整复现

标准流程：

1. 准备真实中英文语料
2. 运行 `search_lopt_configs.py`
3. 运行 `postprocess_search_results.py`
4. 运行 `replay_best_configs.py`
5. 运行 `collect_env_info.py`
6. 运行 `generate_html_report.py` 生成根目录报告

具体命令看：

- [repro_commands.md](repro_commands.md)

## 7. 精度标准

所有优化都必须满足：

- token 数量一致
- token IDs 完全一致
- token hash 一致

性能允许有自然波动，但精度不能退。
